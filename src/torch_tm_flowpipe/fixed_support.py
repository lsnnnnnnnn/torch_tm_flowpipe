"""Canonical configurable fixed-support batched Taylor models.

This module implements the tensor-friendly design point in the same public
package hierarchy as :mod:`torch_tm_flowpipe.batched_dense_tm`.  A support is
data: callers declare monomial exponents and the local-time variable, and the
descriptor freezes multiplication/projection/integration routes plus a stable
manifest hash.  The upstream DiffReach restricted quadratic basis is one
descriptor instance, not a VDP-specific solver.

The numerical route intentionally mirrors DiffReach commit
``dd628eb443b517d6415de93e7035b4baef73963e``.  It uses ordinary device
float64 operations for semantic equivalence; universal directed-rounding is
not claimed.  Soundness qualification is performed by independent outward
replay in the experiment layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch


DIFFREACH_SOURCE_SHA = "dd628eb443b517d6415de93e7035b4baef73963e"


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _unit_exponent(dim: int, index: int) -> tuple[int, ...]:
    result = [0] * int(dim)
    result[int(index)] = 1
    return tuple(result)


def _complete_total_degree_exponents(dim: int, order: int) -> tuple[tuple[int, ...], ...]:
    exponents: list[tuple[int, ...]] = []

    def visit(position: int, remaining: int, prefix: tuple[int, ...]) -> None:
        if position == int(dim):
            if remaining == 0:
                exponents.append(prefix)
            return
        for value in range(remaining + 1):
            visit(position + 1, remaining - value, (*prefix, value))

    for degree in range(int(order) + 1):
        visit(0, degree, ())
    return tuple(exponents)


@dataclass(frozen=True)
class FixedSupportRoute:
    """One ordered signed product contribution to a retained slot."""

    left_slot: int
    right_slot: int
    output_slot: int
    sign: int = 1


@dataclass(frozen=True)
class FixedSupportIntegrationRoute:
    """One source-to-output antiderivative contribution."""

    input_slot: int
    output_slot: int
    factor_numerator: int
    factor_denominator: int

    @property
    def factor(self) -> float:
        return float(self.factor_numerator) / float(self.factor_denominator)


@dataclass(frozen=True)
class FixedSupportDescriptor:
    """Immutable monomial support and precomputed algebra routes.

    ``multiply_routes`` is ordered.  Routes for each output slot are applied
    one stage at a time, preserving a declared expression order without
    data-dependent Python decisions in the tensor loop.
    """

    name: str
    variable_names: tuple[str, ...]
    exponents: tuple[tuple[int, ...], ...]
    local_time_index: int
    multiply_routes: tuple[FixedSupportRoute, ...]
    integration_routes: tuple[FixedSupportIntegrationRoute, ...]
    overflow_policy: str = "monomial_natural_interval"
    range_policy: str = "monomial_natural_interval"
    expression_order: str = "ordered_routes"
    source_contract: str = "generic"
    _exponent_to_slot: Mapping[tuple[int, ...], int] = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        dim = len(self.variable_names)
        if dim <= 0:
            raise ValueError("fixed support requires at least one polynomial variable")
        if not 0 <= int(self.local_time_index) < dim:
            raise ValueError("local_time_index is outside variable_names")
        normalized = tuple(tuple(int(value) for value in exp) for exp in self.exponents)
        if any(len(exp) != dim for exp in normalized):
            raise ValueError("every exponent must have one entry per variable")
        if any(value < 0 for exp in normalized for value in exp):
            raise ValueError("monomial exponents must be nonnegative")
        if len(set(normalized)) != len(normalized):
            raise ValueError("fixed support exponents must be unique")
        zero = (0,) * dim
        if zero not in normalized:
            raise ValueError("fixed support must retain the constant monomial")
        object.__setattr__(self, "exponents", normalized)
        index = {exp: slot for slot, exp in enumerate(normalized)}
        object.__setattr__(self, "_exponent_to_slot", index)
        for route in self.multiply_routes:
            if not (0 <= route.left_slot < len(normalized)):
                raise ValueError("multiply route has invalid left slot")
            if not (0 <= route.right_slot < len(normalized)):
                raise ValueError("multiply route has invalid right slot")
            if not (0 <= route.output_slot < len(normalized)):
                raise ValueError("multiply route has invalid output slot")
            if route.sign not in (-1, 1):
                raise ValueError("multiply route sign must be -1 or 1")
        for route in self.integration_routes:
            if not (0 <= route.input_slot < len(normalized)):
                raise ValueError("integration route has invalid input slot")
            if not (0 <= route.output_slot < len(normalized)):
                raise ValueError("integration route has invalid output slot")
            if route.factor_denominator <= 0:
                raise ValueError("integration denominator must be positive")

    @classmethod
    def from_exponents(
        cls,
        *,
        name: str,
        variable_names: Sequence[str],
        exponents: Sequence[Sequence[int]],
        local_time_index: int,
        overflow_policy: str = "monomial_natural_interval",
        range_policy: str = "monomial_natural_interval",
        source_contract: str = "generic",
    ) -> "FixedSupportDescriptor":
        """Build deterministic projection and integration routes."""

        variables = tuple(str(value) for value in variable_names)
        support = tuple(tuple(int(value) for value in exp) for exp in exponents)
        index = {exp: slot for slot, exp in enumerate(support)}
        multiply: list[FixedSupportRoute] = []
        for output_slot, output_exp in enumerate(support):
            for left_slot, left_exp in enumerate(support):
                for right_slot, right_exp in enumerate(support):
                    product = tuple(a + b for a, b in zip(left_exp, right_exp))
                    if product == output_exp:
                        multiply.append(FixedSupportRoute(left_slot, right_slot, output_slot, 1))
        integrate: list[FixedSupportIntegrationRoute] = []
        for input_slot, exp in enumerate(support):
            output_exp = list(exp)
            denominator = output_exp[int(local_time_index)] + 1
            output_exp[int(local_time_index)] += 1
            output_slot = index.get(tuple(output_exp))
            if output_slot is not None:
                integrate.append(FixedSupportIntegrationRoute(input_slot, output_slot, 1, denominator))
        return cls(
            name=str(name),
            variable_names=variables,
            exponents=support,
            local_time_index=int(local_time_index),
            multiply_routes=tuple(multiply),
            integration_routes=tuple(integrate),
            overflow_policy=str(overflow_policy),
            range_policy=str(range_policy),
            source_contract=str(source_contract),
        )

    @classmethod
    def diffreach_restricted_quadratic(
        cls,
        state_dim: int,
        *,
        state_variable_names: Sequence[str] | None = None,
    ) -> "FixedSupportDescriptor":
        """Exact upstream DiffReach ``c/L/Lt`` support and slot order.

        For state dimension ``D``, variables are ``[t, xi_0, ..., xi_D-1]``
        and slots are ``[1, all linear z, all t*z]``.  Thus VDP has the seven
        slots ``[1,t,xi0,xi1,t^2,t*xi0,t*xi1]``.
        """

        state_dim = int(state_dim)
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if state_variable_names is None:
            spatial_names = tuple(f"xi{index}" for index in range(state_dim))
        else:
            spatial_names = tuple(str(value) for value in state_variable_names)
            if len(spatial_names) != state_dim:
                raise ValueError("state_variable_names length must equal state_dim")
        variables = ("t", *spatial_names)
        dim = state_dim + 1
        zero = (0,) * dim
        linear = tuple(_unit_exponent(dim, index) for index in range(dim))
        time_unit = _unit_exponent(dim, 0)
        time_cross = tuple(tuple(a + b for a, b in zip(time_unit, exp)) for exp in linear)
        exponents = (zero, *linear, *time_cross)

        constant_slot = 0
        linear_slots = tuple(1 + index for index in range(dim))
        cross_slots = tuple(1 + dim + index for index in range(dim))
        routes: list[FixedSupportRoute] = [
            FixedSupportRoute(constant_slot, constant_slot, constant_slot)
        ]
        for variable_index, output_slot in enumerate(linear_slots):
            routes.extend(
                (
                    FixedSupportRoute(constant_slot, linear_slots[variable_index], output_slot),
                    FixedSupportRoute(linear_slots[variable_index], constant_slot, output_slot),
                )
            )
        time_linear_slot = linear_slots[0]
        for variable_index, output_slot in enumerate(cross_slots):
            routes.extend(
                (
                    FixedSupportRoute(constant_slot, cross_slots[variable_index], output_slot),
                    FixedSupportRoute(cross_slots[variable_index], constant_slot, output_slot),
                    FixedSupportRoute(time_linear_slot, linear_slots[variable_index], output_slot),
                    FixedSupportRoute(linear_slots[variable_index], time_linear_slot, output_slot),
                )
            )
            if variable_index == 0:
                # Upstream forms both time-linear products and then subtracts
                # one duplicate t^2 contribution as a final expression stage.
                routes.append(FixedSupportRoute(time_linear_slot, time_linear_slot, output_slot, -1))

        integration: list[FixedSupportIntegrationRoute] = [
            FixedSupportIntegrationRoute(constant_slot, linear_slots[0], 1, 1),
            FixedSupportIntegrationRoute(linear_slots[0], cross_slots[0], 1, 2),
        ]
        integration.extend(
            FixedSupportIntegrationRoute(linear_slots[index], cross_slots[index], 1, 1)
            for index in range(1, dim)
        )
        return cls(
            name=f"diffreach_restricted_quadratic_DR{1 + 2 * dim}",
            variable_names=variables,
            exponents=exponents,
            local_time_index=0,
            multiply_routes=tuple(routes),
            integration_routes=tuple(integration),
            overflow_policy="diffreach_restricted_quadratic_grouped",
            range_policy="diffreach_restricted_quadratic_horner",
            expression_order="diffreach_c_then_L_then_Lt_four_terms_then_t2_correction",
            source_contract=f"DiffReach:{DIFFREACH_SOURCE_SHA}:src/polynomial.py",
        )

    @classmethod
    def complete_total_degree(
        cls,
        *,
        variable_names: Sequence[str],
        order: int,
        local_time_index: int = 0,
    ) -> "FixedSupportDescriptor":
        """Generate a complete total-degree descriptor and all algebra routes."""

        variables = tuple(str(value) for value in variable_names)
        if not variables:
            raise ValueError("complete support requires variables")
        if int(order) < 0:
            raise ValueError("complete support order must be nonnegative")
        return cls.from_exponents(
            name=f"complete_total_degree_O{int(order)}_D{len(variables)}",
            variable_names=variables,
            exponents=_complete_total_degree_exponents(len(variables), int(order)),
            local_time_index=int(local_time_index),
            overflow_policy="monomial_natural_interval",
            range_policy="monomial_natural_interval",
            source_contract=f"complete_total_degree:order={int(order)}",
        )

    @property
    def dim(self) -> int:
        return len(self.variable_names)

    @property
    def num_slots(self) -> int:
        return len(self.exponents)

    @property
    def constant_slot(self) -> int:
        return self._exponent_to_slot[(0,) * self.dim]

    def slot(self, exponent: Sequence[int]) -> int:
        return self._exponent_to_slot[tuple(int(value) for value in exponent)]

    def linear_slot(self, variable_index: int) -> int:
        return self.slot(_unit_exponent(self.dim, int(variable_index)))

    def time_cross_slot(self, variable_index: int) -> int:
        exp = list(_unit_exponent(self.dim, int(variable_index)))
        exp[self.local_time_index] += 1
        return self.slot(exp)

    @property
    def linear_slots(self) -> tuple[int, ...]:
        return tuple(self.linear_slot(index) for index in range(self.dim))

    @property
    def time_cross_slots(self) -> tuple[int, ...]:
        return tuple(self.time_cross_slot(index) for index in range(self.dim))

    def manifest(self) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "schema": "torch_tm_flowpipe_fixed_support_v1",
            "name": self.name,
            "coefficient_layout": ["batch", "state_output", "monomial_slot"],
            "variable_names": list(self.variable_names),
            "local_time_index": self.local_time_index,
            "slots": [
                {"slot": slot, "exponent": list(exp)}
                for slot, exp in enumerate(self.exponents)
            ],
            "multiply_routes": [
                {
                    "left_slot": route.left_slot,
                    "right_slot": route.right_slot,
                    "output_slot": route.output_slot,
                    "sign": route.sign,
                }
                for route in self.multiply_routes
            ],
            "integration_routes": [
                {
                    "input_slot": route.input_slot,
                    "output_slot": route.output_slot,
                    "factor_numerator": route.factor_numerator,
                    "factor_denominator": route.factor_denominator,
                }
                for route in self.integration_routes
            ],
            "overflow_policy": self.overflow_policy,
            "range_policy": self.range_policy,
            "expression_order": self.expression_order,
            "source_contract": self.source_contract,
        }
        if self.source_contract.startswith("complete_total_degree:"):
            multiplication_table = []
            for left_slot, left_exp in enumerate(self.exponents):
                for right_slot, right_exp in enumerate(self.exponents):
                    product_exp = tuple(a + b for a, b in zip(left_exp, right_exp))
                    output_slot = self._exponent_to_slot.get(product_exp)
                    multiplication_table.append(
                        {
                            "left_slot": left_slot,
                            "right_slot": right_slot,
                            "product_exponent": list(product_exp),
                            "output_slot": output_slot,
                            "overflow": output_slot is None,
                        }
                    )
            differentiation_table = []
            for input_slot, exponent in enumerate(self.exponents):
                for variable_index, power in enumerate(exponent):
                    if power == 0:
                        continue
                    output_exp = list(exponent)
                    output_exp[variable_index] -= 1
                    differentiation_table.append(
                        {
                            "input_slot": input_slot,
                            "variable_index": variable_index,
                            "output_slot": self._exponent_to_slot[tuple(output_exp)],
                            "factor": power,
                        }
                    )
            integration_table = []
            endpoint_table = []
            time_index = int(self.local_time_index)
            for input_slot, exponent in enumerate(self.exponents):
                integrated = list(exponent)
                denominator = integrated[time_index] + 1
                integrated[time_index] += 1
                output_slot = self._exponent_to_slot.get(tuple(integrated))
                integration_table.append(
                    {
                        "input_slot": input_slot,
                        "output_exponent": integrated,
                        "output_slot": output_slot,
                        "factor_numerator": 1,
                        "factor_denominator": denominator,
                        "overflow": output_slot is None,
                    }
                )
                reduced = list(exponent)
                time_power = reduced[time_index]
                reduced[time_index] = 0
                endpoint_table.append(
                    {
                        "input_slot": input_slot,
                        "output_slot": self._exponent_to_slot[tuple(reduced)],
                        "time_power": time_power,
                    }
                )
            manifest["complete_algebra_tables"] = {
                "multiplication": multiplication_table,
                "differentiation": differentiation_table,
                "integration": integration_table,
                "endpoint_substitution": endpoint_table,
            }
        return manifest

    @property
    def support_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.manifest())).hexdigest()


@dataclass(frozen=True)
class FixedSupportKernelPlan:
    """Immutable device plan shared by object, functional, and compiled lanes."""

    support_sha256: str
    dtype: torch.dtype
    device: torch.device
    expression_order: str
    overflow_policy: str
    range_policy: str
    constant_slot: int
    local_time_index: int
    support_dim: int
    num_slots: int
    multiply_left: torch.Tensor
    multiply_right: torch.Tensor
    multiply_sign: torch.Tensor
    multiply_valid: torch.Tensor
    multiply_route_indices: tuple[tuple[int, int, int, int], ...]
    overflow_left: torch.Tensor
    overflow_right: torch.Tensor
    overflow_exponents: torch.Tensor
    integration_input: torch.Tensor
    integration_output: torch.Tensor
    integration_factor: torch.Tensor
    integration_input_indices: tuple[int, ...]
    integration_output_indices: tuple[int, ...]
    linear_slots: torch.Tensor
    time_cross_slots: torch.Tensor
    spatial_indices: torch.Tensor
    state_indices: torch.Tensor
    spatial_linear_slots: torch.Tensor
    time_evaluate_output: torch.Tensor
    time_evaluate_power: torch.Tensor
    time_evaluate_output_indices: tuple[int, ...]
    time_evaluate_power_integers: tuple[int, ...]
    spatial_off_diagonal_mask: torch.Tensor
    exponents: torch.Tensor
    exponent_tuples: tuple[tuple[int, ...], ...]


_KERNEL_PLAN_CACHE: dict[tuple[str, torch.dtype, str, int | None, str], FixedSupportKernelPlan] = {}
_KERNEL_PLAN_BUILD_COUNT = 0


def _normalized_device(device: torch.device | str) -> torch.device:
    normalized = torch.device(device)
    if normalized.type == "cuda" and normalized.index is None:
        normalized = torch.device("cuda", torch.cuda.current_device())
    return normalized


def _build_kernel_plan(
    descriptor: FixedSupportDescriptor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> FixedSupportKernelPlan:
    global _KERNEL_PLAN_BUILD_COUNT
    _KERNEL_PLAN_BUILD_COUNT += 1
    per_output: list[list[FixedSupportRoute]] = [[] for _ in range(descriptor.num_slots)]
    for route in descriptor.multiply_routes:
        per_output[route.output_slot].append(route)
    stages = max((len(routes) for routes in per_output), default=0)
    left = torch.zeros((stages, descriptor.num_slots), dtype=torch.long, device=device)
    right = torch.zeros_like(left)
    sign = torch.zeros((stages, descriptor.num_slots), dtype=dtype, device=device)
    valid = torch.zeros((stages, descriptor.num_slots), dtype=torch.bool, device=device)
    for output_slot, routes in enumerate(per_output):
        for stage, route in enumerate(routes):
            left[stage, output_slot] = route.left_slot
            right[stage, output_slot] = route.right_slot
            sign[stage, output_slot] = float(route.sign)
            valid[stage, output_slot] = True
    retained_exponents = set(descriptor.exponents)
    overflow_pairs = [
        (left_slot, right_slot, tuple(a + b for a, b in zip(left_exp, right_exp)))
        for left_slot, left_exp in enumerate(descriptor.exponents)
        for right_slot, right_exp in enumerate(descriptor.exponents)
        if tuple(a + b for a, b in zip(left_exp, right_exp))
        not in retained_exponents
    ]
    integration_input = torch.as_tensor(
        [route.input_slot for route in descriptor.integration_routes],
        dtype=torch.long,
        device=device,
    )
    integration_output = torch.as_tensor(
        [route.output_slot for route in descriptor.integration_routes],
        dtype=torch.long,
        device=device,
    )
    integration_factor = torch.as_tensor(
        [route.factor for route in descriptor.integration_routes],
        dtype=dtype,
        device=device,
    )
    linear_slots = torch.as_tensor(
        descriptor.linear_slots, dtype=torch.long, device=device
    )
    time_cross_slots = torch.as_tensor(
        descriptor.time_cross_slots, dtype=torch.long, device=device
    )
    spatial = tuple(
        index for index in range(descriptor.dim) if index != descriptor.local_time_index
    )
    spatial_indices = torch.as_tensor(spatial, dtype=torch.long, device=device)
    spatial_linear_slots = linear_slots.index_select(0, spatial_indices)
    time_outputs: list[int] = []
    time_powers: list[int] = []
    for exponent in descriptor.exponents:
        reduced = list(exponent)
        time_powers.append(reduced[descriptor.local_time_index])
        reduced[descriptor.local_time_index] = 0
        output_slot = descriptor._exponent_to_slot.get(tuple(reduced))
        if output_slot is None:
            raise ValueError("support is not closed under local-time evaluation")
        time_outputs.append(output_slot)
    spatial_count = len(spatial)
    return FixedSupportKernelPlan(
        support_sha256=descriptor.support_sha256,
        dtype=dtype,
        device=device,
        expression_order=descriptor.expression_order,
        overflow_policy=descriptor.overflow_policy,
        range_policy=descriptor.range_policy,
        constant_slot=descriptor.constant_slot,
        local_time_index=descriptor.local_time_index,
        support_dim=descriptor.dim,
        num_slots=descriptor.num_slots,
        multiply_left=left,
        multiply_right=right,
        multiply_sign=sign,
        multiply_valid=valid,
        multiply_route_indices=tuple(
            (
                route.left_slot,
                route.right_slot,
                route.output_slot,
                route.sign,
            )
            for route in descriptor.multiply_routes
        ),
        overflow_left=torch.as_tensor(
            [left for left, _, _ in overflow_pairs], dtype=torch.long, device=device
        ),
        overflow_right=torch.as_tensor(
            [right for _, right, _ in overflow_pairs], dtype=torch.long, device=device
        ),
        overflow_exponents=torch.as_tensor(
            [exponent for _, _, exponent in overflow_pairs],
            dtype=torch.long,
            device=device,
        ).reshape(-1, descriptor.dim),
        integration_input=integration_input,
        integration_output=integration_output,
        integration_factor=integration_factor,
        integration_input_indices=tuple(
            route.input_slot for route in descriptor.integration_routes
        ),
        integration_output_indices=tuple(
            route.output_slot for route in descriptor.integration_routes
        ),
        linear_slots=linear_slots,
        time_cross_slots=time_cross_slots,
        spatial_indices=spatial_indices,
        state_indices=torch.arange(spatial_count, dtype=torch.long, device=device),
        spatial_linear_slots=spatial_linear_slots,
        time_evaluate_output=torch.as_tensor(
            time_outputs, dtype=torch.long, device=device
        ),
        time_evaluate_power=torch.as_tensor(time_powers, dtype=dtype, device=device),
        time_evaluate_output_indices=tuple(time_outputs),
        time_evaluate_power_integers=tuple(time_powers),
        spatial_off_diagonal_mask=(
            1.0 - torch.eye(spatial_count, dtype=dtype, device=device)
        ),
        exponents=torch.as_tensor(descriptor.exponents, dtype=torch.long, device=device),
        exponent_tuples=descriptor.exponents,
    )


def fixed_support_kernel_plan(
    descriptor: FixedSupportDescriptor,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> FixedSupportKernelPlan:
    """Return the one cached plan for a support/dtype/device/expression order."""

    normalized = _normalized_device(device)
    key = (
        descriptor.support_sha256,
        dtype,
        normalized.type,
        normalized.index,
        descriptor.expression_order,
    )
    plan = _KERNEL_PLAN_CACHE.get(key)
    if plan is None:
        plan = _build_kernel_plan(descriptor, device=normalized, dtype=dtype)
        _KERNEL_PLAN_CACHE[key] = plan
    return plan


def fixed_support_kernel_plan_cache_info() -> dict[str, int]:
    return {"size": len(_KERNEL_PLAN_CACHE), "build_count": _KERNEL_PLAN_BUILD_COUNT}


@dataclass(frozen=True)
class FixedSupportInterval:
    """Batched interval with ordinary DiffReach-compatible arithmetic."""

    lo: torch.Tensor
    hi: torch.Tensor

    def __post_init__(self) -> None:
        if self.lo.shape != self.hi.shape:
            raise ValueError("interval endpoints must have identical shape")
        if self.lo.dtype != self.hi.dtype or self.lo.device != self.hi.device:
            raise ValueError("interval endpoints must share dtype and device")

    @classmethod
    def zeros_like(cls, value: torch.Tensor) -> "FixedSupportInterval":
        zero = torch.zeros_like(value)
        return cls(zero, zero)

    def add(self, other: "FixedSupportInterval | torch.Tensor | float") -> "FixedSupportInterval":
        if isinstance(other, FixedSupportInterval):
            return FixedSupportInterval(self.lo + other.lo, self.hi + other.hi)
        return FixedSupportInterval(self.lo + other, self.hi + other)

    def sub(self, other: "FixedSupportInterval | torch.Tensor | float") -> "FixedSupportInterval":
        if isinstance(other, FixedSupportInterval):
            return FixedSupportInterval(self.lo - other.hi, self.hi - other.lo)
        return FixedSupportInterval(self.lo - other, self.hi - other)

    def scale(self, value: torch.Tensor | float) -> "FixedSupportInterval":
        lo = self.lo * value
        hi = self.hi * value
        return FixedSupportInterval(torch.minimum(lo, hi), torch.maximum(lo, hi))

    def mul(self, other: "FixedSupportInterval | torch.Tensor | float") -> "FixedSupportInterval":
        if not isinstance(other, FixedSupportInterval):
            return self.scale(other)
        products = torch.stack(
            (
                self.lo * other.lo,
                self.lo * other.hi,
                self.hi * other.lo,
                self.hi * other.hi,
            ),
            dim=0,
        )
        return FixedSupportInterval(products.amin(dim=0), products.amax(dim=0))

    def square(self) -> "FixedSupportInterval":
        lower_square = self.lo * self.lo
        upper_square = self.hi * self.hi
        positive = self.lo >= 0
        negative = self.hi <= 0
        lo = torch.where(positive, lower_square, torch.where(negative, upper_square, torch.zeros_like(lower_square)))
        hi = torch.where(positive, upper_square, torch.where(negative, lower_square, torch.maximum(lower_square, upper_square)))
        return FixedSupportInterval(lo, hi)

    def pow(self, exponent: int) -> "FixedSupportInterval":
        exponent = int(exponent)
        if exponent < 0:
            raise ValueError("negative interval powers are not supported")
        if exponent == 0:
            one = torch.ones_like(self.lo)
            return FixedSupportInterval(one, one)
        if exponent == 1:
            return self
        if exponent == 2:
            return self.square()
        if exponent % 2 == 0:
            lo_power = torch.minimum(self.lo**exponent, self.hi**exponent)
            hi_power = torch.maximum(self.lo**exponent, self.hi**exponent)
            lo_power = torch.where((self.lo <= 0) & (self.hi >= 0), torch.zeros_like(lo_power), lo_power)
            return FixedSupportInterval(lo_power, hi_power)
        return FixedSupportInterval(self.lo**exponent, self.hi**exponent)

    def affine(self, matrix: torch.Tensor) -> "FixedSupportInterval":
        positive = matrix >= 0
        positive_matrix = matrix * positive
        negative_matrix = matrix * (~positive)
        lo_col = self.lo.unsqueeze(-1)
        hi_col = self.hi.unsqueeze(-1)
        lo = torch.sum(positive_matrix @ lo_col + negative_matrix @ hi_col, dim=-1)
        hi = torch.sum(positive_matrix @ hi_col + negative_matrix @ lo_col, dim=-1)
        return FixedSupportInterval(lo, hi)

    def subseteq_elem(self, other: "FixedSupportInterval") -> torch.Tensor:
        return (self.lo >= other.lo) & (self.hi <= other.hi)

    def where(self, mask: torch.Tensor, other: "FixedSupportInterval") -> "FixedSupportInterval":
        return FixedSupportInterval(torch.where(mask, self.lo, other.lo), torch.where(mask, self.hi, other.hi))

    @property
    def width(self) -> torch.Tensor:
        return self.hi - self.lo


@dataclass(frozen=True)
class FixedSupportLedger:
    """Named interval sources emitted by one fixed-support operation."""

    entries: tuple[tuple[str, FixedSupportInterval], ...] = ()

    @classmethod
    def from_mapping(cls, values: Mapping[str, FixedSupportInterval]) -> "FixedSupportLedger":
        return cls(tuple((str(name), interval) for name, interval in values.items()))

    def as_dict(self) -> dict[str, FixedSupportInterval]:
        return dict(self.entries)

    def total_like(self, reference: torch.Tensor) -> FixedSupportInterval:
        total = FixedSupportInterval.zeros_like(reference)
        for _, interval in self.entries:
            total = total.add(interval)
        return total

    def prefixed(self, prefix: str) -> "FixedSupportLedger":
        return FixedSupportLedger(tuple((f"{prefix}.{name}", interval) for name, interval in self.entries))

    def extend(self, other: "FixedSupportLedger") -> "FixedSupportLedger":
        return FixedSupportLedger((*self.entries, *other.entries))


def _linear_form_interval(
    coefficients: torch.Tensor,
    box_lo: torch.Tensor,
    box_hi: torch.Tensor,
) -> FixedSupportInterval:
    lower_products = coefficients * box_lo[:, None, :]
    upper_products = coefficients * box_hi[:, None, :]
    return FixedSupportInterval(
        torch.minimum(lower_products, upper_products).sum(dim=-1),
        torch.maximum(lower_products, upper_products).sum(dim=-1),
    )


def _monomial_interval(
    exponent: Sequence[int],
    box_lo: torch.Tensor,
    box_hi: torch.Tensor,
) -> FixedSupportInterval:
    result = FixedSupportInterval(
        torch.ones(box_lo.shape[0], dtype=box_lo.dtype, device=box_lo.device),
        torch.ones(box_hi.shape[0], dtype=box_hi.dtype, device=box_hi.device),
    )
    for variable_index, power in enumerate(exponent):
        if int(power) == 0:
            continue
        variable = FixedSupportInterval(box_lo[:, variable_index], box_hi[:, variable_index]).pow(int(power))
        result = result.mul(variable)
    return result


def _monomial_interval_table(
    exponents: torch.Tensor,
    box_lo: torch.Tensor,
    box_hi: torch.Tensor,
    *,
    maximum_power: int,
) -> FixedSupportInterval:
    """Evaluate many monomial natural intervals without per-route Python work."""

    if exponents.ndim != 2 or exponents.shape[1] != box_lo.shape[1]:
        raise ValueError("monomial exponent table and range box dimensions disagree")
    count = int(exponents.shape[0])
    lo = torch.ones(
        (box_lo.shape[0], count), dtype=box_lo.dtype, device=box_lo.device
    )
    hi = torch.ones_like(lo)
    if count == 0:
        return FixedSupportInterval(lo, hi)
    for variable_index in range(box_lo.shape[1]):
        powers = exponents[:, variable_index]
        factor_lo = torch.ones_like(lo)
        factor_hi = torch.ones_like(hi)
        variable = FixedSupportInterval(
            box_lo[:, variable_index], box_hi[:, variable_index]
        )
        for power in range(1, int(maximum_power) + 1):
            powered = variable.pow(power)
            mask = powers == power
            factor_lo = torch.where(mask[None, :], powered.lo[:, None], factor_lo)
            factor_hi = torch.where(mask[None, :], powered.hi[:, None], factor_hi)
        product = FixedSupportInterval(lo, hi).mul(
            FixedSupportInterval(factor_lo, factor_hi)
        )
        lo, hi = product.lo, product.hi
    return FixedSupportInterval(lo, hi)


@dataclass(frozen=True)
class FixedSupportPolynomial:
    """Polynomial coefficients shaped ``[batch, state_output, slot]``."""

    coeffs: torch.Tensor
    support: FixedSupportDescriptor

    def __post_init__(self) -> None:
        if self.coeffs.ndim != 3:
            raise ValueError("fixed-support coefficients must be rank 3 [batch, output, slot]")
        if self.coeffs.shape[-1] != self.support.num_slots:
            raise ValueError("coefficient slot axis does not match support descriptor")
        if not self.coeffs.is_floating_point():
            raise TypeError("fixed-support coefficients must be floating point")

    @classmethod
    def zeros(
        cls,
        batch: int,
        output_dim: int,
        support: FixedSupportDescriptor,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> "FixedSupportPolynomial":
        return cls(torch.zeros((batch, output_dim, support.num_slots), dtype=dtype, device=device), support)

    @classmethod
    def constant_like(
        cls,
        reference: "FixedSupportPolynomial",
        value: torch.Tensor | float,
    ) -> "FixedSupportPolynomial":
        coeffs = torch.zeros_like(reference.coeffs)
        constant = torch.as_tensor(value, dtype=coeffs.dtype, device=coeffs.device)
        if constant.ndim == 0:
            constant = constant.expand(reference.coeffs.shape[:2])
        elif constant.ndim == 1:
            constant = constant.unsqueeze(0).expand(reference.coeffs.shape[:2])
        else:
            constant = torch.broadcast_to(constant, reference.coeffs.shape[:2])
        coeffs[..., reference.support.constant_slot] = constant
        return cls(coeffs, reference.support)

    @property
    def batch(self) -> int:
        return int(self.coeffs.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.coeffs.shape[1])

    def _check(self, other: "FixedSupportPolynomial") -> None:
        if self.support != other.support:
            raise ValueError("fixed-support polynomial descriptors differ")
        if self.coeffs.shape != other.coeffs.shape:
            raise ValueError("fixed-support polynomial shapes differ")

    def add(self, other: "FixedSupportPolynomial") -> "FixedSupportPolynomial":
        self._check(other)
        return FixedSupportPolynomial(self.coeffs + other.coeffs, self.support)

    def sub(self, other: "FixedSupportPolynomial") -> "FixedSupportPolynomial":
        self._check(other)
        return FixedSupportPolynomial(self.coeffs - other.coeffs, self.support)

    def scale(self, value: torch.Tensor | float) -> "FixedSupportPolynomial":
        factor = torch.as_tensor(value, dtype=self.coeffs.dtype, device=self.coeffs.device)
        while factor.ndim < self.coeffs.ndim:
            factor = factor.unsqueeze(-1)
        return FixedSupportPolynomial(self.coeffs * factor, self.support)

    def component(self, index: int) -> "FixedSupportPolynomial":
        return FixedSupportPolynomial(self.coeffs[:, int(index) : int(index) + 1, :], self.support)

    @classmethod
    def stack(cls, values: Sequence["FixedSupportPolynomial"]) -> "FixedSupportPolynomial":
        if not values:
            raise ValueError("cannot stack an empty polynomial sequence")
        support = values[0].support
        if any(value.support != support for value in values):
            raise ValueError("cannot stack different fixed supports")
        return cls(torch.cat([value.coeffs for value in values], dim=1), support)

    def mul_trunc(self, other: "FixedSupportPolynomial") -> "FixedSupportPolynomial":
        self._check(other)
        plan = fixed_support_kernel_plan(
            self.support, device=self.coeffs.device, dtype=self.coeffs.dtype
        )
        result = torch.zeros_like(self.coeffs)
        for stage in range(plan.multiply_left.shape[0]):
            left = self.coeffs.index_select(-1, plan.multiply_left[stage])
            right = other.coeffs.index_select(-1, plan.multiply_right[stage])
            contribution = left * right * plan.multiply_sign[stage]
            result = result + torch.where(
                plan.multiply_valid[stage], contribution, torch.zeros_like(contribution)
            )
        return FixedSupportPolynomial(result, self.support)

    def _diffreach_groups(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        plan = fixed_support_kernel_plan(
            self.support, device=self.coeffs.device, dtype=self.coeffs.dtype
        )
        linear = self.coeffs.index_select(-1, plan.linear_slots)
        time_cross = self.coeffs.index_select(-1, plan.time_cross_slots)
        constant = self.coeffs[..., self.support.constant_slot]
        return constant, linear, time_cross

    def range(self, box_lo: torch.Tensor, box_hi: torch.Tensor) -> FixedSupportInterval:
        if box_lo.shape != box_hi.shape or box_lo.shape != (self.batch, self.support.dim):
            raise ValueError("range box must have shape [batch, support.dim]")
        if self.support.range_policy == "diffreach_restricted_quadratic_horner":
            plan = fixed_support_kernel_plan(
                self.support, device=self.coeffs.device, dtype=self.coeffs.dtype
            )
            constant, linear, cross = self._diffreach_groups()
            time_index = self.support.local_time_index
            spatial_index = plan.spatial_indices
            time_lo = box_lo[:, time_index]
            time_hi = box_hi[:, time_index]
            time_interval = FixedSupportInterval(
                time_lo[:, None].expand(-1, self.output_dim),
                time_hi[:, None].expand(-1, self.output_dim),
            )
            linear_spatial = linear.index_select(-1, spatial_index)
            cross_spatial = cross.index_select(-1, spatial_index)
            spatial_lo = box_lo.index_select(-1, spatial_index)
            spatial_hi = box_hi.index_select(-1, spatial_index)
            linear_x = _linear_form_interval(linear_spatial, spatial_lo, spatial_hi)
            cross_x = _linear_form_interval(cross_spatial, spatial_lo, spatial_hi)
            linear_time = linear[..., time_index]
            cross_time = cross[..., time_index]
            inner = FixedSupportInterval(cross_time, cross_time).mul(time_interval).add(
                FixedSupportInterval(linear_time, linear_time).add(cross_x)
            )
            base = FixedSupportInterval(constant + linear_x.lo, constant + linear_x.hi)
            return time_interval.mul(inner).add(base)

        result = FixedSupportInterval.zeros_like(self.coeffs[..., 0])
        for slot, exponent in enumerate(self.support.exponents):
            monomial = _monomial_interval(exponent, box_lo, box_hi)
            coefficient = self.coeffs[..., slot]
            term = FixedSupportInterval(coefficient, coefficient).mul(
                FixedSupportInterval(monomial.lo[:, None], monomial.hi[:, None])
            )
            result = result.add(term)
        return result

    def mul_ctrunc(
        self,
        other: "FixedSupportPolynomial",
        box_lo: torch.Tensor,
        box_hi: torch.Tensor,
    ) -> tuple["FixedSupportPolynomial", FixedSupportLedger]:
        self._check(other)
        kept = self.mul_trunc(other)
        if self.support.overflow_policy != "diffreach_restricted_quadratic_grouped":
            plan = fixed_support_kernel_plan(
                self.support, device=self.coeffs.device, dtype=self.coeffs.dtype
            )
            if plan.overflow_left.numel() == 0:
                overflow = FixedSupportInterval.zeros_like(self.coeffs[..., 0])
            else:
                monomials = _monomial_interval_table(
                    plan.overflow_exponents,
                    box_lo,
                    box_hi,
                    maximum_power=2
                    * max(max(exponent) for exponent in self.support.exponents),
                )
                coefficients = self.coeffs.index_select(
                    -1, plan.overflow_left
                ) * other.coeffs.index_select(-1, plan.overflow_right)
                lower = coefficients * monomials.lo[:, None, :]
                upper = coefficients * monomials.hi[:, None, :]
                overflow = FixedSupportInterval(
                    torch.minimum(lower, upper).sum(dim=-1),
                    torch.maximum(lower, upper).sum(dim=-1),
                )
            return kept, FixedSupportLedger.from_mapping(
                {"discarded_product_monomials": overflow}
            )

        _, left_linear, left_cross = self._diffreach_groups()
        _, right_linear, right_cross = other._diffreach_groups()
        plan = fixed_support_kernel_plan(
            self.support, device=self.coeffs.device, dtype=self.coeffs.dtype
        )
        time_index = self.support.local_time_index
        spatial_index = plan.spatial_indices
        spatial_lo = box_lo.index_select(-1, spatial_index)
        spatial_hi = box_hi.index_select(-1, spatial_index)

        left_x = left_linear.index_select(-1, spatial_index)
        right_x = right_linear.index_select(-1, spatial_index)
        square_lo = torch.minimum(spatial_lo * spatial_lo, spatial_hi * spatial_hi)
        square_hi = torch.maximum(spatial_lo * spatial_lo, spatial_hi * spatial_hi)
        square_lo = torch.where(
            (spatial_lo <= 0) & (spatial_hi >= 0), torch.zeros_like(square_lo), square_lo
        )
        diagonal_coefficient = left_x * right_x
        diagonal_lo = torch.minimum(
            diagonal_coefficient * square_lo[:, None, :],
            diagonal_coefficient * square_hi[:, None, :],
        ).sum(dim=-1)
        diagonal_hi = torch.maximum(
            diagonal_coefficient * square_lo[:, None, :],
            diagonal_coefficient * square_hi[:, None, :],
        ).sum(dim=-1)

        lo_i = spatial_lo[:, :, None]
        hi_i = spatial_hi[:, :, None]
        lo_j = spatial_lo[:, None, :]
        hi_j = spatial_hi[:, None, :]
        corners = torch.stack((lo_i * lo_j, lo_i * hi_j, hi_i * lo_j, hi_i * hi_j), dim=0)
        products_lo = corners.amin(dim=0)
        products_hi = corners.amax(dim=0)
        mask = plan.spatial_off_diagonal_mask
        products_lo = products_lo * mask
        products_hi = products_hi * mask
        pair_coefficient = left_x[:, :, :, None] * right_x[:, :, None, :]
        off_lo = torch.minimum(
            pair_coefficient * products_lo[:, None, :, :],
            pair_coefficient * products_hi[:, None, :, :],
        ).sum(dim=(-2, -1))
        off_hi = torch.maximum(
            pair_coefficient * products_lo[:, None, :, :],
            pair_coefficient * products_hi[:, None, :, :],
        ).sum(dim=(-2, -1))
        pure_spatial = FixedSupportInterval(diagonal_lo + off_lo, diagonal_hi + off_hi)

        left_linear_range = _linear_form_interval(left_linear, box_lo, box_hi)
        right_linear_range = _linear_form_interval(right_linear, box_lo, box_hi)
        left_cross_range = _linear_form_interval(left_cross, box_lo, box_hi)
        right_cross_range = _linear_form_interval(right_cross, box_lo, box_hi)
        time_interval = FixedSupportInterval(
            box_lo[:, time_index : time_index + 1].expand(-1, self.output_dim),
            box_hi[:, time_index : time_index + 1].expand(-1, self.output_dim),
        )
        time_cubic = time_interval.mul(left_linear_range.mul(right_cross_range)).add(
            time_interval.mul(right_linear_range.mul(left_cross_range))
        )
        time_quartic = time_interval.mul(time_interval).mul(left_cross_range.mul(right_cross_range))
        ledger = FixedSupportLedger.from_mapping(
            {
                "pure_spatial_quadratic": pure_spatial,
                "time_cubic": time_cubic,
                "time_quartic": time_quartic,
            }
        )
        return kept, ledger

    def integrate_time_trunc(self) -> "FixedSupportPolynomial":
        plan = fixed_support_kernel_plan(
            self.support, device=self.coeffs.device, dtype=self.coeffs.dtype
        )
        result = torch.zeros_like(self.coeffs)
        for route_index, (input_slot, output_slot) in enumerate(
            zip(plan.integration_input_indices, plan.integration_output_indices)
        ):
            contribution = self.coeffs[..., input_slot] * plan.integration_factor[route_index]
            result[..., output_slot] = result[..., output_slot] + contribution
        return FixedSupportPolynomial(result, self.support)

    def differentiate(self, variable_index: int) -> "FixedSupportPolynomial":
        """Differentiate exactly within a downward-closed monomial support."""

        variable = int(variable_index)
        if not 0 <= variable < self.support.dim:
            raise IndexError(variable)
        result = torch.zeros_like(self.coeffs)
        for input_slot, exponent in enumerate(self.support.exponents):
            factor = int(exponent[variable])
            if factor == 0:
                continue
            output_exp = list(exponent)
            output_exp[variable] -= 1
            try:
                output_slot = self.support.slot(output_exp)
            except KeyError as error:
                raise ValueError("support is not closed under differentiation") from error
            result[..., output_slot] = result[..., output_slot] + self.coeffs[..., input_slot] * factor
        return FixedSupportPolynomial(result, self.support)

    def integrate_time_ctrunc(
        self,
        box_lo: torch.Tensor,
        box_hi: torch.Tensor,
    ) -> tuple["FixedSupportPolynomial", FixedSupportLedger]:
        kept = self.integrate_time_trunc()
        if self.support.overflow_policy != "diffreach_restricted_quadratic_grouped":
            retained_inputs = {route.input_slot for route in self.support.integration_routes}
            overflow = FixedSupportInterval.zeros_like(self.coeffs[..., 0])
            for input_slot, exponent in enumerate(self.support.exponents):
                if input_slot in retained_inputs:
                    continue
                integrated_exp = list(exponent)
                denominator = integrated_exp[self.support.local_time_index] + 1
                integrated_exp[self.support.local_time_index] += 1
                monomial = _monomial_interval(integrated_exp, box_lo, box_hi)
                coefficient = self.coeffs[..., input_slot] / float(denominator)
                overflow = overflow.add(
                    FixedSupportInterval(coefficient, coefficient).mul(
                        FixedSupportInterval(monomial.lo[:, None], monomial.hi[:, None])
                    )
                )
            return kept, FixedSupportLedger.from_mapping({"integration_discarded_monomials": overflow})

        _, _, cross = self._diffreach_groups()
        plan = fixed_support_kernel_plan(
            self.support, device=self.coeffs.device, dtype=self.coeffs.dtype
        )
        time_index = self.support.local_time_index
        spatial_index = plan.spatial_indices
        time_lo = box_lo[:, time_index]
        time_hi = box_hi[:, time_index]
        time_cube_lo = (time_lo**3)[:, None]
        time_cube_hi = (time_hi**3)[:, None]
        time_square_lo = (time_lo**2)[:, None, None]
        time_square_hi = (time_hi**2)[:, None, None]
        time_square_coefficient = cross[..., time_index]
        cubic_lo = torch.minimum(time_square_coefficient * time_cube_lo, time_square_coefficient * time_cube_hi) / 3.0
        cubic_hi = torch.maximum(time_square_coefficient * time_cube_lo, time_square_coefficient * time_cube_hi) / 3.0
        cubic = FixedSupportInterval(cubic_lo, cubic_hi)

        spatial_lo = box_lo.index_select(-1, spatial_index)[:, None, :]
        spatial_hi = box_hi.index_select(-1, spatial_index)[:, None, :]
        corners = torch.stack(
            (
                time_square_lo * spatial_lo,
                time_square_lo * spatial_hi,
                time_square_hi * spatial_lo,
                time_square_hi * spatial_hi,
            ),
            dim=0,
        )
        time_square_x_lo = corners.amin(dim=0)
        time_square_x_hi = corners.amax(dim=0)
        spatial_cross = cross.index_select(-1, spatial_index)
        cross_lo = torch.minimum(spatial_cross * time_square_x_lo, spatial_cross * time_square_x_hi).sum(dim=-1) * 0.5
        cross_hi = torch.maximum(spatial_cross * time_square_x_lo, spatial_cross * time_square_x_hi).sum(dim=-1) * 0.5
        squared_spatial = FixedSupportInterval(cross_lo, cross_hi)
        return kept, FixedSupportLedger.from_mapping(
            {
                "integration_time_cubic": cubic,
                "integration_time_squared_spatial": squared_spatial,
            }
        )

    def evaluate_time(self, time_value: float | torch.Tensor) -> "FixedSupportPolynomial":
        time = torch.as_tensor(time_value, dtype=self.coeffs.dtype, device=self.coeffs.device)
        plan = fixed_support_kernel_plan(
            self.support, device=self.coeffs.device, dtype=self.coeffs.dtype
        )
        result = torch.zeros_like(self.coeffs)
        for input_slot in range(self.support.num_slots):
            output_slot = plan.time_evaluate_output_indices[input_slot]
            power = plan.time_evaluate_power_integers[input_slot]
            result[..., output_slot] = (
                result[..., output_slot] + self.coeffs[..., input_slot] * (time**power)
            )
        return FixedSupportPolynomial(result, self.support)


@dataclass(frozen=True)
class FixedSupportTaylorModel:
    polynomial: FixedSupportPolynomial
    remainder: FixedSupportInterval
    ledger: FixedSupportLedger = FixedSupportLedger()

    def __post_init__(self) -> None:
        if self.remainder.lo.shape != self.polynomial.coeffs.shape[:2]:
            raise ValueError("TM remainder shape must equal [batch, output]")

    @classmethod
    def from_polynomial(cls, polynomial: FixedSupportPolynomial) -> "FixedSupportTaylorModel":
        return cls(polynomial, FixedSupportInterval.zeros_like(polynomial.coeffs[..., 0]))

    @classmethod
    def constant_like(
        cls,
        reference: "FixedSupportTaylorModel",
        value: torch.Tensor | float,
    ) -> "FixedSupportTaylorModel":
        return cls.from_polynomial(FixedSupportPolynomial.constant_like(reference.polynomial, value))

    def component(self, index: int) -> "FixedSupportTaylorModel":
        return FixedSupportTaylorModel(
            self.polynomial.component(index),
            FixedSupportInterval(
                self.remainder.lo[:, int(index) : int(index) + 1],
                self.remainder.hi[:, int(index) : int(index) + 1],
            ),
            self.ledger,
        )

    @classmethod
    def stack(cls, values: Sequence["FixedSupportTaylorModel"]) -> "FixedSupportTaylorModel":
        if not values:
            raise ValueError("cannot stack an empty TM sequence")
        polynomial = FixedSupportPolynomial.stack([value.polynomial for value in values])
        remainder = FixedSupportInterval(
            torch.cat([value.remainder.lo for value in values], dim=1),
            torch.cat([value.remainder.hi for value in values], dim=1),
        )
        ledger = FixedSupportLedger()
        for component_index, value in enumerate(values):
            ledger = ledger.extend(value.ledger.prefixed(f"component_{component_index}"))
        return cls(polynomial, remainder, ledger)

    def add(self, other: "FixedSupportTaylorModel") -> "FixedSupportTaylorModel":
        return FixedSupportTaylorModel(
            self.polynomial.add(other.polynomial),
            self.remainder.add(other.remainder),
            self.ledger.extend(other.ledger),
        )

    def sub(self, other: "FixedSupportTaylorModel") -> "FixedSupportTaylorModel":
        return FixedSupportTaylorModel(
            self.polynomial.sub(other.polynomial),
            self.remainder.sub(other.remainder),
            self.ledger.extend(other.ledger),
        )

    def scale(self, value: torch.Tensor | float) -> "FixedSupportTaylorModel":
        return FixedSupportTaylorModel(
            self.polynomial.scale(value),
            self.remainder.scale(value),
            self.ledger,
        )

    def range(self, box_lo: torch.Tensor, box_hi: torch.Tensor) -> FixedSupportInterval:
        return self.polynomial.range(box_lo, box_hi).add(self.remainder)

    def mul(
        self,
        other: "FixedSupportTaylorModel",
        box_lo: torch.Tensor,
        box_hi: torch.Tensor,
    ) -> "FixedSupportTaylorModel":
        kept, polynomial_ledger = self.polynomial.mul_ctrunc(other.polynomial, box_lo, box_hi)
        left_range = self.polynomial.range(box_lo, box_hi)
        right_range = other.polynomial.range(box_lo, box_hi)
        left_times_right_remainder = left_range.mul(other.remainder)
        right_times_left_remainder = right_range.mul(self.remainder)
        remainder_times_remainder = self.remainder.mul(other.remainder)
        polynomial_overflow = polynomial_ledger.total_like(self.remainder.lo)
        remainder = (
            left_times_right_remainder.add(right_times_left_remainder)
            .add(remainder_times_remainder)
            .add(polynomial_overflow)
        )
        ledger = polynomial_ledger.extend(
            FixedSupportLedger.from_mapping(
                {
                    "left_polynomial_times_right_remainder": left_times_right_remainder,
                    "right_polynomial_times_left_remainder": right_times_left_remainder,
                    "remainder_times_remainder": remainder_times_remainder,
                }
            )
        )
        return FixedSupportTaylorModel(kept, remainder, ledger)

    def integrate_time(
        self,
        box_lo: torch.Tensor,
        box_hi: torch.Tensor,
    ) -> "FixedSupportTaylorModel":
        kept, polynomial_ledger = self.polynomial.integrate_time_ctrunc(box_lo, box_hi)
        time_index = self.polynomial.support.local_time_index
        time_magnitude = torch.maximum(
            torch.abs(box_lo[:, time_index : time_index + 1]),
            torch.abs(box_hi[:, time_index : time_index + 1]),
        )
        integrated_remainder = self.remainder.scale(time_magnitude)
        polynomial_overflow = polynomial_ledger.total_like(self.remainder.lo)
        ledger = polynomial_ledger.extend(
            FixedSupportLedger.from_mapping({"integrated_input_remainder": integrated_remainder})
        )
        return FixedSupportTaylorModel(kept, polynomial_overflow.add(integrated_remainder), ledger)

    def evaluate_time(self, time_value: float | torch.Tensor) -> "FixedSupportTaylorModel":
        return FixedSupportTaylorModel(self.polynomial.evaluate_time(time_value), self.remainder, self.ledger)

    def compose_affine(
        self,
        other: "FixedSupportTaylorModel",
        time_value: float | torch.Tensor,
    ) -> "FixedSupportTaylorModel":
        """Compose with a normalized affine state parameterization.

        The local-time variable remains the identity coordinate.  This is the
        dimension-generic form of DiffReach's ``c/L/Lt`` composition, and it
        is exact for the restricted quadratic descriptor.
        """

        support = self.polynomial.support
        if support != other.polynomial.support:
            raise ValueError("composition requires identical supports")
        state_dim = support.dim - 1
        if other.polynomial.output_dim != state_dim:
            raise ValueError("affine parameterization must have support.dim - 1 outputs")
        time_index = support.local_time_index
        if time_index != 0:
            raise NotImplementedError("DiffReach-compatible affine composition currently requires local time at index 0")
        if support.range_policy != "diffreach_restricted_quadratic_horner":
            batch = self.polynomial.batch
            dtype = self.polynomial.coeffs.dtype
            device = self.polynomial.coeffs.device
            time_polynomial = FixedSupportPolynomial.zeros(
                batch, 1, support, dtype=dtype, device=device
            )
            time_coefficients = time_polynomial.coeffs.clone()
            time_coefficients[..., support.linear_slot(time_index)] = 1.0
            variables = [
                FixedSupportTaylorModel.from_polynomial(
                    FixedSupportPolynomial(time_coefficients, support)
                ),
                *(other.component(index) for index in range(state_dim)),
            ]
            box_lo = torch.cat(
                (
                    torch.zeros((batch, 1), dtype=dtype, device=device),
                    -torch.ones((batch, state_dim), dtype=dtype, device=device),
                ),
                dim=1,
            )
            time_hi = torch.as_tensor(time_value, dtype=dtype, device=device)
            if time_hi.ndim == 0:
                time_hi = time_hi.expand(batch)
            box_hi = torch.cat(
                (
                    time_hi.reshape(batch, 1),
                    torch.ones((batch, state_dim), dtype=dtype, device=device),
                ),
                dim=1,
            )
            monomial_reference = self.component(0)
            monomials: list[FixedSupportTaylorModel] = []
            for exponent in support.exponents:
                if not any(exponent):
                    monomials.append(
                        FixedSupportTaylorModel.constant_like(
                            monomial_reference, 1.0
                        )
                    )
                    continue
                variable_index = max(
                    index for index, power in enumerate(exponent) if int(power) > 0
                )
                parent_exponent = list(exponent)
                parent_exponent[variable_index] -= 1
                parent_slot = support.slot(parent_exponent)
                monomials.append(
                    monomials[parent_slot].mul(
                        variables[variable_index], box_lo, box_hi
                    )
                )
            outputs: list[FixedSupportTaylorModel] = []
            for output_index in range(self.polynomial.output_dim):
                reference = self.component(output_index)
                accumulated = FixedSupportTaylorModel.constant_like(reference, 0.0)
                for slot, _exponent in enumerate(support.exponents):
                    coefficient = reference.polynomial.coeffs[..., slot]
                    if not bool(torch.any(coefficient != 0)):
                        continue
                    accumulated = accumulated.add(
                        monomials[slot].scale(coefficient)
                    )
                accumulated = FixedSupportTaylorModel(
                    accumulated.polynomial,
                    accumulated.remainder.add(reference.remainder),
                    accumulated.ledger.extend(reference.ledger),
                )
                outputs.append(accumulated)
            return FixedSupportTaylorModel.stack(outputs)
        plan = fixed_support_kernel_plan(
            support, device=self.polynomial.coeffs.device, dtype=self.polynomial.coeffs.dtype
        )

        self_constant, self_linear, self_cross = self.polynomial._diffreach_groups()
        other_constant, other_linear, _ = other.polynomial._diffreach_groups()
        time_identity = torch.zeros(
            (self.polynomial.batch, 1, support.dim),
            dtype=self.polynomial.coeffs.dtype,
            device=self.polynomial.coeffs.device,
        )
        time_identity[..., time_index] = 1.0
        affine_matrix = torch.cat((time_identity, other_linear), dim=1)
        affine_offset = torch.cat(
            (
                torch.zeros(
                    (self.polynomial.batch, 1),
                    dtype=self.polynomial.coeffs.dtype,
                    device=self.polynomial.coeffs.device,
                ),
                other_constant,
            ),
            dim=1,
        )
        linear_new = torch.einsum("bdv,bvw->bdw", self_linear, affine_matrix)
        constant_new = self_constant + (self_linear * affine_offset[:, None, :]).sum(dim=-1)
        cross_new = torch.einsum("bdv,bvw->bdw", self_cross, affine_matrix)
        linear_new[..., time_index] = linear_new[..., time_index] + (
            self_cross * affine_offset[:, None, :]
        ).sum(dim=-1)
        coefficients = torch.zeros_like(self.polynomial.coeffs)
        coefficients[..., support.constant_slot] = constant_new
        coefficients[..., plan.linear_slots] = linear_new
        coefficients[..., plan.time_cross_slots] = cross_new
        polynomial = FixedSupportPolynomial(coefficients, support)

        spatial_index = plan.spatial_indices
        linear_remainder = other.remainder.affine(self_linear.index_select(-1, spatial_index))
        cross_remainder = other.remainder.affine(self_cross.index_select(-1, spatial_index)).scale(time_value)
        remainder = self.remainder.add(linear_remainder).add(cross_remainder)
        ledger = FixedSupportLedger.from_mapping(
            {
                "composition_linear_parameter_remainder": linear_remainder,
                "composition_time_cross_parameter_remainder": cross_remainder,
            }
        )
        return FixedSupportTaylorModel(polynomial, remainder, ledger)


FixedSupportPolynomialRHS = Callable[
    [FixedSupportPolynomial, torch.Tensor, torch.Tensor], FixedSupportPolynomial
]
FixedSupportTMRHS = Callable[
    [FixedSupportTaylorModel, torch.Tensor, torch.Tensor], FixedSupportTaylorModel
]


def fixed_support_polynomial_picard(
    base: FixedSupportPolynomial,
    rhs: FixedSupportPolynomialRHS,
    box_lo: torch.Tensor,
    box_hi: torch.Tensor,
    *,
    iterations: int = 2,
) -> tuple[FixedSupportPolynomial, tuple[FixedSupportPolynomial, ...]]:
    """Fixed-count polynomial Picard construction used by DiffReach."""

    current = base
    trace: list[FixedSupportPolynomial] = []
    for _ in range(int(iterations)):
        current = base.add(rhs(current, box_lo, box_hi).integrate_time_trunc())
        trace.append(current)
    return current, tuple(trace)


@dataclass(frozen=True)
class FixedSupportDRPicardResult:
    model: FixedSupportTaylorModel
    initial_inclusion_mask: torch.Tensor
    round_inclusion_masks: torch.Tensor
    round_remainder_lo: torch.Tensor
    round_remainder_hi: torch.Tensor
    roundoff_remainder: FixedSupportInterval

    @property
    def initial_inclusion_passed(self) -> bool:
        return bool(torch.all(self.initial_inclusion_mask).item())


def fixed_support_dr_remainder_picard(
    rhs: FixedSupportTMRHS,
    new_x0: FixedSupportTaylorModel,
    seed: FixedSupportTaylorModel,
    box_lo: torch.Tensor,
    box_hi: torch.Tensor,
    *,
    rounds: int,
) -> FixedSupportDRPicardResult:
    """Reproduce upstream DR-RP masks and retain-on-failure semantics.

    The initial Picard image is used only for the returned initial inclusion
    mask and polynomial-difference roundoff interval.  Refinement then starts
    from ``seed`` exactly as in pinned DiffReach.  Every later component keeps
    the candidate interval only when it is a subset, otherwise it retains the
    previous component while adopting the new polynomial part.
    """

    initial_rhs = rhs(seed, box_lo, box_hi)
    initial_next = new_x0.add(initial_rhs.integrate_time(box_lo, box_hi))
    initial_mask = initial_next.remainder.subseteq_elem(seed.remainder)
    polynomial_difference = initial_next.polynomial.sub(seed.polynomial)
    roundoff = polynomial_difference.range(box_lo, box_hi)

    current = seed
    masks: list[torch.Tensor] = []
    remainder_los: list[torch.Tensor] = []
    remainder_his: list[torch.Tensor] = []
    for _ in range(int(rounds)):
        candidate = new_x0.add(rhs(current, box_lo, box_hi).integrate_time(box_lo, box_hi))
        next_remainder = candidate.remainder.add(roundoff)
        mask = next_remainder.subseteq_elem(current.remainder)
        accepted = next_remainder.where(mask, current.remainder)
        current = FixedSupportTaylorModel(candidate.polynomial, accepted, candidate.ledger)
        masks.append(mask)
        remainder_los.append(accepted.lo)
        remainder_his.append(accepted.hi)
    if masks:
        mask_tensor = torch.stack(masks, dim=0)
        lo_tensor = torch.stack(remainder_los, dim=0)
        hi_tensor = torch.stack(remainder_his, dim=0)
    else:
        shape = (0, *seed.remainder.lo.shape)
        mask_tensor = torch.empty(shape, dtype=torch.bool, device=seed.remainder.lo.device)
        lo_tensor = torch.empty(shape, dtype=seed.remainder.lo.dtype, device=seed.remainder.lo.device)
        hi_tensor = torch.empty_like(lo_tensor)
    return FixedSupportDRPicardResult(
        current,
        initial_mask,
        mask_tensor,
        lo_tensor,
        hi_tensor,
        roundoff,
    )


def fixed_support_build_linear_tm(
    center: torch.Tensor,
    scale: torch.Tensor,
    support: FixedSupportDescriptor,
) -> FixedSupportTaylorModel:
    """Build ``center + diag(scale) * xi`` in a declared fixed support."""

    if center.ndim != 2 or center.shape != scale.shape:
        raise ValueError("center and scale must have equal [batch, state] shapes")
    if support.dim != center.shape[1] + 1:
        raise ValueError("support must contain local time plus one generator per state")
    polynomial = FixedSupportPolynomial.zeros(
        center.shape[0],
        center.shape[1],
        support,
        dtype=center.dtype,
        device=center.device,
    )
    coefficients = polynomial.coeffs.clone()
    coefficients[..., support.constant_slot] = center
    for state_index in range(center.shape[1]):
        coefficients[:, state_index, support.linear_slot(state_index + 1)] = scale[:, state_index]
    return FixedSupportTaylorModel.from_polynomial(FixedSupportPolynomial(coefficients, support))


def fixed_support_identity_parameterization(
    batch: int,
    state_dim: int,
    support: FixedSupportDescriptor,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> FixedSupportTaylorModel:
    center = torch.zeros((batch, state_dim), dtype=dtype, device=device)
    scale = torch.ones_like(center)
    return fixed_support_build_linear_tm(center, scale, support)


def fixed_support_step_boxes(
    batch: int,
    state_dim: int,
    step_size: float,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zeros = torch.zeros((batch, 1), dtype=dtype, device=device)
    step_time = torch.full((batch, 1), float(step_size), dtype=dtype, device=device)
    ones = torch.ones((batch, state_dim), dtype=dtype, device=device)
    step_lo = torch.cat((zeros, -ones), dim=1)
    step_hi = torch.cat((step_time, ones), dim=1)
    eval_lo = torch.cat((zeros, -ones), dim=1)
    eval_hi = torch.cat((zeros, ones), dim=1)
    return step_lo, step_hi, eval_lo, eval_hi


@dataclass(frozen=True)
class FixedSupportSymbolicRemainderState:
    """Windowed DiffReach/Flow*-style J/Phi symbolic carry."""

    phi_buffer: torch.Tensor
    j_buffer: FixedSupportInterval
    count: torch.Tensor
    capacity: int
    inverse_scale: torch.Tensor
    slot_indices: torch.Tensor

    @classmethod
    def initialize(
        cls,
        batch: int,
        state_dim: int,
        capacity: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> "FixedSupportSymbolicRemainderState":
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("symbolic remainder capacity must be positive")
        phi = torch.zeros((batch, capacity, state_dim, state_dim), dtype=dtype, device=device)
        j_lo = torch.zeros((batch, capacity, state_dim), dtype=dtype, device=device)
        j_hi = torch.zeros_like(j_lo)
        count = torch.zeros((), dtype=torch.long, device=device)
        inverse_scale = torch.ones((batch, state_dim), dtype=dtype, device=device)
        slot_indices = torch.arange(capacity, dtype=torch.long, device=device)
        return cls(
            phi,
            FixedSupportInterval(j_lo, j_hi),
            count,
            capacity,
            inverse_scale,
            slot_indices,
        )


@dataclass(frozen=True)
class FixedSupportSymbolicStepResult:
    scale: torch.Tensor
    normalized_parameterization: FixedSupportTaylorModel
    state: FixedSupportSymbolicRemainderState


def fixed_support_symbolic_step_linear(
    parameterization: FixedSupportTaylorModel,
    endpoint: FixedSupportTaylorModel,
    state: FixedSupportSymbolicRemainderState,
    eval_lo: torch.Tensor,
    eval_hi: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> FixedSupportSymbolicStepResult:
    """Exact pinned symbolic-carry alignment with cap-triggered queue clear."""

    support = parameterization.polynomial.support
    if support != endpoint.polynomial.support:
        raise ValueError("symbolic carry requires identical supports")
    if support.local_time_index != 0:
        raise NotImplementedError("DiffReach symbolic carry requires local time at index 0")
    plan = fixed_support_kernel_plan(
        support,
        device=parameterization.polynomial.coeffs.device,
        dtype=parameterization.polynomial.coeffs.dtype,
    )
    spatial_slots = plan.spatial_linear_slots
    parameter_linear = parameterization.polynomial.coeffs.index_select(-1, spatial_slots)
    endpoint_linear = endpoint.polynomial.coeffs.index_select(-1, spatial_slots)
    composed_linear = torch.einsum("bij,bjk->bik", endpoint_linear, parameter_linear)
    phi_new = endpoint_linear * state.inverse_scale[:, None, :]

    indices = state.slot_indices
    active = indices < state.count
    phi_updated = torch.einsum("bij,bmjk->bmik", phi_new, state.phi_buffer)
    phi_roll = torch.roll(phi_updated, shifts=-1, dims=1)
    last_active = indices == (state.count - 1)
    phi_contributions = torch.where(
        last_active[None, :, None, None], phi_new[:, None, :, :], phi_roll
    )
    past_all = state.j_buffer.affine(phi_contributions)
    active_float = active.to(dtype=past_all.lo.dtype)[None, :, None]
    past = FixedSupportInterval(
        (past_all.lo * active_float).sum(dim=1),
        (past_all.hi * active_float).sum(dim=1),
    )
    seed_through = parameterization.remainder.affine(endpoint_linear)
    candidate_j = endpoint.remainder.add(seed_through)
    empty = state.count == 0
    # Pinned ``Interval.where`` selects ``self`` when the mask is true:
    # ``(r_x0 + seed_through).where(count == 0, r_x0)``.  Thus the seed is
    # propagated only on an empty queue; once entries exist, J_new is r_x0.
    current_j = FixedSupportInterval(
        torch.where(empty, candidate_j.lo, endpoint.remainder.lo),
        torch.where(empty, candidate_j.hi, endpoint.remainder.hi),
    )
    next_remainder = past.add(current_j)

    coefficients = torch.zeros_like(parameterization.polynomial.coeffs)
    coefficients[..., list(support.linear_slots[1:])] = composed_linear
    next_parameterization = FixedSupportTaylorModel(
        FixedSupportPolynomial(coefficients, support), next_remainder
    )
    range_of_endpoint = next_parameterization.range(eval_lo, eval_hi)
    scale = torch.maximum(torch.abs(range_of_endpoint.hi), torch.abs(range_of_endpoint.lo))
    inverse_scale = 1.0 / (scale + float(epsilon))
    next_parameterization = next_parameterization.scale(inverse_scale)

    active_phi = active[None, :, None, None]
    phi_buffer = torch.where(active_phi, phi_updated, state.phi_buffer)
    insertion_phi = (indices == state.count)[None, :, None, None]
    phi_buffer = torch.where(insertion_phi, phi_new[:, None, :, :], phi_buffer)
    insertion_j = (indices == state.count)[None, :, None]
    j_buffer = FixedSupportInterval(
        torch.where(insertion_j, current_j.lo[:, None, :], state.j_buffer.lo),
        torch.where(insertion_j, current_j.hi[:, None, :], state.j_buffer.hi),
    )
    count_next = torch.minimum(
        state.count + 1,
        torch.as_tensor(state.capacity, dtype=state.count.dtype, device=state.count.device),
    )
    just_full = count_next == state.capacity
    phi_buffer = torch.where(just_full, torch.zeros_like(phi_buffer), phi_buffer)
    j_buffer = FixedSupportInterval(
        torch.where(just_full, torch.zeros_like(j_buffer.lo), j_buffer.lo),
        torch.where(just_full, torch.zeros_like(j_buffer.hi), j_buffer.hi),
    )
    count_next = torch.where(just_full, torch.zeros_like(count_next), count_next)
    next_state = FixedSupportSymbolicRemainderState(
        phi_buffer,
        j_buffer,
        count_next,
        state.capacity,
        inverse_scale,
        state.slot_indices,
    )
    return FixedSupportSymbolicStepResult(scale, next_parameterization, next_state)


@dataclass(frozen=True)
class FixedSupportValidatedStep:
    model: FixedSupportTaylorModel
    parameterization: FixedSupportTaylorModel
    symbolic_state: FixedSupportSymbolicRemainderState
    dr_picard: FixedSupportDRPicardResult
    endpoint: FixedSupportInterval
    full_step_tube: FixedSupportInterval
    normalization_scale: torch.Tensor


@dataclass(frozen=True)
class FixedSupportReachResult:
    times: torch.Tensor
    endpoint_lo: torch.Tensor
    endpoint_hi: torch.Tensor
    tube_lo: torch.Tensor
    tube_hi: torch.Tensor
    initial_inclusion_masks: torch.Tensor
    round_inclusion_masks: torch.Tensor
    validated_steps: int
    requested_steps: int
    completed: bool
    first_failure_step: int | None
    first_failure_reason: str | None
    final_model: FixedSupportTaylorModel
    final_parameterization: FixedSupportTaylorModel
    final_symbolic_state: FixedSupportSymbolicRemainderState
    host_synchronizations: int
    device_transfers: int


class FixedSupportReachability:
    """Generic fixed-support ODE reachability pipeline.

    The plant enters only through polynomial and TM RHS callables.  The class
    implements the pinned two-Picard/DR-RP/normalization/symbolic-carry
    contract and always returns endpoint and full-step-tube arrays separately.
    """

    def __init__(
        self,
        *,
        support: FixedSupportDescriptor,
        state_dim: int,
        polynomial_rhs: FixedSupportPolynomialRHS,
        tm_rhs: FixedSupportTMRHS,
        step_size: float,
        initial_remainder: float | Sequence[float] = 0.01,
        polynomial_picard_iterations: int = 2,
        remainder_rounds: int = 10,
        symbolic_window_size: int = 1000,
        normalization_epsilon: float = 1e-12,
    ) -> None:
        self.support = support
        self.state_dim = int(state_dim)
        if support.dim != self.state_dim + 1:
            raise ValueError("support dimension must equal state_dim + local time")
        self.polynomial_rhs = polynomial_rhs
        self.tm_rhs = tm_rhs
        self.step_size = float(step_size)
        self.polynomial_picard_iterations = int(polynomial_picard_iterations)
        self.remainder_rounds = int(remainder_rounds)
        self.symbolic_window_size = int(symbolic_window_size)
        self.normalization_epsilon = float(normalization_epsilon)
        initial = torch.as_tensor(initial_remainder, dtype=torch.float64)
        if initial.ndim == 0:
            initial = initial.repeat(self.state_dim)
        if initial.shape != (self.state_dim,):
            raise ValueError("initial_remainder must be scalar or one value per state")
        self._initial_remainder = initial

    def step_once(
        self,
        model: FixedSupportTaylorModel,
        parameterization: FixedSupportTaylorModel,
        symbolic_state: FixedSupportSymbolicRemainderState,
        step_lo: torch.Tensor,
        step_hi: torch.Tensor,
        eval_lo: torch.Tensor,
        eval_hi: torch.Tensor,
    ) -> FixedSupportValidatedStep:
        endpoint_previous = model.evaluate_time(self.step_size)
        symbolic = fixed_support_symbolic_step_linear(
            parameterization,
            endpoint_previous,
            symbolic_state,
            eval_lo,
            eval_hi,
            epsilon=self.normalization_epsilon,
        )
        center = endpoint_previous.polynomial.coeffs[..., self.support.constant_slot]
        new_x0 = fixed_support_build_linear_tm(center, symbolic.scale, self.support)
        polynomial, _ = fixed_support_polynomial_picard(
            new_x0.polynomial,
            self.polynomial_rhs,
            step_lo,
            step_hi,
            iterations=self.polynomial_picard_iterations,
        )
        initial_remainder = self._initial_remainder.to(
            dtype=polynomial.coeffs.dtype, device=polynomial.coeffs.device
        ).expand(polynomial.batch, -1)
        seed = FixedSupportTaylorModel(
            polynomial,
            FixedSupportInterval(-initial_remainder, initial_remainder),
        )
        dr_picard = fixed_support_dr_remainder_picard(
            self.tm_rhs,
            new_x0,
            seed,
            step_lo,
            step_hi,
            rounds=self.remainder_rounds,
        )
        composed = dr_picard.model.compose_affine(
            symbolic.normalized_parameterization, self.step_size
        )
        endpoint_lo = step_lo.clone()
        endpoint_lo[:, self.support.local_time_index] = self.step_size
        endpoint = composed.range(endpoint_lo, step_hi)
        tube = composed.range(step_lo, step_hi)
        return FixedSupportValidatedStep(
            dr_picard.model,
            symbolic.normalized_parameterization,
            symbolic.state,
            dr_picard,
            endpoint,
            tube,
            symbolic.scale,
        )

    def verify(
        self,
        initial_lo: torch.Tensor,
        initial_hi: torch.Tensor,
        *,
        steps: int,
        reject_failed_initial_inclusion: bool = True,
    ) -> FixedSupportReachResult:
        if initial_lo.shape != initial_hi.shape or initial_lo.ndim != 2:
            raise ValueError("initial bounds must be equal [batch, state] tensors")
        if initial_lo.shape[1] != self.state_dim:
            raise ValueError("initial bound state dimension mismatch")
        if initial_lo.dtype != initial_hi.dtype or initial_lo.device != initial_hi.device:
            raise ValueError("initial bounds must share dtype and device")
        steps = int(steps)
        batch = int(initial_lo.shape[0])
        center = 0.5 * (initial_lo + initial_hi)
        scale = 0.5 * (initial_hi - initial_lo)
        model = fixed_support_build_linear_tm(center, scale, self.support)
        parameterization = fixed_support_identity_parameterization(
            batch,
            self.state_dim,
            self.support,
            dtype=initial_lo.dtype,
            device=initial_lo.device,
        )
        symbolic_state = FixedSupportSymbolicRemainderState.initialize(
            batch,
            self.state_dim,
            min(self.symbolic_window_size, max(1, steps)),
            dtype=initial_lo.dtype,
            device=initial_lo.device,
        )
        step_lo, step_hi, eval_lo, eval_hi = fixed_support_step_boxes(
            batch,
            self.state_dim,
            self.step_size,
            dtype=initial_lo.dtype,
            device=initial_lo.device,
        )

        endpoint_los = [initial_lo]
        endpoint_his = [initial_hi]
        tube_los: list[torch.Tensor] = []
        tube_his: list[torch.Tensor] = []
        initial_masks: list[torch.Tensor] = []
        round_masks: list[torch.Tensor] = []
        host_synchronizations = 0
        failure_step: int | None = None
        failure_reason: str | None = None
        for step_index in range(steps):
            step = self.step_once(
                model,
                parameterization,
                symbolic_state,
                step_lo,
                step_hi,
                eval_lo,
                eval_hi,
            )
            initial_masks.append(step.dr_picard.initial_inclusion_mask)
            round_masks.append(step.dr_picard.round_inclusion_masks)
            inclusion_passed = bool(torch.all(step.dr_picard.initial_inclusion_mask).item())
            host_synchronizations += 1
            if reject_failed_initial_inclusion and not inclusion_passed:
                failure_step = step_index
                failure_reason = "failed_initial_DR_RP_inclusion"
                break
            model = step.model
            parameterization = step.parameterization
            symbolic_state = step.symbolic_state
            endpoint_los.append(step.endpoint.lo)
            endpoint_his.append(step.endpoint.hi)
            tube_los.append(step.full_step_tube.lo)
            tube_his.append(step.full_step_tube.hi)

        validated_steps = len(tube_los)
        if initial_masks:
            initial_mask_tensor = torch.stack(initial_masks, dim=0)
            round_mask_tensor = torch.stack(round_masks, dim=0)
        else:
            initial_mask_tensor = torch.empty(
                (0, batch, self.state_dim), dtype=torch.bool, device=initial_lo.device
            )
            round_mask_tensor = torch.empty(
                (0, self.remainder_rounds, batch, self.state_dim),
                dtype=torch.bool,
                device=initial_lo.device,
            )
        if tube_los:
            tube_lo_tensor = torch.stack(tube_los, dim=1)
            tube_hi_tensor = torch.stack(tube_his, dim=1)
        else:
            tube_lo_tensor = torch.empty(
                (batch, 0, self.state_dim), dtype=initial_lo.dtype, device=initial_lo.device
            )
            tube_hi_tensor = torch.empty_like(tube_lo_tensor)
        times = torch.arange(
            validated_steps + 1, dtype=initial_lo.dtype, device=initial_lo.device
        ) * self.step_size
        return FixedSupportReachResult(
            times=times,
            endpoint_lo=torch.stack(endpoint_los, dim=1),
            endpoint_hi=torch.stack(endpoint_his, dim=1),
            tube_lo=tube_lo_tensor,
            tube_hi=tube_hi_tensor,
            initial_inclusion_masks=initial_mask_tensor,
            round_inclusion_masks=round_mask_tensor,
            validated_steps=validated_steps,
            requested_steps=steps,
            completed=validated_steps == steps,
            first_failure_step=failure_step,
            first_failure_reason=failure_reason,
            final_model=model,
            final_parameterization=parameterization,
            final_symbolic_state=symbolic_state,
            host_synchronizations=host_synchronizations,
            device_transfers=0,
        )


def diffreach_vdp_polynomial_rhs(
    state: FixedSupportPolynomial,
    box_lo: torch.Tensor,
    box_hi: torch.Tensor,
) -> FixedSupportPolynomial:
    """Official VDP expression order, supplied as a benchmark RHS callable."""

    del box_lo, box_hi
    x = state.component(0)
    y = state.component(1)
    one = FixedSupportPolynomial.constant_like(x, 1.0)
    second = one.sub(x.mul_trunc(x)).mul_trunc(y).sub(x)
    return FixedSupportPolynomial.stack((y, second))


def diffreach_vdp_tm_rhs(
    state: FixedSupportTaylorModel,
    box_lo: torch.Tensor,
    box_hi: torch.Tensor,
) -> FixedSupportTaylorModel:
    """Official VDP expression order with fixed-support TM overflow."""

    x = state.component(0)
    y = state.component(1)
    one = FixedSupportTaylorModel.constant_like(x, 1.0)
    second = one.sub(x.mul(x, box_lo, box_hi)).mul(y, box_lo, box_hi).sub(x)
    return FixedSupportTaylorModel.stack((y, second))


__all__ = [
    "DIFFREACH_SOURCE_SHA",
    "FixedSupportDescriptor",
    "FixedSupportDRPicardResult",
    "FixedSupportIntegrationRoute",
    "FixedSupportInterval",
    "FixedSupportKernelPlan",
    "FixedSupportLedger",
    "FixedSupportPolynomial",
    "FixedSupportReachResult",
    "FixedSupportReachability",
    "FixedSupportRoute",
    "FixedSupportTaylorModel",
    "FixedSupportSymbolicRemainderState",
    "FixedSupportSymbolicStepResult",
    "FixedSupportValidatedStep",
    "diffreach_vdp_polynomial_rhs",
    "diffreach_vdp_tm_rhs",
    "fixed_support_dr_remainder_picard",
    "fixed_support_build_linear_tm",
    "fixed_support_identity_parameterization",
    "fixed_support_kernel_plan",
    "fixed_support_kernel_plan_cache_info",
    "fixed_support_polynomial_picard",
    "fixed_support_step_boxes",
    "fixed_support_symbolic_step_linear",
]
