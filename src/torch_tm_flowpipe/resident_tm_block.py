"""Opt-in device-resident accepted-boundary Taylor-model composition.

The production solver calls this module only through an explicitly installed
task context.  A request owns a complete copy of its numerical inputs.  The
CUDA kernel consumes those inputs once, keeps all Horner intermediates in
device-local storage, and returns point coefficients, paid coefficient-error
records, and the complete Taylor-model remainder once.
"""
from __future__ import annotations

import ctypes as C
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from .interval import Interval
from .polynomial import Polynomial
from .taylor_model import TaylorModel
from .tm_vector import TMVector


MAX_ORDER = 6
MAX_SPLIT = 16
DIAGNOSTIC_FIELDS = (
    "truncation_width",
    "cutoff_width",
    "p_left_times_right_remainder_width",
    "p_right_times_left_remainder_width",
    "remainder_times_remainder_width",
    "retained_coefficient_roundoff_width",
    "factorized_multiplication_count",
    "outer_remainder_width",
    "composed_poly_range_width",
)
FLAGS = (
    "--std=c++14",
    "--gpu-architecture=compute_70",
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
)
_DISPATCH: ContextVar[Callable[..., Any] | None] = ContextVar(
    "resident_tm_block_dispatch", default=None
)


def active_dispatch() -> Callable[..., Any] | None:
    return _DISPATCH.get()


def canonical_exponents(order: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (first, degree - first)
        for degree in range(int(order) + 1)
        for first in range(degree + 1)
    )


def _normalized_split(value: int | None) -> int:
    if value is None or int(value) <= 1:
        return 0
    return int(value)


def _basis_fingerprint(order: int) -> str:
    payload = {
        "ordered_support": canonical_exponents(order),
        "expression": "two_variable_recursive_horner",
        "variable_roles": ["accepted_normal_u0", "accepted_normal_u1"],
        "rules": [
            "directed_binary64_coefficient_interval_mul_add",
            "total_degree_truncate_each_multiply",
            "point_coefficient_cutoff_device_mask_each_stage",
            "natural_or_split_interval_range",
            "pay_retained_coefficient_error_at_output",
        ],
        "variable_order": [0, 1],
    }
    import json

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True)
class ResidentNormalCompositionRequest:
    request_id: str
    outer_point: torch.Tensor
    outer_lo: torch.Tensor
    outer_hi: torch.Tensor
    outer_rem_lo: torch.Tensor
    outer_rem_hi: torch.Tensor
    inner_point: torch.Tensor
    inner_lo: torch.Tensor
    inner_hi: torch.Tensor
    inner_rem_lo: torch.Tensor
    inner_rem_hi: torch.Tensor
    domain_lo: torch.Tensor
    domain_hi: torch.Tensor
    outer_splits: tuple[int, ...]
    inner_splits: tuple[int, ...]
    order: int
    cutoff: float | None
    scalar_output: bool = False
    basis_fingerprint: str = ""

    @property
    def output_dim(self) -> int:
        return int(self.outer_point.shape[0])

    @property
    def term_count(self) -> int:
        return int(self.outer_point.shape[1])

    @property
    def structure_key(self) -> tuple[Any, ...]:
        # The active sparse masks are rebuilt from this request on the device;
        # the cached structure is the complete ordered support and exact graph.
        return (
            "resident-normal-composition-v1",
            2,
            int(self.order),
            self.output_dim,
            self.basis_fingerprint,
            "outer_endpoint_without_constant",
            "inner_previous_accepted_right_map",
            "horner-0-then-1",
            "directed-rn-rd-ru-no-fma-no-ftz",
            None if self.cutoff is None else float(self.cutoff).hex(),
        )

    def owned_copy(self, request_id: str | None = None) -> "ResidentNormalCompositionRequest":
        names = (
            "outer_point",
            "outer_lo",
            "outer_hi",
            "outer_rem_lo",
            "outer_rem_hi",
            "inner_point",
            "inner_lo",
            "inner_hi",
            "inner_rem_lo",
            "inner_rem_hi",
            "domain_lo",
            "domain_hi",
        )
        updates = {name: getattr(self, name).detach().clone() for name in names}
        if request_id is not None:
            updates["request_id"] = request_id
        return replace(self, **updates)


@dataclass(frozen=True)
class ResidentNormalCompositionResult:
    request_id: str
    status: str
    output: TaylorModel | TMVector | None = None
    diagnostics: Mapping[str, Any] | None = None
    coefficient_error_lo: torch.Tensor | None = None
    coefficient_error_hi: torch.Tensor | None = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def caller_copy(self) -> "ResidentNormalCompositionResult":
        return replace(
            self,
            output=deepcopy(self.output),
            diagnostics=deepcopy(self.diagnostics),
            coefficient_error_lo=(
                self.coefficient_error_lo.clone()
                if self.coefficient_error_lo is not None
                else None
            ),
            coefficient_error_hi=(
                self.coefficient_error_hi.clone()
                if self.coefficient_error_hi is not None
                else None
            ),
        )


def _dense_from_models(
    models: Sequence[TaylorModel], order: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    exponents = canonical_exponents(order)
    index = {exponent: slot for slot, exponent in enumerate(exponents)}
    point = torch.zeros((len(models), len(exponents)), dtype=torch.float64)
    rem_lo = torch.empty(len(models), dtype=torch.float64)
    rem_hi = torch.empty_like(rem_lo)
    splits: list[int] = []
    for output, model in enumerate(models):
        if model.n_vars != 2:
            raise ValueError("resident normal composition supports exactly two variables")
        if model.polynomial.dtype != torch.float64 or model.remainder.dtype != torch.float64:
            raise TypeError("resident normal composition requires binary64")
        for exponent, coefficient in model.polynomial.terms.items():
            if exponent not in index:
                raise ValueError(f"term {exponent} lies outside the order-{order} resident basis")
            point[output, index[exponent]] = coefficient.detach().to(device="cpu", dtype=torch.float64)
        rem_lo[output] = model.remainder.lo.detach().to(device="cpu", dtype=torch.float64)
        rem_hi[output] = model.remainder.hi.detach().to(device="cpu", dtype=torch.float64)
        splits.append(_normalized_split(model.truncation_range_split))
    return point, point.clone(), point.clone(), rem_lo, rem_hi, tuple(splits)


def request_from_taylor_models(
    request_id: str,
    outer: TaylorModel | TMVector,
    inner: TMVector,
    order: int,
    cutoff: float | None,
    domain: Sequence[Interval] | None = None,
) -> ResidentNormalCompositionRequest | None:
    order_i = int(order)
    models = list(outer.models) if isinstance(outer, TMVector) else [outer]
    if (
        order_i < 1
        or order_i > MAX_ORDER
        or len(models) not in {1, 2}
        or len(inner) != 2
        or any(model.n_vars != 2 for model in models)
        or inner.n_vars != 2
    ):
        return None
    selected_domain = list(domain or inner.domain)
    if len(selected_domain) != 2:
        return None
    try:
        outer_values = _dense_from_models(models, order_i)
        inner_values = _dense_from_models(list(inner.models), order_i)
    except (TypeError, ValueError):
        return None
    outer_point, outer_lo, outer_hi, outer_rem_lo, outer_rem_hi, outer_splits = outer_values
    inner_point, inner_lo, inner_hi, inner_rem_lo, inner_rem_hi, inner_splits = inner_values
    domain_lo = torch.stack(
        [value.lo.detach().to(device="cpu", dtype=torch.float64) for value in selected_domain]
    )
    domain_hi = torch.stack(
        [value.hi.detach().to(device="cpu", dtype=torch.float64) for value in selected_domain]
    )
    return ResidentNormalCompositionRequest(
        request_id,
        outer_point,
        outer_lo,
        outer_hi,
        outer_rem_lo,
        outer_rem_hi,
        inner_point,
        inner_lo,
        inner_hi,
        inner_rem_lo,
        inner_rem_hi,
        domain_lo,
        domain_hi,
        outer_splits,
        inner_splits,
        order_i,
        None if cutoff is None else float(cutoff),
        not isinstance(outer, TMVector),
        _basis_fingerprint(order_i),
    )


_MODULES: dict[torch.device, "ResidentKernelModule"] = {}
_MODULE_LOCK = threading.Lock()


def _check_cuda(code: int, operation: str) -> None:
    if code:
        raise RuntimeError(f"{operation} returned CUDA/NVRTC error {code}")


class ResidentKernelModule:
    def __init__(self, device: torch.device):
        started = time.perf_counter()
        self.device = device
        torch.cuda.init()
        torch.empty(1, device=device)
        libraries = sorted(
            (Path(torch.__file__).resolve().parent.parent / "nvidia/cuda_nvrtc/lib").glob(
                "libnvrtc.so*"
            )
        )
        if not libraries:
            raise RuntimeError("installed PyTorch NVRTC library not found")
        self.nvrtc = C.CDLL(str(libraries[0]))
        self.driver = C.CDLL("libcuda.so.1")
        self.nvrtc.nvrtcCreateProgram.argtypes = [
            C.POINTER(C.c_void_p),
            C.c_char_p,
            C.c_char_p,
            C.c_int,
            C.c_void_p,
            C.c_void_p,
        ]
        self.nvrtc.nvrtcCompileProgram.argtypes = [
            C.c_void_p,
            C.c_int,
            C.POINTER(C.c_char_p),
        ]
        for name in ("nvrtcGetProgramLogSize", "nvrtcGetPTXSize"):
            getattr(self.nvrtc, name).argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
        for name in ("nvrtcGetProgramLog", "nvrtcGetPTX"):
            getattr(self.nvrtc, name).argtypes = [C.c_void_p, C.c_void_p]
        self.nvrtc.nvrtcDestroyProgram.argtypes = [C.POINTER(C.c_void_p)]
        self.driver.cuModuleLoadData.argtypes = [C.POINTER(C.c_void_p), C.c_void_p]
        self.driver.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        self.driver.cuLaunchKernel.argtypes = [
            C.c_void_p,
            *([C.c_uint] * 7),
            C.c_void_p,
            C.POINTER(C.c_void_p),
            C.c_void_p,
        ]
        source = Path(__file__).with_name("resident_tm_cuda_kernel.cu").read_bytes()
        program = C.c_void_p()
        _check_cuda(
            self.nvrtc.nvrtcCreateProgram(
                C.byref(program), source, b"resident_tm_cuda_kernel.cu", 0, None, None
            ),
            "nvrtcCreateProgram",
        )
        try:
            options = (C.c_char_p * len(FLAGS))(*(flag.encode() for flag in FLAGS))
            code = self.nvrtc.nvrtcCompileProgram(program, len(FLAGS), options)
            size = C.c_size_t()
            _check_cuda(
                self.nvrtc.nvrtcGetProgramLogSize(program, C.byref(size)),
                "nvrtcGetProgramLogSize",
            )
            log = C.create_string_buffer(size.value)
            _check_cuda(self.nvrtc.nvrtcGetProgramLog(program, log), "nvrtcGetProgramLog")
            self.log = log.value.decode()
            if code:
                raise RuntimeError(f"NVRTC compilation failed ({code}): {self.log}")
            _check_cuda(
                self.nvrtc.nvrtcGetPTXSize(program, C.byref(size)), "nvrtcGetPTXSize"
            )
            ptx = C.create_string_buffer(size.value)
            _check_cuda(self.nvrtc.nvrtcGetPTX(program, ptx), "nvrtcGetPTX")
        finally:
            self.nvrtc.nvrtcDestroyProgram(C.byref(program))
        self.module = C.c_void_p()
        _check_cuda(self.driver.cuModuleLoadData(C.byref(self.module), ptx), "cuModuleLoadData")
        self.functions: dict[str, C.c_void_p] = {}
        for name in ("resident_normal_compose", "resident_arithmetic_probe"):
            function = C.c_void_p()
            _check_cuda(
                self.driver.cuModuleGetFunction(C.byref(function), self.module, name.encode()),
                "cuModuleGetFunction",
            )
            self.functions[name] = function
        major, minor = C.c_int(), C.c_int()
        _check_cuda(self.nvrtc.nvrtcVersion(C.byref(major), C.byref(minor)), "nvrtcVersion")
        self.build = {
            "nvrtc_path": str(libraries[0]),
            "nvrtc_version": [major.value, minor.value],
            "flags": list(FLAGS),
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "ptx_sha256": hashlib.sha256(ptx.raw).hexdigest(),
            "compile_and_load_s": time.perf_counter() - started,
            "compiler_log": self.log,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        }

    def launch(self, name: str, count: int, arguments: Sequence[Any]) -> None:
        values: list[Any] = []
        for argument in arguments:
            if isinstance(argument, torch.Tensor):
                values.append(C.c_void_p(argument.data_ptr()))
            elif isinstance(argument, float):
                values.append(C.c_double(argument))
            else:
                values.append(C.c_int(int(argument)))
        pointers = (C.c_void_p * len(values))(
            *(C.cast(C.byref(value), C.c_void_p) for value in values)
        )
        _check_cuda(
            self.driver.cuLaunchKernel(
                self.functions[name],
                max(1, (int(count) + 63) // 64),
                1,
                1,
                64,
                1,
                1,
                0,
                C.c_void_p(torch.cuda.current_stream(self.device).cuda_stream),
                pointers,
                None,
            ),
            name,
        )


def get_resident_module(device: torch.device | str = "cuda:0") -> ResidentKernelModule:
    selected = torch.device(device)
    if selected.index is None:
        selected = torch.device("cuda", torch.cuda.current_device())
    with torch.cuda.device(selected), _MODULE_LOCK:
        if selected not in _MODULES:
            _MODULES[selected] = ResidentKernelModule(selected)
        return _MODULES[selected]


def arithmetic_probe(
    a: Sequence[float], b: Sequence[float], device: torch.device | str = "cuda:0"
) -> torch.Tensor:
    module = get_resident_module(device)
    a_device = torch.tensor(a, dtype=torch.float64, device=module.device)
    b_device = torch.tensor(b, dtype=torch.float64, device=module.device)
    output = torch.empty((len(a), 4), dtype=torch.float64, device=module.device)
    module.launch("resident_arithmetic_probe", len(a), [a_device, b_device, output, len(a)])
    return output.cpu()


_STARTUP_CHECKED: set[tuple[str, str]] = set()
_STARTUP_LOCK = threading.Lock()


def resident_cuda_startup_check(
    device: torch.device | str = "cuda:0",
) -> dict[str, Any]:
    module = get_resident_module(device)
    key = (str(module.device), module.build["ptx_sha256"])
    with _STARTUP_LOCK:
        if key in _STARTUP_CHECKED:
            return {"reused": True, "build": module.build}
        a = [5e-324, -5e-324, 1e-160, -1e-160, 1.0, 1e30]
        b = [0.5, 0.5, 1e-160, 1e-160, 5e-324, -1e30]
        output = arithmetic_probe(a, b, module.device)
        for row, left, right in zip(output, a, b):
            for offset, exact in (
                (0, Fraction(left) + Fraction(right)),
                (2, Fraction(left) * Fraction(right)),
            ):
                if not Fraction(float(row[offset])) <= exact <= Fraction(float(row[offset + 1])):
                    raise FloatingPointError("resident CUDA directed primitive startup check failed")
        _STARTUP_CHECKED.add(key)
        return {
            "reused": False,
            "build": module.build,
            "primitive_bounds": [[float(value).hex() for value in row] for row in output],
        }


def _stack_payload(requests: Sequence[ResidentNormalCompositionRequest]) -> tuple[torch.Tensor, dict[str, slice]]:
    names = (
        "outer_point",
        "outer_lo",
        "outer_hi",
        "outer_rem_lo",
        "outer_rem_hi",
        "inner_point",
        "inner_lo",
        "inner_hi",
        "inner_rem_lo",
        "inner_rem_hi",
        "domain_lo",
        "domain_hi",
    )
    rows: list[torch.Tensor] = []
    offsets: dict[str, slice] = {}
    cursor = 0
    for name in names:
        values = torch.stack([getattr(request, name).reshape(-1) for request in requests])
        width = int(values.shape[1])
        offsets[name] = slice(cursor, cursor + width)
        rows.append(values)
        cursor += width
    return torch.cat(rows, dim=1).contiguous(), offsets


class ResidentTMBlockExecutor:
    """Own the compiled structure and execute homogeneous request groups."""

    def __init__(self, device: torch.device | str = "cuda:0") -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("the resident Taylor-model executor requires CUDA")
        self.module = get_resident_module(self.device)
        self.structure_cache: set[tuple[Any, ...]] = set()

    def evaluate(
        self,
        requests: Sequence[ResidentNormalCompositionRequest],
        *,
        backend: str = "cuda",
        diagnostics: bool = False,
        timings: dict[str, Any] | None = None,
    ) -> dict[str, ResidentNormalCompositionResult]:
        if backend != "cuda":
            raise ValueError("resident Taylor-model execution is CUDA-only")
        if not requests:
            return {}
        key = requests[0].structure_key
        if any(request.structure_key != key for request in requests):
            return {
                request.request_id: ResidentNormalCompositionResult(
                    request.request_id,
                    "unsupported_structure",
                    message="resident request group is not structurally homogeneous",
                )
                for request in requests
            }
        order = requests[0].order
        outputs = requests[0].output_dim
        terms = len(canonical_exponents(order))
        if (
            order < 1
            or order > MAX_ORDER
            or outputs not in {1, 2}
            or any(
                request.term_count != terms
                or request.basis_fingerprint != _basis_fingerprint(order)
                or len(request.outer_splits) != outputs
                or len(request.inner_splits) != 2
                or max((*request.outer_splits, *request.inner_splits), default=0) > MAX_SPLIT
                for request in requests
            )
        ):
            return {
                request.request_id: ResidentNormalCompositionResult(
                    request.request_id,
                    "unsupported_structure",
                    message="resident kernel supports dim=2, output<=2, order<=6 and split<=16",
                )
                for request in requests
            }
        self.structure_cache.add(key)
        host_started = time.perf_counter_ns()
        host_payload, offsets = _stack_payload(requests)
        host_outer_splits = torch.tensor(
            [request.outer_splits for request in requests], dtype=torch.int32
        ).contiguous()
        host_inner_splits = torch.tensor(
            [request.inner_splits for request in requests], dtype=torch.int32
        ).contiguous()
        # Concatenate flattened sections so both device views are contiguous;
        # a [B,O+2] column slice would retain a row stride and corrupt the ABI.
        host_splits = torch.cat(
            [host_outer_splits.reshape(-1), host_inner_splits.reshape(-1)]
        ).contiguous()
        packed_ns = time.perf_counter_ns()
        stream = torch.cuda.current_stream(self.device)
        h2d_begin = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        kernel_end = torch.cuda.Event(enable_timing=True)
        h2d_begin.record(stream)
        device_payload = host_payload.to(device=self.device, copy=True)
        device_splits = host_splits.to(device=self.device, copy=True)
        h2d_end.record(stream)

        def view(name: str, shape: tuple[int, ...]) -> torch.Tensor:
            return device_payload[:, offsets[name]].reshape(shape)

        batch = len(requests)
        outer_shape = (batch, outputs, terms)
        inner_shape = (batch, 2, terms)
        outer_point = view("outer_point", outer_shape)
        outer_lo = view("outer_lo", outer_shape)
        outer_hi = view("outer_hi", outer_shape)
        outer_rem_lo = view("outer_rem_lo", (batch, outputs))
        outer_rem_hi = view("outer_rem_hi", (batch, outputs))
        inner_point = view("inner_point", inner_shape)
        inner_lo = view("inner_lo", inner_shape)
        inner_hi = view("inner_hi", inner_shape)
        inner_rem_lo = view("inner_rem_lo", (batch, 2))
        inner_rem_hi = view("inner_rem_hi", (batch, 2))
        domain_lo = view("domain_lo", (batch, 2))
        domain_hi = view("domain_hi", (batch, 2))
        stride = 3 * terms + 2 + len(DIAGNOSTIC_FIELDS) + 2
        device_output = torch.empty(
            (batch, outputs, stride), dtype=torch.float64, device=self.device
        )
        receipt = torch.zeros(2, dtype=torch.int64, device=self.device)
        cutoff = 0.0 if requests[0].cutoff is None else float(requests[0].cutoff)
        self.module.launch(
            "resident_normal_compose",
            batch * outputs,
            [
                outer_point,
                outer_lo,
                outer_hi,
                outer_rem_lo,
                outer_rem_hi,
                inner_point,
                inner_lo,
                inner_hi,
                inner_rem_lo,
                inner_rem_hi,
                domain_lo,
                domain_hi,
                device_splits[: batch * outputs],
                device_splits[batch * outputs :],
                device_output,
                receipt,
                int(device_payload.stride(0)),
                batch,
                outputs,
                order,
                cutoff,
                int(requests[0].cutoff is not None),
            ],
        )
        kernel_end.record(stream)
        output_cpu = device_output.cpu()
        receipt_cpu = receipt.cpu()
        transfer_completed_ns = time.perf_counter_ns()
        receipt_values = [int(value) for value in receipt_cpu.tolist()]
        expected_magic = int.from_bytes(b"KLBMTSER", "little")
        if receipt_values != [expected_magic, batch * outputs]:
            raise RuntimeError(f"resident CUDA execution receipt mismatch: {receipt_values}")
        h2d_ms = h2d_begin.elapsed_time(h2d_end)
        kernel_ms = h2d_end.elapsed_time(kernel_end)
        results: dict[str, ResidentNormalCompositionResult] = {}
        exponents = canonical_exponents(order)
        for lane, request in enumerate(requests):
            rows = output_cpu[lane]
            status_values = rows[:, -1].to(dtype=torch.int64)
            if bool(torch.any(status_values != 0)):
                results[request.request_id] = ResidentNormalCompositionResult(
                    request.request_id,
                    "nonfinite",
                    message=f"resident kernel status {status_values.tolist()}",
                )
                continue
            coefficient_error_lo = rows[:, terms : 2 * terms].clone()
            coefficient_error_hi = rows[:, 2 * terms : 3 * terms].clone()
            base = 3 * terms
            rem_lo = rows[:, base]
            rem_hi = rows[:, base + 1]
            diag_values = rows[:, base + 2 : base + 2 + len(DIAGNOSTIC_FIELDS)]
            split_values = rows[:, -2].to(dtype=torch.int64)
            domain_values = [
                Interval(request.domain_lo[index], request.domain_hi[index])
                for index in range(2)
            ]
            models: list[TaylorModel] = []
            for output_index in range(outputs):
                terms_map = {
                    exponent: rows[output_index, term_index].clone()
                    for term_index, exponent in enumerate(exponents)
                    if bool(rows[output_index, term_index] != 0.0)
                }
                models.append(
                    TaylorModel(
                        Polynomial(terms_map, 2),
                        Interval(rem_lo[output_index], rem_hi[output_index]),
                        domain_values,
                        order=order,
                        truncation_range_split=(
                            int(split_values[output_index])
                            if int(split_values[output_index]) > 1
                            else None
                        ),
                    )
                )
            output_value: TaylorModel | TMVector = (
                models[0] if request.scalar_output else TMVector(models)
            )
            per_component = [
                {
                    field: float(diag_values[component, index])
                    for index, field in enumerate(DIAGNOSTIC_FIELDS)
                }
                for component in range(outputs)
            ]
            aggregate = {
                field: sum(row[field] for row in per_component)
                for field in DIAGNOSTIC_FIELDS
            }
            aggregate.update(
                {
                    "components": per_component,
                    "terms_after_insertion": sum(len(model.polynomial.terms) for model in models),
                    "basis_fingerprint": request.basis_fingerprint,
                    "structure_key_sha256": hashlib.sha256(repr(key).encode("utf-8")).hexdigest(),
                    "device": str(self.device),
                    "kernel_invocations": 1,
                    "host_numeric_roundtrips_inside_block": 0,
                }
            )
            results[request.request_id] = ResidentNormalCompositionResult(
                request.request_id,
                "ok",
                output_value,
                aggregate,
                coefficient_error_lo,
                coefficient_error_hi,
            )
        rebuilt_ns = time.perf_counter_ns()
        if timings is not None:
            input_bytes = host_payload.numel() * host_payload.element_size()
            split_bytes = host_splits.numel() * host_splits.element_size()
            output_bytes = output_cpu.numel() * output_cpu.element_size()
            receipt_bytes = receipt_cpu.numel() * receipt_cpu.element_size()
            dropped = ((2 * order + 1) * (2 * order + 2) // 2) - terms
            logical_local_bytes = batch * outputs * (
                (9 * terms + 2 * dropped + 32) * 8
            )
            timings.update(
                {
                    "schema": "resident-tm-block-timing-v1",
                    "packing_s": (packed_ns - host_started) / 1e9,
                    "h2d_cuda_event_s": h2d_ms / 1e3,
                    "kernel_cuda_event_s": kernel_ms / 1e3,
                    "transfer_and_sync_host_span_s": (
                        transfer_completed_ns - packed_ns
                    ) / 1e9,
                    "post_d2h_checks_and_rebuild_s": (
                        rebuilt_ns - transfer_completed_ns
                    ) / 1e9,
                    "d2h_checks_and_rebuild_s": (rebuilt_ns - packed_ns) / 1e9
                    - h2d_ms / 1e3
                    - kernel_ms / 1e3,
                    "d2h_host_span_s": (transfer_completed_ns - packed_ns) / 1e9
                    - h2d_ms / 1e3
                    - kernel_ms / 1e3,
                    "block_host_span_s": (rebuilt_ns - host_started) / 1e9,
                    "numeric_h2d_bytes": input_bytes,
                    "metadata_h2d_bytes": split_bytes,
                    "numeric_d2h_bytes": output_bytes,
                    "metadata_d2h_bytes": receipt_bytes,
                    "h2d_copy_operations": 2,
                    "d2h_copy_operations": 2,
                    "kernel_invocations": 1,
                    "host_synchronizations": 1,
                    "logical_device_local_intermediate_bytes": logical_local_bytes,
                    "batch": batch,
                    "outputs": outputs,
                    "terms": terms,
                    "structure_cache_entries": len(self.structure_cache),
                    "receipt": receipt_values,
                }
            )
        return results


def apply_result_diagnostics(
    target: dict[str, Any] | None,
    result: ResidentNormalCompositionResult,
) -> None:
    if target is None or not result.ok or result.diagnostics is None:
        return
    data = result.diagnostics
    target.update(
        {
            "insertion_dependency_preserving_used": True,
            "insertion_device_resident_used": True,
            "insertion_device_resident_contract": "resident-normal-composition-v1",
            "insertion_canonical_variable_order": [0, 1],
            "insertion_components": len(data["components"]),
            "insertion_factorized_multiplication_count": int(
                data["factorized_multiplication_count"]
            ),
            "insertion_truncation_width": data["truncation_width"],
            "insertion_cutoff_width": data["cutoff_width"],
            "insertion_inner_remainder_times_poly_width": data[
                "p_left_times_right_remainder_width"
            ],
            "insertion_accumulated_remainder_times_inner_poly_width": data[
                "p_right_times_left_remainder_width"
            ],
            "insertion_remainder_times_poly_width": data[
                "p_left_times_right_remainder_width"
            ]
            + data["p_right_times_left_remainder_width"],
            "insertion_remainder_times_remainder_width": data[
                "remainder_times_remainder_width"
            ],
            "insertion_retained_coefficient_roundoff_width": data[
                "retained_coefficient_roundoff_width"
            ],
            "composed_poly_range_width": data["composed_poly_range_width"],
            "output_remainder_width": sum(
                float(model.remainder.width())
                for model in (
                    result.output.models
                    if isinstance(result.output, TMVector)
                    else [result.output]
                )
            ),
            "terms_after_insertion": data["terms_after_insertion"],
            "insertion_resident_basis_fingerprint": data["basis_fingerprint"],
            "insertion_resident_structure_key_sha256": data[
                "structure_key_sha256"
            ],
            "insertion_resident_kernel_invocations": data["kernel_invocations"],
            "insertion_resident_host_numeric_roundtrips_inside_block": data[
                "host_numeric_roundtrips_inside_block"
            ],
            "_dependency_preserving_stage_rows": [],
            "_dependency_preserving_top_components": [],
        }
    )
    names = ["x", "y"] if len(data["components"]) == 2 else ["scalar"]
    for name, row in zip(names, data["components"]):
        target[f"insertion_truncation_width_{name}"] = row["truncation_width"]
        target[f"insertion_cutoff_width_{name}"] = row["cutoff_width"]
        target[f"composed_poly_range_width_{name}"] = row[
            "composed_poly_range_width"
        ]


__all__ = [
    "DIAGNOSTIC_FIELDS",
    "MAX_ORDER",
    "MAX_SPLIT",
    "ResidentNormalCompositionRequest",
    "ResidentNormalCompositionResult",
    "ResidentTMBlockExecutor",
    "_DISPATCH",
    "active_dispatch",
    "apply_result_diagnostics",
    "arithmetic_probe",
    "canonical_exponents",
    "request_from_taylor_models",
    "resident_cuda_startup_check",
]
