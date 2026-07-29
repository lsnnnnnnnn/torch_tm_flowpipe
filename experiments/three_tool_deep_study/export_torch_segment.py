#!/usr/bin/env python3
"""Export one native Torch TM segment to the common read-only representation."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (HERE, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from common import (
    canonical_record,
    deterministic_points,
    git_sha,
    load_spec,
    unavailable,
    write_json,
)
from torch_tm_flowpipe import Interval, TaylorModel, TMVector, flowpipe_step

torch.set_default_dtype(torch.float64)


def _power(value: Any, exponent: int) -> Any:
    result: Any = 1.0
    for _ in range(int(exponent)):
        result = result * value
    return result


def rhs_from_spec(system: Mapping[str, Any]):
    def rhs(state: TMVector) -> TMVector:
        outputs: list[TaylorModel] = []
        for polynomial in system["rhs"]:
            value: Any = 0.0
            for term in polynomial["terms"]:
                product: Any = float(term["coefficient"])
                for coordinate, exponent in zip(state, term["powers"]):
                    product = product * _power(coordinate, int(exponent))
                value = value + product
            outputs.append(value)
        return TMVector(outputs)

    return rhs


def _bounds(interval: Interval) -> list[float]:
    return [
        float(interval.lo.detach().cpu()),
        float(interval.hi.detach().cpu()),
    ]


def _state(model: TaylorModel) -> dict[str, Any]:
    return {
        "polynomial_terms": [
            {
                "exponents": list(map(int, exponent)),
                "coefficient": float(coefficient.detach().cpu()),
            }
            for exponent, coefficient in sorted(
                model.polynomial.terms.items(),
                key=lambda item: (sum(item[0]), item[0]),
            )
        ],
        "independent_interval_remainder": _bounds(model.remainder),
        "native_structured_symbolic_remainder": unavailable(
            "Torch TM exposes one independent interval remainder only"
        ),
    }


def _native_samples(vector: TMVector) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for point in deterministic_points([_bounds(domain) for domain in vector.domain], limit=16):
        samples.append(
            {
                "point": point,
                "polynomial_values": [
                    float(model.polynomial.evaluate_point(point).detach().cpu())
                    for model in vector
                ],
                "total_intervals": [
                    [
                        float(model.polynomial.evaluate_point(point).detach().cpu())
                        + _bounds(model.remainder)[0],
                        float(model.polynomial.evaluate_point(point).detach().cpu())
                        + _bounds(model.remainder)[1],
                    ]
                    for model in vector
                ],
            }
        )
    return samples


def export_segment(
    spec: Mapping[str, Any],
    *,
    system_name: str,
    h: float,
    order: int,
) -> dict[str, Any]:
    system = spec["systems"][system_name]
    diagnostics: list[dict[str, Any]] = []
    propagation_started = time.perf_counter()
    segment = flowpipe_step(
        rhs_from_spec(system),
        [Interval(*bounds) for bounds in system["initial_box"]],
        float(h),
        int(order),
        max_validation_attempts=int(spec["torch"]["max_validation_attempts"]),
        validation_mode=str(spec["torch"]["validation_mode"]),
        diagnostics=diagnostics,
        diagnostics_context={
            "protocol": "common_segment_export",
            "endpoint_semantics": "raw_and_tightened_separate",
        },
    )
    propagation_s = time.perf_counter() - propagation_started
    if segment.endpoint_raw_tm is None:
        raise RuntimeError("Torch segment did not expose a raw endpoint")
    tube = segment.tm
    endpoint = segment.endpoint_raw_tm
    variable_names = [
        *[f"x0_{name}" for name in system["state_names"]],
        "tau",
    ]
    roles = ["state_generator"] * len(system["state_names"]) + ["local_time"]
    domains = [_bounds(domain) for domain in tube.domain]
    export_started = time.perf_counter()
    tube_states = [_state(model) for model in tube]
    endpoint_states = [_state(model) for model in endpoint]
    raw_endpoint_box = [
        _bounds(interval) for interval in endpoint.range_box()
    ]
    tube_box = [_bounds(interval) for interval in tube.range_box()]
    tightened = (
        segment.endpoint_tightened_tm
        if (
            segment.endpoint_tightened_tm is not None
            and segment.endpoint_tightening_applied
        )
        else None
    )
    tightened_states = (
        [_state(model) for model in tightened]
        if tightened is not None
        else None
    )
    tightened_box = (
        [_bounds(interval) for interval in tightened.range_box()]
        if tightened is not None
        else None
    )
    export_s = time.perf_counter() - export_started
    template = tube[0].remainder.lo
    record = canonical_record(
        tool="torch_tm_flowpipe",
        variant=f"complete_total_degree_{order}",
        system=system_name,
        h=h,
        variable_names=variable_names,
        variable_roles=roles,
        domains=domains,
        states=tube_states,
        raw_endpoint=endpoint_states,
        raw_endpoint_box=raw_endpoint_box,
        tube_box=tube_box,
        validation_trace=diagnostics,
        reset_metadata={
            "reset": "none_one_step",
            "preconditioning": "none",
            "endpoint_tightening_applied": False,
            "supplemental_tightened_endpoint_available": (
                segment.endpoint_tightened_tm is not None
                and segment.endpoint_tightening_applied
            ),
        },
        native_metadata={
            "status": segment.status,
            "validation_attempts": segment.validation_attempts,
            "message": segment.message,
            "order": order,
            "dtype": str(torch.get_default_dtype()),
            "device": "cpu",
            "directed_rounding_or_mpfr": "torch_nextafter_outward",
            "floating_point_enclosure_candidate": True,
            "native_point_samples": _native_samples(tube),
        },
        system_definition={
            "name": system_name,
            "state_names": list(system["state_names"]),
            "equations": system["rhs"],
            "initial_domain": system["initial_box"],
        },
        accepted_step=h if segment.status == "validated" else None,
        tightened_endpoint=tightened_states,
        tightened_endpoint_box=tightened_box,
        outcome={
            "status": (
                "success" if segment.status == "validated" else "failure"
            ),
            "category": (
                "" if segment.status == "validated" else "validation_failure"
            ),
            "reason": segment.message or "",
            "requested_horizon_reached": segment.status == "validated",
        },
        execution_metadata={
            "backend": "torch",
            "dtype": str(template.dtype),
            "device": str(template.device),
            "repository_commit": git_sha(REPO_ROOT),
            "runtime": {
                "setup_s": unavailable(
                    "setup is included in one-step propagation timing"
                ),
                "propagation_s": propagation_s,
                "export_s": export_s,
            },
        },
        basis_metadata={
            "name": f"complete_total_degree_{order}",
            "requested_order": order,
            "native_order": order,
            "coefficient_representation": (
                "Torch sparse exponent tuple to tensor coefficient"
            ),
        },
    )
    record["native_validation_passed"] = segment.status == "validated"
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--system", default="coupled_quadratic")
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    record = export_segment(
        load_spec(args.spec),
        system_name=args.system,
        h=args.h,
        order=args.order,
    )
    write_json(args.output, record)
    print(args.output)


if __name__ == "__main__":
    main()
