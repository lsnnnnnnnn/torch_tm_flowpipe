"""Run the finite independent exact-contract matrix for the resident block."""
from __future__ import annotations

import argparse
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path

import torch

from experiments.resident_tm_block.exact_oracle import verify_result
from torch_tm_flowpipe import Interval, Polynomial, TaylorModel, TMVector
from torch_tm_flowpipe.flowpipe import insert_ctrunc_normal_dependency_preserving
from torch_tm_flowpipe.resident_tm_block import (
    ResidentTMBlockExecutor,
    arithmetic_probe,
    request_from_taylor_models,
    resident_cuda_startup_check,
)


def _tm(terms, domain, order, remainder=(0.0, 0.0), split=None):
    return TaylorModel(
        Polynomial(terms, 2),
        Interval(*remainder),
        domain,
        order=order,
        truncation_range_split=split,
    )


def make_case(index: int, order: int, *, interval_coefficients: bool):
    domain = [Interval(-1.0, 1.0), Interval(-0.75, 1.0)]
    scale = 1.0 + index * 2.0**-14
    threshold_above = torch.nextafter(
        torch.tensor(1.0e-10, dtype=torch.float64),
        torch.tensor(torch.inf, dtype=torch.float64),
    ).item()
    outer = TMVector(
        [
            _tm(
                {
                    (1, 0): -1.1 * scale,
                    (0, 2): 0.3,
                    (3, 0): 2.0e-9,
                    (0, 1): 1.0e-10 if index % 2 else threshold_above,
                    **({(1, 1): 5.0e-324} if index % 3 == 0 else {}),
                },
                domain,
                order,
                (-1.0e-12, 2.0e-12),
                2 if index % 4 == 0 else None,
            ),
            _tm(
                {(0, 1): 0.75, (2, 1): -0.125 * scale, (0, 0): -3.0e-11},
                domain,
                order,
                (-2.0e-12, 4.0e-12),
            ),
        ]
    )
    inner = TMVector(
        [
            _tm(
                {(1, 0): 1.1, (0, 1): -0.1, (2, 0): 2.0e-5},
                domain,
                order,
                (-1.0e-6, 2.0e-6),
            ),
            _tm(
                {(0, 1): -0.9, (1, 0): -0.05, (0, 2): 3.0e-5},
                domain,
                order,
                (-2.0e-6, 1.0e-6),
                3 if index % 5 == 0 else None,
            ),
        ]
    )
    request = request_from_taylor_models(
        f"o{order}-lane-{index}", outer, inner, order, 1.0e-10, domain
    )
    assert request is not None
    if interval_coefficients:
        outer_lo, outer_hi = request.outer_lo.clone(), request.outer_hi.clone()
        inner_lo, inner_hi = request.inner_lo.clone(), request.inner_hi.clone()
        neg_inf = torch.full((), -torch.inf, dtype=torch.float64)
        pos_inf = torch.full((), torch.inf, dtype=torch.float64)
        for tensor_lo, tensor_hi, row, column in (
            (outer_lo, outer_hi, 0, 1),
            (inner_lo, inner_hi, 1, 2),
        ):
            tensor_lo[row, column] = torch.nextafter(tensor_lo[row, column], neg_inf)
            tensor_hi[row, column] = torch.nextafter(tensor_hi[row, column], pos_inf)
        # A zero point with a nonzero coefficient interval exercises severe
        # cancellation without letting a CPU sparse merge decide the answer.
        outer_lo[1, 4] = -2.0**-100
        outer_hi[1, 4] = 2.0**-100
        request = replace(
            request,
            outer_lo=outer_lo,
            outer_hi=outer_hi,
            inner_lo=inner_lo,
            inner_hi=inner_hi,
        )
    return request


def signature(result):
    models = result.output.models
    return {
        "models": [
            {
                "terms": [
                    [list(exponent), float(coefficient).hex()]
                    for exponent, coefficient in model.polynomial.terms.items()
                ],
                "remainder": [
                    float(model.remainder.lo).hex(),
                    float(model.remainder.hi).hex(),
                ],
                "split": model.truncation_range_split,
            }
            for model in models
        ],
        "coefficient_error_lo": [
            float(value).hex() for value in result.coefficient_error_lo.reshape(-1)
        ],
        "coefficient_error_hi": [
            float(value).hex() for value in result.coefficient_error_hi.reshape(-1)
        ],
    }


def run_checks() -> dict:
    startup = resident_cuda_startup_check()
    executor = ResidentTMBlockExecutor()
    matrix = []
    total = {
        "point_coefficient_checks": 0,
        "coefficient_error_interval_checks": 0,
        "remainder_interval_checks": 0,
    }
    all_requests = []
    for order, batch in ((4, 32), (6, 32)):
        requests = [
            make_case(index, order, interval_coefficients=index % 3 == 1)
            for index in range(batch)
        ]
        all_requests.extend(requests)
        timing = {}
        together = executor.evaluate(requests, timings=timing)
        reversed_results = executor.evaluate(list(reversed(requests)))
        chunk_results = {}
        for width in (1, 2, 8, 32):
            current = {}
            for begin in range(0, batch, width):
                current.update(executor.evaluate(requests[begin : begin + width]))
            for request in requests:
                if signature(current[request.request_id]) != signature(
                    together[request.request_id]
                ):
                    raise AssertionError(f"chunk width {width} changed {request.request_id}")
            chunk_results[str(width)] = True
        for request in requests:
            result = together[request.request_id]
            if signature(reversed_results[request.request_id]) != signature(result):
                raise AssertionError(f"task reorder changed {request.request_id}")
            checks = verify_result(request, result)
            for name in total:
                total[name] += int(checks[name])
        matrix.append(
            {
                "order": order,
                "batch": batch,
                "heterogeneous_support": True,
                "heterogeneous_interval_coefficients": True,
                "heterogeneous_splits": True,
                "task_reorder_bitwise": True,
                "chunk_widths_bitwise": chunk_results,
                "timing": timing,
            }
        )

    # A concrete legacy omission witness: the old sparse block returns a point
    # coefficient for 1.1*1.1 but does not own its retained multiply roundoff.
    domain = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    outer = TMVector([_tm({(1, 0): 1.1}, domain, 4)])
    inner = TMVector(
        [_tm({(1, 0): 1.1}, domain, 4), _tm({(0, 1): 1.0}, domain, 4)]
    )
    legacy = insert_ctrunc_normal_dependency_preserving(
        outer, inner, 4, None, domain
    )[0]
    witness = request_from_taylor_models(
        "retained-roundoff-witness", outer, inner, 4, None, domain
    )
    assert witness is not None
    resident = executor.evaluate([witness])[witness.request_id]
    verify_result(witness, resident)
    exact_error = Fraction(1.1) * Fraction(1.1) - Fraction(float(1.1 * 1.1))
    legacy_contains = (
        Fraction(float(legacy.remainder.lo))
        <= exact_error
        <= Fraction(float(legacy.remainder.hi))
    )
    resident_error = (
        Fraction(float(resident.coefficient_error_lo[0, 2])),
        Fraction(float(resident.coefficient_error_hi[0, 2])),
    )
    if legacy_contains or not resident_error[0] <= exact_error <= resident_error[1]:
        raise AssertionError("retained coefficient roundoff witness did not separate paths")

    wrong = replace(all_requests[0], request_id="wrong-fingerprint", basis_fingerprint="0" * 64)
    wrong_result = executor.evaluate([wrong])[wrong.request_id]
    nonfinite_point = all_requests[0].inner_point.clone()
    nonfinite_lo = all_requests[0].inner_lo.clone()
    nonfinite_hi = all_requests[0].inner_hi.clone()
    nonfinite_point[0, 0] = nonfinite_lo[0, 0] = nonfinite_hi[0, 0] = torch.inf
    nonfinite = replace(
        all_requests[0],
        request_id="nonfinite",
        inner_point=nonfinite_point,
        inner_lo=nonfinite_lo,
        inner_hi=nonfinite_hi,
    )
    nonfinite_result = executor.evaluate([nonfinite])[nonfinite.request_id]
    if wrong_result.status != "unsupported_structure" or nonfinite_result.status != "nonfinite":
        raise AssertionError("failure policy matrix did not fail closed")

    probe = arithmetic_probe(
        [5e-324, -5e-324, 1e-160, -1e-160, 1.0, 1e30],
        [0.5, 0.5, 1e-160, 1e-160, 5e-324, -1e30],
    )
    return {
        "schema": "resident-tm-block-local-exact-checks-v1",
        "passed": True,
        "matrix": matrix,
        "totals": total,
        "retained_roundoff_witness": {
            "left_input_hex": float(1.1).hex(),
            "exact_error_numerator": exact_error.numerator,
            "exact_error_denominator": exact_error.denominator,
            "legacy_remainder": [
                float(legacy.remainder.lo).hex(),
                float(legacy.remainder.hi).hex(),
            ],
            "legacy_contains_exact_error": legacy_contains,
            "resident_coefficient_error": [
                float(resident.coefficient_error_lo[0, 2]).hex(),
                float(resident.coefficient_error_hi[0, 2]).hex(),
            ],
            "resident_contains_exact_error": True,
        },
        "failure_policy": {
            "wrong_fingerprint": wrong_result.status,
            "nonfinite_input": nonfinite_result.status,
            "hidden_cpu_recompute": False,
        },
        "primitive_probe": [
            [float(value).hex() for value in row] for row in probe
        ],
        "startup": startup,
        "independence": {
            "oracle_numeric_type": "fractions.Fraction",
            "legacy_cpu_output_used_as_truth": False,
            "random_trajectory_sampling_used": False,
            "fraction_or_high_precision_in_hot_path": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": result["passed"], **result["totals"]}))


if __name__ == "__main__":
    main()
