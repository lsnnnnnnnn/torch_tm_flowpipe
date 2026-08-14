#!/usr/bin/env python3
"""Independent exact/discrete verifier for the bounded G1 source ledger."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import FlowstarNormalFlowpipeState, Interval, TaylorModel, TMVector
from torch_tm_flowpipe.batched_dense_tm import (
    dense_picard_validate_step,
    dense_polynomial_picard,
    sparse_tmvector_to_dense,
)
from torch_tm_flowpipe.polynomial import Polynomial
from torch_tm_flowpipe.source_ledger import (
    BoundedSourceLedgerState,
    accepted_successor,
    affine_lift_interval,
    collapse_source_polynomial,
    commit_or_preserve,
    metadata_tamper,
    source_payload_hash,
)


def exact(value: Any) -> Fraction:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu())
    return Fraction.from_float(float(value))


def record(rows: list[dict[str, Any]], name: str, check: Callable[[], dict[str, Any]]) -> None:
    try:
        detail = check()
        rows.append({"oracle": name, "status": "PASS", **detail})
    except Exception as exc:
        rows.append({"oracle": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    unit = Interval(-1.0, 1.0)

    def affine_two_step() -> dict[str, Any]:
        z = Polynomial.variable(0, 1)
        value = (Polynomial.constant(1.0, 1) + z * 2.0) * 3.0 * -0.5
        require(exact(value.terms[(1,)]) == Fraction(-3), "two-step affine coefficient changed")
        return {"proof_class": "exact_rational", "coefficient": "-3"}

    record(rows, "affine_source_two_step_exact_coefficient", affine_two_step)

    def cancellation() -> dict[str, Any]:
        z = Polynomial.variable(0, 1)
        shared = (Polynomial.constant(1.0, 1) + z) - (Polynomial.constant(1.0, 1) + z)
        independent = (
            Polynomial.variable(0, 2) - Polynomial.variable(1, 2)
        ).evaluate_interval([unit, unit])
        require(not shared.terms, "shared-source cancellation was not exact")
        require(exact(independent.lo) <= -2 and exact(independent.hi) >= 2, "rebox excess missing")
        return {"proof_class": "exact_rational_plus_directed_interval", "legacy_width": 4.0}

    record(rows, "shared_source_cancellation_actual_polynomial", cancellation)

    def cubic() -> dict[str, Any]:
        z = Polynomial.variable(0, 1)
        poly = (Polynomial.constant(1.0, 1) + z) * (Polynomial.constant(1.0, 1) + z) * (Polynomial.constant(1.0, 1) + z)
        got = {exp[0]: exact(coef) for exp, coef in poly.terms.items()}
        require(got == {0: 1, 1: 3, 2: 3, 3: 1}, "x^2 y source paths did not merge")
        return {"proof_class": "exact_rational", "coefficients": [1, 3, 3, 1]}

    record(rows, "quadratic_cubic_x2y_shared_identity", cubic)

    def mixed_products() -> dict[str, Any]:
        structured = TaylorModel(
            Polynomial.constant(2.0, 1) + Polynomial.variable(0, 1) * 0.5,
            Interval(-0.2, 0.3), [unit], order=4,
        )
        ordinary = TaylorModel.constant(1.0, [unit], order=4, remainder=Interval(-0.4, 0.1))
        bound = (structured * ordinary).range_box()
        checked = 0
        for z in (-1, 1):
            for left_rem in (Fraction(-1, 5), Fraction(3, 10)):
                for right_rem in (Fraction(-2, 5), Fraction(1, 10)):
                    value = (Fraction(2) + Fraction(z, 2) + left_rem) * (1 + right_rem)
                    require(exact(bound.lo) <= value <= exact(bound.hi), "mixed-product corner escaped")
                    checked += 1
        return {"proof_class": "exact_rational_corners_plus_directed_interval", "corners": checked}

    record(rows, "ordinary_structured_nonlinear_asymmetric", mixed_products)

    def ownership() -> dict[str, Any]:
        domain = [unit, Interval(0.0, 0.1)]
        sparse = TMVector([TaylorModel(
            Polynomial.variable(0, 2) + Polynomial.variable(1, 2),
            Interval(-0.01, 0.02), domain, order=4,
        )])
        dense = sparse_tmvector_to_dense(sparse, order=4)
        fourth = dense.mul_trunc(dense).mul_trunc(dense).mul_trunc(dense)
        fifth = fourth.mul_trunc(dense)
        integrated = fourth.integrate(1)
        cut = fourth.apply_cutoff(10.0)
        for ledger, category in (
            (fifth.ledger, "polynomial_truncation"),
            (integrated.ledger, "integration_overflow"),
            (cut.ledger, "cutoff"),
        ):
            require(category in ledger.entries, f"missing {category} owner")
            require(bool(torch.any(ledger.entries[category][1] > 0)), f"empty {category} owner")
        return {"proof_class": "deterministic_dense_ledger", "owners": ["polynomial_truncation", "integration_overflow", "cutoff"]}

    record(rows, "degree4_truncation_integration_cutoff_ownership", ownership)

    def tau_merge() -> dict[str, Any]:
        u = Polynomial.variable(0, 2)
        tau = Polynomial.variable(1, 2)
        endpoint = (u * tau + u * 2).substitute_const(1, 3).drop_variable(1)
        require(list(endpoint.terms) == [(1,)], "endpoint has duplicate exponent")
        require(exact(endpoint.terms[(1,)]) == 5, "tau endpoint coefficient mismatch")
        return {"proof_class": "exact_rational", "merged_coefficient": 5}

    record(rows, "duplicate_exponent_merge_tau_endpoint", tau_merge)

    def retire() -> dict[str, Any]:
        u = Polynomial.variable(0, 2)
        z = Polynomial.variable(1, 2)
        poly = Polynomial.constant(2.0, 2) + u + u * z + z * Fraction(1, 4)
        witness = collapse_source_polynomial(poly, [unit, unit], [1])
        for u_value in (-1, 1):
            for z_value in (-1, 1):
                source_value = Fraction(u_value * z_value) + Fraction(z_value, 4)
                require(exact(witness.collapsed.lo) <= source_value <= exact(witness.collapsed.hi), "retire collapse escaped")
        return {"proof_class": "exact_rational_corners_plus_directed_interval", "source_support_sha256": witness.source_support_sha256}

    record(rows, "source_retire_collapse_containment", retire)

    def retry() -> dict[str, Any]:
        initial = BoundedSourceLedgerState.initial(2)
        proposed = accepted_successor(initial, torch.tensor([[0.1, 0.2]], dtype=torch.float64), ("picard_residual",))
        preserved = commit_or_preserve(initial, proposed, accepted=False)
        require(preserved is initial, "rejection changed state identity")
        require(preserved.fingerprint == initial.fingerprint, "rejection changed state hash")
        return {"proof_class": "deterministic_atomicity", "state_sha256": initial.fingerprint}

    record(rows, "retry_rejection_state_hash_unchanged", retry)

    def batch_permutation() -> dict[str, Any]:
        checked = []
        for batch in (1, 8, 64):
            generator = torch.Generator().manual_seed(20260814 + batch)
            lo = torch.rand((batch, 3), generator=generator, dtype=torch.float64) - 1
            hi = lo + torch.rand((batch, 3), generator=generator, dtype=torch.float64)
            perm = torch.arange(batch - 1, -1, -1)
            direct = affine_lift_interval(lo, hi)
            shuffled = affine_lift_interval(lo[perm], hi[perm])
            require(torch.equal(direct.midpoint[perm], shuffled.midpoint), "batch midpoint permutation failed")
            require(torch.equal(direct.radius[perm], shuffled.radius), "batch radius permutation failed")
            checked.append(batch)
        return {"proof_class": "deterministic_tensor_equivariance", "batches": checked}

    record(rows, "B1_B8_B64_permutation_equivariance", batch_permutation)

    def cuda_parity() -> dict[str, Any]:
        require(torch.cuda.is_available(), "CUDA unavailable")
        lo = torch.tensor([[-0.3, -1e-12], [0.1, 2.0]], dtype=torch.float64)
        hi = torch.tensor([[0.7, 2e-12], [0.10000000000000003, 3.0]], dtype=torch.float64)
        cpu = affine_lift_interval(lo, hi)
        gpu = affine_lift_interval(lo.cuda(), hi.cuda())
        require(torch.equal(cpu.contains_input, gpu.contains_input.cpu()), "CPU/CUDA decisions differ")
        require(bool(torch.all(gpu.represented_lo.cpu() <= lo)), "CUDA lower containment failed")
        require(bool(torch.all(gpu.represented_hi.cpu() >= hi)), "CUDA upper containment failed")
        return {"proof_class": "implementation_consistency_not_formal_rounding", "device": torch.cuda.get_device_name(0)}

    record(rows, "CPU_CUDA_same_containment_decision", cuda_parity)

    def exact_initial_consumer() -> dict[str, Any]:
        state = FlowstarNormalFlowpipeState.from_exact_decimal_box([("1.1", "1.4"), ("2.35", "2.45")], 4)
        reset = state.normalized_initial_tm(4)
        dense = sparse_tmvector_to_dense(reset.extend_domain(Interval(0.0, 0.01)), order=4)
        for center, scale, bounds in zip(state.center, state.scales, ((Fraction(11, 10), Fraction(7, 5)), (Fraction(47, 20), Fraction(49, 20)))):
            require(exact(center) - exact(scale) <= bounds[0], "exact lower endpoint escaped")
            require(exact(center) + exact(scale) >= bounds[1], "exact upper endpoint escaped")
        require(dense.poly.out_dim == 2, "exact reset was not consumed by dense path")
        return {"proof_class": "exact_rational_plus_actual_dense_conversion", "initialization_contract": state.diagnostics["initialization_contract"]}

    record(rows, "exact_decimal_compensation_actual_consumer", exact_initial_consumer)

    def observer_parity() -> dict[str, Any]:
        domain = [unit, Interval(0.0, 0.01)]
        sparse = TMVector([TaylorModel(Polynomial.variable(0, 2), Interval.zero(), domain, order=4)])
        base = sparse_tmvector_to_dense(sparse, order=4)
        observed: list[str] = []

        def rhs(value):
            return value.component(0).mul_trunc(value.component(0)).add(1.0)

        plain, plain_trace = dense_polynomial_picard(rhs, base, tau_index=1, order=4, cutoff_threshold=1e-12)

        def observer(iteration, pre_cutoff, retained):
            observed.append(hashlib.sha256(retained.poly.coeffs.detach().cpu().numpy().tobytes()).hexdigest())

        watched, watched_trace = dense_polynomial_picard(rhs, base, tau_index=1, order=4, cutoff_threshold=1e-12, observer=observer)
        require(torch.equal(plain.poly.coeffs, watched.poly.coeffs), "observer changed coefficients")
        require(plain_trace == watched_trace, "observer changed trace")
        require(len(observed) == 4, "observer missed an actual iterate")
        return {"proof_class": "bitwise_actual_consumer_parity", "coefficient_sha256": observed[-1]}

    record(rows, "observer_default_off_bitwise_parity", observer_parity)

    def tamper_consumer() -> dict[str, Any]:
        midpoint = torch.zeros((1, 2), dtype=torch.float64)
        radius = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
        state = accepted_successor(BoundedSourceLedgerState.initial(2), radius, ("picard_residual",))
        metadata = metadata_tamper(state, "audit")
        original_payload = source_payload_hash(midpoint, radius)
        require(state.fingerprint != metadata.fingerprint, "metadata tamper did not alter metadata hash")
        require(original_payload == source_payload_hash(midpoint, radius), "metadata entered consumer payload")
        changed = source_payload_hash(midpoint, torch.tensor([[0.1, 0.21]], dtype=torch.float64))
        require(changed != original_payload, "carry payload tamper did not alter consumer hash")
        return {"proof_class": "deterministic_actual_payload_hash", "payload_sha256": original_payload}

    record(rows, "tamper_actual_payload_vs_metadata", tamper_consumer)

    passed = all(row["status"] == "PASS" for row in rows)
    result = {
        "schema": "bounded_source_ledger_independent_oracle_v1",
        "candidate": "normalized_insertion_bounded_source_ledger_o4_g1",
        "passed": passed,
        "oracle_count": len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "oracle_count": len(rows)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
