#!/usr/bin/env python3
"""Independent plant-only proof-kernel audit for a clean flowstar_gpu tree.

The script imports and exercises the shipped implementation.  Exact rational
arithmetic (``fractions.Fraction``) is the oracle for binary operations and dot
products; mpmath is used only for square root.  It also owns the auditable
proof-to-code claim table so schema and source/test references are checked by
focused Torch-repository tests.
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import importlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Callable, Iterable


CSV_FIELDS = (
    "claim_id",
    "paper_section",
    "theorem_or_assumption",
    "mathematical_hypotheses",
    "source_file",
    "symbol_or_function",
    "kernel_or_operation",
    "runtime_assertion",
    "unit_test",
    "oracle",
    "status",
    "gap",
)
ALLOWED_STATUSES = {
    "MAPPED_AND_TESTED",
    "MAPPED_NOT_TESTED",
    "PARTIALLY_MAPPED",
    "SOURCE_MISSING",
    "ASSUMPTION_ONLY",
    "CONTRADICTED",
}


CLAIMS: tuple[dict[str, str], ...] = (
    {
        "claim_id": "FP_ELEMENTWISE_OUTWARD",
        "paper_section": "3 / Floating-point soundness without directed rounding",
        "theorem_or_assumption": "Proposition bg-outward; engine text preceding Assumption libulp",
        "mathematical_hypotheses": "binary64 correctly rounded operation; finite inputs/result; gradual underflow",
        "source_file": "src/flowstar_gpu/rounding.py; src/flowstar_gpu/interval.py; src/flowstar_gpu/transcendental.py; src/flowstar_gpu/cuda_kernels.py",
        "symbol_or_function": "next_up,next_down; add,sub,mul,div; sqrt_iv; iv_* CUDA kernels",
        "kernel_or_operation": "RN+nextafter CPU fallback or directed __dadd/__dmul CUDA endpoints",
        "runtime_assertion": "dtype f64 guards; division bad mask; certificate-boundary finite assertions and per-lane non-finite mask",
        "unit_test": "tests/properties/test_interval_props.py; tests/properties/test_transcendental_props.py; tests/unit/test_cuda_kernels.py; tests/soundness/test_strict_proof_contract.py; huan_proof_kernel_audit.py D1",
        "oracle": "Fraction exact endpoints and mpmath sqrt",
        "status": "MAPPED_AND_TESTED",
        "gap": "primitives may return extended intervals internally; only finite validated flowpipes are certificates",
    },
    {
        "claim_id": "FP_NO_FTZ_STARTUP",
        "paper_section": "3 / Floating-point soundness without directed rounding",
        "theorem_or_assumption": "equation engine-fpmodel startup condition",
        "mathematical_hypotheses": "float64 subnormals are preserved by every deployed path",
        "source_file": "src/flowstar_gpu/determinism.py; src/flowstar_gpu/flowpipe.py",
        "symbol_or_function": "assert_gradual_underflow; enable_determinism; reach",
        "kernel_or_operation": "startup and CUDA float64 arithmetic",
        "runtime_assertion": "CPU always checked; selected CUDA and active custom interval-multiply route additionally checked before reach",
        "unit_test": "tests/unit/test_denormal_ftz.py; tests/soundness/test_strict_proof_contract.py::test_no_ftz_startup_rejects_simulated_flush; huan_proof_kernel_audit.py no_ftz",
        "oracle": "smallest-subnormal identities",
        "status": "MAPPED_AND_TESTED",
        "gap": "direct low-level advance_sparse callers must invoke production initialization; public plant reach does so",
    },
    {
        "claim_id": "FP_OVERFLOW_DIVZERO_FAIL_CLOSED",
        "paper_section": "3 / Floating-point soundness without directed rounding",
        "theorem_or_assumption": "finite-intermediate hypotheses and fail-closed prose",
        "mathematical_hypotheses": "finite inputs; denominator excludes zero and reciprocal remains finite",
        "source_file": "src/flowstar_gpu/interval.py; src/flowstar_gpu/flowpipe.py",
        "symbol_or_function": "rec,div,assert_valid; _validate_and_refine",
        "kernel_or_operation": "reciprocal-then-multiply; contraction predicate",
        "runtime_assertion": "division/domain bad masks plus generic per-lane finite checks; assert_valid rejects infinite certificate endpoints",
        "unit_test": "tests/properties/test_interval_props.py; tests/soundness/test_strict_proof_contract.py; huan_proof_kernel_audit.py D1",
        "oracle": "exact extended-real enclosure and explicit bad-mask checks",
        "status": "MAPPED_AND_TESTED",
        "gap": "status value 3 is retained for compatibility and the audit trace distinguishes generic non-finite from domain failures",
    },
    {
        "claim_id": "FP_ANY_ORDER_REDUCTION",
        "paper_section": "3 / Theorem engine-fpsum and Corollary engine-order",
        "theorem_or_assumption": "Theorem 3.5 any-order inflated reductions",
        "mathematical_hypotheses": "finite intermediates; binary64 RN/FMA; m*u <= 1/4; no FTZ",
        "source_file": "src/flowstar_gpu/rounding.py; src/flowstar_gpu/interval.py",
        "symbol_or_function": "dot_error_bound; sum; dot_point_iv",
        "kernel_or_operation": "RN reductions plus symmetric a-priori inflation",
        "runtime_assertion": "m*U < 0.25 checked for int and tensor lengths",
        "unit_test": "tests/properties/test_rounding_props.py; huan_proof_kernel_audit.py D2",
        "oracle": "Fraction exact dot across sequential,pairwise,permuted,chunked,FMA,Torch schedules",
        "status": "MAPPED_AND_TESTED",
        "gap": "the theorem does not cover overflowing intermediates; fused directed CUDA interval kernels bypass this inflation",
    },
    {
        "claim_id": "FP_COMPUTABLE_INFLATION",
        "paper_section": "3 / Remark engine-computable",
        "theorem_or_assumption": "computable inflation of the absolute-product sum",
        "mathematical_hypotheses": "m < 2^51; finite RN absolute reduction; gradual underflow",
        "source_file": "src/flowstar_gpu/rounding.py",
        "symbol_or_function": "sum_error_bound; dot_error_bound",
        "kernel_or_operation": "(2m-1)u(1+2mu)*abs_dot_rn + m*eta with next_up",
        "runtime_assertion": "reduction length and f64 checked; downstream per-lane nonfinite mask and assert_valid reject a nonfinite certificate",
        "unit_test": "tests/properties/test_rounding_props.py; huan_proof_kernel_audit.py D2",
        "oracle": "Fraction exact error versus the shipped computed bound",
        "status": "MAPPED_AND_TESTED",
        "gap": "the primitive may return a nonfinite inflation on overflow; public validation classifies that lane nonfinite and cannot issue a finite certificate",
    },
    {
        "claim_id": "STRICT_VERSUS_PARITY",
        "paper_section": "3 / Remark engine-repro",
        "theorem_or_assumption": "strict charges point-coefficient roundoff; parity adopts Flow* trust model",
        "mathematical_hypotheses": "every retained point-coefficient reduction error is ranged into the remainder in strict mode",
        "source_file": "src/flowstar_gpu/config.py; src/flowstar_gpu/composition.py; src/flowstar_gpu/flowpipe.py; src/flowstar_gpu/symbolic_remainder.py",
        "symbol_or_function": "Settings.mode; compose(strict); _validate_and_refine; propagate",
        "kernel_or_operation": "composition GEMM inflation and validated point-vs-interval defect",
        "runtime_assertion": "strict Phi requires interval Phi; finite/m*u guards; coefficient errors ranged on the actual dense/sparse support",
        "unit_test": "tests/unit/test_composition.py; tests/unit/test_sparse_exec.py; tests/soundness/test_strict_proof_contract.py",
        "oracle": "Fraction Phi and dense/sparse convolution oracles plus source-path accounting audit",
        "status": "MAPPED_AND_TESTED",
        "gap": "parity deliberately preserves Flow*'s point-coefficient trust model and is not promoted to strict soundness",
    },
    {
        "claim_id": "LANE_CERT_MASK_FREEZE_RETRY_REFINE",
        "paper_section": "3 / Batched flowpipe construction",
        "theorem_or_assumption": "per-lane contraction certificates and member-separable pinned regions",
        "mathematical_hypotheses": "each accepted lane has its own closed subset check",
        "source_file": "src/flowstar_gpu/flowpipe.py; src/flowstar_gpu/sparse_exec.py",
        "symbol_or_function": "_validate_and_refine; advance; advance_adaptive; refine_loop",
        "kernel_or_operation": "per-lane masks, torch.where freeze, halving retries",
        "runtime_assertion": "status and ok/bad masks",
        "unit_test": "tests/unit/test_flowpipe.py; tests/unit/test_sparse_exec.py; tests/adversarial/test_batch_invariance.py",
        "oracle": "solo-vs-batch bit equality inside pinned paths and explicit status transitions",
        "status": "MAPPED_AND_TESTED",
        "gap": "bitwise claims remain conditional on the fused/fixed-reduction path",
    },
    {
        "claim_id": "REFINE_FIRST_MAP_NOT_RESCUED",
        "paper_section": "3 / Batched flowpipe construction",
        "theorem_or_assumption": "refinement begins only after a successful initial self-map",
        "mathematical_hypotheses": "ok is the initial closed subset certificate",
        "source_file": "src/flowstar_gpu/flowpipe.py",
        "symbol_or_function": "_validate_and_refine; refine_loop",
        "kernel_or_operation": "refining = ok.clone()",
        "runtime_assertion": "none beyond the ok mask",
        "unit_test": "tests/unit/test_flowpipe.py; tests/unit/test_tape_kernels.py; audit refinement control tests",
        "oracle": "call-count and state-freeze assertions",
        "status": "MAPPED_AND_TESTED",
        "gap": "CUDA fused execution is cross-checked against the shared eager semantics; audit callbacks intentionally select the observable eager route",
    },
    {
        "claim_id": "SPARSE_SUPPORT_SUPERSET",
        "paper_section": "3 / Proposition engine-support",
        "theorem_or_assumption": "declared support contains every nonzero coefficient",
        "mathematical_hypotheses": "support identifiers sorted; constant id 0 present; omitted slots are exact zeros",
        "source_file": "src/flowstar_gpu/support.py; src/flowstar_gpu/sparse_exec.py",
        "symbol_or_function": "Support; make_support; pair_tables; _accum_sups",
        "kernel_or_operation": "support-specialized gather/reduction schedules",
        "runtime_assertion": "support constructor invariants",
        "unit_test": "tests/unit/test_sparse_support.py; tests/unit/test_sparse_exec.py; tests/soundness/test_sparse_containment.py",
        "oracle": "dense embedding, exact Fraction cases, and direct containment",
        "status": "MAPPED_AND_TESTED",
        "gap": "CUDA dense/sparse point coefficients carry ulp-level, not bitwise, agreement",
    },
    {
        "claim_id": "MEMBER_CHUNKING_SCOPE",
        "paper_section": "3 / Proposition engine-chunk",
        "theorem_or_assumption": "all kernels in a claimed region are member-separable",
        "mathematical_hypotheses": "per-member schedule does not depend on batch extent",
        "source_file": "src/flowstar_gpu/cuda_kernels.py; src/flowstar_gpu/safety.py; src/flowstar_gpu/graphing.py",
        "symbol_or_function": "seg_* kernels; _spatial_images; GraphCache",
        "kernel_or_operation": "fixed in-member reductions and member-axis slicing",
        "runtime_assertion": "CUDA kernel availability gates GraphCache; no universal member-separability assertion",
        "unit_test": "tests/unit/test_cuda_kernels.py; tests/unit/test_safety.py::test_spatial_images_member_chunking_is_bitwise_neutral; tests/adversarial/test_batch_invariance.py",
        "oracle": "B=1 embedded in B>1 and multiple forced chunk sizes",
        "status": "MAPPED_AND_TESTED",
        "gap": "fallback torch.segment_reduce is shape-sensitive; the bitwise gate passed only after fused kernels actually built",
    },
    {
        "claim_id": "NONLINEAR_HISTORY_COMPOSITION",
        "paper_section": "3 / Batched flowpipe construction",
        "theorem_or_assumption": "history remainder is substituted through nonlinear flowmaps",
        "mathematical_hypotheses": "Taylor-model multiplication remainder algebra encloses every cross term",
        "source_file": "src/flowstar_gpu/composition.py; src/flowstar_gpu/sparse_exec.py",
        "symbol_or_function": "compose; compose_s",
        "kernel_or_operation": "I1*I2 + P2*I1 + P1*I2 at every monomial-image DAG node",
        "runtime_assertion": "optional constant-removed precondition",
        "unit_test": "tests/unit/test_composition.py; tests/unit/test_sparse_exec.py",
        "oracle": "Fraction evaluation and sampled Taylor-model containment",
        "status": "MAPPED_AND_TESTED",
        "gap": "strict widens the ordinary remainder; parity intentionally omits the additional coefficient-error range",
    },
    {
        "claim_id": "SYMBOLIC_QUEUE_RECONSTRUCT_CLEAR",
        "paper_section": "3 / Symbolic remainder propagation",
        "theorem_or_assumption": "full remainder is emitted before queue clearing; reset reanchors composition",
        "mathematical_hypotheses": "Phi/J pairing and reset placement remain synchronized",
        "source_file": "src/flowstar_gpu/symbolic_remainder.py; src/flowstar_gpu/flowpipe.py; src/flowstar_gpu/sparse_exec.py",
        "symbol_or_function": "propagate; append_j; reset_if_full; advance",
        "kernel_or_operation": "Phi queue products and interval J reconstruction",
        "runtime_assertion": "queue overflow guard; >= capacity reset; strict propagation requires exact-enclosing interval Phi/scalars",
        "unit_test": "tests/unit/test_flowpipe.py; tests/unit/test_sparse_exec.py; tests/soundness/test_containment.py; tests/soundness/test_strict_proof_contract.py",
        "oracle": "Fraction matrix/J reconstruction at sizes 1--3 and all queue/reset boundaries",
        "status": "MAPPED_AND_TESTED",
        "gap": "phi_buf remains the parity point path; phi_iv_buf is allocated only for strict mode",
    },
    {
        "claim_id": "TRANSCENDENTAL_ASSUMPTIONS",
        "paper_section": "3 / Assumption engine-libulp",
        "theorem_or_assumption": "exp/log <=3 ulp and sin/cos <=4 ulp on deployed libraries",
        "mathematical_hypotheses": "calibrated library/version remains within the stated budget",
        "source_file": "src/flowstar_gpu/transcendental.py; tests/properties/test_transcendental_props.py",
        "symbol_or_function": "exp_iv,log_iv,sin_iv,cos_iv",
        "kernel_or_operation": "library evaluation plus calibrated nextafter chain",
        "runtime_assertion": "no automatic startup calibration in production entrypoint",
        "unit_test": "tests/properties/test_transcendental_props.py; tests/unit/test_tape_kernels.py",
        "oracle": "mpmath calibration grid",
        "status": "ASSUMPTION_ONLY",
        "gap": "conditional and tested out of band; not a theorem or startup assertion",
    },
    {
        "claim_id": "POLYNOMIAL_ONLY_UNCONDITIONAL",
        "paper_section": "3 / floating-point scope statement",
        "theorem_or_assumption": "polynomial ODE uses only unconditional arithmetic claims",
        "mathematical_hypotheses": "no FTZ; finite intermediates; all point-coefficient roundoff charged in strict mode",
        "source_file": "src/flowstar_gpu/flowpipe.py; src/flowstar_gpu/composition.py; src/flowstar_gpu/symbolic_remainder.py",
        "symbol_or_function": "reach/advance; compose; propagate",
        "kernel_or_operation": "polynomial Picard, composition, and optional symbolic remainder",
        "runtime_assertion": "f64/length guards, production no-FTZ assertion, generic finite-lane mask, and strict coefficient accounting",
        "unit_test": "plant-only upstream suite; tests/soundness/test_strict_proof_contract.py; huan_proof_kernel_audit.py",
        "oracle": "proof-to-code closure audit with Fraction coefficient/Phi witnesses",
        "status": "MAPPED_AND_TESTED",
        "gap": "promotion applies only to strict polynomial plant reach under enforced finite hypotheses; transcendental library claims remain conditional",
    },
)


def write_proof_map(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(CLAIMS)


def validate_proof_map(path: Path, engine_root: Path) -> list[str]:
    errors: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            errors.append("proof map header does not match the required schema")
            return errors
        rows = list(reader)
    if len(rows) < 14:
        errors.append(f"proof map has {len(rows)} claims; expected at least 14")
    seen: set[str] = set()
    for number, row in enumerate(rows, 2):
        if row["claim_id"] in seen:
            errors.append(f"row {number}: duplicate claim_id {row['claim_id']}")
        seen.add(row["claim_id"])
        if row["status"] not in ALLOWED_STATUSES:
            errors.append(f"row {number}: invalid status {row['status']}")
        for source in row["source_file"].split("; "):
            if source.startswith("src/") and not (engine_root / source).is_file():
                errors.append(f"row {number}: missing source {source}")
        if not row["gap"]:
            errors.append(f"row {number}: empty gap")
    return errors


def _exact_dot(a: Iterable[float], b: Iterable[float]) -> Fraction:
    return sum((Fraction(x) * Fraction(y) for x, y in zip(a, b, strict=True)), Fraction())


def _sequential(values: list[float]) -> float:
    acc = 0.0
    for value in values:
        acc = float(acc + value)
    return acc


def _pairwise(values: list[float]) -> float:
    work = list(values)
    if not work:
        return 0.0
    while len(work) > 1:
        nxt = [float(work[i] + work[i + 1]) for i in range(0, len(work) - 1, 2)]
        if len(work) % 2:
            nxt.append(work[-1])
        work = nxt
    return work[0]


def _chunked(values: list[float], size: int) -> float:
    return _pairwise([_sequential(values[i : i + size]) for i in range(0, len(values), size)])


def _fma_dot(a: list[float], b: list[float]) -> float:
    if not hasattr(math, "fma"):
        return _sequential([float(x * y) for x, y in zip(a, b, strict=True)])
    acc = 0.0
    for x, y in zip(a, b, strict=True):
        acc = math.fma(x, y, acc)
    return acc


def reduction_cases(seed: int = 20260826) -> list[tuple[str, list[float], list[float]]]:
    rng = random.Random(seed)
    cases: list[tuple[str, list[float], list[float]]] = [
        ("m1", [0.1], [0.2]),
        ("m2", [1e307, -1e307], [1.0, 1.0]),
        ("m3-cancel", [1e16, 1.0, -1e16], [1.0, 1.0, 1.0]),
        ("underflow", [5e-324] * 8, [0.5] * 8),
        ("mixed", [1e150, -1e150, 1e-150, -1e-150], [1e-150, 1e-150, 1e150, 1e150]),
    ]
    for m in (31, 32, 33, 63, 64, 65, 257, 4097):
        a = [math.ldexp(rng.uniform(-1.0, 1.0), rng.randint(-450, 450)) for _ in range(m)]
        b = [math.ldexp(rng.uniform(-1.0, 1.0), rng.randint(-450, 450)) for _ in range(m)]
        cases.append((f"random-m{m}", a, b))
    return cases


def _contains_fraction(lo: float, hi: float, exact: Fraction) -> bool:
    return not math.isnan(lo) and not math.isnan(hi) and Fraction(lo) <= exact <= Fraction(hi)


def _audit_d1(torch: Any, iv: Any, trans: Any, device: str) -> dict[str, Any]:
    import mpmath as mp

    mp.mp.dps = 200
    rows: list[dict[str, Any]] = []

    def tensor(pair: tuple[float, float]) -> Any:
        return torch.tensor(pair, dtype=torch.float64, device=device)

    def binary(name: str, fn: Callable[[Any, Any], Any], a: tuple[float, float], b: tuple[float, float], op: Callable[[Fraction, Fraction], Fraction]) -> None:
        out = fn(tensor(a), tensor(b))
        exact = [op(Fraction(x), Fraction(y)) for x in a for y in b]
        lo, hi = out.tolist()
        rows.append({"case": name, "pass": _contains_fraction(lo, hi, min(exact)) and _contains_fraction(lo, hi, max(exact)), "out": [lo, hi]})

    binary("add-ordinary", iv.add, (0.1, 0.2), (0.3, 0.4), lambda x, y: x + y)
    binary("sub-cancellation", iv.sub, (1.0, 1.0), (1.0, 1.0), lambda x, y: x - y)
    binary("mul-signed-zero", iv.mul, (-0.0, 0.0), (-3.0, 4.0), lambda x, y: x * y)
    binary("mul-subnormal", iv.mul, (5e-324, 1e-320), (0.5, 2.0), lambda x, y: x * y)

    div_out, div_bad = iv.div(tensor((1.0, 2.0)), tensor((-1.0, 1.0)))
    rows.append({"case": "div-zero-containing", "pass": bool(div_bad.item()) and div_out.tolist() == [-math.inf, math.inf], "out": div_out.tolist()})

    sqrt_out, sqrt_bad = trans.sqrt_iv(tensor((5e-324, 1.7976931348623157e308)))
    sqrt_lo, sqrt_hi = sqrt_out.tolist()
    exact_lo = mp.sqrt(mp.mpf(5e-324))
    exact_hi = mp.sqrt(mp.mpf(1.7976931348623157e308))
    rows.append({"case": "sqrt-subnormal-max", "pass": not bool(sqrt_bad.item()) and mp.mpf(sqrt_lo) <= exact_lo and exact_hi <= mp.mpf(sqrt_hi), "out": sqrt_out.tolist()})

    overflow = iv.mul(tensor((1e300, 1.5e308)), tensor((1e300, 1.5e308)))
    finite_probe = tensor((-1.0, 1.0))
    finite_certificate_rejected = not bool(iv.contains(overflow, finite_probe).item())
    try:
        iv.assert_valid(overflow)
    except FloatingPointError:
        assert_valid_rejected = True
    else:
        assert_valid_rejected = False
    rows.append({
        "case": "mul-overflow-extended",
        "pass": (
            not bool(torch.isnan(overflow).any())
            and math.isinf(overflow[1].item())
            and finite_certificate_rejected
            and assert_valid_rejected
        ),
        "out": overflow.tolist(),
        "finite_certificate_rejected": finite_certificate_rejected,
        "assert_valid_rejected": assert_valid_rejected,
        "classification": "NONFINITE_EXTENDED_ENCLOSURE_NOT_A_FINITE_CERTIFICATE",
    })

    tiny = torch.tensor(1e-310, dtype=torch.float64, device=device)
    no_ftz = (tiny * 0.5).item() == 5e-311 and torch.nextafter(torch.zeros_like(tiny), torch.full_like(tiny, math.inf)).item() == 5e-324
    return {"cases": rows, "case_count": len(rows), "passed": sum(bool(row["pass"]) for row in rows), "no_ftz_observed": no_ftz}


def _route_row(
    *,
    case_name: str,
    m: int,
    schedule_name: str,
    execution_backend: str,
    actual_device: str,
    kernel_path: str,
    kernel_invocation_observed: bool,
    exact: Fraction,
    hat: float | None,
    abs_hat: float | None,
    interval: tuple[float, float] | None,
    rounding: Any,
    torch: Any,
    bound_device: str,
) -> dict[str, Any]:
    """Build one route-tagged D2 result.

    Host schedules are mathematical any-order witnesses.  Device routes name
    the operation that actually executed; an enclosing interval route records
    containment directly, while a point route applies the shipped inflation.
    """
    finite = (
        interval is not None
        and all(math.isfinite(value) for value in interval)
    ) or (
        interval is None
        and hat is not None
        and abs_hat is not None
        and math.isfinite(hat)
        and math.isfinite(abs_hat)
    )
    base: dict[str, Any] = {
        "case": case_name,
        "m": m,
        "schedule_name": schedule_name,
        "execution_backend": execution_backend,
        "actual_device": actual_device,
        "kernel_path": kernel_path,
        "kernel_invocation_observed": kernel_invocation_observed,
        "finite_hypotheses_satisfied": finite,
        "m_u_gate": m * rounding.U <= 0.25,
    }
    if not finite:
        return {
            **base,
            "status": "OUTSIDE_FINITE_INTERMEDIATE_HYPOTHESIS",
            "exact_error": None,
            "computed_inflation": None,
            "containment": None,
        }
    if interval is not None:
        lo, hi = interval
        contained = _contains_fraction(lo, hi, exact)
        return {
            **base,
            "status": "PASS" if contained else "FAIL",
            "interval": [lo, hi],
            "exact_error": None,
            "computed_inflation": hi - lo,
            "containment": contained,
        }
    assert hat is not None and abs_hat is not None
    bound = rounding.dot_error_bound(
        torch.tensor(abs_hat, dtype=torch.float64, device=bound_device), m
    ).item()
    error = abs(Fraction(hat) - exact)
    contained = error <= Fraction(bound)
    return {
        **base,
        "status": "PASS" if contained else "FAIL",
        "hat": hat,
        "exact_error": float(error),
        "computed_inflation": bound,
        "containment": contained,
    }


def _audit_d2(torch: Any, rounding: Any, iv: Any, ck: Any, device: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    rng = random.Random(20260826)
    all_cases = reduction_cases()
    for _ in range(128):
        m = rng.randint(1, 32)
        a = [math.ldexp(rng.uniform(-1.0, 1.0), rng.randint(-500, 500)) for _ in range(m)]
        b = [math.ldexp(rng.uniform(-1.0, 1.0), rng.randint(-500, 500)) for _ in range(m)]
        all_cases.append((f"search-{len(all_cases)}", a, b))

    for case_name, a, b in all_cases:
        m = len(a)
        exact = _exact_dot(a, b)
        rounded_products = [float(x * y) for x, y in zip(a, b, strict=True)]
        abs_products = [abs(value) for value in rounded_products]
        perm = list(range(m))
        rng.shuffle(perm)
        schedules: dict[str, tuple[float, float]] = {
            "host_sequential": (_sequential(rounded_products), _sequential(abs_products)),
            "host_pairwise": (_pairwise(rounded_products), _pairwise(abs_products)),
            "host_chunk3": (_chunked(rounded_products, 3), _chunked(abs_products, 3)),
            "host_chunk17": (_chunked(rounded_products, 17), _chunked(abs_products, 17)),
            "host_permuted": (_pairwise([rounded_products[i] for i in perm]), _pairwise([abs_products[i] for i in perm])),
            "host_fma": (_fma_dot(a, b), _pairwise(abs_products)),
        }
        for schedule, (hat, abs_hat) in schedules.items():
            row = _route_row(
                case_name=case_name,
                m=m,
                schedule_name=schedule,
                execution_backend="host_python",
                actual_device="host_python_float",
                kernel_path=f"scripts.huan_proof_kernel_audit.{schedule}",
                kernel_invocation_observed=True,
                exact=exact,
                hat=hat,
                abs_hat=abs_hat,
                interval=None,
                rounding=rounding,
                torch=torch,
                bound_device="cpu",
            )
            rows.append(row)
            if row["status"] == "FAIL":
                failures.append(row)

        ta = torch.tensor(a, dtype=torch.float64, device=device)
        tb = torch.tensor(b, dtype=torch.float64, device=device)
        backend = "torch_cuda" if ta.is_cuda else "torch_cpu"
        actual_device = str(ta.device)

        device_routes: list[tuple[str, float, float, str]] = []
        dot_hat = torch.dot(ta, tb)
        dot_abs = torch.dot(ta.abs(), tb.abs())
        sum_hat = torch.sum(ta * tb)
        sum_abs = torch.sum(ta.abs() * tb.abs())
        einsum_hat = torch.einsum("i,i->", ta, tb)
        einsum_abs = torch.einsum("i,i->", ta.abs(), tb.abs())
        if ta.is_cuda:
            torch.cuda.synchronize(ta.device)
        device_routes.extend(
            [
                ("torch_dot", dot_hat.item(), dot_abs.item(), "torch.dot"),
                ("torch_sum_mul", sum_hat.item(), sum_abs.item(), "torch.sum(a*b)"),
                ("engine_point_einsum", einsum_hat.item(), einsum_abs.item(), "torch.einsum(i,i->)"),
            ]
        )
        for schedule, hat, abs_hat, kernel_path in device_routes:
            row = _route_row(
                case_name=case_name,
                m=m,
                schedule_name=schedule,
                execution_backend=backend,
                actual_device=actual_device,
                kernel_path=kernel_path,
                kernel_invocation_observed=True,
                exact=exact,
                hat=hat,
                abs_hat=abs_hat,
                interval=None,
                rounding=rounding,
                torch=torch,
                bound_device=device,
            )
            rows.append(row)
            if row["status"] == "FAIL":
                failures.append(row)

        # Production interval-dot route.  On CUDA this dispatches through
        # flowstar_gpu.interval._ck.iv_dot_point_iv; a narrowly scoped wrapper
        # proves the custom operation was invoked instead of merely available.
        custom_calls = 0
        original_custom = ck.iv_dot_point_iv

        def counted_custom(*args: Any, **kwargs: Any) -> Any:
            nonlocal custom_calls
            custom_calls += 1
            return original_custom(*args, **kwargs)

        if ta.is_cuda:
            ck.iv_dot_point_iv = counted_custom
        try:
            engine_iv = iv.dot_point_iv(ta, iv.from_point(tb), dim=-1)
            if ta.is_cuda:
                torch.cuda.synchronize(ta.device)
        finally:
            if ta.is_cuda:
                ck.iv_dot_point_iv = original_custom
        engine_interval = tuple(float(value) for value in engine_iv.tolist())
        engine_backend = "custom_cuda" if custom_calls else backend
        row = _route_row(
            case_name=case_name,
            m=m,
            schedule_name="engine_interval_dot",
            execution_backend=engine_backend,
            actual_device=actual_device,
            kernel_path="flowstar_gpu.interval.dot_point_iv",
            kernel_invocation_observed=(custom_calls > 0) if ta.is_cuda else True,
            exact=exact,
            hat=None,
            abs_hat=None,
            interval=engine_interval,
            rounding=rounding,
            torch=torch,
            bound_device=device,
        )
        row["custom_cuda_invocation_count"] = custom_calls
        rows.append(row)
        if row["status"] == "FAIL":
            failures.append(row)
    checked = sum(row["status"] in {"PASS", "FAIL"} for row in rows)
    route_counts: dict[str, int] = {}
    invocation_counts: dict[str, int] = {}
    for row in rows:
        backend = row["execution_backend"]
        route_counts[backend] = route_counts.get(backend, 0) + 1
        if row["kernel_invocation_observed"]:
            invocation_counts[backend] = invocation_counts.get(backend, 0) + 1
    return {
        "rows": rows,
        "checked": checked,
        "passed": sum(row["status"] == "PASS" for row in rows),
        "failures": failures,
        "route_counts": route_counts,
        "kernel_invocation_counts": invocation_counts,
    }


def run(engine_root: Path, device: str) -> dict[str, Any]:
    sys.path.insert(0, str(engine_root / "src"))
    torch = importlib.import_module("torch")
    iv = importlib.import_module("flowstar_gpu.interval")
    trans = importlib.import_module("flowstar_gpu.transcendental")
    rounding = importlib.import_module("flowstar_gpu.rounding")
    ck = importlib.import_module("flowstar_gpu.cuda_kernels")
    determinism = importlib.import_module("flowstar_gpu.determinism")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    kernel_available = bool(ck.available()) if device == "cuda" else False
    determinism.assert_gradual_underflow(device)
    d1 = _audit_d1(torch, iv, trans, device)
    d2 = _audit_d2(torch, rounding, iv, ck, device)
    return {
        "schema": "torch_tm_flowpipe.huan_proof_kernel_audit/2",
        "engine_root": str(engine_root.resolve()),
        "engine_head": _git_head(engine_root),
        "device": device,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_kernel_available": kernel_available,
        "production_no_ftz_assertion_passed": True,
        "d1": d1,
        "d2": d2,
        "gate_passed": (
            d1["passed"] == d1["case_count"]
            and d1["no_ftz_observed"]
            and not d2["failures"]
        ),
    }


def _git_head(repo: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


D2_ROUTE_INVENTORY: tuple[dict[str, str], ...] = (
    {
        "schedule_name": "host_sequential|host_pairwise|host_chunk3|host_chunk17|host_permuted|host_fma",
        "execution_backend": "host_python",
        "kernel_path": "Python binary64 helpers",
        "evidence_scope": "any-order mathematical oracle; never CPU/CUDA kernel evidence",
    },
    {
        "schedule_name": "torch_dot",
        "execution_backend": "torch_cpu|torch_cuda",
        "kernel_path": "torch.dot",
        "evidence_scope": "actual point reduction on the tagged tensor device",
    },
    {
        "schedule_name": "torch_sum_mul",
        "execution_backend": "torch_cpu|torch_cuda",
        "kernel_path": "torch.sum(a*b)",
        "evidence_scope": "actual product then reduction on the tagged tensor device",
    },
    {
        "schedule_name": "engine_point_einsum",
        "execution_backend": "torch_cpu|torch_cuda",
        "kernel_path": "torch.einsum(i,i->)",
        "evidence_scope": "point-reduction primitive used by engine composition paths",
    },
    {
        "schedule_name": "engine_interval_dot",
        "execution_backend": "torch_cpu|custom_cuda",
        "kernel_path": "flowstar_gpu.interval.dot_point_iv",
        "evidence_scope": "actual production interval-dot route; custom CUDA requires a counted dispatch",
    },
)


def write_d2_outputs(payload: dict[str, Any], output_root: Path) -> None:
    """Write the corrected route-separated evidence requested by Phase 1."""
    output_root.mkdir(parents=True, exist_ok=True)
    inventory = output_root / "d2_schedule_inventory.csv"
    with inventory.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(D2_ROUTE_INVENTORY[0]))
        writer.writeheader()
        writer.writerows(D2_ROUTE_INVENTORY)

    d2 = payload["d2"]
    common = {
        "schema": "torch_tm_flowpipe.huan_d2_route_evidence/1",
        "engine_head": payload["engine_head"],
        "torch": payload["torch"],
        "cuda_runtime": payload["cuda_runtime"],
    }
    host_rows = [row for row in d2["rows"] if row["execution_backend"] == "host_python"]
    actual_rows = [row for row in d2["rows"] if row["execution_backend"] != "host_python"]
    if payload["device"] == "cpu":
        host_payload = {
            **common,
            "scope": "host binary64 reduction-order oracle; not device-kernel evidence",
            "rows": host_rows,
            "checked": sum(row["status"] in {"PASS", "FAIL"} for row in host_rows),
            "passed": sum(row["status"] == "PASS" for row in host_rows),
        }
        (output_root / "d2_host_order_oracle.json").write_text(
            json.dumps(host_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    actual_payload = {
        **common,
        "scope": f"actual {payload['device']} routes only",
        "rows": actual_rows,
        "checked": sum(row["status"] in {"PASS", "FAIL"} for row in actual_rows),
        "passed": sum(row["status"] == "PASS" for row in actual_rows),
        "kernel_invocation_counts": d2["kernel_invocation_counts"],
    }
    (output_root / f"d2_actual_{payload['device']}.json").write_text(
        json.dumps(actual_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proof-map", type=Path, required=True)
    parser.add_argument("--d2-output-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    write_proof_map(args.proof_map)
    map_errors = validate_proof_map(args.proof_map, args.engine_root)
    payload = run(args.engine_root, args.device)
    payload["proof_map_errors"] = map_errors
    payload["gate_passed"] = payload["gate_passed"] and not map_errors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.d2_output_root is not None:
        write_d2_outputs(payload, args.d2_output_root)
    print(json.dumps({"device": args.device, "gate_passed": payload["gate_passed"], "d1": [payload["d1"]["passed"], payload["d1"]["case_count"]], "d2": [payload["d2"]["passed"], payload["d2"]["checked"]], "kernel": payload["cuda_kernel_available"]}, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
