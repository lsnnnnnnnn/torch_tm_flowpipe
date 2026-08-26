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
        "runtime_assertion": "dtype f64 guards; division bad mask; no general finite-output guard at primitive boundary",
        "unit_test": "tests/properties/test_interval_props.py; tests/properties/test_transcendental_props.py; tests/unit/test_cuda_kernels.py; huan_proof_kernel_audit.py D1",
        "oracle": "Fraction exact endpoints and mpmath sqrt",
        "status": "MAPPED_AND_TESTED",
        "gap": "overflow returns an extended interval; fail-closed status is supplied only by division or downstream contraction, not every primitive",
    },
    {
        "claim_id": "FP_NO_FTZ_STARTUP",
        "paper_section": "3 / Floating-point soundness without directed rounding",
        "theorem_or_assumption": "equation engine-fpmodel startup condition",
        "mathematical_hypotheses": "float64 subnormals are preserved by every deployed path",
        "source_file": "src/flowstar_gpu/determinism.py; tests/unit/test_denormal_ftz.py",
        "symbol_or_function": "enable_determinism; denormal tests",
        "kernel_or_operation": "startup and CUDA float64 arithmetic",
        "runtime_assertion": "none in production startup",
        "unit_test": "tests/unit/test_denormal_ftz.py; huan_proof_kernel_audit.py no_ftz",
        "oracle": "smallest-subnormal identities",
        "status": "CONTRADICTED",
        "gap": "paper says the engine asserts no-FTZ at startup; source only tests it out of band",
    },
    {
        "claim_id": "FP_OVERFLOW_DIVZERO_FAIL_CLOSED",
        "paper_section": "3 / Floating-point soundness without directed rounding",
        "theorem_or_assumption": "finite-intermediate hypotheses and fail-closed prose",
        "mathematical_hypotheses": "finite inputs; denominator excludes zero and reciprocal remains finite",
        "source_file": "src/flowstar_gpu/interval.py; src/flowstar_gpu/flowpipe.py",
        "symbol_or_function": "rec,div,assert_valid; _validate_and_refine",
        "kernel_or_operation": "reciprocal-then-multiply; contraction predicate",
        "runtime_assertion": "division returns bad mask; contraction rejects non-contained extended results",
        "unit_test": "tests/properties/test_interval_props.py; huan_proof_kernel_audit.py D1",
        "oracle": "exact extended-real enclosure and explicit bad-mask checks",
        "status": "PARTIALLY_MAPPED",
        "gap": "division is explicit; generic overflow is not converted to a primitive status and assert_valid permits infinity",
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
        "runtime_assertion": "reduction length and f64 checked; abs_dot finiteness not checked",
        "unit_test": "tests/properties/test_rounding_props.py; huan_proof_kernel_audit.py D2",
        "oracle": "Fraction exact error versus the shipped computed bound",
        "status": "MAPPED_AND_TESTED",
        "gap": "an overflowing absolute reduction yields a non-finite bound rather than an explicit status",
    },
    {
        "claim_id": "STRICT_VERSUS_PARITY",
        "paper_section": "3 / Remark engine-repro",
        "theorem_or_assumption": "strict charges point-coefficient roundoff; parity adopts Flow* trust model",
        "mathematical_hypotheses": "every retained point-coefficient reduction error is ranged into the remainder in strict mode",
        "source_file": "src/flowstar_gpu/config.py; src/flowstar_gpu/composition.py; src/flowstar_gpu/flowpipe.py; src/flowstar_gpu/symbolic_remainder.py",
        "symbol_or_function": "Settings.mode; compose(strict); _validate_and_refine; propagate",
        "kernel_or_operation": "composition GEMM inflation and validated point-vs-interval defect",
        "runtime_assertion": "mode enum only",
        "unit_test": "tests/unit/test_composition.py; tests/unit/test_sparse_exec.py",
        "oracle": "strict remainder superset of parity plus source-path accounting audit",
        "status": "CONTRADICTED",
        "gap": "strict does not charge symbolic Phi einsums and does not visibly charge retained monomial-image point-product roundoff",
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
        "gap": "no public refinement ledger records every proposal/commit",
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
        "gap": "strict accounting of the retained point-product coefficients is incomplete",
    },
    {
        "claim_id": "SYMBOLIC_QUEUE_RECONSTRUCT_CLEAR",
        "paper_section": "3 / Symbolic remainder propagation",
        "theorem_or_assumption": "full remainder is emitted before queue clearing; reset reanchors composition",
        "mathematical_hypotheses": "Phi/J pairing and reset placement remain synchronized",
        "source_file": "src/flowstar_gpu/symbolic_remainder.py; src/flowstar_gpu/flowpipe.py; src/flowstar_gpu/sparse_exec.py",
        "symbol_or_function": "propagate; append_j; reset_if_full; advance",
        "kernel_or_operation": "Phi queue products and interval J reconstruction",
        "runtime_assertion": "queue overflow guard; >= capacity reset",
        "unit_test": "tests/unit/test_flowpipe.py; tests/unit/test_sparse_exec.py; tests/soundness/test_containment.py",
        "oracle": "reset-boundary replay against a non-reset queue and containment trajectories",
        "status": "PARTIALLY_MAPPED",
        "gap": "queue semantics are tested, but strict-mode Phi-product roundoff is uncharged",
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
        "runtime_assertion": "f64 and selected length guards only",
        "unit_test": "plant-only upstream suite and huan_proof_kernel_audit.py",
        "oracle": "proof-to-code closure audit",
        "status": "CONTRADICTED",
        "gap": "unconditional scope is not established when no-FTZ is not asserted and strict point-roundoff coverage has identified holes",
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
    rows.append({"case": "mul-overflow-extended", "pass": not bool(torch.isnan(overflow).any()) and math.isinf(overflow[1].item()), "out": overflow.tolist(), "classification": "NONFINITE_EXTENDED_ENCLOSURE_NOT_A_FINITE_CERTIFICATE"})

    tiny = torch.tensor(1e-310, dtype=torch.float64, device=device)
    no_ftz = (tiny * 0.5).item() == 5e-311 and torch.nextafter(torch.zeros_like(tiny), torch.full_like(tiny, math.inf)).item() == 5e-324
    return {"cases": rows, "case_count": len(rows), "passed": sum(bool(row["pass"]) for row in rows), "no_ftz_observed": no_ftz}


def _audit_d2(torch: Any, rounding: Any, device: str) -> dict[str, Any]:
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
            "sequential": (_sequential(rounded_products), _sequential(abs_products)),
            "pairwise": (_pairwise(rounded_products), _pairwise(abs_products)),
            "chunk3": (_chunked(rounded_products, 3), _chunked(abs_products, 3)),
            "chunk17": (_chunked(rounded_products, 17), _chunked(abs_products, 17)),
            "permuted": (_pairwise([rounded_products[i] for i in perm]), _pairwise([abs_products[i] for i in perm])),
            "fma": (_fma_dot(a, b), _pairwise(abs_products)),
        }
        ta = torch.tensor(a, dtype=torch.float64, device=device)
        tb = torch.tensor(b, dtype=torch.float64, device=device)
        schedules["torch_dot"] = (torch.dot(ta, tb).item(), torch.dot(ta.abs(), tb.abs()).item())
        for schedule, (hat, abs_hat) in schedules.items():
            if not math.isfinite(hat) or not math.isfinite(abs_hat):
                rows.append({"case": case_name, "m": m, "schedule": schedule, "status": "OUTSIDE_FINITE_INTERMEDIATE_HYPOTHESIS"})
                continue
            bound = rounding.dot_error_bound(torch.tensor(abs_hat, dtype=torch.float64, device=device), m).item()
            error = abs(Fraction(hat) - exact)
            passed = error <= Fraction(bound)
            row = {"case": case_name, "m": m, "schedule": schedule, "status": "PASS" if passed else "FAIL", "bound": bound, "error": float(error), "precondition_m_u_le_quarter": m * rounding.U <= 0.25}
            rows.append(row)
            if not passed:
                failures.append(row)
    checked = sum(row["status"] in {"PASS", "FAIL"} for row in rows)
    return {"rows": rows, "checked": checked, "passed": sum(row["status"] == "PASS" for row in rows), "failures": failures}


def run(engine_root: Path, device: str) -> dict[str, Any]:
    sys.path.insert(0, str(engine_root / "src"))
    torch = importlib.import_module("torch")
    iv = importlib.import_module("flowstar_gpu.interval")
    trans = importlib.import_module("flowstar_gpu.transcendental")
    rounding = importlib.import_module("flowstar_gpu.rounding")
    ck = importlib.import_module("flowstar_gpu.cuda_kernels")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    kernel_available = bool(ck.available()) if device == "cuda" else False
    d1 = _audit_d1(torch, iv, trans, device)
    d2 = _audit_d2(torch, rounding, device)
    return {
        "schema": "torch_tm_flowpipe.huan_proof_kernel_audit/1",
        "engine_root": str(engine_root.resolve()),
        "engine_head": _git_head(engine_root),
        "device": device,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_kernel_available": kernel_available,
        "d1": d1,
        "d2": d2,
        "gate_passed": d1["passed"] == d1["case_count"] and d1["no_ftz_observed"] and not d2["failures"],
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proof-map", type=Path, required=True)
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
    print(json.dumps({"device": args.device, "gate_passed": payload["gate_passed"], "d1": [payload["d1"]["passed"], payload["d1"]["case_count"]], "d2": [payload["d2"]["passed"], payload["d2"]["checked"]], "kernel": payload["cuda_kernel_available"]}, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
