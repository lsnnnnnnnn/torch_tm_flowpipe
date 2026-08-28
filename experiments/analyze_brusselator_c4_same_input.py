#!/usr/bin/env python3
"""Close the frozen Brusselator operator ledger and the one allowed C4 micro-gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import (  # noqa: E402
    DenseRangePolicy,
    FLOWSTAR_RAW_REMAINDER_GENERIC_REFINED_MODE,
    GENERIC_ACCEPTED_BOUNDARY_SYMBOLIC_REMAINDER,
    accepted_boundary_sr_queue_sha256,
    flowpipe_step_flowstar_style_adaptive,
    load_terminal_checkpoint,
    tmvector_hashes,
)
from torch_tm_flowpipe.ode_examples import brusselator_ode  # noqa: E402
from experiments.run_brusselator_second_system_torch import (  # noqa: E402
    _advance_local_samples,
    _sample_points,
)
from experiments.run_brusselator_sr1000_parity import (  # noqa: E402
    CONFIG,
    CUTOFF,
    ORDER,
    QUEUE_CAPACITY,
    REMAINDER_RADIUS,
    STEP,
    VALIDATION_EPS,
    _policy,
)


FLOWSTAR_SHA = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
BASELINE_COMMIT = "beb0daf310c360a28a0ecce04554a29bc30d0dbe"
MATERIAL_THRESHOLD = 1.0e-12
LEGACY_MODE = "flowstar_raw_remainder_compat"
C4_MODE = FLOWSTAR_RAW_REMAINDER_GENERIC_REFINED_MODE
BOUND_FIELDS = tuple(
    f"{prefix}_{component}_{bound}"
    for prefix in ("endpoint", "tube")
    for component in ("x", "y")
    for bound in ("lo", "hi")
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _number(value: Mapping[str, str]) -> float:
    decimal = float(value["decimal"])
    hexadecimal = float.fromhex(value["hex"])
    if decimal != hexadecimal or not math.isfinite(decimal):
        raise ValueError(f"invalid binary64 trace number: {value}")
    return hexadecimal


def _interval(value: Mapping[str, Any]) -> tuple[float, float]:
    return _number(value["lower"]), _number(value["upper"])


def _read_trace(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or any(row.get("source_commit") != FLOWSTAR_SHA for row in rows):
        raise ValueError("Flow* trace is empty or has the wrong source commit")
    if {int(row["accepted_step_index"]) for row in rows} != {0}:
        raise ValueError("Flow* trace is not restricted to the first accepted step")
    return rows


def _flowstar_polynomial(
    rows: Sequence[Mapping[str, Any]], stage: str, component: int
) -> dict[tuple[int, int, int], tuple[float, float]]:
    result: dict[tuple[int, int, int], tuple[float, float]] = {}
    for row in rows:
        if (
            row.get("stage") != stage
            or int(row.get("component", -1)) != component
            or row.get("record_type") != "polynomial_term"
        ):
            continue
        exponent = tuple(int(value) for value in row["exponents"])
        if len(exponent) != 4 or exponent[3] != 0:
            raise ValueError("plant polynomial depends on the deterministic clock state")
        canonical = (exponent[1], exponent[2], exponent[0])
        result[canonical] = (
            _number(row["coefficient_lower"]),
            _number(row["coefficient_upper"]),
        )
    return result


def _torch_polynomial(model: Any) -> dict[tuple[int, int, int], float]:
    result: dict[tuple[int, int, int], float] = {}
    for exponent, coefficient in model.polynomial.terms.items():
        normalized_exponent = tuple(int(value) for value in exponent)
        if len(normalized_exponent) == 2:
            normalized_exponent = normalized_exponent + (0,)
        if len(normalized_exponent) != 3:
            raise ValueError(
                "Brusselator comparison expects two state variables and optional tau, "
                f"got exponent {normalized_exponent!r}"
            )
        result[normalized_exponent] = float(coefficient.detach().cpu())
    return result


def _polynomial_comparison(
    torch_models: Sequence[Any],
    flowstar_rows: Sequence[Mapping[str, Any]],
    stage: str,
) -> dict[str, Any]:
    distances: list[dict[str, Any]] = []
    support_equal = True
    for component in range(2):
        torch_poly = _torch_polynomial(torch_models[component])
        stock_poly = _flowstar_polynomial(flowstar_rows, stage, component)
        support_equal &= set(torch_poly) == set(stock_poly)
        for exponent in sorted(set(torch_poly) | set(stock_poly)):
            torch_value = torch_poly.get(exponent, 0.0)
            stock_lo, stock_hi = stock_poly.get(exponent, (0.0, 0.0))
            distance = max(stock_lo - torch_value, torch_value - stock_hi, 0.0)
            distances.append(
                {
                    "component": component,
                    "exponent_ux_uy_tau": list(exponent),
                    "torch": torch_value,
                    "flowstar_lo": stock_lo,
                    "flowstar_hi": stock_hi,
                    "interval_distance": distance,
                }
            )
    maximum = max(distances, key=lambda row: row["interval_distance"])
    return {
        "support_equal": support_equal,
        "max_coefficient_interval_distance": maximum["interval_distance"],
        "max_coefficient_witness": maximum,
        "material": maximum["interval_distance"] > MATERIAL_THRESHOLD,
    }


def _torch_right_map_is_identity(models: Sequence[Any]) -> bool:
    if len(models) != 2:
        return False
    for component, model in enumerate(models):
        expected_exponent = tuple(1 if index == component else 0 for index in range(2))
        if set(model.polynomial.terms) != {expected_exponent}:
            return False
        if float(model.polynomial.terms[expected_exponent].detach().cpu()) != 1.0:
            return False
        if float(model.remainder.lo.detach().cpu()) != 0.0:
            return False
        if float(model.remainder.hi.detach().cpu()) != 0.0:
            return False
    return True


def _flowstar_right_map_is_identity(rows: Sequence[Mapping[str, Any]]) -> bool:
    for component in range(2):
        models = [
            row
            for row in rows
            if row.get("stage") == "right_map_input"
            and row.get("record_type") == "taylor_model"
            and int(row["component"]) == component
        ]
        terms = [
            row
            for row in rows
            if row.get("stage") == "right_map_input"
            and row.get("record_type") == "polynomial_term"
            and int(row["component"]) == component
        ]
        expected_exponents = [0, 0, 0, 0]
        expected_exponents[component + 1] = 1
        if len(models) != 1 or _interval(models[0]["remainder"]) != (0.0, 0.0):
            return False
        if len(terms) != 1 or terms[0]["exponents"] != expected_exponents:
            return False
        if _number(terms[0]["coefficient_lower"]) != 1.0:
            return False
        if _number(terms[0]["coefficient_upper"]) != 1.0:
            return False
    return True


def _box(segment: Any, prefix: str) -> list[Any]:
    model = segment.endpoint_raw_tm if prefix == "endpoint" else segment.tm
    if model is None:
        raise RuntimeError(f"accepted segment lacks {prefix} model")
    return model.range_box()


def _box_record(segment: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for prefix in ("endpoint", "tube"):
        for component, (name, interval) in enumerate(zip(("x", "y"), _box(segment, prefix))):
            del component
            lo = float(interval.lo.detach().cpu())
            hi = float(interval.hi.detach().cpu())
            result[f"{prefix}_{name}_lo"] = lo
            result[f"{prefix}_{name}_hi"] = hi
            result[f"{prefix}_{name}_lo_hex"] = lo.hex()
            result[f"{prefix}_{name}_hi_hex"] = hi.hex()
    return result


def _run_mode(checkpoint_dir: Path, mode: str) -> tuple[Any, list[dict[str, Any]], Any]:
    checkpoint = load_terminal_checkpoint(checkpoint_dir)
    diagnostics: list[dict[str, Any]] = []
    segment = flowpipe_step_flowstar_style_adaptive(
        brusselator_ode,
        checkpoint.current,
        h=STEP,
        h_min=STEP,
        h_max=STEP,
        order=ORDER,
        target_remainder_radius=REMAINDER_RADIUS,
        cutoff_threshold=CUTOFF,
        max_validation_attempts=int(CONFIG["max_validation_attempts"]),
        validation_eps=VALIDATION_EPS,
        validation_mode=mode,
        reset_mode=GENERIC_ACCEPTED_BOUNDARY_SYMBOLIC_REMAINDER,
        flowstar_normal_state=checkpoint.normal_state,
        flowstar_symbolic_queue_max_size=QUEUE_CAPACITY,
        right_map_range_mode=CONFIG["right_map_range_mode"],
        right_map_center_mode=CONFIG["right_map_center_mode"],
        tm_backend="dense",
        dense_device="cpu",
        dense_dtype=torch.float64,
        dense_range_policy=_policy(),
        diagnostics=diagnostics,
        diagnostics_context={"system": "brusselator", "lane": mode, "segment_index": 0},
    )
    return segment, diagnostics, checkpoint


def _validation(diagnostics: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    rows = [row for row in diagnostics if row.get("phase") == "remainder_validation"]
    if len(rows) != 1:
        raise ValueError(f"expected one initial validation record, found {len(rows)}")
    return rows[0]


def _stock_step1(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    if int(row["step"]) != 1:
        raise ValueError("stock CSV does not begin at step one")
    return row


def _max_bound_delta(row: Mapping[str, Any], stock: Mapping[str, str]) -> float:
    return max(abs(float(row[field]) - float(stock[field])) for field in BOUND_FIELDS)


def _remainder_pairs(segment: Any) -> tuple[list[float], list[float]]:
    return (
        [float(model.remainder.lo.detach().cpu()) for model in segment.tm],
        [float(model.remainder.hi.detach().cpu()) for model in segment.tm],
    )


def _l1_interval_error(
    actual: tuple[Sequence[float], Sequence[float]],
    expected: tuple[Sequence[float], Sequence[float]],
) -> float:
    return sum(abs(left - right) for left, right in zip(actual[0], expected[0])) + sum(
        abs(left - right) for left, right in zip(actual[1], expected[1])
    )


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    trace_path = args.flowstar_trace.resolve()
    compose_trace_path = args.compose_trace.resolve()
    compose_result_path = args.compose_result.resolve()
    checkpoint_dir = args.prestate.resolve()
    stock_csv = args.stock_csv.resolve()
    observed_csv = args.observed_csv.resolve()
    unobserved_csv = args.unobserved_csv.resolve()
    sr1000_dir = args.sr1000_dir.resolve()
    flowstar_rows = _read_trace(trace_path)
    compose_rows = _read_trace(compose_trace_path)
    stock = _stock_step1(stock_csv)
    compose_result = json.loads(compose_result_path.read_text(encoding="utf-8"))
    sr1000_summary = json.loads((sr1000_dir / "summary.json").read_text(encoding="utf-8"))
    if sr1000_summary["capacity_reset_decision"] != "NOT_SOLELY_QUEUE_RESET_CAPACITY":
        raise ValueError("operator localization requires the frozen non-capacity verdict")
    frozen_stock_sha256 = "08e184e2b0a99be48417be8971ed6632eccec0630849787a1048b9962d15f567"
    observer_output_equivalent = (
        _sha256(stock_csv)
        == _sha256(observed_csv)
        == _sha256(unobserved_csv)
        == frozen_stock_sha256
    )

    baseline, baseline_diagnostics, baseline_input = _run_mode(checkpoint_dir, LEGACY_MODE)
    candidate, candidate_diagnostics, candidate_input = _run_mode(checkpoint_dir, C4_MODE)
    if baseline.status != "validated" or candidate.status != "validated":
        raise RuntimeError("paired step-one operator evaluation did not validate")
    baseline_validation = _validation(baseline_diagnostics)
    candidate_validation = _validation(candidate_diagnostics)
    baseline_box = _box_record(baseline)
    candidate_box = _box_record(candidate)

    original_rows = list(csv.DictReader((sr1000_dir / "segments.csv").open(encoding="utf-8")))
    original_step1 = original_rows[0]
    baseline_reproduces_sr1000 = all(
        baseline_box[f"{field}_hex"] == original_step1[f"{field}_hex"]
        for field in BOUND_FIELDS
    )
    input_hashes_equal = (
        baseline_input.manifest["full_checkpoint_sha256"]
        == candidate_input.manifest["full_checkpoint_sha256"]
        and tmvector_hashes(baseline_input.current) == tmvector_hashes(candidate_input.current)
        and tmvector_hashes(baseline_input.normal_state.tmv_pre)
        == tmvector_hashes(candidate_input.normal_state.tmv_pre)
        and tmvector_hashes(baseline_input.normal_state.tmv_right)
        == tmvector_hashes(candidate_input.normal_state.tmv_right)
    )

    input_polynomial = _polynomial_comparison(
        tuple(baseline_input.current), flowstar_rows, "pre_map_input"
    )
    final_polynomial = _polynomial_comparison(tuple(baseline.tm), flowstar_rows, "candidate_target")
    queue_input = next(row for row in flowstar_rows if row["stage"] == "symbolic_queue_input")
    torch_right_map_identity = _torch_right_map_is_identity(
        baseline_input.normal_state.tmv_right
    )
    flowstar_right_map_identity = _flowstar_right_map_is_identity(flowstar_rows)
    stage_input = {
        **input_polynomial,
        "flowstar_queue_j_size": int(queue_input["j_size"]),
        "flowstar_queue_phi_size": int(queue_input["phi_size"]),
        "torch_queue_absent": baseline_input.normal_state.symbolic_queue is None,
        "torch_right_map_identity": torch_right_map_identity,
        "flowstar_right_map_identity": flowstar_right_map_identity,
    }
    stage_input["material"] = bool(
        stage_input["max_coefficient_interval_distance"] > MATERIAL_THRESHOLD
        or stage_input["flowstar_queue_j_size"]
        or stage_input["flowstar_queue_phi_size"]
        or not stage_input["torch_queue_absent"]
        or not torch_right_map_identity
        or not flowstar_right_map_identity
    )

    stock_intermediate = [
        _interval(row["interval"])
        for row in flowstar_rows
        if row["stage"] == "candidate_intermediate_ranges"
    ]
    stock_truncation = (stock_intermediate[7], stock_intermediate[14])
    stock_hooks = [
        _interval(row["interval"])
        for row in flowstar_rows
        if row["stage"] == "operator_degree_truncation" and int(row["attempt_index"]) == 0
    ]
    if stock_truncation != (stock_hooks[4], stock_hooks[9]):
        raise ValueError("Flow* terminal truncation hook/intermediate mapping changed")
    torch_truncation_ledger = baseline_validation["raw_remainder_ledger_intervals"][
        "polynomial_truncation"
    ]
    torch_truncation = tuple(
        (
            float(torch_truncation_ledger["lo"][0][component]),
            float(torch_truncation_ledger["hi"][0][component]),
        )
        for component in range(2)
    )
    truncation_deltas = [
        abs(torch_truncation[component][bound] - stock_truncation[component][bound])
        for component in range(2)
        for bound in range(2)
    ]
    stock_cutoffs = [
        _interval(row["interval"])
        for row in flowstar_rows
        if row["stage"] == "operator_cutoff" and int(row["attempt_index"]) == 0
    ]
    torch_cutoff = baseline_validation["raw_remainder_ledger_intervals"]["cutoff"]
    cutoff_max = max(
        [abs(value) for interval in stock_cutoffs for value in interval]
        + [
            abs(float(torch_cutoff[bound][0][component]))
            for bound in ("lo", "hi")
            for component in range(2)
        ]
    )
    truncation_stage = {
        "flowstar_terminal_degree_truncation": stock_truncation,
        "torch_degree_truncation_ledger": torch_truncation,
        "max_degree_truncation_bound_delta": max(truncation_deltas),
        "flowstar_candidate_cutoff_hooks_all_zero": all(
            lo == 0.0 and hi == 0.0 for lo, hi in stock_cutoffs
        ),
        "max_cutoff_owner_magnitude": cutoff_max,
        "material": max(truncation_deltas) > MATERIAL_THRESHOLD,
        "mapping": (
            "Flow* candidate_intermediate_ranges[7,14] are the terminal x/y ODE "
            "degree-discard intervals and equal candidate attempt hooks[4,9]"
        ),
    }

    stock_raw = tuple(
        _interval(row["image"])
        for row in flowstar_rows
        if row["stage"] == "candidate_subset" and int(row["component"]) < 2
    )
    torch_raw = tuple(
        (
            float(baseline_validation["picard_image_remainder_lo"][0][component]),
            float(baseline_validation["picard_image_remainder_hi"][0][component]),
        )
        for component in range(2)
    )
    raw_deltas = [
        abs(torch_raw[component][bound] - stock_raw[component][bound])
        for component in range(2)
        for bound in range(2)
    ]
    raw_stage = {
        "flowstar_initial_subset_image": stock_raw,
        "torch_initial_subset_image": torch_raw,
        "max_bound_delta": max(raw_deltas),
        "material": max(raw_deltas) > MATERIAL_THRESHOLD,
    }

    stock_accepted = tuple(
        _interval(row["remainder"])
        for row in flowstar_rows
        if row["stage"] == "accepted_remainder"
        and row["record_type"] == "taylor_model"
        and int(row["component"]) < 2
    )
    stock_remainder = (
        tuple(pair[0] for pair in stock_accepted),
        tuple(pair[1] for pair in stock_accepted),
    )
    baseline_remainder = _remainder_pairs(baseline)
    candidate_remainder = _remainder_pairs(candidate)
    baseline_remainder_error = _l1_interval_error(baseline_remainder, stock_remainder)
    candidate_remainder_error = _l1_interval_error(candidate_remainder, stock_remainder)
    refinement_rows = [
        row for row in candidate.backend_trace if row.get("phase") == "post_accept_refinement"
    ]
    first_acceptance_identical = all(
        baseline_validation[key] == candidate_validation[key]
        for key in (
            "validation_status",
            "subset_result",
            "target_subset_result",
            "candidate_remainder_lo",
            "candidate_remainder_hi",
            "picard_image_remainder_lo",
            "picard_image_remainder_hi",
            "raw_rhs_remainder_lo",
            "raw_rhs_remainder_hi",
            "poly_diff_range_lo",
            "poly_diff_range_hi",
        )
    )
    refinement_gate = {
        "mode": C4_MODE,
        "first_acceptance_identical_to_legacy": first_acceptance_identical,
        "retained_polynomial_equal": all(
            set(left.polynomial.terms) == set(right.polynomial.terms)
            and all(
                torch.equal(
                    left.polynomial.terms[exponent],
                    right.polynomial.terms[exponent],
                )
                for exponent in left.polynomial.terms
            )
            for left, right in zip(baseline.tm, candidate.tm)
        ),
        "refinement_iterations": len(refinement_rows),
        "all_commits_subset": bool(refinement_rows)
        and all(
            row["committed"] and all(component["subset"] for component in row["components"])
            for row in refinement_rows
        ),
        "stop_reason": refinement_rows[-1]["stop_reason"] if refinement_rows else "",
        "baseline_stock_remainder_l1_error": baseline_remainder_error,
        "candidate_stock_remainder_l1_error": candidate_remainder_error,
        "stock_remainder_error_ratio": candidate_remainder_error / baseline_remainder_error,
        "validated_decomposition_contains_image": bool(
            torch.all(candidate.validated_remainder_decomposition.contains_image)
        ),
    }

    samples = _sample_points()
    _, endpoint_violations, tube_violations, solver_ok = _advance_local_samples(
        samples, _box(candidate, "endpoint"), _box(candidate, "tube")
    )
    refinement_gate.update(
        {
            "sample_solver_ok": solver_ok,
            "sample_endpoint_violations": endpoint_violations,
            "sample_tube_violations": tube_violations,
            "queue_present_after_accept": candidate.flowstar_normal_state is not None
            and candidate.flowstar_normal_state.symbolic_queue is not None,
            "queue_hash_after_accept": (
                accepted_boundary_sr_queue_sha256(candidate.flowstar_normal_state.symbolic_queue)
                if candidate.flowstar_normal_state is not None
                and candidate.flowstar_normal_state.symbolic_queue is not None
                else ""
            ),
        }
    )

    compose_matches_stock = all(
        float(compose_result[f"{prefix}_{component}"][bound])
        == float(stock[f"{prefix}_{component}_{bound}"])
        for prefix in ("endpoint", "tube")
        for component in ("x", "y")
        for bound in ("lo", "hi")
    )
    endpoint_stage = {
        "flowstar_compose_probe_matches_frozen_stock_bit_exact": compose_matches_stock,
        "baseline_stock_max_bound_delta": _max_bound_delta(baseline_box, stock),
        "candidate_stock_max_bound_delta": _max_bound_delta(candidate_box, stock),
        "flowstar_range_semantics": "compose_normal then per-term intEvalNormal without tau-term collection",
        "torch_range_semantics": "substitute tau=h, collect equal state monomials, then interval-evaluate",
        "later_material_observation": _max_bound_delta(baseline_box, stock) > MATERIAL_THRESHOLD,
    }

    stages = [
        {
            "search_index": 1,
            "stage": "accepted_input_right_map_queue",
            **stage_input,
        },
        {
            "search_index": 2,
            "stage": "polynomial_picard_pre_truncation",
            "flowstar_observer_stage": "candidate_target",
            **final_polynomial,
        },
        {
            "search_index": 3,
            "stage": "truncation_cutoff_owners",
            **truncation_stage,
        },
        {"search_index": 4, "stage": "raw_rhs_remainder_picard_residual", **raw_stage},
        {
            "search_index": 5,
            "stage": "validated_remainder_subset",
            "flowstar_accepted_remainder": stock_remainder,
            "torch_baseline_remainder": baseline_remainder,
            "torch_c4_remainder": candidate_remainder,
            "baseline_stock_l1_error": baseline_remainder_error,
            "c4_stock_l1_error": candidate_remainder_error,
            "material": baseline_remainder_error > MATERIAL_THRESHOLD,
        },
        {"search_index": 6, "stage": "endpoint_tube_range", **endpoint_stage},
    ]
    first_material = next(stage for stage in stages if stage.get("material") is True)

    gate_passed = all(
        (
            input_hashes_equal,
            baseline_reproduces_sr1000,
            first_acceptance_identical,
            refinement_gate["retained_polynomial_equal"],
            refinement_gate["all_commits_subset"],
            refinement_gate["stop_reason"] == "stop_ratio",
            refinement_gate["stock_remainder_error_ratio"] < 0.25,
            refinement_gate["validated_decomposition_contains_image"],
            solver_ok,
            endpoint_violations == 0,
            tube_violations == 0,
            compose_matches_stock,
            observer_output_equivalent,
        )
    )
    c4_status = "C4_FIX_AUTHORIZED" if gate_passed else "NO_C4_FIX_AUTHORIZED"
    git_diff = subprocess.run(
        [
            "git",
            "diff",
            "--binary",
            BASELINE_COMMIT,
            "--",
            "src/torch_tm_flowpipe/batched_dense_tm.py",
            "src/torch_tm_flowpipe/flowpipe.py",
            "src/torch_tm_flowpipe/__init__.py",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    result = {
        "schema": "torch_tm_flowpipe.brusselator_c4_same_input_result/1",
        "capacity_verdict": sr1000_summary["capacity_reset_decision"],
        "sr1000_accepted_steps": sr1000_summary["accepted_steps"],
        "stock_accepted_steps": 1000,
        "first_material_operator_divergence": first_material["stage"],
        "first_material_search_index": first_material["search_index"],
        "first_material_max_delta": first_material["max_degree_truncation_bound_delta"],
        "same_input_checkpoint_sha256": baseline_input.manifest["full_checkpoint_sha256"],
        "same_input_hashes_equal": input_hashes_equal,
        "baseline_reproduces_frozen_sr1000_step1": baseline_reproduces_sr1000,
        "c4_status": c4_status,
        "c4_fix_budget": 1,
        "c4_numeric_fixes_authorized": 1 if gate_passed else 0,
        "c4_mode": C4_MODE,
        "c4_change": "generic atomic post-accept ordered-terms raw-remainder refinement",
        "c4_core_diff_sha256": hashlib.sha256(git_diff).hexdigest(),
        "same_input_gate": refinement_gate,
        "endpoint_range_semantics_is_later_not_first": True,
        "flowstar_observer_output_equivalent": observer_output_equivalent,
        "full_c4_prefix_rerun_performed": False,
        "full_c4_prefix_rerun_reason": "frozen contract permits one paired local operator evaluation, not another benchmark",
    }
    provenance = {
        "schema": "torch_tm_flowpipe.brusselator_c4_provenance/1",
        "flowstar_commit": FLOWSTAR_SHA,
        "baseline_commit": BASELINE_COMMIT,
        "head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "inputs": {
            str(path): {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (
                trace_path,
                compose_trace_path,
                compose_result_path,
                stock_csv,
                observed_csv,
                unobserved_csv,
                sr1000_dir / "summary.json",
                sr1000_dir / "segments.csv",
                checkpoint_dir / "terminal_state.json",
                checkpoint_dir / "terminal_state_manifest.json",
            )
        },
        "flowstar_observer_trace_is_output_equivalent": observer_output_equivalent,
        "compose_trace_record_count": len(compose_rows),
    }
    _write_json(output / "operator_ledger.json", {"material_threshold": MATERIAL_THRESHOLD, "stages": stages})
    _write_json(output / "same_input_gate.json", refinement_gate)
    _write_json(output / "RESULT.json", result)
    _write_json(output / "provenance.json", provenance)
    manifest = {
        "schema": "torch_tm_flowpipe.brusselator_c4_manifest/1",
        "files": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(output.iterdir())
            if path.is_file()
        },
        "result": result,
    }
    _write_json(output / "MANIFEST.json", manifest)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flowstar-trace", type=Path, required=True)
    parser.add_argument("--compose-trace", type=Path, required=True)
    parser.add_argument("--compose-result", type=Path, required=True)
    parser.add_argument("--stock-csv", type=Path, required=True)
    parser.add_argument("--observed-csv", type=Path, required=True)
    parser.add_argument("--unobserved-csv", type=Path, required=True)
    parser.add_argument("--sr1000-dir", type=Path, required=True)
    parser.add_argument("--prestate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = analyze(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["c4_status"] == "C4_FIX_AUTHORIZED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
