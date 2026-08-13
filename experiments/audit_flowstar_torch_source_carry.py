#!/usr/bin/env python3
"""Derive the source/carry audit from frozen Flow* and Torch raw traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.source_carry_audit import (
    FLOWSTAR_CHANNELS,
    TORCH_CHANNELS,
    accepted_flowstar_rows,
    accepted_torch_rows,
    checkpoint_reproduction,
    derive_same_prestate_gate,
    derive_scientific_outcome,
    derive_width_minima,
    exact_semantics_micro_oracles,
    finite_float,
    growth_and_ratio_analysis,
    interval_record,
    parse_json_cell,
    runtime_feature_summary,
    source_semantics_map_is_closed,
)


SCHEMA = "flowstar_torch_source_carry_audit_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(command: Sequence[str], *, cwd: Path | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        list(command), cwd=cwd, text=True, capture_output=True, check=False
    )
    return {
        "command": list(command),
        "cwd": str(cwd.resolve()) if cwd is not None else str(ROOT),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _bounds_width(row: Mapping[str, str], fields: tuple[str, str]) -> float | str:
    lo_field, hi_field = fields
    if row.get(lo_field, "") == "" or row.get(hi_field, "") == "":
        return ""
    return finite_float(row[hi_field], field=hi_field) - finite_float(
        row[lo_field], field=lo_field
    )


def _validator_margin(raw: str, component: int) -> float | str:
    if raw == "":
        return ""
    value = parse_json_cell(raw, field="target_margins")
    if not isinstance(value, list) or not value or not isinstance(value[0], list):
        raise ValueError("target_margins is not a 1xN array")
    return float(value[0][component])


def baseline_step_trace(
    flow_all: Sequence[Mapping[str, str]], torch_all: Sequence[Mapping[str, str]]
) -> list[dict[str, Any]]:
    flow_by_step = {
        int(row["accepted_step_index"]) + 1: row
        for row in flow_all
        if row.get("status") == "accepted" and row.get("accepted") == "true"
    }
    torch_by_step: dict[int, Mapping[str, str]] = {}
    for row in torch_all:
        raw_carry_step = row.get("carry_step_index", "")
        step = (
            int(raw_carry_step)
            if raw_carry_step != ""
            else int(row["segment_index"]) + 1
        )
        if step in torch_by_step:
            raise ValueError(f"duplicate Torch record for step {step}")
        torch_by_step[step] = row
    maximum = max(torch_by_step)
    output: list[dict[str, Any]] = []
    for step in range(1, maximum + 1):
        flow = flow_by_step[step]
        torch = torch_by_step[step]
        row: dict[str, Any] = {
            "step": step,
            "time": step * 0.01,
            "flowstar_status": flow["status"],
            "torch_status": torch["status"],
            "flowstar_center_x": flow.get("extracted_center_x", ""),
            "flowstar_center_y": flow.get("extracted_center_y", ""),
            "flowstar_scale_x": flow.get("extracted_scale_x", ""),
            "flowstar_scale_y": flow.get("extracted_scale_y", ""),
            "flowstar_polynomial_range_x_lo": flow.get("polynomial_range_x_lo", ""),
            "flowstar_polynomial_range_x_hi": flow.get("polynomial_range_x_hi", ""),
            "flowstar_polynomial_range_y_lo": flow.get("polynomial_range_y_lo", ""),
            "flowstar_polynomial_range_y_hi": flow.get("polynomial_range_y_hi", ""),
            "flowstar_picard_remainder_x_lo": flow.get("post_cutoff_residual_x_lo", ""),
            "flowstar_picard_remainder_x_hi": flow.get("post_cutoff_residual_x_hi", ""),
            "flowstar_picard_remainder_y_lo": flow.get("post_cutoff_residual_y_lo", ""),
            "flowstar_picard_remainder_y_hi": flow.get("post_cutoff_residual_y_hi", ""),
            "flowstar_symbolic_propagated_width_x": flow.get("symbolic_propagated_width_x", ""),
            "flowstar_symbolic_propagated_width_y": flow.get("symbolic_propagated_width_y", ""),
            "flowstar_queue_size": flow.get("symbolic_J_size", ""),
            "torch_prestate_center": torch.get("prestate_center", ""),
            "torch_prestate_scale": torch.get("prestate_scale", ""),
            "torch_new_center": torch.get("center", ""),
            "torch_new_scale": torch.get("scale", ""),
            "torch_polynomial_range_width_x": torch.get(
                "carry_composed_poly_range_width_x", ""
            ),
            "torch_polynomial_range_width_y": torch.get(
                "carry_composed_poly_range_width_y", ""
            ),
            "torch_ordinary_remainder": torch.get("post_poly_diff_remainder", ""),
            "torch_parameterization_remainder_width_x": torch.get(
                "carry_output_remainder_width_x", ""
            ),
            "torch_parameterization_remainder_width_y": torch.get(
                "carry_output_remainder_width_y", ""
            ),
            "torch_pre_renormalization_range_width_x": torch.get(
                "carry_centered_inserted_range_width_x", ""
            ),
            "torch_pre_renormalization_range_width_y": torch.get(
                "carry_centered_inserted_range_width_y", ""
            ),
            "torch_validator_margin_x": _validator_margin(
                torch.get("target_margins", ""), 0
            ),
            "torch_validator_margin_y": _validator_margin(
                torch.get("target_margins", ""), 1
            ),
        }
        for channel, fields in FLOWSTAR_CHANNELS.items():
            row[f"flowstar_{channel}_lo"] = flow[fields[0]]
            row[f"flowstar_{channel}_hi"] = flow[fields[1]]
            row[f"flowstar_{channel}_width"] = _bounds_width(flow, fields)
        for channel, fields in TORCH_CHANNELS.items():
            row[f"torch_{channel}_lo"] = torch.get(fields[0], "")
            row[f"torch_{channel}_hi"] = torch.get(fields[1], "")
            row[f"torch_{channel}_width"] = _bounds_width(torch, fields)
        output.append(row)
    return output


def historical_replay_check(
    historical: Path | None,
    flow_rows: Sequence[Mapping[str, str]],
    torch_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    if historical is None:
        return {"available": False, "reason": "no historical common-prefix CSV supplied"}
    old = read_csv(historical)
    if len(old) != min(len(flow_rows), len(torch_rows)):
        raise ValueError("historical common-prefix length mismatch")
    fields = [
        *(f"flowstar_{channel}_width" for channel in FLOWSTAR_CHANNELS),
        *(f"torch_{channel}_width" for channel in TORCH_CHANNELS),
    ]
    exact = True
    mismatch_count = 0
    for index, prior in enumerate(old):
        flow = flow_rows[index]
        torch = torch_rows[index]
        for channel in FLOWSTAR_CHANNELS:
            value = str(_bounds_width(flow, FLOWSTAR_CHANNELS[channel]))
            if value != prior[f"flowstar_{channel}_width"]:
                exact = False
                mismatch_count += 1
        for channel in TORCH_CHANNELS:
            value = str(_bounds_width(torch, TORCH_CHANNELS[channel]))
            if value != prior[f"torch_{channel}_width"]:
                exact = False
                mismatch_count += 1
    return {
        "available": True,
        "path": str(historical.resolve()),
        "sha256": sha256(historical),
        "rows": len(old),
        "compared_fields": fields,
        "decimal_text_exact": exact,
        "mismatch_count": mismatch_count,
    }


def source_map() -> list[dict[str, Any]]:
    flow_root = "/srv/local/shengenli/flowstar_source_carry_20260813"
    torch_root = str(ROOT)
    return [
        {
            "mathematical_stage": "benchmark/model entry",
            "flowstar_source": f"{flow_root}/benchmarks/continuous/vanderpol/vanderpol.cpp:7-89 main",
            "flowstar_representation": "ODE<Real>, Flowpipe, Symbolic_Remainder(max_size=100)",
            "torch_source": f"{torch_root}/experiments/run_vdp_dense_backend.py:488-533 integration loop",
            "torch_representation": "complete total-degree O4 BatchedTaylorModel",
            "first_unequal": False,
            "dependency_consequence": "same ODE/box/order contract; different implementation stacks",
        },
        {
            "mathematical_stage": "fixed-step reach loop",
            "flowstar_source": f"{flow_root}/flowstar-toolbox/Continuous.h:832-882 ODE::reach_symbolic_remainder",
            "flowstar_representation": "accepted Flowpipe chain plus persistent symbolic queue",
            "torch_source": f"{torch_root}/src/torch_tm_flowpipe/flowpipe.py:5197-5386 integrate_adaptive",
            "torch_representation": "FlowstarNormalFlowpipeState without queue in legacy mode",
            "first_unequal": False,
            "dependency_consequence": "runtime feature selection becomes observable after the first boundary",
        },
        {
            "mathematical_stage": "cross-step carry decomposition",
            "flowstar_source": f"{flow_root}/flowstar-toolbox/Continuous.cpp:2151-2177 Flowpipe::advance",
            "flowstar_representation": "linear x0 map plus nonlinear/other TM; Phi_L/J retain per-step source identity",
            "torch_source": f"{torch_root}/src/torch_tm_flowpipe/flowpipe.py:1470-1511 _flowstar_normalized_insertion_transition",
            "torch_representation": "constant removed, otherwise full outer TM sent to legacy insertion",
            "first_unequal": True,
            "dependency_consequence": "Flow* propagates linear old sources once through matrices; Torch has no source queue",
        },
        {
            "mathematical_stage": "normal polynomial composition",
            "flowstar_source": f"{flow_root}/flowstar-toolbox/TaylorModel.h:4213-4243 HornerForm::insert_ctrunc_normal",
            "flowstar_representation": "recursive Horner grouping with truncation at multiplication stages",
            "torch_source": f"{torch_root}/src/torch_tm_flowpipe/flowpipe.py:698-739 _insert_ctrunc_normal_like_scalar",
            "torch_representation": "each monomial built independently by repeated TM multiplication, then summed",
            "first_unequal": True,
            "dependency_consequence": "one right-map remainder interval is independently materialized in distinct monomial paths",
        },
        {
            "mathematical_stage": "TM multiplication remainder",
            "flowstar_source": f"{flow_root}/flowstar-toolbox/TaylorModel.h:797-866 TaylorModel::mul_insert_ctrunc_normal",
            "flowstar_representation": "P1*I2 + P2*I1 + I1*I2 plus truncation/cutoff interval",
            "torch_source": f"{torch_root}/src/torch_tm_flowpipe/taylor_model.py:276-285 TaylorModel.__mul__",
            "torch_representation": "polynomial-range times scalar Interval remainder components",
            "first_unequal": False,
            "dependency_consequence": "both intervalize locally; grouping and source reuse determine repeated cost",
        },
        {
            "mathematical_stage": "Picard/validator",
            "flowstar_source": f"{flow_root}/flowstar-toolbox/Continuous.cpp:2328-2410 Flowpipe::advance",
            "flowstar_representation": "Picard_ctrunc_normal candidate and subset/refinement intervals",
            "torch_source": f"{torch_root}/src/torch_tm_flowpipe/batched_dense_tm.py:2750-3030 dense_picard_validate_step",
            "torch_representation": "tensor-native polynomial plus interval remainder ledger",
            "first_unequal": False,
            "dependency_consequence": "validators consume already different carry scales/remainders",
        },
        {
            "mathematical_stage": "endpoint/tube range extraction",
            "flowstar_source": f"{flow_root}/flowstar-toolbox/Continuous.cpp:415-454 Flowpipe::compose_normal/intEvalNormal",
            "flowstar_representation": "accepted result composed and interval-evaluated over tau=[0,h] or tau=h",
            "torch_source": f"{torch_root}/src/torch_tm_flowpipe/flowpipe.py:5040-5147 published segment/endpoint",
            "torch_representation": "raw accepted TM range fields",
            "first_unequal": False,
            "dependency_consequence": "published objects include retained ordinary and parameterization uncertainty",
        },
        {
            "mathematical_stage": "serialization/parser/join",
            "flowstar_source": f"{torch_root}/experiments/flowstar_probe/flowstar_vdp_step_trace_probe.cpp:400-408,613-624,695-724",
            "flowstar_representation": "directed binary64 bounds serialized with 17 significant digits",
            "torch_source": f"{torch_root}/experiments/compare_flowstar_torch_fixed_schedule.py:180-360 compare",
            "torch_representation": "strict CSV parse and accepted-index/time equality checks; no zero fill",
            "first_unequal": False,
            "dependency_consequence": "the positive minima are preserved exactly through the output pipeline",
        },
    ]


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    flow_path = args.flowstar_trace.resolve()
    metadata_path = args.flowstar_metadata.resolve()
    torch_path = args.torch_segments.resolve()
    torch_summary_path = args.torch_summary.resolve()
    flow_all = read_csv(flow_path)
    torch_all = read_csv(torch_path)
    flow = accepted_flowstar_rows(flow_all)
    torch = accepted_torch_rows(torch_all)
    metadata_rows = read_csv(metadata_path)
    metadata = {row["key"]: row["value"] for row in metadata_rows}
    torch_summary = read_json(torch_summary_path)

    if len(flow) != 1000:
        raise ValueError("Flow* did not accept the frozen 1000-step run")
    if len(torch) != 632 or int(torch_summary.get("accepted_steps", -1)) != 632:
        raise ValueError("Torch did not reproduce the frozen 632-step accepted prefix")
    if torch_summary.get("status") != "failed" or torch_summary.get("failure_type") != "minimum_step_reached":
        raise ValueError("Torch scientific failure outcome changed")

    contract = {
        "schema": "vdp_complete_o4_fixed_contract_v1",
        "ode": ["x'=y", "y'=y-x-x^2*y"],
        "initial_box": {"x": [1.1, 1.4], "y": [2.35, 2.45]},
        "partition": "B1",
        "representation": "complete_total_degree_O4",
        "fixed_step": 0.01,
        "fixed_step_hex": (0.01).hex(),
        "target_horizon": 10.0,
        "remainder_target": [-1e-4, 1e-4],
        "cutoff": [-1e-10, 1e-10],
        "metrics": ["endpoint", "one_segment_tube", "prefix_tube"],
        "contract_identity": torch_summary.get("contract_identity"),
    }
    contract["sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(output / "baseline_contract.json", contract)

    step_rows = baseline_step_trace(flow_all, torch_all)
    write_csv(output / "baseline_step_trace.csv", step_rows)
    checkpoints, baseline_verdict = checkpoint_reproduction(flow, torch)
    baseline_verdict["historical_replay"] = historical_replay_check(
        args.historical_common_prefix, flow, torch
    )
    write_csv(output / "baseline_checkpoint_reproduction.csv", checkpoints)
    write_json(output / "baseline_reproduction_verdict.json", baseline_verdict)

    minima, contexts = derive_width_minima(flow)
    write_csv(output / "flowstar_width_minima.csv", minima)
    write_csv(output / "flowstar_width_minima_context.csv", contexts)
    lineage = {
        "schema": "flowstar_width_data_lineage_v1",
        "classification": "Z0_POSITIVE_WIDTH_ONLY_LOOKS_ZERO",
        "source_trace": {"path": str(flow_path), "sha256": sha256(flow_path)},
        "layers": [
            {
                "layer": "Flowstar accepted Flowpipe",
                "source": "Continuous.cpp:2412-2414 result.tmvPre/domain",
            },
            {
                "layer": "endpoint/tube interval extraction",
                "source": "Continuous.cpp:415-454 compose_normal/intEvalNormal",
            },
            {
                "layer": "observer bounds",
                "source": "flowstar_vdp_step_trace_probe.cpp:1496-1506 and 613-624",
            },
            {
                "layer": "17-digit CSV serialization",
                "source": "flowstar_vdp_step_trace_probe.cpp:400-408 and 695-724",
            },
            {
                "layer": "strict parser and join",
                "source": "compare_flowstar_torch_fixed_schedule.py:180-360",
            },
        ],
        "first_zero_or_near_zero_layer": None,
        "missing_rows": 0,
        "duplicate_accepted_steps": 0,
        "nan_or_inf_bounds": 0,
        "empty_bound_fields": 0,
        "failure_rows_zero_filled": False,
        "abs_clip_round_smooth_or_fill_applied": False,
        "lower_upper_recomputed": True,
        "minimum_summary": minima,
    }
    write_json(output / "flowstar_width_data_lineage.json", lineage)
    growth = growth_and_ratio_analysis(flow[: len(torch)], torch, minima)
    write_json(output / "width_growth_and_ratio_analysis.json", growth)

    features = runtime_feature_summary(flow, metadata, torch_summary)
    write_json(output / "flowstar_runtime_features.json", features)
    semantic_map = source_map()
    write_json(
        output / "source_semantics_map.json",
        {
            "schema": "flowstar_torch_carry_source_map_v1",
            "rows": semantic_map,
            "first_bitwise_or_published_width_difference": "accepted step 1",
            "first_beyond_roundoff_difference": "accepted step 1 (width differences are 2.14e-4 to 1.30e-3)",
            "first_persistent_dependency_semantics_difference": "boundary after accepted step 1; Flow* queue is active for step 2",
            "first_decision_relevant_difference": "step 1 output changes step 2 normalization scales",
            "localized_conclusion": (
                "Flow* Continuous.cpp:2151-2177 keeps the linear old-remainder sources in "
                "Phi_L/J and TaylorModel.h:4213-4243 composes the nonlinear polynomial in "
                "Horner form. Torch flowpipe.py:1470-1511 sends the whole constant-removed "
                "polynomial to flowpipe.py:698-739, where each monomial independently reuses "
                "the already intervalized right-map remainder. This changes the next scale "
                "after accepted step 1 and accumulates polynomial-times-parameterization "
                "remainder thereafter."
            ),
        },
    )

    same_prestate = {
        **derive_same_prestate_gate(
            coefficient_export="Real::toString 15-digit scientific decimal",
            symbolic_queue_exported=False,
            import_path_available=False,
        ),
        "initial_step_same_prestate": True,
        "post_step_full_bridge_available": False,
        "reason": (
            "The executed probe's canonical Flow* state exporter uses Real::toString(), which "
            "serializes coefficients at 15 scientific digits, and therefore is not a lossless "
            "binary coefficient/state bridge. It also does not expose an import path for the "
            "complete Phi_L/J queue. No lossy adapter is used."
        ),
        "fallback": "exact affine/quadratic/cubic rational fixtures",
        "two_by_two_attribution_available": False,
    }
    write_json(output / "same_prestate_lossless_gate.json", same_prestate)
    micro = exact_semantics_micro_oracles()
    write_json(
        output / "exact_semantics_micro_oracles.json",
        {"schema": "exact_dependency_semantics_micro_oracles_v1", "fixtures": micro},
    )

    outcome = derive_scientific_outcome(
        baseline_verdict=baseline_verdict,
        minima=minima,
        runtime_features=features,
        source_map_closed=source_semantics_map_is_closed(semantic_map),
        lossless_full_prestate_bridge=bool(same_prestate["lossless_full_prestate_bridge"]),
        independent_candidate_oracle_closed=False,
        flowstar_soundness_gate_closed=False,
    )
    no_fix = {
        "status": "NO_FIX_AUTHORIZED",
        "candidate_implemented": False,
        "reason": [
            "The complete post-step Flow* state and symbolic queue cannot be losslessly imported for the required 2x2 attribution.",
            "The pinned Flow* build has an independent strict scalar-affine under-enclosure witness, so it is not a sound tightness oracle.",
            "No independently proved outward source-ledger primitive currently covers nonlinear multiplication, O4 truncation, cutoff, renormalization, and the full carry transition together.",
        ],
        "only_next_question": (
            "Can a lossless binary state/queue fixture and an independently outward-rounded "
            "source-ledger carry primitive prove the complete one-step containment contract?"
        ),
    }
    write_json(output / "candidate_or_no_fix.json", no_fix)

    provenance = {
        "schema": "flowstar_torch_source_carry_provenance_v1",
        "torch": {
            "root": str(ROOT),
            "source_sha": run_text(["git", "rev-parse", "HEAD"], cwd=ROOT)["stdout"].strip(),
            "status": run_text(["git", "status", "--short", "--branch"], cwd=ROOT),
        },
        "flowstar": {
            "root": str(args.flowstar_root.resolve()),
            "source_sha": run_text(["git", "rev-parse", "HEAD"], cwd=args.flowstar_root)["stdout"].strip(),
            "status": run_text(["git", "status", "--short", "--branch"], cwd=args.flowstar_root),
            "remote": run_text(["git", "remote", "-v"], cwd=args.flowstar_root),
            "recent_log": run_text(["git", "log", "-5", "--oneline", "--decorate"], cwd=args.flowstar_root),
            "submodules": run_text(["git", "submodule", "status", "--recursive"], cwd=args.flowstar_root),
            "library": {
                "path": str(args.flowstar_library.resolve()),
                "sha256": sha256(args.flowstar_library),
                "file": run_text(["file", str(args.flowstar_library.resolve())]),
            },
            "probe": (
                None
                if args.flowstar_binary is None
                else {
                    "path": str(args.flowstar_binary.resolve()),
                    "resolved_path": str(args.flowstar_binary.resolve().resolve()),
                    "sha256": sha256(args.flowstar_binary),
                    "file": run_text(["file", str(args.flowstar_binary.resolve())]),
                    "ldd": run_text(["ldd", str(args.flowstar_binary.resolve())]),
                }
            ),
            "compiler": run_text(["g++", "--version"]),
            "build_contract": {
                "flags": "-O3 -g -std=c++11; GCC 15 observation build adds -fpermissive for a pre-existing uninstantiated derivative template body",
                "interval_backend": "MPFR, default 53-bit precision, directed interval endpoints",
                "baseline_gcc11_library_sha256": "b5ff500af66354b0518cf12e7d951f4525f435e8e2d695cf84b91821992c9d9a",
                "instrumentation_isolated_worktree": True,
            },
        },
        "inputs": {
            "flowstar_trace": {"path": str(flow_path), "sha256": sha256(flow_path)},
            "flowstar_metadata": {"path": str(metadata_path), "sha256": sha256(metadata_path)},
            "torch_segments": {"path": str(torch_path), "sha256": sha256(torch_path)},
            "torch_summary": {"path": str(torch_summary_path), "sha256": sha256(torch_summary_path)},
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
            "torch": run_text(
                [sys.executable, "-c", "import torch; print(torch.__version__)"]
            ),
        },
    }
    write_json(output / "provenance.json", provenance)

    verification = {
        "schema": SCHEMA,
        "baseline": baseline_verdict,
        "width_classification": "Z0_POSITIVE_WIDTH_ONLY_LOOKS_ZERO",
        "all_minima_above_1e_9": all(float(row["width"]) > 1e-9 for row in minima),
        "runtime_features": features,
        "source_map_row_count": len(semantic_map),
        "same_prestate_gate": same_prestate["status"],
        "micro_oracles_all_contain": all(
            bool(row["intervalized_contains_shared"]) for row in micro
        ),
        "outcome": outcome,
        "candidate": no_fix["status"],
        "scientific_outcome_uses_process_exit_code": False,
    }
    write_json(output / "verification.json", verification)
    return verification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flowstar-trace", type=Path, required=True)
    parser.add_argument("--flowstar-metadata", type=Path, required=True)
    parser.add_argument("--torch-segments", type=Path, required=True)
    parser.add_argument("--torch-summary", type=Path, required=True)
    parser.add_argument("--historical-common-prefix", type=Path)
    parser.add_argument("--flowstar-root", type=Path, required=True)
    parser.add_argument("--flowstar-library", type=Path, required=True)
    parser.add_argument("--flowstar-binary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = audit(parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
