#!/usr/bin/env python3
"""Build separate native-capability and pairwise tables without rankings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


FLOWSTAR_SHA = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
DIFFREACH_SHA = "dd628eb443b517d6415de93e7035b4baef73963e"


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty table {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    flow = _json(args.flow_summary)
    stock = _json(args.stock_diffreach_summary)
    diff = _json(args.diffreach_summary)
    torch_dr7 = _json(args.torch_dr7_summary)
    comparison = _json(args.diff_comparison)
    common = _csv(args.flow_common_prefix)
    if flow["outcome"] != "FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_ONLY":
        raise RuntimeError("unexpected Flow*/Torch outcome")
    if comparison["outcome"] != "DIFFREACH_TORCH_DR7_FULL_HORIZON_DIVERGED":
        raise RuntimeError("unexpected DiffReach/Torch outcome")
    if any(row["both_completed"] != "True" for row in common):
        raise RuntimeError("M-F row extends beyond common completion")

    native = [
        {
            "tool": "Flow*",
            "source_commit": FLOWSTAR_SHA,
            "native_or_observer": "read-only fixed-schedule observer",
            "representation": "complete total-degree O4",
            "partition": "B1",
            "schedule": "fixed h=0.01, 1000 steps",
            "requested_horizon": 10.0,
            "validated_horizon": flow["flowstar"]["validated_horizon"],
            "completion": "completed",
            "available_output_object": "endpoint, segment tube, prefix tube",
            "soundness_scope": "Flow* build; scalar-affine under-enclosure qualification open",
            "runtime_scope": f"cold process wall {flow['flowstar']['process_wall_s']} s; no matched ratio",
        },
        {
            "tool": "Torch complete-O4",
            "source_commit": args.torch_complete_source_sha,
            "native_or_observer": "native solver plus read-only trace",
            "representation": "fixed R35 complete total-degree O4",
            "partition": "B1",
            "schedule": "fixed h=0.01 to first failure",
            "requested_horizon": 10.0,
            "validated_horizon": flow["torch"]["validated_horizon"],
            "completion": "failed candidate step 633",
            "available_output_object": "endpoint, segment tube, prefix tube through step 632",
            "soundness_scope": "ordinary-float64 empirical",
            "runtime_scope": f"cold core {flow['torch']['process_wall_s']} s; no matched ratio",
        },
        {
            "tool": "DiffReach stock",
            "source_commit": DIFFREACH_SHA,
            "native_or_observer": "unmodified stock driver",
            "representation": "fixed DR7, mixed builder dtype",
            "partition": "B64",
            "schedule": "fixed h=0.01, 1000 steps",
            "requested_horizon": stock["horizon_requested"],
            "validated_horizon": stock["horizon_validated"],
            "completion": "completed",
            "available_output_object": "endpoint sequence only",
            "soundness_scope": "native capability only; mixed builder dtype",
            "runtime_scope": f"stock process wall {stock['process_wall_seconds']} s; not pairwise eligible",
        },
        {
            "tool": "DiffReach explicit-f64",
            "source_commit": diff["source_sha"],
            "native_or_observer": "minimal read-only upstream patch",
            "representation": "fixed DR7 explicit float64",
            "partition": "B64",
            "schedule": "fixed h=0.01, 1000 steps",
            "requested_horizon": 10.0,
            "validated_horizon": diff["validated_horizon"],
            "completion": diff["completion_status"],
            "available_output_object": "endpoint, segment tube, prefix tube and operator trace",
            "soundness_scope": "ordinary-float64 empirical",
            "runtime_scope": f"hash/capture diagnostic {diff['runtime_with_hashing_and_capture_s']} s; pairwise timing ineligible",
        },
        {
            "tool": "Torch DR7",
            "source_commit": torch_dr7["source_sha"],
            "native_or_observer": "read-only native step fields",
            "representation": "fixed DR7 explicit float64",
            "partition": "B64",
            "schedule": "fixed h=0.01, 1000 steps",
            "requested_horizon": 10.0,
            "validated_horizon": torch_dr7["validated_horizon"],
            "completion": torch_dr7["completion_status"],
            "available_output_object": "endpoint, segment tube, prefix tube and operator trace",
            "soundness_scope": "ordinary-float64 empirical",
            "runtime_scope": f"hash/capture diagnostic {torch_dr7['runtime_with_hashing_and_capture_s']} s; pairwise timing ineligible",
        },
    ]
    native_path = output / "table_n_native_capability.csv"
    _write(native_path, native)

    mf_fields = (
        "time", "both_completed",
        "flowstar_endpoint_x_width", "flowstar_endpoint_y_width",
        "torch_endpoint_x_width", "torch_endpoint_y_width",
        "flowstar_segment_tube_x_width", "flowstar_segment_tube_y_width",
        "torch_segment_tube_x_width", "torch_segment_tube_y_width",
        "flowstar_prefix_tube_x_width", "flowstar_prefix_tube_y_width",
        "torch_prefix_tube_x_width", "torch_prefix_tube_y_width",
        "flowstar_margin_x", "flowstar_margin_y", "torch_margin_x", "torch_margin_y",
        "flowstar_cumulative_runtime_s", "torch_cumulative_runtime_s", "qualification",
    )
    mf = [
        {
            **{field: row[field] for field in mf_fields},
            "runtime_to_common_prefix_scope": "diagnostic_cumulative_not_matched_timing_ratio",
        }
        for row in common
    ]
    mf_path = output / "table_mf_flowstar_torch_common_prefix.csv"
    _write(mf_path, mf)

    md = [
        {
            "scope": comparison["scope"],
            "batch": comparison["batch_size"],
            "h": comparison["step_size"],
            "requested_horizon": comparison["steps"] * comparison["step_size"],
            "operator_equality": comparison["operator_equality"],
            "mask_equality": comparison["mask_equality"],
            "remainder_equality": comparison["first_divergence_by_field"]["retained_R_lo"] is None and comparison["first_divergence_by_field"]["retained_R_hi"] is None,
            "endpoint_tube_equality": comparison["endpoint_tube_equality"],
            "j_phi_equality": comparison["j_phi_equality"],
            "cpu_runtime": "diagnostic traces only; semantics gate diverged",
            "gpu_runtime": "unavailable; CPU semantics gate diverged",
            "soundness_scope": comparison["soundness_scope"],
            "outcome": comparison["outcome"],
        }
    ]
    md_path = output / "table_md_diffreach_torch_explicit_f64.csv"
    _write(md_path, md)
    artifacts = [
        {"path": path.name, "sha256": _sha(path), "rows": sum(1 for _ in path.open(encoding="utf-8")) - 1}
        for path in (native_path, mf_path, md_path)
    ]
    summary = {
        "schema": "full_horizon_pairwise_tables_v1",
        "outcome": "PAIRWISE_TABLES_BUILT_WITHOUT_UNIVERSAL_RANKING",
        "artifacts": artifacts,
        "prohibited_ratios_emitted": False,
        "universal_ranking_emitted": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return summary


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-summary", type=Path, required=True)
    parser.add_argument("--flow-common-prefix", type=Path, required=True)
    parser.add_argument("--stock-diffreach-summary", type=Path, required=True)
    parser.add_argument("--diffreach-summary", type=Path, required=True)
    parser.add_argument("--torch-dr7-summary", type=Path, required=True)
    parser.add_argument("--diff-comparison", type=Path, required=True)
    parser.add_argument("--torch-complete-source-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(_args())
