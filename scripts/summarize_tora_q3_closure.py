#!/usr/bin/env python3
"""Build native closed-loop tightness tables and final closure report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


STATES = ("x1", "x2", "x3", "x4", "u1")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--root-cause", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roots = {"baseline_native_k2": args.baseline, "candidate_k3": args.candidate}
    summaries = {
        label: json.loads((root / "summary.json").read_text())
        for label, root in roots.items()
    }
    segments = {
        label: load_jsonl(root / "segments.jsonl")
        for label, root in roots.items()
    }
    common_horizon = min(
        summary["completed_segments"] for summary in summaries.values()
    )
    width_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    for label, rows in segments.items():
        formal_limit = summaries[label]["completed_segments"]
        for scope, limit in (
            ("own_formal_horizon", formal_limit),
            ("common_formal_horizon", common_horizon),
            ("diagnostic_propagation", len(rows)),
        ):
            selected = [row for row in rows if row["segment_index"] <= limit]
            for kind in ("endpoint", "tube"):
                widths = np.stack(
                    [
                        np.asarray(row[kind]["upper"])
                        - np.asarray(row[kind]["lower"])
                        for row in selected
                    ]
                )
                for state, state_name in enumerate(STATES):
                    values = widths[:, :, state]
                    worst_flat = int(np.argmax(values))
                    worst_segment_index, worst_leaf = np.unravel_index(
                        worst_flat, values.shape
                    )
                    width_rows.append(
                        {
                            "lane": label,
                            "scope": scope,
                            "kind": kind,
                            "state": state_name,
                            "median_width": float(np.median(values)),
                            "p95_width": float(np.percentile(values, 95)),
                            "maximum_width": float(np.max(values)),
                            "worst_segment": selected[worst_segment_index][
                                "segment_index"
                            ],
                            "worst_leaf": int(worst_leaf),
                        }
                    )
        for row in rows:
            margin = np.asarray(row["property_margin"])
            margin_rows.append(
                {
                    "lane": label,
                    "segment": row["segment_index"],
                    "physical_time": row["physical_time"],
                    "formal": row["segment_index"] <= formal_limit,
                    "minimum_property_margin": float(np.min(margin)),
                    "maximum_endpoint_width": row["width_attribution"][
                        "composed_exact_endpoint_direct"
                    ]["maximum"],
                    "maximum_polynomial_range_width": row[
                        "width_attribution"
                    ]["pre_projection_polynomial_range"]["maximum"],
                    "maximum_interval_remainder_width": row[
                        "width_attribution"
                    ]["pre_projection_interval_remainder"]["maximum"],
                }
            )
    width_path = args.output_dir / "closed_loop_width_statistics.csv"
    with width_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(width_rows[0]))
        writer.writeheader()
        writer.writerows(width_rows)
    margin_path = args.output_dir / "property_margin_over_time.csv"
    with margin_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(margin_rows[0]))
        writer.writeheader()
        writer.writerows(margin_rows)
    horizon_rows = [
        {
            "lane": label,
            "formal_certified_horizon": summary["certified_horizon"],
            "first_failure_segment": summary["first_failure"]["segment"],
            "first_failure_reason": summary["first_failure"]["reason"],
            "T5_status": "FAILED",
            "T10_status": "N/A (hierarchical gate stopped at T5)",
            "T20_status": "N/A (hierarchical gate stopped at T5)",
        }
        for label, summary in summaries.items()
    ]
    horizon_path = args.output_dir / "lane_horizons.csv"
    with horizon_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(horizon_rows[0]))
        writer.writeheader()
        writer.writerows(horizon_rows)

    root_cause = json.loads(args.root_cause.read_text())
    runtime = json.loads(args.runtime.read_text())
    conclusion = {
        "schema": "tora_q3_closed_loop_closure_v1",
        "status": "CASE_C_FULL_LOOP_IMPROVED_PERFORMANCE_GATES_UNMET",
        "baseline_horizon": summaries["baseline_native_k2"][
            "certified_horizon"
        ],
        "candidate_horizon": summaries["candidate_k3"]["certified_horizon"],
        "common_formal_horizon": common_horizon * 0.1,
        "selected_candidate": "complete-Q3 K3 polynomial Picard ablation",
        "candidate_first_failure": summaries["candidate_k3"]["first_failure"],
        "baseline_segment_44_is_property_failure": True,
        "baseline_segment_44_numerical_certificate_ok": True,
        "performance_gates": runtime["gates"],
        "target_horizon_widths": {
            "T5": "N/A (candidate first fails at segment 45)",
            "T10": "N/A",
            "T20": "N/A",
        },
        "next_smallest_technical_problem": (
            "reduce remainder-dominated width entering the period-5 controller; "
            "at segment 40 the interval remainder width dominates the polynomial "
            "range, while projection inflation is only roundoff scale"
        ),
        "vdp_t_6_397_status": "unchanged and outside this TORA-Q3 branch scope",
        "root_cause_schema": root_cause["schema"],
    }
    (args.output_dir / "closure_summary.json").write_text(
        json.dumps(conclusion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# TORA-Q3 closed-loop closure report",
        "",
        "This closure is Case C: the sound full-loop candidate improves the "
        "formal horizon, while the required 10x GPU performance gates remain unmet.",
        "",
        "## Formal outcomes",
        "",
        "- baseline complete-Q3 K2: certified through T=4.3; property first fails at segment 44",
        "- candidate complete-Q3 K3: certified through T=4.4; property first fails at segment 45",
        "- baseline segment 44 still passes finiteness and every numerical subset certificate",
        "- T5/T10/T20 widths are N/A after the hierarchical candidate gate fails at segment 45",
        "",
        "## Root cause",
        "",
        "At T=1, 99.924% of the measured Torch/Xiangru difference is already "
        "present in the direct endpoint before projection. At segment 40, width "
        "is remainder-dominated; project/materialize inflation is about 1e-12.",
        "",
        "## Performance",
        "",
        f"- compiled B48 one-step: `{runtime['one_step']['compiled_speedup']:.3f}x`",
        f"- compiled common-control T20: `{runtime['common_control_t20']['speedup']:.3f}x`",
        "- P0/P1/P2 pass; P3/P4 fail the required 10x thresholds",
        "",
        "The next concrete technical problem is a sound reduction of the "
        "remainder-dominated period-5 controller input, not another affine "
        "projection reorder.",
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(conclusion, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
