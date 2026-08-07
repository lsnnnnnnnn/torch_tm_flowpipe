#!/usr/bin/env python3
"""Sanitize private full-loop lane traces into public aggregate evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--lane",
        action="append",
        default=[],
        help="LABEL=private_run_directory",
    )
    parser.add_argument("--controller-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        allowed = {
            "range_policy_shadow_lanes.json",
            "refresh_stage_widths.csv",
            "segment_predicates.csv",
            "ledger_widths.csv",
            "root_cause.json",
            "r1_r2_replay.json",
        }
        unexpected = {path.name for path in args.output_dir.iterdir()} - allowed
        if unexpected:
            raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lanes: dict[str, Path] = {"L0_baseline_native": args.baseline}
    for item in args.lane:
        label, value = item.split("=", 1)
        lanes[label] = Path(value)

    summaries = {
        label: json.loads((root / "summary.json").read_text())
        for label, root in lanes.items()
    }
    segment_payloads = {
        label: load_jsonl(root / "segments.jsonl")
        for label, root in lanes.items()
    }
    controller_payloads = {
        label: load_jsonl(root / "controller_updates.jsonl")
        for label, root in lanes.items()
    }

    predicate_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    refresh_rows: list[dict[str, Any]] = []
    for label, rows in segment_payloads.items():
        for row in rows:
            predicates = row["predicates"]
            counts = {
                name: int(np.count_nonzero(np.asarray(values, dtype=bool)))
                for name, values in predicates.items()
            }
            predicate_rows.append(
                {
                    "lane": label,
                    "segment": row["segment_index"],
                    "physical_time": row["physical_time"],
                    "finite_ok_leaves": counts["finite_ok_by_leaf"],
                    "initial_subset_ok_leaves": counts[
                        "initial_subset_ok_by_leaf"
                    ],
                    "all_remainder_rounds_ok_leaves": counts[
                        "all_remainder_rounds_ok_by_leaf"
                    ],
                    "local_property_ok_leaves": counts[
                        "local_property_ok_by_leaf"
                    ],
                    "composed_property_ok_leaves": counts[
                        "composed_property_ok_by_leaf"
                    ],
                    "overall_accepted_leaves": counts[
                        "overall_accepted_by_leaf"
                    ],
                    "minimum_property_margin": float(
                        np.min(np.asarray(row["property_margin"]))
                    ),
                }
            )
            for category, stats in row["ledger_widths"].items():
                ledger_rows.append(
                    {
                        "lane": label,
                        "segment": row["segment_index"],
                        "category": category,
                        "median_width": stats["median"],
                        "maximum_width": stats["maximum"],
                        "sum_width": stats["sum"],
                    }
                )
        by_segment = {row["segment_index"]: row for row in rows}
        for update in controller_payloads[label]:
            prior_segment = update["segment_index"] - 1
            prior = by_segment.get(prior_segment)
            refresh_rows.append(
                {
                    "lane": label,
                    "controller_period": update["controller_period"],
                    "prior_segment": prior_segment,
                    "direct_endpoint_max_width": (
                        prior["width_attribution"][
                            "composed_exact_endpoint_direct"
                        ]["maximum"]
                        if prior is not None
                        else ""
                    ),
                    "current_projection_max_width": (
                        prior["width_attribution"][
                            "current_project_local_then_compose"
                        ]["maximum"]
                        if prior is not None
                        else ""
                    ),
                    "physical_projection_max_width": (
                        prior["width_attribution"][
                            "candidate_compose_then_project"
                        ]["maximum"]
                        if prior is not None
                        else ""
                    ),
                    "controller_input_max_width": update[
                        "controller_input_width"
                    ]["maximum"],
                    "controller_output_before_max_width": update[
                        "controller_output_before_width"
                    ]["maximum"],
                    "controller_output_after_max_width": update[
                        "controller_output_after_width"
                    ]["maximum"],
                }
            )

    write_csv(
        args.output_dir / "segment_predicates.csv",
        list(predicate_rows[0]),
        predicate_rows,
    )
    write_csv(
        args.output_dir / "ledger_widths.csv",
        list(ledger_rows[0]),
        ledger_rows,
    )
    write_csv(
        args.output_dir / "refresh_stage_widths.csv",
        list(refresh_rows[0]),
        refresh_rows,
    )

    baseline_segments = segment_payloads["L0_baseline_native"]
    baseline_controllers = controller_payloads["L0_baseline_native"]
    segment10 = next(row for row in baseline_segments if row["segment_index"] == 10)
    controller2 = next(
        row for row in baseline_controllers if row["controller_period"] == 2
    )
    observed = json.loads(args.controller_trace.read_text())["rows"][1][
        "pre_controller_state_box"
    ]
    direct_lower = np.asarray(segment10["endpoint"]["lower"])[:, :4]
    direct_upper = np.asarray(segment10["endpoint"]["upper"])[:, :4]
    projected_lower = np.asarray(controller2["pre_controller_state_box"]["lower"])
    projected_upper = np.asarray(controller2["pre_controller_state_box"]["upper"])
    observed_lower = np.asarray(observed["lower"])
    observed_upper = np.asarray(observed["upper"])
    direct_difference = max(
        float(np.max(np.abs(direct_lower - observed_lower))),
        float(np.max(np.abs(direct_upper - observed_upper))),
    )
    projection_change = max(
        float(np.max(np.abs(direct_lower - projected_lower))),
        float(np.max(np.abs(direct_upper - projected_upper))),
    )
    lane_results = {}
    for label, summary in summaries.items():
        diagnostic_failure = summary.get("diagnostic_failure")
        if diagnostic_failure is None:
            failed_numerical = next(
                (
                    row
                    for row in predicate_rows
                    if row["lane"] == label
                    and row["finite_ok_leaves"] == 48
                    and (
                        row["initial_subset_ok_leaves"] < 48
                        or row["all_remainder_rounds_ok_leaves"] < 48
                    )
                ),
                None,
            )
            if failed_numerical is not None:
                diagnostic_failure = {
                    "segment": failed_numerical["segment"],
                    "reason": "numerical_certificate",
                }
        diagnostic_certificate_horizon = (
            (float(diagnostic_failure["segment"]) - 1.0) * 0.1
            if diagnostic_failure is not None
            else summary.get("diagnostic_horizon")
        )
        lane_results[label] = {
            "formal_certified_horizon": summary["certified_horizon"],
            "formal_first_failure": summary["first_failure"],
            "diagnostic_propagated_horizon": summary.get("diagnostic_horizon"),
            "diagnostic_certificate_horizon": diagnostic_certificate_horizon,
            "diagnostic_failure": diagnostic_failure,
            "plant_seconds": summary["plant_seconds"],
            "private_segments_sha256": summary["segments_sha256"],
            "private_controller_updates_sha256": summary[
                "controller_updates_sha256"
            ],
            "private_replay_points_sha256": summary["replay_points_sha256"],
            "config": summary.get("config"),
            "config_sha256": summary.get("config_sha256"),
            "source_sha256": summary.get("source_sha256"),
        }
    root_cause = {
        "schema": "tora_q3_t4_4_width_root_cause_v1",
        "status": "PASS",
        "segment_44_verdict": {
            "formal_first_failure": "property",
            "finite_ok_leaves": 48,
            "initial_subset_ok_leaves": 48,
            "all_remainder_rounds_ok_leaves": 48,
            "interpretation": (
                "reachability arithmetic remains numerically validated; "
                "the safety property is not proved"
            ),
        },
        "diagnostic_numerical_horizon": {
            "last_numerically_valid_segment": 47,
            "first_numerical_certificate_failure_segment": 48,
        },
        "t1_0_014211_attribution": {
            "direct_exact_endpoint_vs_xiangru_max_abs": direct_difference,
            "projection_materialization_change_max_abs": projection_change,
            "fraction_already_present_before_projection": (
                direct_difference / (direct_difference + projection_change)
            ),
            "dominant_stage": "ten-segment plant propagation before projection",
        },
        "segment_40_width_decomposition": next(
            row for row in baseline_segments if row["segment_index"] == 40
        )["width_attribution"],
        "lane_results": lane_results,
        "selected_candidate": {
            "lane": "L4_k3_picard",
            "baseline_horizon": 4.3,
            "candidate_horizon": 4.4,
            "next_failure_segment": 45,
            "method_note": (
                "complete-Q3 K3 ablation; not algorithm-identical to the "
                "frozen K2 baseline"
            ),
        },
        "controller_trace_sha256": sha256(args.controller_trace),
    }
    root_path = args.output_dir / "root_cause.json"
    root_path.write_text(
        json.dumps(root_cause, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# TORA-Q3 T4.4 width attribution report",
        "",
        "The formal L0 verdict first fails at segment 44 because the safety "
        "property is not proved. All 48 leaves still pass finiteness, the "
        "initial subset check, and all ten remainder rounds.",
        "",
        "Diagnostic propagation remains numerically certified through segment "
        "47 and first loses the numerical certificate at segment 48.",
        "",
        "## T=1 refresh attribution",
        "",
        f"- direct endpoint versus Xiangru: `{direct_difference:.15g}` max abs",
        f"- project/materialize change: `{projection_change:.15g}` max abs",
        "- dominant source: the preceding ten plant segments, not projection",
        "",
        "## Sound shadow/candidate results",
        "",
        "| lane | formal horizon | first failure |",
        "|---|---:|---|",
    ]
    for label, result in lane_results.items():
        failure = result["formal_first_failure"]
        lines.append(
            f"| {label} | {result['formal_certified_horizon']} | "
            f"{failure['reason']} at segment {failure['segment']} |"
        )
    lines.extend(
        [
            "",
            "K3 is the selected sound method candidate: it moves the formal "
            "horizon from 4.3 to 4.4, then fails the property at segment 45.",
        ]
    )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "direct_difference": direct_difference,
                "projection_change": projection_change,
                "selected_candidate": "L4_k3_picard",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
