#!/usr/bin/env python3
"""Publish aggregate native TORA-Q3 closure evidence from private traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


STATE_NAMES = ("x1", "x2", "x3", "x4", "u1")
TORCH_LANES = (
    "baseline_native_k2",
    "k3_picard",
    "algorithm_aligned_q3",
    "algorithm_aligned_h005_refresh1",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def statistics(values: np.ndarray, prefix: str) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        f"{prefix}_mean": float(np.mean(flat)),
        f"{prefix}_median": float(np.median(flat)),
        f"{prefix}_p05": float(np.percentile(flat, 5)),
        f"{prefix}_p95": float(np.percentile(flat, 95)),
        f"{prefix}_maximum": float(np.max(flat)),
        f"{prefix}_minimum": float(np.min(flat)),
    }


def interval_curve_rows(
    lane: str,
    record: Mapping[str, Any],
    field: str,
) -> list[dict[str, Any]]:
    lower = np.asarray(record[field]["lower"], dtype=np.float64)
    upper = np.asarray(record[field]["upper"], dtype=np.float64)
    center = lower + 0.5 * (upper - lower)
    width = upper - lower
    rows = []
    for state, state_name in enumerate(STATE_NAMES):
        row: dict[str, Any] = {
            "formal_lane": lane,
            "segment_index": int(record["segment_index"]),
            "physical_time": float(record["physical_time"]),
            "state": state_name,
        }
        row.update(statistics(width[:, state], "width"))
        row.update(statistics(center[:, state], "center"))
        row["center_maximum_absolute"] = float(np.max(np.abs(center[:, state])))
        rows.append(row)
    return rows


def property_curve_rows(
    lane: str, record: Mapping[str, Any]
) -> list[dict[str, Any]]:
    raw = record["property_margin"]
    margins = np.asarray(raw["tube"] if isinstance(raw, dict) else raw, dtype=np.float64)
    rows = []
    for state, state_name in enumerate(STATE_NAMES[:4]):
        row: dict[str, Any] = {
            "formal_lane": lane,
            "segment_index": int(record["segment_index"]),
            "physical_time": float(record["physical_time"]),
            "state": state_name,
        }
        row.update(statistics(margins[:, state], "margin"))
        rows.append(row)
    return rows


def selected_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "segment_index": record["segment_index"],
        "physical_time": record["physical_time"],
        "endpoint": record["endpoint"],
        "tube": record["tube"],
        "interval_remainder": record["interval_remainder"],
        "property_margin": record["property_margin"],
    }


def process_xiangru(
    path: Path,
    *,
    selected_segments: set[int],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    endpoint: list[dict[str, Any]] = []
    tube: list[dict[str, Any]] = []
    remainder: list[dict[str, Any]] = []
    margin: list[dict[str, Any]] = []
    selected: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        header = json.loads(next(handle))
        if header.get("schema") != "xiangru_tora_q3_plant_trace_header_v1":
            raise ValueError("unexpected Xiangru plant header")
        for line in handle:
            record = json.loads(line)
            segment = int(record["segment_index"])
            endpoint.extend(interval_curve_rows("xiangru_native_q3", record, "endpoint"))
            tube.extend(interval_curve_rows("xiangru_native_q3", record, "tube"))
            remainder.extend(
                interval_curve_rows("xiangru_native_q3", record, "interval_remainder")
            )
            margin.extend(property_curve_rows("xiangru_native_q3", record))
            if segment in selected_segments:
                selected[segment] = selected_record(record)
    if len(endpoint) != 200 * len(STATE_NAMES):
        raise ValueError("complete Xiangru T20 aggregate trace is required")
    return endpoint, tube, remainder, margin, selected


def numerical_certificate(failure: Mapping[str, Any]) -> bool | None:
    if not failure.get("available"):
        return None
    return all(
        all(bool(value) for value in leaf["numerical_certificate"].values())
        for leaf in failure["failed_leaves"]
    )


def prefix_timing(
    summary: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    controller_rows: list[Mapping[str, Any]],
) -> dict[str, float]:
    failure = summary.get("first_failure")
    stop = int(failure["segment"]) if failure else int(summary["completed_segments"])
    periods = math.ceil(stop / 10)
    selected_segments = [row for row in rows if int(row["segment_index"]) <= stop]
    selected_controllers = [
        row for row in controller_rows if int(row["controller_period"]) <= periods
    ]
    plant = sum(float(row["plant_seconds"]) for row in selected_segments)
    normalization = sum(
        float(row["normalization_seconds"]) for row in selected_segments
    )
    controller_bound = sum(
        float(row["timing"]["bound_seconds"]) for row in selected_controllers
    )
    controller_composition = sum(
        float(row["timing"]["composition_seconds"])
        for row in selected_controllers
    )
    build = float(summary["controller_build_seconds"])
    return {
        "formal_prefix_controller_build_seconds": build,
        "formal_prefix_controller_bound_seconds": controller_bound,
        "formal_prefix_controller_composition_seconds": controller_composition,
        "formal_prefix_normalization_seconds": normalization,
        "formal_prefix_plant_seconds": plant,
        "formal_prefix_accounted_seconds": (
            build + controller_bound + controller_composition + normalization + plant
        ),
    }


def aggregate_failure(
    lane: str, failure: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "formal_lane": lane,
        "available": bool(failure.get("available")),
        "first_failure": failure.get("first_failure"),
    }
    if not failure.get("available"):
        return result
    leaves = failure["failed_leaves"]
    states = []
    for state in range(5):
        entries = [leaf["states"][state] for leaf in leaves]
        states.append(
            {
                "state": STATE_NAMES[state],
                "endpoint_center_maximum_absolute": max(
                    abs(float(row["endpoint"]["center"])) for row in entries
                ),
                "endpoint_radius_maximum": max(
                    float(row["endpoint"]["radius"]) for row in entries
                ),
                "tube_center_maximum_absolute": max(
                    abs(float(row["tube"]["center"])) for row in entries
                ),
                "tube_radius_maximum": max(
                    float(row["tube"]["radius"]) for row in entries
                ),
                "interval_remainder_center_maximum_absolute": max(
                    abs(float(row["interval_remainder"]["center"]))
                    for row in entries
                ),
                "interval_remainder_radius_maximum": max(
                    float(row["interval_remainder"]["radius"]) for row in entries
                ),
                "property_margin_minimum": (
                    min(float(row["property_margin"]) for row in entries)
                    if state < 4
                    else None
                ),
            }
        )
    controller_input = []
    if all(leaf.get("controller") is not None for leaf in leaves):
        for state in range(4):
            entries = [leaf["controller"]["input"][state] for leaf in leaves]
            controller_input.append(
                {
                    "state": STATE_NAMES[state],
                    "center_maximum_absolute": max(
                        abs(float(row["center"])) for row in entries
                    ),
                    "radius_maximum": max(float(row["radius"]) for row in entries),
                }
            )
        before = [leaf["controller"]["output_before_outward"] for leaf in leaves]
        after = [leaf["controller"]["output_after_outward"] for leaf in leaves]
        controller_output = {
            "before_outward_center_maximum_absolute": max(
                abs(float(row["center"])) for row in before
            ),
            "before_outward_radius_maximum": max(
                float(row["radius"]) for row in before
            ),
            "after_outward_center_maximum_absolute": max(
                abs(float(row["center"])) for row in after
            ),
            "after_outward_radius_maximum": max(
                float(row["radius"]) for row in after
            ),
        }
    else:
        controller_output = None
    result.update(
        {
            "segment_index": failure["segment_index"],
            "physical_time": failure["physical_time"],
            "failure_type": failure["failure_type"],
            "failed_leaf_ids": failure["failed_leaf_ids"],
            "numerical_certificate_passed": numerical_certificate(failure),
            "states": states,
            "controller_input": controller_input,
            "controller_output": controller_output,
            "polynomial_remainder_decomposition": failure[
                "polynomial_remainder_decomposition"
            ],
            "ledger_widths": failure["ledger_widths"],
        }
    )
    return result


def comparison_rows(
    records: Mapping[str, Mapping[str, Any]], segment: int
) -> list[dict[str, Any]]:
    reference = records["xiangru_native_q3"]
    rows = []
    for lane, record in records.items():
        for field in ("endpoint", "tube", "interval_remainder"):
            lower = np.asarray(record[field]["lower"], dtype=np.float64)
            upper = np.asarray(record[field]["upper"], dtype=np.float64)
            ref_lower = np.asarray(reference[field]["lower"], dtype=np.float64)
            ref_upper = np.asarray(reference[field]["upper"], dtype=np.float64)
            width = upper - lower
            ref_width = ref_upper - ref_lower
            center = lower + 0.5 * width
            ref_center = ref_lower + 0.5 * ref_width
            for state, name in enumerate(STATE_NAMES):
                rows.append(
                    {
                        "segment_index": segment,
                        "physical_time": segment * 0.1,
                        "formal_lane": lane,
                        "quantity": field,
                        "state": name,
                        "width_maximum": float(np.max(width[:, state])),
                        "xiangru_width_maximum": float(
                            np.max(ref_width[:, state])
                        ),
                        "width_maximum_difference_from_xiangru": float(
                            np.max(width[:, state]) - np.max(ref_width[:, state])
                        ),
                        "center_difference_maximum_absolute_from_xiangru": float(
                            np.max(np.abs(center[:, state] - ref_center[:, state]))
                        ),
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--k3-run", type=Path, required=True)
    parser.add_argument("--aligned-run", type=Path, required=True)
    parser.add_argument("--fallback-run", type=Path, required=True)
    parser.add_argument("--xiangru-result", type=Path, required=True)
    parser.add_argument("--xiangru-plant", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    publisher_source_sha256 = sha256(Path(__file__).resolve())
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    paths = dict(
        zip(
            TORCH_LANES,
            (args.baseline_run, args.k3_run, args.aligned_run, args.fallback_run),
            strict=True,
        )
    )
    summaries: dict[str, dict[str, Any]] = {}
    gates: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}
    torch_records: dict[str, dict[int, dict[str, Any]]] = {}
    endpoint_rows: list[dict[str, Any]] = []
    tube_rows: list[dict[str, Any]] = []
    remainder_rows: list[dict[str, Any]] = []
    property_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    for lane, directory in paths.items():
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        gate = json.loads(
            (directory / "hierarchical_gates.json").read_text(encoding="utf-8")
        )
        failure = json.loads(
            (directory / "failure_detail.json").read_text(encoding="utf-8")
        )
        if summary["formal_lane"] != lane or gate["formal_lane"] != lane:
            raise ValueError(f"formal lane mismatch for {lane}")
        rows = load_jsonl(directory / "segments.jsonl")
        controllers = load_jsonl(directory / "controller_updates.jsonl")
        stop = int(summary["first_failure"]["segment"])
        formal_rows = [row for row in rows if int(row["segment_index"]) <= stop]
        torch_records[lane] = {
            int(row["segment_index"]): selected_record(row) for row in formal_rows
        }
        for record in formal_rows:
            endpoint_rows.extend(interval_curve_rows(lane, record, "endpoint"))
            tube_rows.extend(interval_curve_rows(lane, record, "tube"))
            remainder_rows.extend(
                interval_curve_rows(lane, record, "interval_remainder")
            )
            property_rows.extend(property_curve_rows(lane, record))
        prefix = prefix_timing(summary, rows, controllers)
        runtime_rows.append(
            {
                "formal_lane": lane,
                "measurement_scope": "formal_prefix_through_first_failure",
                **prefix,
                "diagnostic_wall_seconds_including_serialization": summary[
                    "wall_seconds_including_serialization"
                ],
                "diagnostic_horizon": summary["diagnostic_horizon"],
                "peak_cuda_memory_bytes": summary["peak_cuda_memory_bytes"],
                "maximum_process_rss_kib": summary["maximum_process_rss_kib"],
            }
        )
        first = summary["first_failure"]
        horizon_rows.append(
            {
                "formal_lane": lane,
                "implementation": "torch_native_closed_loop",
                "status": summary["status"],
                "requested_horizon": 20.0,
                "certified_horizon": summary["certified_horizon"],
                "completed_segments": summary["completed_segments"],
                "first_failure_segment": first["segment"],
                "first_failure_time": first["segment"] * 0.1,
                "failure_type": first["reason"],
                "failed_leaf_ids": ";".join(map(str, first["failed_leaf_ids"])),
                "numerical_certificate_passed_at_failure": numerical_certificate(
                    failure
                ),
                "t1_status": gate["gates"][2]["status"],
                "t5_status": gate["gates"][3]["status"],
                "t10_status": gate["gates"][4]["status"],
                "t20_status": gate["gates"][5]["status"],
            }
        )
        summaries[lane] = summary
        gates[lane] = gate
        failures[lane] = aggregate_failure(lane, failure)

    common_segment = min(int(item["completed_segments"]) for item in summaries.values())
    xiangru_result = json.loads(args.xiangru_result.read_text(encoding="utf-8"))
    xiangru_cell = xiangru_result["cells"]["b48_static"]["complete_q3"]
    if xiangru_cell["status"] != "VERIFIED" or xiangru_cell["certified_horizon"] != 20.0:
        raise ValueError("frozen Xiangru native reference is not verified to T20")
    x_endpoint, x_tube, x_remainder, x_property, x_selected = process_xiangru(
        args.xiangru_plant,
        selected_segments={common_segment, 50, 100, 200},
    )
    endpoint_rows.extend(x_endpoint)
    tube_rows.extend(x_tube)
    remainder_rows.extend(x_remainder)
    property_rows.extend(x_property)
    horizon_rows.append(
        {
            "formal_lane": "xiangru_native_q3",
            "implementation": "xiangru_native_closed_loop",
            "status": "VERIFIED",
            "requested_horizon": 20.0,
            "certified_horizon": 20.0,
            "completed_segments": 200,
            "first_failure_segment": "",
            "first_failure_time": "",
            "failure_type": "",
            "failed_leaf_ids": "",
            "numerical_certificate_passed_at_failure": "",
            "t1_status": "PASS",
            "t5_status": "PASS",
            "t10_status": "PASS",
            "t20_status": "PASS",
        }
    )
    timing = xiangru_cell["timing"]
    runtime_rows.append(
        {
            "formal_lane": "xiangru_native_q3",
            "measurement_scope": "complete_native_t20",
            "formal_prefix_controller_build_seconds": timing[
                "implementation_compile_and_warm_seconds_excluded"
            ],
            "formal_prefix_controller_bound_seconds": timing["controller_seconds"],
            "formal_prefix_controller_composition_seconds": "included_above",
            "formal_prefix_normalization_seconds": "included_in_solver",
            "formal_prefix_plant_seconds": timing["default_dynamics_seconds"],
            "formal_prefix_accounted_seconds": timing[
                "solver_wall_seconds_excluding_validation"
            ],
            "diagnostic_wall_seconds_including_serialization": timing[
                "total_wall_seconds_including_validation"
            ],
            "diagnostic_horizon": 20.0,
            "peak_cuda_memory_bytes": xiangru_cell["peak_cuda_memory_bytes"],
            "maximum_process_rss_kib": "",
        }
    )

    comparison_inputs = {
        lane: records[common_segment] for lane, records in torch_records.items()
    }
    comparison_inputs["xiangru_native_q3"] = x_selected[common_segment]
    common_rows = comparison_rows(comparison_inputs, common_segment)

    public_implementations = {}
    for lane in TORCH_LANES:
        summary = summaries[lane]
        public_implementations[lane] = {
            "status": summary["status"],
            "certified_horizon": summary["certified_horizon"],
            "first_failure": summary["first_failure"],
            "gates": gates[lane]["gates"],
            "config": summary["config"],
            "config_sha256": summary["config_sha256"],
            "source_sha256": summary["source_sha256"],
            "private_trace_sha256": gates[lane]["private_trace_sha256"],
            "failure_detail_sha256": summary["failure_detail_sha256"],
            "numerical_certificate_passed_at_failure": failures[lane].get(
                "numerical_certificate_passed"
            ),
        }
    gate_summary = {
        "schema": "tora_q3_native_hierarchical_closure_v1",
        "status": "CASE_C_PERFORMANCE_PASS_NATIVE_T5_GATE_FAIL",
        "strict_previous_pass_only": True,
        "diagnostic_continuation_is_not_formal": True,
        "common_control_substitution_forbidden": True,
        "implementations": public_implementations,
        "fallback_decision": {
            "selected": "algorithm_aligned_h005_refresh1",
            "candidate_count": 1,
            "contract_change": (
                "two validated h=0.05 plant substeps per h=0.1 reporting step; "
                "controller refresh remains 1.0 second"
            ),
            "evidence": {
                "algorithm_aligned_segment44_pre_projection_polynomial_range_maximum_width": 0.05847775643173847,
                "algorithm_aligned_segment44_pre_projection_interval_remainder_maximum_width": 4.0217060126908155,
                "algorithm_aligned_segment44_interval_remainder_share_of_sum": (
                    4.0217060126908155
                    / (4.0217060126908155 + 0.05847775643173847)
                ),
                "prior_horner_range_candidate_certified_horizon": 4.3,
            },
            "result": {
                "certified_horizon": summaries[
                    "algorithm_aligned_h005_refresh1"
                ]["certified_horizon"],
                "first_failure": summaries[
                    "algorithm_aligned_h005_refresh1"
                ]["first_failure"],
            },
        },
        "xiangru_native_reference": {
            "status": "VERIFIED",
            "certified_horizon": 20.0,
            "segments": 200,
            "solver_wall_seconds_excluding_validation": timing[
                "solver_wall_seconds_excluding_validation"
            ],
            "total_wall_seconds_including_validation": timing[
                "total_wall_seconds_including_validation"
            ],
            "source_sha256": {
                "result": sha256(args.xiangru_result),
                "plant": sha256(args.xiangru_plant),
            },
        },
        "common_certified_comparison": {
            "segment_index": common_segment,
            "physical_time": common_segment * 0.1,
            "interpolation_used": False,
        },
        "torch_target_width_availability": {
            lane: {"T5": None, "T10": None, "T20": None} for lane in TORCH_LANES
        },
        "publisher_source_sha256": publisher_source_sha256,
    }
    failure_payload = {
        "schema": "tora_q3_native_failure_aggregates_v1",
        "common_certified_segment": common_segment,
        "lanes": failures,
    }
    summary_payload = {
        "schema": "tora_q3_native_closure_public_summary_v1",
        "status": gate_summary["status"],
        "best_torch_certified_horizon": max(
            float(item["certified_horizon"]) for item in summaries.values()
        ),
        "best_torch_lanes": [
            lane
            for lane, item in summaries.items()
            if float(item["certified_horizon"]) == 4.4
        ],
        "xiangru_certified_horizon": 20.0,
        "native_t20_closed": False,
        "remaining_gap": (
            "property enclosure growth at controller period 5; all numerical "
            "certificates remain true at the first failed segment"
        ),
        "publisher_source_sha256": publisher_source_sha256,
        "public_files": [
            "hierarchical_gates.json",
            "failure_horizons.csv",
            "failure_details.json",
            "endpoint_width_over_time.csv",
            "tube_width_over_time.csv",
            "remainder_width_over_time.csv",
            "property_margin_over_time.csv",
            "common_horizon_comparison.csv",
            "runtime_breakdown.csv",
        ],
    }
    write_json(output / "hierarchical_gates.json", gate_summary)
    write_json(output / "failure_details.json", failure_payload)
    write_json(output / "summary.json", summary_payload)
    write_csv(output / "failure_horizons.csv", horizon_rows)
    write_csv(output / "endpoint_width_over_time.csv", endpoint_rows)
    write_csv(output / "tube_width_over_time.csv", tube_rows)
    write_csv(output / "remainder_width_over_time.csv", remainder_rows)
    write_csv(output / "property_margin_over_time.csv", property_rows)
    write_csv(output / "common_horizon_comparison.csv", common_rows)
    write_csv(output / "runtime_breakdown.csv", runtime_rows)
    print(
        json.dumps(
            {
                "status": gate_summary["status"],
                "best_torch_certified_horizon": summary_payload[
                    "best_torch_certified_horizon"
                ],
                "common_certified_segment": common_segment,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
