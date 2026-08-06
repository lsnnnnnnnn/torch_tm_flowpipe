#!/usr/bin/env python3
"""Fail-closed streaming comparison for two TORA common-control replays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterator


STATES = ("x1", "x2", "x3", "x4", "u1")
KINDS = ("endpoint", "tube")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def exact_contract(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    fields = ("controller_trace_sha256", "basis_variables", "basis_exponents", "slot_count")
    mismatches = [name for name in fields if left.get(name) != right.get(name)]
    if mismatches:
        raise ValueError(f"header contract mismatch: {mismatches}")
    if left.get("period_local_observation_restart") is not True or right.get("period_local_observation_restart") is not True:
        raise ValueError("both lanes must declare period-local observation restart")
    return {
        "matched_fields": list(fields),
        "basis_slot_permutation": list(range(int(left["slot_count"]))),
        "basis_bijective": True,
        "coefficient_comparison": "unavailable",
        "coefficient_blocker": (
            "The raw lanes do not expose a proved, tested conversion between their "
            "per-segment normalization center/scale coordinates."
        ),
    }


def assert_aligned_segment_keys(
    xiangru: dict[str, Any], torch_row: dict[str, Any]
) -> None:
    exact_key_fields = (
        "segment_index",
        "physical_time",
        "controller_period",
        "local_segment",
        "leaf_id",
    )
    mismatches = [
        field
        for field in exact_key_fields
        if xiangru.get(field) != torch_row.get(field)
    ]
    if mismatches:
        raise ValueError(
            f"exact alignment failed at next segment: {mismatches}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xiangru-jsonl", type=Path, required=True)
    parser.add_argument("--torch-jsonl", type=Path, required=True)
    parser.add_argument("--xiangru-summary", type=Path, required=True)
    parser.add_argument("--torch-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    xiangru_stream = rows(args.xiangru_jsonl)
    torch_stream = rows(args.torch_jsonl)
    xiangru_header = next(xiangru_stream)
    torch_header = next(torch_stream)
    contract = exact_contract(xiangru_header, torch_header)
    width_rows: list[dict[str, Any]] = []
    property_rows: list[dict[str, Any]] = []
    overlay_rows: list[dict[str, Any]] = []
    ratio_extremes: dict[tuple[str, str], dict[str, Any]] = {}
    target_horizon_rows: list[dict[str, Any]] = []
    first_status = None
    first_endpoint = None
    first_tube = None
    first_remainder = None
    compared_segments = 0

    sentinel = object()
    while True:
        xiangru = next(xiangru_stream, sentinel)
        torch_row = next(torch_stream, sentinel)
        if xiangru is sentinel and torch_row is sentinel:
            break
        if xiangru is sentinel or torch_row is sentinel:
            raise ValueError("segment count mismatch")
        assert_aligned_segment_keys(xiangru, torch_row)
        segment = int(xiangru["segment_index"])
        time = float(xiangru["physical_time"])
        period = int(xiangru["controller_period"])
        leaf_ids = xiangru["leaf_id"]
        if leaf_ids != list(range(48)):
            raise ValueError(f"noncanonical B48 leaf order at segment {segment}")
        if xiangru["accepted"] != torch_row["accepted"] and first_status is None:
            first_status = {"segment_index": segment, "physical_time": time}

        for kind in KINDS:
            x_lower = xiangru[kind]["lower"]
            x_upper = xiangru[kind]["upper"]
            t_lower = torch_row[kind]["lower"]
            t_upper = torch_row[kind]["upper"]
            for state_index, state in enumerate(STATES):
                x_width = [float(x_upper[leaf][state_index]) - float(x_lower[leaf][state_index]) for leaf in range(48)]
                t_width = [float(t_upper[leaf][state_index]) - float(t_lower[leaf][state_index]) for leaf in range(48)]
                x_remainder_width = [
                    float(xiangru["interval_remainder"]["upper"][leaf][state_index])
                    - float(xiangru["interval_remainder"]["lower"][leaf][state_index])
                    for leaf in range(48)
                ]
                t_remainder_width = [
                    float(torch_row["interval_remainder"]["upper"][leaf][state_index])
                    - float(torch_row["interval_remainder"]["lower"][leaf][state_index])
                    for leaf in range(48)
                ]
                x_polynomial_width = [
                    max(0.0, total - remainder)
                    for total, remainder in zip(x_width, x_remainder_width, strict=True)
                ]
                t_polynomial_width = [
                    max(0.0, total - remainder)
                    for total, remainder in zip(t_width, t_remainder_width, strict=True)
                ]
                leaf_ratios = [
                    (leaf, t_width[leaf] / x_width[leaf])
                    for leaf in range(48)
                    if x_width[leaf] != 0.0
                ]
                ratios = [ratio for _leaf, ratio in leaf_ratios]
                lower_diff = [abs(float(t_lower[leaf][state_index]) - float(x_lower[leaf][state_index])) for leaf in range(48)]
                upper_diff = [abs(float(t_upper[leaf][state_index]) - float(x_upper[leaf][state_index])) for leaf in range(48)]
                torch_contains = sum(
                    float(t_lower[leaf][state_index]) <= float(x_lower[leaf][state_index])
                    and float(t_upper[leaf][state_index]) >= float(x_upper[leaf][state_index])
                    for leaf in range(48)
                )
                xiangru_contains = sum(
                    float(x_lower[leaf][state_index]) <= float(t_lower[leaf][state_index])
                    and float(x_upper[leaf][state_index]) >= float(t_upper[leaf][state_index])
                    for leaf in range(48)
                )
                row = {
                    "lane": "common_control_plant_replay",
                    "segment_index": segment,
                    "physical_time": time,
                    "controller_period": period,
                    "state": state,
                    "enclosure_kind": kind,
                    "torch_width_median": median(t_width),
                    "torch_width_p95": percentile(t_width, 0.95),
                    "torch_width_max": max(t_width),
                    "xiangru_width_median": median(x_width),
                    "xiangru_width_p95": percentile(x_width, 0.95),
                    "xiangru_width_max": max(x_width),
                    "torch_over_xiangru_ratio_median": median(ratios) if ratios else "N/A",
                    "torch_over_xiangru_ratio_p95": percentile(ratios, 0.95) if ratios else "N/A",
                    "torch_over_xiangru_ratio_max": max(ratios) if ratios else "N/A",
                    "torch_hull_width": max(float(row[state_index]) for row in t_upper) - min(float(row[state_index]) for row in t_lower),
                    "xiangru_hull_width": max(float(row[state_index]) for row in x_upper) - min(float(row[state_index]) for row in x_lower),
                    "torch_polynomial_range_width_median": median(t_polynomial_width),
                    "torch_polynomial_range_width_p95": percentile(t_polynomial_width, 0.95),
                    "torch_polynomial_range_width_max": max(t_polynomial_width),
                    "xiangru_polynomial_range_width_median": median(x_polynomial_width),
                    "xiangru_polynomial_range_width_p95": percentile(x_polynomial_width, 0.95),
                    "xiangru_polynomial_range_width_max": max(x_polynomial_width),
                    "torch_interval_remainder_width_median": median(t_remainder_width),
                    "torch_interval_remainder_width_p95": percentile(t_remainder_width, 0.95),
                    "torch_interval_remainder_width_max": max(t_remainder_width),
                    "xiangru_interval_remainder_width_median": median(x_remainder_width),
                    "xiangru_interval_remainder_width_p95": percentile(x_remainder_width, 0.95),
                    "xiangru_interval_remainder_width_max": max(x_remainder_width),
                    "maximum_lower_abs_difference": max(lower_diff),
                    "maximum_upper_abs_difference": max(upper_diff),
                    "torch_contains_xiangru_leaf_count": torch_contains,
                    "xiangru_contains_torch_leaf_count": xiangru_contains,
                    "minimum_torch_property_margin": min(float(row[state_index]) for row in torch_row["property_margin"][kind]) if state_index < 4 else "N/A",
                    "minimum_xiangru_property_margin": min(float(row[state_index]) for row in xiangru["property_margin"][kind]) if state_index < 4 else "N/A",
                }
                width_rows.append(row)
                if leaf_ratios:
                    key = (kind, state)
                    smallest = min(leaf_ratios, key=lambda item: item[1])
                    largest = max(leaf_ratios, key=lambda item: item[1])
                    record = ratio_extremes.setdefault(key, {
                        "enclosure_kind": kind,
                        "state": state,
                        "minimum_ratio": math.inf,
                        "maximum_ratio": -math.inf,
                    })
                    if smallest[1] < record["minimum_ratio"]:
                        record.update({
                            "minimum_ratio": smallest[1],
                            "minimum_ratio_segment": segment,
                            "minimum_ratio_time": time,
                            "minimum_ratio_leaf_id": smallest[0],
                        })
                    if largest[1] > record["maximum_ratio"]:
                        record.update({
                            "maximum_ratio": largest[1],
                            "maximum_ratio_segment": segment,
                            "maximum_ratio_time": time,
                            "maximum_ratio_leaf_id": largest[0],
                        })
                    if segment == 200:
                        target_horizon_rows.append({
                            "physical_time": time,
                            "state": state,
                            "enclosure_kind": kind,
                            "torch_over_xiangru_ratio_median": median(ratios),
                            "torch_over_xiangru_ratio_p95": percentile(ratios, 0.95),
                            "torch_over_xiangru_ratio_max": largest[1],
                            "maximum_ratio_leaf_id": largest[0],
                            "torch_width_median": median(t_width),
                            "xiangru_width_median": median(x_width),
                        })
                for leaf in (0, 23, 47):
                    overlay_rows.append({
                        "lane": "common_control_plant_replay",
                        "segment_index": segment,
                        "physical_time": time,
                        "controller_period": period,
                        "leaf_id": leaf,
                        "state": state,
                        "enclosure_kind": kind,
                        "torch_lower": float(t_lower[leaf][state_index]),
                        "torch_upper": float(t_upper[leaf][state_index]),
                        "torch_width": t_width[leaf],
                        "xiangru_lower": float(x_lower[leaf][state_index]),
                        "xiangru_upper": float(x_upper[leaf][state_index]),
                        "xiangru_width": x_width[leaf],
                    })
                if state_index < 4:
                    for tool, margins in (
                        ("torch", torch_row["property_margin"][kind]),
                        ("xiangru", xiangru["property_margin"][kind]),
                    ):
                        values_for_state = [
                            float(leaf_margin[state_index])
                            for leaf_margin in margins
                        ]
                        property_rows.append({
                            "lane": "common_control_plant_replay",
                            "tool": tool,
                            "segment_index": segment,
                            "physical_time": time,
                            "controller_period": period,
                            "state": state,
                            "enclosure_kind": kind,
                            "margin_min": min(values_for_state),
                            "margin_median": median(values_for_state),
                            "margin_p95": percentile(values_for_state, 0.95),
                            "margin_max": max(values_for_state),
                        })
                if (max(lower_diff) != 0.0 or max(upper_diff) != 0.0):
                    target = "first_endpoint" if kind == "endpoint" else "first_tube"
                    if (target == "first_endpoint" and first_endpoint is None) or (target == "first_tube" and first_tube is None):
                        detail = {
                            "segment_index": segment,
                            "physical_time": time,
                            "controller_period": period,
                            "state": state,
                            "enclosure_kind": kind,
                            "maximum_lower_abs_difference": max(lower_diff),
                            "maximum_upper_abs_difference": max(upper_diff),
                        }
                        if kind == "endpoint":
                            first_endpoint = detail
                        else:
                            first_tube = detail

        x_rem = xiangru["interval_remainder"]
        t_rem = torch_row["interval_remainder"]
        if first_remainder is None:
            for leaf in range(48):
                for state_index, state in enumerate(STATES):
                    x_width = float(x_rem["upper"][leaf][state_index]) - float(x_rem["lower"][leaf][state_index])
                    t_width = float(t_rem["upper"][leaf][state_index]) - float(t_rem["lower"][leaf][state_index])
                    if x_width != t_width:
                        first_remainder = {
                            "segment_index": segment,
                            "physical_time": time,
                            "leaf_id": leaf,
                            "state": state,
                            "torch_width": t_width,
                            "xiangru_width": x_width,
                            "absolute_width_difference": abs(t_width - x_width),
                        }
                        break
                if first_remainder is not None:
                    break
        compared_segments += 1

    xiangru_summary = json.loads(args.xiangru_summary.read_text(encoding="utf-8"))
    torch_summary = json.loads(args.torch_summary.read_text(encoding="utf-8"))
    horizons = []
    for tool, summary in (("xiangru", xiangru_summary), ("torch", torch_summary)):
        horizons.append({
            "tool": tool,
            "status": summary["status"],
            "completed_segments": summary["completed_segments"],
            "certified_horizon": summary["certified_horizon"],
            "first_failure": summary["first_failure"],
        })
    if compared_segments != 200 or any(row["status"] != "VERIFIED" or row["certified_horizon"] != 20.0 for row in horizons):
        target_horizon_ratio = "N/A"
    else:
        target_horizon_ratio = "available_in_width_over_time_rows_at_physical_time_20.0"
    result = {
        "schema": "tora_q3_common_control_comparison_v2",
        "lane": "common_control_plant_replay",
        "status": "FORMALLY_ALIGNED" if compared_segments == 200 else "INCOMPLETE",
        "period_local_observation_restart": True,
        "not_independent_closed_loop": True,
        "compared_segments": compared_segments,
        "aligned_scalar_enclosures": compared_segments * 48 * len(STATES) * len(KINDS),
        "ratio_definition": "Torch interval width divided by Xiangru interval width; zero Xiangru width is N/A.",
        "target_horizon_ratio": target_horizon_ratio,
        "contract": contract,
        "horizons": horizons,
        "first_divergence": {
            "status": first_status,
            "endpoint": first_endpoint,
            "tube": first_tube,
            "remainder_width": first_remainder,
            "coefficient": "unavailable_without_normalization_bijection",
        },
        "tightness_summary": {
            "target_horizon_available": target_horizon_ratio != "N/A",
            "interpretation": (
                "ratios below one mean Torch is tighter; ratios above one mean "
                "Xiangru is tighter"
            ),
            "time_state_kind_rows_torch_tighter_by_median": sum(
                isinstance(row["torch_over_xiangru_ratio_median"], float)
                and row["torch_over_xiangru_ratio_median"] < 1.0
                for row in width_rows
            ),
            "time_state_kind_rows_xiangru_tighter_by_median": sum(
                isinstance(row["torch_over_xiangru_ratio_median"], float)
                and row["torch_over_xiangru_ratio_median"] > 1.0
                for row in width_rows
            ),
            "time_state_kind_rows_equal_by_median": sum(
                row["torch_over_xiangru_ratio_median"] == 1.0
                for row in width_rows
            ),
        },
        "source_hashes": {
            "controller_trace": xiangru_header["controller_trace_sha256"],
            "xiangru_segments": sha256(args.xiangru_jsonl),
            "torch_segments": sha256(args.torch_jsonl),
            "xiangru_summary": sha256(args.xiangru_summary),
            "torch_summary": sha256(args.torch_summary),
        },
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "width_over_time.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(width_rows[0]))
        writer.writeheader()
        writer.writerows(width_rows)
    for kind in KINDS:
        selected_rows = [
            row for row in width_rows if row["enclosure_kind"] == kind
        ]
        with (output / f"{kind}_width_over_time.csv").open(
            "x", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
            writer.writeheader()
            writer.writerows(selected_rows)
    with (output / "property_margin_over_time.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(property_rows[0]))
        writer.writeheader()
        writer.writerows(property_rows)
    with (output / "selected_leaf_overlays.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overlay_rows[0]))
        writer.writeheader()
        writer.writerows(overlay_rows)
    with (output / "target_horizon_ratios.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(target_horizon_rows[0])
        )
        writer.writeheader()
        writer.writerows(target_horizon_rows)
    extreme_rows = [ratio_extremes[key] for key in sorted(ratio_extremes)]
    with (output / "worst_leaf_cases.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(extreme_rows[0]))
        writer.writeheader()
        writer.writerows(extreme_rows)
    first_segment = min(
        detail["segment_index"]
        for detail in (first_endpoint, first_tube, first_remainder)
        if isinstance(detail, dict)
    )
    divergence_window = {
        "schema": "tora_q3_first_divergence_window_v1",
        "first_segment": first_segment,
        "window_segments": list(range(max(1, first_segment - 1), min(200, first_segment + 2) + 1)),
        "rows": [
            row
            for row in width_rows
            if max(1, first_segment - 1)
            <= int(row["segment_index"])
            <= min(200, first_segment + 2)
        ],
    }
    (output / "first_divergence_window.json").write_text(
        json.dumps(divergence_window, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "failure_horizons.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(horizons[0]))
        writer.writeheader()
        writer.writerows(horizons)
    print(json.dumps({"status": result["status"], "compared_segments": compared_segments, "first_divergence": result["first_divergence"]}))
    return 0 if result["status"] == "FORMALLY_ALIGNED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
