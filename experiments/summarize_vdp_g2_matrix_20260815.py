#!/usr/bin/env python3
"""Recompute the G2 production decision from raw fixed/native matrix files."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


CHANNEL_FIELDS = {
    "endpoint_x": ("endpoint_x_lo", "endpoint_x_hi", "endpoint_x_width"),
    "endpoint_y": ("endpoint_y_lo", "endpoint_y_hi", "endpoint_y_width"),
    "segment_tube_x": ("segment_x_lo", "segment_x_hi", "segment_x_width"),
    "segment_tube_y": ("segment_y_lo", "segment_y_hi", "segment_y_width"),
}
CHECKPOINTS = (1.0, 3.0, 6.32)
THRESHOLDS = (1.1, 1.5, 2.0, 5.0)
MODES = ("legacy", "g1", "g2")
LEGACY_NATIVE_REFERENCE = 6.397083942944808


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def raw_width(row: Mapping[str, str], channel: str) -> dict[str, float]:
    lo_key, hi_key, width_key = CHANNEL_FIELDS[channel]
    lo = float(row[lo_key])
    hi = float(row[hi_key])
    stored = float(row[width_key])
    recomputed = hi - lo
    if recomputed != stored:
        raise ValueError(f"stored width is not raw upper-lower for {channel}")
    return {"lo": lo, "hi": hi, "width": recomputed}


def deterministic_gzip_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed:
            import io

            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--flowstar-width-ledger", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.matrix_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    matrix = json.loads((root / "matrix.json").read_text(encoding="utf-8"))
    if matrix["request_count"] != 36:
        raise ValueError("fresh matrix must contain all 36 independent requests")
    request = {
        (row["schedule"], row["mode"], float(row["requested_horizon"])): row
        for row in matrix["rows"]
    }
    if len(request) != 36:
        raise ValueError("fresh request matrix has duplicate/missing keys")
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    oracle_passed = oracle.get("status") == "PASS" and oracle.get("implementation_independent") is True

    flowstar: dict[tuple[int, str], dict[str, float]] = {}
    with args.flowstar_width_ledger.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lo = float(row["flowstar_lo"])
            hi = float(row["flowstar_hi"])
            width = hi - lo
            if width != float(row["flowstar_width"]):
                raise ValueError("Flow* reference width is not raw upper-lower")
            flowstar[(int(row["step"]), row["channel"])] = {"lo": lo, "hi": hi, "width": width}
    if len(flowstar) != 632 * 4:
        raise ValueError("Flow* fixed reference does not cover 632 steps x four channels")

    mode_segments: dict[str, list[dict[str, str]]] = {}
    for mode in MODES:
        long_request = request[("fixed", mode, 6.32)]
        path = root / long_request["relative_output"] / "segments.csv.gz"
        rows = [row for row in read_csv_gz(path) if row["status"] == "accepted"]
        if len(rows) != 632:
            raise ValueError(f"{mode} long fixed trace is incomplete")
        mode_segments[mode] = rows

    curve_rows: list[dict[str, Any]] = []
    previous: dict[tuple[str, str], float] = {}
    for step in range(1, 633):
        for channel in CHANNEL_FIELDS:
            fs = flowstar[(step, channel)]
            record: dict[str, Any] = {
                "step": step,
                "time": step * 0.01,
                "time_hex": float(step * 0.01).hex(),
                "channel": channel,
                "flowstar_lo": fs["lo"],
                "flowstar_hi": fs["hi"],
                "flowstar_width": fs["width"],
            }
            for mode in MODES:
                values = raw_width(mode_segments[mode][step - 1], channel)
                prior = previous.get((mode, channel))
                record.update({
                    f"{mode}_lo": values["lo"],
                    f"{mode}_hi": values["hi"],
                    f"{mode}_width": values["width"],
                    f"{mode}_excess": values["width"] - fs["width"],
                    f"{mode}_ratio": values["width"] / fs["width"],
                    f"{mode}_width_increment": "" if prior is None else values["width"] - prior,
                })
                previous[(mode, channel)] = values["width"]
            record["g1_reduction_vs_legacy"] = record["legacy_width"] - record["g1_width"]
            record["g2_reduction_vs_legacy"] = record["legacy_width"] - record["g2_width"]
            record["g2_reduction_vs_g1"] = record["g1_width"] - record["g2_width"]
            curve_rows.append(record)
    deterministic_gzip_csv(output / "fixed_curve.csv.gz", curve_rows, list(curve_rows[0]))

    resource_rows: list[dict[str, Any]] = []
    for mode in MODES:
        for step, row in enumerate(mode_segments[mode], 1):
            resource_rows.append({
                "mode": mode,
                "step": step,
                "time": step * 0.01,
                "active_variables": row.get("next_boundary_active_variables", ""),
                "term_count": row.get("next_boundary_term_count", ""),
                "collapse_count": row.get("carry_g2_collapse_count", row.get("carry_source_ledger_collapse_count", "")),
                "ordinary_mass": row.get("carry_g2_ordinary_collapsed_width_mass", row.get("carry_source_ledger_ordinary_width_mass", "")),
                "retained_shared_source_mass": row.get("carry_g2_retained_shared_source_width_mass", ""),
                "fresh_source_mass": row.get("carry_g2_fresh_structured_width_mass", row.get("carry_source_ledger_structured_width_mass", "")),
                "fallback_count": row.get("sparse_fallback_count", ""),
                "stage_runtime_s": row.get("stage_runtime_s", ""),
            })
    deterministic_gzip_csv(output / "resource_curve.csv.gz", resource_rows, list(resource_rows[0]))

    selected: dict[str, Any] = {}
    for horizon in CHECKPOINTS:
        step = round(horizon / 0.01)
        checkpoint: dict[str, Any] = {}
        for channel in CHANNEL_FIELDS:
            row = next(item for item in curve_rows if item["step"] == step and item["channel"] == channel)
            legacy_excess = row["legacy_excess"]
            reduction = row["g2_reduction_vs_legacy"]
            checkpoint[channel] = {
                key: row[key]
                for key in (
                    "flowstar_lo", "flowstar_hi", "flowstar_width",
                    "legacy_lo", "legacy_hi", "legacy_width", "legacy_excess",
                    "g1_lo", "g1_hi", "g1_width", "g1_excess",
                    "g2_lo", "g2_hi", "g2_width", "g2_excess",
                    "g1_reduction_vs_legacy", "g2_reduction_vs_legacy", "g2_reduction_vs_g1",
                )
            }
            checkpoint[channel]["g2_fraction_of_legacy_excess_removed"] = (
                reduction / legacy_excess if legacy_excess > 0 else None
            )
        selected[format(horizon, "g")] = checkpoint

    crossings: list[dict[str, Any]] = []
    for mode in MODES:
        for channel in CHANNEL_FIELDS:
            rows = [row for row in curve_rows if row["channel"] == channel]
            for threshold in THRESHOLDS:
                match = next((row for row in rows if row[f"{mode}_ratio"] >= threshold), None)
                crossings.append({
                    "mode": mode,
                    "channel": channel,
                    "threshold": threshold,
                    "crossed": match is not None,
                    "step": None if match is None else match["step"],
                    "time": None if match is None else match["time"],
                    "ratio": None if match is None else match[f"{mode}_ratio"],
                })
    with (output / "ratio_crossings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(crossings[0]))
        writer.writeheader()
        writer.writerows(crossings)

    fixed_all_completed = all(
        request[("fixed", mode, horizon)]["completed_requested_horizon"]
        for mode in MODES
        for horizon in (0.1, 0.5, 1.0, 2.0, 3.0, 6.32)
    )
    t1_narrow = all(
        selected["1"][channel]["g2_width"] <= selected["1"][channel]["g1_width"]
        for channel in CHANNEL_FIELDS
    )
    excess_gate = all(
        selected[label][channel]["g2_fraction_of_legacy_excess_removed"] >= 0.10
        for label in ("3", "6.32")
        for channel in CHANNEL_FIELDS
    )
    native_rows = {
        mode: request[("native", mode, 10.0)]
        for mode in MODES
    }
    terminal_details: dict[str, Any] = {}
    for mode, row in native_rows.items():
        reference_path = root / row["relative_output"] / "terminal_checkpoint/terminal_reference.json"
        terminal_details[mode] = (
            json.loads(reference_path.read_text(encoding="utf-8"))
            if reference_path.is_file()
            else None
        )
    g2_native = float(native_rows["g2"]["completed_horizon"])
    native_gate = g2_native >= LEGACY_NATIVE_REFERENCE
    if native_rows["g2"]["completed_requested_horizon"]:
        terminal_margin_gate = True
    else:
        g2_terminal = terminal_details["g2"]
        legacy_terminal = terminal_details["legacy"]
        terminal_margin_gate = bool(
            g2_terminal is not None
            and legacy_terminal is not None
            and float(g2_terminal["subset_margin"][0][1])
            > float(legacy_terminal["subset_margin"][0][1])
        )
    no_failures = fixed_all_completed and all(
        request[("fixed", "g2", horizon)]["rejected_attempts"] == 0
        for horizon in (0.1, 0.5, 1.0, 2.0, 3.0, 6.32)
    )
    mechanism_improved = all(
        selected[label][channel]["g2_width"] <= selected[label][channel]["g1_width"]
        for label in ("1", "3", "6.32")
        for channel in CHANNEL_FIELDS
    ) and any(
        selected[label][channel]["g2_width"] < selected[label][channel]["g1_width"]
        for label in ("1", "3", "6.32")
        for channel in CHANNEL_FIELDS
    )
    production_success = all((
        oracle_passed,
        t1_narrow,
        excess_gate,
        no_failures,
        native_gate,
        terminal_margin_gate,
    ))
    if production_success and native_rows["g2"]["completed_requested_horizon"]:
        conclusion = "G2_VDP_T10_VALIDATED"
    elif mechanism_improved:
        conclusion = "G2_MECHANISM_IMPROVED__PRODUCTION_GATE_NOT_MET"
    else:
        conclusion = "G2_SHARED_COLUMN_CARRY_REJECTED"

    native_table = []
    for mode in MODES:
        for horizon in (1.0, 3.0, 6.0, 6.5, 7.5, 10.0):
            native_table.append(request[("native", mode, horizon)])
    summary = {
        "schema": "vdp_g2_shared_column_scientific_summary_v1",
        "conclusion": conclusion,
        "total_cause_conclusion": "LOSSLESS_CROSS_OPERATOR_CELL_UNAVAILABLE__TOTAL_CAUSE_OPEN",
        "fixed_checkpoints": selected,
        "ratio_crossings": crossings,
        "native_requests": native_table,
        "native_horizon_T10_requests": native_rows,
        "native_terminal_details": terminal_details,
        "gates": {
            "independent_oracle": oracle_passed,
            "fixed_T1_all_four_no_wider_than_G1": t1_narrow,
            "fixed_T3_T6p32_all_four_remove_at_least_10pct_legacy_excess": excess_gate,
            "all_fixed_requests_complete_without_G2_rejection": no_failures,
            "native_G2_at_least_legacy_6p397083942944808": native_gate,
            "terminal_y_subset_margin_better_than_legacy_if_failure_remains": terminal_margin_gate,
            "production_success": production_success,
        },
        "mechanism_improved_at_selected_fixed_checkpoints": mechanism_improved,
        "g2_native_horizon": g2_native,
        "legacy_native_reference": LEGACY_NATIVE_REFERENCE,
        "g2_reached_T10": native_rows["g2"]["completed_requested_horizon"],
        "unexplained_T1_T3_residual": "NOT_IDENTIFIABLE_WITHOUT_LOSSLESS_CROSS_OPERATOR_CELLS",
        "width_recomputation": "every stored width above and in fixed_curve.csv.gz was recomputed as raw_hi-raw_lo",
    }
    (output / "scientific_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"conclusion": conclusion, "gates": summary["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
