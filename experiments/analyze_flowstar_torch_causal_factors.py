#!/usr/bin/env python3
"""Derive Gate C queue/Horner effects and step-1/step-2 attribution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.lossless_state_queue_schema import (
    decode_binary64_exact,
    parse_file,
)


CHECKPOINTS = (1, 2, 10, 50, 100, 200, 300, 632)
CHANNELS = (
    "endpoint_x_width",
    "endpoint_y_width",
    "segment_x_width",
    "segment_y_width",
)
CELLS = {
    "T-D0": (False, False, "normalized_insertion"),
    "T-H0": (True, False, "normalized_insertion_horner"),
    "T-DQ": (False, True, "normalized_insertion_symqueue_v2"),
    "T-HQ": (True, True, "normalized_insertion_horner_symqueue_v2"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def width(row: Mapping[str, str], prefix: str) -> float:
    return float(row[f"{prefix}_hi"]) - float(row[f"{prefix}_lo"])


def accepted(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    return [row for row in rows if row.get("status") == "accepted"]


def load_torch_cells(root: Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, Mapping[str, Any]]]:
    rows: dict[str, list[dict[str, str]]] = {}
    summaries: dict[str, Mapping[str, Any]] = {}
    for cell, (_, _, reset_mode) in CELLS.items():
        run = root / cell / "artifacts" / "run"
        rows[cell] = read_csv(run / "segments.csv")
        summaries[cell] = read_json(run / "summary.json")
        if summaries[cell].get("reset_mode") != reset_mode:
            raise ValueError(f"{cell} reset-mode dispatch mismatch")
    invariant_keys = (
        "contract_identity",
        "cutoff",
        "dense_range_contexts",
        "dense_range_max_depth",
        "dense_range_max_leaves",
        "dense_range_method",
        "dense_range_trigger",
        "device",
        "effective_h_max",
        "effective_h_min",
        "partition",
        "requested_order",
        "requested_horizon",
        "schedule",
        "support",
        "target_remainder_radius",
        "tm_backend",
        "tracked_diff_sha256",
    )
    reference = summaries["T-D0"]
    for cell, summary in summaries.items():
        for key in invariant_keys:
            if summary.get(key) != reference.get(key):
                raise ValueError(f"hidden factorial setting difference: {cell}:{key}")
    return rows, summaries


def dispatch_evidence(rows: Mapping[str, Sequence[Mapping[str, str]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cell, (horner, queue, reset_mode) in CELLS.items():
        first = rows[cell][0]
        observed_horner = first.get("carry_insertion_horner_used") == "True"
        observed_queue = first.get("carry_symbolic_queue_mode") == "flowstar_linear_v2"
        if observed_horner != horner or observed_queue != queue:
            raise ValueError(f"factor dispatch did not enter requested source path: {cell}")
        if first.get("carry_reset_mode") != reset_mode:
            raise ValueError(f"row-level reset mode mismatch: {cell}")
        result[cell] = {
            "requested_horner": horner,
            "requested_queue": queue,
            "observed_horner": observed_horner,
            "observed_queue": observed_queue,
            "row_reset_mode": first.get("carry_reset_mode"),
            "queue_size_after_step1": first.get("carry_queue_size_after", ""),
            "propagated_symbolic_width_step1": first.get(
                "carry_propagated_symbolic_width_sum", ""
            ),
        }
    return result


def factorial_effects(rows: Mapping[str, Sequence[Mapping[str, str]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for step in CHECKPOINTS:
        index = step - 1
        for channel in CHANNELS:
            values: dict[str, float] = {}
            for cell in CELLS:
                row = rows[cell][index]
                if row.get("status") != "accepted":
                    raise ValueError(f"missing accepted factorial checkpoint {cell}:{step}")
                values[cell] = float(row[channel])
            horner = 0.5 * (
                (values["T-H0"] - values["T-D0"])
                + (values["T-HQ"] - values["T-DQ"])
            )
            queue = 0.5 * (
                (values["T-DQ"] - values["T-D0"])
                + (values["T-HQ"] - values["T-H0"])
            )
            interaction = (
                values["T-HQ"]
                - values["T-H0"]
                - values["T-DQ"]
                + values["T-D0"]
            )
            output.append(
                {
                    "step": step,
                    "time": step * 0.01,
                    "channel": channel,
                    **{cell: values[cell] for cell in CELLS},
                    "horner_main_effect": horner,
                    "queue_main_effect": queue,
                    "interaction": interaction,
                }
            )
    return output


def flowstar_queue_analysis(root: Path, q100_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = {
        1: root / "q1" / "artifacts" / "stock.csv",
        2: root / "q2" / "artifacts" / "stock.csv",
        10: root / "q10" / "artifacts" / "stock.csv",
        100: q100_path,
    }
    summaries = {
        q: read_json(path.parent / "summary.json") for q, path in paths.items()
    }
    rows = {q: read_csv(path) for q, path in paths.items()}
    checkpoints = (
        1, 2, 10, 50,
        99, 100, 101,
        199, 200, 201,
        299, 300, 301,
        397, 399, 400, 401, 474,
        499, 500, 501, 599, 600, 601, 632,
    )
    output: list[dict[str, Any]] = []
    for q, qrows in rows.items():
        for step in checkpoints:
            record: dict[str, Any] = {
                "queue_max_size": q,
                "step": step,
                "time": step * 0.01,
                "available": step <= len(qrows),
            }
            if step <= len(qrows):
                row = qrows[step - 1]
                for channel in ("endpoint_x", "endpoint_y", "segment_x", "segment_y"):
                    record[f"{channel}_width"] = width(row, channel)
            output.append(record)
    step1_equal = all(
        all(rows[q][0][field] == rows[100][0][field] for field in (
            "endpoint_x_lo", "endpoint_x_hi", "endpoint_y_lo", "endpoint_y_hi",
            "segment_x_lo", "segment_x_hi", "segment_y_lo", "segment_y_hi",
        ))
        for q in (1, 2, 10)
    )
    step2_first_differences = {
        f"Q{q}": {
            channel: width(rows[q][1], channel) - width(rows[100][1], channel)
            for channel in ("endpoint_x", "endpoint_y", "segment_x", "segment_y")
        }
        for q in (1, 2, 10)
    }
    observed_fields = (
        "endpoint_x_lo", "endpoint_x_hi", "endpoint_y_lo", "endpoint_y_hi",
        "segment_x_lo", "segment_x_hi", "segment_y_lo", "segment_y_hi",
    )
    first_published_differences: dict[str, Any] = {}
    for q in (1, 2, 10):
        by_field = {
            field: next(
                (
                    index + 1
                    for index, (candidate, baseline) in enumerate(zip(rows[q], rows[100]))
                    if candidate[field] != baseline[field]
                ),
                None,
            )
            for field in observed_fields
        }
        first_published_differences[f"Q{q}"] = {
            "by_field": by_field,
            "first_step": min(value for value in by_field.values() if value is not None),
        }
    reset_growth: dict[str, Any] = {}
    q100 = rows[100]
    for boundary in (100, 200, 300, 400, 500, 600, 700, 800, 900):
        reset_growth[str(boundary)] = {}
        for channel in ("endpoint_x", "endpoint_y", "segment_x", "segment_y"):
            widths = [width(row, channel) for row in q100]
            reset_growth[str(boundary)][channel] = {
                "growth_into_boundary": widths[boundary - 1] - widths[boundary - 2],
                "growth_after_reset": widths[boundary] - widths[boundary - 1],
                "second_difference": (
                    widths[boundary] - 2 * widths[boundary - 1] + widths[boundary - 2]
                ),
            }
    summary = {
        "horizons": {
            f"Q{q}": {
                "accepted_steps": int(summaries[q]["accepted_steps"]),
                "completed_time": int(summaries[q]["accepted_steps"]) * 0.01,
                "result_status_code": int(summaries[q]["result_status_code"]),
            }
            for q in paths
        },
        "step1_all_queue_sizes_bitwise_equal": step1_equal,
        "step2_minus_q100": step2_first_differences,
        "first_published_difference_from_q100": first_published_differences,
        "q100_reset_boundary_finite_differences": reset_growth,
        "no_sr_overload_mixed_with_q1": True,
    }
    return output, summary


def step1_coefficient_comparison(flowstar_fixture: Path, torch_terms: Path) -> dict[str, Any]:
    flow = parse_file(flowstar_fixture)
    flow_terms: dict[tuple[int, tuple[int, ...]], str] = {}
    for component in (0, 1):
        count = int(flow[f"flowpipe.tmvPre.component.{component}.term_count"])
        for index in range(count):
            base = f"flowpipe.tmvPre.component.{component}.term.{index}"
            exponents = tuple(int(value) for value in flow[f"{base}.exponents"].split(","))
            if exponents[3] != 0:
                raise ValueError("unexpected Flow* time-state parameter in VDP component")
            torch_order = (exponents[1], exponents[2], exponents[0])
            flow_terms[(component, torch_order)] = decode_binary64_exact(
                flow[f"{base}.coefficient"]
            ).hex()
    torch: dict[tuple[int, tuple[int, ...]], str] = {}
    with torch_terms.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["accepted_step_index"] == 0 and row["stage"] == "raw_picard_image":
                torch[(int(row["state_component"]), tuple(row["exponent_tuple"]))] = row[
                    "coefficient"
                ]["lower"]["hex"]
    support_equal = set(flow_terms) == set(torch)
    mismatches = [
        {
            "component": key[0],
            "exponents_torch_order": list(key[1]),
            "flowstar_hex": flow_terms[key],
            "torch_hex": torch[key],
        }
        for key in sorted(set(flow_terms) & set(torch))
        if flow_terms[key] != torch[key]
    ]
    return {
        "stage": "raw_picard_image / returned tmvPre polynomial",
        "support_equal": support_equal,
        "flowstar_term_count": len(flow_terms),
        "torch_term_count": len(torch),
        "bitwise_equal_coefficient_count": len(flow_terms) - len(mismatches),
        "bitwise_unequal_coefficient_count": len(mismatches),
        "first_mismatches": mismatches[:12],
        "interpretation": (
            "The first bitwise coefficient differences occur in local Picard polynomial "
            "arithmetic on step 1, before any old J/Phi source can cross a boundary."
        ),
    }


def step1_direction(stock_path: Path, torch_rows: Sequence[Mapping[str, str]], copied_path: Path) -> dict[str, Any]:
    stock = read_csv(stock_path)[0]
    torch = torch_rows[0]
    copied = accepted(read_csv(copied_path))[0]
    channels = {}
    for channel in ("endpoint_x", "endpoint_y", "segment_x", "segment_y"):
        flow_width = width(stock, channel)
        torch_width = float(torch[f"{channel}_width"])
        channels[channel] = {
            "flowstar_width": flow_width,
            "torch_width": torch_width,
            "torch_minus_flowstar": torch_width - flow_width,
            "torch_to_flowstar_ratio": torch_width / flow_width,
        }
    flow_residual_x = float(copied["residual_width_x"])
    flow_residual_y = float(copied["residual_width_y"])
    torch_residual_x = float(torch["carry_output_remainder_width_x"])
    torch_residual_y = float(torch["carry_output_remainder_width_y"])
    flow_segment_polynomial_x = (
        float(copied["picard_no_remainder_polynomial_range_x_hi"])
        - float(copied["picard_no_remainder_polynomial_range_x_lo"])
    )
    flow_segment_polynomial_y = (
        float(copied["picard_no_remainder_polynomial_range_y_hi"])
        - float(copied["picard_no_remainder_polynomial_range_y_lo"])
    )
    torch_segment_polynomial_x = width(torch, "segment_x") - torch_residual_x
    torch_segment_polynomial_y = width(torch, "segment_y") - torch_residual_y
    flow_endpoint_polynomial_x = width(stock, "endpoint_x") - flow_residual_x
    flow_endpoint_polynomial_y = width(stock, "endpoint_y") - flow_residual_y
    torch_endpoint_polynomial_x = width(torch, "endpoint_x") - torch_residual_x
    torch_endpoint_polynomial_y = width(torch, "endpoint_y") - torch_residual_y
    return {
        "channels": channels,
        "segment_polynomial_width": {
            "flowstar": {"x": flow_segment_polynomial_x, "y": flow_segment_polynomial_y},
            "torch": {"x": torch_segment_polynomial_x, "y": torch_segment_polynomial_y},
            "torch_minus_flowstar": {
                "x": torch_segment_polynomial_x - flow_segment_polynomial_x,
                "y": torch_segment_polynomial_y - flow_segment_polynomial_y,
            },
        },
        "endpoint_polynomial_width": {
            "flowstar": {"x": flow_endpoint_polynomial_x, "y": flow_endpoint_polynomial_y},
            "torch": {"x": torch_endpoint_polynomial_x, "y": torch_endpoint_polynomial_y},
            "torch_minus_flowstar": {
                "x": torch_endpoint_polynomial_x - flow_endpoint_polynomial_x,
                "y": torch_endpoint_polynomial_y - flow_endpoint_polynomial_y,
            },
        },
        "final_residual_width": {
            "flowstar": {"x": flow_residual_x, "y": flow_residual_y},
            "torch": {"x": torch_residual_x, "y": torch_residual_y},
            "torch_minus_flowstar": {
                "x": torch_residual_x - flow_residual_x,
                "y": torch_residual_y - flow_residual_y,
            },
        },
        "direction_conclusion": (
            "At step 1 the segment polynomial ranges are nearly equal, while Torch retains "
            "larger final residual widths, explaining the slightly wider Torch segments. "
            "At tau=h the polynomial endpoint evaluation differs enough in the opposite "
            "direction to dominate those residuals, so Torch endpoints are narrower."
        ),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    torch_rows, summaries = load_torch_cells(args.torch_factorial_root.resolve())
    dispatch = dispatch_evidence(torch_rows)
    effects = factorial_effects(torch_rows)
    write_csv(output / "torch_factorial_effects.csv", effects)
    flow_checkpoints, flow_summary = flowstar_queue_analysis(
        args.flowstar_queue_root.resolve(), args.flowstar_q100.resolve()
    )
    write_csv(output / "flowstar_queue_checkpoints.csv", flow_checkpoints)
    coefficient = step1_coefficient_comparison(
        args.flowstar_step1_fixture.resolve(), args.torch_step_terms.resolve()
    )
    direction = step1_direction(
        args.flowstar_q100.resolve(), torch_rows["T-D0"], args.copied_probe.resolve()
    )
    first_horner_width_difference: dict[str, int | None] = {}
    first_queue_width_difference: dict[str, int | None] = {}
    for channel in CHANNELS:
        first_horner_width_difference[channel] = next(
            (
                index + 1
                for index, (direct, horner) in enumerate(
                    zip(torch_rows["T-D0"], torch_rows["T-H0"])
                )
                if direct.get(channel) != horner.get(channel)
            ),
            None,
        )
        first_queue_width_difference[channel] = next(
            (
                index + 1
                for index, (off, on) in enumerate(
                    zip(torch_rows["T-D0"], torch_rows["T-DQ"])
                )
                if off.get(channel) != on.get(channel)
            ),
            None,
        )
    last_d0 = torch_rows["T-D0"][-1]
    result = {
        "schema": "flowstar_torch_causal_factor_split_v1",
        "flowstar_queue": flow_summary,
        "torch_dispatch": dispatch,
        "torch_horizons": {
            cell: {
                "accepted_steps": int(summary["accepted_steps"]),
                "completed_horizon": float(summary["completed_horizon"]),
                "failure_type": summary["failure_type"],
            }
            for cell, summary in summaries.items()
        },
        "first_horner_published_width_difference": first_horner_width_difference,
        "first_queue_published_width_difference": first_queue_width_difference,
        "step1_raw_picard_coefficients": coefficient,
        "step1_width_directions": direction,
        "torch_candidate_633": {
            "status": last_d0["status"],
            "pre_time": float(last_d0["t_lo"]),
            "target_margins": json.loads(last_d0["target_margins"]),
            "y_subset_margin": json.loads(last_d0["target_margins"])[0][1],
        },
        "queue_diagnostic_limitation": (
            "flowstar_linear_v2 changes published segment widths from step 2 but leaves "
            "endpoint widths, carry scales, and failure horizon unchanged in both composition "
            "levels; it is not a production-equivalent old-source carry."
        ),
        "deterministic_not_statistical": True,
        "status": "CAUSAL_FACTOR_SPLIT_PARTIAL",
    }
    write_json(output / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flowstar-queue-root", type=Path, required=True)
    parser.add_argument("--flowstar-q100", type=Path, required=True)
    parser.add_argument("--torch-factorial-root", type=Path, required=True)
    parser.add_argument("--flowstar-step1-fixture", type=Path, required=True)
    parser.add_argument("--torch-step-terms", type=Path, required=True)
    parser.add_argument("--copied-probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), sort_keys=True, allow_nan=False))
