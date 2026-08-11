#!/usr/bin/env python3
"""Build the five causal figures and their source CSVs from runner artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _svg(path: Path, width: int, height: int, body: Sequence[str]) -> None:
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#202124}.title{font-size:18px;font-weight:600}.axis{font-size:11px}.label{font-size:12px}.small{font-size:10px}.grid{stroke:#d8dce2;stroke-width:1}.zero{stroke:#30343b;stroke-width:1.4}</style>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def _text(x: float, y: float, value: Any, css: str = "label", anchor: str = "start") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" class="{css}" text-anchor="{anchor}">{escape(str(value))}</text>'


def _raw_waterfall(run_root: Path, output: Path) -> tuple[Path, Path]:
    source = (
        run_root
        / "07_flowstar_torch_raw_remainder/independent_analysis/artifacts/run"
        / "raw_remainder_node_comparison.csv"
    )
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    selected = [row for row in rows if int(row["suboperation_order"]) <= 8]
    for row in selected:
        row.update(
            {
                "units": "interval_width",
                "eligibility": "same_frozen_prestate_diagnostic",
                "sample_count": 1,
            }
        )
    csv_path = output / "raw_remainder_expression_width_waterfall.csv"
    fields = list(selected[0]) if selected else []
    _write_csv(csv_path, fields, selected)

    width, left, right, top, row_h = 1080, 330, 1025, 66, 35
    height = top + row_h * len(selected) + 65
    values = [float(row["width_delta_flowstar_minus_torch"]) for row in selected]
    maximum = max((abs(value) for value in values), default=1.0) or 1.0
    zero = (left + right) / 2
    scale = (right - left) / (2 * maximum)
    body = [
        _text(20, 30, "First-split raw-remainder expression width waterfall", "title"),
        _text(20, 50, "Flow* width minus Torch width; interval-width units; n=1 frozen prestate", "axis"),
        f'<line x1="{zero:.2f}" y1="{top - 18}" x2="{zero:.2f}" y2="{height - 35}" class="zero"/>',
    ]
    for index, (row, value) in enumerate(zip(selected, values)):
        y = top + index * row_h
        name = f'{row["suboperation_order"]}. {row["semantic_node"]}/{row["operation"]}'
        body.append(_text(left - 12, y + 15, name, "small", "end"))
        bar_x = min(zero, zero + value * scale)
        bar_width = max(abs(value * scale), 1.0)
        color = "#d95f02" if value >= 0 else "#1b9e77"
        body.append(f'<rect x="{bar_x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="20" fill="{color}"/>')
        body.append(_text(zero + value * scale + (6 if value >= 0 else -6), y + 15, f"{value:.6g}", "small", "start" if value >= 0 else "end"))
    svg_path = output / "raw_remainder_expression_width_waterfall.svg"
    _svg(svg_path, width, height, body)
    return csv_path, svg_path


def _validator_matrix(run_root: Path, output: Path) -> tuple[Path, Path]:
    source = (
        run_root
        / "08_schedule_validator_matrix/adaptive_schedule/artifacts/run"
        / "schedule_validator_matrix.json"
    )
    matrix = _json(source)
    checkpoint = next(
        row
        for row in matrix["checkpoints"]
        if row["checkpoint"] == "last_common_prestate_before_first_split"
    )
    rows: list[dict[str, Any]] = []
    for candidate in checkpoint["rows"]:
        for validator in ("flowstar_validator", "torch_validator"):
            result = candidate[validator]
            rows.append(
                {
                    "candidate_producer": candidate["candidate_producer"],
                    "receiving_validator": validator.removesuffix("_validator"),
                    "minimum_margin": min(float(value) for value in result["margins"]),
                    "decision": result["decision"],
                    "units": "target_radius_margin",
                    "eligibility": "same_prestate_same_candidate_componentwise_subset",
                    "sample_count": 1,
                    "t_pre": checkpoint["t_pre"],
                    "h": checkpoint["h"],
                }
            )
    csv_path = output / "same_prestate_validator_margin_matrix.csv"
    _write_csv(csv_path, tuple(rows[0]), rows)
    producers = ["torch_complete_o4", "flowstar_complete_o4"]
    validators = ["flowstar", "torch"]
    body = [
        _text(20, 30, "Same-prestate 2×2 validator margin matrix", "title"),
        _text(20, 50, "minimum target-radius margin; n=1 candidate per producer", "axis"),
    ]
    x0, y0, cell_w, cell_h = 330, 100, 260, 100
    for column, validator in enumerate(validators):
        body.append(_text(x0 + column * cell_w + cell_w / 2, y0 - 18, f"receiver: {validator}", "label", "middle"))
    for row_index, producer in enumerate(producers):
        body.append(_text(x0 - 18, y0 + row_index * cell_h + cell_h / 2, f"producer: {producer}", "label", "end"))
        for column, validator in enumerate(validators):
            record = next(row for row in rows if row["candidate_producer"] == producer and row["receiving_validator"] == validator)
            x, y = x0 + column * cell_w, y0 + row_index * cell_h
            color = "#c7e9c0" if record["decision"] == "accept" else "#fcbba1"
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w - 8}" height="{cell_h - 8}" fill="{color}" stroke="#666"/>')
            body.append(_text(x + (cell_w - 8) / 2, y + 38, record["decision"], "label", "middle"))
            body.append(_text(x + (cell_w - 8) / 2, y + 62, f'{record["minimum_margin"]:.8g}', "small", "middle"))
    svg_path = output / "same_prestate_validator_margin_matrix.svg"
    _svg(svg_path, 900, 345, body)
    return csv_path, svg_path


def _bridge_changes(run_root: Path, output: Path) -> tuple[Path, Path]:
    source = run_root / "10_bridge_ladder/G3/artifacts/run/bridge_ladder.json"
    ladder = _json(source)
    rows = []
    for item in ladder["adjacent_factor_attribution"]:
        row = dict(item)
        row.update(
            {
                "units": "margin_or_interval_width_delta",
                "eligibility": item["comparison_eligibility"],
                "sample_count": 1,
            }
        )
        rows.append(row)
    csv_path = output / "bridge_per_factor_margin_width_changes.csv"
    fields = list(rows[0]) if rows else []
    _write_csv(csv_path, fields, rows)

    factors = ["support", "picard", "validator", "carry"]
    batches = [1, 64]
    colors = {1: "#377eb8", 64: "#e41a1c"}
    panels = [
        ("margin_delta", "T1 minimum-margin delta"),
        ("t1_max_endpoint_width_delta", "T1 max endpoint-width delta"),
    ]
    body = [
        _text(20, 30, "Bridge A0→A4 per-factor changes", "title"),
        _text(20, 50, "same B/h/T1/output/success; ordinary-float64 diagnostic", "axis"),
    ]
    for panel_index, (field, title) in enumerate(panels):
        panel_y = 82 + panel_index * 250
        left, right, zero = 330, 1040, 685
        panel_values = [float(row[field]) for row in rows]
        maximum = max((abs(value) for value in panel_values), default=1.0) or 1.0
        scale = (right - left) / (2 * maximum)
        body.append(_text(20, panel_y + 14, title, "label"))
        body.append(f'<line x1="{zero}" y1="{panel_y + 23}" x2="{zero}" y2="{panel_y + 210}" class="zero"/>')
        for factor_index, factor in enumerate(factors):
            y = panel_y + 42 + factor_index * 42
            body.append(_text(left - 16, y + 13, factor, "label", "end"))
            for batch_index, batch in enumerate(batches):
                record = next(
                    (
                        row
                        for row in rows
                        if row["changed_factor"] == factor
                        and int(row["batch"]) == batch
                    ),
                    None,
                )
                if record is None:
                    continue
                value = float(record[field])
                shifted_y = y + batch_index * 15
                x = min(zero, zero + value * scale)
                body.append(f'<rect x="{x:.2f}" y="{shifted_y:.2f}" width="{max(abs(value * scale), 1.0):.2f}" height="11" fill="{colors[batch]}"/>')
        body.append(_text(right - 100, panel_y + 14, "B1", "small"))
        body.append(f'<rect x="{right - 125}" y="{panel_y + 5}" width="18" height="10" fill="{colors[1]}"/>')
        body.append(_text(right - 45, panel_y + 14, "B64", "small"))
        body.append(f'<rect x="{right - 72}" y="{panel_y + 5}" width="18" height="10" fill="{colors[64]}"/>')
    svg_path = output / "bridge_per_factor_margin_width_changes.svg"
    _svg(svg_path, 1080, 590, body)
    return csv_path, svg_path


def _runtime_breakdown(run_root: Path, output: Path) -> tuple[Path, Path]:
    source = run_root / "10_bridge_ladder/G3/artifacts/run/bridge_ladder.json"
    ladder = _json(source)
    rows = []
    stages = ("polynomial_picard", "validation", "carry", "output_object")
    for cell in ladder["cells"]:
        for stage in stages:
            rows.append(
                {
                    "cell": cell["cell"],
                    "batch": cell["batch"],
                    "completed_steps": cell["completed_steps"],
                    "validated_horizon": cell["validated_horizon"],
                    "completed_requested_gate": cell["completed_requested_gate"],
                    "stage": stage,
                    "seconds": cell["runtime_by_stage_seconds"][stage],
                    "units": "process_wall_seconds",
                    "eligibility": "same_runner_horizon_gated_diagnostic_no_cross_tool_ratio",
                    "sample_count": 1,
                }
            )
    csv_path = output / "horizon_gated_runtime_breakdown.csv"
    _write_csv(csv_path, tuple(rows[0]), rows)
    keys = [(cell["cell"], int(cell["batch"])) for cell in ladder["cells"]]
    totals = {key: sum(float(row["seconds"]) for row in rows if (row["cell"], int(row["batch"])) == key) for key in keys}
    maximum = max(totals.values(), default=1.0) or 1.0
    colors = {
        "polynomial_picard": "#4daf4a",
        "validation": "#377eb8",
        "carry": "#984ea3",
        "output_object": "#ff7f00",
    }
    body = [
        _text(20, 30, "Horizon-gated bridge runtime breakdown", "title"),
        _text(20, 50, "process-wall seconds by stage; diagnostic only; one run per row", "axis"),
    ]
    left, right, top, row_h = 165, 1040, 82, 34
    for index, key in enumerate(keys):
        y = top + index * row_h
        body.append(_text(left - 12, y + 14, f"{key[0]} / B{key[1]}", "label", "end"))
        x = left
        for stage in stages:
            seconds = next(float(row["seconds"]) for row in rows if row["cell"] == key[0] and int(row["batch"]) == key[1] and row["stage"] == stage)
            bar_width = seconds / maximum * (right - left)
            body.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="20" fill="{colors[stage]}"/>')
            x += bar_width
    legend_x = left
    legend_y = top + len(keys) * row_h + 25
    for stage in stages:
        body.append(f'<rect x="{legend_x}" y="{legend_y}" width="16" height="11" fill="{colors[stage]}"/>')
        body.append(_text(legend_x + 22, legend_y + 10, stage, "small"))
        legend_x += 205
    svg_path = output / "horizon_gated_runtime_breakdown.svg"
    _svg(svg_path, 1080, legend_y + 55, body)
    return csv_path, svg_path


def _first(mapping: Mapping[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _native_capability_table(run_root: Path, output: Path) -> tuple[Path, Path]:
    specifications = (
        ("Flow*", "complete-O4 native", "03_native_flowstar/official_vdp"),
        ("DiffReach", "fixed-DR7 native", "04_native_diffreach/official_vdp"),
        ("Torch", "complete-O4 native", "05_native_torch_complete_o4/authoritative"),
        ("Torch", "fixed-DR7 matched", "06_native_torch_fixed_dr7/t10_cpu"),
    )
    rows = []
    for tool, lane, relative in specifications:
        runner = run_root / relative
        config = _json(runner / "config.json")
        wrapper = _json(runner / "summary.json")
        artifact_path = runner / "artifacts/run/summary.json"
        artifact = _json(artifact_path) if artifact_path.is_file() else {}
        schedule = artifact.get("schedule", {})
        if isinstance(schedule, Mapping):
            schedule_label = schedule.get("kind", "")
            if schedule.get("h") is not None:
                schedule_label += f' h={schedule["h"]}'
        else:
            schedule_label = str(schedule)
        rows.append(
            {
                "tool": tool,
                "lane": lane,
                "representation": _first(artifact, "representation", "tm_backend"),
                "partition": _first(artifact, "partition_count", "batch"),
                "schedule": schedule_label,
                "requested_horizon": _first(artifact, "horizon_requested", "requested_horizon", "horizon"),
                "validated_horizon": _first(artifact, "horizon_validated", "validated_horizon", "completed_horizon"),
                "result_status": _first(artifact, "result_status", "status", "outcome", default=wrapper["status"]),
                "endpoint_available": _first(artifact, "endpoint_available"),
                "segment_tube_available": _first(artifact, "segment_tube_available"),
                "prefix_tube_available": _first(artifact, "prefix_tube_available"),
                "eligibility": config["eligibility_status"],
                "sample_count": 1,
            }
        )
    csv_path = output / "native_capability_eligibility_table.csv"
    _write_csv(csv_path, tuple(rows[0]), rows)
    columns = ("tool", "lane", "representation", "partition", "validated_horizon", "result_status", "eligibility")
    widths = (90, 165, 150, 75, 120, 120, 245)
    x0, y0, row_h = 20, 76, 43
    body = [
        _text(20, 30, "Native capability / eligibility table", "title"),
        _text(20, 50, "capability rows only; no transitive ranking or speedup", "axis"),
    ]
    x = x0
    for column, column_width in zip(columns, widths):
        body.append(f'<rect x="{x}" y="{y0}" width="{column_width}" height="{row_h}" fill="#d9e6f2" stroke="#888"/>')
        body.append(_text(x + 5, y0 + 26, column, "small"))
        x += column_width
    for row_index, row in enumerate(rows, start=1):
        x = x0
        y = y0 + row_index * row_h
        for column, column_width in zip(columns, widths):
            body.append(f'<rect x="{x}" y="{y}" width="{column_width}" height="{row_h}" fill="white" stroke="#aaa"/>')
            value = str(row[column])
            if len(value) > 34:
                value = value[:31] + "..."
            body.append(_text(x + 5, y + 26, value, "small"))
            x += column_width
    svg_path = output / "native_capability_eligibility_table.svg"
    _svg(svg_path, sum(widths) + 40, y0 + (len(rows) + 1) * row_h + 28, body)
    return csv_path, svg_path


def build(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    builders = (
        _raw_waterfall,
        _validator_matrix,
        _bridge_changes,
        _runtime_breakdown,
        _native_capability_table,
    )
    artifacts = []
    for builder in builders:
        csv_path, svg_path = builder(run_root, output)
        artifacts.append(
            {
                "source_csv": csv_path.name,
                "source_csv_sha256": _sha(csv_path),
                "figure": svg_path.name,
                "figure_sha256": _sha(svg_path),
            }
        )
    summary = {
        "schema": "three_tool_causal_figures_v1",
        "outcome": "CAUSAL_FIGURES_BUILT",
        "figure_count": len(artifacts),
        "artifacts": artifacts,
        "constraints": {
            "universal_three_tool_ranking": False,
            "incomplete_lane_speedup": False,
            "every_figure_has_source_csv": True,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(build(parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
