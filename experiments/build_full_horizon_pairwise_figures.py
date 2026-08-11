#!/usr/bin/env python3
"""Build the five preregistered causal figures, each with its own source CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


COLORS = ("#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#a65628")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty source rows for {path.name}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _svg(path: Path, width: int, height: int, body: Iterable[str]) -> None:
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#202124}.title{font-size:18px;font-weight:600}.label{font-size:12px}.tick{font-size:10px}.grid{stroke:#d8dce2;stroke-width:1}.axis{stroke:#30343b;stroke-width:1.2}.line{fill:none;stroke-width:1.7}</style>',
                *body,
                "</svg>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _text(x: float, y: float, value: Any, css: str = "label", anchor: str = "start") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" class="{css}" text-anchor="{anchor}">{escape(str(value))}</text>'


def _finite(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        raw = row.get(field, "")
        if raw not in (None, "") and math.isfinite(value := float(raw)):
            values.append(value)
    return values


def _line_panels(
    path: Path,
    *,
    title: str,
    subtitle: str,
    rows: Sequence[Mapping[str, Any]],
    panels: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    x_field: str = "time",
) -> None:
    width, height = 1120, 75 + 225 * len(panels) + 45
    left, right = 85.0, 1080.0
    body = [_text(18, 28, title, "title"), _text(18, 48, subtitle, "tick")]
    x_values = _finite(rows, x_field)
    x_min, x_max = min(x_values), max(x_values)
    if x_max == x_min:
        x_max = x_min + 1.0
    for panel_index, (panel_title, series) in enumerate(panels):
        top = 75.0 + panel_index * 225.0
        bottom = top + 165.0
        all_y = [value for field, _ in series for value in _finite(rows, field)]
        y_min = min(0.0, min(all_y))
        y_max = max(0.0, max(all_y))
        if y_max == y_min:
            y_max = y_min + 1.0
        body.extend(
            [
                _text(18, top + 14, panel_title, "label"),
                f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>',
                _text(left, bottom + 16, f"{x_min:.4g}", "tick", "middle"),
                _text(right, bottom + 16, f"{x_max:.4g}", "tick", "middle"),
                _text(left - 8, top + 5, f"{y_max:.4g}", "tick", "end"),
                _text(left - 8, bottom, f"{y_min:.4g}", "tick", "end"),
            ]
        )
        for series_index, (field, label) in enumerate(series):
            points = []
            for row in rows:
                if row.get(field, "") in (None, ""):
                    continue
                x, y = float(row[x_field]), float(row[field])
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                px = left + (x - x_min) / (x_max - x_min) * (right - left)
                py = bottom - (y - y_min) / (y_max - y_min) * (bottom - top)
                points.append(f"{px:.2f},{py:.2f}")
            color = COLORS[series_index % len(COLORS)]
            if points:
                body.append(f'<polyline points="{" ".join(points)}" class="line" stroke="{color}"/>')
            lx = right - 145 * (len(series) - series_index)
            body.append(f'<line x1="{lx}" y1="{top + 12}" x2="{lx + 20}" y2="{top + 12}" stroke="{color}" stroke-width="2"/>')
            body.append(_text(lx + 25, top + 16, label, "tick"))
    body.append(_text((left + right) / 2, height - 14, x_field, "label", "middle"))
    _svg(path, width, height, body)


def _with_common(rows: Sequence[Mapping[str, str]], *, schema: str, eligibility: str) -> list[dict[str, Any]]:
    return [
        {
            "schema": schema,
            **row,
            "eligibility": eligibility,
            "sample_count": 1,
        }
        for row in rows
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    artifacts: list[dict[str, Any]] = []

    common_raw = _read_csv(args.flow_common_prefix)
    if not common_raw or any(row["both_completed"] != "True" for row in common_raw):
        raise RuntimeError("Flow*/Torch figure source extends beyond the completed common prefix")
    width_fields = (
        "step", "time", "time_hex", "both_completed", "qualification",
        "flowstar_endpoint_x_width", "torch_endpoint_x_width",
        "flowstar_endpoint_y_width", "torch_endpoint_y_width",
        "flowstar_segment_tube_x_width", "torch_segment_tube_x_width",
        "flowstar_segment_tube_y_width", "torch_segment_tube_y_width",
    )
    width_rows = _with_common(
        [{field: row[field] for field in width_fields} for row in common_raw],
        schema="flowstar_torch_fixed_schedule_width_figure_row_v1",
        eligibility="common_prefix_only_flowstar_build_qualification_open",
    )
    width_csv = output / "flowstar_torch_endpoint_tube_widths.csv"
    width_svg = output / "flowstar_torch_endpoint_tube_widths.svg"
    _write_csv(width_csv, width_rows)
    _line_panels(
        width_svg,
        title="Flow*/Torch fixed-schedule common-prefix widths",
        subtitle="h=0.01, B1, complete-O4; trace stops at Torch's last accepted step 632",
        rows=width_rows,
        panels=(
            ("endpoint x width", (("flowstar_endpoint_x_width", "Flow*"), ("torch_endpoint_x_width", "Torch"))),
            ("endpoint y width", (("flowstar_endpoint_y_width", "Flow*"), ("torch_endpoint_y_width", "Torch"))),
            ("segment tube x width", (("flowstar_segment_tube_x_width", "Flow*"), ("torch_segment_tube_x_width", "Torch"))),
            ("segment tube y width", (("flowstar_segment_tube_y_width", "Flow*"), ("torch_segment_tube_y_width", "Torch"))),
        ),
    )
    artifacts.append({"figure": width_svg.name, "source_csv": width_csv.name})

    margin_fields = ("step", "time", "time_hex", "both_completed", "qualification", "flowstar_margin_y", "torch_margin_y")
    margin_rows = _with_common(
        [{field: row[field] for field in margin_fields} for row in common_raw],
        schema="flowstar_torch_fixed_schedule_y_margin_figure_row_v1",
        eligibility="common_prefix_only_flowstar_build_qualification_open",
    )
    margin_csv = output / "flowstar_torch_y_margin.csv"
    margin_svg = output / "flowstar_torch_y_margin.svg"
    _write_csv(margin_csv, margin_rows)
    _line_panels(
        margin_svg,
        title="Flow*/Torch fixed-schedule y-margin",
        subtitle="target-radius margin; negative rejected-candidate values are not appended to the accepted common prefix",
        rows=margin_rows,
        panels=(("y margin", (("flowstar_margin_y", "Flow*"), ("torch_margin_y", "Torch"))),),
    )
    artifacts.append({"figure": margin_svg.name, "source_csv": margin_csv.name})

    delta_rows = _with_common(
        _read_csv(args.diff_delta),
        schema="diffreach_torch_full_horizon_delta_figure_row_v1",
        eligibility="full_horizon_diagnostic_diverged_not_timing_eligible",
    )
    delta_csv = output / "diffreach_torch_full_horizon_deltas.csv"
    delta_svg = output / "diffreach_torch_full_horizon_deltas.svg"
    _write_csv(delta_csv, delta_rows)
    _line_panels(
        delta_svg,
        title="DiffReach/Torch DR7 full-horizon deltas",
        subtitle="explicit-f64, B64, h=0.01; endpoint/tube are not within the preregistered 2-ULP envelope",
        rows=delta_rows,
        x_field="step",
        panels=(
            ("maximum absolute delta", (("endpoint_max_abs", "endpoint"), ("tube_max_abs", "tube"))),
            ("maximum ULP delta", (("endpoint_max_ulp", "endpoint"), ("tube_max_ulp", "tube"))),
        ),
    )
    artifacts.append({"figure": delta_svg.name, "source_csv": delta_csv.name})

    carry_rows: list[dict[str, Any]] = []
    for cell, source in (("A3", args.a3_metrics), ("A4", args.a4_metrics)):
        for row in _read_csv(source):
            carry_rows.append(
                {
                    "schema": "a3_a4_carry_metric_figure_row_v1",
                    "cell": cell,
                    **row,
                    "eligibility": "ordinary_float64_empirical_carry_diagnostic",
                    "sample_count": 1,
                }
            )
    carry_csv = output / "a3_a4_scale_composition_remainder.csv"
    carry_svg = output / "a3_a4_scale_composition_remainder.svg"
    _write_csv(carry_csv, carry_rows)
    flattened = []
    for row in carry_rows:
        flat = dict(row)
        for field in ("scale_max", "composition_ledger_width_max", "parameterization_remainder_width_max"):
            flat[f"{row['cell']}_{field}"] = row[field]
        flattened.append(flat)
    _line_panels(
        carry_svg,
        title="A3/A4 carry growth",
        subtitle="A4 stops at its rejected candidate step 320; no post-failure zero fill or connecting line",
        rows=flattened,
        panels=(
            ("scale max", (("A3_scale_max", "A3"), ("A4_scale_max", "A4"))),
            ("composition remainder width", (("A3_composition_ledger_width_max", "A3"), ("A4_composition_ledger_width_max", "A4"))),
            ("parameterization remainder width", (("A3_parameterization_remainder_width_max", "A3"), ("A4_parameterization_remainder_width_max", "A4"))),
        ),
    )
    artifacts.append({"figure": carry_svg.name, "source_csv": carry_csv.name})

    source_rows: list[dict[str, Any]] = []
    reports = ((2, 0.01, _json(args.first_accounting)), (320, 3.19, _json(args.failure_accounting)))
    categories = (
        "degree_gt4_dropped_polynomial",
        "polynomial_times_parameterization_remainder",
        "endpoint_remainder_times_parameterization_polynomial",
        "remainder_times_remainder",
        "outer_endpoint_remainder",
    )
    for step, time_value, report in reports:
        checkpoint = report["checkpoints"][0]
        for category in categories:
            value = float(checkpoint["source_intervals"][category]["max_width"])
            source_rows.append(
                {
                    "schema": "a4_remainder_source_figure_row_v1",
                    "step": step,
                    "time": time_value,
                    "category": category,
                    "max_width": value,
                    "max_width_hex": value.hex(),
                    "eligibility": "same_prestate_native_observer_bit_exact_composition_accounting",
                    "sample_count": 1,
                }
            )
    source_csv = output / "a4_remainder_sources.csv"
    source_svg = output / "a4_remainder_sources.svg"
    _write_csv(source_csv, source_rows)
    chart_width, chart_height = 1120, 440
    left, right, top, bottom = 130.0, 1060.0, 85.0, 350.0
    maximum = max(float(row["max_width"]) for row in source_rows) or 1.0
    body = [
        _text(18, 28, "A4 composition remainder sources", "title"),
        _text(18, 48, "stacked maximum component widths at first material and pre-failure checkpoints", "tick"),
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>',
        _text(left - 8, top + 4, f"{maximum:.4g}", "tick", "end"),
        _text(left - 8, bottom, "0", "tick", "end"),
    ]
    for bar_index, step in enumerate((2, 320)):
        x, bar_width = 300.0 + bar_index * 430.0, 190.0
        y = bottom
        for category_index, category in enumerate(categories):
            value = next(float(row["max_width"]) for row in source_rows if int(row["step"]) == step and row["category"] == category)
            height = value / maximum * (bottom - top)
            y -= height
            body.append(f'<rect x="{x}" y="{y:.2f}" width="{bar_width}" height="{height:.2f}" fill="{COLORS[category_index]}"/>')
        body.append(_text(x + bar_width / 2, bottom + 20, f"before step {step}", "label", "middle"))
    for index, category in enumerate(categories):
        x = 120.0 + (index % 3) * 340.0
        y = 385.0 + (index // 3) * 22.0
        body.append(f'<rect x="{x}" y="{y - 10}" width="15" height="10" fill="{COLORS[index]}"/>')
        body.append(_text(x + 21, y, category, "tick"))
    _svg(source_svg, chart_width, chart_height, body)
    artifacts.append({"figure": source_svg.name, "source_csv": source_csv.name})

    for artifact in artifacts:
        artifact["figure_sha256"] = _sha(output / artifact["figure"])
        artifact["source_csv_sha256"] = _sha(output / artifact["source_csv"])
    summary = {
        "schema": "full_horizon_pairwise_causal_figures_v1",
        "outcome": "CAUSAL_FIGURES_BUILT",
        "figure_count": len(artifacts),
        "artifacts": artifacts,
        "constraints": {
            "every_figure_has_source_csv": len(artifacts) == 5,
            "post_failure_zero_fill": False,
            "universal_winner_chart": False,
            "fix_horizon_chart": "not_generated_no_fix_authorized",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return summary


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-common-prefix", type=Path, required=True)
    parser.add_argument("--diff-delta", type=Path, required=True)
    parser.add_argument("--a3-metrics", type=Path, required=True)
    parser.add_argument("--a4-metrics", type=Path, required=True)
    parser.add_argument("--first-accounting", type=Path, required=True)
    parser.add_argument("--failure-accounting", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(_args())
