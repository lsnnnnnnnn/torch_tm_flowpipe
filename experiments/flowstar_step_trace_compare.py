"""Generate Flow* and PyTorch accepted-step traces for Van der Pol diagnostics.

This is intentionally a local diagnostic runner, not a new reachability mode.
It compiles a repo-local C++ probe against the local Flow* toolbox and converts
existing PyTorch normalized-insertion diagnostics into the same step schema.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import FlowstarNormalFlowpipeState, Interval, TMVector, flowpipe_step_flowstar_style_adaptive
from torch_tm_flowpipe.ode_examples import van_der_pol_ode

PROBE_CPP = ROOT / "experiments" / "flowstar_probe" / "flowstar_vdp_step_trace_probe.cpp"
DEFAULT_OUT = ROOT / "outputs" / "flowstar_step_trace_compare"

TRACE_FIELDS = [
    "source",
    "mode",
    "step_index",
    "adaptive_attempt_index",
    "t_before",
    "h",
    "t_after",
    "accepted",
    "rejected",
    "status",
    "message",
    "tmv_pre_range_width_x",
    "tmv_pre_range_width_y",
    "tmv_pre_range_width_sum",
    "tmv_right_range_width_x",
    "tmv_right_range_width_y",
    "tmv_right_range_width_sum",
    "tmv_right_normal_range_width_x",
    "tmv_right_normal_range_width_y",
    "tmv_right_normal_range_width_sum",
    "endpoint_pre_center_width_x",
    "endpoint_pre_center_width_y",
    "endpoint_pre_center_width_sum",
    "center_x",
    "center_y",
    "scale_x",
    "scale_y",
    "inv_scale_x",
    "inv_scale_y",
    "new_x0_width_x",
    "new_x0_width_y",
    "new_x0_width_sum",
    "target_remainder_width_x",
    "target_remainder_width_y",
    "target_remainder_width_sum",
    "picard_no_remainder_residual_width_x",
    "picard_no_remainder_residual_width_y",
    "picard_no_remainder_residual_width_sum",
    "picard_ctrunc_normal_residual_width_x",
    "picard_ctrunc_normal_residual_width_y",
    "picard_ctrunc_normal_residual_width_sum",
    "cutoff_polynomial_difference_width_x",
    "cutoff_polynomial_difference_width_y",
    "cutoff_polynomial_difference_width_sum",
    "symbolic_J_size",
    "symbolic_Phi_L_size",
    "scalar_x",
    "scalar_y",
    "symbolic_J_width_x",
    "symbolic_J_width_y",
    "symbolic_J_width_sum",
    "symbolic_propagated_width_x",
    "symbolic_propagated_width_y",
    "symbolic_propagated_width_sum",
    "final_flowpipe_width_x",
    "final_flowpipe_width_y",
    "final_flowpipe_width_sum",
    "residual_width_x",
    "residual_width_y",
    "residual_width_sum",
    "residual_over_target_x",
    "residual_over_target_y",
    "residual_over_target_sum",
]

DIFF_FIELDS = [
    "step_index",
    "t_flowstar",
    "t_noqueue",
    "t_v2",
    "flowstar_h",
    "noqueue_h",
    "v2_h",
    "flowstar_width_sum",
    "noqueue_width_sum",
    "v2_width_sum",
    "noqueue_width_ratio",
    "v2_width_ratio",
    "flowstar_residual_sum",
    "noqueue_residual_sum",
    "v2_residual_sum",
    "noqueue_residual_ratio",
    "v2_residual_ratio",
    "center_scale_delta_noqueue",
    "center_scale_delta_v2",
    "inserted_endpoint_ratio_noqueue",
    "inserted_endpoint_ratio_v2",
    "right_map_ratio_noqueue",
    "right_map_ratio_v2",
    "target_remainder_ratio_noqueue",
    "target_remainder_ratio_v2",
    "picard_residual_ratio_noqueue",
    "picard_residual_ratio_v2",
    "cutoff_poly_ratio_noqueue",
    "cutoff_poly_ratio_v2",
    "symbolic_queue_ratio_v2",
    "first_material_channel",
    "notes",
]


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.17g}"
    return str(value)


def _ratio(numerator: Any, denominator: Any) -> float | None:
    num = _float(numerator)
    den = _float(denominator)
    if num is None or den is None or abs(den) <= 0.0:
        return None
    return num / den


def _abs_delta(a: Any, b: Any) -> float | None:
    fa = _float(a)
    fb = _float(b)
    if fa is None or fb is None:
        return None
    return abs(fa - fb)


def _max_abs_delta(reference: Mapping[str, Any], candidate: Mapping[str, Any], fields: Iterable[str]) -> float | None:
    values = [_abs_delta(reference.get(field), candidate.get(field)) for field in fields]
    finite = [value for value in values if value is not None]
    return max(finite) if finite else None


def _sum_widths(row: Mapping[str, Any], prefix: str) -> float | None:
    explicit = _float(row.get(f"{prefix}_width_sum"))
    if explicit is not None:
        return explicit
    x = _float(row.get(f"{prefix}_width_x")) or 0.0
    y = _float(row.get(f"{prefix}_width_y")) or 0.0
    return x + y


def _put_widths(out: dict[str, Any], target_prefix: str, source: Mapping[str, Any], source_prefix: str) -> None:
    for suffix in ("x", "y", "sum"):
        source_key = f"{source_prefix}_width_{suffix}"
        if suffix == "sum":
            value = source.get(source_key)
            if value in (None, ""):
                value = _sum_widths(source, source_prefix)
        else:
            value = source.get(source_key)
        out[f"{target_prefix}_width_{suffix}"] = value if value is not None else ""


def _zero_widths(out: dict[str, Any], target_prefix: str) -> None:
    for suffix in ("x", "y", "sum"):
        out[f"{target_prefix}_width_{suffix}"] = 0.0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _str(row.get(field, "")) for field in fieldnames})


def compile_probe(flowstar_root: Path, out_dir: Path, compiler: str = "g++") -> Path:
    exe = out_dir / "flowstar_vdp_step_trace_probe"
    cmd = [
        compiler,
        "-O3",
        "-w",
        "-std=c++11",
        "-I",
        str(flowstar_root / "flowstar-toolbox"),
        "-I",
        "/usr/local/include",
        str(PROBE_CPP),
        "-L",
        str(flowstar_root / "flowstar-toolbox"),
        "-L",
        "/usr/local/lib",
        "-o",
        str(exe),
        "-lflowstar",
        "-lmpfr",
        "-lgmp",
        "-lgsl",
        "-lgslcblas",
        "-lm",
        "-lglpk",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    (out_dir / "flowstar_probe_compile.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (out_dir / "flowstar_probe_compile.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Flow* probe compilation failed; see {out_dir / 'flowstar_probe_compile.stderr.txt'}")
    return exe


def run_flowstar_probe(exe: Path, out_dir: Path, horizon: float, max_segments: int | None) -> Path:
    trace = out_dir / "flowstar_trace.csv"
    cmd = [str(exe), str(trace), f"{horizon:.17g}"]
    if max_segments:
        cmd.append(str(max_segments))
    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=str(out_dir), check=False)
    (out_dir / "flowstar_probe_run.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (out_dir / "flowstar_probe_run.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Flow* probe run failed; see {out_dir / 'flowstar_probe_run.stderr.txt'}")
    return trace


def _validation_by_adaptive_attempt(diagnostics: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for row in diagnostics:
        adaptive = row.get("adaptive_attempt_index")
        if adaptive in (None, ""):
            continue
        try:
            key = int(adaptive)
        except (TypeError, ValueError):
            continue
        grouped[key] = row
    return grouped


def _common_torch_row(
    *,
    mode: str,
    step_index: int,
    t_before: float,
    validation: Mapping[str, Any],
    accepted: bool,
    normal_stats: Mapping[str, Any] | None,
    target_radius: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": "torch",
        "mode": mode,
        "step_index": step_index,
        "adaptive_attempt_index": validation.get("adaptive_attempt_index", ""),
        "t_before": t_before,
        "h": validation.get("h_try", validation.get("h", "")),
        "accepted": accepted,
        "rejected": not accepted,
        "status": "accepted" if accepted else "rejected",
        "message": validation.get("validation_message", "") or validation.get("rejection_reason", ""),
    }
    h = _float(row["h"])
    if h is not None:
        row["t_after"] = t_before + h

    _put_widths(row, "tmv_pre_range", validation, "candidate_segment")
    _put_widths(row, "final_flowpipe", validation, "candidate_final")
    _put_widths(row, "residual", validation, "residual")
    _put_widths(row, "picard_no_remainder_residual", validation, "residual")

    for suffix in ("x", "y"):
        row[f"target_remainder_width_{suffix}"] = 2.0 * target_radius
    row["target_remainder_width_sum"] = 4.0 * target_radius

    if "tmp_remainder_width_sum" in validation:
        _put_widths(row, "picard_ctrunc_normal_residual", validation, "tmp_remainder")
    else:
        _put_widths(row, "picard_ctrunc_normal_residual", validation, "residual")

    if "poly_diff_range_width_sum" in validation:
        _put_widths(row, "cutoff_polynomial_difference", validation, "poly_diff_range")
    elif normal_stats:
        _put_widths(row, "cutoff_polynomial_difference", normal_stats, "insertion_cutoff")
    else:
        _zero_widths(row, "cutoff_polynomial_difference")

    residual_sum = _float(row.get("residual_width_sum"))
    if residual_sum is not None:
        row["residual_over_target_sum"] = residual_sum / (4.0 * target_radius)
    for suffix in ("x", "y"):
        residual = _float(row.get(f"residual_width_{suffix}"))
        if residual is not None:
            row[f"residual_over_target_{suffix}"] = residual / (2.0 * target_radius)

    if normal_stats:
        _put_widths(row, "tmv_right_range", normal_stats, "old_right_map_range")
        _put_widths(row, "tmv_right_normal_range", normal_stats, "normal_right_map_range")
        _put_widths(row, "endpoint_pre_center", normal_stats, "inserted_endpoint")
        _put_widths(row, "new_x0", normal_stats, "normalized_reset")
        for key in ("center_x", "center_y", "scale_x", "scale_y"):
            row[key] = normal_stats.get(key, "")
        for suffix in ("x", "y"):
            scale = _float(normal_stats.get(f"scale_{suffix}"))
            row[f"inv_scale_{suffix}"] = "" if scale is None or scale == 0.0 else 1.0 / scale
        row["symbolic_J_size"] = normal_stats.get("j_count", normal_stats.get("queue_size_after", 0 if mode == "torch_noqueue" else ""))
        row["symbolic_Phi_L_size"] = normal_stats.get("phi_l_count", "")
        row["scalar_x"] = normal_stats.get("scalar_x", row.get("inv_scale_x", ""))
        row["scalar_y"] = normal_stats.get("scalar_y", row.get("inv_scale_y", ""))
        if mode == "torch_v2":
            _put_widths(row, "symbolic_J", normal_stats, "new_symbolic")
            _put_widths(row, "symbolic_propagated", normal_stats, "propagated_symbolic")
        else:
            _zero_widths(row, "symbolic_J")
            _zero_widths(row, "symbolic_propagated")
    else:
        _zero_widths(row, "tmv_right_range")
        _zero_widths(row, "tmv_right_normal_range")
        _zero_widths(row, "endpoint_pre_center")
        _zero_widths(row, "new_x0")
        _zero_widths(row, "symbolic_J")
        _zero_widths(row, "symbolic_propagated")

    return row


def generate_torch_trace(
    *,
    mode: str,
    horizon: float,
    out_path: Path,
    target_radius: float = 1e-4,
    h_min: float = 0.002,
    h_max: float = 0.1,
    order: int = 4,
    validation_mode: str = "target_remainder_flowstar_ctrunc",
    max_segments: int | None = None,
) -> list[dict[str, Any]]:
    if mode not in {"torch_noqueue", "torch_v2"}:
        raise ValueError(f"unsupported torch trace mode: {mode}")
    current: TMVector | list[Interval] = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    normal_state: FlowstarNormalFlowpipeState | None = None
    h = h_max
    t = 0.0
    step_index = 0
    trace_rows: list[dict[str, Any]] = []
    reset_mode = "normalized_insertion" if mode == "torch_noqueue" else "normalized_insertion_symqueue_v2"
    symbolic_queue_mode = "" if mode == "torch_noqueue" else "flowstar_linear_v2"

    while t < horizon - 1e-15:
        if max_segments is not None and step_index >= max_segments:
            break
        h_try = min(h, h_max, horizon - t)
        if h_try < h_min:
            h_try = h_min
        diagnostics: list[dict[str, Any]] = []
        seg = flowpipe_step_flowstar_style_adaptive(
            van_der_pol_ode,
            current,
            h=h_try,
            h_min=h_min,
            h_max=h_max,
            order=order,
            target_remainder_radius=target_radius,
            cutoff_threshold=1e-10,
            max_validation_attempts=2,
            validation_mode=validation_mode,
            reset_mode=reset_mode,
            symbolic_queue_mode=symbolic_queue_mode,
            flowstar_symbolic_queue_max_size=100,
            flowstar_normal_state=normal_state,
            diagnostics=diagnostics,
            diagnostics_context={"mode": mode, "segment_index": step_index, "t_before": t},
        )

        grouped = _validation_by_adaptive_attempt(diagnostics)
        accepted_attempt = None
        if seg.status == "validated":
            accepted_h = _float(seg.h)
            for attempt, validation in grouped.items():
                if _float(validation.get("h_try", validation.get("h"))) == accepted_h:
                    accepted_attempt = attempt
            if accepted_attempt is None and grouped:
                accepted_attempt = max(grouped)

        for attempt in sorted(grouped):
            validation = grouped[attempt]
            accepted = bool(seg.status == "validated" and attempt == accepted_attempt)
            trace_rows.append(
                _common_torch_row(
                    mode=mode,
                    step_index=step_index,
                    t_before=t,
                    validation=validation,
                    accepted=accepted,
                    normal_stats=seg.flowstar_normal_stats if accepted else None,
                    target_radius=target_radius,
                )
            )

        if seg.status != "validated" or seg.reset_tm is None:
            break

        t += float(seg.h)
        current = seg.reset_tm
        normal_state = seg.flowstar_normal_state
        h = float(seg.next_h if seg.next_h is not None else min(float(seg.h) * 1.5, h_max))
        step_index += 1

    _write_rows(out_path, TRACE_FIELDS, trace_rows)
    return trace_rows


def _accepted(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if str(row.get("accepted", "")).lower() == "true" or row.get("status") == "accepted"]


def _channel_ratios(flow: Mapping[str, Any], torch_row: Mapping[str, Any]) -> dict[str, float | None]:
    center_scale_delta = _max_abs_delta(
        flow,
        torch_row,
        ("center_x", "center_y", "scale_x", "scale_y", "inv_scale_x", "inv_scale_y"),
    )
    return {
        "center_scale": center_scale_delta,
        "inserted_endpoint": _ratio(torch_row.get("endpoint_pre_center_width_sum"), flow.get("endpoint_pre_center_width_sum")),
        "right_map": _ratio(torch_row.get("tmv_right_normal_range_width_sum"), flow.get("tmv_right_normal_range_width_sum")),
        "target_remainder": _ratio(torch_row.get("target_remainder_width_sum"), flow.get("target_remainder_width_sum")),
        "picard_residual": _ratio(torch_row.get("picard_ctrunc_normal_residual_width_sum"), flow.get("picard_ctrunc_normal_residual_width_sum")),
        "cutoff_poly": _ratio(torch_row.get("cutoff_polynomial_difference_width_sum"), flow.get("cutoff_polynomial_difference_width_sum")),
        "symbolic_queue": _ratio(torch_row.get("symbolic_J_width_sum"), flow.get("symbolic_J_width_sum")),
    }


def _material_channel(noq: dict[str, float | None], v2: dict[str, float | None], *, ratio_threshold: float, delta_threshold: float) -> str:
    channels = [
        ("center/scaling", "center_scale"),
        ("inserted endpoint map", "inserted_endpoint"),
        ("right-map range", "right_map"),
        ("target remainder", "target_remainder"),
        ("Picard residual", "picard_residual"),
        ("cutoff / polynomial truncation", "cutoff_poly"),
        ("symbolic queue", "symbolic_queue"),
    ]
    for label, key in channels:
        values = [noq.get(key), v2.get(key)]
        for value in values:
            if value is None:
                continue
            if key == "center_scale":
                if value > delta_threshold:
                    return label
            elif value > ratio_threshold or value < 1.0 / ratio_threshold:
                return label
    return ""


def align_traces(flowstar_rows: list[Mapping[str, Any]], noqueue_rows: list[Mapping[str, Any]], v2_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flow_acc = _accepted(flowstar_rows)
    noq_acc = _accepted(noqueue_rows)
    v2_acc = _accepted(v2_rows)
    total = min(len(flow_acc), len(noq_acc), len(v2_acc))
    rows: list[dict[str, Any]] = []
    for index in range(total):
        flow = flow_acc[index]
        noq = noq_acc[index]
        v2 = v2_acc[index]
        noq_channels = _channel_ratios(flow, noq)
        v2_channels = _channel_ratios(flow, v2)
        channel = _material_channel(noq_channels, v2_channels, ratio_threshold=1.25, delta_threshold=1e-6)
        row = {
            "step_index": index,
            "t_flowstar": flow.get("t_before", ""),
            "t_noqueue": noq.get("t_before", ""),
            "t_v2": v2.get("t_before", ""),
            "flowstar_h": flow.get("h", ""),
            "noqueue_h": noq.get("h", ""),
            "v2_h": v2.get("h", ""),
            "flowstar_width_sum": flow.get("final_flowpipe_width_sum", ""),
            "noqueue_width_sum": noq.get("final_flowpipe_width_sum", ""),
            "v2_width_sum": v2.get("final_flowpipe_width_sum", ""),
            "noqueue_width_ratio": _ratio(noq.get("final_flowpipe_width_sum"), flow.get("final_flowpipe_width_sum")),
            "v2_width_ratio": _ratio(v2.get("final_flowpipe_width_sum"), flow.get("final_flowpipe_width_sum")),
            "flowstar_residual_sum": flow.get("picard_ctrunc_normal_residual_width_sum", flow.get("residual_width_sum", "")),
            "noqueue_residual_sum": noq.get("picard_ctrunc_normal_residual_width_sum", noq.get("residual_width_sum", "")),
            "v2_residual_sum": v2.get("picard_ctrunc_normal_residual_width_sum", v2.get("residual_width_sum", "")),
            "noqueue_residual_ratio": _ratio(noq.get("picard_ctrunc_normal_residual_width_sum"), flow.get("picard_ctrunc_normal_residual_width_sum")),
            "v2_residual_ratio": _ratio(v2.get("picard_ctrunc_normal_residual_width_sum"), flow.get("picard_ctrunc_normal_residual_width_sum")),
            "center_scale_delta_noqueue": noq_channels["center_scale"],
            "center_scale_delta_v2": v2_channels["center_scale"],
            "inserted_endpoint_ratio_noqueue": noq_channels["inserted_endpoint"],
            "inserted_endpoint_ratio_v2": v2_channels["inserted_endpoint"],
            "right_map_ratio_noqueue": noq_channels["right_map"],
            "right_map_ratio_v2": v2_channels["right_map"],
            "target_remainder_ratio_noqueue": noq_channels["target_remainder"],
            "target_remainder_ratio_v2": v2_channels["target_remainder"],
            "picard_residual_ratio_noqueue": noq_channels["picard_residual"],
            "picard_residual_ratio_v2": v2_channels["picard_residual"],
            "cutoff_poly_ratio_noqueue": noq_channels["cutoff_poly"],
            "cutoff_poly_ratio_v2": v2_channels["cutoff_poly"],
            "symbolic_queue_ratio_v2": v2_channels["symbolic_queue"],
            "first_material_channel": channel,
            "notes": "ratio threshold 1.25, center/scale absolute delta threshold 1e-6" if channel else "",
        }
        rows.append(row)
    return rows


def write_report(out_dir: Path, aligned: list[Mapping[str, Any]], *, horizon: float, docs: bool = True) -> str:
    first = next((row for row in aligned if row.get("first_material_channel")), None)
    first_ratio = next(
        (
            row
            for row in aligned
            if (_float(row.get("noqueue_width_ratio")) or 0.0) > 1.25
            or (_float(row.get("v2_width_ratio")) or 0.0) > 1.25
            or (_float(row.get("noqueue_residual_ratio")) or 0.0) > 1.25
            or (_float(row.get("v2_residual_ratio")) or 0.0) > 1.25
        ),
        None,
    )
    lines = [
        "# Flow* Accepted-Step Trace Divergence Report",
        "",
        "This is a diagnostic probe, not a Flow* parity claim.",
        "",
        f"- Horizon traced: T={horizon:g}",
        "- Flow* source: local toolbox probe linked against `/srv/local/shengenli/flowstar/flowstar-toolbox/libflowstar.a`",
        "- PyTorch modes: existing `normalized_insertion` no_queue and `normalized_insertion_symqueue_v2` with `flowstar_linear_v2`",
        "- Material threshold: width/residual/channel ratio outside `[0.8, 1.25]`, or center/scaling absolute delta above `1e-6`",
        "",
    ]
    if first:
        lines.extend(
            [
                "## First Channel Divergence",
                "",
                f"- Step: {first.get('step_index')}",
                f"- Channel: {first.get('first_material_channel')}",
                f"- Flow* h: {first.get('flowstar_h')}",
                f"- no_queue width ratio: {first.get('noqueue_width_ratio')}",
                f"- v2 width ratio: {first.get('v2_width_ratio')}",
                f"- no_queue residual ratio: {first.get('noqueue_residual_ratio')}",
                f"- v2 residual ratio: {first.get('v2_residual_ratio')}",
                "",
            ]
        )
    else:
        lines.extend(["## First Channel Divergence", "", "- No material channel divergence found over aligned accepted steps.", ""])
    if first_ratio:
        lines.extend(
            [
                "## First Width Or Residual Divergence",
                "",
                f"- Step: {first_ratio.get('step_index')}",
                f"- no_queue width ratio: {first_ratio.get('noqueue_width_ratio')}",
                f"- v2 width ratio: {first_ratio.get('v2_width_ratio')}",
                f"- no_queue residual ratio: {first_ratio.get('noqueue_residual_ratio')}",
                f"- v2 residual ratio: {first_ratio.get('v2_residual_ratio')}",
                "",
            ]
        )
    else:
        lines.extend(["## First Width Or Residual Divergence", "", "- No material width/residual divergence found over aligned accepted steps.", ""])
    lines.extend(
        [
            "## Output Files",
            "",
            "- `outputs/flowstar_step_trace_compare/flowstar_trace.csv`",
            "- `outputs/flowstar_step_trace_compare/torch_noqueue_trace.csv`",
            "- `outputs/flowstar_step_trace_compare/torch_v2_trace.csv`",
            "- `outputs/flowstar_step_trace_compare/aligned_trace_diff.csv`",
            "",
            "## Limitations",
            "",
            "- The Flow* C++ probe mirrors the local adaptive symbolic-remainder path for this benchmark and logs public internals; it does not patch or commit Flow* source.",
            "- PyTorch cutoff/Picard fields use existing diagnostics. Fields absent in a mode are left blank or zeroed when the channel is not present.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (out_dir / "trace_divergence_report.md").write_text(text, encoding="utf-8")
    if docs:
        (ROOT / "docs" / "flowstar_step_trace_divergence_report.md").write_text(text, encoding="utf-8")
    return text


def write_plan_doc() -> None:
    text = """# Flow* Accepted-Step Trace Plan

This diagnostic uses a repo-local C++ probe in `experiments/flowstar_probe/` and does not commit Flow* source changes.

The probe links against `/srv/local/shengenli/flowstar/flowstar-toolbox/libflowstar.a` and mirrors the local Van der Pol adaptive symbolic-remainder step path with original Flow* settings: adaptive step `0.002..0.1`, order `4`, cutoff `1e-10`, target remainder `[-1e-4, 1e-4]`, initial box `x=[1.1,1.4]`, `y=[2.35,2.45]`.

Trace rows are emitted per adaptive attempt, including rejected shrink attempts and accepted steps. The aligned comparator keeps Flow* as the diagnostic reference and converts existing PyTorch normalized insertion diagnostics for `normalized_insertion` no_queue and `normalized_insertion_symqueue_v2`/`flowstar_linear_v2` into the same columns.

Required channels:

- accepted-step timing: `t_before`, `h`, accepted/rejected status
- pre/right map range widths and normal range widths
- endpoint range before center extraction, center, scale, inverse scale
- normalized `new_x0` box/range and target remainder
- Picard no-remainder and ctrunc residual widths
- cutoff/polynomial-difference contribution when available
- symbolic queue sizes, scalar values, and symbolic width contributions
- final flowpipe segment width

This is not a Flow* parity proof. It is a short-horizon probe for localizing the first material divergence channel.
"""
    (ROOT / "docs" / "flowstar_accepted_step_trace_plan.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flowstar-root", default="/srv/local/shengenli/flowstar")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--horizon", type=float, default=1.0)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--compiler", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--skip-flowstar", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_plan_doc()

    flowstar_trace = out_dir / "flowstar_trace.csv"
    if not args.skip_flowstar:
        exe = compile_probe(Path(args.flowstar_root).resolve(), out_dir, compiler=args.compiler)
        flowstar_trace = run_flowstar_probe(exe, out_dir, args.horizon, args.max_segments or None)
    flowstar_rows = _read_rows(flowstar_trace)

    noqueue_rows = generate_torch_trace(
        mode="torch_noqueue",
        horizon=args.horizon,
        out_path=out_dir / "torch_noqueue_trace.csv",
        max_segments=args.max_segments or None,
    )
    v2_rows = generate_torch_trace(
        mode="torch_v2",
        horizon=args.horizon,
        out_path=out_dir / "torch_v2_trace.csv",
        max_segments=args.max_segments or None,
    )
    aligned = align_traces(flowstar_rows, noqueue_rows, v2_rows)
    _write_rows(out_dir / "aligned_trace_diff.csv", DIFF_FIELDS, aligned)
    write_report(out_dir, aligned, horizon=args.horizon)
    print(f"Wrote traces and report to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
