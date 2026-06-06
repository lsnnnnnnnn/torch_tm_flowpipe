#!/usr/bin/env python3
'''Replay Flow* and PyTorch from shared normalized-insertion checkpoints.'''
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
FLOWSTAR_RUNNER_ROOT = REPO_ROOT / "comparisons" / "flowstar"
for p in (SRC_ROOT, FLOWSTAR_RUNNER_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_flowstar import run_flowstar_toolbox  # noqa: E402
from torch_tm_flowpipe import Interval, flowpipe_step_flowstar_style_adaptive  # noqa: E402
from torch_tm_flowpipe.ode_examples import van_der_pol_ode  # noqa: E402
from torch_tm_flowpipe.safety import intervals_are_finite  # noqa: E402

NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
NUMERIC_PAIR_RE = re.compile(rf"^\s*(?P<t>{NUMBER_RE})\s+(?P<v>{NUMBER_RE})(?:\s|$)")
COMPLETED_RE = re.compile(r"FLOWSTAR_COMPLETED\s+(?P<ok>[01])")
RUNTIME_RE = re.compile(rf"FLOWSTAR_RUNTIME_S\s+(?P<runtime>{NUMBER_RE})")
RUN_SPECS = {
    "o4_insert": {"run_id": "flowstar_style_o4_target_insert", "order": 4, "candidate_order": None},
    "o6_insert": {"run_id": "flowstar_style_o6_candidate8_output6_insert", "order": 6, "candidate_order": 8},
}
CHECKPOINTS = [0.75, 1.30, 1.68, 2.12, 3.0, 5.0, 6.4, 7.4]
SUMMARY_FIELDS = [
    "checkpoint_label", "checkpoint_t", "path_id", "run_id", "order", "candidate_order",
    "source_segment_index", "source_t_hi", "h_try", "reset_box_width_x", "reset_box_width_y",
    "reset_box_width_sum", "flowstar_one_step_status", "flowstar_one_step_completed",
    "flowstar_one_step_width_sum", "pytorch_one_step_status", "pytorch_one_step_completed",
    "pytorch_one_step_width_sum", "flowstar_mini_status", "flowstar_mini_completed",
    "flowstar_mini_width_sum", "pytorch_mini_status", "pytorch_mini_completed",
    "pytorch_mini_width_sum", "width_ratio_pytorch_over_flowstar", "step_rejection_count",
    "residual_failure_dimension", "residual_failure_width_x", "residual_failure_width_y",
    "residual_failure_width_sum", "failure_reason",
]
SEGMENT_FIELDS = [
    "checkpoint_label", "checkpoint_t", "path_id", "engine", "horizon_kind", "segment_index",
    "t_lo", "t_hi", "x_lo", "x_hi", "y_lo", "y_hi", "width_x", "width_y",
    "width_sum", "status",
]


def _fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.17g}" if math.isfinite(value) else ""
    return value


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _finite_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _box_from_row(row: Mapping[str, Any]) -> list[Interval]:
    return [Interval(float(row["x_lo"]), float(row["x_hi"])), Interval(float(row["y_lo"]), float(row["y_hi"]))]


def _has_box_bounds(row: Mapping[str, Any]) -> bool:
    return all(_finite_float(row.get(key)) is not None for key in ("x_lo", "x_hi", "y_lo", "y_hi"))


def _checkpoint_row_from_segment(row: Mapping[str, str]) -> dict[str, Any] | None:
    if row.get("status") != "validated" or not _has_box_bounds(row):
        return None
    width_x = _finite_float(row.get("reset_width_x")) or _finite_float(row.get("width_x"))
    width_y = _finite_float(row.get("reset_width_y")) or _finite_float(row.get("width_y"))
    width_sum = _finite_float(row.get("reset_width_sum"))
    if width_sum is None and width_x is not None and width_y is not None:
        width_sum = width_x + width_y
    return {
        "run_id": row.get("run_id", ""),
        "segment_index": row.get("segment_index", ""),
        "t_lo": row.get("t_lo", ""),
        "t_hi": row.get("t_hi", ""),
        "x_lo": row.get("x_lo", ""),
        "x_hi": row.get("x_hi", ""),
        "y_lo": row.get("y_lo", ""),
        "y_hi": row.get("y_hi", ""),
        "h": row.get("h", ""),
        "order": row.get("order", ""),
        "reset_box_source": row.get("reset_box_source", "segment_final_box") or "segment_final_box",
        "validation_mode": row.get("validation_mode", ""),
        "reset_mode": row.get("reset_mode", ""),
        "width_x": width_x if width_x is not None else row.get("width_x", ""),
        "width_y": width_y if width_y is not None else row.get("width_y", ""),
        "width_sum": width_sum if width_sum is not None else row.get("width_sum", ""),
    }


def _checkpoint_rows(source_dir: Path, segment_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in _read_csv(source_dir / "rescue_reset_boxes.csv"):
        if not _has_box_bounds(row):
            continue
        key = (row.get("run_id", ""), row.get("segment_index", ""))
        rows.append(dict(row))
        seen.add(key)
    for row in _read_csv(source_dir / "normalized_insertion_h10_reset_diagnostics.csv"):
        if not _has_box_bounds(row):
            continue
        key = (row.get("run_id", ""), row.get("segment_index", ""))
        if key in seen:
            continue
        rows.append(dict(row))
        seen.add(key)
    for row in segment_rows:
        checkpoint_row = _checkpoint_row_from_segment(row)
        if checkpoint_row is None:
            continue
        key = (str(checkpoint_row.get("run_id", "")), str(checkpoint_row.get("segment_index", "")))
        if key in seen:
            continue
        rows.append(checkpoint_row)
        seen.add(key)
    return rows


def _widths_from_box(box: Sequence[Interval]) -> tuple[float, float, float]:
    wx = float(box[0].width())
    wy = float(box[1].width())
    return wx, wy, wx + wy


def _nearest_reset_row(rows: Sequence[Mapping[str, str]], checkpoint: float) -> Mapping[str, str] | None:
    candidates = [row for row in rows if (_finite_float(row.get("t_hi")) or -math.inf) <= checkpoint + 1e-9]
    if not candidates:
        candidates = list(rows)
    if not candidates:
        return None
    return min(candidates, key=lambda row: abs((_finite_float(row.get("t_hi")) or 0.0) - checkpoint))


def _next_h(run_segments: Sequence[Mapping[str, str]], segment_index: int, fallback: float) -> float:
    for row in run_segments:
        idx = int(_finite_float(row.get("segment_index")) or -1)
        if idx == segment_index + 1:
            return _finite_float(row.get("h")) or fallback
    return fallback


def _parse_gnuplot_segments(path: Path) -> list[tuple[float, float, float, float]]:
    if not path.exists():
        return []
    blocks: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    def flush() -> None:
        nonlocal current
        if current:
            blocks.append(current)
            current = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = NUMERIC_PAIR_RE.match(line)
        if not match:
            flush()
            continue
        current.append((float(match.group("t")), float(match.group("v"))))
    flush()
    out = []
    for block in blocks:
        ts = [p[0] for p in block]
        vals = [p[1] for p in block]
        out.append((min(ts), max(ts), min(vals), max(vals)))
    return out


def _combine_segments(x_path: Path, y_path: Path) -> list[dict[str, Any]]:
    rows = []
    for i, (x_seg, y_seg) in enumerate(zip(_parse_gnuplot_segments(x_path), _parse_gnuplot_segments(y_path))):
        x_t_lo, x_t_hi, x_lo, x_hi = x_seg
        y_t_lo, y_t_hi, y_lo, y_hi = y_seg
        wx = max(0.0, x_hi - x_lo)
        wy = max(0.0, y_hi - y_lo)
        rows.append({"segment_index": i, "t_lo": min(x_t_lo, y_t_lo), "t_hi": max(x_t_hi, y_t_hi), "x_lo": x_lo, "x_hi": x_hi, "y_lo": y_lo, "y_hi": y_hi, "width_x": wx, "width_y": wy, "width_sum": wx + wy})
    return rows


def _stdout_value(pattern: re.Pattern[str], path: Path | None, group: str) -> str:
    if path is None or not path.exists():
        return ""
    matches = list(pattern.finditer(path.read_text(encoding="utf-8", errors="ignore")))
    return matches[-1].group(group) if matches else ""


def _render_cpp(stem: str, order: int, step_h: float, horizon: float, reset_box: Mapping[str, Any]) -> str:
    return f'''#include "Continuous.h"
#include <ctime>
#include <cstdio>
#include <vector>
using namespace flowstar;
using namespace std;

int main()
{{
  Variables vars;
  int x_id = vars.declareVar("x");
  int y_id = vars.declareVar("y");
  int t_id = vars.declareVar("t");
  ODE<Real> ode({{"y", "y - x - x^2*y", "1"}}, vars);
  Computational_Setting setting(vars);
  setting.setFixedStepsize({step_h:.17g}, {order});
  setting.setCutoffThreshold(1e-10);
  vector<Interval> remainder_estimation(vars.size());
  for(unsigned int i = 0; i < vars.size(); ++i) remainder_estimation[i] = Interval(-0.0001, 0.0001);
  setting.setRemainderEstimation(remainder_estimation);
  setting.printOn();
  vector<Interval> box(vars.size());
  box[x_id] = Interval({float(reset_box['x_lo']):.17g}, {float(reset_box['x_hi']):.17g});
  box[y_id] = Interval({float(reset_box['y_lo']):.17g}, {float(reset_box['y_hi']):.17g});
  box[t_id] = Interval(0.0, 0.0);
  Flowpipe initialSet(box);
  vector<Constraint> safeSet;
  Result_of_Reachability result;
  clock_t begin = clock();
  ode.reach(result, initialSet, {horizon:.17g}, setting, safeSet);
  clock_t end = clock();
  printf("FLOWSTAR_RUNTIME_S %.17g\\n", (double)(end - begin) / CLOCKS_PER_SEC);
  printf("FLOWSTAR_COMPLETED %d\\n", result.isCompleted() ? 1 : 0);
  result.transformToTaylorModels(setting);
  Plot_Setting plot_setting(vars);
  plot_setting.printOn();
  plot_setting.setOutputDims("t", "x");
  plot_setting.plot_2D_interval_GNUPLOT("./", "{stem}_t_x", result.tmv_flowpipes, setting);
  plot_setting.setOutputDims("t", "y");
  plot_setting.plot_2D_interval_GNUPLOT("./", "{stem}_t_y", result.tmv_flowpipes, setting);
  return 0;
}}
'''


def _run_flowstar(out_dir: Path, label: str, order: int, step_h: float, horizon: float, reset_row: Mapping[str, Any], flowstar_root: str | None, timeout_s: float | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_dir = out_dir / "flowstar_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{label}_o{order}"
    cpp = model_dir / f"{stem}.cpp"
    cpp.write_text(_render_cpp(stem, order, step_h, horizon, reset_row), encoding="utf-8", newline="\n")
    run = run_flowstar_toolbox(cpp, flowstar_root=flowstar_root, output_dir=model_dir, timeout_s=timeout_s, build_lib=True)
    completed = run.status == "completed" and _stdout_value(COMPLETED_RE, run.stdout_path, "ok") == "1"
    status = "completed" if completed else ("not_completed" if run.status == "completed" else run.status)
    segments = _combine_segments(model_dir / f"{stem}_t_x.plt", model_dir / f"{stem}_t_y.plt")
    last = segments[-1] if segments else {}
    return {"status": status, "completed": completed, "width_sum": last.get("width_sum", ""), "runtime_s": _finite_float(_stdout_value(RUNTIME_RE, run.stdout_path, "runtime")) or run.runtime_s, "failure_reason": run.message or ("Flow* reach did not complete" if not completed else "")}, segments


def _pytorch_one_step(box: Sequence[Interval], spec: Mapping[str, Any], h_try: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    seg = flowpipe_step_flowstar_style_adaptive(van_der_pol_ode, box, h=h_try, order=int(spec["order"]), h_min=h_try, h_max=h_try, target_remainder_radius=1e-4, cutoff_threshold=1e-10, max_validation_attempts=2, validation_mode="target_remainder", candidate_order=spec.get("candidate_order"), reset_mode="normalized_insertion", diagnostics=diagnostics, diagnostics_context={"run_id": spec["run_id"], "segment_index": 0})
    final_box = seg.final_tm.range_box()
    wx, wy, width_sum = _widths_from_box(final_box)
    failed = [row for row in diagnostics if row.get("validation_status") == "failed"]
    fail = failed[-1] if failed else {}
    rx = _finite_float(fail.get("residual_width_x"))
    ry = _finite_float(fail.get("residual_width_y"))
    summary = {
        "status": seg.status,
        "completed": seg.status == "validated" and intervals_are_finite(final_box),
        "width_sum": width_sum,
        "step_rejections": getattr(seg, "step_rejections", 0),
        "failure_reason": getattr(seg, "message", ""),
        "residual_failure_width_x": rx if rx is not None else "",
        "residual_failure_width_y": ry if ry is not None else "",
        "residual_failure_width_sum": fail.get("residual_width_sum", ""),
        "residual_failure_dimension": "x" if rx is not None and (ry is None or rx >= ry) else ("y" if ry is not None else ""),
    }
    rows = [
        {
            "segment_index": 0,
            "t_lo": 0.0,
            "t_hi": h_try,
            "x_lo": float(final_box[0].lo),
            "x_hi": float(final_box[0].hi),
            "y_lo": float(final_box[1].lo),
            "y_hi": float(final_box[1].hi),
            "width_x": wx,
            "width_y": wy,
            "width_sum": width_sum,
            "status": seg.status,
        }
    ]
    return summary, rows


def _pytorch_mini(box: Sequence[Interval], spec: Mapping[str, Any], h_try: float, horizon: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    current: Any = list(box)
    t = 0.0
    rows: list[dict[str, Any]] = []
    total_rejections = 0
    failure = ""
    while t < horizon - 1e-15:
        h = min(max(h_try, 0.002), 0.1, horizon - t)
        seg = flowpipe_step_flowstar_style_adaptive(van_der_pol_ode, current, h=h, order=int(spec["order"]), h_min=min(0.002, h), h_max=0.1, target_remainder_radius=1e-4, cutoff_threshold=1e-10, max_validation_attempts=2, validation_mode="target_remainder", candidate_order=spec.get("candidate_order"), reset_mode="normalized_insertion")
        final_box = seg.final_tm.range_box()
        wx, wy, width_sum = _widths_from_box(final_box)
        status = "validated" if seg.status == "validated" and intervals_are_finite(final_box) else "failed"
        rows.append({"segment_index": len(rows), "t_lo": t, "t_hi": t + float(seg.h), "x_lo": float(final_box[0].lo), "x_hi": float(final_box[0].hi), "y_lo": float(final_box[1].lo), "y_hi": float(final_box[1].hi), "width_x": wx, "width_y": wy, "width_sum": width_sum, "status": status})
        total_rejections += int(getattr(seg, "step_rejections", 0) or 0)
        if status != "validated":
            failure = getattr(seg, "message", "") or "validation failed"
            break
        current = seg.reset_tm if seg.reset_tm is not None else seg.final_tm
        t += float(seg.h)
    completed = bool(rows) and rows[-1]["status"] == "validated" and rows[-1]["t_hi"] >= horizon - 1e-12
    return {"status": "completed" if completed else "failed", "completed": completed, "width_sum": rows[-1]["width_sum"] if rows else "", "step_rejections": total_rejections, "failure_reason": failure}, rows


def _decorate_segments(rows: Sequence[Mapping[str, Any]], *, label: str, checkpoint: float, path_id: str, engine: str, horizon_kind: str, status: str) -> list[dict[str, Any]]:
    return [dict(row, checkpoint_label=label, checkpoint_t=checkpoint, path_id=path_id, engine=engine, horizon_kind=horizon_kind, status=row.get("status", status)) for row in rows]


def _make_plot(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    pts = [(_finite_float(row.get("checkpoint_t")), _finite_float(row.get("width_ratio_pytorch_over_flowstar")), str(row.get("path_id", ""))) for row in rows]
    pts = [(t, r, p) for t, r, p in pts if t is not None and r is not None]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for path_id in sorted({p for _t, _r, p in pts}):
        sub = sorted([(t, r) for t, r, p in pts if p == path_id])
        ax.plot([t for t, _ in sub], [r for _, r in sub], marker="o", linewidth=1.0, label=path_id)
    ax.axhline(1.0, color="#111111", linewidth=0.9, linestyle="--")
    ax.set_xlabel("checkpoint t")
    ax.set_ylabel("PyTorch mini width / Flow* mini width")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "checkpoint_width_ratio.png", dpi=160)
    plt.close(fig)


def _write_report(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    first_flow_fail = next((r for r in rows if str(r.get("flowstar_mini_completed", "")).lower() not in {"true", "1", "yes"}), None)
    first_wide = next((r for r in rows if (_finite_float(r.get("width_ratio_pytorch_over_flowstar")) or 0.0) >= 2.0), None)
    o4_last = max((_finite_float(r.get("checkpoint_t")) or 0.0 for r in rows if r.get("path_id") == "o4_insert" and str(r.get("flowstar_mini_completed", "")).lower() in {"true", "1", "yes"}), default=0.0)
    o6_last = max((_finite_float(r.get("checkpoint_t")) or 0.0 for r in rows if r.get("path_id") == "o6_insert" and str(r.get("flowstar_mini_completed", "")).lower() in {"true", "1", "yes"}), default=0.0)
    target = first_wide or first_flow_fail or (rows[-1] if rows else {})
    lines = [
        "# Flowstar Checkpoint Replay Report", "",
        f"At which checkpoint does Flow* first fail from the PyTorch local box? `{first_flow_fail.get('checkpoint_label', 'none') if first_flow_fail else 'none observed'}`.",
        f"At which checkpoint does PyTorch first become much wider than Flow*? `{first_wide.get('checkpoint_label', 'none') if first_wide else 'none observed'}`.",
        f"Does o4 stay Flow*-replayable longer than o6? {'yes' if o4_last > o6_last else 'no'}; o4 last replayable checkpoint=`{o4_last}`, o6=`{o6_last}`.",
        f"Is h10 failure caused by boxes already too wide, or by a local kernel mismatch? {'boxes already too wide/local step too hard' if first_flow_fail else 'no Flow* local failure observed in this replay window'}.",
        f"Which earlier time should be targeted for width reduction? `{target.get('checkpoint_label', '')}` near t=`{target.get('checkpoint_t', '')}`.",
        "", "## Summary", "", "| checkpoint | path | Flow* mini | PyTorch mini | ratio | reset_width | failure |", "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(f"| {row.get('checkpoint_label', '')} | {row.get('path_id', '')} | {row.get('flowstar_mini_status', '')} | {row.get('pytorch_mini_status', '')} | {row.get('width_ratio_pytorch_over_flowstar', '')} | {row.get('reset_box_width_sum', '')} | {row.get('failure_reason', '')} |")
    (out_dir / "checkpoint_replay_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    segment_rows = _read_csv(source_dir / "normalized_insertion_h10_segments.csv") or _read_csv(source_dir / "rescue_segments.csv")
    reset_rows = _checkpoint_rows(source_dir, segment_rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    replay_segments: list[dict[str, Any]] = []
    for path_id, spec in RUN_SPECS.items():
        path_resets = [row for row in reset_rows if row.get("run_id") == spec["run_id"]]
        path_segments = [row for row in segment_rows if row.get("run_id") == spec["run_id"]]
        for checkpoint in CHECKPOINTS:
            if path_id == "o4_insert" and checkpoint > 6.45:
                continue
            if path_id == "o6_insert" and checkpoint > 7.45:
                continue
            reset = _nearest_reset_row(path_resets, checkpoint)
            if reset is None:
                continue
            segment_index = int(_finite_float(reset.get("segment_index")) or 0)
            h_try = min(max(_next_h(path_segments, segment_index, _finite_float(reset.get("h")) or 0.01), 1e-6), 0.1)
            box = _box_from_row(reset)
            wx, wy, wsum = _widths_from_box(box)
            label = f"{path_id}_t{checkpoint:g}".replace(".", "p")
            flow_one, flow_one_segments = _run_flowstar(out_dir, f"{label}_one", int(spec["order"]), h_try, h_try, reset, args.flowstar_root, args.timeout_s)
            flow_mini, flow_mini_segments = _run_flowstar(out_dir, f"{label}_mini", int(spec["order"]), h_try, min(float(args.mini_horizon), 0.1), reset, args.flowstar_root, args.timeout_s)
            py_one, py_one_segments = _pytorch_one_step(box, spec, h_try)
            py_mini, py_mini_segments = _pytorch_mini(box, spec, h_try, min(float(args.mini_horizon), 0.1))
            ratio = ""
            flow_width = _finite_float(flow_mini.get("width_sum"))
            py_width = _finite_float(py_mini.get("width_sum"))
            if flow_width is not None and py_width is not None and flow_width > 0:
                ratio = py_width / flow_width
            failure = py_mini.get("failure_reason") or flow_mini.get("failure_reason") or py_one.get("failure_reason") or flow_one.get("failure_reason")
            summary_rows.append(
                {
                    "checkpoint_label": label,
                    "checkpoint_t": checkpoint,
                    "path_id": path_id,
                    "run_id": spec["run_id"],
                    "order": spec["order"],
                    "candidate_order": spec.get("candidate_order") or "",
                    "source_segment_index": segment_index,
                    "source_t_hi": reset.get("t_hi", ""),
                    "h_try": h_try,
                    "reset_box_width_x": wx,
                    "reset_box_width_y": wy,
                    "reset_box_width_sum": wsum,
                    "flowstar_one_step_status": flow_one.get("status", ""),
                    "flowstar_one_step_completed": flow_one.get("completed", False),
                    "flowstar_one_step_width_sum": flow_one.get("width_sum", ""),
                    "pytorch_one_step_status": py_one.get("status", ""),
                    "pytorch_one_step_completed": py_one.get("completed", False),
                    "pytorch_one_step_width_sum": py_one.get("width_sum", ""),
                    "flowstar_mini_status": flow_mini.get("status", ""),
                    "flowstar_mini_completed": flow_mini.get("completed", False),
                    "flowstar_mini_width_sum": flow_mini.get("width_sum", ""),
                    "pytorch_mini_status": py_mini.get("status", ""),
                    "pytorch_mini_completed": py_mini.get("completed", False),
                    "pytorch_mini_width_sum": py_mini.get("width_sum", ""),
                    "width_ratio_pytorch_over_flowstar": ratio,
                    "step_rejection_count": int(py_one.get("step_rejections") or 0) + int(py_mini.get("step_rejections") or 0),
                    "residual_failure_dimension": py_one.get("residual_failure_dimension", ""),
                    "residual_failure_width_x": py_one.get("residual_failure_width_x", ""),
                    "residual_failure_width_y": py_one.get("residual_failure_width_y", ""),
                    "residual_failure_width_sum": py_one.get("residual_failure_width_sum", ""),
                    "failure_reason": failure,
                }
            )
            replay_segments.extend(_decorate_segments(flow_one_segments, label=label, checkpoint=checkpoint, path_id=path_id, engine="flowstar", horizon_kind="one_step", status=flow_one.get("status", "")))
            replay_segments.extend(_decorate_segments(flow_mini_segments, label=label, checkpoint=checkpoint, path_id=path_id, engine="flowstar", horizon_kind="mini_horizon", status=flow_mini.get("status", "")))
            replay_segments.extend(_decorate_segments(py_one_segments, label=label, checkpoint=checkpoint, path_id=path_id, engine="pytorch", horizon_kind="one_step", status=py_one.get("status", "")))
            replay_segments.extend(_decorate_segments(py_mini_segments, label=label, checkpoint=checkpoint, path_id=path_id, engine="pytorch", horizon_kind="mini_horizon", status=py_mini.get("status", "")))
    _write_csv(out_dir / "checkpoint_replay_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(out_dir / "checkpoint_replay_segments.csv", SEGMENT_FIELDS, replay_segments)
    _write_report(out_dir, summary_rows)
    _make_plot(out_dir, summary_rows)
    print(f"wrote checkpoint replay to {out_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=REPO_ROOT / "outputs" / "flowstar_normalized_insertion_h10")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "flowstar_checkpoint_replay")
    parser.add_argument("--flowstar-root", default=None)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--mini-horizon", type=float, default=0.1)
    args = parser.parse_args(argv)
    start = time.perf_counter()
    run(args)
    print(f"runtime_s={time.perf_counter() - start:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
