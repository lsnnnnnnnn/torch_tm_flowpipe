#!/usr/bin/env python3
"""Generate structured Van der Pol trajectory audit data and plots."""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
FLOWSTAR_DIR = REPO_ROOT / "comparisons" / "flowstar"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(FLOWSTAR_DIR) not in sys.path:
    sys.path.insert(0, str(FLOWSTAR_DIR))

import torch

torch.set_num_threads(1)

from torch_tm_flowpipe import Interval, flowpipe_multi_step
from torch_tm_flowpipe.ode_examples import van_der_pol_ode

from export_flowstar_model import export_model, load_config
from run_flowstar import find_flowstar_root, run_flowstar_toolbox

CONFIG_PATH = FLOWSTAR_DIR / "configs" / "van_der_pol.yaml"
STATE_VARS = ("x", "y")
INITIAL_BOX = {"x": (1.1, 1.4), "y": (2.35, 2.45)}
DEFAULT_TORCH_H = 0.01
DEFAULT_TORCH_STEPS = 10
DEFAULT_TORCH_ORDERS = tuple(range(2, 9))
TORCH_MODES = ("range_only", "dependency_preserving")

FLOWSTAR_SUMMARY_FIELDS = [
    "case_id",
    "system",
    "h",
    "steps",
    "horizon",
    "order",
    "setting_label",
    "remainder_estimation",
    "cutoff",
    "status",
    "failure_reason",
    "endpoint_box_available",
    "endpoint_source",
    "endpoint_width_x",
    "endpoint_width_y",
    "endpoint_width_sum",
    "last_segment_width_x",
    "last_segment_width_y",
    "last_segment_width_sum",
    "tube_width_x",
    "tube_width_y",
    "tube_width_sum",
    "flowstar_internal_reach_s",
    "flowstar_wall_compile_s",
    "flowstar_wall_run_s",
    "flowstar_wall_total_s",
    "num_segments",
    "box_source",
    "stdout_path",
    "stderr_path",
    "model_path",
    "plot_paths",
]

SEGMENT_FIELDS = [
    "case_id",
    "segment_index",
    "t_lo",
    "t_hi",
    "x_lo",
    "x_hi",
    "y_lo",
    "y_hi",
    "width_x",
    "width_y",
    "width_sum",
    "box_source",
]

TORCH_SUMMARY_FIELDS = [
    "case_id",
    "system",
    "mode",
    "h",
    "steps",
    "horizon",
    "requested_order",
    "status",
    "endpoint_box_available",
    "endpoint_width_x",
    "endpoint_width_y",
    "endpoint_width_sum",
    "last_segment_width_x",
    "last_segment_width_y",
    "last_segment_width_sum",
    "tube_width_x",
    "tube_width_y",
    "tube_width_sum",
    "runtime_s",
    "num_segments",
    "max_final_degree",
    "degree_by_dim",
    "term_count_total",
    "remainder_width_x",
    "remainder_width_y",
    "remainder_width_sum",
    "poly_range_width_x",
    "poly_range_width_y",
    "poly_range_width_sum",
    "validation_attempts",
    "containment_failures",
    "box_source",
]

SAMPLE_FIELDS = [
    "case_id",
    "sample_id",
    "sample_kind",
    "integrator",
    "diagnostic_only",
    "initial_x",
    "initial_y",
    "step_index",
    "t",
    "x",
    "y",
]

OVERLAY_FIELDS = [
    "case_id",
    "system",
    "h",
    "steps",
    "horizon",
    "order",
    "setting_label",
    "torch_mode",
    "flowstar_status",
    "torch_status",
    "last_segment_ratio_available",
    "last_segment_width_ratio_torch_over_flowstar",
    "tube_ratio_available",
    "tube_width_ratio_torch_over_flowstar",
    "endpoint_ratio_available",
    "endpoint_width_ratio_torch_over_flowstar",
    "ratio_note",
]

CROSSCHECK_FIELDS = [
    "comparison_id",
    "category",
    "metric",
    "new_source",
    "old_source",
    "case_id",
    "system",
    "mode",
    "h",
    "steps",
    "order",
    "setting_label",
    "new_value",
    "old_value",
    "abs_diff",
    "rel_diff",
    "tolerance",
    "pass_fail",
    "note",
]

_RUNTIME_RE = re.compile(r"FLOWSTAR_RUNTIME_S\s+(?P<runtime>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
_PLOT_RE = re.compile(r"FLOWSTAR_PLOT\s+(?P<stem>\S+)\s+(?P<xvar>\S+)\s+(?P<yvar>\S+)")
_NUMERIC_PAIR = re.compile(r"^\s*(?P<t>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+(?P<v>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(?:\s|$)")


@dataclass(frozen=True)
class FlowstarCase:
    setting_label: str
    h: float
    steps: int
    order: int
    remainder_estimation: float
    cutoff: float
    expected: str

    @property
    def case_id(self) -> str:
        return f"flowstar_{self.setting_label}_{h_tag(self.h)}_s{self.steps}_o{self.order}"


@dataclass(frozen=True)
class TorchCase:
    h: float
    steps: int
    order: int
    mode: str

    @property
    def case_id(self) -> str:
        return f"torch_{self.mode}_{h_tag(self.h)}_s{self.steps}_o{self.order}"

    @property
    def group_id(self) -> str:
        return f"torch_modes_{h_tag(self.h)}_s{self.steps}_o{self.order}"


def representative_flowstar_cases() -> list[FlowstarCase]:
    return [
        FlowstarCase("rem1e-4_cut1e-10", 0.01, 10, 4, 1.0e-4, 1.0e-10, "completed"),
        FlowstarCase("rem1e-4_cut1e-10", 0.01, 10, 2, 1.0e-4, 1.0e-10, "failed"),
        FlowstarCase("rem1e-10_cut1e-15", 0.0025, 10, 8, 1.0e-10, 1.0e-15, "completed"),
    ]


def h_tag(h: float) -> str:
    text = f"{float(h):g}".replace("-", "m").replace(".", "p")
    return f"h{text}"


def _fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.17g}"
        return ""
    return value


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in fields})


def _width(iv: Interval) -> float:
    return float(iv.width().detach().cpu())


def _interval_tuple(iv: Interval) -> tuple[float, float]:
    return (float(iv.lo.detach().cpu()), float(iv.hi.detach().cpu()))


def _interval_width_pair(lo: float, hi: float) -> float:
    return max(0.0, float(hi) - float(lo))


def _hull_pairs(pairs: Iterable[tuple[float, float]]) -> tuple[float, float] | None:
    vals = list(pairs)
    if not vals:
        return None
    return (min(lo for lo, _hi in vals), max(hi for _lo, hi in vals))


def _initial_box_intervals() -> list[Interval]:
    return [Interval(*INITIAL_BOX["x"]), Interval(*INITIAL_BOX["y"])]


def _rk4_step(state: tuple[float, float], dt: float) -> tuple[float, float]:
    def f(s: tuple[float, float]) -> tuple[float, float]:
        x, y = s
        return (y, y - x - x * x * y)

    x, y = state
    k1 = f((x, y))
    k2 = f((x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1]))
    k3 = f((x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1]))
    k4 = f((x + dt * k3[0], y + dt * k3[1]))
    return (
        x + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        y + dt * (k1[1] + 2.0 * k2[1] + k3[1] * 2.0 + k4[1]) / 6.0,
    )


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if n == 1:
        return [(lo + hi) / 2.0]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def initial_sample_points() -> list[tuple[str, str, float, float]]:
    x_lo, x_hi = INITIAL_BOX["x"]
    y_lo, y_hi = INITIAL_BOX["y"]
    samples: list[tuple[str, str, float, float]] = [
        ("corner_00", "corner", x_lo, y_lo),
        ("corner_01", "corner", x_lo, y_hi),
        ("corner_10", "corner", x_hi, y_lo),
        ("corner_11", "corner", x_hi, y_hi),
        ("center", "center", (x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0),
    ]
    xs = _linspace(x_lo, x_hi, 5)
    ys = _linspace(y_lo, y_hi, 5)
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            samples.append((f"grid_{ix}_{iy}", "grid5x5", x, y))
    return samples


def sample_trajectories(case_id: str, h: float, steps: int, *, rk4_substeps: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id, sample_kind, x0, y0 in initial_sample_points():
        state = (float(x0), float(y0))
        rows.append({
            "case_id": case_id,
            "sample_id": sample_id,
            "sample_kind": sample_kind,
            "integrator": "rk4",
            "diagnostic_only": True,
            "initial_x": x0,
            "initial_y": y0,
            "step_index": 0,
            "t": 0.0,
            "x": state[0],
            "y": state[1],
        })
        dt = float(h) / max(1, int(rk4_substeps))
        for step in range(1, int(steps) + 1):
            for _ in range(max(1, int(rk4_substeps))):
                state = _rk4_step(state, dt)
            rows.append({
                "case_id": case_id,
                "sample_id": sample_id,
                "sample_kind": sample_kind,
                "integrator": "rk4",
                "diagnostic_only": True,
                "initial_x": x0,
                "initial_y": y0,
                "step_index": step,
                "t": float(h) * step,
                "x": state[0],
                "y": state[1],
            })
    return rows


def _sample_final_points(h: float, steps: int) -> list[tuple[float, float]]:
    samples = sample_trajectories("_samples", h, steps)
    return [(float(r["x"]), float(r["y"])) for r in samples if int(r["step_index"]) == int(steps)]


def _containment_failures(endpoint_box: Sequence[Interval], h: float, steps: int) -> int:
    failures = 0
    for x, y in _sample_final_points(h, steps):
        if not endpoint_box[0].contains(x, tol=1e-8):
            failures += 1
        if not endpoint_box[1].contains(y, tol=1e-8):
            failures += 1
    return failures


def _poly_range_width(model: Any) -> float:
    return _width(model.polynomial.evaluate_interval(model.domain))


def torch_case_grid(flowstar_cases: Sequence[FlowstarCase]) -> list[TorchCase]:
    keys = {(DEFAULT_TORCH_H, DEFAULT_TORCH_STEPS, order) for order in DEFAULT_TORCH_ORDERS}
    for case in flowstar_cases:
        keys.add((case.h, case.steps, case.order))
    cases: list[TorchCase] = []
    for h, steps, order in sorted(keys):
        for mode in TORCH_MODES:
            cases.append(TorchCase(float(h), int(steps), int(order), mode))
    return cases


def run_torch_case(case: TorchCase, out_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    start = time.perf_counter()
    result = flowpipe_multi_step(
        van_der_pol_ode,
        _initial_box_intervals(),
        h=case.h,
        steps=case.steps,
        order=case.order,
        mode=case.mode,
    )
    runtime_s = time.perf_counter() - start

    segment_rows: list[dict[str, Any]] = []
    for i, seg in enumerate(result.segments):
        box = seg.tm.range_box()
        x_lo, x_hi = _interval_tuple(box[0])
        y_lo, y_hi = _interval_tuple(box[1])
        wx = _interval_width_pair(x_lo, x_hi)
        wy = _interval_width_pair(y_lo, y_hi)
        segment_rows.append({
            "case_id": case.case_id,
            "segment_index": i,
            "t_lo": i * case.h,
            "t_hi": (i + 1) * case.h,
            "x_lo": x_lo,
            "x_hi": x_hi,
            "y_lo": y_lo,
            "y_hi": y_hi,
            "width_x": wx,
            "width_y": wy,
            "width_sum": wx + wy,
            "box_source": "torch_tm_segment_range",
        })

    endpoint_box = result.final_tm.range_box()
    endpoint_widths = [_width(endpoint_box[0]), _width(endpoint_box[1])]
    last_row = segment_rows[-1]
    tube_x = _hull_pairs((float(r["x_lo"]), float(r["x_hi"])) for r in segment_rows)
    tube_y = _hull_pairs((float(r["y_lo"]), float(r["y_hi"])) for r in segment_rows)
    if tube_x is None or tube_y is None:
        raise ValueError("torch flowpipe produced no segments")
    degrees = [model.polynomial.degree() for model in result.final_tm.models]
    term_counts = [len(model.polynomial.terms) for model in result.final_tm.models]
    rem_widths = [_width(model.remainder) for model in result.final_tm.models]
    poly_widths = [_poly_range_width(model) for model in result.final_tm.models]
    summary = {
        "case_id": case.case_id,
        "system": "van_der_pol",
        "mode": case.mode,
        "h": case.h,
        "steps": case.steps,
        "horizon": case.h * case.steps,
        "requested_order": case.order,
        "status": result.status,
        "endpoint_box_available": True,
        "endpoint_width_x": endpoint_widths[0],
        "endpoint_width_y": endpoint_widths[1],
        "endpoint_width_sum": sum(endpoint_widths),
        "last_segment_width_x": last_row["width_x"],
        "last_segment_width_y": last_row["width_y"],
        "last_segment_width_sum": last_row["width_sum"],
        "tube_width_x": _interval_width_pair(*tube_x),
        "tube_width_y": _interval_width_pair(*tube_y),
        "tube_width_sum": _interval_width_pair(*tube_x) + _interval_width_pair(*tube_y),
        "runtime_s": runtime_s,
        "num_segments": len(segment_rows),
        "max_final_degree": max(degrees) if degrees else 0,
        "degree_by_dim": repr(degrees),
        "term_count_total": sum(term_counts),
        "remainder_width_x": rem_widths[0],
        "remainder_width_y": rem_widths[1],
        "remainder_width_sum": sum(rem_widths),
        "poly_range_width_x": poly_widths[0],
        "poly_range_width_y": poly_widths[1],
        "poly_range_width_sum": sum(poly_widths),
        "validation_attempts": result.validation_attempts,
        "containment_failures": _containment_failures(endpoint_box, case.h, case.steps),
        "box_source": "torch_tm_segment_range_and_final_time_tm",
    }
    sample_rows = sample_trajectories(case.case_id, case.h, case.steps)
    _write_csv(out_dir / "torch_segments" / f"{case.case_id}_segments.csv", SEGMENT_FIELDS, segment_rows)
    _write_csv(out_dir / "samples" / f"{case.case_id}_samples.csv", SAMPLE_FIELDS, sample_rows)
    return summary, segment_rows, sample_rows


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _flowstar_internal_runtime(stdout_path: Path | None) -> float | None:
    text = _read_text(stdout_path)
    matches = list(_RUNTIME_RE.finditer(text))
    if not matches:
        return None
    return float(matches[-1].group("runtime"))


def _flowstar_declared_failure(stdout_path: Path | None, stderr_path: Path | None) -> str:
    text = _read_text(stdout_path) + "\n" + _read_text(stderr_path)
    if "FLOWSTAR_COMPLETED 0" not in text:
        return ""
    for line in text.splitlines():
        lowered = line.lower()
        if "terminated" in lowered or "failed" in lowered:
            return line.strip()
    return "Flow* reported FLOWSTAR_COMPLETED 0"


def _flowstar_plot_paths(stdout_path: Path | None, model_dir: Path) -> dict[str, Path]:
    text = _read_text(stdout_path)
    paths: dict[str, Path] = {}
    for match in _PLOT_RE.finditer(text):
        var = match.group("yvar")
        paths[var] = model_dir / f"{match.group('stem')}.plt"
    return paths


def _parse_gnuplot_segments(path: Path) -> list[tuple[float, float, float, float]]:
    if not path.exists():
        return []
    blocks: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    in_data = False

    def flush() -> None:
        nonlocal current
        if current:
            blocks.append(current)
            current = []

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("plot "):
            in_data = True
            continue
        if not in_data:
            continue
        if not line:
            flush()
            continue
        if line == "e":
            flush()
            break
        match = _NUMERIC_PAIR.match(line)
        if not match:
            flush()
            continue
        current.append((float(match.group("t")), float(match.group("v"))))
    flush()

    segments: list[tuple[float, float, float, float]] = []
    for block in blocks:
        if not block:
            continue
        ts = [p[0] for p in block]
        vals = [p[1] for p in block]
        segments.append((min(ts), max(ts), min(vals), max(vals)))
    return segments


def _combine_flowstar_segments(case_id: str, x_path: Path | None, y_path: Path | None) -> list[dict[str, Any]]:
    x_segments = _parse_gnuplot_segments(x_path) if x_path is not None else []
    y_segments = _parse_gnuplot_segments(y_path) if y_path is not None else []
    n = min(len(x_segments), len(y_segments))
    rows: list[dict[str, Any]] = []
    for i in range(n):
        x_t_lo, x_t_hi, x_lo, x_hi = x_segments[i]
        y_t_lo, y_t_hi, y_lo, y_hi = y_segments[i]
        t_lo = min(x_t_lo, y_t_lo)
        t_hi = max(x_t_hi, y_t_hi)
        wx = _interval_width_pair(x_lo, x_hi)
        wy = _interval_width_pair(y_lo, y_hi)
        rows.append({
            "case_id": case_id,
            "segment_index": i,
            "t_lo": t_lo,
            "t_hi": t_hi,
            "x_lo": x_lo,
            "x_hi": x_hi,
            "y_lo": y_lo,
            "y_hi": y_hi,
            "width_x": wx,
            "width_y": wy,
            "width_sum": wx + wy,
            "box_source": "flowstar_gnuplot_segment",
        })
    return rows


def run_flowstar_case(case: FlowstarCase, cfg_path: Path, out_dir: Path, flowstar_root: str | None, timeout_s: float | None, build_lib: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_dir = out_dir / "flowstar_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{case.case_id}.cpp"
    export_model(
        cfg_path,
        model_path,
        h=case.h,
        steps=case.steps,
        order=case.order,
        plot_output_name=case.case_id,
        target="toolbox_cpp",
        remainder_radius=case.remainder_estimation,
        cutoff=case.cutoff,
    )

    root = find_flowstar_root(flowstar_root)
    if root is None:
        run_status = "skipped"
        compile_s: float | str = ""
        run_s: float | str = ""
        total_s: float | str = ""
        stdout_path = model_dir / f"{model_path.stem}.stdout.txt"
        stderr_path = model_dir / f"{model_path.stem}.stderr.txt"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("Flow* toolbox root not found; set FLOWSTAR_ROOT.\n", encoding="utf-8")
        failure_reason = "Flow* toolbox root not found; set FLOWSTAR_ROOT"
        plot_paths: dict[str, Path] = {}
    else:
        run = run_flowstar_toolbox(
            model_path,
            flowstar_root=root,
            output_dir=model_dir,
            timeout_s=timeout_s,
            build_lib=build_lib,
        )
        run_status = run.status
        compile_s = run.compile_s
        run_s = run.run_s
        total_s = run.runtime_s
        stdout_path = run.stdout_path
        stderr_path = run.stderr_path
        failure_reason = run.message
        declared_failure = _flowstar_declared_failure(stdout_path, stderr_path)
        if declared_failure:
            run_status = "failed"
            failure_reason = declared_failure
        plot_paths = _flowstar_plot_paths(stdout_path, model_dir)

    segment_rows = _combine_flowstar_segments(case.case_id, plot_paths.get("x"), plot_paths.get("y"))
    _write_csv(out_dir / "flowstar_segments" / f"{case.case_id}_segments.csv", SEGMENT_FIELDS, segment_rows)

    tube_x = _hull_pairs((float(r["x_lo"]), float(r["x_hi"])) for r in segment_rows)
    tube_y = _hull_pairs((float(r["y_lo"]), float(r["y_hi"])) for r in segment_rows)
    last = segment_rows[-1] if segment_rows else None
    internal_runtime = _flowstar_internal_runtime(stdout_path)
    if run_status == "completed" and not segment_rows:
        run_status = "unparsed"
        failure_reason = "no Flow* GNUPLOT segment boxes parsed"
    summary = {
        "case_id": case.case_id,
        "system": "van_der_pol",
        "h": case.h,
        "steps": case.steps,
        "horizon": case.h * case.steps,
        "order": case.order,
        "setting_label": case.setting_label,
        "remainder_estimation": case.remainder_estimation,
        "cutoff": case.cutoff,
        "status": run_status,
        "failure_reason": failure_reason,
        "endpoint_box_available": False,
        "endpoint_source": "",
        "endpoint_width_x": "",
        "endpoint_width_y": "",
        "endpoint_width_sum": "",
        "last_segment_width_x": last["width_x"] if last else "",
        "last_segment_width_y": last["width_y"] if last else "",
        "last_segment_width_sum": last["width_sum"] if last else "",
        "tube_width_x": _interval_width_pair(*tube_x) if tube_x else "",
        "tube_width_y": _interval_width_pair(*tube_y) if tube_y else "",
        "tube_width_sum": (_interval_width_pair(*tube_x) + _interval_width_pair(*tube_y)) if tube_x and tube_y else "",
        "flowstar_internal_reach_s": internal_runtime,
        "flowstar_wall_compile_s": compile_s,
        "flowstar_wall_run_s": run_s,
        "flowstar_wall_total_s": total_s,
        "num_segments": len(segment_rows),
        "box_source": "flowstar_gnuplot_segment_boxes" if segment_rows else "flowstar_no_segment_boxes",
        "stdout_path": str(stdout_path) if stdout_path else "",
        "stderr_path": str(stderr_path) if stderr_path else "",
        "model_path": str(model_path),
        "plot_paths": ";".join(str(p) for _var, p in sorted(plot_paths.items())),
    }
    return summary, segment_rows


def _rows_to_float(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [float(r[key]) for r in rows]


def _case_samples_by_id(sample_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in sample_rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    return grouped


def _add_initial_box(ax: Any, *, label: str = "initial box") -> None:
    import matplotlib.patches as patches

    x_lo, x_hi = INITIAL_BOX["x"]
    y_lo, y_hi = INITIAL_BOX["y"]
    ax.add_patch(patches.Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo, fill=False, edgecolor="black", linewidth=1.4, linestyle="--", label=label))


def _add_phase_boxes(ax: Any, rows: Sequence[Mapping[str, Any]], *, edgecolor: str, label: str, alpha: float = 0.18, linewidth: float = 0.9) -> None:
    import matplotlib.patches as patches

    for i, row in enumerate(rows):
        rect = patches.Rectangle(
            (float(row["x_lo"]), float(row["y_lo"])),
            float(row["width_x"]),
            float(row["width_y"]),
            fill=True,
            facecolor=edgecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            label=label if i == 0 else None,
        )
        ax.add_patch(rect)
    if rows:
        last = rows[-1]
        ax.add_patch(patches.Rectangle(
            (float(last["x_lo"]), float(last["y_lo"])),
            float(last["width_x"]),
            float(last["width_y"]),
            fill=False,
            edgecolor=edgecolor,
            linewidth=2.2,
            label=f"{label} last segment",
        ))


def _add_tx_boxes(ax: Any, rows: Sequence[Mapping[str, Any]], var: str, *, edgecolor: str, label: str, alpha: float = 0.18, linewidth: float = 0.9) -> None:
    import matplotlib.patches as patches

    lo_key = f"{var}_lo"
    width_key = f"width_{var}"
    for i, row in enumerate(rows):
        rect = patches.Rectangle(
            (float(row["t_lo"]), float(row[lo_key])),
            float(row["t_hi"]) - float(row["t_lo"]),
            float(row[width_key]),
            fill=True,
            facecolor=edgecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            label=label if i == 0 else None,
        )
        ax.add_patch(rect)
    if rows:
        last = rows[-1]
        ax.add_patch(patches.Rectangle(
            (float(last["t_lo"]), float(last[lo_key])),
            float(last["t_hi"]) - float(last["t_lo"]),
            float(last[width_key]),
            fill=False,
            edgecolor=edgecolor,
            linewidth=2.2,
            label=f"{label} last segment",
        ))


def _add_sample_lines(ax: Any, sample_rows: Sequence[Mapping[str, Any]], *, x_key: str, y_key: str, label: str) -> None:
    grouped = _case_samples_by_id(sample_rows)
    for i, rows in enumerate(grouped.values()):
        ordered = sorted(rows, key=lambda r: int(r["step_index"]))
        ax.plot(
            [float(r[x_key]) for r in ordered],
            [float(r[y_key]) for r in ordered],
            color="black",
            alpha=0.18,
            linewidth=0.8,
            label=label if i == 0 else None,
        )


def _finish_axes(fig: Any, ax: Any, path: Path) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    import matplotlib.pyplot as plt

    plt.close(fig)


def make_torch_plots(case: TorchCase, summary: Mapping[str, Any], segments: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]], figures_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    title_base = f"van_der_pol h={case.h:g} steps={case.steps} order={case.order} mode={case.mode}"

    fig, ax = plt.subplots(figsize=(7, 5))
    _add_phase_boxes(ax, segments, edgecolor="#1f77b4", label="PyTorch source = TM segment range")
    _add_sample_lines(ax, samples, x_key="x", y_key="y", label="RK4 sample trajectories")
    _add_initial_box(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title_base)
    path = figures_dir / f"{case.case_id}_torch_phase_xy.png"
    _finish_axes(fig, ax, path)
    paths.append(path)

    for var in ("x", "y"):
        fig, ax = plt.subplots(figsize=(7, 5))
        _add_tx_boxes(ax, segments, var, edgecolor="#1f77b4", label="PyTorch source = TM segment range")
        _add_sample_lines(ax, samples, x_key="t", y_key=var, label="RK4 sample trajectories")
        ax.set_xlabel("t")
        ax.set_ylabel(var)
        ax.set_title(title_base)
        path = figures_dir / f"{case.case_id}_torch_t_{var}.png"
        _finish_axes(fig, ax, path)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 5))
    t_mid = [(float(r["t_lo"]) + float(r["t_hi"])) / 2.0 for r in segments]
    ax.plot(t_mid, _rows_to_float(segments, "width_x"), marker="o", label="width x")
    ax.plot(t_mid, _rows_to_float(segments, "width_y"), marker="o", label="width y")
    ax.plot(t_mid, _rows_to_float(segments, "width_sum"), marker="o", label="width sum")
    ax.scatter([t_mid[-1]], [float(segments[-1]["width_sum"])], s=60, facecolor="none", edgecolor="black", label="last segment highlight")
    ax.set_xlabel("t midpoint")
    ax.set_ylabel("segment width")
    ax.set_title(title_base)
    path = figures_dir / f"{case.case_id}_torch_width_over_time.png"
    _finish_axes(fig, ax, path)
    paths.append(path)
    return paths


def make_torch_modes_overlay(group_case_id: str, h: float, steps: int, order: int, mode_data: Mapping[str, tuple[Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]], figures_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"range_only": "#1f77b4", "dependency_preserving": "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 5))
    sample_rows: Sequence[Mapping[str, Any]] = []
    for mode, (_summary, segments, samples) in sorted(mode_data.items()):
        sample_rows = samples
        _add_phase_boxes(ax, segments, edgecolor=colors[mode], label=f"PyTorch {mode} source = TM segment range", alpha=0.12)
    _add_sample_lines(ax, sample_rows, x_key="x", y_key="y", label="RK4 sample trajectories")
    _add_initial_box(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"van_der_pol h={h:g} steps={steps} order={order} mode=range_only+dependency_preserving")
    path = figures_dir / f"{group_case_id}_torch_modes_overlay_phase_xy.png"
    _finish_axes(fig, ax, path)
    return path


def make_flowstar_overlay_plots(flow_summary: Mapping[str, Any], flow_segments: Sequence[Mapping[str, Any]], torch_matches: Mapping[str, tuple[Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]], figures_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    case_id = str(flow_summary["case_id"])
    h = float(flow_summary["h"])
    steps = int(flow_summary["steps"])
    order = int(flow_summary["order"])
    setting = str(flow_summary["setting_label"])
    title_base = f"van_der_pol h={h:g} steps={steps} order={order} setting={setting} mode=range_only+dependency_preserving"
    colors = {
        "flowstar": "#2ca02c",
        "range_only": "#1f77b4",
        "dependency_preserving": "#d62728",
    }
    sample_rows: Sequence[Mapping[str, Any]] = []
    if torch_matches:
        sample_rows = next(iter(torch_matches.values()))[2]

    fig, ax = plt.subplots(figsize=(7, 5))
    _add_phase_boxes(ax, flow_segments, edgecolor=colors["flowstar"], label="Flow* source = GNUPLOT segment", alpha=0.14)
    for mode, (_summary, segments, _samples) in sorted(torch_matches.items()):
        _add_phase_boxes(ax, segments, edgecolor=colors[mode], label=f"PyTorch {mode} source = TM segment range", alpha=0.10)
    _add_sample_lines(ax, sample_rows, x_key="x", y_key="y", label="RK4 sample trajectories")
    _add_initial_box(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title_base)
    path = figures_dir / f"{case_id}_overlay_phase_xy.png"
    _finish_axes(fig, ax, path)
    paths.append(path)

    for var in ("x", "y"):
        fig, ax = plt.subplots(figsize=(7, 5))
        _add_tx_boxes(ax, flow_segments, var, edgecolor=colors["flowstar"], label="Flow* source = GNUPLOT segment", alpha=0.14)
        for mode, (_summary, segments, _samples) in sorted(torch_matches.items()):
            _add_tx_boxes(ax, segments, var, edgecolor=colors[mode], label=f"PyTorch {mode} source = TM segment range", alpha=0.10)
        _add_sample_lines(ax, sample_rows, x_key="t", y_key=var, label="RK4 sample trajectories")
        ax.set_xlabel("t")
        ax.set_ylabel(var)
        ax.set_title(title_base)
        path = figures_dir / f"{case_id}_overlay_t_{var}.png"
        _finish_axes(fig, ax, path)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 5))
    t_mid = [(float(r["t_lo"]) + float(r["t_hi"])) / 2.0 for r in flow_segments]
    ax.plot(t_mid, _rows_to_float(flow_segments, "width_sum"), marker="o", color=colors["flowstar"], label="Flow* GNUPLOT segment width sum")
    for mode, (_summary, segments, _samples) in sorted(torch_matches.items()):
        t_mid_torch = [(float(r["t_lo"]) + float(r["t_hi"])) / 2.0 for r in segments]
        ax.plot(t_mid_torch, _rows_to_float(segments, "width_sum"), marker="o", color=colors[mode], label=f"PyTorch {mode} TM segment width sum")
    ax.scatter([t_mid[-1]], [float(flow_segments[-1]["width_sum"])], s=60, facecolor="none", edgecolor="black", label="last segment highlight")
    ax.set_xlabel("t midpoint")
    ax.set_ylabel("segment width sum")
    ax.set_title(title_base)
    path = figures_dir / f"{case_id}_overlay_width_over_time.png"
    _finish_axes(fig, ax, path)
    paths.append(path)
    return paths


def overlay_summary_rows(flow_rows: Sequence[Mapping[str, Any]], torch_data: Mapping[tuple[float, int, int, str], tuple[Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for flow in flow_rows:
        h = float(flow["h"])
        steps = int(flow["steps"])
        order = int(flow["order"])
        flow_last = _safe_float(flow.get("last_segment_width_sum"))
        flow_tube = _safe_float(flow.get("tube_width_sum"))
        for mode in TORCH_MODES:
            match = torch_data.get((h, steps, order, mode))
            if not match:
                continue
            torch_summary = match[0]
            torch_last = _safe_float(torch_summary.get("last_segment_width_sum"))
            torch_tube = _safe_float(torch_summary.get("tube_width_sum"))
            last_ratio = torch_last / flow_last if flow_last and torch_last is not None else None
            tube_ratio = torch_tube / flow_tube if flow_tube and torch_tube is not None else None
            rows.append({
                "case_id": flow["case_id"],
                "system": "van_der_pol",
                "h": h,
                "steps": steps,
                "horizon": h * steps,
                "order": order,
                "setting_label": flow["setting_label"],
                "torch_mode": mode,
                "flowstar_status": flow["status"],
                "torch_status": torch_summary["status"],
                "last_segment_ratio_available": last_ratio is not None,
                "last_segment_width_ratio_torch_over_flowstar": last_ratio,
                "tube_ratio_available": tube_ratio is not None,
                "tube_width_ratio_torch_over_flowstar": tube_ratio,
                "endpoint_ratio_available": False,
                "endpoint_width_ratio_torch_over_flowstar": "",
                "ratio_note": "endpoint ratio disabled because Flow* GNUPLOT boxes are segment boxes, not final-time endpoint boxes",
            })
    return rows


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None



def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _find_row(rows: Sequence[Mapping[str, str]], **criteria: Any) -> Mapping[str, str] | None:
    def matches(row: Mapping[str, str], key: str, expected: Any) -> bool:
        value = row.get(key, "")
        if isinstance(expected, float):
            parsed = _safe_float(value)
            return parsed is not None and math.isclose(parsed, expected, rel_tol=1e-12, abs_tol=1e-12)
        if isinstance(expected, int):
            parsed = _safe_float(value)
            return parsed is not None and int(round(parsed)) == expected
        return str(value) == str(expected)

    for row in rows:
        if all(matches(row, key, expected) for key, expected in criteria.items()):
            return row
    return None


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _compare_values(new_value: Any, old_value: Any, tolerance: float) -> tuple[bool, str, str]:
    new_text = _stringify_value(new_value)
    old_text = _stringify_value(old_value)
    if new_text == "" and old_text == "":
        return True, "", ""
    new_float = _safe_float(new_text)
    old_float = _safe_float(old_text)
    if new_float is not None and old_float is not None:
        diff = abs(new_float - old_float)
        denom = max(abs(new_float), abs(old_float), 1.0)
        rel = diff / denom
        return diff <= tolerance or rel <= tolerance, f"{diff:.17g}", f"{rel:.17g}"
    return new_text == old_text, "", ""


def _append_crosscheck_row(
    rows: list[dict[str, Any]],
    *,
    category: str,
    metric: str,
    new_source: str,
    old_source: str,
    case_id: str,
    context: Mapping[str, Any],
    new_value: Any,
    old_value: Any,
    tolerance: float = 1.0e-10,
    note: str = "",
) -> None:
    passed, abs_diff, rel_diff = _compare_values(new_value, old_value, tolerance)
    rows.append({
        "comparison_id": f"{category}:{case_id}:{metric}",
        "category": category,
        "metric": metric,
        "new_source": new_source,
        "old_source": old_source,
        "case_id": case_id,
        "system": context.get("system", "van_der_pol"),
        "mode": context.get("mode", ""),
        "h": context.get("h", ""),
        "steps": context.get("steps", ""),
        "order": context.get("order", ""),
        "setting_label": context.get("setting_label", ""),
        "new_value": new_value,
        "old_value": old_value,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "tolerance": tolerance,
        "pass_fail": "pass" if passed else "fail",
        "note": note,
    })


def _append_missing_crosscheck_rows(
    rows: list[dict[str, Any]],
    *,
    category: str,
    metrics: Sequence[str],
    case_id: str,
    context: Mapping[str, Any],
    note: str,
) -> None:
    for metric in metrics:
        rows.append({
            "comparison_id": f"{category}:{case_id}:{metric}",
            "category": category,
            "metric": metric,
            "new_source": "outputs/trajectory_audit",
            "old_source": "outputs",
            "case_id": case_id,
            "system": context.get("system", "van_der_pol"),
            "mode": context.get("mode", ""),
            "h": context.get("h", ""),
            "steps": context.get("steps", ""),
            "order": context.get("order", ""),
            "setting_label": context.get("setting_label", ""),
            "new_value": "",
            "old_value": "",
            "abs_diff": "",
            "rel_diff": "",
            "tolerance": 1.0e-10,
            "pass_fail": "fail",
            "note": note,
        })


def write_crosscheck_outputs(out_dir: Path) -> list[dict[str, Any]]:
    torch_rows = _read_csv_rows(out_dir / "torch_structured_summary.csv")
    flow_rows = _read_csv_rows(out_dir / "flowstar_structured_summary.csv")
    diagnostic_rows = _read_csv_rows(REPO_ROOT / "outputs" / "van_der_pol_diagnostics_by_order_v2.csv")
    order_rows = _read_csv_rows(REPO_ROOT / "outputs" / "tm_order_audit_vdp_order2_8.csv")
    flow_sweep_rows = _read_csv_rows(REPO_ROOT / "outputs" / "flowstar_vdp_remainder_cutoff_sweep.csv")
    rows: list[dict[str, Any]] = []

    torch_metrics = [
        ("endpoint_width_sum", "final_width_sum", "outputs/van_der_pol_diagnostics_by_order_v2.csv"),
        ("remainder_width_sum", "remainder_width_sum", "outputs/van_der_pol_diagnostics_by_order_v2.csv"),
        ("poly_range_width_sum", "poly_range_width_sum", "outputs/van_der_pol_diagnostics_by_order_v2.csv"),
        ("max_final_degree", "max_final_degree", "outputs/van_der_pol_diagnostics_by_order_v2.csv"),
        ("term_count_total", "term_count_total", "outputs/tm_order_audit_vdp_order2_8.csv"),
    ]
    for order in DEFAULT_TORCH_ORDERS:
        for mode in TORCH_MODES:
            context = {"system": "van_der_pol", "mode": mode, "h": DEFAULT_TORCH_H, "steps": DEFAULT_TORCH_STEPS, "order": order}
            case_id = TorchCase(DEFAULT_TORCH_H, DEFAULT_TORCH_STEPS, order, mode).case_id
            new_row = _find_row(torch_rows, system="van_der_pol", mode=mode, h=DEFAULT_TORCH_H, steps=DEFAULT_TORCH_STEPS, requested_order=order)
            diagnostic_row = _find_row(diagnostic_rows, system="van_der_pol", mode=mode, h=DEFAULT_TORCH_H, steps=DEFAULT_TORCH_STEPS, requested_order=order)
            order_row = _find_row(order_rows, system="van_der_pol", mode=mode, h=DEFAULT_TORCH_H, steps=DEFAULT_TORCH_STEPS, requested_order=order)
            if new_row is None or diagnostic_row is None or order_row is None:
                _append_missing_crosscheck_rows(rows, category="torch_diagnostics", metrics=[m[0] for m in torch_metrics], case_id=case_id, context=context, note="missing new or old torch comparison row")
                continue
            for new_metric, old_metric, source in torch_metrics:
                old_row = order_row if old_metric == "term_count_total" else diagnostic_row
                metric_name = f"{new_metric}_vs_{old_metric}" if new_metric != old_metric else new_metric
                _append_crosscheck_row(
                    rows,
                    category="torch_diagnostics",
                    metric=metric_name,
                    new_source="outputs/trajectory_audit/torch_structured_summary.csv",
                    old_source=source,
                    case_id=case_id,
                    context=context,
                    new_value=new_row.get(new_metric, ""),
                    old_value=old_row.get(old_metric, ""),
                    note="runtime intentionally excluded",
                )

    flow_cases = [
        FlowstarCase("rem1e-4_cut1e-10", 0.01, 10, 4, 1.0e-4, 1.0e-10, "completed"),
        FlowstarCase("rem1e-4_cut1e-10", 0.01, 10, 2, 1.0e-4, 1.0e-10, "failed"),
        FlowstarCase("rem1e-10_cut1e-15", 0.0025, 10, 8, 1.0e-10, 1.0e-15, "completed"),
    ]
    flow_metrics = ["status", "failure_reason", "num_segments", "last_segment_width_sum", "tube_width_sum"]
    for case in flow_cases:
        context = {"system": "van_der_pol", "mode": "fixed", "h": case.h, "steps": case.steps, "order": case.order, "setting_label": case.setting_label}
        new_row = _find_row(flow_rows, system="van_der_pol", h=case.h, steps=case.steps, order=case.order, setting_label=case.setting_label)
        old_row = _find_row(flow_sweep_rows, system="van_der_pol", tool="flowstar", mode="fixed", h=case.h, steps=case.steps, order=case.order, setting_label=case.setting_label)
        if new_row is None or old_row is None:
            _append_missing_crosscheck_rows(rows, category="flowstar_sweep", metrics=flow_metrics, case_id=case.case_id, context=context, note="missing new or old Flow* comparison row")
            continue
        for metric in flow_metrics:
            if metric == "num_segments" and new_row.get("status") == "failed" and old_row.get("status") == "failed":
                new_segments = _safe_float(new_row.get("num_segments", ""))
                old_segments = _safe_float(old_row.get("num_segments", ""))
                semantic_pass = new_segments == 0 and old_segments == case.steps
                rows.append({
                    "comparison_id": f"flowstar_sweep:{case.case_id}:num_segments",
                    "category": "flowstar_sweep",
                    "metric": "num_segments",
                    "new_source": "outputs/trajectory_audit/flowstar_structured_summary.csv",
                    "old_source": "outputs/flowstar_vdp_remainder_cutoff_sweep.csv",
                    "case_id": case.case_id,
                    "system": context.get("system", "van_der_pol"),
                    "mode": context.get("mode", ""),
                    "h": context.get("h", ""),
                    "steps": context.get("steps", ""),
                    "order": context.get("order", ""),
                    "setting_label": context.get("setting_label", ""),
                    "new_value": new_row.get("num_segments", ""),
                    "old_value": old_row.get("num_segments", ""),
                    "abs_diff": abs((new_segments or 0.0) - (old_segments or 0.0)),
                    "rel_diff": "",
                    "tolerance": 1.0e-10,
                    "pass_fail": "pass" if semantic_pass else "fail",
                    "note": "failed case semantics: new num_segments is parsed GNUPLOT boxes; old sweep num_segments stores requested steps",
                })
                continue
            _append_crosscheck_row(
                rows,
                category="flowstar_sweep",
                metric=metric,
                new_source="outputs/trajectory_audit/flowstar_structured_summary.csv",
                old_source="outputs/flowstar_vdp_remainder_cutoff_sweep.csv",
                case_id=case.case_id,
                context=context,
                new_value=new_row.get(metric, ""),
                old_value=old_row.get(metric, ""),
                note="runtime intentionally excluded",
            )

    _write_csv(out_dir / "crosscheck_summary.csv", CROSSCHECK_FIELDS, rows)
    write_crosscheck_markdown(out_dir, rows)
    return rows


def write_crosscheck_markdown(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("pass_fail") == "pass")
    failed = total - passed
    torch_total = sum(1 for row in rows if row.get("category") == "torch_diagnostics")
    flow_total = sum(1 for row in rows if row.get("category") == "flowstar_sweep")
    failure_rows = [row for row in rows if row.get("pass_fail") != "pass"]
    lines = [
        "# Trajectory Audit Cross-check",
        "",
        "This file compares the generated trajectory audit CSVs against the older authoritative CSVs without rerunning the older experiments.",
        "Runtime columns are intentionally excluded from exact matching.",
        "",
        "## Sources",
        "",
        "- New PyTorch TM: `outputs/trajectory_audit/torch_structured_summary.csv`",
        "- New Flow*: `outputs/trajectory_audit/flowstar_structured_summary.csv`",
        "- Old PyTorch diagnostics: `outputs/van_der_pol_diagnostics_by_order_v2.csv`",
        "- Old TM order audit: `outputs/tm_order_audit_vdp_order2_8.csv`",
        "- Old Flow* sweep: `outputs/flowstar_vdp_remainder_cutoff_sweep.csv`",
        "",
        "## Result",
        "",
        f"- Overall: {passed}/{total} comparisons passed; {failed} failed.",
        f"- PyTorch diagnostics: {torch_total} comparisons for h=0.01, steps=10, orders 2..8, both modes.",
        f"- Flow* sweep: {flow_total} comparisons for the three representative fixed-step/fixed-order cases.",
        "",
    ]
    if failure_rows:
        lines.extend(["## Failures", "", "| comparison_id | new_value | old_value | note |", "|---|---:|---:|---|"])
        for row in failure_rows:
            lines.append(f"| `{row.get('comparison_id', '')}` | `{row.get('new_value', '')}` | `{row.get('old_value', '')}` | {row.get('note', '')} |")
    else:
        lines.extend(["All required comparisons passed.", ""])
    path = out_dir / "crosscheck_summary.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    return path


def _make_contact_sheet(image_specs: Sequence[tuple[Path, str]], out_path: Path, *, cols: int, thumb_size: tuple[int, int]) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = max(1, math.ceil(len(image_specs) / cols))
    margin = 18
    caption_h = 34
    cell_w = thumb_size[0] + margin * 2
    cell_h = thumb_size[1] + caption_h + margin * 2
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)

    for idx, (image_path, title) in enumerate(image_specs):
        row = idx // cols
        col = idx % cols
        x0 = col * cell_w + margin
        y0 = row * cell_h + margin
        if image_path.exists():
            with Image.open(image_path) as src:
                img = src.convert("RGB")
                img.thumbnail(thumb_size, resample)
        else:
            img = Image.new("RGB", thumb_size, "#f2f2f2")
            placeholder = ImageDraw.Draw(img)
            placeholder.text((12, 12), "missing", fill="#555555", font=font)
        px = x0 + (thumb_size[0] - img.size[0]) // 2
        py = y0 + caption_h + (thumb_size[1] - img.size[1]) // 2
        draw.text((x0, y0), title[:70], fill="#222222", font=font)
        sheet.paste(img, (px, py))
    sheet.save(out_path)
    return out_path


def make_contact_sheets(out_dir: Path) -> list[Path]:
    figures = out_dir / "figures"
    outputs: list[Path] = []

    torch_order_specs = [
        (figures / f"torch_modes_h0p01_s10_o{order}_torch_modes_overlay_phase_xy.png", f"Torch modes overlay phase, order {order}")
        for order in DEFAULT_TORCH_ORDERS
    ]
    outputs.append(_make_contact_sheet(torch_order_specs, figures / "contact_sheet_torch_orders.png", cols=4, thumb_size=(390, 285)))

    flowstar_specs = []
    for case_id, label in [
        ("flowstar_rem1e-4_cut1e-10_h0p01_s10_o4", "Flow* loose order 4"),
        ("flowstar_rem1e-10_cut1e-15_h0p0025_s10_o8", "Flow* strict order 8"),
    ]:
        for suffix, view in [
            ("overlay_phase_xy", "phase"),
            ("overlay_t_x", "t-x"),
            ("overlay_t_y", "t-y"),
            ("overlay_width_over_time", "width over time"),
        ]:
            flowstar_specs.append((figures / f"{case_id}_{suffix}.png", f"{label}: {view}"))
    outputs.append(_make_contact_sheet(flowstar_specs, figures / "contact_sheet_flowstar_overlays.png", cols=4, thumb_size=(390, 285)))

    width_specs = []
    for order in DEFAULT_TORCH_ORDERS:
        width_specs.append((figures / f"torch_range_only_h0p01_s10_o{order}_torch_width_over_time.png", f"range_only order {order}"))
        width_specs.append((figures / f"torch_dependency_preserving_h0p01_s10_o{order}_torch_width_over_time.png", f"dependency_preserving order {order}"))
    width_specs.extend([
        (figures / "flowstar_rem1e-4_cut1e-10_h0p01_s10_o4_overlay_width_over_time.png", "Flow* loose order 4 overlay"),
        (figures / "flowstar_rem1e-10_cut1e-15_h0p0025_s10_o8_overlay_width_over_time.png", "Flow* strict order 8 overlay"),
    ])
    outputs.append(_make_contact_sheet(width_specs, figures / "contact_sheet_width_trends.png", cols=4, thumb_size=(390, 285)))
    return outputs


def _fmt_report_value(value: Any) -> str:
    parsed = _safe_float(value)
    if parsed is not None:
        return f"{parsed:.6g}"
    text = _stringify_value(value)
    return text if text else "n/a"


def _relative_md_path(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def write_visual_audit_report(out_dir: Path, contact_sheets: Sequence[Path]) -> Path:
    torch_rows = _read_csv_rows(out_dir / "torch_structured_summary.csv")
    flow_rows = _read_csv_rows(out_dir / "flowstar_structured_summary.csv")
    overlay_rows = _read_csv_rows(out_dir / "flowstar_vs_torch_overlay_summary.csv")

    def torch_row(mode: str, order: int) -> Mapping[str, str]:
        row = _find_row(torch_rows, mode=mode, h=DEFAULT_TORCH_H, steps=DEFAULT_TORCH_STEPS, requested_order=order)
        if row is None:
            raise ValueError(f"missing torch row for {mode} order {order}")
        return row

    flow_o4 = _find_row(flow_rows, h=0.01, steps=10, order=4, setting_label="rem1e-4_cut1e-10")
    flow_o2 = _find_row(flow_rows, h=0.01, steps=10, order=2, setting_label="rem1e-4_cut1e-10")
    flow_o8 = _find_row(flow_rows, h=0.0025, steps=10, order=8, setting_label="rem1e-10_cut1e-15")
    if flow_o4 is None or flow_o2 is None or flow_o8 is None:
        raise ValueError("missing representative Flow* summary rows")

    range_o2 = torch_row("range_only", 2)
    range_o8 = torch_row("range_only", 8)
    dep_o2 = torch_row("dependency_preserving", 2)
    dep_o8 = torch_row("dependency_preserving", 8)
    overlay_o4_range = _find_row(overlay_rows, case_id=flow_o4["case_id"], torch_mode="range_only")
    overlay_o8_range = _find_row(overlay_rows, case_id=flow_o8["case_id"], torch_mode="range_only")

    contact_lines = [f"![{path.name}]({_relative_md_path(path, out_dir)})" for path in contact_sheets]
    figures = out_dir / "figures"
    key_figures = [
        figures / "torch_modes_h0p01_s10_o4_torch_modes_overlay_phase_xy.png",
        figures / "torch_range_only_h0p01_s10_o4_torch_t_x.png",
        figures / "torch_range_only_h0p01_s10_o4_torch_t_y.png",
        figures / "flowstar_rem1e-4_cut1e-10_h0p01_s10_o4_overlay_phase_xy.png",
        figures / "flowstar_rem1e-10_cut1e-15_h0p0025_s10_o8_overlay_phase_xy.png",
    ]
    key_lines = [f"![{path.name}]({_relative_md_path(path, out_dir)})" for path in key_figures]

    text = f"""# Trajectory Visual Audit Report

This report is generated from the existing trajectory audit CSV and PNG artifacts. It is a visual QA report for fixed-step/fixed-order Van der Pol trajectories, not a new reachability algorithm.

## Contact Sheets

{chr(10).join(contact_lines)}

## Visual QA Summary

- PyTorch order 2..8 phase trend: the mode overlay phase plots follow the same Van der Pol direction across orders. The range_only endpoint width sum drops from {_fmt_report_value(range_o2['endpoint_width_sum'])} at order 2 to {_fmt_report_value(range_o8['endpoint_width_sum'])} at order 8; dependency_preserving drops from {_fmt_report_value(dep_o2['endpoint_width_sum'])} to {_fmt_report_value(dep_o8['endpoint_width_sum'])}.
- range_only vs dependency_preserving overlay: the phase overlays share the same sampled trajectory trend. dependency_preserving keeps original-variable dependence and is usually wider in segment/tube views; range_only is narrower at higher orders but collapses dependency after each step.
- Flow* order 4 loose completed vs torch overlay: status={flow_o4['status']}; Flow* last segment width sum is {_fmt_report_value(flow_o4['last_segment_width_sum'])}. The range_only last-segment ratio torch/Flow* is {_fmt_report_value(overlay_o4_range.get('last_segment_width_ratio_torch_over_flowstar') if overlay_o4_range else '')}.
- Flow* order 8 strict completed vs torch overlay: status={flow_o8['status']}; Flow* last segment width sum is {_fmt_report_value(flow_o8['last_segment_width_sum'])}. The range_only last-segment ratio torch/Flow* is {_fmt_report_value(overlay_o8_range.get('last_segment_width_ratio_torch_over_flowstar') if overlay_o8_range else '')}.
- Flow* order 2 loose failed: status={flow_o2['status']}; failure_reason={flow_o2['failure_reason']}
- t-x, t-y, phase, and width-over-time views are included in the individual figures and in the contact sheets above.
- Sampling uses corners, center, and a 5x5 grid with RK4 trajectories. Sampling is diagnostic only, not proof.

## Key Figures

{chr(10).join(key_lines)}

## Scope Guard

This report is not CROWN, not auto_LiRPA, not a Jacobian-bound experiment, not sin/cos support, not hybrid automata, not a Flow* Python binding, not an NN controller workflow, and not a new algorithm.
"""
    path = out_dir / "visual_audit_report.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def generate_audit_reports(out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    write_readme(out_dir, flowstar_patched=False)
    contact_sheets = make_contact_sheets(out_dir)
    crosscheck_rows = write_crosscheck_outputs(out_dir)
    visual_report = write_visual_audit_report(out_dir, contact_sheets)
    return {
        "contact_sheets": contact_sheets,
        "crosscheck_rows": crosscheck_rows,
        "visual_report": visual_report,
    }

def write_readme(out_dir: Path, flowstar_patched: bool) -> None:
    patch_text = "No Flow* source patch was used; patch path, patch sha256, rebuild command, and patched libflowstar.a sha256 are not applicable."
    if flowstar_patched:
        patch_text = "A Flow* patch was used; see comparisons/flowstar/flowstar_patches/ for patch metadata."
    text = f"""# Trajectory Visual Audit

This directory is a structured-output and plotting audit for the plant-only Van der Pol benchmark. It is a trend/visual audit of existing fixed-step/fixed-order behavior, not a new algorithm.

## Scope

- System: `dx/dt = y`, `dy/dt = y - x - x^2*y`
- Initial set: `x in [1.1, 1.4]`, `y in [2.35, 2.45]`
- PyTorch TM modes: `range_only`, `dependency_preserving`
- Flow* backend: generated toolbox C++ linked against `$FLOWSTAR_ROOT/flowstar-toolbox/libflowstar.a`
- Flow* patch status: {patch_text}

## Semantics

- Flow* GNUPLOT boxes are flowpipe segment boxes. They are not final-time endpoint boxes.
- `endpoint_box_available=false` for Flow* rows in this audit, so endpoint widths are blank and endpoint ratios are disabled.
- `last_segment_width_*` is the width of the final flowpipe segment box.
- `tube_width_*` is the hull over all segment boxes.
- PyTorch `endpoint_width_*` comes from the final-time Taylor model range box; PyTorch segment rows come from Taylor-model segment ranges.
- Flow* internal runtime is parsed from `FLOWSTAR_RUNTIME_S`; compile, run, and total wall times are recorded separately.
- Samples are RK4 trajectories from corners, center, and a 5x5 grid. They are visual diagnostics only and are not proof.

## Files

- `flowstar_structured_summary.csv`
- `flowstar_segments/*_segments.csv`
- `torch_structured_summary.csv`
- `torch_segments/*_segments.csv`
- `samples/*_samples.csv`
- `figures/*.png`
- `figures/contact_sheet_*.png`
- `flowstar_vs_torch_overlay_summary.csv`
- `visual_audit_report.md`
- `crosscheck_summary.csv`
- `crosscheck_summary.md`

This audit is fixed-step/fixed-order. It is not Flow*_adaptive, not full CROWN-Reach, not CROWN, not auto_LiRPA, not a Jacobian-bound experiment, not sin/cos support, not hybrid automata, not a Flow* Python binding workflow, not an NN controller workflow, and not a new algorithm.
"""
    path = out_dir / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(CONFIG_PATH)
    if args.system != "van_der_pol":
        raise SystemExit("trajectory_visual_audit currently supports --system van_der_pol")

    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    flowstar_cases = representative_flowstar_cases()
    torch_cases = torch_case_grid(flowstar_cases)

    torch_rows: list[dict[str, Any]] = []
    torch_data: dict[tuple[float, int, int, str], tuple[Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]] = {}
    print(f"torch: {len(torch_cases)} case(s)")
    for case in torch_cases:
        summary, segments, samples = run_torch_case(case, out_dir)
        torch_rows.append(summary)
        torch_data[(case.h, case.steps, case.order, case.mode)] = (summary, segments, samples)
        make_torch_plots(case, summary, segments, samples, figures_dir)
        print(f"  {case.case_id}: status={summary['status']} endpoint_sum={float(summary['endpoint_width_sum']):.6g}")

    by_group: dict[tuple[float, int, int], dict[str, tuple[Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]]] = {}
    for (h, steps, order, mode), data in torch_data.items():
        by_group.setdefault((h, steps, order), {})[mode] = data
    for (h, steps, order), mode_data in sorted(by_group.items()):
        group_id = TorchCase(h, steps, order, "range_only").group_id
        make_torch_modes_overlay(group_id, h, steps, order, mode_data, figures_dir)

    flowstar_rows: list[dict[str, Any]] = []
    flowstar_segments_by_case: dict[str, list[dict[str, Any]]] = {}
    if args.skip_flowstar:
        print("flowstar: skipped by --skip-flowstar")
    print(f"flowstar: {len(flowstar_cases)} representative case(s)")
    for case in flowstar_cases:
        if args.skip_flowstar:
            summary = {
                "case_id": case.case_id,
                "system": "van_der_pol",
                "h": case.h,
                "steps": case.steps,
                "horizon": case.h * case.steps,
                "order": case.order,
                "setting_label": case.setting_label,
                "remainder_estimation": case.remainder_estimation,
                "cutoff": case.cutoff,
                "status": "skipped",
                "failure_reason": "skip_flowstar requested",
                "endpoint_box_available": False,
                "endpoint_source": "",
                "num_segments": 0,
                "box_source": "flowstar_not_run",
            }
            segments: list[dict[str, Any]] = []
            _write_csv(out_dir / "flowstar_segments" / f"{case.case_id}_segments.csv", SEGMENT_FIELDS, segments)
        else:
            summary, segments = run_flowstar_case(
                case,
                CONFIG_PATH,
                out_dir,
                args.flowstar_root,
                args.flowstar_timeout_s,
                not args.no_build_flowstar_lib,
            )
        flowstar_rows.append(summary)
        flowstar_segments_by_case[case.case_id] = segments
        print(f"  {case.case_id}: status={summary['status']} segments={summary.get('num_segments', 0)}")
        if summary.get("status") == "completed" and segments:
            matches = {
                mode: torch_data[(case.h, case.steps, case.order, mode)]
                for mode in TORCH_MODES
                if (case.h, case.steps, case.order, mode) in torch_data
            }
            make_flowstar_overlay_plots(summary, segments, matches, figures_dir)

    _write_csv(out_dir / "torch_structured_summary.csv", TORCH_SUMMARY_FIELDS, torch_rows)
    _write_csv(out_dir / "flowstar_structured_summary.csv", FLOWSTAR_SUMMARY_FIELDS, flowstar_rows)
    overlay_rows = overlay_summary_rows(flowstar_rows, torch_data)
    _write_csv(out_dir / "flowstar_vs_torch_overlay_summary.csv", OVERLAY_FIELDS, overlay_rows)
    report_result = generate_audit_reports(out_dir)
    return {
        "torch_rows": torch_rows,
        "flowstar_rows": flowstar_rows,
        "overlay_rows": overlay_rows,
        "crosscheck_rows": report_result["crosscheck_rows"],
        "contact_sheets": report_result["contact_sheets"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Flow* and PyTorch TM trajectory/trend visual audit for Van der Pol.")
    parser.add_argument("--system", default="van_der_pol")
    parser.add_argument("--out-dir", default="outputs/trajectory_audit")
    parser.add_argument("--flowstar-root", default=None)
    parser.add_argument("--flowstar-timeout-s", type=float, default=60.0)
    parser.add_argument("--no-build-flowstar-lib", action="store_true")
    parser.add_argument("--skip-flowstar", action="store_true")
    parser.add_argument("--reports-only", action="store_true", help="Generate contact sheets, visual report, and cross-checks from existing audit outputs.")
    args = parser.parse_args()
    if args.reports_only:
        if args.system != "van_der_pol":
            raise SystemExit("trajectory_visual_audit currently supports --system van_der_pol")
        result = generate_audit_reports(Path(args.out_dir))
        print(f"wrote {Path(args.out_dir) / 'visual_audit_report.md'}")
        print(f"wrote {Path(args.out_dir) / 'crosscheck_summary.csv'} ({len(result['crosscheck_rows'])} rows)")
        for path in result["contact_sheets"]:
            print(f"wrote {path}")
        return
    result = run_audit(args)
    print(f"wrote {Path(args.out_dir) / 'torch_structured_summary.csv'} ({len(result['torch_rows'])} rows)")
    print(f"wrote {Path(args.out_dir) / 'flowstar_structured_summary.csv'} ({len(result['flowstar_rows'])} rows)")
    print(f"wrote {Path(args.out_dir) / 'flowstar_vs_torch_overlay_summary.csv'} ({len(result['overlay_rows'])} rows)")
    print(f"wrote {Path(args.out_dir) / 'crosscheck_summary.csv'} ({len(result['crosscheck_rows'])} rows)")


if __name__ == "__main__":
    main()
