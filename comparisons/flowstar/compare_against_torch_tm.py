"""Plant-only comparison suite: torch_tm_flowpipe vs. Flow*.

This script deliberately compares isolated polynomial plant ODEs.  It does not
run CROWN, auto_LiRPA, NNCS controllers, Jacobian bounds, or Flow* bindings in
the core library.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# Allow direct execution from the repository root without requiring installation.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment guard.
    raise SystemExit("PyYAML is required for comparisons/flowstar configs") from exc

import torch

# Sparse scalar polynomial operations are latency-bound; one thread avoids
# repeated oversubscription in large comparison grids.
torch.set_num_threads(1)

from torch_tm_flowpipe import Interval, TMVector, flowpipe_multi_step
from torch_tm_flowpipe.ode_examples import harmonic_oscillator_ode, scalar_quadratic_ode, van_der_pol_ode

from export_flowstar_model import export_model, load_config
from parse_flowstar_output import parse_files, widths as parsed_widths
from run_flowstar import find_flowstar_executable, find_flowstar_root, run_flowstar_legacy_model, run_flowstar_toolbox
from summarize_comparison import generate_summary

CSV_FIELDS = [
    "system", "tool", "mode", "h", "steps", "order", "setting_label", "status",
    "endpoint_width_sum", "endpoint_width_max", "last_segment_width_sum", "last_segment_width_max",
    "tube_width_sum", "tube_width_max", "box_source", "endpoint_box_available",
    "last_segment_box_available", "tube_box_available", "final_width_sum", "final_width_max",
    "flowpipe_width_sum", "flowpipe_width_max", "runtime_s", "num_segments",
    "validation_attempts", "term_count", "actual_degree", "remainder_radius",
    "containment_failures", "flowstar_internal_reach_s", "flowstar_wall_compile_s",
    "flowstar_wall_run_s", "flowstar_wall_total_s", "flowstar_compile_s", "flowstar_run_s",
    "flowstar_model_path", "flowstar_stdout_path", "flowstar_stderr_path", "failure_reason",
]

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_CONFIGS = [
    CONFIG_DIR / "scalar_quadratic.yaml",
    CONFIG_DIR / "harmonic_oscillator.yaml",
    CONFIG_DIR / "van_der_pol.yaml",
    CONFIG_DIR / "affine_controlled.yaml",
]

_RUNTIME_RE = re.compile(r"FLOWSTAR_RUNTIME_S\s+(?P<runtime>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


def _flowstar_internal_runtime(run) -> float | str:
    if run.stdout_path is None or not Path(run.stdout_path).exists():
        return ""
    text = Path(run.stdout_path).read_text(encoding="utf-8", errors="ignore")
    matches = list(_RUNTIME_RE.finditer(text))
    if not matches:
        return ""
    return float(matches[-1].group("runtime"))


def _interval_box_from_config(cfg: Mapping[str, Any]) -> list[Interval]:
    init = cfg["initial"]
    return [Interval(init[var][0], init[var][1]) for var in cfg["state_vars"]]


def _metric_indices(cfg: Mapping[str, Any]) -> list[int]:
    state_vars = list(cfg["state_vars"])
    metric_vars = list(cfg.get("metric_vars", state_vars))
    return [state_vars.index(v) for v in metric_vars]


def _metric_vars(cfg: Mapping[str, Any]) -> list[str]:
    return list(cfg.get("metric_vars", cfg["state_vars"]))


def affine_controlled_folded_ode(x: TMVector, u: TMVector | None = None) -> TMVector:
    """dx/dt = 0.5*x + e, de/dt = 0 for e in [-0.01, 0.01]."""
    return TMVector([0.5 * x[0] + x[1], 0.0 * x[1]])


def ode_for_config(cfg: Mapping[str, Any]) -> Callable[..., TMVector]:
    name = cfg.get("torch_ode", cfg["system"])
    if name == "scalar_quadratic":
        return scalar_quadratic_ode
    if name == "harmonic_oscillator":
        return harmonic_oscillator_ode
    if name == "van_der_pol":
        return van_der_pol_ode
    if name == "affine_controlled_folded":
        return affine_controlled_folded_ode
    raise ValueError(f"unknown torch_ode: {name}")


def _tensor_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu())
    return float(x)


def _width(iv: Interval) -> float:
    return _tensor_float(iv.width())


def _radius(iv: Interval) -> float:
    return _tensor_float(iv.radius())


def _box_width_metrics(box: Sequence[Interval], indices: Sequence[int]) -> tuple[float, float]:
    vals = [_width(box[i]) for i in indices]
    if not vals:
        return 0.0, 0.0
    return float(sum(vals)), float(max(vals))


def _hull_interval(intervals: Iterable[Interval]) -> Interval | None:
    items = list(intervals)
    if not items:
        return None
    return Interval.hull(*items)


def _flowpipe_tube_box(result, n_state_vars: int) -> list[Interval]:
    hulls: list[Interval] = []
    for dim in range(n_state_vars):
        intervals = [seg.tm.range_box()[dim] for seg in result.segments]
        hull = _hull_interval(intervals)
        if hull is None:
            raise ValueError("empty segment list")
        hulls.append(hull)
    return hulls


def _term_count(result, indices: Sequence[int]) -> int:
    return int(sum(len(result.final_tm[i].polynomial.terms) for i in indices))


def _actual_degree(result, indices: Sequence[int]) -> int:
    if not indices:
        return 0
    return int(max(result.final_tm[i].polynomial.degree() for i in indices))


def _remainder_radius(result, indices: Sequence[int]) -> float:
    if not indices:
        return 0.0
    return float(max(_radius(result.final_tm[i].remainder) for i in indices))


def _contains_box(box: Mapping[str, tuple[float, float]], sample: Mapping[str, float], metric_vars: Sequence[str], *, tol: float) -> bool:
    for var in metric_vars:
        if var not in box or var not in sample:
            return False
        lo, hi = box[var]
        if sample[var] < lo - tol or sample[var] > hi + tol:
            return False
    return True


def _torch_contains_box(box: Sequence[Interval], sample_values: Sequence[float], indices: Sequence[int], *, tol: float) -> bool:
    for idx in indices:
        if not box[idx].contains(sample_values[idx], tol=tol):
            return False
    return True


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
        y + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
    )


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if n == 1:
        return [(lo + hi) / 2.0]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def sample_final_states(cfg: Mapping[str, Any], h: float, steps: int) -> list[dict[str, float]]:
    """Regression-only sampling checks; not a formal containment proof."""
    system = cfg["system"]
    init = cfg["initial"]
    T = float(h) * int(steps)
    samples: list[dict[str, float]] = []

    if system == "scalar_quadratic":
        for x0 in _linspace(init["x"][0], init["x"][1], 21):
            samples.append({"x": math.tan(T + math.atan(x0))})
        return samples

    if system == "harmonic_oscillator":
        for x0 in _linspace(init["x"][0], init["x"][1], 7):
            for y0 in _linspace(init["y"][0], init["y"][1], 7):
                samples.append({
                    "x": x0 * math.cos(T) + y0 * math.sin(T),
                    "y": -x0 * math.sin(T) + y0 * math.cos(T),
                })
        return samples

    if system == "van_der_pol":
        substeps = max(1, steps * 10)
        dt = T / substeps if substeps else 0.0
        for x0 in _linspace(init["x"][0], init["x"][1], 6):
            for y0 in _linspace(init["y"][0], init["y"][1], 6):
                state = (x0, y0)
                for _ in range(substeps):
                    state = _rk4_step(state, dt)
                samples.append({"x": state[0], "y": state[1]})
        return samples

    if system == "affine_controlled":
        factor = math.exp(0.5 * T)
        for x0 in _linspace(init["x"][0], init["x"][1], 11):
            for e in _linspace(init["e"][0], init["e"][1], 5):
                x = factor * x0 + 2.0 * (factor - 1.0) * e
                samples.append({"x": x, "e": e})
        return samples

    return samples


def torch_containment_failures(result, cfg: Mapping[str, Any], h: float, steps: int) -> int:
    state_vars = list(cfg["state_vars"])
    indices = _metric_indices(cfg)
    final_box = result.final_tm.range_box()
    failures = 0
    for sample in sample_final_states(cfg, h, steps):
        values = [sample.get(v, float("nan")) for v in state_vars]
        if not _torch_contains_box(final_box, values, indices, tol=1e-8):
            failures += 1
    return failures


def flowstar_containment_failures(parsed_box: Mapping[str, tuple[float, float]], cfg: Mapping[str, Any], h: float, steps: int) -> int | str:
    metric_vars = _metric_vars(cfg)
    samples = sample_final_states(cfg, h, steps)
    if not parsed_box or not samples:
        return ""
    return sum(0 if _contains_box(parsed_box, sample, metric_vars, tol=1e-8) else 1 for sample in samples)


def _flowstar_declared_failure(run) -> str:
    text = ""
    if run.stdout_path is not None and Path(run.stdout_path).exists():
        text += Path(run.stdout_path).read_text(encoding="utf-8", errors="ignore")
    if run.stderr_path is not None and Path(run.stderr_path).exists():
        text += "\n" + Path(run.stderr_path).read_text(encoding="utf-8", errors="ignore")
    if "FLOWSTAR_COMPLETED 0" in text:
        for line in text.splitlines():
            if "terminated" in line.lower() or "failed" in line.lower():
                return line.strip()
        return "Flow* reported FLOWSTAR_COMPLETED 0"
    return ""


def _flowstar_plot_paths(run, model_dir: Path) -> list[Path]:
    if run.stdout_path is None or not Path(run.stdout_path).exists():
        return []
    text = Path(run.stdout_path).read_text(encoding="utf-8", errors="ignore")
    paths = []
    for match in re.finditer(r"FLOWSTAR_PLOT\s+(\S+)", text):
        paths.append(model_dir / f"{match.group(1)}.plt")
    return paths


def torch_row(cfg: Mapping[str, Any], *, h: float, steps: int, order: int, mode: str) -> dict[str, Any]:
    ode_fn = ode_for_config(cfg)
    x0_box = _interval_box_from_config(cfg)
    state_vars = list(cfg["state_vars"])
    indices = _metric_indices(cfg)
    start = time.perf_counter()
    result = flowpipe_multi_step(
        ode_fn,
        x0_box,
        h=h,
        steps=steps,
        order=order,
        mode=mode,
    )
    runtime = time.perf_counter() - start
    endpoint_sum, endpoint_max = _box_width_metrics(result.final_tm.range_box(), indices)
    last_segment_box = result.segments[-1].tm.range_box() if result.segments else result.final_tm.range_box()
    last_segment_sum, last_segment_max = _box_width_metrics(last_segment_box, indices)
    tube_sum, tube_max = _box_width_metrics(_flowpipe_tube_box(result, len(state_vars)), indices)
    return {
        "system": cfg["system"],
        "tool": "torch_tm_flowpipe",
        "mode": mode,
        "h": h,
        "steps": steps,
        "order": order,
        "status": result.status,
        "setting_label": "torch_default",
        "endpoint_width_sum": endpoint_sum,
        "endpoint_width_max": endpoint_max,
        "last_segment_width_sum": last_segment_sum,
        "last_segment_width_max": last_segment_max,
        "tube_width_sum": tube_sum,
        "tube_width_max": tube_max,
        "box_source": "torch_endpoint_last_segment_tube",
        "endpoint_box_available": True,
        "last_segment_box_available": True,
        "tube_box_available": True,
        "final_width_sum": endpoint_sum,
        "final_width_max": endpoint_max,
        "flowpipe_width_sum": tube_sum,
        "flowpipe_width_max": tube_max,
        "runtime_s": runtime,
        "num_segments": len(result.segments),
        "validation_attempts": result.validation_attempts,
        "term_count": _term_count(result, indices),
        "actual_degree": _actual_degree(result, indices),
        "remainder_radius": _remainder_radius(result, indices),
        "containment_failures": torch_containment_failures(result, cfg, h, steps),
        "flowstar_internal_reach_s": "",
        "flowstar_wall_compile_s": "",
        "flowstar_wall_run_s": "",
        "flowstar_wall_total_s": "",
        "flowstar_compile_s": "",
        "flowstar_run_s": "",
        "flowstar_model_path": "",
        "flowstar_stdout_path": "",
        "flowstar_stderr_path": "",
        "failure_reason": "",
    }


def flowstar_row(
    cfg: Mapping[str, Any],
    config_path: Path,
    *,
    h: float,
    steps: int,
    order: int,
    model_dir: Path,
    flowstar_target: str,
    flowstar_bin: str | None,
    flowstar_root: str | None,
    skip_flowstar: bool,
    timeout_s: float | None,
    build_flowstar_lib: bool,
    flowstar_remainder_radius: float | None = None,
    flowstar_cutoff: float | None = None,
    flowstar_setting_label: str = "default",
) -> dict[str, Any]:
    system = str(cfg["system"])
    base = f"{system}_h{h:g}_s{steps}_o{order}"
    if flowstar_target == "toolbox_cpp":
        model_path = model_dir / f"{base}.cpp"
        plot_name = base
    else:
        model_path = model_dir / f"{base}.model"
        plot_name = f"{base}.plt"
    # Always export the Flow* input artifact, even when Flow* itself is not
    # installed.  This makes skipped rows actionable: users can copy/compile the
    # generated C++ case on a server with Flow* and reproduce the exact case.
    export_model(
        config_path,
        model_path,
        h=h,
        steps=steps,
        order=order,
        plot_output_name=plot_name,
        target=flowstar_target,
        remainder_radius=flowstar_remainder_radius,
        cutoff=flowstar_cutoff,
    )
    if skip_flowstar:
        return {
            "system": system,
            "tool": "flowstar",
            "mode": "fixed",
            "setting_label": flowstar_setting_label,
            "h": h,
            "steps": steps,
            "order": order,
            "status": "skipped",
            "final_width_sum": "",
            "final_width_max": "",
            "flowpipe_width_sum": "",
            "flowpipe_width_max": "",
            "runtime_s": "",
            "num_segments": steps,
            "validation_attempts": "",
            "term_count": "",
            "actual_degree": "",
            "remainder_radius": "",
            "containment_failures": "",
            "flowstar_compile_s": "",
            "flowstar_run_s": "",
            "flowstar_model_path": str(model_path),
            "flowstar_stdout_path": "",
            "flowstar_stderr_path": "",
            "failure_reason": "skip_flowstar requested or Flow* root/executable unavailable",
        }

    if flowstar_target == "toolbox_cpp":
        run = run_flowstar_toolbox(
            model_path,
            flowstar_root=flowstar_root,
            output_dir=model_dir,
            timeout_s=timeout_s,
            build_lib=build_flowstar_lib,
        )
    else:
        run = run_flowstar_legacy_model(model_path, flowstar_bin=flowstar_bin, output_dir=model_dir, timeout_s=timeout_s)
    if run.status != "completed":
        return {
            "system": system,
            "tool": "flowstar",
            "mode": "fixed",
            "setting_label": flowstar_setting_label,
            "h": h,
            "steps": steps,
            "order": order,
            "status": run.status,
            "final_width_sum": "",
            "final_width_max": "",
            "flowpipe_width_sum": "",
            "flowpipe_width_max": "",
            "runtime_s": run.runtime_s,
            "num_segments": steps,
            "validation_attempts": "",
            "term_count": "",
            "actual_degree": "",
            "remainder_radius": "",
            "containment_failures": "",
            "flowstar_internal_reach_s": "",
            "flowstar_wall_compile_s": run.compile_s,
            "flowstar_wall_run_s": run.run_s,
            "flowstar_wall_total_s": run.runtime_s,
            "flowstar_compile_s": run.compile_s,
            "flowstar_run_s": run.run_s,
            "flowstar_model_path": str(model_path),
            "flowstar_stdout_path": str(run.stdout_path) if run.stdout_path is not None else "",
            "flowstar_stderr_path": str(run.stderr_path) if run.stderr_path is not None else "",
            "failure_reason": run.message,
        }

    declared_failure = _flowstar_declared_failure(run)
    if declared_failure:
        internal_runtime = _flowstar_internal_runtime(run)
        runtime_s = internal_runtime if internal_runtime != "" else run.runtime_s
        return {
            "system": system,
            "tool": "flowstar",
            "mode": "fixed",
            "setting_label": flowstar_setting_label,
            "h": h,
            "steps": steps,
            "order": order,
            "status": "failed",
            "final_width_sum": "",
            "final_width_max": "",
            "flowpipe_width_sum": "",
            "flowpipe_width_max": "",
            "runtime_s": runtime_s,
            "num_segments": steps,
            "validation_attempts": "",
            "term_count": "",
            "actual_degree": "",
            "remainder_radius": "",
            "containment_failures": "",
            "flowstar_internal_reach_s": internal_runtime,
            "flowstar_wall_compile_s": run.compile_s,
            "flowstar_wall_run_s": run.run_s,
            "flowstar_wall_total_s": run.runtime_s,
            "flowstar_compile_s": run.compile_s,
            "flowstar_run_s": run.run_s,
            "flowstar_model_path": str(model_path),
            "flowstar_stdout_path": str(run.stdout_path) if run.stdout_path is not None else "",
            "flowstar_stderr_path": str(run.stderr_path) if run.stderr_path is not None else "",
            "failure_reason": declared_failure,
        }

    parse_candidates = [p for p in [run.stdout_path, run.stderr_path, model_dir / plot_name] if p is not None]
    parse_candidates.extend(_flowstar_plot_paths(run, model_dir))
    parse_candidates.extend(run.artifact_paths)
    parse_candidates.extend(model_dir.glob(f"{base}*"))
    # Deduplicate while preserving order.  Toolbox output often consists of
    # GNUPLOT interval files rather than explicit "x in [lo, hi]" ranges.
    deduped_candidates = list(dict.fromkeys(parse_candidates))
    parsed = parse_files(
        deduped_candidates,
        variables=cfg["state_vars"],
        numeric_plot_vars=cfg.get("flowstar", {}).get("plot_vars", cfg["state_vars"]),
    )
    metric_vars = _metric_vars(cfg)
    endpoint_widths = parsed_widths(parsed.endpoint_box, metric_vars)
    last_segment_widths = parsed_widths(parsed.last_segment_box, metric_vars)
    tube_widths = parsed_widths(parsed.tube_box, metric_vars)
    if parsed.status != "parsed" or not (endpoint_widths or last_segment_widths or tube_widths):
        status = "unparsed"
    else:
        status = "completed"
    internal_runtime = _flowstar_internal_runtime(run)
    runtime_s = internal_runtime if internal_runtime != "" else run.runtime_s
    return {
        "system": system,
        "tool": "flowstar",
        "mode": "fixed",
        "setting_label": flowstar_setting_label,
        "h": h,
        "steps": steps,
        "order": order,
        "status": status,
        "endpoint_width_sum": sum(endpoint_widths) if endpoint_widths else "",
        "endpoint_width_max": max(endpoint_widths) if endpoint_widths else "",
        "last_segment_width_sum": sum(last_segment_widths) if last_segment_widths else "",
        "last_segment_width_max": max(last_segment_widths) if last_segment_widths else "",
        "tube_width_sum": sum(tube_widths) if tube_widths else "",
        "tube_width_max": max(tube_widths) if tube_widths else "",
        "box_source": "flowstar_gnuplot_last_segment_and_tube" if last_segment_widths else "endpoint_unavailable_from_gnuplot",
        "endpoint_box_available": bool(endpoint_widths),
        "last_segment_box_available": bool(last_segment_widths),
        "tube_box_available": bool(tube_widths),
        "final_width_sum": sum(endpoint_widths) if endpoint_widths else "",
        "final_width_max": max(endpoint_widths) if endpoint_widths else "",
        "flowpipe_width_sum": sum(tube_widths) if tube_widths else "",
        "flowpipe_width_max": max(tube_widths) if tube_widths else "",
        "runtime_s": runtime_s,
        "num_segments": steps,
        "validation_attempts": "",
        "term_count": "",
        "actual_degree": "",
        "remainder_radius": "",
        "containment_failures": flowstar_containment_failures(parsed.endpoint_box, cfg, h, steps) if endpoint_widths else "",
        "flowstar_internal_reach_s": internal_runtime,
        "flowstar_wall_compile_s": run.compile_s,
        "flowstar_wall_run_s": run.run_s,
        "flowstar_wall_total_s": run.runtime_s,
        "flowstar_compile_s": run.compile_s,
        "flowstar_run_s": run.run_s,
        "flowstar_model_path": str(model_path),
        "flowstar_stdout_path": str(run.stdout_path) if run.stdout_path is not None else "",
        "flowstar_stderr_path": str(run.stderr_path) if run.stderr_path is not None else "",
        "failure_reason": parsed.message if status == "unparsed" else "",
    }


def _case_grid(
    cfg: Mapping[str, Any],
    *,
    all_cases: bool,
    h_values: Sequence[float] | None = None,
    steps_values: Sequence[int] | None = None,
    orders: Sequence[int] | None = None,
) -> list[tuple[float, int, int]]:
    hs = list(h_values if h_values is not None else cfg["h"])
    steps = list(steps_values if steps_values is not None else cfg["steps"])
    order_values = list(orders if orders is not None else cfg["order"])
    if not all_cases and h_values is None and steps_values is None and orders is None:
        return [(float(hs[0]), int(steps[0]), int(order_values[0]))]
    return [(float(h), int(s), int(o)) for h, s, o in itertools.product(hs, steps, order_values)]


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _read_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v: Any) -> float | None:
    try:
        if v in (None, ""):
            return None
        x = float(v)
        if math.isfinite(x):
            return x
    except (TypeError, ValueError):
        return None
    return None


def _plot_placeholder(ax, title: str, msg: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def make_plots(csv_path: str | Path, output_dir: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt

    rows = _read_rows(csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def aggregate(metric: str, predicate) -> dict[tuple[str, str, str, int], list[float]]:
        grouped: dict[tuple[str, str, str, int], list[float]] = {}
        for r in rows:
            if not predicate(r):
                continue
            val = _to_float(r.get(metric))
            if val is None:
                continue
            key = (r["system"], r["tool"], r["mode"], int(float(r["steps"])))
            grouped.setdefault(key, []).append(val)
        return grouped

    for filename, metric, title, ylabel in [
        ("final_width_vs_steps.png", "final_width_sum", "Final width vs. steps", "final width sum"),
        ("runtime_vs_steps.png", "runtime_s", "Runtime vs. steps", "runtime (s)"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        grouped = aggregate(metric, lambda r: r.get("status") in {"validated", "completed"})
        if grouped:
            for (system, tool, mode, _steps), vals in sorted(grouped.items()):
                # Aggregate over h/order for each step.
                pass
            by_label: dict[str, dict[int, list[float]]] = {}
            for (system, tool, mode, steps), vals in grouped.items():
                label = f"{system}/{tool}/{mode}"
                by_label.setdefault(label, {}).setdefault(steps, []).extend(vals)
            for label, by_steps in sorted(by_label.items()):
                xs = sorted(by_steps)
                ys = [sum(by_steps[x]) / len(by_steps[x]) for x in xs]
                ax.plot(xs, ys, marker="o", label=label)
            ax.set_xlabel("steps")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(fontsize=7, ncol=2)
        else:
            _plot_placeholder(ax, title, "No parsed successful rows")
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    # Width ratios use matching semantics only: GNUPLOT-derived last segment and
    # tube boxes are not endpoint boxes.
    for ratio_type, width_col, ylabel in [
        ("last_segment", "last_segment_width_sum", "torch last-segment width sum / Flow* last-segment width sum"),
        ("tube", "tube_width_sum", "torch tube width sum / Flow* tube width sum"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        flow_by_case: dict[tuple[str, str, str, str], float] = {}
        for r in rows:
            if r.get("tool") == "flowstar" and r.get("status") == "completed":
                val = _to_float(r.get(width_col))
                if val and val > 0.0:
                    flow_by_case[(r["system"], r["h"], r["steps"], r["order"])] = val
        ratio_by_label: dict[str, dict[int, list[float]]] = {}
        for r in rows:
            if r.get("tool") != "torch_tm_flowpipe" or r.get("status") != "validated":
                continue
            val = _to_float(r.get(width_col))
            flow_val = flow_by_case.get((r["system"], r["h"], r["steps"], r["order"]))
            if val is None or flow_val is None or flow_val <= 0.0:
                continue
            label = f"{r['system']}/{r['mode']}"
            ratio_by_label.setdefault(label, {}).setdefault(int(float(r["steps"])), []).append(val / flow_val)
        title = f"{ratio_type.replace('_', '-')} width ratio: torch over Flow*"
        if ratio_by_label:
            for label, by_steps in sorted(ratio_by_label.items()):
                xs = sorted(by_steps)
                ys = [sum(by_steps[x]) / len(by_steps[x]) for x in xs]
                ax.plot(xs, ys, marker="o", label=label)
            ax.axhline(1.0, linestyle="--", linewidth=1)
            ax.set_xlabel("steps")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(fontsize=7, ncol=2)
        else:
            _plot_placeholder(ax, title, f"No parsed Flow* {ratio_type.replace('_', '-')} width data")
        fig.tight_layout()
        path = out_dir / f"torch_over_flowstar_{ratio_type}_width_ratio.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    # Dependency preserving vs range-only ratio.
    fig, ax = plt.subplots(figsize=(9, 5))
    range_by_case: dict[tuple[str, str, str, str], float] = {}
    dep_by_case: dict[tuple[str, str, str, str], float] = {}
    for r in rows:
        if r.get("tool") != "torch_tm_flowpipe" or r.get("status") != "validated":
            continue
        val = _to_float(r.get("final_width_sum"))
        if val is None:
            continue
        key = (r["system"], r["h"], r["steps"], r["order"])
        if r.get("mode") == "range_only":
            range_by_case[key] = val
        elif r.get("mode") == "dependency_preserving":
            dep_by_case[key] = val
    dep_ratio_by_system: dict[str, dict[int, list[float]]] = {}
    for key, dep_val in dep_by_case.items():
        rng_val = range_by_case.get(key)
        if rng_val is None or rng_val <= 0.0:
            continue
        system, _h, steps, _order = key
        dep_ratio_by_system.setdefault(system, {}).setdefault(int(float(steps)), []).append(dep_val / rng_val)
    if dep_ratio_by_system:
        for system, by_steps in sorted(dep_ratio_by_system.items()):
            xs = sorted(by_steps)
            ys = [sum(by_steps[x]) / len(by_steps[x]) for x in xs]
            ax.plot(xs, ys, marker="o", label=system)
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xlabel("steps")
        ax.set_ylabel("dependency-preserving / range-only final width sum")
        ax.set_title("Dependency-preserving vs. range-only")
        ax.legend(fontsize=8)
    else:
        _plot_placeholder(ax, "Dependency-preserving vs. range-only", "No matching torch rows")
    fig.tight_layout()
    path = out_dir / "dependency_preserving_vs_range_only.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    return paths


def _select_config_paths(config_paths: list[Path], systems: Sequence[str] | None) -> list[Path]:
    if not systems:
        return config_paths
    wanted = set(systems)
    selected = []
    for path in config_paths:
        cfg = load_config(path)
        if cfg["system"] in wanted:
            selected.append(path)
    found = {load_config(p)["system"] for p in selected}
    missing = wanted - found
    if missing:
        raise SystemExit(f"unknown system(s): {', '.join(sorted(missing))}")
    return selected


def run_comparison(args: argparse.Namespace) -> list[dict[str, Any]]:
    config_paths = [Path(p) for p in args.config] if args.config else DEFAULT_CONFIGS
    config_paths = _select_config_paths(config_paths, args.systems)
    rows: list[dict[str, Any]] = []
    model_dir = Path(args.model_dir)

    flowstar_exe: str | None = None
    flowstar_root: str | None = None
    if args.flowstar_target == "toolbox_cpp":
        root = find_flowstar_root(args.flowstar_root)
        flowstar_root = str(root) if root is not None else None
        skip_flowstar = bool(args.skip_flowstar or flowstar_root is None)
        if flowstar_root is None and not args.skip_flowstar:
            print("Flow* toolbox root not found; set FLOWSTAR_ROOT or pass --flowstar-root. Flow* rows will be marked skipped.")
    else:
        flowstar_exe = find_flowstar_executable(args.flowstar_bin)
        skip_flowstar = bool(args.skip_flowstar or flowstar_exe is None)
        if flowstar_exe is None and not args.skip_flowstar:
            print("Flow* executable not found; Flow* rows will be marked skipped.")

    for config_path in config_paths:
        cfg = load_config(config_path)
        cases = _case_grid(
            cfg,
            all_cases=args.all,
            h_values=args.h_values,
            steps_values=args.steps_values,
            orders=args.orders,
        )
        print(f"{cfg['system']}: {len(cases)} case(s)")
        for h, steps, order in cases:
            if not args.flowstar_only:
                for mode in ["range_only", "dependency_preserving"]:
                    row = torch_row(cfg, h=h, steps=steps, order=order, mode=mode)
                    rows.append(row)
                    print(
                        f"  torch/{mode}: h={h:g} steps={steps} order={order} "
                        f"status={row['status']} final_sum={float(row['final_width_sum']):.6g} "
                        f"time={float(row['runtime_s']):.4f}s failures={row['containment_failures']}"
                    )
            frow = flowstar_row(
                cfg,
                config_path,
                h=h,
                steps=steps,
                order=order,
                model_dir=model_dir,
                flowstar_target=args.flowstar_target,
                flowstar_bin=flowstar_exe,
                flowstar_root=flowstar_root,
                skip_flowstar=skip_flowstar,
                timeout_s=args.flowstar_timeout_s,
                build_flowstar_lib=not args.no_build_flowstar_lib,
                flowstar_remainder_radius=args.flowstar_remainder_radius,
                flowstar_cutoff=args.flowstar_cutoff,
                flowstar_setting_label=args.flowstar_setting_label,
            )
            rows.append(frow)
            print(f"  flowstar/fixed: h={h:g} steps={steps} order={order} status={frow['status']}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare torch_tm_flowpipe against plant-only Flow* runs.")
    parser.add_argument("--all", action="store_true", help="run the full configured grids; otherwise run the first case per config")
    parser.add_argument("--config", nargs="*", default=None, help="YAML config path(s); defaults to all bundled configs")
    parser.add_argument("--systems", nargs="+", default=None, help="system name(s) from bundled configs")
    parser.add_argument("--orders", type=int, nargs="+", default=None)
    parser.add_argument("--h-values", type=float, nargs="+", default=None)
    parser.add_argument("--steps-values", type=int, nargs="+", default=None)
    parser.add_argument("--csv", default="outputs/flowstar_comparison.csv")
    parser.add_argument("--model-dir", default="outputs/flowstar_models")
    parser.add_argument("--flowstar-target", choices=["toolbox_cpp", "legacy_model"], default="toolbox_cpp", help="Flow* backend interface to use; toolbox_cpp matches chenxin415/flowstar")
    parser.add_argument("--flowstar-root", default=None, help="path to chenxin415/flowstar root; alternatively set FLOWSTAR_ROOT")
    parser.add_argument("--flowstar-bin", default=None, help="legacy Flow* executable for --flowstar-target legacy_model")
    parser.add_argument("--skip-flowstar", action="store_true", help="do not run Flow*, only write skipped Flow* rows")
    parser.add_argument("--flowstar-only", action="store_true", help="skip torch rows when sweeping Flow* settings")
    parser.add_argument("--flowstar-timeout-s", type=float, default=None)
    parser.add_argument("--no-build-flowstar-lib", action="store_true", help="do not run make in flowstar-toolbox when libflowstar.a is missing")
    parser.add_argument("--flowstar-remainder-radius", type=float, default=None)
    parser.add_argument("--flowstar-cutoff", type=float, default=None)
    parser.add_argument("--flowstar-setting-label", default="default")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    rows = run_comparison(args)
    write_csv(args.csv, rows)
    print(f"wrote {args.csv}")
    summary_path = Path(args.csv).with_name(Path(args.csv).stem + "_summary.md")
    generate_summary(args.csv, summary_path)
    print(f"wrote {summary_path}")
    if not args.no_plots:
        plot_paths = make_plots(args.csv, Path(args.csv).parent)
        for p in plot_paths:
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
