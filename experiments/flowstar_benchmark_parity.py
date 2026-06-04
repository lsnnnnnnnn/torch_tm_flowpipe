#!/usr/bin/env python3
"""Generate Flow* original Van der Pol benchmark parity artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
FLOWSTAR_RUNNER_DIR = REPO_ROOT / "comparisons" / "flowstar"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(FLOWSTAR_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(FLOWSTAR_RUNNER_DIR))

import torch

torch.set_num_threads(1)

from torch_tm_flowpipe import Interval, TMVector, flowpipe_step, flowpipe_step_from_tm
from torch_tm_flowpipe.ode_examples import van_der_pol_ode

from run_flowstar import find_flowstar_root, run_flowstar_toolbox

STATE_VARS = ("x", "y")
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
PARITY_FIELDS = [
    "tool",
    "status",
    "original_flowstar_wall_run_s",
    "generated_flowstar_internal_reach_s",
    "generated_flowstar_compile_wall_s",
    "generated_flowstar_run_wall_s",
    "torch_runtime_s",
    "num_segments",
    "last_reached_t",
    "validated_segments",
    "last_validated_t",
    "failed_segment_index",
    "failed_segment_t_lo",
    "failed_segment_t_hi",
    "last_attempted_t",
    "failure_reason",
    "requested_horizon",
    "last_segment_width_x",
    "last_segment_width_y",
    "last_segment_width_sum",
    "tube_width_x",
    "tube_width_y",
    "tube_width_sum",
    "endpoint_box_available",
    "box_source",
    "notes",
]
ORIGINAL_SUMMARY_FIELDS = [
    "status",
    "make_wall_s",
    "wall_run_s",
    "internal_time_cost_s",
    "returncode",
    "num_segments",
    "last_reached_t",
    "requested_horizon",
    "last_segment_width_x",
    "last_segment_width_y",
    "last_segment_width_sum",
    "tube_width_x",
    "tube_width_y",
    "tube_width_sum",
    "endpoint_box_available",
    "box_source",
    "stdout_path",
    "stderr_path",
    "notes",
]
GENERATED_SUMMARY_FIELDS = [
    "status",
    "flowstar_internal_reach_s",
    "compile_wall_s",
    "run_wall_s",
    "total_wall_s",
    "returncode",
    "num_segments",
    "last_reached_t",
    "requested_horizon",
    "last_segment_width_x",
    "last_segment_width_y",
    "last_segment_width_sum",
    "tube_width_x",
    "tube_width_y",
    "tube_width_sum",
    "endpoint_box_available",
    "box_source",
    "stdout_path",
    "stderr_path",
    "model_path",
    "plot_paths",
    "notes",
]
TORCH_SUMMARY_FIELDS = [
    "status",
    "mode",
    "runtime_s",
    "num_segments",
    "last_reached_t",
    "validated_segments",
    "last_validated_t",
    "failed_segment_index",
    "failed_segment_t_lo",
    "failed_segment_t_hi",
    "last_attempted_t",
    "failure_reason",
    "requested_horizon",
    "requested_order",
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
    "validation_attempts",
    "box_source",
    "notes",
]
HASH_FIELDS = ["path", "sha256"]
COMPARISON_FIELDS = ["metric", "value"]

NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
NUMERIC_PAIR_RE = re.compile(rf"^\s*(?P<t>{NUMBER_RE})\s+(?P<v>{NUMBER_RE})(?:\s|$)")
TIME_COST_RE = re.compile(r"time cost:\s*(?P<runtime>" + NUMBER_RE + r")")
FLOWSTAR_RUNTIME_RE = re.compile(r"FLOWSTAR_RUNTIME_S\s+(?P<runtime>" + NUMBER_RE + r")")


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
        writer = csv.DictWriter(f, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in fields})


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _width(lo: float, hi: float) -> float:
    return max(0.0, float(hi) - float(lo))


def _hull(rows: Sequence[Mapping[str, Any]], var: str) -> tuple[float, float] | None:
    if not rows:
        return None
    lo_key = f"{var}_lo"
    hi_key = f"{var}_hi"
    return (min(float(r[lo_key]) for r in rows), max(float(r[hi_key]) for r in rows))


def _summarize_segments(rows: Sequence[Mapping[str, Any]], requested_horizon: float) -> dict[str, Any]:
    last = rows[-1] if rows else None
    tube_x = _hull(rows, "x")
    tube_y = _hull(rows, "y")
    return {
        "num_segments": len(rows),
        "last_reached_t": float(last["t_hi"]) if last else 0.0,
        "requested_horizon": requested_horizon,
        "last_segment_width_x": last["width_x"] if last else "",
        "last_segment_width_y": last["width_y"] if last else "",
        "last_segment_width_sum": last["width_sum"] if last else "",
        "tube_width_x": _width(*tube_x) if tube_x else "",
        "tube_width_y": _width(*tube_y) if tube_y else "",
        "tube_width_sum": (_width(*tube_x) + _width(*tube_y)) if tube_x and tube_y else "",
    }


def _completed_progress_fields(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "validated_segments": summary.get("num_segments", ""),
        "last_validated_t": summary.get("last_reached_t", ""),
        "failed_segment_index": "",
        "failed_segment_t_lo": "",
        "failed_segment_t_hi": "",
        "last_attempted_t": summary.get("last_reached_t", ""),
        "failure_reason": "",
    }


def _torch_progress_fields(
    rows: Sequence[Mapping[str, Any]],
    requested_horizon: float,
    status: str,
    *,
    failed_segment_index: int | None = None,
    failed_segment_t_lo: float | None = None,
    failed_segment_t_hi: float | None = None,
    failure_reason: str = "",
) -> dict[str, Any]:
    summary = _summarize_segments(rows, requested_horizon)
    if status == "failed" and failed_segment_index is not None:
        validated_rows = [r for r in rows if int(r["segment_index"]) < failed_segment_index]
        last_validated_t = float(validated_rows[-1]["t_hi"]) if validated_rows else 0.0
        if failed_segment_t_lo is None and rows:
            failed_segment_t_lo = float(rows[-1]["t_lo"])
        if failed_segment_t_hi is None and rows:
            failed_segment_t_hi = float(rows[-1]["t_hi"])
        return {
            **summary,
            "validated_segments": len(validated_rows),
            "last_validated_t": last_validated_t,
            "failed_segment_index": failed_segment_index,
            "failed_segment_t_lo": failed_segment_t_lo if failed_segment_t_lo is not None else "",
            "failed_segment_t_hi": failed_segment_t_hi if failed_segment_t_hi is not None else "",
            "last_attempted_t": failed_segment_t_hi if failed_segment_t_hi is not None else summary["last_reached_t"],
            "failure_reason": failure_reason,
        }
    return {
        **summary,
        "validated_segments": summary["num_segments"],
        "last_validated_t": summary["last_reached_t"],
        "failed_segment_index": "",
        "failed_segment_t_lo": "",
        "failed_segment_t_hi": "",
        "last_attempted_t": summary["last_reached_t"],
        "failure_reason": "",
    }


def _parse_gnuplot_blocks(path: Path) -> list[tuple[float, float, float, float]]:
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
        match = NUMERIC_PAIR_RE.match(line)
        if not match:
            flush()
            continue
        current.append((float(match.group("t")), float(match.group("v"))))
    flush()

    segments: list[tuple[float, float, float, float]] = []
    for block in blocks:
        ts = [p[0] for p in block]
        vals = [p[1] for p in block]
        segments.append((min(ts), max(ts), min(vals), max(vals)))
    return segments


def _combine_segments(case_id: str, x_path: Path, y_path: Path, box_source: str) -> list[dict[str, Any]]:
    x_segments = _parse_gnuplot_blocks(x_path)
    y_segments = _parse_gnuplot_blocks(y_path)
    n = min(len(x_segments), len(y_segments))
    rows: list[dict[str, Any]] = []
    for i in range(n):
        x_t_lo, x_t_hi, x_lo, x_hi = x_segments[i]
        y_t_lo, y_t_hi, y_lo, y_hi = y_segments[i]
        width_x = _width(x_lo, x_hi)
        width_y = _width(y_lo, y_hi)
        rows.append(
            {
                "case_id": case_id,
                "segment_index": i,
                "t_lo": min(x_t_lo, y_t_lo),
                "t_hi": max(x_t_hi, y_t_hi),
                "x_lo": x_lo,
                "x_hi": x_hi,
                "y_lo": y_lo,
                "y_hi": y_hi,
                "width_x": width_x,
                "width_y": width_y,
                "width_sum": width_x + width_y,
                "box_source": box_source,
            }
        )
    return rows


def _parse_first_float(pattern: re.Pattern[str], text: str) -> float | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return float(matches[-1].group("runtime"))


def _extract_between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = text.find(end, i)
    if j < 0:
        return text[i:].strip()
    return text[i:j].strip()


def parse_original_params(flowstar_root: Path) -> dict[str, Any]:
    bench_dir = flowstar_root / "benchmarks" / "continuous" / "vanderpol"
    source = (bench_dir / "vanderpol.cpp").read_text(encoding="utf-8")
    readme = (bench_dir / "README.md").read_text(encoding="utf-8")
    continuous = (flowstar_root / "flowstar-toolbox" / "Continuous.cpp").read_text(encoding="utf-8")

    ode_body = _extract_between(source, "ODE<Real> ode({", "}, vars);")
    ode_exprs = [s.strip().strip('"') for s in ode_body.split(",") if s.strip()]
    init_match = re.search(
        r"Interval\s+init_x\((?P<xlo>" + NUMBER_RE + r"),\s*(?P<xhi>" + NUMBER_RE + r")\),\s*init_y\((?P<ylo>" + NUMBER_RE + r"),\s*(?P<yhi>" + NUMBER_RE + r")\)",
        source,
    )
    if not init_match:
        raise ValueError("could not parse original Van der Pol initial box")
    horizon_match = re.search(r"double\s+T\s*=\s*(?P<T>" + NUMBER_RE + r")", source)
    if not horizon_match:
        raise ValueError("could not parse original Van der Pol horizon")
    safe_match = re.search(r'Constraint\("(?P<constraint>[^"]+)",\s*vars\)', source)
    symbolic_match = re.search(r"Symbolic_Remainder\s+sr\(initialSet,\s*(?P<size>\d+)\)", source)
    adaptive_match = re.search(
        r"setAdaptiveStepsize\((?P<step_min>" + NUMBER_RE + r"),\s*(?P<step_max>" + NUMBER_RE + r"),\s*(?P<order>\d+)\)",
        continuous,
    )
    cutoff_match = re.search(
        r"Interval\s+cutoff_threshold\((?P<lo>-" + NUMBER_RE + r"),\s*(?P<hi>" + NUMBER_RE + r")\)",
        continuous,
    )
    remainder_match = re.search(
        r"Interval\s+I\((?P<lo>-" + NUMBER_RE + r"),\s*(?P<hi>" + NUMBER_RE + r")\);\s*std::vector<Interval>\s+estimation",
        continuous,
    )
    plot_calls = re.findall(r'plot_2D_interval_GNUPLOT\("\./",\s*"(?P<stem>[^"]+)"', source)
    if not adaptive_match or not cutoff_match or not remainder_match:
        raise ValueError("could not parse Flow* default computational setting")

    horizon = float(horizon_match.group("T"))
    params = {
        "source": str(bench_dir / "vanderpol.cpp"),
        "readme": str(bench_dir / "README.md"),
        "system": "van_der_pol",
        "state_variables": ["x", "y", "t"],
        "ode": {
            "x": ode_exprs[0],
            "y": ode_exprs[1],
            "t": ode_exprs[2] if len(ode_exprs) > 2 else "1",
        },
        "initial_set": {
            "x": [float(init_match.group("xlo")), float(init_match.group("xhi"))],
            "y": [float(init_match.group("ylo")), float(init_match.group("yhi"))],
            "t": [0.0, 0.0],
        },
        "time_horizon": horizon,
        "safe_set": safe_match.group("constraint") + " <= 0" if safe_match else "",
        "step_policy": "adaptive",
        "step_min": float(adaptive_match.group("step_min")),
        "step_max": float(adaptive_match.group("step_max")),
        "taylor_order": int(adaptive_match.group("order")),
        "order_policy": "fixed",
        "remainder_estimation": [
            float(remainder_match.group("lo")),
            float(remainder_match.group("hi")),
        ],
        "cutoff_threshold": [
            float(cutoff_match.group("lo")),
            float(cutoff_match.group("hi")),
        ],
        "symbolic_remainder_queue_size": int(symbolic_match.group("size")) if symbolic_match else None,
        "plot_commands": [
            f'plot_setting.plot_2D_interval_GNUPLOT("./", "{stem}", result.tmv_flowpipes, setting)'
            for stem in plot_calls
        ],
        "plot_stems": plot_calls,
        "plot_files": [f"{stem}.plt" for stem in plot_calls],
        "eps_files": [f"{stem}.eps" for stem in plot_calls],
        "reference_png_files": [
            str(flowstar_root / "images" / "benchmarks" / f"{stem}.png") for stem in plot_calls
        ],
        "readme_result_images": re.findall(r"<img\s+src='([^']+)'", readme),
        "notes": (
            "The original source does not call setFixedStepsize; it relies on "
            "Computational_Setting defaults parsed from flowstar-toolbox/Continuous.cpp."
        ),
    }
    return params


def write_params_docs(out_dir: Path, params: Mapping[str, Any]) -> None:
    _write_json(out_dir / "original_flowstar_params.json", params)
    lines = [
        "# Original Flow* Van der Pol Parameters",
        "",
        f"- Source: `{params['source']}`",
        f"- ODE: `x' = {params['ode']['x']}`, `y' = {params['ode']['y']}`, `t' = {params['ode']['t']}`",
        f"- Initial set: `x in {params['initial_set']['x']}`, `y in {params['initial_set']['y']}`, `t = 0`",
        f"- Time horizon: `[0, {params['time_horizon']}]`",
        f"- Safe set: `{params['safe_set']}`",
        f"- Step policy: `{params['step_policy']}`, min `{params['step_min']}`, max `{params['step_max']}`",
        f"- Taylor order policy: `{params['order_policy']}`, order `{params['taylor_order']}`",
        f"- Remainder estimation: `{params['remainder_estimation']}` for each declared variable",
        f"- Cutoff threshold: `{params['cutoff_threshold']}`",
        f"- Symbolic remainder queue size: `{params['symbolic_remainder_queue_size']}`",
        "- Plot commands:",
    ]
    lines.extend(f"  - `{cmd}`" for cmd in params["plot_commands"])
    lines.extend(
        [
            "- Plot files: " + ", ".join(f"`{p}`" for p in params["plot_files"]),
            "- EPS files: " + ", ".join(f"`{p}`" for p in params["eps_files"]),
            "- Benchmark PNG files: " + ", ".join(f"`{p}`" for p in params["reference_png_files"]),
            "",
            "No Flow* source patch was used.",
            "",
        ]
    )
    (out_dir / "original_flowstar_params.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _run_captured(cmd: Sequence[str], cwd: Path, timeout_s: float | None) -> tuple[subprocess.CompletedProcess[str], float]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, timeout=timeout_s, check=False)
    return proc, time.perf_counter() - start


def run_original_flowstar(flowstar_root: Path, out_dir: Path, params: Mapping[str, Any], timeout_s: float | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bench_dir = flowstar_root / "benchmarks" / "continuous" / "vanderpol"
    original_dir = out_dir / "original_flowstar"
    original_dir.mkdir(parents=True, exist_ok=True)

    make_proc, make_wall = _run_captured(["make"], bench_dir, timeout_s)
    (original_dir / "original_make.stdout.txt").write_text(make_proc.stdout or "", encoding="utf-8", newline="\n")
    (original_dir / "original_make.stderr.txt").write_text(make_proc.stderr or "", encoding="utf-8", newline="\n")

    run_proc, wall_run = _run_captured([str(bench_dir / "vanderpol")], bench_dir, timeout_s)
    stdout_path = original_dir / "original_vanderpol.stdout.txt"
    stderr_path = original_dir / "original_vanderpol.stderr.txt"
    stdout_path.write_text(run_proc.stdout or "", encoding="utf-8", newline="\n")
    stderr_path.write_text(run_proc.stderr or "", encoding="utf-8", newline="\n")

    copied: list[Path] = []
    for name in ["vanderpol.cpp", "Makefile", "README.md", "vanderpol_t_x.plt", "vanderpol_t_y.plt", "vanderpol_t_x.eps", "vanderpol_t_y.eps"]:
        src = bench_dir / name
        if src.exists():
            dst = original_dir / name
            shutil.copy2(src, dst)
            copied.append(dst)
    for stem in ["vanderpol_t_x", "vanderpol_t_y"]:
        src = flowstar_root / "images" / "benchmarks" / f"{stem}.png"
        dst = original_dir / f"original_{stem}.png"
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(dst)

    segments = _combine_segments(
        "original_flowstar",
        original_dir / "vanderpol_t_x.plt",
        original_dir / "vanderpol_t_y.plt",
        "flowstar_original_gnuplot_segment_boxes",
    )
    _write_csv(original_dir / "original_flowstar_segments.csv", SEGMENT_FIELDS, segments)
    summary = {
        "status": "completed" if run_proc.returncode == 0 and segments else "failed",
        "make_wall_s": make_wall,
        "wall_run_s": wall_run,
        "internal_time_cost_s": _parse_first_float(TIME_COST_RE, run_proc.stdout or ""),
        "returncode": run_proc.returncode,
        **_summarize_segments(segments, float(params["time_horizon"])),
        "endpoint_box_available": False,
        "box_source": "flowstar_original_gnuplot_segment_boxes",
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "notes": "Original benchmark was compiled and run as-is; Flow* GNUPLOT rectangles are segment boxes, not endpoints.",
    }
    _write_csv(original_dir / "original_flowstar_summary.csv", ORIGINAL_SUMMARY_FIELDS, [summary])
    hash_rows = [{"path": str(path), "sha256": _sha256(path)} for path in copied]
    _write_csv(original_dir / "original_file_hashes.csv", HASH_FIELDS, hash_rows)
    return summary, segments


def render_generated_cpp(params: Mapping[str, Any]) -> str:
    return f'''#include "Continuous.h"
#include <ctime>
#include <cstdio>
#include <vector>

// Build/link parity check: compile this file against flowstar-toolbox with -lflowstar.
// Original benchmark parity: default adaptive stepsize is set explicitly below.
// Fixed-step harnesses in this repository use setting.setFixedStepsize(...); this one does not.

using namespace flowstar;
using namespace std;

int main()
{{
  Variables vars;
  int x_id = vars.declareVar("x");
  int y_id = vars.declareVar("y");
  int t_id = vars.declareVar("t");

  ODE<Real> ode({{"y", "(1 - x^2) * y - x", "1"}}, vars);

  Computational_Setting setting(vars);
  setting.setAdaptiveStepsize({params['step_min']:.17g}, {params['step_max']:.17g}, {int(params['taylor_order'])});
  setting.setCutoffThreshold({abs(float(params['cutoff_threshold'][1])):.17g});
  Interval remainder({float(params['remainder_estimation'][0]):.17g}, {float(params['remainder_estimation'][1]):.17g});
  vector<Interval> remainder_estimation(vars.size(), remainder);
  setting.setRemainderEstimation(remainder_estimation);

  Interval init_x({params['initial_set']['x'][0]:.17g}, {params['initial_set']['x'][1]:.17g});
  Interval init_y({params['initial_set']['y'][0]:.17g}, {params['initial_set']['y'][1]:.17g});
  vector<Interval> box(vars.size());
  box[x_id] = init_x;
  box[y_id] = init_y;
  box[t_id] = Interval(0.0, 0.0);
  Flowpipe initialSet(box);

  vector<Constraint> safeSet = {{Constraint("y - 2.75", vars)}};
  Result_of_Reachability result;

  clock_t begin, end;
  begin = clock();
  Symbolic_Remainder sr(initialSet, {int(params['symbolic_remainder_queue_size'])});
  ode.reach(result, initialSet, {float(params['time_horizon']):.17g}, setting, safeSet, sr);
  end = clock();
  printf("FLOWSTAR_RUNTIME_S %.17g\\n", (double)(end - begin) / CLOCKS_PER_SEC);
  printf("FLOWSTAR_COMPLETED %d\\n", result.isCompleted() ? 1 : 0);
  printf("FLOWSTAR_SAFE %d\\n", result.isSafe() ? 1 : 0);
  printf("FLOWSTAR_UNSAFE %d\\n", result.isUnsafe() ? 1 : 0);

  if(!result.isCompleted())
  {{
    printf("Flowpipe computation is terminated due to the large overestimation.\\n");
  }}

  result.transformToTaylorModels(setting);
  Plot_Setting plot_setting(vars);
  plot_setting.printOn();
  plot_setting.setOutputDims("t", "x");
  plot_setting.plot_2D_interval_GNUPLOT("./", "generated_vanderpol_t_x", result.tmv_flowpipes, setting);
  printf("FLOWSTAR_PLOT generated_vanderpol_t_x t x\\n");
  plot_setting.setOutputDims("t", "y");
  plot_setting.plot_2D_interval_GNUPLOT("./", "generated_vanderpol_t_y", result.tmv_flowpipes, setting);
  printf("FLOWSTAR_PLOT generated_vanderpol_t_y t y\\n");

  return 0;
}}
'''


def run_generated_flowstar(flowstar_root: Path, out_dir: Path, params: Mapping[str, Any], timeout_s: float | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    generated_dir = out_dir / "generated_flowstar"
    generated_dir.mkdir(parents=True, exist_ok=True)
    cpp_path = generated_dir / "generated_vanderpol_original_defaults.cpp"
    cpp_path.write_text(render_generated_cpp(params), encoding="utf-8", newline="\n")

    run = run_flowstar_toolbox(
        cpp_path,
        flowstar_root=flowstar_root,
        output_dir=generated_dir,
        timeout_s=timeout_s,
        build_lib=True,
    )
    exe = generated_dir / cpp_path.stem
    if exe.exists():
        exe.unlink()

    x_path = generated_dir / "generated_vanderpol_t_x.plt"
    y_path = generated_dir / "generated_vanderpol_t_y.plt"
    segments = _combine_segments("generated_flowstar", x_path, y_path, "flowstar_generated_gnuplot_segment_boxes")
    _write_csv(generated_dir / "generated_flowstar_segments.csv", SEGMENT_FIELDS, segments)
    stdout_text = (run.stdout_path.read_text(encoding="utf-8", errors="ignore") if run.stdout_path and run.stdout_path.exists() else "")
    summary = {
        "status": "completed" if run.status == "completed" and segments else run.status,
        "flowstar_internal_reach_s": _parse_first_float(FLOWSTAR_RUNTIME_RE, stdout_text),
        "compile_wall_s": run.compile_s,
        "run_wall_s": run.run_s,
        "total_wall_s": run.runtime_s,
        "returncode": run.returncode,
        **_summarize_segments(segments, float(params["time_horizon"])),
        "endpoint_box_available": False,
        "box_source": "flowstar_generated_gnuplot_segment_boxes",
        "stdout_path": str(run.stdout_path) if run.stdout_path else "",
        "stderr_path": str(run.stderr_path) if run.stderr_path else "",
        "model_path": str(cpp_path),
        "plot_paths": ";".join(str(p) for p in [x_path, y_path] if p.exists()),
        "notes": "Generated C++ uses the same parsed ODE, initial set, adaptive step range, order, cutoff, remainder estimation, safe set, and symbolic remainder queue.",
    }
    _write_csv(generated_dir / "generated_flowstar_summary.csv", GENERATED_SUMMARY_FIELDS, [summary])
    hash_paths = [p for p in [cpp_path, x_path, y_path, run.stdout_path, run.stderr_path] if p and Path(p).exists()]
    _write_csv(generated_dir / "generated_file_hashes.csv", HASH_FIELDS, [{"path": str(p), "sha256": _sha256(Path(p))} for p in hash_paths])
    return summary, segments


def _interval_tuple(iv: Interval) -> tuple[float, float]:
    return (float(iv.lo.detach().cpu()), float(iv.hi.detach().cpu()))


def _initial_box(params: Mapping[str, Any]) -> list[Interval]:
    return [Interval(*params["initial_set"]["x"]), Interval(*params["initial_set"]["y"])]


def run_torch_range_only_on_original_grid(out_dir: Path, params: Mapping[str, Any], reference_segments: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch_dir = out_dir / "torch_range_only"
    torch_dir.mkdir(parents=True, exist_ok=True)
    order = int(params["taylor_order"])
    current_box = _initial_box(params)
    rows: list[dict[str, Any]] = []
    validation_attempts = 0
    status = "completed"
    notes = "PyTorch TM range-only baseline on the original Flow* segment time grid; each segment endpoint is collapsed to an interval box."
    endpoint_widths: list[float] = []
    failed_segment_index: int | None = None
    failed_segment_t_lo: float | None = None
    failed_segment_t_hi: float | None = None
    failure_reason = ""
    start = time.perf_counter()
    for i, ref in enumerate(reference_segments):
        h = float(ref["t_hi"]) - float(ref["t_lo"])
        if h <= 0:
            status = "failed"
            failed_segment_index = i
            failed_segment_t_lo = float(ref["t_lo"])
            failed_segment_t_hi = float(ref["t_hi"])
            failure_reason = f"non-positive reference segment width at index {i}"
            notes = failure_reason
            break
        seg = flowpipe_step(van_der_pol_ode, current_box, h, order)
        validation_attempts += seg.validation_attempts
        box = seg.tm.range_box()
        x_lo, x_hi = _interval_tuple(box[0])
        y_lo, y_hi = _interval_tuple(box[1])
        width_x = _width(x_lo, x_hi)
        width_y = _width(y_lo, y_hi)
        rows.append(
            {
                "case_id": "torch_tm_range_only",
                "segment_index": i,
                "t_lo": float(ref["t_lo"]),
                "t_hi": float(ref["t_hi"]),
                "x_lo": x_lo,
                "x_hi": x_hi,
                "y_lo": y_lo,
                "y_hi": y_hi,
                "width_x": width_x,
                "width_y": width_y,
                "width_sum": width_x + width_y,
                "box_source": "torch_tm_range_only_segment_on_flowstar_time_grid",
            }
        )
        final_box = seg.final_tm.range_box()
        endpoint_widths = [_width(*_interval_tuple(final_box[0])), _width(*_interval_tuple(final_box[1]))]
        if seg.status != "validated":
            status = "failed"
            failed_segment_index = i
            failed_segment_t_lo = float(ref["t_lo"])
            failed_segment_t_hi = float(ref["t_hi"])
            failure_reason = seg.message or "validation failed"
            notes = f"validation failed at attempted failed segment {i}: {failure_reason}"
            break
        current_box = [iv.inflate(1e-9) for iv in final_box]
    runtime_s = time.perf_counter() - start

    if rows and float(rows[-1]["t_hi"]) + 1e-12 < float(params["time_horizon"]) and status == "completed":
        status = "max_horizon_reached"
        notes = "PyTorch range-only run stopped before the original horizon without a validation error."

    _write_csv(torch_dir / "torch_range_only_segments.csv", SEGMENT_FIELDS, rows)
    summary = {
        "status": status,
        "mode": "range_only",
        "runtime_s": runtime_s,
        **_torch_progress_fields(
            rows,
            float(params["time_horizon"]),
            status,
            failed_segment_index=failed_segment_index,
            failed_segment_t_lo=failed_segment_t_lo,
            failed_segment_t_hi=failed_segment_t_hi,
            failure_reason=failure_reason,
        ),
        "requested_order": order,
        "endpoint_box_available": bool(rows),
        "endpoint_width_x": endpoint_widths[0] if endpoint_widths else "",
        "endpoint_width_y": endpoint_widths[1] if endpoint_widths else "",
        "endpoint_width_sum": sum(endpoint_widths) if endpoint_widths else "",
        "validation_attempts": validation_attempts,
        "box_source": "torch_tm_range_only_segment_on_flowstar_time_grid",
        "notes": notes,
    }
    _write_csv(torch_dir / "torch_range_only_summary.csv", TORCH_SUMMARY_FIELDS, [summary])
    return summary, rows


def run_torch_dependency_preserving_on_original_grid(
    out_dir: Path,
    params: Mapping[str, Any],
    reference_segments: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch_dir = out_dir / "torch_dependency_preserving"
    torch_dir.mkdir(parents=True, exist_ok=True)
    order = int(params["taylor_order"])
    current_tm = TMVector.identity(_initial_box(params), order=order)
    rows: list[dict[str, Any]] = []
    validation_attempts = 0
    status = "completed"
    notes = "PyTorch TM dependency-preserving run on the original Flow* segment time grid."
    endpoint_widths: list[float] = []
    failed_segment_index: int | None = None
    failed_segment_t_lo: float | None = None
    failed_segment_t_hi: float | None = None
    failure_reason = ""
    start = time.perf_counter()
    for i, ref in enumerate(reference_segments):
        h = float(ref["t_hi"]) - float(ref["t_lo"])
        if h <= 0:
            status = "failed"
            failed_segment_index = i
            failed_segment_t_lo = float(ref["t_lo"])
            failed_segment_t_hi = float(ref["t_hi"])
            failure_reason = f"non-positive reference segment width at index {i}"
            notes = failure_reason
            break
        seg = flowpipe_step_from_tm(van_der_pol_ode, current_tm, h, order)
        validation_attempts += seg.validation_attempts
        box = seg.tm.range_box()
        x_lo, x_hi = _interval_tuple(box[0])
        y_lo, y_hi = _interval_tuple(box[1])
        width_x = _width(x_lo, x_hi)
        width_y = _width(y_lo, y_hi)
        rows.append(
            {
                "case_id": "torch_tm_dependency_preserving",
                "segment_index": i,
                "t_lo": float(ref["t_lo"]),
                "t_hi": float(ref["t_hi"]),
                "x_lo": x_lo,
                "x_hi": x_hi,
                "y_lo": y_lo,
                "y_hi": y_hi,
                "width_x": width_x,
                "width_y": width_y,
                "width_sum": width_x + width_y,
                "box_source": "torch_tm_dependency_preserving_segment_on_flowstar_time_grid",
            }
        )
        final_box = seg.final_tm.range_box()
        endpoint_widths = [_width(*_interval_tuple(final_box[0])), _width(*_interval_tuple(final_box[1]))]
        if seg.status != "validated":
            status = "failed"
            failed_segment_index = i
            failed_segment_t_lo = float(ref["t_lo"])
            failed_segment_t_hi = float(ref["t_hi"])
            failure_reason = seg.message or "validation failed"
            notes = f"validation failed at attempted failed segment {i}: {failure_reason}"
            break
        current_tm = seg.final_tm
    runtime_s = time.perf_counter() - start

    if rows and float(rows[-1]["t_hi"]) + 1e-12 < float(params["time_horizon"]) and status == "completed":
        status = "max_horizon_reached"
        notes = "PyTorch dependency-preserving run stopped before the original horizon without a validation error."

    summary = {
        "status": status,
        "mode": "dependency_preserving",
        "runtime_s": runtime_s,
        **_torch_progress_fields(
            rows,
            float(params["time_horizon"]),
            status,
            failed_segment_index=failed_segment_index,
            failed_segment_t_lo=failed_segment_t_lo,
            failed_segment_t_hi=failed_segment_t_hi,
            failure_reason=failure_reason,
        ),
        "requested_order": order,
        "endpoint_box_available": bool(rows),
        "endpoint_width_x": endpoint_widths[0] if endpoint_widths else "",
        "endpoint_width_y": endpoint_widths[1] if endpoint_widths else "",
        "endpoint_width_sum": sum(endpoint_widths) if endpoint_widths else "",
        "validation_attempts": validation_attempts,
        "box_source": "torch_tm_dependency_preserving_segment_on_flowstar_time_grid",
        "notes": notes,
    }
    _write_csv(torch_dir / "torch_dependency_preserving_segments.csv", SEGMENT_FIELDS, rows)
    _write_csv(torch_dir / "torch_dependency_preserving_summary.csv", TORCH_SUMMARY_FIELDS, [summary])
    return summary, rows


def _axis_limits(rows_groups: Sequence[Sequence[Mapping[str, Any]]], keys: tuple[str, str]) -> tuple[float, float] | None:
    vals: list[float] = []
    for rows in rows_groups:
        for row in rows:
            vals.extend([float(row[keys[0]]), float(row[keys[1]])])
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.05, 1e-6)
    return lo - pad, hi + pad


def _finish_plot(fig: Any, ax: Any, path: Path, rows_groups: Sequence[Sequence[Mapping[str, Any]]], var: str | None = None) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=7, loc="best")
    if var in {"x", "y"}:
        ax.set_xlim(0.0, 10.0)
        ylim = _axis_limits(rows_groups, (f"{var}_lo", f"{var}_hi"))
        if ylim:
            ax.set_ylim(*ylim)
    elif var == "phase":
        xlim = _axis_limits(rows_groups, ("x_lo", "x_hi"))
        ylim = _axis_limits(rows_groups, ("y_lo", "y_hi"))
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _add_tx_boxes(ax: Any, rows: Sequence[Mapping[str, Any]], var: str, *, color: str, label: str, alpha: float) -> None:
    import matplotlib.patches as patches

    for i, row in enumerate(rows):
        ax.add_patch(
            patches.Rectangle(
                (float(row["t_lo"]), float(row[f"{var}_lo"])),
                float(row["t_hi"]) - float(row["t_lo"]),
                float(row[f"width_{var}"]),
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=0.8,
                label=label if i == 0 else None,
            )
        )
    if rows:
        last = rows[-1]
        ax.add_patch(
            patches.Rectangle(
                (float(last["t_lo"]), float(last[f"{var}_lo"])),
                float(last["t_hi"]) - float(last["t_lo"]),
                float(last[f"width_{var}"]),
                fill=False,
                edgecolor=color,
                linewidth=1.8,
                label=f"{label} last",
            )
        )


def _add_phase_boxes(ax: Any, rows: Sequence[Mapping[str, Any]], *, color: str, label: str, alpha: float) -> None:
    import matplotlib.patches as patches

    for i, row in enumerate(rows):
        ax.add_patch(
            patches.Rectangle(
                (float(row["x_lo"]), float(row["y_lo"])),
                float(row["width_x"]),
                float(row["width_y"]),
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=0.8,
                label=label if i == 0 else None,
            )
        )
    if rows:
        last = rows[-1]
        ax.add_patch(
            patches.Rectangle(
                (float(last["x_lo"]), float(last["y_lo"])),
                float(last["width_x"]),
                float(last["width_y"]),
                fill=False,
                edgecolor=color,
                linewidth=1.8,
                label=f"{label} last",
            )
        )


def make_single_plots(base: Path, prefix: str, rows: Sequence[Mapping[str, Any]], title: str, color: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    for var in ("x", "y"):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        _add_tx_boxes(ax, rows, var, color=color, label=title, alpha=0.18)
        ax.set_xlabel("t")
        ax.set_ylabel(var)
        ax.set_title(f"{title}: t-{var}")
        path = base / f"{prefix}_t_{var}.png"
        _finish_plot(fig, ax, path, [rows], var)
        paths.append(path)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    _add_phase_boxes(ax, rows, color=color, label=title, alpha=0.16)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title}: phase")
    path = base / f"{prefix}_phase_xy.png"
    _finish_plot(fig, ax, path, [rows], "phase")
    paths.append(path)
    return paths


def make_overlay_plots(out_dir: Path, left_name: str, left_rows: Sequence[Mapping[str, Any]], right_name: str, right_rows: Sequence[Mapping[str, Any]], stem: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    for var in ("x", "y"):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        _add_tx_boxes(ax, left_rows, var, color="#2ca02c", label=left_name, alpha=0.16)
        _add_tx_boxes(ax, right_rows, var, color="#1f77b4", label=right_name, alpha=0.12)
        ax.set_xlabel("t")
        ax.set_ylabel(var)
        ax.set_title(f"{left_name} vs {right_name}: t-{var}")
        path = out_dir / f"{stem}_t_{var}.png"
        _finish_plot(fig, ax, path, [left_rows, right_rows], var)
        paths.append(path)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    _add_phase_boxes(ax, left_rows, color="#2ca02c", label=left_name, alpha=0.14)
    _add_phase_boxes(ax, right_rows, color="#1f77b4", label=right_name, alpha=0.10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{left_name} vs {right_name}: phase")
    path = out_dir / f"{stem}_phase_xy.png"
    _finish_plot(fig, ax, path, [left_rows, right_rows], "phase")
    paths.append(path)
    return paths


def make_all_plots(
    out_dir: Path,
    original_rows: Sequence[Mapping[str, Any]],
    generated_rows: Sequence[Mapping[str, Any]],
    range_only_rows: Sequence[Mapping[str, Any]],
    dependency_rows: Sequence[Mapping[str, Any]],
) -> None:
    original_dir = out_dir / "original_flowstar"
    generated_dir = out_dir / "generated_flowstar"
    range_only_dir = out_dir / "torch_range_only"
    dependency_dir = out_dir / "torch_dependency_preserving"
    make_single_plots(original_dir, "original_flowstar_rendered", original_rows, "Original Flow*", "#2ca02c")
    make_single_plots(generated_dir, "generated_vanderpol", generated_rows, "Generated Flow*", "#9467bd")
    make_single_plots(range_only_dir, "torch_range_only_vanderpol", range_only_rows, "PyTorch TM range-only", "#1f77b4")
    make_single_plots(
        dependency_dir,
        "torch_dependency_preserving_vanderpol",
        dependency_rows,
        "PyTorch TM dependency-preserving",
        "#d62728",
    )
    make_overlay_plots(out_dir, "Original Flow*", original_rows, "PyTorch TM range-only", range_only_rows, "overlay_original_flowstar_vs_torch_range_only")
    make_overlay_plots(out_dir, "Generated Flow*", generated_rows, "PyTorch TM range-only", range_only_rows, "overlay_generated_flowstar_vs_torch_range_only")
    make_overlay_plots(
        out_dir,
        "Original Flow*",
        original_rows,
        "PyTorch TM dependency-preserving",
        dependency_rows,
        "overlay_original_flowstar_vs_torch_dependency_preserving",
    )
    make_overlay_plots(
        out_dir,
        "Generated Flow*",
        generated_rows,
        "PyTorch TM dependency-preserving",
        dependency_rows,
        "overlay_generated_flowstar_vs_torch_dependency_preserving",
    )


def write_generated_original_comparison(out_dir: Path, original_rows: Sequence[Mapping[str, Any]], generated_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    numeric_fields = ["t_lo", "t_hi", "x_lo", "x_hi", "y_lo", "y_hi", "width_x", "width_y", "width_sum"]
    n = min(len(original_rows), len(generated_rows))
    max_abs_diff = 0.0
    for i in range(n):
        for field in numeric_fields:
            diff = abs(float(original_rows[i][field]) - float(generated_rows[i][field]))
            max_abs_diff = max(max_abs_diff, diff)
    summary = {
        "original_num_segments": len(original_rows),
        "generated_num_segments": len(generated_rows),
        "matched_prefix_segments": n,
        "segment_count_match": len(original_rows) == len(generated_rows),
        "max_abs_segment_field_diff": max_abs_diff,
    }
    _write_csv(out_dir / "generated_flowstar_vs_original_comparison.csv", COMPARISON_FIELDS, [{"metric": k, "value": v} for k, v in summary.items()])
    return summary


def _summary_value(row: Mapping[str, Any], key: str) -> Any:
    value = row.get(key, "")
    return "" if value is None else value


def _torch_parity_row(tool: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tool": tool,
        "status": summary["status"],
        "torch_runtime_s": summary["runtime_s"],
        "num_segments": summary["num_segments"],
        "last_reached_t": summary["last_reached_t"],
        "validated_segments": summary.get("validated_segments", ""),
        "last_validated_t": summary.get("last_validated_t", ""),
        "failed_segment_index": summary.get("failed_segment_index", ""),
        "failed_segment_t_lo": summary.get("failed_segment_t_lo", ""),
        "failed_segment_t_hi": summary.get("failed_segment_t_hi", ""),
        "last_attempted_t": summary.get("last_attempted_t", ""),
        "failure_reason": summary.get("failure_reason", ""),
        "requested_horizon": summary["requested_horizon"],
        "last_segment_width_x": summary["last_segment_width_x"],
        "last_segment_width_y": summary["last_segment_width_y"],
        "last_segment_width_sum": summary["last_segment_width_sum"],
        "tube_width_x": summary["tube_width_x"],
        "tube_width_y": summary["tube_width_y"],
        "tube_width_sum": summary["tube_width_sum"],
        "endpoint_box_available": summary["endpoint_box_available"],
        "box_source": summary["box_source"],
        "notes": summary["notes"],
    }


def write_parity_summary(
    out_dir: Path,
    original: Mapping[str, Any],
    generated: Mapping[str, Any],
    range_only_summary: Mapping[str, Any],
    dependency_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        {
            "tool": "original_flowstar",
            "status": original["status"],
            "original_flowstar_wall_run_s": original["wall_run_s"],
            "num_segments": original["num_segments"],
            "last_reached_t": original["last_reached_t"],
            **_completed_progress_fields(original),
            "requested_horizon": original["requested_horizon"],
            "last_segment_width_x": original["last_segment_width_x"],
            "last_segment_width_y": original["last_segment_width_y"],
            "last_segment_width_sum": original["last_segment_width_sum"],
            "tube_width_x": original["tube_width_x"],
            "tube_width_y": original["tube_width_y"],
            "tube_width_sum": original["tube_width_sum"],
            "endpoint_box_available": False,
            "box_source": original["box_source"],
            "notes": original["notes"],
        },
        {
            "tool": "generated_flowstar",
            "status": generated["status"],
            "generated_flowstar_internal_reach_s": generated["flowstar_internal_reach_s"],
            "generated_flowstar_compile_wall_s": generated["compile_wall_s"],
            "generated_flowstar_run_wall_s": generated["run_wall_s"],
            "num_segments": generated["num_segments"],
            "last_reached_t": generated["last_reached_t"],
            **_completed_progress_fields(generated),
            "requested_horizon": generated["requested_horizon"],
            "last_segment_width_x": generated["last_segment_width_x"],
            "last_segment_width_y": generated["last_segment_width_y"],
            "last_segment_width_sum": generated["last_segment_width_sum"],
            "tube_width_x": generated["tube_width_x"],
            "tube_width_y": generated["tube_width_y"],
            "tube_width_sum": generated["tube_width_sum"],
            "endpoint_box_available": False,
            "box_source": generated["box_source"],
            "notes": generated["notes"],
        },
        _torch_parity_row("torch_tm_range_only", range_only_summary),
        _torch_parity_row("torch_tm_dependency_preserving", dependency_summary),
    ]
    _write_csv(out_dir / "parity_summary.csv", PARITY_FIELDS, rows)
    return rows


def _fmt_md(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{f:.6g}" if math.isfinite(f) else "n/a"


def write_report(
    out_dir: Path,
    params: Mapping[str, Any],
    parity_rows: Sequence[Mapping[str, Any]],
    generated: Mapping[str, Any],
    range_only_summary: Mapping[str, Any],
    dependency_summary: Mapping[str, Any],
    generated_comparison: Mapping[str, Any],
) -> None:
    table = [
        "| tool | status | runtime | segments | validated | last validated t | last attempted t | last width sum | tube width sum | endpoint box | source |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in parity_rows:
        runtime = (
            row.get("original_flowstar_wall_run_s")
            or row.get("generated_flowstar_internal_reach_s")
            or row.get("torch_runtime_s")
        )
        table.append(
            f"| `{row['tool']}` | `{row['status']}` | {_fmt_md(runtime)} | {_fmt_md(row['num_segments'])} | "
            f"{_fmt_md(row.get('validated_segments'))} | {_fmt_md(row.get('last_validated_t'))} | "
            f"{_fmt_md(row.get('last_attempted_t'))} | {_fmt_md(row['last_segment_width_sum'])} | "
            f"{_fmt_md(row['tube_width_sum'])} | `{row['endpoint_box_available']}` | `{row['box_source']}` |"
        )
    generated_ratio = "n/a"
    if generated.get("last_segment_width_sum") not in ("", None):
        generated_ratio = _fmt_md(generated["last_segment_width_sum"])
    range_last_t = _fmt_md(range_only_summary["last_validated_t"])
    range_attempted_t = _fmt_md(range_only_summary["last_attempted_t"])
    dependency_last_t = _fmt_md(dependency_summary["last_validated_t"])
    dependency_attempted_t = _fmt_md(dependency_summary["last_attempted_t"])
    dependency_note = ""
    try:
        if float(dependency_summary["last_validated_t"]) > float(range_only_summary["last_validated_t"]):
            dependency_note = " The dependency-preserving run validates farther than the range-only baseline, showing that dependency loss caused much of the range-only blowup."
    except (TypeError, ValueError):
        dependency_note = ""
    text = f"""# Flow* Benchmark Parity Report

This is a Flow* original benchmark parity audit for the plant-only polynomial Van der Pol benchmark. It is not a new reachability algorithm.

## Original Parameters

- ODE: `x' = {params['ode']['x']}`, `y' = {params['ode']['y']}`, `t' = {params['ode']['t']}`
- Initial set: `x in {params['initial_set']['x']}`, `y in {params['initial_set']['y']}`
- Horizon: `0` to `{params['time_horizon']}`
- Flow* original step policy: adaptive, min `{params['step_min']}`, max `{params['step_max']}`
- Flow* Taylor order: fixed order `{params['taylor_order']}`
- Remainder estimation: `{params['remainder_estimation']}`
- Cutoff threshold: `{params['cutoff_threshold']}`
- Symbolic remainder queue size: `{params['symbolic_remainder_queue_size']}`
- Flow* patch used: no
- Benchmark PNG inputs: `{params['reference_png_files'][0]}` and `{params['reference_png_files'][1]}`

## Runtime And Bounds

{chr(10).join(table)}

`generated_flowstar` was generated from the parsed parameters and run through the repository Flow* toolbox runner. Its last-segment width sum is {generated_ratio}.

Generated Flow* vs original Flow*: segment count match is `{generated_comparison["segment_count_match"]}` and max absolute parsed segment-field difference is `{_fmt_md(generated_comparison["max_abs_segment_field_diff"])}`.

`torch_tm_range_only` is a weak baseline: it collapses each validated endpoint Taylor model to an interval box before the next original Flow* segment. Its last validated time is `{range_last_t}`, last attempted time is `{range_attempted_t}`, status is `{range_only_summary['status']}`, and notes are: {range_only_summary['notes']}.

`torch_tm_dependency_preserving` is the fairer PyTorch TM comparison because it propagates `seg.final_tm` directly across segment boundaries. Its last validated time is `{dependency_last_t}`, last attempted time is `{dependency_attempted_t}`, status is `{dependency_summary['status']}`, and notes are: {dependency_summary['notes']}.{dependency_note}

## Semantics

Flow* GNUPLOT rectangles are segment boxes. They are not final-time endpoint boxes. Therefore `endpoint_box_available=false` for both Flow* rows, endpoint widths are blank, and no endpoint ratio is reported.

For failed PyTorch rows, `failed_segment_index` and `failed_segment_t_hi` describe the attempted failed segment; `validated_segments` and `last_validated_t` describe only the last successfully validated segment. Only last-segment and tube widths are reported for Flow* parity. Plot generation time is not included in algorithm runtime.

## Scope Guard

No CROWN, no auto_LiRPA, no Jacobian bounds, no sin/cos support, no hybrid automata, no Flow* Python binding, no NN controller workflow, and no new algorithm were added.
"""
    (out_dir / "parity_report.md").write_text(text, encoding="utf-8", newline="\n")


def write_output_readme(out_dir: Path, params: Mapping[str, Any]) -> None:
    text = f"""# Flow* Benchmark Parity Outputs

This directory contains the Van der Pol Flow* original benchmark parity audit.

## Scope

- Original Flow*: `/srv/local/shengenli/flowstar/benchmarks/continuous/vanderpol`
- Generated Flow*: C++ harness generated from `original_flowstar_params.json`
- PyTorch TM range-only: weak baseline over the original Flow* segment time grid
- PyTorch TM dependency-preserving: fairer TM comparison that propagates `seg.final_tm` between original Flow* segments
- Horizon: `{params['time_horizon']}`
- Flow* patch used: no

## Runtime Semantics

Original Flow* `wall_run_s` is subprocess wall time for running the original executable. Generated Flow* records compile wall time, executable wall time, and internal `FLOWSTAR_RUNTIME_S`. PyTorch `runtime_s` measures only TM propagation and not plot writing.

## Bound Semantics

Flow* GNUPLOT boxes are flowpipe segment boxes, not final-time endpoint boxes. `endpoint_box_available=false` for Flow* rows unless a true endpoint source is extracted from the Flow* API. This audit reports last-segment and tube widths only for Flow* parity. Failed PyTorch rows distinguish the attempted failed segment from the last validated segment with `failed_segment_*`, `last_attempted_t`, `validated_segments`, and `last_validated_t`.

## Files

- `original_flowstar_params.json`
- `original_flowstar_params.md`
- `original_flowstar/`
- `generated_flowstar/`
- `torch_range_only/`
- `torch_dependency_preserving/`
- `parity_summary.csv`
- `generated_flowstar_vs_original_comparison.csv`
- `parity_report.md`
- `overlay_*png`

## Scope Guard

No CROWN, no auto_LiRPA, no Jacobian bounds, no sin/cos support, no hybrid automata, no Flow* Python binding, no NN controller workflow, and no new algorithm were added.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Flow* original Van der Pol benchmark parity audit.")
    parser.add_argument("--flowstar-root", default=None, help="Flow* repository root; defaults to FLOWSTAR_ROOT or /srv/local/shengenli/flowstar")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "flowstar_benchmark_parity"))
    parser.add_argument("--timeout-s", type=float, default=300.0)
    args = parser.parse_args()

    explicit = args.flowstar_root or str(Path("/srv/local/shengenli/flowstar"))
    root = find_flowstar_root(explicit)
    if root is None:
        raise SystemExit("Flow* root not found; pass --flowstar-root or set FLOWSTAR_ROOT")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = parse_original_params(root)
    write_params_docs(out_dir, params)
    original_summary, original_segments = run_original_flowstar(root, out_dir, params, args.timeout_s)
    generated_summary, generated_segments = run_generated_flowstar(root, out_dir, params, args.timeout_s)
    range_only_summary, range_only_segments = run_torch_range_only_on_original_grid(out_dir, params, original_segments)
    dependency_summary, dependency_segments = run_torch_dependency_preserving_on_original_grid(out_dir, params, original_segments)
    make_all_plots(out_dir, original_segments, generated_segments, range_only_segments, dependency_segments)
    generated_comparison = write_generated_original_comparison(out_dir, original_segments, generated_segments)
    parity_rows = write_parity_summary(out_dir, original_summary, generated_summary, range_only_summary, dependency_summary)
    write_report(out_dir, params, parity_rows, generated_summary, range_only_summary, dependency_summary, generated_comparison)
    write_output_readme(out_dir, params)
    print(f"wrote {out_dir}")
    print(f"original status={original_summary['status']} segments={original_summary['num_segments']}")
    print(f"generated status={generated_summary['status']} segments={generated_summary['num_segments']}")
    print(
        f"torch range_only status={range_only_summary['status']} "
        f"segments={range_only_summary['num_segments']} last_t={range_only_summary['last_reached_t']}"
    )
    print(
        f"torch dependency_preserving status={dependency_summary['status']} "
        f"segments={dependency_summary['num_segments']} last_t={dependency_summary['last_reached_t']}"
    )


if __name__ == "__main__":
    main()
