#!/usr/bin/env python3
"""Local one-step Flow* oracle for the Flowstar-style Van der Pol rescue."""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
FLOWSTAR_RUNNER_ROOT = REPO_ROOT / "comparisons" / "flowstar"
LOCAL_FLOWSTAR_ROOT = REPO_ROOT.parent / "flowstar"
for p in (SRC_ROOT, FLOWSTAR_RUNNER_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_flowstar import run_flowstar_toolbox  # noqa: E402

SOURCE_RUN_ID = "flowstar_style_o6_candidate8_output6_cutoff"
SOURCE_DIR = REPO_ROOT / "outputs" / "flowstar_style_candidate_order"
RUN_ID = "flowstar_one_step_oracle_candidate8_cutoff"
NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
NUMERIC_PAIR_RE = re.compile(rf"^\s*(?P<t>{NUMBER_RE})\s+(?P<v>{NUMBER_RE})(?:\s|$)")
COMPLETED_RE = re.compile(r"FLOWSTAR_COMPLETED\s+(?P<ok>[01])")
RUNTIME_RE = re.compile(rf"FLOWSTAR_RUNTIME_S\s+(?P<runtime>{NUMBER_RE})")

SUMMARY_FIELDS = [
    "run_id",
    "source_run_id",
    "reset_segment_index",
    "t_lo",
    "t_hi",
    "h_try",
    "status",
    "flowstar_validated",
    "pytorch_validated",
    "flowstar_orders",
    "flowstar_best_order",
    "flowstar_last_width_sum",
    "pytorch_candidate_final_width_sum",
    "width_ratio_flowstar_over_pytorch",
    "flowstar_runtime_s",
    "failure_reason",
    "notes",
]

FLOWSTAR_SEGMENT_FIELDS = [
    "run_id",
    "order",
    "flowstar_status",
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

PYTORCH_ATTEMPT_FIELDS = [
    "run_id",
    "source_run_id",
    "segment_index",
    "t_lo",
    "t_hi",
    "h_try",
    "x_lo",
    "x_hi",
    "y_lo",
    "y_hi",
    "order",
    "candidate_order",
    "output_order",
    "validation_status",
    "subset_result",
    "candidate_final_width_sum",
    "residual_width_sum",
    "rejection_reason",
    "reset_box_source",
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


def _candidate_source_paths() -> tuple[Path, Path, Path]:
    return (
        SOURCE_DIR / "rescue_reset_boxes.csv",
        SOURCE_DIR / "rescue_segments.csv",
        SOURCE_DIR / "rescue_validation_attempts.csv",
    )


def _load_last_reset_box() -> dict[str, Any]:
    reset_path, segments_path, _attempts_path = _candidate_source_paths()
    rows = [row for row in _read_csv(reset_path) if row.get("run_id") == SOURCE_RUN_ID]
    if not rows:
        rows = [
            {
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
                "reset_box_source": "derived_from_rescue_segments",
            }
            for row in _read_csv(segments_path)
            if row.get("run_id") == SOURCE_RUN_ID and row.get("status") == "validated"
        ]
    if not rows:
        raise FileNotFoundError(f"no reset/segment rows found for {SOURCE_RUN_ID} in {SOURCE_DIR}")
    rows.sort(key=lambda r: (_finite_float(r.get("t_hi")) or 0.0, _finite_float(r.get("segment_index")) or 0.0))
    return rows[-1]


def _load_failed_attempt(reset_box: Mapping[str, Any]) -> dict[str, Any]:
    _reset_path, _segments_path, attempts_path = _candidate_source_paths()
    attempts = [row for row in _read_csv(attempts_path) if row.get("run_id") == SOURCE_RUN_ID]
    failed = [row for row in attempts if row.get("validation_status") == "failed"]
    reset_t = _finite_float(reset_box.get("t_hi"))
    if reset_t is not None:
        near = [row for row in failed if abs((_finite_float(row.get("t_lo")) or -1.0) - reset_t) <= 1e-9]
        if near:
            failed = near
    if not failed:
        raise FileNotFoundError(f"no failed attempt rows found for {SOURCE_RUN_ID} in {attempts_path}")
    failed.sort(key=lambda r: (_finite_float(r.get("t_lo")) or 0.0, _finite_float(r.get("adaptive_attempt_index")) or 0.0, _finite_float(r.get("attempt_index")) or 0.0))
    return failed[-1]


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
    segments: list[tuple[float, float, float, float]] = []
    for block in blocks:
        ts = [p[0] for p in block]
        vals = [p[1] for p in block]
        segments.append((min(ts), max(ts), min(vals), max(vals)))
    return segments


def _combine_segments(run_id: str, order: int, status: str, x_path: Path, y_path: Path) -> list[dict[str, Any]]:
    x_segments = _parse_gnuplot_segments(x_path)
    y_segments = _parse_gnuplot_segments(y_path)
    rows: list[dict[str, Any]] = []
    for i, (x_seg, y_seg) in enumerate(zip(x_segments, y_segments)):
        x_t_lo, x_t_hi, x_lo, x_hi = x_seg
        y_t_lo, y_t_hi, y_lo, y_hi = y_seg
        width_x = max(0.0, x_hi - x_lo)
        width_y = max(0.0, y_hi - y_lo)
        rows.append(
            {
                "run_id": run_id,
                "order": order,
                "flowstar_status": status,
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
                "box_source": "flowstar_gnuplot_segment_box",
            }
        )
    return rows


def _render_cpp(order: int, h: float, reset_box: Mapping[str, Any]) -> str:
    stem = f"oracle_flowstar_o{order}"
    x_lo = float(reset_box["x_lo"])
    x_hi = float(reset_box["x_hi"])
    y_lo = float(reset_box["y_lo"])
    y_hi = float(reset_box["y_hi"])
    return f"""#include "Continuous.h"
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
  setting.setFixedStepsize({h:.17g}, {order});
  setting.setCutoffThreshold(1e-10);
  vector<Interval> remainder_estimation(vars.size());
  for(unsigned int i = 0; i < vars.size(); ++i)
  {{
    remainder_estimation[i] = Interval(-0.0001, 0.0001);
  }}
  setting.setRemainderEstimation(remainder_estimation);
  setting.printOn();

  vector<Interval> box(vars.size());
  box[x_id] = Interval({x_lo:.17g}, {x_hi:.17g});
  box[y_id] = Interval({y_lo:.17g}, {y_hi:.17g});
  box[t_id] = Interval(0.0, 0.0);
  Flowpipe initialSet(box);
  vector<Constraint> safeSet;
  Result_of_Reachability result;

  clock_t begin, end;
  begin = clock();
  ode.reach(result, initialSet, {h:.17g}, setting, safeSet);
  end = clock();
  printf("FLOWSTAR_RUNTIME_S %.17g\\n", (double)(end - begin) / CLOCKS_PER_SEC);
  printf("FLOWSTAR_COMPLETED %d\\n", result.isCompleted() ? 1 : 0);
  printf("FLOWSTAR_SAFE %d\\n", result.isSafe() ? 1 : 0);
  printf("FLOWSTAR_UNSAFE %d\\n", result.isUnsafe() ? 1 : 0);

  result.transformToTaylorModels(setting);
  Plot_Setting plot_setting(vars);
  plot_setting.printOn();
  plot_setting.setOutputDims("t", "x");
  plot_setting.plot_2D_interval_GNUPLOT("./", "{stem}_t_x", result.tmv_flowpipes, setting);
  printf("FLOWSTAR_PLOT {stem}_t_x t x\\n");
  plot_setting.setOutputDims("t", "y");
  plot_setting.plot_2D_interval_GNUPLOT("./", "{stem}_t_y", result.tmv_flowpipes, setting);
  printf("FLOWSTAR_PLOT {stem}_t_y t y\\n");
  return 0;
}}
"""


def _stdout_value(pattern: re.Pattern[str], path: Path | None, group: str) -> str:
    if path is None or not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = list(pattern.finditer(text))
    return matches[-1].group(group) if matches else ""


def _run_flowstar_orders(out_dir: Path, reset_box: Mapping[str, Any], h_try: float, orders: Sequence[int], flowstar_root: str | None, timeout_s: float | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_dir = out_dir / "flowstar_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    for order in orders:
        cpp_path = model_dir / f"oracle_flowstar_o{order}.cpp"
        cpp_path.write_text(_render_cpp(order, h_try, reset_box), encoding="utf-8", newline="\n")
        run = run_flowstar_toolbox(cpp_path, flowstar_root=flowstar_root, output_dir=model_dir, timeout_s=timeout_s, build_lib=True)
        completed_flag = _stdout_value(COMPLETED_RE, run.stdout_path, "ok")
        runtime = _stdout_value(RUNTIME_RE, run.stdout_path, "runtime")
        if run.status == "completed" and completed_flag == "1":
            status = "completed"
        elif run.status == "completed" and completed_flag == "0":
            status = "not_completed"
        else:
            status = run.status
        failure_reason = run.message or ("Flow* reach did not complete; no segment boxes emitted" if status == "not_completed" else "")
        x_path = model_dir / f"oracle_flowstar_o{order}_t_x.plt"
        y_path = model_dir / f"oracle_flowstar_o{order}_t_y.plt"
        order_segments = _combine_segments(RUN_ID, order, status, x_path, y_path)
        segments.extend(order_segments)
        last = order_segments[-1] if order_segments else {}
        summaries.append(
            {
                "order": order,
                "status": status,
                "flowstar_validated": status == "completed" and bool(order_segments),
                "runtime_s": _finite_float(runtime) if runtime else run.runtime_s,
                "last_width_sum": last.get("width_sum", ""),
                "num_segments": len(order_segments),
                "failure_reason": failure_reason,
            }
        )
    return summaries, segments


def run_oracle(out_dir: Path, *, flowstar_root: str | None = None, timeout_s: float | None = 600.0, orders: Sequence[int] = (4, 6, 8)) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    reset_box = _load_last_reset_box()
    failed_attempt = _load_failed_attempt(reset_box)
    h_try = _finite_float(failed_attempt.get("h_try")) or _finite_float(failed_attempt.get("h"))
    if h_try is None:
        raise ValueError("failed attempt does not contain h_try/h")
    if flowstar_root is None and (LOCAL_FLOWSTAR_ROOT / "flowstar-toolbox" / "Continuous.h").exists():
        flowstar_root = str(LOCAL_FLOWSTAR_ROOT)
    py_row = {
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "segment_index": reset_box.get("segment_index", ""),
        "t_lo": reset_box.get("t_hi", ""),
        "t_hi": (_finite_float(reset_box.get("t_hi")) or 0.0) + h_try,
        "h_try": h_try,
        "x_lo": reset_box.get("x_lo", ""),
        "x_hi": reset_box.get("x_hi", ""),
        "y_lo": reset_box.get("y_lo", ""),
        "y_hi": reset_box.get("y_hi", ""),
        "order": failed_attempt.get("order", ""),
        "candidate_order": failed_attempt.get("candidate_order", ""),
        "output_order": failed_attempt.get("output_order", ""),
        "validation_status": failed_attempt.get("validation_status", ""),
        "subset_result": failed_attempt.get("subset_result", ""),
        "candidate_final_width_sum": failed_attempt.get("candidate_final_width_sum", ""),
        "residual_width_sum": failed_attempt.get("residual_width_sum", ""),
        "rejection_reason": failed_attempt.get("rejection_reason", ""),
        "reset_box_source": reset_box.get("reset_box_source", ""),
    }
    order_summaries, flowstar_segments = _run_flowstar_orders(out_dir, reset_box, h_try, orders, flowstar_root, timeout_s)
    validated = [row for row in order_summaries if row.get("flowstar_validated")]
    best = min(validated, key=lambda r: int(r["order"]), default=(order_summaries[-1] if order_summaries else {}))
    py_width = _finite_float(py_row.get("candidate_final_width_sum"))
    flow_width = _finite_float(best.get("last_width_sum"))
    flowstar_validated = bool(validated)
    summary = {
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "reset_segment_index": reset_box.get("segment_index", ""),
        "t_lo": reset_box.get("t_hi", ""),
        "t_hi": (_finite_float(reset_box.get("t_hi")) or 0.0) + h_try,
        "h_try": h_try,
        "status": "completed" if flowstar_validated else (best.get("status", "skipped") if best else "skipped"),
        "flowstar_validated": flowstar_validated,
        "pytorch_validated": str(py_row.get("validation_status", "")).lower() == "validated",
        "flowstar_orders": ";".join(str(o) for o in orders),
        "flowstar_best_order": best.get("order", ""),
        "flowstar_last_width_sum": best.get("last_width_sum", ""),
        "pytorch_candidate_final_width_sum": py_row.get("candidate_final_width_sum", ""),
        "width_ratio_flowstar_over_pytorch": (flow_width / py_width) if flow_width is not None and py_width and py_width > 0 else "",
        "flowstar_runtime_s": best.get("runtime_s", ""),
        "failure_reason": "" if flowstar_validated else best.get("failure_reason", "Flow* did not validate or no boxes parsed"),
        "notes": "local one-step diagnostic only; no full parity claim",
    }
    _write_csv(out_dir / "oracle_summary.csv", SUMMARY_FIELDS, [summary])
    _write_csv(out_dir / "oracle_flowstar_segments.csv", FLOWSTAR_SEGMENT_FIELDS, flowstar_segments)
    _write_csv(out_dir / "oracle_pytorch_attempt.csv", PYTORCH_ATTEMPT_FIELDS, [py_row])
    _write_report(out_dir, summary, order_summaries, py_row)
    try:
        from flowstar_style_rescue_vanderpol import write_rescue_next4_outputs

        write_rescue_next4_outputs(trigger_out_dir=out_dir)
    except Exception:
        pass
    return summary, flowstar_segments, py_row


def _write_report(out_dir: Path, summary: Mapping[str, Any], order_summaries: Sequence[Mapping[str, Any]], py_row: Mapping[str, Any]) -> None:
    flowstar_validated = str(summary.get("flowstar_validated", "")).lower() in {"true", "1", "yes"}
    pytorch_validated = str(summary.get("pytorch_validated", "")).lower() in {"true", "1", "yes"}
    if flowstar_validated and not pytorch_validated:
        conclusion = "Flow* validates the same local box and h_try where PyTorch rejects; the PyTorch kernel is missing or tighter than a Flow* mechanism."
    elif not flowstar_validated:
        conclusion = "Flow* does not validate the same local box and h_try; the local reset box is already too wide or the step is too hard."
    else:
        conclusion = "Both tools validate this local one-step case."
    lines = [
        "# Flowstar One-Step Oracle Report",
        "",
        f"Source run: `{SOURCE_RUN_ID}`.",
        f"Reset segment index: `{summary.get('reset_segment_index', '')}` at t=`{summary.get('t_lo', '')}`.",
        f"h_try: `{summary.get('h_try', '')}`.",
        f"Does Flow* validate the same local box and h_try where PyTorch rejects? {'yes' if flowstar_validated and not pytorch_validated else 'no'}.",
        conclusion,
        f"Flow* local one-step width sum: `{summary.get('flowstar_last_width_sum', '')}`.",
        f"PyTorch failed candidate final width sum: `{summary.get('pytorch_candidate_final_width_sum', '')}`.",
        f"Flow*/PyTorch width ratio: `{summary.get('width_ratio_flowstar_over_pytorch', '')}`.",
        "",
        "## Order Comparison",
        "",
        "| order | status | validated | runtime_s | last_width_sum | segments | failure_reason |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in order_summaries:
        lines.append(
            f"| {row.get('order', '')} | {row.get('status', '')} | {_fmt(row.get('flowstar_validated', ''))} | "
            f"{_fmt(row.get('runtime_s', ''))} | {_fmt(row.get('last_width_sum', ''))} | {row.get('num_segments', '')} | {row.get('failure_reason', '')} |"
        )
    lines.extend(
        [
            "",
            "## PyTorch Attempt",
            "",
            f"PyTorch validation status: `{py_row.get('validation_status', '')}`; rejection reason: `{py_row.get('rejection_reason', '')}`.",
            "This is a local diagnostic only, not full Flow* parity.",
        ]
    )
    (out_dir / "oracle_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/flowstar_one_step_oracle"))
    parser.add_argument("--flowstar-root", default=None)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--orders", nargs="*", type=int, default=[4, 6, 8])
    args = parser.parse_args(argv)
    summary, segments, _py = run_oracle(args.out_dir, flowstar_root=args.flowstar_root, timeout_s=args.timeout_s, orders=args.orders)
    print(f"wrote oracle outputs to {args.out_dir}; flowstar_validated={summary.get('flowstar_validated')} segments={len(segments)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
