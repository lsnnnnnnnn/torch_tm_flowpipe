#!/usr/bin/env python3
"""Diagnostic sample containment check for normalized-insertion Van der Pol boxes."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import flowstar_style_rescue_vanderpol as rescue

SUMMARY_FIELDS = [
    "run_id",
    "integrator",
    "diagnostic_only",
    "num_samples",
    "checked_sample_time_pairs",
    "segment_times_checked",
    "violations_count",
    "max_outside_distance",
    "max_outside_sample_id",
    "max_outside_t",
    "max_outside_x",
    "max_outside_y",
    "status",
    "notes",
]

INITIAL_X = (1.1, 1.4)
INITIAL_Y = (2.35, 2.45)


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: rescue._fmt(row.get(field, "")) for field in fields})


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _finite_float(value: Any) -> float | None:
    return rescue._finite_float(value)


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1:
        return [(lo + hi) / 2.0]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def sample_initial_points(num_samples: int) -> list[tuple[str, float, float]]:
    n = max(1, int(num_samples))
    nx = max(2, int(math.ceil(math.sqrt(n))))
    ny = max(2, int(math.ceil(n / nx)))
    points: list[tuple[str, float, float]] = []
    for ix, x in enumerate(_linspace(INITIAL_X[0], INITIAL_X[1], nx)):
        for iy, y in enumerate(_linspace(INITIAL_Y[0], INITIAL_Y[1], ny)):
            points.append((f"grid_{ix}_{iy}", x, y))
    return points[:n]


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


def _rk4_advance(state: tuple[float, float], dt_total: float, max_dt: float) -> tuple[float, float]:
    steps = max(1, int(math.ceil(abs(dt_total) / max(float(max_dt), 1e-12))))
    dt = float(dt_total) / steps
    current = state
    for _ in range(steps):
        current = _rk4_step(current, dt)
    return current


def _select_run(summary_rows: Sequence[Mapping[str, str]], run_id: str | None) -> str:
    if run_id:
        return run_id
    candidates = [row for row in summary_rows if row.get("reset_mode") == "normalized_insertion_symqueue_v2"]
    if not candidates:
        candidates = [row for row in summary_rows if row.get("reset_mode") == "normalized_insertion_symqueue_split"]
    if not candidates:
        candidates = [row for row in summary_rows if row.get("reset_mode") == "normalized_insertion_symqueue"]
    if not candidates:
        candidates = [row for row in summary_rows if row.get("reset_mode") == "normalized_insertion"]
    if not candidates:
        candidates = list(summary_rows)
    best = max(candidates, key=lambda row: _finite_float(row.get("last_validated_t")) or 0.0)
    return str(best.get("run_id", ""))


def _rows_for_run(segment_rows: Sequence[Mapping[str, str]], run_id: str) -> list[Mapping[str, str]]:
    rows = [row for row in segment_rows if row.get("run_id") == run_id and row.get("status") == "validated"]
    return sorted(rows, key=lambda row: _finite_float(row.get("t_hi")) or 0.0)


def _outside_distance(x: float, y: float, row: Mapping[str, str], tol: float) -> float:
    x_lo = _finite_float(row.get("x_lo"))
    x_hi = _finite_float(row.get("x_hi"))
    y_lo = _finite_float(row.get("y_lo"))
    y_hi = _finite_float(row.get("y_hi"))
    if x_lo is None or x_hi is None or y_lo is None or y_hi is None:
        return math.inf
    return max(x_lo - x - tol, x - x_hi - tol, y_lo - y - tol, y - y_hi - tol, 0.0)


def check_sample_containment(
    out_dir: Path,
    *,
    run_id: str | None,
    num_samples: int,
    max_rk4_dt: float,
    tol: float,
) -> dict[str, Any]:
    summary_rows = _read_rows(out_dir / "rescue_summary.csv")
    segment_rows = _read_rows(out_dir / "rescue_segments.csv")
    selected_run_id = _select_run(summary_rows, run_id)
    rows = _rows_for_run(segment_rows, selected_run_id)
    if not rows:
        raise SystemExit(f"no validated segment rows found for {selected_run_id}")

    samples = sample_initial_points(num_samples)
    violations = 0
    max_outside = 0.0
    max_sample_id = ""
    max_t = ""
    max_x = ""
    max_y = ""
    checked = 0
    for sample_id, x0, y0 in samples:
        state = (float(x0), float(y0))
        current_t = 0.0
        for row in rows:
            t = _finite_float(row.get("t_hi"))
            if t is None:
                continue
            dt = t - current_t
            if dt < -1e-12:
                continue
            state = _rk4_advance(state, max(dt, 0.0), max_rk4_dt)
            current_t = t
            checked += 1
            outside = _outside_distance(state[0], state[1], row, tol)
            if outside > 0.0:
                violations += 1
                if outside > max_outside:
                    max_outside = outside
                    max_sample_id = sample_id
                    max_t = t
                    max_x = state[0]
                    max_y = state[1]

    status = "passed" if violations == 0 else "failed"
    return {
        "run_id": selected_run_id,
        "integrator": "rk4",
        "diagnostic_only": True,
        "num_samples": len(samples),
        "checked_sample_time_pairs": checked,
        "segment_times_checked": len(rows),
        "violations_count": violations,
        "max_outside_distance": max_outside,
        "max_outside_sample_id": max_sample_id,
        "max_outside_t": max_t,
        "max_outside_x": max_x,
        "max_outside_y": max_y,
        "status": status,
        "notes": "diagnostic-only numerical sample sanity check; not a proof",
    }


def write_report(out_dir: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Sample Containment Report",
        "",
        "This diagnostic checks deterministic RK4 sample trajectories against PyTorch normalized-insertion boxes at accepted segment end times.",
        "It is a sanity check only, not a reachability proof.",
        "",
        "## Result",
        "",
        "| run_id | samples | checked_pairs | violations | max_outside_distance | status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        f"| {summary.get('run_id', '')} | {summary.get('num_samples', '')} | {summary.get('checked_sample_time_pairs', '')} | "
        f"{summary.get('violations_count', '')} | {summary.get('max_outside_distance', '')} | {summary.get('status', '')} |",
        "",
        f"Max outside sample id: `{summary.get('max_outside_sample_id', '')}`.",
        f"Max outside time: `{summary.get('max_outside_t', '')}`.",
        f"Integrator: `{summary.get('integrator', '')}` with deterministic grid initial points in x=[1.1,1.4], y=[2.35,2.45].",
        "Conclusion: this diagnostic can catch obvious enclosure misses, but passing it does not prove containment for the full initial box.",
    ]
    (out_dir / "sample_containment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_h10_report(out_dir: Path, max_horizon: float) -> None:
    summary_path = out_dir / "normalized_insertion_h10_summary.csv"
    if summary_path.exists():
        comparison_path = out_dir / "normalized_insertion_h10_vs_flowstar_comparison.csv"
        summary_rows = _read_rows(summary_path)
        comparison_rows = _read_rows(comparison_path) if comparison_path.exists() else []
        rescue.write_normalized_insertion_h10_report(out_dir, summary_rows, comparison_rows, max_horizon=max_horizon)

    normal_eval_summary_path = out_dir / "normal_eval_summary.csv"
    if normal_eval_summary_path.exists():
        comparison_path = out_dir / "normal_eval_vs_flowstar_comparison.csv"
        summary_rows = _read_rows(normal_eval_summary_path)
        comparison_rows = _read_rows(comparison_path) if comparison_path.exists() else []
        rescue.write_normal_eval_h10_report(out_dir, summary_rows, comparison_rows, max_horizon=max_horizon)

    symqueue_summary_path = out_dir / "symqueue_h10_summary.csv"
    if symqueue_summary_path.exists():
        comparison_path = out_dir / "symqueue_h10_vs_flowstar_comparison.csv"
        summary_rows = _read_rows(symqueue_summary_path)
        comparison_rows = _read_rows(comparison_path) if comparison_path.exists() else []
        rescue.write_symqueue_h10_report(out_dir, summary_rows, comparison_rows, max_horizon=max_horizon)

    split_summary_path = out_dir / "symqueue_split_summary.csv"
    if split_summary_path.exists():
        comparison_path = out_dir / "symqueue_split_vs_flowstar_comparison.csv"
        summary_rows = _read_rows(split_summary_path)
        comparison_rows = _read_rows(comparison_path) if comparison_path.exists() else []
        rescue.write_symqueue_split_h10_report(out_dir, summary_rows, comparison_rows, max_horizon=max_horizon)

    v2_summary_path = out_dir / "symqueue_v2_summary.csv"
    if v2_summary_path.exists():
        comparison_path = out_dir / "symqueue_v2_vs_flowstar_comparison.csv"
        segment_path = out_dir / "symqueue_v2_segments.csv"
        summary_rows = _read_rows(v2_summary_path)
        segment_rows = _read_rows(segment_path) if segment_path.exists() else []
        comparison_rows = _read_rows(comparison_path) if comparison_path.exists() else []
        rescue.write_symqueue_v2_h10_report(out_dir, summary_rows, segment_rows, comparison_rows, max_horizon=max_horizon)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory containing rescue_summary.csv and rescue_segments.csv.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/flowstar_normalized_insertion_h10"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--max-rk4-dt", type=float, default=0.001)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--max-horizon", type=float, default=10.0)
    args = parser.parse_args(argv)

    input_dir = args.input_dir if args.input_dir is not None else args.out_dir
    summary = check_sample_containment(
        input_dir,
        run_id=args.run_id,
        num_samples=args.samples,
        max_rk4_dt=args.max_rk4_dt,
        tol=args.tol,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "sample_containment_summary.csv", SUMMARY_FIELDS, [summary])
    write_report(args.out_dir, summary)
    refresh_h10_report(args.out_dir, max_horizon=float(args.max_horizon))
    print(f"sample containment {summary['status']} for {summary['run_id']} with {summary['violations_count']} violations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
