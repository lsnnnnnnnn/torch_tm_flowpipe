#!/usr/bin/env python3
"""Focused diagnostics for the Van der Pol benchmark.

The main comparison showed dependency-preserving is wider than range_only on
Van der Pol.  This diagnostic decomposes final width into polynomial interval
range and interval remainder contributions, and compares against a dense RK4
sampling estimate of the true final box.  The sampling estimate is not a proof.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

torch.set_num_threads(1)

from torch_tm_flowpipe import Interval, flowpipe_multi_step
from torch_tm_flowpipe.ode_examples import van_der_pol_ode

FIELDS = [
    "h", "steps", "order", "mode", "status", "dim", "final_width",
    "poly_range_width", "remainder_width", "remainder_radius",
    "poly_degree", "term_count", "true_sample_width_dim",
    "true_sample_width_sum", "over_sample_ratio_dim", "validation_attempts",
]


def width(iv: Interval) -> float:
    return float(iv.width().detach().cpu())


def radius(iv: Interval) -> float:
    return float(iv.radius().detach().cpu())


def rk4_step(state: tuple[float, float], dt: float) -> tuple[float, float]:
    def f(s: tuple[float, float]) -> tuple[float, float]:
        x, y = s
        return (y, y - x - x * x * y)

    x, y = state
    k1 = f((x, y))
    k2 = f((x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1]))
    k3 = f((x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1]))
    k4 = f((x + dt * k3[0], y + dt * k3[1]))
    return (
        x + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6,
        y + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6,
    )


def linspace(lo: float, hi: float, n: int) -> list[float]:
    if n == 1:
        return [(lo + hi) / 2]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def sample_true_widths(T: float, grid: int, substeps: int) -> tuple[list[float], list[tuple[float, float]]]:
    vals = []
    dt = T / substeps
    for x0 in linspace(1.1, 1.4, grid):
        for y0 in linspace(2.35, 2.45, grid):
            s = (x0, y0)
            for _ in range(substeps):
                s = rk4_step(s, dt)
            vals.append(s)
    xs = [v[0] for v in vals]
    ys = [v[1] for v in vals]
    boxes = [(min(xs), max(xs)), (min(ys), max(ys))]
    return [boxes[0][1] - boxes[0][0], boxes[1][1] - boxes[1][0]], boxes


def diagnostic_rows(h: float, steps: int, order: int, grid: int, substeps: int) -> list[dict[str, object]]:
    T = h * steps
    true_widths, true_boxes = sample_true_widths(T, grid=grid, substeps=substeps)
    true_sum = sum(true_widths)
    rows = []
    x0_box = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    for mode in ["range_only", "dependency_preserving"]:
        result = flowpipe_multi_step(van_der_pol_ode, x0_box, h=h, steps=steps, order=order, mode=mode)
        final_box = result.final_tm.range_box()
        for dim, model in enumerate(result.final_tm.models):
            poly_iv = model.polynomial.evaluate_interval(model.domain)
            fw = width(final_box[dim])
            rows.append({
                "h": h,
                "steps": steps,
                "order": order,
                "mode": mode,
                "status": result.status,
                "dim": dim,
                "final_width": fw,
                "poly_range_width": width(poly_iv),
                "remainder_width": width(model.remainder),
                "remainder_radius": radius(model.remainder),
                "poly_degree": model.polynomial.degree(),
                "term_count": len(model.polynomial.terms),
                "true_sample_width_dim": true_widths[dim],
                "true_sample_width_sum": true_sum,
                "over_sample_ratio_dim": fw / true_widths[dim] if true_widths[dim] > 0 else "",
                "validation_attempts": result.validation_attempts,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose why Van der Pol dependency-preserving can be wider.")
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--orders", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--sample-grid", type=int, default=51)
    parser.add_argument("--rk4-substeps", type=int, default=100)
    parser.add_argument("--csv", default="outputs/van_der_pol_diagnostics.csv")
    args = parser.parse_args()

    rows = []
    for order in args.orders:
        rows.extend(diagnostic_rows(args.h, args.steps, order, args.sample_grid, args.rk4_substeps))
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    for r in rows:
        if r["dim"] == 1:
            print(f"order={r['order']} mode={r['mode']}: y_width={float(r['final_width']):.6g} "
                  f"poly_width={float(r['poly_range_width']):.6g} rem_width={float(r['remainder_width']):.6g} "
                  f"sample_width={float(r['true_sample_width_dim']):.6g}")


if __name__ == "__main__":
    main()
