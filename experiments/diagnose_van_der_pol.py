#!/usr/bin/env python3
"""Focused diagnostics for the Van der Pol benchmark.

This diagnostic decomposes final Taylor-model width into polynomial interval
range and interval remainder contributions, and compares against a dense RK4
sampling estimate of the true final box.  The sampling estimate is not a proof.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
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
    "system", "mode", "h", "steps", "requested_order", "status",
    "final_width_sum", "final_width_by_dim", "poly_range_width_sum",
    "poly_range_width_by_dim", "remainder_width_sum", "remainder_width_by_dim",
    "remainder_radius_by_dim", "max_final_degree", "degree_by_dim",
    "term_count_by_dim", "runtime_s", "validation_attempts",
    "containment_failures", "sampled_width_sum", "sampled_width_by_dim",
    "remainder_width_frac", "poly_range_width_frac", "width_over_sampled_ratio",
    "quality_label",
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




def quality_label(status: str, final_sum: float, poly_sum: float, rem_sum: float, sampled_sum: float) -> str:
    if status != "validated" or final_sum <= 0.0:
        return "failed"
    rem_frac = rem_sum / final_sum
    poly_frac = poly_sum / final_sum
    ratio = final_sum / sampled_sum if sampled_sum > 0.0 else float("inf")
    if ratio > 10.0:
        return "very_loose"
    if rem_frac >= 0.5:
        return "remainder_dominated"
    if poly_frac >= rem_frac:
        return "polynomial_range_dominated"
    return "ok"


def _containment_failures(final_box: list[Interval], sample_boxes: list[tuple[float, float]]) -> int:
    failures = 0
    for dim, (lo, hi) in enumerate(sample_boxes):
        if not final_box[dim].contains(lo, tol=1e-8):
            failures += 1
        if not final_box[dim].contains(hi, tol=1e-8):
            failures += 1
    return failures


def diagnostic_rows(h: float, steps: int, order: int, grid: int, substeps: int) -> list[dict[str, object]]:
    T = h * steps
    true_widths, true_boxes = sample_true_widths(T, grid=grid, substeps=substeps)
    true_sum = sum(true_widths)
    rows = []
    x0_box = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    for mode in ["range_only", "dependency_preserving"]:
        start = time.perf_counter()
        result = flowpipe_multi_step(van_der_pol_ode, x0_box, h=h, steps=steps, order=order, mode=mode)
        runtime = time.perf_counter() - start
        final_box = result.final_tm.range_box()
        final_widths = [width(iv) for iv in final_box]
        poly_widths = []
        rem_widths = []
        rem_radii = []
        degrees = []
        term_counts = []
        for model in result.final_tm.models:
            poly_iv = model.polynomial.evaluate_interval(model.domain)
            poly_widths.append(width(poly_iv))
            rem_widths.append(width(model.remainder))
            rem_radii.append(radius(model.remainder))
            degrees.append(model.polynomial.degree())
            term_counts.append(len(model.polynomial.terms))
        final_sum = sum(final_widths)
        poly_sum = sum(poly_widths)
        rem_sum = sum(rem_widths)
        rows.append({
            "system": "van_der_pol",
            "mode": mode,
            "h": h,
            "steps": steps,
            "requested_order": order,
            "status": result.status,
            "final_width_sum": final_sum,
            "final_width_by_dim": repr(final_widths),
            "poly_range_width_sum": poly_sum,
            "poly_range_width_by_dim": repr(poly_widths),
            "remainder_width_sum": rem_sum,
            "remainder_width_by_dim": repr(rem_widths),
            "remainder_radius_by_dim": repr(rem_radii),
            "max_final_degree": max(degrees) if degrees else 0,
            "degree_by_dim": repr(degrees),
            "term_count_by_dim": repr(term_counts),
            "runtime_s": runtime,
            "validation_attempts": result.validation_attempts,
            "containment_failures": _containment_failures(final_box, true_boxes),
            "sampled_width_sum": true_sum,
            "sampled_width_by_dim": repr(true_widths),
            "remainder_width_frac": rem_sum / final_sum if final_sum > 0 else "",
            "poly_range_width_frac": poly_sum / final_sum if final_sum > 0 else "",
            "width_over_sampled_ratio": final_sum / true_sum if true_sum > 0 else "",
            "quality_label": quality_label(result.status, final_sum, poly_sum, rem_sum, true_sum),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose why Van der Pol dependency-preserving can be wider.")
    parser.add_argument("--h", type=float, default=None, help="single step size; kept for compatibility")
    parser.add_argument("--steps", type=int, default=None, help="single step count; kept for compatibility")
    parser.add_argument("--h-values", type=float, nargs="+", default=None)
    parser.add_argument("--steps-values", type=int, nargs="+", default=None)
    parser.add_argument("--orders", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--sample-grid", type=int, default=51)
    parser.add_argument("--rk4-substeps", type=int, default=100)
    parser.add_argument("--csv", default="outputs/van_der_pol_diagnostics.csv")
    args = parser.parse_args()

    h_values = args.h_values if args.h_values is not None else [args.h if args.h is not None else 0.01]
    steps_values = args.steps_values if args.steps_values is not None else [args.steps if args.steps is not None else 10]

    rows = []
    for h in h_values:
        for steps in steps_values:
            for order in args.orders:
                rows.extend(diagnostic_rows(float(h), int(steps), int(order), args.sample_grid, args.rk4_substeps))
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    for r in rows:
        if r["h"] == 0.01 and r["steps"] == 10:
            print(
                f"order={r['requested_order']} mode={r['mode']}: width={float(r['final_width_sum']):.6g} "
                f"poly={float(r['poly_range_width_sum']):.6g} rem={float(r['remainder_width_sum']):.6g} "
                f"sample={float(r['sampled_width_sum']):.6g}"
            )


if __name__ == "__main__":
    main()
