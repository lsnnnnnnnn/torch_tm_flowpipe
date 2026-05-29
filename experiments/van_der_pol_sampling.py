from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from torch_tm_flowpipe import Interval, flowpipe_multi_step
from torch_tm_flowpipe.ode_examples import van_der_pol_ode

from _common import dtype_device, interval_contains_all, max_final_width, max_flowpipe_width, now, write_csv


def f(state, mu=1.0):
    x, y = state
    return (y, mu * (1.0 - x * x) * y - x)


def rk4(state, h, steps):
    x, y = state
    for _ in range(steps):
        k1 = f((x, y))
        k2 = f((x + 0.5 * h * k1[0], y + 0.5 * h * k1[1]))
        k3 = f((x + 0.5 * h * k2[0], y + 0.5 * h * k2[1]))
        k4 = f((x + h * k3[0], y + h * k3[1]))
        x += h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
        y += h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
    return x, y


def run_case(h: float, order: int, steps: int, mode: str) -> dict:
    start = now()
    result = flowpipe_multi_step(
        van_der_pol_ode,
        [Interval(1.0, 1.05), Interval(0.0, 0.05)],
        h=h,
        steps=steps,
        order=order,
        mode=mode,
    )
    runtime = now() - start
    xs, ys = [], []
    for i in range(6):
        x0 = 1.0 + 0.05 * i / 5
        for j in range(6):
            y0 = 0.05 * j / 5
            x, y = rk4((x0, y0), h / 5.0, steps * 5)
            xs.append(x)
            ys.append(y)
    failures = interval_contains_all(result.final_tm.range_box()[0], xs, tol=1e-8) + interval_contains_all(result.final_tm.range_box()[1], ys, tol=1e-8)
    device, dtype = dtype_device(result)
    return {
        "system": f"van_der_pol/{mode}",
        "h": h,
        "order": order,
        "status": result.status,
        "final_width": max_final_width(result),
        "flowpipe_width": max_flowpipe_width(result),
        "runtime_s": runtime,
        "validation_attempts": result.validation_attempts,
        "containment_failures": failures,
        "device": device,
        "dtype": dtype,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--full", action="store_true", help="run the larger regression grid")
    args = parser.parse_args()
    rows = []
    cases = [(0.005, 4, 5)]
    if args.full:
        cases.append((0.01, 4, 5))
    for h, order, steps in cases:
        for mode in ["range_only", "dependency_preserving"]:
            rows.append(run_case(h, order, steps, mode))
    for row in rows:
        print(
            f"{row['system']}: h={row['h']} order={row['order']} "
            f"status={row['status']} final_width={row['final_width']:.6g} "
            f"flowpipe_width={row['flowpipe_width']:.6g} failures={row['containment_failures']} "
            f"time={row['runtime_s']:.4f}s"
        )
    write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
