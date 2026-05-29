from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from torch_tm_flowpipe import Interval, flowpipe_multi_step
from torch_tm_flowpipe.ode_examples import harmonic_oscillator_ode

from _common import dtype_device, interval_contains_all, max_final_width, max_flowpipe_width, now, write_csv


def exact(x0: float, v0: float, t: float) -> tuple[float, float]:
    return (x0 * math.cos(t) + v0 * math.sin(t), -x0 * math.sin(t) + v0 * math.cos(t))


def run_case(h: float, order: int, steps: int, mode: str) -> dict:
    start = now()
    result = flowpipe_multi_step(
        harmonic_oscillator_ode,
        [Interval(0.9, 1.0), Interval(-0.05, 0.05)],
        h=h,
        steps=steps,
        order=order,
        mode=mode,
    )
    runtime = now() - start
    T = h * steps
    xs, vs = [], []
    for i in range(6):
        x0 = 0.9 + 0.1 * i / 5
        for j in range(6):
            v0 = -0.05 + 0.1 * j / 5
            x, v = exact(x0, v0, T)
            xs.append(x)
            vs.append(v)
    failures = interval_contains_all(result.final_tm.range_box()[0], xs) + interval_contains_all(result.final_tm.range_box()[1], vs)
    device, dtype = dtype_device(result)
    return {
        "system": f"harmonic_oscillator/{mode}",
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
    cases = [(0.02, 4, 10)]
    if args.full:
        cases.append((0.01, 5, 20))
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
