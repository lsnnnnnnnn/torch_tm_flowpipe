from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from torch_tm_flowpipe import Interval, flowpipe_multi_step
from torch_tm_flowpipe.ode_examples import scalar_quadratic_ode

from _common import dtype_device, interval_contains_all, max_final_width, max_flowpipe_width, now, write_csv


def exact_scalar_quadratic(x0: float, t: float) -> float:
    return math.tan(t + math.atan(x0))


def run_case(h: float, order: int, steps: int, mode: str) -> dict:
    start = now()
    result = flowpipe_multi_step(
        scalar_quadratic_ode,
        [Interval(0.0, 0.1)],
        h=h,
        steps=steps,
        order=order,
        mode=mode,
    )
    runtime = now() - start
    T = h * steps
    samples = [exact_scalar_quadratic(i / 1000.0, T) for i in range(0, 101, 5)]
    failures = interval_contains_all(result.final_tm.range_box()[0], samples)
    device, dtype = dtype_device(result)
    return {
        "system": f"scalar_quadratic/{mode}",
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
    cases = [(0.01, 4, 5), (0.02, 4, 5)]
    if args.full:
        cases.append((0.01, 5, 10))
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
