#!/usr/bin/env python3
"""Compute exact endpoint hulls and deterministic simulation-only trajectories."""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np
from scipy.integrate import solve_ivp

from common import (
    evaluate_rhs,
    exact_endpoint,
    iter_configurations,
    load_spec,
    output_dir_from_args,
    resolved_spec,
    write_json,
)

REFERENCE_FIELDS = [
    "system", "h", "horizon", "step_index", "time", "state_index",
    "interval_kind", "lower", "upper", "proof",
]
TRAJECTORY_FIELDS = [
    "system", "h", "horizon", "trajectory_id", "time", "state_index", "value",
]


def _initial_points(system: Mapping[str, Any], count: int, seed: int) -> np.ndarray:
    box = np.asarray(system["initial_box"], dtype=float)
    corners = np.asarray(list(itertools.product(*[(lo, hi) for lo, hi in box])), dtype=float)
    rng = np.random.default_rng(seed)
    random_count = max(0, count - len(corners))
    random = rng.uniform(box[:, 0], box[:, 1], size=(random_count, box.shape[0]))
    return np.concatenate([corners, random], axis=0) if random_count else corners[:count]


def _trajectory(
    system_name: str,
    system: Mapping[str, Any],
    initial: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    if system_name == "riccati":
        x0 = initial[0]
        return (x0 / (1.0 - x0 * times))[:, None]
    if system_name == "harmonic":
        c, s = np.cos(times), np.sin(times)
        x0, y0 = initial
        return np.column_stack((c * x0 + s * y0, -s * x0 + c * y0))

    def rhs(_: float, state: np.ndarray) -> np.ndarray:
        return np.asarray(evaluate_rhs(list(state), system), dtype=float)

    solution = solve_ivp(
        rhs,
        (0.0, float(times[-1])),
        initial,
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-14,
        t_eval=times,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y.T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--systems", nargs="*")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(args.output_dir)
    reference_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for config_index, config in enumerate(
        iter_configurations(spec, smoke=args.smoke, systems=args.systems)
    ):
        system_name = config["system"]
        system = spec["systems"][system_name]
        h, horizon, steps = float(config["h"]), float(config["horizon"]), int(config["steps"])
        for step_index in range(steps + 1):
            time_value = step_index * h
            exact = exact_endpoint(system_name, time_value, system["initial_box"])
            if exact is None:
                continue
            for state_index, (lower, upper) in enumerate(exact):
                reference_rows.append(
                    {
                        "system": system_name,
                        "h": h,
                        "horizon": horizon,
                        "step_index": step_index,
                        "time": time_value,
                        "state_index": state_index,
                        "interval_kind": "endpoint",
                        "lower": lower,
                        "upper": upper,
                        "proof": "closed_form_exact_interval_hull",
                    }
                )
        substeps = int(spec["sample_substeps_per_step"])
        fine_times = np.linspace(0.0, horizon, steps * substeps + 1)
        points = _initial_points(
            system,
            int(spec["sample_trajectories"]),
            int(spec["random_seed"]) + config_index,
        )
        for trajectory_id, initial in enumerate(points):
            values = _trajectory(system_name, system, initial, fine_times)
            for time_value, state in zip(fine_times, values):
                for state_index, value in enumerate(state):
                    trajectory_rows.append(
                        {
                            "system": system_name,
                            "h": h,
                            "horizon": horizon,
                            "trajectory_id": trajectory_id,
                            "time": float(time_value),
                            "state_index": state_index,
                            "value": float(value),
                        }
                    )
        print(
            f"reference {system_name} h={h} T={horizon} trajectories={len(points)}",
            flush=True,
        )
    for path, rows, fields in (
        (output_dir / "references.csv", reference_rows, REFERENCE_FIELDS),
        (output_dir / "trajectories.csv", trajectory_rows, TRAJECTORY_FIELDS),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    write_json(output_dir / "benchmark_spec_resolved.json", resolved_spec(spec))
    print(output_dir)


if __name__ == "__main__":
    main()
