#!/usr/bin/env python3
"""Three-step CPU plant smoke for the accepted-segment octagon stream."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from torch_tm_flowpipe import Interval, flowpipe_multi_step
from torch_tm_flowpipe.ode_examples import harmonic_oscillator_ode
from torch_tm_flowpipe.tm_octagon_nohash import (
    OctagonJSONLObserver, directional_intervals, geometry_from_stream, render_geometry,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    stream_path = output / "harmonic_stream.jsonl"
    initial = [[1.0, 1.2], [0.0, 0.0]]
    started = time.perf_counter()
    with OctagonJSONLObserver(stream_path, initial,
                             source_label="Harmonic oscillator CPU plant-only stream smoke") as observer:
        result = flowpipe_multi_step(
            harmonic_oscillator_ode, [Interval(*pair) for pair in initial],
            h=.05, steps=3, order=4, mode="dependency_preserving",
            accepted_segment_observer=observer,
        )
    solver_observer_s = time.perf_counter() - started
    if result.status != "validated" or observer.step != 3 or len(result.segments) != 3:
        raise RuntimeError("short harmonic solver or stream did not complete all three steps")
    geometry = geometry_from_stream(stream_path)
    if geometry["accepted_steps"] != 3 or not math.isclose(geometry["numerical_horizon"], .15):
        raise AssertionError("stream replay has an unexpected accepted horizon")
    sampled_cases = 0
    for frame, segment in zip(geometry["frames"], result.segments):
        for view, tm in (("tube", segment.tm), ("endpoint", segment.final_tm)):
            bounds = frame[f"{view}_supports"]
            if bounds != directional_intervals(tm, 0, 1):
                raise AssertionError("streamed directional supports differ from accepted in-memory TM")
            times = ([frame["t_start"] + (frame["t_end"] - frame["t_start"]) * j / 10
                      for j in range(11)] if view == "tube" else [frame["t_end"]])
            for t in times:
                for x0 in (1.0, 1.1, 1.2):
                    x, y = x0 * math.cos(t), -x0 * math.sin(t)
                    for name, value in (("x", x), ("y", y), ("x_plus_y", x + y),
                                        ("x_minus_y", x - y)):
                        lo, hi = bounds[name]
                        if not lo - 1e-12 <= value <= hi + 1e-12:
                            raise AssertionError(f"exact sample outside {view} {name}")
                    sampled_cases += 1
    geometry_path = output / "harmonic_stream.geometry.json"
    geometry_path.write_text(json.dumps(geometry, indent=2, allow_nan=False) + "\n")
    for view in ("tube", "endpoint"):
        render_geometry(geometry, output / f"harmonic_stream_{view}", view=view,
                        geometry_path=geometry_path)
    receipt = {"schema": "torch-tm-flowpipe-octagon-stream-smoke-nohash-v1",
               "status": result.status, "accepted_steps": observer.step,
               "numerical_horizon": observer.time, "h": .05, "order": 4,
               "mode": "dependency_preserving", "initial_box": initial,
               "ode": ["x1'=x2", "x2'=-x1"],
               "sampled_exact_solution_cases": sampled_cases,
               "stream_supports_equal_accepted_tm": True,
               "solver_plus_observer_wall_seconds": solver_observer_s,
               "qualification": "CPU plant-only three-step stream smoke; no NNCS or full benchmark proof",
               "content_digest_policy": "none computed"}
    (output / "RUN.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": result.status, "steps": observer.step,
                      "sampled_exact_solution_cases": sampled_cases}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
