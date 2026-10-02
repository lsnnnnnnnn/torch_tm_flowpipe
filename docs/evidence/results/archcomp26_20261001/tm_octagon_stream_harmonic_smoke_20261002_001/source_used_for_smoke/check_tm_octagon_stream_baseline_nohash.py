#!/usr/bin/env python3
"""Compare saved three-step stream supports with an unobserved CPU run."""
import argparse
import json
from pathlib import Path

from torch_tm_flowpipe import Interval, flowpipe_multi_step
from torch_tm_flowpipe.ode_examples import harmonic_oscillator_ode
from torch_tm_flowpipe.tm_octagon_nohash import directional_intervals, geometry_from_stream


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    run = parser.parse_args().run
    target = run / "BASELINE_COMPARE.json"
    if target.exists():
        raise FileExistsError(target)
    saved = geometry_from_stream(run / "harmonic_stream.jsonl")
    plain = flowpipe_multi_step(
        harmonic_oscillator_ode, [Interval(1.0, 1.2), Interval(0.0, 0.0)],
        h=.05, steps=3, order=4, mode="dependency_preserving",
    )
    if plain.status != "validated" or len(plain.segments) != saved["accepted_steps"]:
        raise AssertionError("unobserved baseline acceptance differs from streamed run")
    for frame, segment in zip(saved["frames"], plain.segments):
        if (frame["tube_supports"] != directional_intervals(segment.tm, 0, 1)
                or frame["endpoint_supports"] != directional_intervals(segment.final_tm, 0, 1)):
            raise AssertionError("unobserved baseline directional support differs")
    receipt = {"status": "pass", "plain_status": plain.status,
               "plain_accepted_steps": len(plain.segments),
               "stream_accepted_steps": saved["accepted_steps"],
               "all_tube_and_endpoint_supports_equal": True,
               "scope": "same CPU harmonic plant, 3 steps; observer does not change acceptance or directional results here",
               "content_digest_policy": "none computed"}
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
