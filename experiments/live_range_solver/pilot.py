"""Exactly two preregistered wait limits, evaluated only on B8/two-step prefixes."""
import argparse
import json
from pathlib import Path

import torch

from experiments.range_batch_device.common import save
from .runner import run_case, write_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for plant in ("van_der_pol", "brusselator"):
        for route in ("Q", "G"):
            name = f"warmup-{plant}-{route}"
            result, events, _ = run_case(plant, [0], 1, route, run_id=name)
            write_run(args.output/name, result, events)
            assert result["successful_tasks"] == 1
    summaries = []
    for i, plant in enumerate(("van_der_pol", "brusselator")):
        for candidate in ((.002, .020) if i == 0 else (.020, .002)):
            for route in (("Q", "G") if candidate == .002 else ("G", "Q")):
                name = f"pilot-{plant}-{route}-{candidate}"
                result, events, _ = run_case(plant, list(range(0, 32, 4)), 2, route,
                    run_id=name, max_wait_s=candidate)
                summary = write_run(args.output/name, result, events)
                assert result["successful_tasks"] == 8 and result["accepted_lane_steps"] == 16
                summaries.append(summary)
                print(json.dumps(dict(name=name, wall_s=result["wall_s"], counts=result["counts"])), flush=True)
    scores = {str(candidate): sum(r["wall_s"] for r in summaries if r["max_wait_s"] == candidate)
              for candidate in (.002, .020)}
    chosen = min(scores, key=lambda k: (scores[k], float(k)))
    decision = dict(candidates=[.002, .020], scores=scores, chosen_wait_s=float(chosen), max_group=32,
        rule="minimum summed Q/G wall time, two systems B8 first two live steps; tie prefers shorter wait",
        formal_results_used=False, sample_names=[r["run_id"] for r in summaries],
        source_sha=summaries[0]["source_sha"])
    save(args.output/"POLICY_SELECTION.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
