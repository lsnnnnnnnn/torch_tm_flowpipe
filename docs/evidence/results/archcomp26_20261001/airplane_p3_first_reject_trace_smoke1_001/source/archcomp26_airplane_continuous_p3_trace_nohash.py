#!/usr/bin/env python3
"""Trace one existing Airplane P3 full-box attempt through its first rejection."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import archcomp26_airplane_continuous_p3_nohash as airplane


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_known_args()[0].output
    original_prepare = airplane.prepare_p3

    def prepare(backend):
        torch, driver, engine, cache = original_prepare(backend)
        trace_path = output / "refinement_trace.jsonl"
        trace_path.open("x").close()
        (output / "TRACE_METHOD.json").write_text(json.dumps({
            "schema": "archcomp26-airplane-p3-first-rejection-trace-v1",
            "physical_initial_boxes": 1,
            "maximum_control_periods": 1,
            "maximum_ode_substeps": 10,
            "stop_policy": "first numerical rejection or saved tube outside safe set",
            "validation_policy": "solution_order",
            "callback_effect": "refinement_callback selects the engine eager refinement path; diagnostic time is not a production timing sample",
        }, indent=2) + "\n")
        original_advance = driver.advance_sparse
        substep = 0

        def traced_advance(*args, **kwargs):
            nonlocal substep
            substep += 1
            args = list(args)
            if args[4].refinement_callback is not None:
                raise RuntimeError("unexpected preexisting refinement callback")
            with trace_path.open("a") as trace:
                def record(event):
                    trace.write(json.dumps({"substep": substep, **event}, default=repr) + "\n")
                    trace.flush()

                args[4] = replace(args[4], refinement_callback=record)
                state, accepted = original_advance(*args, **kwargs)
                record({"event": "advance_return", "lane": 0,
                        "accepted": bool(accepted[0].item()),
                        "status_code": int(state.status[0].item())})
                return state, accepted

        driver.advance_sparse = traced_advance
        return torch, driver, engine, cache

    airplane.prepare_p3 = prepare
    return airplane.main()


if __name__ == "__main__":
    raise SystemExit(main())
