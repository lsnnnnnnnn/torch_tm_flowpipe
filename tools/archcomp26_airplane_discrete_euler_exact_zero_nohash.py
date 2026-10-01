#!/usr/bin/env python3
"""Bounded Airplane Euler interval prefix with exact-zero angular identity."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import archcomp26_airplane_discrete_euler_interval_smoke_nohash as base


MAX_TRANSFERS = 20
PROPERTY_IDS = (1, 6, 7, 8)


def run(output):
    import torch
    sys.path.insert(0, str(base.ENGINE))
    from flowstar_gpu import interval as iv, transcendental as tr

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    started = perf_counter()
    layers = base.load_layers(base.ASSETS / "controller_airplane.onnx", torch)
    state = torch.tensor(base.INITIAL, dtype=torch.float64)
    iv.assert_valid(state, "Airplane initial box")
    dt = torch.tensor([math.nextafter(0.1, -math.inf),
                       math.nextafter(0.1, math.inf)], dtype=torch.float64)
    if any(float(state[i, 0]) < -1 or float(state[i, 1]) > 1 for i in PROPERTY_IDS):
        raise AssertionError("Airplane k=0 initial box is outside the closed property band")

    events = []
    nn_calls = 0
    accepted = 0
    stopped_at = None
    stop_reason = None
    with (output / "EVENTS.jsonl").open("w") as stream:
        for k in range(1, MAX_TRANSFERS + 1):
            event = {"k": k, "old_state": state.tolist(), "old_index": k - 1}
            try:
                control = base.network_interval(state, layers, iv, torch)
                nn_calls += 1
                derivative = base.euler_derivative(state, control, iv, tr)
                # Fixed dynamics.m lines 41-58: a3=[p;q;r], a4=mat_2*a3.
                # If p=q=r are exactly [0,0], all three angular derivatives
                # are algebraically zero whenever cos(theta) excludes zero.
                exact_zero = bool(torch.equal(state[9:12], torch.zeros_like(state[9:12])))
                if exact_zero:
                    derivative[6:9] = 0.0
                iv.assert_valid(derivative, f"Airplane derivative k={k}")
                endpoint = iv.add(state, iv.mul(dt, derivative))
                if exact_zero:
                    # Exact Euler identity x+0=x, avoiding spurious ULP growth.
                    endpoint[6:9] = state[6:9]
                iv.assert_valid(endpoint, f"Airplane endpoint k={k}")
                inside = {base.STATE_NAMES[i]: bool(endpoint[i, 0] >= -1 and endpoint[i, 1] <= 1)
                          for i in PROPERTY_IDS}
                event.update(control_interval=control.tolist(), derivative_interval=derivative.tolist(),
                             endpoint=endpoint.tolist(), exact_zero_angular_identity=exact_zero,
                             property_coordinate_inside=inside,
                             status="accepted_safe_endpoint" if all(inside.values()) else "property_unknown")
                if all(inside.values()):
                    accepted = k
                    state = endpoint
                else:
                    stopped_at = k
                    stop_reason = "endpoint_box_not_contained_in_closed_property_band"
            except Exception as exc:
                stopped_at = k
                stop_reason = "entry_failed_before_endpoint"
                event.update(status="entry_failed_before_endpoint", error=repr(exc))
            events.append(event)
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
            if stopped_at is not None:
                break

    return {
        "schema": "archcomp26-airplane-paper-euler-exact-zero-prefix-nohash-v1",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "profile": "airplane-discrete/paper-Euler-controller-first/exact-zero-identity-diagnostic",
        "method": "CPU directed interval arithmetic; outside four benchmark methods",
        "qualification": "diagnostic prefix only; not participant discrete implementation or independent floating-point certificate",
        "source": {"model": str(base.ASSETS / "controller_airplane.onnx"),
                   "dynamics": str(base.ASSETS / "dynamics.m"),
                   "specification": str(base.ASSETS / "specifications.txt"),
                   "interval_engine": str(base.ENGINE / "flowstar_gpu/interval.py"),
                   "transcendental_engine": str(base.ENGINE / "flowstar_gpu/transcendental.py")},
        "state_order": base.STATE_NAMES,
        "control_order": ("Fx", "Fy", "Fz", "Mx", "My", "Mz"),
        "initial": base.INITIAL, "full_initial_box_unsplit": True,
        "step_exact_rational": "1/10", "step_float64_enclosure": dt.tolist(),
        "maximum_transfers": MAX_TRANSFERS, "property_check_indices": list(range(MAX_TRANSFERS + 1)),
        "accepted_safe_endpoint_prefix": accepted,
        "attempted_transfer_count": len(events), "controller_call_count": nn_calls,
        "first_stopped_index": stopped_at, "stop_reason": stop_reason,
        "status": "full_20_endpoints_inside" if accepted == MAX_TRANSFERS else "stopped_at_first_unknown_or_failure",
        "last_accepted_endpoint": state.tolist(),
        "events_file": "EVENTS.jsonl", "wall_s": perf_counter() - started,
        "torch_version": torch.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = run(args.output)
    except Exception as exc:
        result = {"schema": "archcomp26-airplane-paper-euler-exact-zero-prefix-nohash-v1",
                  "recorded_utc": datetime.now(timezone.utc).isoformat(),
                  "status": "failed_before_first_transfer", "error": repr(exc)}
        (args.output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
        raise
    (args.output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["status"], result["accepted_safe_endpoint_prefix"],
          result["first_stopped_index"], result["wall_s"])


if __name__ == "__main__":
    main()
