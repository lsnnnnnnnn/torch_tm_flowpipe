"""Preserve a bounded real execution and locate the first sequence mismatch."""
from collections import defaultdict
import gzip
import json
from pathlib import Path

import torch

from experiments.live_range_solver.analyze import compare_state
from experiments.live_range_solver.runner import ROOT, run_case
from experiments.live_range_solver.verify import load_run


def sequence(events):
    result = defaultdict(list)
    for event in events:
        if event["event"] == "submit":
            request = dict(event["request"])
            request.pop("request_id")
            result[event["task"]].append((event["generation"], event["attempt"], event["counter"], event["source_state"], request))
    return dict(result)


def first_difference(left, right, path="root"):
    if left == right:
        return None
    if type(left) is not type(right):
        return dict(path=path, left_type=type(left).__name__, right_type=type(right).__name__,
                    left=repr(left)[:500], right=repr(right)[:500])
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return dict(path=path, left_keys=list(left), right_keys=list(right))
        for key in left:
            found = first_difference(left[key], right[key], path+"/"+str(key))
            if found:
                return found
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return dict(path=path, left_length=len(left), right_length=len(right))
        for index, (a, b) in enumerate(zip(left, right)):
            found = first_difference(a, b, path+"/"+str(index))
            if found:
                return found
    return dict(path=path, left=repr(left)[:500], right=repr(right)[:500])


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
saved = ROOT.parent/"package_failure_01"
run_root = ROOT/"artifacts/runs/live_range_solver_20260909T053007Z"
for plant in ("van_der_pol", "brusselator"):
    for route in ("S", "Q", "S_gpu", "G"):
        reference_path = run_root/"diagnostic"/f"small-b2-{plant}-{route}"
        reference, reference_events = load_run(reference_path)
        actual, events, _ = run_case(plant, [0, 31], 2, route, diagnostic=True,
                                   run_id=f"fresh-{plant}-{route}")
        equality = compare_state(reference, actual)
        old, fresh = sequence(reference_events), sequence(events)
        record = dict(plant=plant, route=route, source_sha=actual["source_sha"],
                      complete_state_comparison=equality,
                      accepted_lane_steps=actual["accepted_lane_steps"],
                      reference_path=str(reference_path.relative_to(ROOT)),
                      strict_python_sequence_equal=old == fresh,
                      json_value_sequence_equal=json.loads(json.dumps(old)) == json.loads(json.dumps(fresh)),
                      first_difference=first_difference(old, fresh))
        payload = saved/f"fresh-{plant}-{route}.json.gz"
        with gzip.open(payload, "wt") as stream:
            json.dump(dict(run=actual, events=events), stream, sort_keys=True)
        record["fresh_payload"] = payload.name
        (saved/f"diagnosis-{plant}-{route}.json").write_text(json.dumps(record, indent=2)+"\n")
        print(json.dumps(record), flush=True)
        if old != fresh:
            raise SystemExit(0)
raise SystemExit("No mismatch reproduced in the eight real cases")
