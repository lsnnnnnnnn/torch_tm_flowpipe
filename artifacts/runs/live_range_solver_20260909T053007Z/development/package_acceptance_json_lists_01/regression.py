"""Regression evidence for JSON/native request sequence comparison only."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

import torch

from experiments.live_range_solver.runner import ROOT
from experiments.live_range_solver.verify import load_run
from experiments.live_range_solver.verify_package import bounded_live_replay, request_sequence


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
saved = Path(__file__).resolve().parent
run_root = ROOT/"artifacts/runs/live_range_solver_20260909T053007Z"
_, reference = load_run(run_root/"diagnostic/small-b2-van_der_pol-S")
with gzip.open(saved/"fresh-van_der_pol-S.json.gz", "rt") as stream:
    native = json.load(stream)["events"]
for event in native:
    if event["event"] == "submit":
        event["request"]["exponents"] = tuple(tuple(row) for row in event["request"]["exponents"])
assert request_sequence(native) == request_sequence(reference)
rejected = []
for kind in ("task", "generation", "attempt", "counter", "source_state", "coefficient", "domain", "support_order"):
    changed = deepcopy(native)
    event = next(row for row in changed if row["event"] == "submit")
    if kind == "task":
        event["task"] = "31"
    elif kind in {"generation", "attempt", "counter"}:
        event[kind] += 1
    elif kind == "source_state":
        event[kind] = "different-source-state"
    elif kind == "coefficient":
        event["request"]["coefficients_lo"]["values"][0] = float(0).hex()
    elif kind == "domain":
        event["request"]["domain_hi"]["values"][0] = float(2).hex()
    elif kind == "support_order":
        event["request"]["exponents"] = event["request"]["exponents"][::-1]
    assert request_sequence(changed) != request_sequence(reference), f"semantic change was discarded: {kind}"
    rejected.append(kind)
print(json.dumps(dict(native_json_equivalence=True, semantic_mutations_rejected=rejected)), flush=True)
replay = bounded_live_replay(run_root)
assert len(replay) == 8 and sum(row["accepted_lane_steps"] for row in replay) == 32
receipt = dict(scope="PACKAGE_VERIFIER_REGRESSION_ONLY_NOT_FORMAL_PERFORMANCE",
               native_json_equivalence=True, semantic_mutations_rejected=rejected,
               bounded_actual_replay=replay, fresh_accepted_lane_steps=32,
               counted_as_additional_root_test_identities=False)
(saved/"REGRESSION_RESULT.json").write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps(receipt, indent=2), flush=True)
