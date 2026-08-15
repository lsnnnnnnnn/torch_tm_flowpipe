#!/usr/bin/env python3
"""Fresh native-G1 terminal owner interventions on the last accepted boundary."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "experiments"))

from torch_tm_flowpipe import FlowstarNormalFlowpipeState, PolynomialODE
from torch_tm_flowpipe.flowpipe import flowpipe_step_flowstar_style_adaptive
from torch_tm_flowpipe.source_ledger import metadata_tamper

from audit_g1_owner_interventions_20260815 import (
    CANDIDATE,
    consume,
    frozen_range_policy,
    max_source_coefficient,
    tamper_first_source,
    tm_hash,
    variants,
)
from run_vdp_dense_backend import load_contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-cap-s", type=float, default=1200.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    contract = load_contract()
    ode = PolynomialODE.from_system_spec(contract["canonical_system_spec"])
    normal = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [("1.1", "1.4"), ("2.35", "2.45")], 4
    ).with_bounded_source_g1(4)
    current = normal.normalized_initial_tm(4)
    current_time = 0.0
    h_next = float(contract["h_max"])
    accepted_steps = 0
    last_before = None
    last_accepted = None
    started = time.perf_counter()
    failed = None
    while current_time < 10.0 - 1e-12:
        if time.perf_counter() - started > float(args.wall_cap_s):
            raise TimeoutError("terminal owner audit exceeded wall cap")
        h_try = min(h_next, float(contract["h_max"]), 10.0 - current_time)
        before = normal
        segment = flowpipe_step_flowstar_style_adaptive(
            ode,
            current,
            h=h_try,
            h_min=float(contract["h_min"]),
            h_max=float(contract["h_max"]),
            order=4,
            target_remainder_radius=1e-4,
            cutoff_threshold=1e-10,
            max_validation_attempts=2,
            validation_eps=1e-12,
            validation_mode="flowstar_raw_remainder_compat",
            reset_mode=CANDIDATE,
            step_policy_mode="flowstar_compat",
            flowstar_normal_state=normal,
            tm_backend="dense",
            dense_range_policy=frozen_range_policy(),
        )
        if segment.status != "validated" or segment.reset_tm is None or segment.flowstar_normal_state is None:
            failed = segment
            break
        last_before = before
        last_accepted = segment
        accepted_steps += 1
        current_time += float(segment.h)
        h_next = float(segment.next_h if segment.next_h is not None else min(float(segment.h) * 1.1, contract["h_max"]))
        current = segment.reset_tm
        normal = segment.flowstar_normal_state
    if failed is None or last_before is None or last_accepted is None:
        raise RuntimeError("native G1 did not yield the required terminal prestate")

    candidate_variants, owners = variants(last_before, last_accepted)
    failed_h = float(failed.h)
    consumers: dict[str, Any] = {}
    controls: dict[str, Any] = {}
    for name, reset in candidate_variants.items():
        consumers[name] = consume(ode, reset, h=failed_h)
        if name == "g1_actual":
            continue
        indices = (2, 3)
        maximum = max_source_coefficient(reset, indices)
        if maximum <= 1e-10:
            controls[name] = {
                "payload_control": "NOT_APPLICABLE_OWNER_ABSENT_OR_BELOW_FROZEN_CUTOFF",
                "maximum_source_coefficient": maximum,
            }
            continue
        payload = tamper_first_source(reset, indices)
        consumers[f"{name}__payload_tamper_x2"] = consume(ode, payload, h=failed_h)
        consumers[f"{name}__metadata_tamper"] = consume(
            ode,
            reset,
            h=failed_h,
            nonconsumer_metadata={"owner_label": f"{name}:terminal-metadata-only"},
        )
        payload_changed = (
            consumers[name]["consumer_output_sha256"]
            != consumers[f"{name}__payload_tamper_x2"]["consumer_output_sha256"]
        )
        metadata_preserved = (
            consumers[name]["consumer_output_sha256"]
            == consumers[f"{name}__metadata_tamper"]["consumer_output_sha256"]
        )
        if not payload_changed or not metadata_preserved:
            raise RuntimeError(f"terminal actual-consumer control failed for {name}")
        controls[name] = {
            "payload_control": "PASS_CHANGED_REAL_CONSUMER_OUTPUT",
            "metadata_control": "PASS_PRESERVED_REAL_CONSUMER_OUTPUT",
        }

    actual_state = last_accepted.flowstar_normal_state
    assert actual_state is not None and actual_state.bounded_source_ledger_state is not None
    changed_metadata = replace(
        actual_state,
        bounded_source_ledger_state=metadata_tamper(
            actual_state.bounded_source_ledger_state,
            "terminal-negative-control",
        ),
    )
    metadata_reset = changed_metadata.normalized_initial_tm(4)
    if tm_hash(metadata_reset) != tm_hash(candidate_variants["g1_actual"]):
        raise RuntimeError("terminal metadata changed actual G1 polynomial")
    consumers["g1_actual__metadata_tamper"] = consume(
        ode,
        metadata_reset,
        h=failed_h,
        nonconsumer_metadata={
            "pre_fingerprint": actual_state.bounded_source_ledger_state.fingerprint,
            "post_fingerprint": changed_metadata.bounded_source_ledger_state.fingerprint,
        },
    )
    if (
        consumers["g1_actual"]["consumer_output_sha256"]
        != consumers["g1_actual__metadata_tamper"]["consumer_output_sha256"]
    ):
        raise RuntimeError("terminal G1 metadata changed real consumer output")

    result = {
        "schema": "g1_native_terminal_owner_interventions_v1",
        "candidate": CANDIDATE,
        "initialization": "exact_decimal_contract",
        "accepted_steps": accepted_steps,
        "terminal_time": current_time,
        "terminal_time_hex": current_time.hex(),
        "failed_h": failed_h,
        "failed_h_hex": failed_h.hex(),
        "failure_message": failed.message,
        "failed_raw_picard_image": failed.picard_image_remainder,
        "failed_subset_margin": failed.subset_margin,
        "owners": owners,
        "consumers": consumers,
        "controls": controls,
        "runtime_s": time.perf_counter() - started,
        "selection_rule": "same two recoverable owner transformations as fixed audit; cutoff decides applicability",
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "accepted_steps": accepted_steps,
        "terminal_time": current_time,
        "failed_h": failed_h,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
