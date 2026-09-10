"""Verify streamed full-horizon continuity, widths, queues, timings and checkpoints."""
from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import json
import math
from pathlib import Path
import statistics

from experiments.endpoint_roundoff_repair.frozen import setup
from torch_tm_flowpipe.live_range_checkpoint import load_live_range_checkpoint
from torch_tm_flowpipe.live_range_service import LiveRangeService

from .long_horizon import checkpoint_steps, sha256, state_fingerprint


QUEUE_OWNER_SCHEMAS = {
    "van_der_pol": "c3_cross_step_sr_v1",
    "brusselator": "accepted_boundary_sr_v1",
}


def expected_queue_owner_schema(plant):
    return QUEUE_OWNER_SCHEMAS[plant]


def percentile(values, fraction):
    values = sorted(values)
    return values[round((len(values)-1)*fraction)] if values else None


def check_hex(value):
    parsed = float.fromhex(value)
    assert math.isfinite(parsed)
    return parsed


def check_compact(value):
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, list):
        for item in value:
            check_compact(item)
        return
    assert isinstance(value, dict)
    if set(value) == {"float_hex"}:
        check_hex(value["float_hex"]); return
    if set(value) == {"interval"}:
        lo, hi = map(check_hex, value["interval"])
        assert lo <= hi; return
    if set(value) == {"tensor"}:
        tensor = value["tensor"]
        count = math.prod(tensor["shape"])
        assert len(tensor["values"]) == count
        if "float" in tensor["dtype"]:
            for item in tensor["values"]:
                check_hex(item)
        return
    for item in value.values():
        check_compact(item)


def width_rows(row):
    output = []
    for view in ("endpoint", "tube"):
        bounds = row["bounds"][view]
        assert len(bounds) == 2
        for coordinate, pair in zip(("x", "y"), bounds):
            lo, hi = map(check_hex, pair)
            assert lo <= hi
            output.append(dict(step=row["step"], view=view, coordinate=coordinate,
                               lo=lo, hi=hi, width=hi-lo))
    return output


def verify_long_horizon(path, *, require_achieved=False):
    path = Path(path)
    result = json.loads((path/"LONG_HORIZON_RESULT.json").read_text())
    assert result["schema"] == "live-gpu-packet-full-horizon-result-v1"
    assert result["scope"] == "ORIGINAL_UNPARTITIONED_B1"
    assert result["route"] in {"Gp", "G0"}
    assert not result["previous_answers_loaded"]
    assert result["each_next_input_from_preceding_complete_state"]
    assert not result["full_gpu_engine"] and not result["whole_solver_formal_proof"]
    assert sha256(path/"INITIAL_SOURCE.json") == result["files"]["initial_source"]
    assert sha256(path/"full_horizon_steps.jsonl.gz") == result["files"]["steps"]
    assert sha256(path/"RUN_METADATA.json") == result["files"]["run_metadata"]
    initial = json.loads((path/"INITIAL_SOURCE.json").read_text())
    config = setup(result["plant"])[0]
    assert initial["scope"] == result["scope"]
    assert initial["config"] == json.loads(json.dumps(config.as_dict()))
    assert initial["initial_decimal_box"] == config.as_dict()["initial_decimal_box"]
    metadata = json.loads((path/"RUN_METADATA.json").read_text())
    assert metadata["plant"] == result["plant"] and metadata["route"] == result["route"]
    assert metadata["original"] is True and metadata["scope"] == "ORIGINAL_B1"
    assert metadata["previous_answers_loaded"] is False
    assert metadata["retained_service_groups"] is False
    assert metadata["retained_service_wait_samples"] is False
    assert metadata["source_sha"] == result["source_sha"] == initial["source_sha"]

    expected_h = float(.01 if result["plant"] == "van_der_pol" else .02)
    capacity = 100 if result["plant"] == "van_der_pol" else 1000
    queue_owner_schema = expected_queue_owner_schema(result["plant"])
    rows, widths, step_hashes = [], [], {}
    previous_after = initial["state"]
    previous_end = metadata["start_ns"]
    exact_elapsed = Fraction(0)
    total_requests = total_groups = total_kernels = total_packets = 0
    with gzip.open(path/"full_horizon_steps.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            rows.append(row)
            step = len(rows)
            assert row["schema"] == "live-gpu-packet-full-step-v1"
            assert row["plant"] == result["plant"] and row["route"] == result["route"]
            assert row["offset"] == row["step"] == step
            assert row["accepted"] and row["status"] == "validated"
            step_h = float.fromhex(row["h_hex"])
            assert step_h > 0 and math.isfinite(step_h)
            assert step_h == expected_h
            if row["next_h_hex"] is not None:
                assert float.fromhex(row["next_h_hex"]) == expected_h
            exact_elapsed += Fraction(step_h)
            assert Fraction(row["exact_time"]) == exact_elapsed
            assert math.isclose(row["nominal_time"], float(Fraction(row["exact_time"])),
                                rel_tol=0, abs_tol=0)
            assert previous_end <= row["step_start_ns"] < row["step_end_ns"] <= metadata["end_ns"]
            assert math.isclose(row["solver_step_wall_s"],
                (row["step_end_ns"]-row["step_start_ns"])/1e9, rel_tol=1e-12, abs_tol=1e-12)
            previous_end = row["step_end_ns"]
            assert row["state_before"] == previous_after, "state continuity was broken"
            assert row["state_before"]["payload"]["step_index"] == step-1
            assert row["state_after"]["payload"]["step_index"] == step
            previous_after = row["state_after"]
            step_hashes[step] = previous_after["sha256"]
            before_queue, after_queue = row["queue_before"], row["queue_after"]
            if before_queue is not None:
                assert before_queue["size"] == (step-1) % capacity
                assert before_queue["accepted_boundary_index"] == step-1
            assert after_queue is not None
            assert after_queue["size"] == step % capacity
            assert after_queue["phi_size"] == after_queue["size"]
            assert after_queue["accepted_boundary_index"] == step
            assert after_queue["generation"] == step
            assert after_queue["max_size"] == capacity
            assert after_queue["reset_count"] == step // capacity
            cycle_start = step - after_queue["size"]
            expected_owners = list(range(cycle_start + 1, step + 1))
            assert after_queue["owner_generations"] == expected_owners
            assert after_queue["owner_boundary_indices"] == expected_owners
            assert len(after_queue["scalars"]) == 2
            assert all(math.isfinite(float.fromhex(value)) for value in after_queue["scalars"])
            assert after_queue["owner_schema"] == queue_owner_schema
            assert after_queue["sha256"] == row["state_after"]["payload"]["symbolic_queue_sha256"]
            if before_queue is not None:
                assert before_queue["sha256"] == row["state_before"]["payload"]["symbolic_queue_sha256"]
            assert row["validation_attempts"] >= 1 and row["step_rejections"] >= 0
            assert 0 <= row["post_accept_refinement_count"] <= 491
            assert row["refinement_stop_reason"] in {
                "0.99_stop_ratio", "fixed_point", "491_replay_cap",
                "post_accept_failure", "no_post_accept_stop_flag", "not_recorded"}
            for name in (
                "endpoint_substitution_E", "endpoint_cutoff_remainder",
                "endpoint_ordinary_remainder", "endpoint_total_remainder",
                "tube_ordinary_remainder", "tube_total_remainder",
                "endpoint_total_structured_remainder", "tube_total_structured_remainder",
                "candidate_remainder", "picard_image_remainder", "subset_margin",
                "validated_remainder_ledger", "validated_remainder_decomposition",
                "dense_endpoint_ledger", "backend_counters"):
                check_compact(row[name])
            widths.extend(width_rows(row))
            service = row["service"]
            assert service["groups"] > 0 and service["requests"] > 0
            assert service["wait_samples"] == service["requests"]
            assert service["service_span_s"] <= row["solver_step_wall_s"] + 1e-6
            scheduler = service["scheduler_costs_s"]
            assert all(value >= 0 for value in scheduler.values())
            assert scheduler["ownership_copy_union_s"] <= row["solver_step_wall_s"] + 1e-6
            assert scheduler["future_wait_union_s"] <= row["solver_step_wall_s"] + 1e-6
            counts = service["counts"]
            if result["route"] == "Gp":
                assert counts["actual_kernel_invocations"] == 4*counts["packet_count"]
                assert counts["h2d_copy_operations"] == 2*counts["packet_count"]
                assert counts["d2h_copy_operations"] == 2*counts["packet_count"]
                assert counts["oversize_parent_requests"] == 0
            else:
                assert counts["packet_count"] == 0
                assert counts["actual_kernel_invocations"] % 4 == 0
            total_requests += service["requests"]
            total_groups += service["groups"]
            total_kernels += counts["actual_kernel_invocations"]
            total_packets += counts["packet_count"]

    assert len(rows) == result["recorded_steps"] == result["accepted_lane_steps"]
    assert metadata["accepted_lane_steps"] == result["accepted_lane_steps"]
    assert metadata["counts"]["submitted"] == total_requests
    assert metadata["counts"]["groups"] == total_groups
    assert previous_after["sha256"] == result["final_state_sha256"]
    achieved = len(rows) == result["requested_steps"] and result["successful_tasks"] == 1
    assert result["achieved"] == achieved
    if require_achieved:
        assert achieved
    if achieved:
        assert result["first_failure"] is None
        assert Fraction(result["exact_final_time"]) == exact_elapsed
        if result["requested_steps"] == 1000:
            target = 10.0 if result["plant"] == "van_der_pol" else 20.0
            assert result["target"] == ("T10" if target == 10.0 else "T20")
            assert float(exact_elapsed) == target
        receipts = result["checkpoint_receipts"]
        assert [row["step"] for row in receipts] == checkpoint_steps(
            result["plant"], result["requested_steps"])
        assert len({row["path"] for row in receipts}) == len(receipts)
        for receipt in receipts:
            checkpoint = (path / receipt["path"]).resolve()
            assert checkpoint.is_relative_to(path.resolve()) and checkpoint.is_dir()
            assert receipt["state_sha256"] == step_hashes[receipt["step"]]
            with LiveRangeService() as service:
                task = load_live_range_checkpoint(checkpoint, service)
                loaded = state_fingerprint(*task.accepted_state)
            assert loaded["sha256"] == receipt["state_sha256"]
        final_receipt = max(receipts, key=lambda row: row["step"])
        assert final_receipt["step"] == len(rows)
        checkpoint = path/final_receipt["path"]
        with LiveRangeService() as service:
            task = load_live_range_checkpoint(checkpoint, service)
            loaded = state_fingerprint(*task.accepted_state)
        assert loaded == previous_after
        assert loaded["sha256"] == final_receipt["state_sha256"]

    summary = []
    for view in ("endpoint", "tube"):
        for coordinate in ("x", "y"):
            chosen = [row for row in widths if row["view"] == view and row["coordinate"] == coordinate]
            values = [row["width"] for row in chosen]
            worst = max(chosen, key=lambda row: row["width"]) if chosen else None
            summary.append(dict(view=view, coordinate=coordinate, samples=len(values),
                width_p50=statistics.median(values) if values else None,
                width_p95=percentile(values, .95), width_max=worst["width"] if worst else None,
                width_max_step=worst["step"] if worst else None))
    return dict(verified=True, achieved=achieved, plant=result["plant"], route=result["route"],
        steps=len(rows), requests=total_requests, groups=total_groups,
        packet_kernel_invocations=total_kernels, packets=total_packets,
        exact_final_time=result["exact_final_time"], queue_owner_schema=queue_owner_schema,
        width_summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--require-achieved", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_long_horizon(args.path, require_achieved=args.require_achieved),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
