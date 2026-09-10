"""Fresh original-box full-horizon solve with compact per-step evidence.

Only the initial mathematical box is constructed.  Every next input is the
complete state committed by the preceding accepted step.  Full request answers
are never loaded.  Width observation and evidence export are timed separately
from each solver step.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import fields, is_dataclass
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import threading
import time

import torch
import torch_tm_flowpipe as core

from experiments.endpoint_roundoff_repair.frozen import setup
from experiments.live_range_solver.runner import ROOT, initial, observe, run_case
from torch_tm_flowpipe.live_range_checkpoint import save_live_range_checkpoint
from torch_tm_flowpipe.symbolic_remainder import accepted_boundary_sr_queue_sha256
from torch_tm_flowpipe.terminal_checkpoint import tmvector_hashes


COST_FIELDS = (
    "validation_and_layout_s", "host_packet_build_s", "envelope_check_s",
    "device_allocation_s",
    "host_stage_owned_payload_s", "metadata_h2d_enqueue_s", "numeric_h2d_enqueue_s",
    "h2d_sync_s", "kernel_enqueue_s", "kernel_and_sync_host_s", "completion_sync_s",
    "metadata_d2h_enqueue_s", "numeric_d2h_enqueue_s", "d2h_sync_s",
    "scatter_and_private_wrap_s", "device_metadata_h2d_s", "device_numeric_h2d_s",
    "device_kernel_s", "device_metadata_d2h_s", "device_numeric_d2h_s",
    "grouping_and_packing_s", "compute_and_transfers_s", "scatter_and_wrap_s",
    "fallback_s", "total_s",
)
COUNT_FIELDS = (
    "actual_kernel_invocations", "h2d_copy_operations", "d2h_copy_operations",
    "metadata_h2d_bytes", "numeric_h2d_bytes", "metadata_d2h_bytes",
    "numeric_d2h_bytes", "device_allocation_operations",
    "pinned_allocation_operations", "packet_count", "oversize_parent_requests",
    "external_fallback_requests",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def float_hex(value):
    return float(value).hex()


def tensor_record(value):
    value = value.detach().cpu().contiguous()
    return dict(dtype=str(value.dtype), shape=list(value.shape),
        values=([float(item).hex() for item in value.reshape(-1).tolist()]
                if value.is_floating_point() else value.reshape(-1).tolist()))


def interval_record(value):
    return [float(value.lo.detach().cpu()).hex(), float(value.hi.detach().cpu()).hex()]


def compact(value):
    """Lossless encoding for the bounded remainder objects selected below."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite long-horizon evidence scalar")
        return {"float_hex": value.hex()}
    if isinstance(value, torch.Tensor):
        return {"tensor": tensor_record(value)}
    if isinstance(value, core.Interval):
        return {"interval": interval_record(value)}
    if isinstance(value, dict):
        return {str(key): compact(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [compact(item) for item in value]
    if is_dataclass(value):
        return {"type": type(value).__name__,
                "fields": {field.name: compact(getattr(value, field.name)) for field in fields(value)}}
    raise TypeError(f"unsupported long-horizon evidence value: {type(value).__name__}")


def ledger_record(ledger):
    if ledger is None:
        return None
    if not hasattr(ledger, "entries"):
        return compact(ledger)
    return {name: {"lo": tensor_record(lo), "hi": tensor_record(hi)}
            for name, (lo, hi) in ledger.entries.items()}


def decomposition_record(value):
    if value is None:
        return None
    return dict(source_schema=value.source_schema,
        source_schema_version=value.source_schema_version,
        ledger=ledger_record(value.ledger),
        decomposition_lo=tensor_record(value.decomposition_lo),
        decomposition_hi=tensor_record(value.decomposition_hi),
        padding_lo=tensor_record(value.padding_lo), padding_hi=tensor_record(value.padding_hi),
        contains_image=tensor_record(value.contains_image))


def queue_record(queue, claimed_hash=None):
    if queue is None:
        return None
    actual_hash = accepted_boundary_sr_queue_sha256(queue)
    if claimed_hash is not None and actual_hash != claimed_hash:
        raise AssertionError("solver queue hash differs from the accepted state")
    return dict(size=len(queue.J), phi_size=len(queue.Phi_L), max_size=queue.max_size,
        generation=queue.generation, accepted_boundary_index=queue.accepted_boundary_index,
        reset_count=queue.reset_count, owner_schema=queue.owner_schema,
        owner_generations=list(queue.owner_generations),
        owner_boundary_indices=list(queue.owner_boundary_indices),
        scalars=[float(value).hex() for value in queue.scalars], sha256=actual_hash)


def state_fingerprint(current, state, *, queue_hash=None):
    payload = dict(step_index=state.step_index, current=tmvector_hashes(current),
        tmv_pre=tmvector_hashes(state.tmv_pre), tmv_right=tmvector_hashes(state.tmv_right),
        domain=[interval_record(value) for value in state.domain],
        center=[float(value).hex() for value in state.center],
        scales=[float(value).hex() for value in state.scales],
        initial_remainders=([interval_record(value) for value in state.initial_remainders]
                            if state.initial_remainders is not None else None),
        complete_initial_tm=(tmvector_hashes(state.complete_initial_tm)
                             if state.complete_initial_tm is not None else None),
        retained_source_tm=(tmvector_hashes(state.g2_retained_source_tm)
                            if state.g2_retained_source_tm is not None else None),
        symbolic_queue_sha256=(queue_hash or accepted_boundary_sr_queue_sha256(state.symbolic_queue)
                               if state.symbolic_queue is not None else None),
        symbolic_queue_max_size=state.symbolic_queue_max_size,
        structured_remainder_state_type=(type(state.structured_remainder_state).__name__
                                         if state.structured_remainder_state is not None else None),
        bounded_source_ledger_state_type=(type(state.bounded_source_ledger_state).__name__
                                          if state.bounded_source_ledger_state is not None else None),
        g2_shared_column_state_type=(type(state.g2_shared_column_state).__name__
                                     if state.g2_shared_column_state is not None else None))
    return dict(payload=payload, sha256=json_digest(payload))


def stop_reason(counters):
    if not counters:
        return "not_recorded"
    if counters.get("post_accept_failure_count"):
        return "post_accept_failure"
    if counters.get("post_accept_replay_cap_count"):
        return "491_replay_cap"
    if counters.get("post_accept_fixed_point_count"):
        return "fixed_point"
    if counters.get("post_accept_stop_ratio_count"):
        return "0.99_stop_ratio"
    return "no_post_accept_stop_flag"


def percentile(values, fraction):
    values = sorted(values)
    return values[round((len(values)-1)*fraction)] if values else 0


def interval_union_ns(intervals):
    intervals = sorted((start, end) for start, end in intervals if start > 0 and end >= start)
    total = 0
    for start, end in intervals:
        if not total or start > current_end:
            total += end - start
            current_end = end
        elif end > current_end:
            total += end - current_end
            current_end = end
    return total


class GroupAccumulator:
    def __init__(self):
        self._lock = threading.Lock()
        self._groups = defaultdict(list)

    def __call__(self, group):
        generations = set(group["request_generations"])
        if len(generations) != 1:
            raise AssertionError("one original B1 step packet crossed accepted generations")
        with self._lock:
            self._groups[generations.pop()].append(group)

    def pop(self, generation):
        with self._lock:
            groups = self._groups.pop(generation, [])
        waits = [value for group in groups for value in group["request_wait_ns"]]
        costs = {name: sum(float(group["timing"].get(name, 0)) for group in groups)
                 for name in COST_FIELDS}
        counts = {name: sum(int(group["timing"].get(name, 0)) for group in groups)
                  for name in COUNT_FIELDS}
        scheduler = {name: sum(group["scheduler_costs"][name] for group in groups)/1e9
                     for name in ("ownership_copy_sum_ns", "submit_lock_wait_sum_ns",
                                  "deliver_lock_wait_sum_ns", "future_set_sum_ns",
                                  "consume_lock_wait_sum_ns", "future_wait_sum_ns")}
        scheduler.update(
            ownership_copy_union_s=interval_union_ns([
                interval for group in groups
                for interval in group["scheduler_costs"]["ownership_copy_intervals_ns"]])/1e9,
            submit_lock_union_s=interval_union_ns([
                interval for group in groups
                for interval in group["scheduler_costs"]["submit_lock_intervals_ns"]])/1e9,
            future_wait_union_s=interval_union_ns([
                interval for group in groups
                for interval in group["scheduler_costs"]["future_wait_intervals_ns"]])/1e9)
        return dict(groups=len(groups), requests=sum(group["size"] for group in groups),
            reasons=dict(Counter(group["reason"] for group in groups)),
            service_span_s=sum((group["end_ns"]-group["start_ns"])/1e9 for group in groups),
            service_thread_cpu_s=sum(group["thread_cpu_ns"] for group in groups)/1e9,
            selection_wall_s=sum(group["selection_wall_ns"] for group in groups)/1e9,
            selection_thread_cpu_s=sum(group["selection_thread_cpu_ns"] for group in groups)/1e9,
            wait_samples=len(waits), wait_p50_s=statistics.median(waits)/1e9 if waits else 0,
            wait_p95_s=percentile(waits, .95)/1e9, wait_max_s=max(waits, default=0)/1e9,
            request_wait_sum_s=sum(waits)/1e9,
            ready_requests_max=max((group["selection"]["ready_requests"] for group in groups), default=0),
            ready_keys_max=max((group["selection"]["ready_keys"] for group in groups), default=0),
            costs=costs, counts=counts, scheduler_costs_s=scheduler)

    def assert_empty(self):
        with self._lock:
            if self._groups:
                raise AssertionError(f"unattributed service generations: {sorted(self._groups)}")


def checkpoint_steps(plant, steps):
    if plant == "van_der_pol":
        values = {99, 100, 101, 999, 1000, steps}
    else:
        values = {999, 1000, steps}
    return sorted(value for value in values if 0 < value <= steps)


def run_long_horizon(plant, route, steps, output):
    if plant not in {"van_der_pol", "brusselator"}:
        raise ValueError("unknown plant")
    if route not in {"Gp", "G0"}:
        raise ValueError("long-horizon publication route must be Gp or G0")
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    config = setup(plant)[0]
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    original_current, original_state = initial(plant, 0, original=True)
    initial_record = dict(scope="ORIGINAL_UNPARTITIONED_B1", config=config.as_dict(),
        state=state_fingerprint(original_current, original_state),
        initial_decimal_box=config.as_dict()["initial_decimal_box"],
        normalized_domain=[interval_record(value) for value in original_state.domain],
        current_range=[interval_record(value) for value in original_current.range_box()],
        source_sha=source_sha)
    (output / "INITIAL_SOURCE.json").write_text(
        json.dumps(initial_record, indent=2, sort_keys=True, allow_nan=False)+"\n")

    accumulator = GroupAccumulator()
    milestones = set(checkpoint_steps(plant, steps))
    export_totals = Counter()
    row_count = 0
    last_after = None
    exact_elapsed = Fraction(0)
    checkpoint_receipts = []
    step_file = output / "full_horizon_steps.jsonl.gz"
    with gzip.open(step_file, "wt", compresslevel=6) as stream:
        def on_step(task, offset, current, state, segment, runner_row):
            nonlocal row_count, last_after, exact_elapsed
            export_start = time.perf_counter()
            stats = segment.flowstar_normal_stats or {}
            before_queue = queue_record(state.symbolic_queue, stats.get("c3_queue_hash_before"))
            after_state = segment.flowstar_normal_state
            after_queue = queue_record(after_state.symbolic_queue, stats.get("c3_queue_hash_after"))
            observer_start = time.perf_counter()
            bounds = observe(segment)
            observer_s = time.perf_counter() - observer_start
            before = state_fingerprint(current, state,
                queue_hash=before_queue["sha256"] if before_queue else None)
            after = state_fingerprint(segment.reset_tm, after_state,
                queue_hash=after_queue["sha256"] if after_queue else None)
            if last_after is not None and before != last_after:
                raise AssertionError("long-horizon next input is not the preceding complete state")
            service = accumulator.pop(state.step_index)
            exact_elapsed += Fraction(float(segment.h))
            exact_time = exact_elapsed
            counters = dict(segment.backend_counters or {})
            row = dict(schema="live-gpu-packet-full-step-v1", plant=plant, route=route,
                offset=offset, step=after_state.step_index, accepted=True, status=segment.status,
                h_hex=float(segment.h).hex(), next_h_hex=(float(segment.next_h).hex()
                    if segment.next_h is not None else None), exact_time=str(exact_time),
                nominal_time=float(exact_time), step_start_ns=runner_row["step_start_ns"],
                step_end_ns=runner_row["step_end_ns"],
                solver_step_wall_s=(runner_row["step_end_ns"]-runner_row["step_start_ns"])/1e9,
                validation_attempts=segment.validation_attempts,
                step_rejections=segment.step_rejections,
                post_accept_refinement_count=counters.get("post_accept_replay_calls", 0),
                refinement_stop_reason=stop_reason(counters),
                endpoint_tightening_applied=segment.endpoint_tightening_applied,
                endpoint_tightening_validation_method=segment.endpoint_tightening_validation_method,
                bounds=bounds, endpoint_substitution_E=compact(segment.endpoint_substitution_roundoff),
                endpoint_cutoff_remainder=compact(segment.endpoint_cutoff_remainder),
                endpoint_ordinary_remainder=compact(segment.endpoint_ordinary_remainder),
                endpoint_total_remainder=compact(segment.endpoint_total_remainder),
                tube_ordinary_remainder=compact(segment.tube_ordinary_remainder),
                tube_total_remainder=compact(segment.tube_total_remainder),
                endpoint_total_structured_remainder=compact(segment.endpoint_total_structured_remainder),
                tube_total_structured_remainder=compact(segment.tube_total_structured_remainder),
                candidate_remainder=compact(segment.candidate_remainder),
                picard_image_remainder=compact(segment.picard_image_remainder),
                subset_margin=compact(segment.subset_margin),
                validated_remainder_ledger=ledger_record(segment.validated_remainder_ledger),
                validated_remainder_decomposition=decomposition_record(
                    segment.validated_remainder_decomposition),
                dense_endpoint_ledger=ledger_record(segment.dense_endpoint_ledger),
                backend_counters=compact(counters), service=service,
                queue_before=before_queue, queue_after=after_queue,
                state_before=before, state_after=after, observer_s=observer_s)
            checkpoint_start = time.perf_counter()
            if after_state.step_index in milestones:
                checkpoint = output / "checkpoints" / f"step_{after_state.step_index:04d}"
                save_live_range_checkpoint(checkpoint, task,
                    scheduler=dict(accepted_steps=after_state.step_index,
                        time_exact=str(exact_time), next_h_hex=float(
                            segment.next_h if segment.next_h is not None else segment.h).hex()),
                    contract=dict(plant=plant, config=config.as_dict(), order=config.order,
                                  route=route, original_unpartitioned=True),
                    provenance=dict(purpose="fresh original-box GPU-range full horizon",
                        scientific_sha=source_sha, packet_mode=route == "Gp"))
                checkpoint_receipts.append(dict(step=after_state.step_index,
                    path=str(checkpoint.relative_to(output)), state_sha256=after["sha256"]))
            checkpoint_s = time.perf_counter() - checkpoint_start
            row["checkpoint_export_s"] = checkpoint_s
            row["evidence_prepare_s"] = time.perf_counter() - export_start
            write_start = time.perf_counter()
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
            export_totals["json_gzip_write_s"] += time.perf_counter() - write_start
            export_totals["observer_s"] += observer_s
            export_totals["checkpoint_s"] += checkpoint_s
            export_totals["evidence_prepare_s"] += row["evidence_prepare_s"]
            row_count += 1
            last_after = after

        invocation_start = time.perf_counter()
        run, _, states = run_case(plant, [0], steps, route,
            run_id=f"full-horizon-{plant}-{route}", diagnostic=False, original=True,
            on_step=on_step, service_group_callback=accumulator,
            retain_service_groups=False, retain_service_waits=False)
        invocation_s = time.perf_counter() - invocation_start
    accumulator.assert_empty()
    final_state = states["0"]
    final_fingerprint = state_fingerprint(*final_state)
    if last_after is not None and final_fingerprint != last_after:
        raise AssertionError("returned final state differs from final streamed step")
    failure = [row for rows in run["records"].values() for row in rows if not row["accepted"]]
    metadata = {key: value for key, value in run.items()
                if key not in {"groups", "wait_ns", "records"}}
    metadata.update(retained_service_groups=False, retained_service_wait_samples=False,
                    export_totals_s=dict(export_totals), invocation_s=invocation_s)
    (output / "RUN_METADATA.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)+"\n")
    result = dict(schema="live-gpu-packet-full-horizon-result-v1", plant=plant, route=route,
        source_sha=source_sha, scope="ORIGINAL_UNPARTITIONED_B1",
        requested_steps=steps, recorded_steps=row_count,
        accepted_lane_steps=run["accepted_lane_steps"], successful_tasks=run["successful_tasks"],
        achieved=row_count == steps and run["successful_tasks"] == 1,
        target=("T10" if plant == "van_der_pol" else "T20") if steps == 1000 else "DEVELOPMENT_PREFIX",
        exact_final_time=str(exact_elapsed) if row_count == steps else None,
        first_failure=failure[0] if failure else None, previous_answers_loaded=False,
        each_next_input_from_preceding_complete_state=True,
        gpu_scope="GPU range service participating in the complete solver chain",
        full_gpu_engine=False, whole_solver_formal_proof=False,
        checkpoint_receipts=checkpoint_receipts, final_state_sha256=final_fingerprint["sha256"],
        files={"initial_source": sha256(output/"INITIAL_SOURCE.json"),
               "steps": sha256(step_file), "run_metadata": sha256(output/"RUN_METADATA.json")})
    (output / "LONG_HORIZON_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plant", choices=["van_der_pol", "brusselator"], required=True)
    parser.add_argument("--route", choices=["Gp", "G0"], default="Gp")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run_long_horizon(args.plant, args.route, args.steps, args.output)


if __name__ == "__main__":
    main()
