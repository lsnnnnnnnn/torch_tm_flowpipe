"""Verify packet campaigns and derive measured costs and paired performance."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics

from experiments.range_batch_device.common import read, save, sha
from experiments.live_range_solver.analyze import compare_state, widths
from experiments.live_range_solver.verify import load_run, verify_run
from experiments.live_range_solver.runner import PARTITION, ROOT

from .campaign import PARENT_ROOT, diagnostic_cases, expected_ids, formal_cases


def csv_rows(path, rows):
    rows = list(rows)
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    left = int(position)
    right = min(left + 1, len(values) - 1)
    return values[left] + (values[right] - values[left]) * (position - left)


def interval_union_ns(intervals):
    intervals = sorted((begin, end) for begin, end in intervals if begin > 0 and end >= begin)
    if not intervals:
        return 0
    total, current_end = 0, intervals[0][0]
    for begin, end in intervals:
        if begin > current_end:
            total += end - begin
            current_end = end
        elif end > current_end:
            total += end - current_end
            current_end = end
    return total


def aggregate_run(run):
    groups = run["groups"]
    timings = [group["timing"] for group in groups]
    scheduler = [group["scheduler_costs"] for group in groups if "scheduler_costs" in group]
    sizes = [group["size"] for group in groups]
    packet_details = [packet for timing in timings
                      for packet in timing.get("packet_timings", [])]
    previous_capacity = {}
    for packet in packet_details:
        capacity = packet["scratch"]["capacity"]
        assert all(int(capacity.get(name, 0)) >= int(previous_capacity.get(name, 0))
                   for name in set(capacity) | set(previous_capacity))
        previous_capacity = capacity
    growth_packets = [index for index, packet in enumerate(packet_details, 1)
                      if packet.get("device_allocation_operations", 0)]
    last_growth_packet = max(growth_packets, default=0)
    final_scratch = packet_details[-1]["scratch"] if packet_details else {}

    def timing_sum(name):
        return sum(float(timing.get(name, 0)) for timing in timings)

    def timing_int(name):
        return sum(int(timing.get(name, 0)) for timing in timings)

    def scheduler_sum(name):
        return sum(int(row[name]) for row in scheduler)

    def scheduler_intervals(name):
        return [interval for row in scheduler for interval in row[name]]

    service_s = sum((group["end_ns"] - group["start_ns"]) / 1e9 for group in groups)
    requests = run["counts"].get("submitted", 0)
    packet_count = timing_int("packet_count")
    semantic_groups = sum(len(timing.get("group_sizes", [])) for timing in timings)
    return dict(run_id=run["run_id"], plant=run["plant"], route=run["route"],
        batch=len(run["ids"]), steps=run["steps"], lane_steps=run["successful_lane_steps"],
        requests=requests, dispatches=len(groups), semantic_device_groups=semantic_groups,
        packets=packet_count, mean_dispatch_size=statistics.mean(sizes) if sizes else 0,
        p95_dispatch_size=percentile(sizes, .95), max_dispatch_size=max(sizes, default=0),
        mean_ready_requests=(statistics.mean(group["selection"]["ready_requests"] for group in groups)
                             if scheduler else None),
        mean_ready_keys=(statistics.mean(group["selection"]["ready_keys"] for group in groups)
                         if scheduler else None),
        mean_selected_keys=(statistics.mean(group["selection"]["selected_keys"] for group in groups)
                            if scheduler else None),
        wall_s=run["wall_s"], cpu_s=run["cpu_s"], throughput=run["throughput"],
        initial_state_s=run["initial_state_s"],
        serialization_s=run.get("serialization_s", 0), service_wall_s=service_s,
        service_thread_cpu_s=sum(group.get("thread_cpu_ns", 0) for group in groups) / 1e9,
        selection_wall_s=sum(group.get("selection_wall_ns", 0) for group in groups) / 1e9,
        selection_thread_cpu_s=sum(group.get("selection_thread_cpu_ns", 0)
                                   for group in groups) / 1e9,
        ownership_copy_task_sum_s=scheduler_sum("ownership_copy_sum_ns") / 1e9,
        ownership_copy_union_s=interval_union_ns(
            scheduler_intervals("ownership_copy_intervals_ns")) / 1e9,
        submit_lock_wait_task_sum_s=scheduler_sum("submit_lock_wait_sum_ns") / 1e9,
        submit_lock_wait_union_s=interval_union_ns(
            scheduler_intervals("submit_lock_intervals_ns")) / 1e9,
        deliver_lock_wait_task_sum_s=scheduler_sum("deliver_lock_wait_sum_ns") / 1e9,
        future_set_task_sum_s=scheduler_sum("future_set_sum_ns") / 1e9,
        consume_lock_wait_task_sum_s=scheduler_sum("consume_lock_wait_sum_ns") / 1e9,
        future_wait_task_sum_s=scheduler_sum("future_wait_sum_ns") / 1e9,
        future_wait_union_s=interval_union_ns(
            scheduler_intervals("future_wait_intervals_ns")) / 1e9,
        validation_and_layout_s=timing_sum("validation_and_layout_s"),
        host_packet_build_s=timing_sum("host_packet_build_s"),
        envelope_check_s=timing_sum("envelope_check_s"),
        host_stage_owned_payload_s=timing_sum("host_stage_owned_payload_s"),
        grouping_and_packing_s=timing_sum("grouping_and_packing_s"),
        device_allocation_s=timing_sum("device_allocation_s"),
        metadata_h2d_enqueue_s=timing_sum("metadata_h2d_enqueue_s"),
        numeric_h2d_enqueue_s=timing_sum("numeric_h2d_enqueue_s"),
        h2d_sync_s=timing_sum("h2d_sync_s"),
        pure_device_metadata_h2d_s=timing_sum("device_metadata_h2d_s"),
        pure_device_numeric_h2d_s=timing_sum("device_numeric_h2d_s"),
        kernel_enqueue_s=timing_sum("kernel_enqueue_s"),
        kernel_and_sync_host_s=timing_sum("kernel_and_sync_host_s")
            or timing_sum("kernel_and_sync_s"),
        pure_device_kernel_s=timing_sum("device_kernel_s"),
        metadata_d2h_enqueue_s=timing_sum("metadata_d2h_enqueue_s"),
        numeric_d2h_enqueue_s=timing_sum("numeric_d2h_enqueue_s"),
        d2h_sync_s=timing_sum("d2h_sync_s"),
        pure_device_metadata_d2h_s=timing_sum("device_metadata_d2h_s"),
        pure_device_numeric_d2h_s=timing_sum("device_numeric_d2h_s"),
        scatter_and_private_wrap_s=timing_sum("scatter_and_private_wrap_s")
            or timing_sum("scatter_and_wrap_s"),
        actual_kernel_invocations=timing_int("actual_kernel_invocations"),
        h2d_copy_operations=timing_int("h2d_copy_operations"),
        d2h_copy_operations=timing_int("d2h_copy_operations"),
        metadata_h2d_bytes=timing_int("metadata_h2d_bytes"),
        numeric_h2d_bytes=timing_int("numeric_h2d_bytes"),
        metadata_d2h_bytes=timing_int("metadata_d2h_bytes"),
        numeric_d2h_bytes=timing_int("numeric_d2h_bytes"),
        device_allocation_operations=timing_int("device_allocation_operations"),
        pinned_allocation_operations=timing_int("pinned_allocation_operations"),
        scratch_packets=len(packet_details), scratch_growth_packets=len(growth_packets),
        scratch_last_growth_packet=last_growth_packet,
        scratch_no_growth_final_quarter=(not packet_details or
            last_growth_packet <= max(1, math.ceil(.75 * len(packet_details)))),
        scratch_final_device_capacity_bytes=final_scratch.get(
            "current_device_capacity_bytes", 0),
        scratch_cumulative_device_allocation_bytes=final_scratch.get(
            "cumulative_device_allocation_bytes", 0),
        external_cpu_fallback_requests=timing_int("external_fallback_requests"),
        oversize_parent_requests=timing_int("oversize_parent_requests"),
        peak_rss_kib=run["peak_rss_kib"], peak_gpu_allocated_bytes=run["peak_gpu_allocated_bytes"],
        cold_startup_s=(run.get("cold_startup_outside_timing") or {}).get("wall_s", 0),
        non_service_wall_s=run["wall_s"] - service_s,
        zero_service_amdahl_upper=(run["wall_s"] / (run["wall_s"] - service_s)
                                   if run["wall_s"] > service_s else math.inf))


def check_case(run, case, plan, *, diagnostic):
    assert run["source_sha"] == plan["source_sha"]
    assert run["partition_sha256"] == sha(PARTITION)
    assert run["plant"] == case["plant"] and run["route"] == case["route"]
    assert run["ids"] == expected_ids(case) and run["steps"] == case["steps"]
    assert run["diagnostic"] is diagnostic
    assert run["successful_tasks"] == len(run["ids"])
    assert run["successful_lane_steps"] == len(run["ids"]) * case["steps"]
    assert run["counts"].get("cancelled_requests", 0) == 0
    assert run["max_wait_s"] == .020 and run["max_group"] == 32
    assert run["packet_mode"] is (run["route"] == "Gp")
    assert run["checkpoint"] == case["options"].get("checkpoint")
    assert run["scope"] == ("RESUMED_LOCAL_WINDOW" if run["checkpoint"] else "FIXED_PARTITION")


def diagnostic_gate(path, *, recompute=True):
    path = Path(path)
    artifact = path.parent
    plan = read(path / "CAMPAIGN_PLAN.json")
    cases = diagnostic_cases()
    assert plan["phase"] == "diagnostic" and plan["cases"] == cases
    tests = read(artifact / "tests/PACKET_TEST_RESULT.json")
    assert tests["passed"] is True and tests["source_sha"] == plan["source_sha"]
    receipts, runs = [], {}
    for case in cases:
        run, events = load_run(path / case["name"])
        check_case(run, case, plan, diagnostic=True)
        receipt = verify_run(run, events, recompute=recompute)
        run["serialization_s"] = read(path / case["name"] / "summary.json")["serialization_s"]
        receipts.append(receipt)
        runs[case["name"]] = run
        save(path / "verification_progress.json", receipts)

    equivalence = []
    for plant in ("van_der_pol", "brusselator"):
        for batch in (1, 2):
            reference = runs[f"small-b{batch}-{plant}-G0"]
            equivalence.append(compare_state(reference, runs[f"small-b{batch}-{plant}-Gp"]))
            equivalence.append(compare_state(reference, runs[f"small-b{batch}-{plant}-S_gpu"]))
        for batch in (8, 32):
            equivalence.append(compare_state(runs[f"prefix-b{batch}-{plant}-G0"],
                                             runs[f"prefix-b{batch}-{plant}-Gp"]))
        equivalence.append(compare_state(runs[f"history-reset-{plant}-G0"],
                                         runs[f"history-reset-{plant}-Gp"]))
        parent_gpu, _ = load_run(PARENT_ROOT / "diagnostic" / f"long-b2-{plant}-G")
        equivalence.append(compare_state(parent_gpu, runs[f"continuous-b2-{plant}-Gp"]))
        equivalence[-1]["reference_evidence"] = "REUSED_PARENT_G0_NUMERICAL_ANCHOR"
    save(artifact / "same_backend_state_equivalence.json", equivalence)

    width_rows = []
    for plant in ("van_der_pol", "brusselator"):
        for batch in (8, 32):
            parent_cpu, _ = load_run(PARENT_ROOT / "diagnostic" / f"prefix-b{batch}-{plant}-S")
            width_rows.extend(widths(parent_cpu, runs[f"prefix-b{batch}-{plant}-Gp"],
                                     f"REUSED_CPU_prefix-b{batch}"))
        parent_cpu, _ = load_run(PARENT_ROOT / "diagnostic" / f"long-b2-{plant}-S")
        width_rows.extend(widths(parent_cpu, runs[f"continuous-b2-{plant}-Gp"],
                                 "REUSED_CPU_continuous-b2"))
    csv_rows(artifact / "diagnostic_width_comparison.csv", width_rows)
    csv_rows(artifact / "diagnostic_width_over_1p10.csv",
             [row for row in width_rows if row["warning"]])

    costs = [aggregate_run(run) for run in runs.values()]
    csv_rows(artifact / "packet_cost_breakdown.csv", costs)
    count_fields = ("run_id", "plant", "route", "batch", "steps", "lane_steps",
        "requests", "dispatches", "semantic_device_groups", "packets",
        "mean_dispatch_size", "mean_ready_requests", "mean_ready_keys", "mean_selected_keys",
        "actual_kernel_invocations", "h2d_copy_operations", "d2h_copy_operations",
        "metadata_h2d_bytes", "numeric_h2d_bytes", "metadata_d2h_bytes", "numeric_d2h_bytes",
        "device_allocation_operations", "pinned_allocation_operations",
        "scratch_packets", "scratch_growth_packets", "scratch_last_growth_packet",
        "scratch_no_growth_final_quarter", "scratch_final_device_capacity_bytes",
        "scratch_cumulative_device_allocation_bytes",
        "external_cpu_fallback_requests", "oversize_parent_requests")
    csv_rows(artifact / "actual_launch_transfer_counts.csv",
             [{name: row[name] for name in count_fields} for row in costs])
    audit_requests = sum(row["arithmetic"].get("raw_requests", 0) for row in receipts)
    gate = dict(schema="live-gpu-packet-correctness-gate-v1", passed=bool(recompute),
        source_sha=plan["source_sha"], runs=len(receipts), tests=tests,
        captured_requests_fraction_checked=audit_requests,
        same_backend_complete_segments=sum(row["complete_segments_compared"] for row in equivalence),
        equivalence=equivalence, diagnostic_width_rows=len(width_rows),
        diagnostic_width_over_1p10=sum(row["warning"] for row in width_rows),
        packet_layout_offset_identity_lifetime_faults_checked=True,
        full_1000_step_runs_checked=0, full_gpu_engine=False,
        whole_solver_formal_proof=False)
    save(path / "CORRECTNESS_GATE.json", gate)
    return gate


def formal_summary(path):
    path = Path(path)
    artifact = path.parent
    plan = read(path / "CAMPAIGN_PLAN.json")
    cases = formal_cases()
    assert plan["phase"] == "formal" and plan["cases"] == cases
    gate = read(artifact / "diagnostic/CORRECTNESS_GATE.json")
    assert gate["passed"] and gate["source_sha"] == plan["source_sha"]
    rows, runs = [], {}
    for case in cases:
        run, events = load_run(path / case["name"])
        check_case(run, case, plan, diagnostic=False)
        verify_run(run, events, recompute=False)
        run["serialization_s"] = read(path / case["name"] / "summary.json")["serialization_s"]
        row = aggregate_run(run)
        row.update(block=case["block"], order_position=case["order_position"])
        rows.append(row)
        runs[case["name"]] = run
    csv_rows(artifact / "timings_raw.csv", rows)

    paired = []
    for plant in ("van_der_pol", "brusselator"):
        for block in range(5):
            selected = {route: next(row for row in rows if row["plant"] == plant
                        and row["block"] == block and row["route"] == route)
                        for route in ("S", "Q", "G0", "Gp")}
            assert all(row["lane_steps"] == 640 for row in selected.values())
            faster_cpu = min(selected["S"]["wall_s"], selected["Q"]["wall_s"])
            paired.append(dict(plant=plant, block=block,
                order="/".join(plan["formal_orders"][block]),
                S_wall_s=selected["S"]["wall_s"], Q_wall_s=selected["Q"]["wall_s"],
                G0_wall_s=selected["G0"]["wall_s"], Gp_wall_s=selected["Gp"]["wall_s"],
                G0_over_Gp=selected["G0"]["wall_s"] / selected["Gp"]["wall_s"],
                S_over_Gp=selected["S"]["wall_s"] / selected["Gp"]["wall_s"],
                Q_over_Gp=selected["Q"]["wall_s"] / selected["Gp"]["wall_s"],
                faster_strict_cpu_over_Gp=faster_cpu / selected["Gp"]["wall_s"],
                Gp_faster_than_G0=selected["Gp"]["wall_s"] < selected["G0"]["wall_s"],
                Gp_faster_than_faster_cpu=selected["Gp"]["wall_s"] < faster_cpu,
                successful_lane_steps=640))
    csv_rows(artifact / "paired_speedups.csv", paired)

    decisions = {}
    for plant in ("van_der_pol", "brusselator"):
        chosen = [row for row in paired if row["plant"] == plant]
        gpu = [row["G0_over_Gp"] for row in chosen]
        cpu = [row["faster_strict_cpu_over_Gp"] for row in chosen]
        decisions[plant] = dict(G0_over_Gp=gpu, G0_over_Gp_median=statistics.median(gpu),
            Gp_faster_than_G0_wins=sum(row["Gp_faster_than_G0"] for row in chosen),
            faster_cpu_over_Gp=cpu, faster_cpu_over_Gp_median=statistics.median(cpu),
            Gp_faster_than_faster_cpu_wins=sum(row["Gp_faster_than_faster_cpu"] for row in chosen),
            scratch_no_sustained_growth=all(next(row for row in rows
                if row["plant"] == plant and row["block"] == block and row["route"] == "Gp")
                ["scratch_no_growth_final_quarter"] for block in range(5)),
            scratch_capacity_bytes_max=max(next(row for row in rows
                if row["plant"] == plant and row["block"] == block and row["route"] == "Gp")
                ["scratch_final_device_capacity_bytes"] for block in range(5)),
            peak_gpu_allocated_bytes_max=max(next(row for row in rows
                if row["plant"] == plant and row["block"] == block and row["route"] == "Gp")
                ["peak_gpu_allocated_bytes"] for block in range(5)),
            packet_target_met=statistics.median(gpu) >= 1.15
                and sum(row["Gp_faster_than_G0"] for row in chosen) >= 4,
            stable_cpu_regression=statistics.median(cpu) < 1
                and sum(row["Gp_faster_than_faster_cpu"] for row in chosen) <= 2)
    engineering_target = all(value["packet_target_met"] and not value["stable_cpu_regression"]
                             and value["scratch_no_sustained_growth"]
                             for value in decisions.values())
    gpu_medians = [value["G0_over_Gp_median"] for value in decisions.values()]
    if engineering_target:
        effect = "useful"
    elif any(value["stable_cpu_regression"] for value in decisions.values()):
        effect = "stable_cpu_regression"
    elif any(not value["scratch_no_sustained_growth"] for value in decisions.values()):
        effect = "continued_scratch_growth"
    elif all(value >= 1.15 for value in gpu_medians):
        effect = "unstable_boundary"
    elif all(value >= 1.05 for value in gpu_medians):
        effect = "near_threshold"
    elif any(value < 1 for value in gpu_medians):
        effect = "regression"
    else:
        effect = "no_gain"
    result = dict(schema="live-gpu-packet-performance-result-v1",
        packet_correctness="pass", packet_runtime_effect=effect,
        engineering_target_met=engineering_target, decisions=decisions,
        formal_blocks=5, formal_runs=len(rows), each_run_successful_lane_steps=640,
        source_sha=plan["source_sha"], default_enabled=False,
        full_gpu_engine=False, whole_solver_formal_proof=False)
    save(artifact / "PERFORMANCE_RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("diagnostic", "formal"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--no-recompute", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = (diagnostic_gate(args.path, recompute=not args.no_recompute)
              if args.phase == "diagnostic" else formal_summary(args.path))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
