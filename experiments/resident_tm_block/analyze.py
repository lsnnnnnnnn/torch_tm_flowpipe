"""Verify resident-block campaigns and derive correctness/performance evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any

from experiments.live_range_solver.analyze import widths
from experiments.live_range_solver.verify import load_run, verify_run
from experiments.range_batch_device.common import read, save, sha

from .campaign import (
    PARENT_CPU_DIAGNOSTIC,
    PAIR_ORDERS,
    diagnostic_cases,
    expected_ids,
    formal_cases,
)


def csv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def aggregate_run(run: dict) -> dict:
    groups = run["groups"]
    resident_groups = [g for g in groups if g.get("operation") == "resident_tm_block"]
    range_groups = [g for g in groups if g.get("operation") != "resident_tm_block"]

    def resident_sum(name: str) -> float:
        return sum(float(group["timing"].get(name, 0)) for group in resident_groups)

    def resident_int(name: str) -> int:
        return sum(int(group["timing"].get(name, 0)) for group in resident_groups)

    def range_sum(name: str) -> float:
        return sum(float(group["timing"].get(name, 0)) for group in range_groups)

    def range_int(name: str) -> int:
        return sum(int(group["timing"].get(name, 0)) for group in range_groups)

    resident_sizes = [group["size"] for group in resident_groups]
    range_sizes = [group["size"] for group in range_groups]
    host = run.get("resident_host_costs", {})
    return dict(
        run_id=run["run_id"], plant=run["plant"], route=run["route"],
        batch=len(run["ids"]), steps=run["steps"],
        successful_lane_steps=run["successful_lane_steps"], wall_s=run["wall_s"],
        cpu_s=run["cpu_s"], throughput=run["throughput"],
        initial_state_s=run["initial_state_s"],
        total_service_groups=len(groups), range_service_groups=len(range_groups),
        resident_groups=len(resident_groups),
        total_requests=run["counts"].get("submitted", 0),
        range_requests=run["counts"].get("submitted", 0)
        - run["counts"].get("resident_requests", 0),
        resident_requests=run["counts"].get("resident_requests", 0),
        resident_lane_outputs=run["counts"].get("resident_lane_outputs", 0),
        resident_structure_fallbacks=run["counts"].get("resident_structure_fallbacks", 0),
        hardware_fallback_requests=run["counts"].get("hardware_fallback_requests", 0),
        mean_range_group=(statistics.mean(range_sizes) if range_sizes else None),
        max_range_group=max(range_sizes, default=0),
        mean_resident_group=(statistics.mean(resident_sizes) if resident_sizes else None),
        max_resident_group=max(resident_sizes, default=0),
        resident_block_host_span_sum_s=resident_sum("block_host_span_s"),
        resident_request_packing_s=resident_sum("packing_s"),
        resident_transfer_and_sync_host_span_s=resident_sum("transfer_and_sync_host_span_s"),
        resident_h2d_device_s=resident_sum("h2d_cuda_event_s"),
        resident_kernel_device_s=resident_sum("kernel_cuda_event_s"),
        resident_post_d2h_checks_and_rebuild_s=resident_sum("post_d2h_checks_and_rebuild_s"),
        resident_numeric_h2d_bytes=resident_int("numeric_h2d_bytes"),
        resident_metadata_h2d_bytes=resident_int("metadata_h2d_bytes"),
        resident_numeric_d2h_bytes=resident_int("numeric_d2h_bytes"),
        resident_metadata_d2h_bytes=resident_int("metadata_d2h_bytes"),
        resident_h2d_copy_operations=resident_int("h2d_copy_operations"),
        resident_d2h_copy_operations=resident_int("d2h_copy_operations"),
        resident_kernel_invocations=resident_int("kernel_invocations"),
        resident_host_synchronizations=resident_int("host_synchronizations"),
        resident_logical_device_local_intermediate_bytes_sum=resident_int(
            "logical_device_local_intermediate_bytes"
        ),
        request_build_calls=host.get("request_build_calls", 0),
        request_build_worker_wall_span_sum_s=host.get("request_build_wall_span_sum_ns", 0) / 1e9,
        request_build_worker_thread_cpu_sum_s=host.get("request_build_thread_cpu_sum_ns", 0) / 1e9,
        result_private_copy_worker_wall_span_sum_s=host.get(
            "result_caller_copy_wall_span_sum_ns", 0
        ) / 1e9,
        result_private_copy_worker_thread_cpu_sum_s=host.get(
            "result_caller_copy_thread_cpu_sum_ns", 0
        ) / 1e9,
        result_apply_worker_wall_span_sum_s=host.get(
            "result_apply_and_return_wall_span_sum_ns", 0
        ) / 1e9,
        range_packet_kernel_invocations=range_int("actual_kernel_invocations"),
        range_packet_h2d_copy_operations=range_int("h2d_copy_operations"),
        range_packet_d2h_copy_operations=range_int("d2h_copy_operations"),
        range_packet_kernel_device_s=range_sum("device_kernel_s"),
        service_mutually_exclusive_wall_s=sum(
            (group["end_ns"] - group["start_ns"]) / 1e9 for group in groups
        ),
        service_thread_cpu_sum_s=sum(group.get("thread_cpu_ns", 0) for group in groups) / 1e9,
        future_wait_worker_span_sum_s=sum(
            group.get("scheduler_costs", {}).get("future_wait_sum_ns", 0)
            for group in groups
        ) / 1e9,
        cold_startup_s=(run.get("cold_startup_outside_timing") or {}).get("wall_s", 0),
        peak_rss_kib=run["peak_rss_kib"],
        peak_gpu_allocated_bytes=run["peak_gpu_allocated_bytes"],
    )


def _walk_polynomials(value: Any, path: str = "") -> list[dict]:
    records: list[dict] = []
    if isinstance(value, dict):
        if value.get("type") == "Polynomial":
            items = value["fields"]["terms"]["items"]
            terms = []
            for exponent, coefficient in items:
                exponent_tuple = tuple(int(item) for item in exponent)
                assert coefficient["tensor"] == "torch.float64" and coefficient["shape"] == []
                terms.append((exponent_tuple, coefficient["values"][0]))
            records.append(dict(path=path, terms=terms))
        for key, child in value.items():
            records.extend(_walk_polynomials(child, f"{path}.{key}" if path else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            records.extend(_walk_polynomials(child, f"{path}[{index}]"))
    return records


def _ledger_categories(segment: dict) -> list[str]:
    ledger = segment["fields"].get("validated_remainder_ledger")
    if ledger is None:
        return []
    entries = ledger["fields"]["entries"]["items"]
    return sorted(str(key) for key, _ in entries)


def _polynomial_comparison(left: dict, right: dict) -> tuple[bool, float, int]:
    a = _walk_polynomials(left)
    b = _walk_polynomials(right)
    if [row["path"] for row in a] != [row["path"] for row in b]:
        return False, math.inf, 0
    support_equal = True
    max_abs = 0.0
    compared = 0
    for first, second in zip(a, b):
        first_terms = dict(first["terms"])
        second_terms = dict(second["terms"])
        support_equal &= first_terms.keys() == second_terms.keys()
        for exponent in first_terms.keys() & second_terms.keys():
            max_abs = max(
                max_abs,
                abs(float.fromhex(first_terms[exponent]) - float.fromhex(second_terms[exponent])),
            )
            compared += 1
    return support_equal, max_abs, compared


def compare_online(reference: dict, candidate: dict, label: str) -> tuple[dict, list[dict]]:
    assert reference["plant"] == candidate["plant"]
    assert reference["ids"] == candidate["ids"]
    assert reference["steps"] == candidate["steps"]
    decision_mismatches: list[dict] = []
    support_mismatches: list[dict] = []
    ledger_mismatches: list[dict] = []
    polynomial_max_abs = 0.0
    polynomial_coefficients_compared = 0
    next_state_exact = 0
    segment_exact = 0
    for task in reference["records"]:
        left_rows = reference["records"][task]
        right_rows = candidate["records"][task]
        assert len(left_rows) == len(right_rows)
        for offset, (left, right) in enumerate(zip(left_rows, right_rows), 1):
            decision_fields = (
                "accepted", "status", "state_step", "h_hex", "step_rejections",
                "validation_attempts",
            )
            changed = [field for field in decision_fields if left[field] != right[field]]
            if changed:
                decision_mismatches.append(dict(task=task, offset=offset, fields=changed))
            if left["after"] == right["after"]:
                next_state_exact += 1
            if left["segment"] == right["segment"]:
                segment_exact += 1
            support_equal, coefficient_abs, compared = _polynomial_comparison(
                left["segment"], right["segment"]
            )
            polynomial_max_abs = max(polynomial_max_abs, coefficient_abs)
            polynomial_coefficients_compared += compared
            if not support_equal:
                support_mismatches.append(dict(task=task, offset=offset))
            left_categories = _ledger_categories(left["segment"])
            right_categories = _ledger_categories(right["segment"])
            if left_categories != right_categories:
                ledger_mismatches.append(
                    dict(task=task, offset=offset, reference=left_categories, candidate=right_categories)
                )
    width_rows = widths(reference, candidate, label)
    ratios = [row["gpu_cpu_width_ratio"] for row in width_rows if row["gpu_cpu_width_ratio"] is not None]
    worst = max(width_rows, key=lambda row: row["max_abs_diff"])
    ratio_worst = max(
        (row for row in width_rows if row["gpu_cpu_width_ratio"] is not None),
        key=lambda row: row["gpu_cpu_width_ratio"],
        default=None,
    )
    steps = sum(len(rows) for rows in reference["records"].values())
    result = dict(
        case=label, plant=reference["plant"], batch=len(reference["ids"]),
        steps_per_task=reference["steps"], complete_lane_steps=steps,
        reference_run=reference["run_id"], candidate_run=candidate["run_id"],
        decision_mismatches=decision_mismatches,
        ordered_support_mismatches=support_mismatches,
        ledger_category_mismatches=ledger_mismatches,
        polynomial_coefficients_compared=polynomial_coefficients_compared,
        polynomial_coefficient_max_abs_diff=polynomial_max_abs,
        complete_segment_bitwise_equal=segment_exact,
        complete_next_state_bitwise_equal=next_state_exact,
        expected_roundoff_repair_can_change_binary64_state=True,
        width_rows=len(width_rows), width_ratio_nonzero_count=len(ratios),
        width_ratio_max=max(ratios, default=None),
        width_ratio_p50=percentile(ratios, 0.5),
        width_over_1p10=sum(row["warning"] for row in width_rows),
        worst_absolute_location={
            key: worst[key]
            for key in (
                "task", "step", "time", "view", "variable", "cpu_lo_hex", "cpu_hi_hex",
                "gpu_lo_hex", "gpu_hi_hex", "max_abs_diff",
            )
        },
        worst_width_ratio_location=(
            {
                key: ratio_worst[key]
                for key in (
                    "task", "step", "time", "view", "variable", "cpu_width", "gpu_width",
                    "gpu_cpu_width_ratio",
                )
            }
            if ratio_worst is not None
            else None
        ),
    )
    return result, width_rows


def _check_run(run: dict, case: dict, plan: dict, diagnostic: bool) -> dict:
    assert run["source_sha"] == plan["source_sha"]
    assert run["plant"] == case["plant"] and run["route"] == case["route"]
    assert run["ids"] == expected_ids(case) and run["steps"] == case["steps"]
    assert run["diagnostic"] is diagnostic
    assert run["successful_lane_steps"] == len(run["ids"]) * case["steps"]
    assert run["counts"].get("hardware_fallback_requests", 0) == 0
    if run["route"] == "Gr":
        assert run["counts"].get("resident_requests", 0) == run["successful_lane_steps"]
        assert run["counts"].get("resident_structure_fallbacks", 0) == 0
    return verify_run(run, [], recompute=False) if not diagnostic else None


def diagnostic_gate(path: Path) -> dict:
    path = path.resolve()
    artifact = path.parent
    plan = read(path / "CAMPAIGN_PLAN.json")
    cases = diagnostic_cases()
    assert plan["phase"] == "diagnostic" and plan["cases"] == cases
    local = read(artifact / "local_exact_checks.json")
    assert local["passed"] is True
    runs: dict[str, dict] = {}
    verification = []
    device_rows = []
    for case in cases:
        run, events = load_run(path / case["name"])
        assert run["source_sha"] == plan["source_sha"]
        assert run["plant"] == case["plant"] and run["route"] == case["route"]
        assert run["ids"] == expected_ids(case) and run["steps"] == case["steps"]
        assert run["diagnostic"] is True
        assert run["successful_lane_steps"] == len(run["ids"]) * case["steps"]
        receipt = verify_run(run, events, recompute=True)
        verification.append(receipt)
        runs[case["name"]] = run
        device_rows.append(aggregate_run(run))
        save(path / "verification_progress.json", verification)

    comparisons = []
    same_device_widths = []
    parent_cpu_widths = []
    for plant in ("van_der_pol", "brusselator"):
        for label in (
            "small-b1", "small-b2", "prefix-b8", "prefix-b32", "continuous-b2",
            "history-reset",
        ):
            comparison, rows = compare_online(
                runs[f"{label}-{plant}-Gp"], runs[f"{label}-{plant}-Gr"], label
            )
            comparisons.append(comparison)
            same_device_widths.extend(rows)
        parent_labels = {
            "small-b1": "small-b1", "small-b2": "small-b2", "prefix-b8": "prefix-b8",
            "prefix-b32": "prefix-b32", "continuous-b2": "long-b2",
        }
        for label, parent_label in parent_labels.items():
            cpu, _ = load_run(PARENT_CPU_DIAGNOSTIC / f"{parent_label}-{plant}-S")
            resident = runs[f"{label}-{plant}-Gr"]
            count = min(cpu["steps"], resident["steps"])
            cpu_view = dict(cpu, records={task: rows[:count] for task, rows in cpu["records"].items()})
            resident_view = dict(
                resident,
                records={task: rows[:count] for task, rows in resident["records"].items()},
            )
            parent_cpu_widths.extend(
                widths(cpu_view, resident_view, f"REUSED_PARENT_CPU_{label}")
            )

    csv_rows(artifact / "online_widths_same_device.csv", same_device_widths)
    csv_rows(artifact / "online_widths_vs_reused_cpu.csv", parent_cpu_widths)
    warnings = [row for row in same_device_widths + parent_cpu_widths if row["warning"]]
    csv_rows(artifact / "online_widths_over_1p10.csv", warnings)
    online = dict(
        schema="resident-tm-block-online-state-comparison-v1",
        passed=(
            not warnings
            and all(not row["decision_mismatches"] for row in comparisons)
            and all(not row["ordered_support_mismatches"] for row in comparisons)
            and all(not row["ledger_category_mismatches"] for row in comparisons)
        ),
        source_sha=plan["source_sha"], comparisons=comparisons,
        same_device_width_rows=len(same_device_widths),
        reused_parent_cpu_width_rows=len(parent_cpu_widths), width_over_1p10=len(warnings),
        parent_cpu_source="REUSED_PARENT_DIAGNOSTIC_S__DEFAULT_MATH_UNCHANGED",
        history_source="REUSED_PARENT_FULL_STATE_CHECKPOINT__RESUMED_LOCAL_WINDOW",
        next_step_consumes_current_candidate_output=True,
        saved_answers_used_to_advance=False,
        entire_solver_formal_proof=False,
    )
    save(artifact / "online_state_comparison.json", online)
    csv_rows(artifact / "device_residency_and_calls.csv", device_rows)
    gate = dict(
        schema="resident-tm-block-correctness-gate-v1",
        passed=online["passed"], source_sha=plan["source_sha"],
        local_exact_contract=local["passed"], online_integration=online["passed"],
        online_runs=len(runs), complete_lane_steps=sum(
            run["successful_lane_steps"] for run in runs.values()
        ),
        same_device_comparisons=len(comparisons),
        width_over_1p10=len(warnings), resident_structure_fallbacks=sum(
            run["counts"].get("resident_structure_fallbacks", 0) for run in runs.values()
        ),
        hardware_fallbacks=sum(
            run["counts"].get("hardware_fallback_requests", 0) for run in runs.values()
        ),
        full_1000_step_runs=0, entire_solver_formal_proof=False, full_gpu_engine=False,
    )
    save(path / "CORRECTNESS_GATE.json", gate)
    return gate


def formal_summary(path: Path) -> dict:
    path = path.resolve()
    artifact = path.parent
    plan = read(path / "CAMPAIGN_PLAN.json")
    cases = formal_cases()
    assert plan["phase"] == "formal" and plan["cases"] == cases
    gate = read(artifact / "diagnostic/CORRECTNESS_GATE.json")
    assert gate["passed"] and gate["source_sha"] == plan["source_sha"]
    rows = []
    for case in cases:
        run, events = load_run(path / case["name"])
        assert run["source_sha"] == plan["source_sha"] and not run["diagnostic"]
        assert run["plant"] == case["plant"] and run["route"] == case["route"]
        assert run["ids"] == expected_ids(case) and run["steps"] == 20
        assert run["successful_lane_steps"] == 640
        verify_run(run, events, recompute=False)
        row = aggregate_run(run)
        row.update(
            pair=case.get("pair"), order_position=case.get("order_position"),
            cpu_repetition=case.get("cpu_repetition"),
        )
        rows.append(row)
    csv_rows(artifact / "timings_raw.csv", rows)
    csv_rows(artifact / "device_residency_and_calls.csv", rows)

    paired = []
    decisions = {}
    for plant in ("van_der_pol", "brusselator"):
        q_rows = [row for row in rows if row["plant"] == plant and row["route"] == "Q"]
        s_row = next(row for row in rows if row["plant"] == plant and row["route"] == "S")
        q_median = statistics.median(row["wall_s"] for row in q_rows)
        for pair in range(5):
            selected = {
                route: next(
                    row for row in rows
                    if row["plant"] == plant and row["route"] == route and row["pair"] == pair
                )
                for route in ("Gp", "Gr")
            }
            assert selected["Gp"]["successful_lane_steps"] == selected["Gr"]["successful_lane_steps"] == 640
            paired.append(
                dict(
                    plant=plant, pair=pair, order="/".join(PAIR_ORDERS[pair]),
                    Gp_wall_s=selected["Gp"]["wall_s"], Gr_wall_s=selected["Gr"]["wall_s"],
                    Gp_over_Gr=selected["Gp"]["wall_s"] / selected["Gr"]["wall_s"],
                    Gr_faster_than_Gp=selected["Gr"]["wall_s"] < selected["Gp"]["wall_s"],
                    Q_median_wall_s=q_median, S_single_wall_s=s_row["wall_s"],
                    Q_median_over_Gr=q_median / selected["Gr"]["wall_s"],
                    S_single_over_Gr=s_row["wall_s"] / selected["Gr"]["wall_s"],
                    successful_lane_steps=640,
                )
            )
        chosen = [row for row in paired if row["plant"] == plant]
        speeds = [row["Gp_over_Gr"] for row in chosen]
        q_ratios = [row["Q_median_over_Gr"] for row in chosen]
        decisions[plant] = dict(
            Gp_over_Gr=speeds, median_Gp_over_Gr=statistics.median(speeds),
            Gr_faster_than_Gp_wins=sum(row["Gr_faster_than_Gp"] for row in chosen),
            end_to_end_target_met=(
                statistics.median(speeds) >= 1.20
                and sum(row["Gr_faster_than_Gp"] for row in chosen) >= 4
            ),
            Q_wall_s=[row["wall_s"] for row in q_rows], Q_median_wall_s=q_median,
            S_single_wall_s=s_row["wall_s"],
            median_Q_over_Gr=statistics.median(q_ratios),
            stable_strict_cpu_regression=(statistics.median(q_ratios) < 1.0),
        )
    csv_rows(artifact / "paired_speedups.csv", paired)
    medians = [value["median_Gp_over_Gr"] for value in decisions.values()]
    if all(value["end_to_end_target_met"] for value in decisions.values()):
        effect = "useful"
    elif any(value["stable_strict_cpu_regression"] for value in decisions.values()):
        effect = "regression"
    elif any(value < 1 for value in medians):
        effect = "no_gain"
    else:
        effect = "small_gain"
    result = dict(
        schema="resident-tm-block-performance-result-v1", source_sha=plan["source_sha"],
        formal_pairs_per_plant=5, formal_gpu_runs=20, cpu_Q_runs_per_plant=3,
        cpu_S_runs_per_plant=1, successful_lane_steps_per_run=640,
        decisions=decisions, end_to_end_effect=effect,
        engineering_target_met=all(
            value["end_to_end_target_met"] and not value["stable_strict_cpu_regression"]
            for value in decisions.values()
        ),
        default_enabled=False, full_gpu_engine=False, entire_solver_formal_proof=False,
    )
    save(artifact / "PERFORMANCE_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("diagnostic", "formal"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    result = diagnostic_gate(args.path) if args.phase == "diagnostic" else formal_summary(args.path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
