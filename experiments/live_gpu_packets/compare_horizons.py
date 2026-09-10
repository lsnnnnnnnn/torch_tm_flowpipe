"""Re-observe reused CPU/Flow* objects and compare both fresh GPU-range horizons."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import gzip
import json
import math
from pathlib import Path
import statistics

from experiments.range_batch_device.common import save, sha
from experiments.live_range_solver.runner import ROOT, core_range_cpu
from torch_tm_flowpipe import Interval, Polynomial, TaylorModel, TMVector
from torch_tm_flowpipe.packed_boundary_range import packed_boundary_execution

from .analyze import csv_rows, percentile
from .verify_long_horizon import verify_long_horizon


CPU_OBJECTS = {
    "van_der_pol": ROOT / "artifacts/runs/endpoint_roundoff_repair_20260908/raw_minimal/vdp_full/models.jsonl.gz",
    "brusselator": ROOT / "artifacts/runs/endpoint_roundoff_repair_20260908/raw_minimal/brusselator_full/models.jsonl.gz",
}
FLOWSTAR_OBJECTS = {
    "van_der_pol": ROOT / "artifacts/runs/xiangru_adoption_20260907T032448Z/raw_minimal/native_vdp/models.jsonl.gz",
    "brusselator": ROOT / "artifacts/runs/xiangru_adoption_20260907T032448Z/raw_minimal/native_brusselator/models.jsonl.gz",
}
ORDERS = {"van_der_pol": 4, "brusselator": 6}
STEPS = {"van_der_pol": .01, "brusselator": .02}


def number(value):
    if isinstance(value, str):
        lowered = value.strip().lower()
        value = float.fromhex(value) if "p" in lowered or "0x" in lowered else float(value)
    else:
        value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite saved comparison object")
    return value


def decode_model(model, order):
    domain = [Interval(number(lo), number(hi)) for lo, hi in model["domain"]]
    output = []
    for component in model["components"]:
        terms = {}
        for term in component["terms"]:
            lo, hi = map(number, term["coefficient"])
            if lo != hi:
                raise ValueError("the reused polynomial object has interval coefficients")
            terms[tuple(term["degrees"])] = lo
        output.append(TaylorModel(Polynomial(terms, len(domain)),
            Interval(*map(number, component["remainder"])), domain, order=order))
    return TMVector(output)


def observe_objects(models, order):
    result = {}
    with packed_boundary_execution(True), core_range_cpu():
        for view in ("endpoint", "tube"):
            result[view] = [[float(interval.lo), float(interval.hi)]
                            for interval in decode_model(models[view], order).range_box()]
    return result


def metric_row(plant, step, exact_time, view, coordinate, gpu, reference,
               reference_name, reference_origin):
    gpu_lo, gpu_hi = gpu
    ref_lo, ref_hi = reference
    gpu_width, ref_width = gpu_hi - gpu_lo, ref_hi - ref_lo
    assert gpu_width >= 0 and ref_width >= 0
    near_zero = ref_width <= 1e-10
    ratio = None if near_zero else gpu_width / ref_width
    return dict(plant=plant, step=step, exact_time=exact_time, view=view,
        coordinate=coordinate, reference=reference_name,
        reference_data_origin=reference_origin,
        observer="CURRENT_STRICT_CORRECTED_CPU_RANGE_ON_COMPLETE_SAVED_OBJECT",
        gpu_lo_hex=gpu_lo.hex(), gpu_hi_hex=gpu_hi.hex(),
        reference_lo_hex=ref_lo.hex(), reference_hi_hex=ref_hi.hex(),
        gpu_width=gpu_width, reference_width=ref_width, width_ratio=ratio,
        lower_abs_diff=abs(gpu_lo - ref_lo), upper_abs_diff=abs(gpu_hi - ref_hi),
        center_abs_diff=abs((gpu_lo + gpu_hi - ref_lo - ref_hi) / 2),
        near_zero_reference=near_zero, width_ratio_over_1p10=ratio is not None and ratio > 1.10)


def service_totals(rows):
    totals = Counter()
    for row in rows:
        service = row["service"]
        totals["solver_step_wall_s"] += row["solver_step_wall_s"]
        totals["observer_s"] += row["observer_s"]
        totals["evidence_prepare_s"] += row["evidence_prepare_s"]
        totals["checkpoint_export_s"] += row["checkpoint_export_s"]
        totals["service_wall_s"] += service["service_span_s"]
        totals["service_thread_cpu_s"] += service["service_thread_cpu_s"]
        totals["requests"] += service["requests"]
        totals["groups"] += service["groups"]
        for name, value in service["costs"].items():
            totals[name] += value
        for name, value in service["counts"].items():
            totals[name] += value
        for name, value in service["scheduler_costs_s"].items():
            totals[f"scheduler_{name}"] += value
    return totals


def compare_one(plant, horizon):
    horizon = Path(horizon)
    verification = verify_long_horizon(horizon, require_achieved=True)
    result = json.loads((horizon / "LONG_HORIZON_RESULT.json").read_text())
    metadata = json.loads((horizon / "RUN_METADATA.json").read_text())
    assert result["plant"] == plant and result["route"] == "Gp"
    assert result["requested_steps"] == 1000 and result["recorded_steps"] == 1000
    assert metadata["scope"] == "ORIGINAL_B1" and metadata["previous_answers_loaded"] is False
    h = STEPS[plant]
    elapsed = Fraction(0)
    comparisons, gpu_rows = [], []
    with gzip.open(horizon / "full_horizon_steps.jsonl.gz", "rt") as gpu_stream, \
            gzip.open(CPU_OBJECTS[plant], "rt") as cpu_stream, \
            gzip.open(FLOWSTAR_OBJECTS[plant], "rt") as flowstar_stream:
        for step, lines in enumerate(zip(gpu_stream, cpu_stream, flowstar_stream, strict=True), 1):
            gpu_row, cpu_row, flowstar_row = map(json.loads, lines)
            gpu_rows.append(gpu_row)
            assert gpu_row["step"] == cpu_row["step"] == flowstar_row["step"] == step
            assert gpu_row["h_hex"] == h.hex()
            assert float(flowstar_row["h"]) == h
            assert flowstar_row["state_dimension"] == 2
            assert flowstar_row["safety_check_passed"] is True
            elapsed += Fraction(h)
            assert Fraction(gpu_row["exact_time"]) == elapsed
            assert cpu_row["t_end_exact"] == str(elapsed)
            cpu = observe_objects(cpu_row["models"], ORDERS[plant])
            flowstar = observe_objects(flowstar_row["models"], ORDERS[plant])
            for view in ("endpoint", "tube"):
                for coordinate, gpu_pair, cpu_pair, flowstar_pair in zip(
                        ("x", "y"), gpu_row["bounds"][view], cpu[view], flowstar[view]):
                    gpu_pair = list(map(float.fromhex, gpu_pair))
                    comparisons.append(metric_row(plant, step, str(elapsed), view,
                        coordinate, gpu_pair, cpu_pair, "strict_cpu",
                        "REUSED_FRESH_ENDPOINT_REPAIRED_COMPLETE_OBJECT"))
                    comparisons.append(metric_row(plant, step, str(elapsed), view,
                        coordinate, gpu_pair, flowstar_pair, "stock_flowstar",
                        "REUSED_MATCHED_REFERENCE_COMPLETE_OBJECT"))
    assert len(gpu_rows) == 1000 and len(comparisons) == 8000
    totals = service_totals(gpu_rows)
    matrix = dict(plant=plant, route="Gp", scope="ORIGINAL_UNPARTITIONED_B1",
        requested_steps=1000, accepted_steps=verification["steps"], achieved=verification["achieved"],
        exact_final_time=verification["exact_final_time"], nominal_target="T10" if plant == "van_der_pol" else "T20",
        invocation_s=metadata["invocation_s"], solver_step_wall_s=totals["solver_step_wall_s"],
        service_wall_s=totals["service_wall_s"], service_thread_cpu_s=totals["service_thread_cpu_s"],
        observer_s=totals["observer_s"], evidence_prepare_s=totals["evidence_prepare_s"],
        gzip_write_s=metadata["export_totals_s"]["json_gzip_write_s"],
        checkpoint_export_s=totals["checkpoint_export_s"], requests=int(totals["requests"]),
        dispatches=int(totals["groups"]), packets=int(totals["packet_count"]),
        actual_kernel_invocations=int(totals["actual_kernel_invocations"]),
        h2d_copy_operations=int(totals["h2d_copy_operations"]),
        d2h_copy_operations=int(totals["d2h_copy_operations"]),
        final_state_sha256=result["final_state_sha256"], previous_answers_loaded=False,
        cpu_periodic_correction=False, full_gpu_engine=False, whole_solver_formal_proof=False)
    return comparisons, matrix


def summarize(comparisons):
    output = []
    keys = sorted({(row["plant"], row["reference"], row["view"], row["coordinate"])
                   for row in comparisons})
    for plant, reference, view, coordinate in keys:
        chosen = [row for row in comparisons if (row["plant"], row["reference"],
                  row["view"], row["coordinate"]) == (plant, reference, view, coordinate)]
        ratios = [row["width_ratio"] for row in chosen if row["width_ratio"] is not None]
        assert ratios
        gpu_widths = [row["gpu_width"] for row in chosen]
        reference_widths = [row["reference_width"] for row in chosen]
        lower_diffs = [row["lower_abs_diff"] for row in chosen]
        upper_diffs = [row["upper_abs_diff"] for row in chosen]
        center_diffs = [row["center_abs_diff"] for row in chosen]
        worst = max(chosen, key=lambda row: row["width_ratio"]
                    if row["width_ratio"] is not None else -math.inf)
        lower_worst = max(chosen, key=lambda row: row["lower_abs_diff"])
        upper_worst = max(chosen, key=lambda row: row["upper_abs_diff"])
        output.append(dict(plant=plant, reference=reference, view=view, coordinate=coordinate,
            samples=len(chosen), near_zero_references=sum(row["near_zero_reference"] for row in chosen),
            ratio_over_1p10=sum(row["width_ratio_over_1p10"] for row in chosen),
            gpu_width_p50=statistics.median(gpu_widths),
            gpu_width_p95=percentile(gpu_widths, .95), gpu_width_max=max(gpu_widths),
            reference_width_p50=statistics.median(reference_widths),
            reference_width_p95=percentile(reference_widths, .95),
            reference_width_max=max(reference_widths),
            width_ratio_p50=statistics.median(ratios), width_ratio_p95=percentile(ratios, .95),
            width_ratio_max=max(ratios), worst_step=worst["step"], worst_exact_time=worst["exact_time"],
            worst_gpu_lo_hex=worst["gpu_lo_hex"], worst_gpu_hi_hex=worst["gpu_hi_hex"],
            worst_reference_lo_hex=worst["reference_lo_hex"],
            worst_reference_hi_hex=worst["reference_hi_hex"],
            worst_lower_abs_diff=worst["lower_abs_diff"],
            worst_upper_abs_diff=worst["upper_abs_diff"],
            lower_abs_diff_p50=statistics.median(lower_diffs),
            lower_abs_diff_p95=percentile(lower_diffs, .95),
            lower_abs_diff_max=max(lower_diffs),
            lower_abs_diff_max_step=lower_worst["step"],
            lower_abs_diff_max_exact_time=lower_worst["exact_time"],
            upper_abs_diff_p50=statistics.median(upper_diffs),
            upper_abs_diff_p95=percentile(upper_diffs, .95),
            upper_abs_diff_max=max(upper_diffs),
            upper_abs_diff_max_step=upper_worst["step"],
            upper_abs_diff_max_exact_time=upper_worst["exact_time"],
            center_abs_diff_p50=statistics.median(center_diffs),
            center_abs_diff_p95=percentile(center_diffs, .95),
            center_abs_diff_max=max(center_diffs)))
    return output


def compare_horizons(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    comparisons, matrix = [], []
    for plant in ("van_der_pol", "brusselator"):
        rows, item = compare_one(plant, root / f"{plant}-Gp")
        comparisons.extend(rows)
        matrix.append(item)
    summary = summarize(comparisons)
    csv_rows(output / "full_width_comparison.csv", comparisons)
    csv_rows(output / "full_width_summary.csv", summary)
    csv_rows(output / "full_horizon_matrix.csv", matrix)
    alerts = [row for row in comparisons if row["width_ratio_over_1p10"]]
    csv_rows(output / "full_width_over_1p10.csv", alerts)
    near_zero = [row for row in comparisons if row["near_zero_reference"]]
    csv_rows(output / "full_width_near_zero.csv", near_zero)
    result = dict(schema="live-gpu-packet-full-horizon-comparison-v1",
        complete_horizon_vdp="achieved" if matrix[0]["achieved"] else "not_achieved",
        complete_horizon_brusselator="achieved" if matrix[1]["achieved"] else "not_achieved",
        width_relation="measured", comparison_rows=len(comparisons),
        width_ratio_over_1p10=len(alerts), near_zero_reference_rows=len(near_zero),
        current_common_observer_reexecuted=True,
        gpu_runs_data_origin="FRESH_ORIGINAL_UNPARTITIONED_B1",
        cpu_data_origin="REUSED_FRESH_ENDPOINT_REPAIRED_COMPLETE_OBJECT",
        flowstar_data_origin="REUSED_MATCHED_REFERENCE_COMPLETE_OBJECT",
        reference_sources={str(path.relative_to(ROOT)): sha(path) for path in
                           (*CPU_OBJECTS.values(), *FLOWSTAR_OBJECTS.values())},
        full_gpu_engine=False, whole_solver_formal_proof=False)
    save(output / "FULL_HORIZON_COMPARISON_RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_horizons(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
