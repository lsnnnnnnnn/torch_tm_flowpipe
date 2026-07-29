#!/usr/bin/env python3
"""Run one-step, common-affine-carry, and common-box-carry protocols."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from common import (
    analytic_contained,
    evaluate_polynomial_interval,
    load_spec,
    write_csv,
    write_json,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _width(box: Sequence[float]) -> float:
    return float(box[1]) - float(box[0])


def _record_rows(
    record: Mapping[str, Any],
    *,
    protocol: str,
    step: int,
    time_value: float,
    initial_box: Sequence[Sequence[float]],
    runtime_s: float,
    carry: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exact = analytic_contained(
        str(record["system"]),
        initial_box,
        time_value,
        record["raw_endpoint_box"],
    )
    for kind, states, boxes in (
        ("tube", record["states"], record["whole_tube_box"]),
        ("endpoint_raw", record["raw_endpoint"], record["raw_endpoint_box"]),
    ):
        domains = (
            record["domains"]
            if kind == "tube"
            else record["generator_domains"]
        )
        for state_index, (state, box) in enumerate(zip(states, boxes)):
            poly = evaluate_polynomial_interval(
                state["polynomial_terms"], domains
            )
            remainder = state["independent_interval_remainder"]
            rows.append(
                {
                    "tool": record["tool"],
                    "variant": record["variant"],
                    "protocol": protocol,
                    "system": record["system"],
                    "h": record["h"],
                    "horizon": time_value,
                    "step_index": step,
                    "time": time_value,
                    "state_index": state_index,
                    "interval_kind": kind,
                    "lower": box[0],
                    "upper": box[1],
                    "width": _width(box),
                    "polynomial_width": _width(poly),
                    "remainder_width": _width(remainder),
                    "monomial_families": json.dumps(
                        state["monomial_families"], sort_keys=True
                    ),
                    "term_count": state["term_count"],
                    "max_degree": state["max_degree"],
                    "native_validation_passed": record[
                        "native_validation_passed"
                    ],
                    "analytic_reference_contained": (
                        exact if kind == "endpoint_raw" else ""
                    ),
                    "trajectory_sanity_passed": "",
                    "directed_rounding_or_mpfr": record["native_metadata"].get(
                        "directed_rounding_or_mpfr", ""
                    ),
                    "floating_point_enclosure_candidate": record[
                        "native_metadata"
                    ].get("floating_point_enclosure_candidate", ""),
                    "carry_contract": carry,
                    "runtime_s": runtime_s,
                    "compile_time_s": record["native_metadata"].get(
                        "compile_time_s", 0.0
                    ),
                    "validation_trace_count": len(record["validation_trace"]),
                    "failure_category": "",
                    "message": "",
                }
            )
    return rows


def _one_step_cases(spec: Mapping[str, Any], smoke: bool):
    for system_name, all_h in spec["one_step"].items():
        hs = [float(spec["smoke"][system_name]["h"])] if smoke else list(map(float, all_h))
        for h in hs:
            yield system_name, h


def _save_common_segment(output: Path, record: Mapping[str, Any]) -> None:
    variant = str(record["variant"]).replace("/", "_")
    requested_order = record.get("native_metadata", {}).get(
        "requested_order"
    )
    order_suffix = (
        f"_o{requested_order}" if requested_order not in (None, "") else ""
    )
    name = (
        f"{record['tool']}_{variant}_{record['system']}_"
        f"h{float(record['h']):g}{order_suffix}.json"
    )
    write_json(output / "common_segments" / name, record)


def run_torch(spec: dict[str, Any], output: Path, *, smoke: bool) -> dict[str, Any]:
    src_root = REPO_ROOT / "src"
    followup = HERE.parent / "first_order_followup"
    for candidate in (src_root, followup):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from export_torch_segment import export_segment
    from torch_basis import normalized_initial_tm, project_to_basis
    from torch_tm_flowpipe import flowpipe_step_from_tm
    from export_torch_segment import rhs_from_spec

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    orders = [1] if smoke else [1, 2]
    for system_name, h in _one_step_cases(spec, smoke):
        for order in orders:
            started = time.perf_counter()
            record = export_segment(
                spec, system_name=system_name, h=h, order=order
            )
            _save_common_segment(output, record)
            runtime = time.perf_counter() - started
            rows.extend(
                _record_rows(
                    record,
                    protocol="one_step_common_input",
                    step=1,
                    time_value=h,
                    initial_box=spec["systems"][system_name]["initial_box"],
                    runtime_s=runtime,
                    carry="none_one_step",
                )
            )
            summaries.append(
                {
                    "tool": "torch_tm_flowpipe",
                    "variant": record["variant"],
                    "protocol": "one_step_common_input",
                    "system": system_name,
                    "h": h,
                    "requested_steps": 1,
                    "completed_steps": int(record["native_validation_passed"]),
                    "successful_horizon": h
                    if record["native_validation_passed"]
                    else 0.0,
                    "runtime_s": runtime,
                }
            )
    for system_name, configurations in spec["multi_step"].items():
        configs = [spec["smoke"][system_name]] if smoke else configurations
        for configuration in configs:
            h = float(configuration["h"])
            horizon = float(configuration["horizon"])
            if smoke:
                horizon = float(spec["smoke"][system_name]["horizon"])
            for protocol in ("common_affine_carry", "common_box_carry"):
                system = spec["systems"][system_name]
                current = normalized_initial_tm(system["initial_box"], order=1)
                rhs = rhs_from_spec(system)
                steps = round(horizon / h)
                completed = 0
                timings: list[float] = []
                discarded_total = 0
                for step in range(1, steps + 1):
                    started = time.perf_counter()
                    diagnostics: list[dict[str, Any]] = []
                    segment = flowpipe_step_from_tm(
                        rhs,
                        current,
                        h,
                        1,
                        diagnostics=diagnostics,
                        diagnostics_context={"protocol": protocol},
                    )
                    elapsed = time.perf_counter() - started
                    timings.append(elapsed)
                    if (
                        segment.status != "validated"
                        or segment.endpoint_raw_tm is None
                    ):
                        break
                    endpoint = segment.endpoint_raw_tm
                    if protocol == "common_affine_carry":
                        projected, discarded = project_to_basis(
                            endpoint,
                            "B1",
                            tau_index=None,
                            stage="common_affine_endpoint_projection",
                            iteration=step,
                        )
                        current = projected
                        discarded_total += len(discarded)
                    else:
                        current = normalized_initial_tm(
                            [
                                interval.to_tuple()
                                for interval in endpoint.range_box()
                            ],
                            order=1,
                        )
                    # Reuse the canonical serializer without rebuilding a
                    # segment; its private helpers are deliberately read-only.
                    from export_torch_segment import _bounds, _state
                    from common import canonical_record

                    record = canonical_record(
                        tool="torch_tm_flowpipe",
                        variant="complete_total_degree_1",
                        system=system_name,
                        h=h,
                        variable_names=[
                            *[
                                f"xi_{index}"
                                for index in range(segment.tm.n_vars - 1)
                            ],
                            "tau",
                        ],
                        variable_roles=[
                            *["state_generator"] * (segment.tm.n_vars - 1),
                            "local_time",
                        ],
                        domains=[_bounds(domain) for domain in segment.tm.domain],
                        states=[_state(model) for model in segment.tm],
                        raw_endpoint=[_state(model) for model in endpoint],
                        raw_endpoint_box=[
                            _bounds(interval) for interval in endpoint.range_box()
                        ],
                        tube_box=[
                            _bounds(interval) for interval in segment.tm.range_box()
                        ],
                        validation_trace=diagnostics,
                        reset_metadata={
                            "reset": protocol,
                            "discarded_term_count": discarded_total,
                        },
                        native_metadata={
                            "status": segment.status,
                            "directed_rounding_or_mpfr": "torch_nextafter_outward",
                            "floating_point_enclosure_candidate": True,
                        },
                    )
                    record["native_validation_passed"] = True
                    rows.extend(
                        _record_rows(
                            record,
                            protocol=protocol,
                            step=step,
                            time_value=step * h,
                            initial_box=system["initial_box"],
                            runtime_s=elapsed,
                            carry=(
                                "c+A*xi+fresh_independent_interval"
                                if protocol == "common_affine_carry"
                                else "componentwise_box"
                            ),
                        )
                    )
                    completed = step
                summaries.append(
                    {
                        "tool": "torch_tm_flowpipe",
                        "variant": "complete_total_degree_1",
                        "protocol": protocol,
                        "system": system_name,
                        "h": h,
                        "horizon": horizon,
                        "requested_steps": steps,
                        "completed_steps": completed,
                        "successful_horizon": completed * h,
                        "first_call_time_s": timings[0] if timings else math.nan,
                        "steady_step_time_s": (
                            statistics.median(timings[1:] or timings)
                            if timings
                            else math.nan
                        ),
                        "discarded_term_count": discarded_total,
                    }
                )
    write_csv(output / "controlled_torch.csv", rows)
    write_json(output / "controlled_torch_summary.json", summaries)
    return {"rows": len(rows), "summaries": len(summaries)}


def run_diffreach(
    spec: dict[str, Any], output: Path, *, smoke: bool
) -> dict[str, Any]:
    from export_diffreach_segment import export_segment

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    variants = [True] if smoke else [True, False]
    for system_name, h in _one_step_cases(spec, smoke):
        for affine in variants:
            started = time.perf_counter()
            record = export_segment(
                spec, system_name=system_name, h=h, affine=affine
            )
            _save_common_segment(output, record)
            runtime = time.perf_counter() - started
            rows.extend(
                _record_rows(
                    record,
                    protocol="one_step_common_input",
                    step=1,
                    time_value=h,
                    initial_box=spec["systems"][system_name]["initial_box"],
                    runtime_s=runtime,
                    carry="none_one_step",
                )
            )
            summaries.append(
                {
                    "tool": "diffreach",
                    "variant": record["variant"],
                    "protocol": "one_step_common_input",
                    "system": system_name,
                    "h": h,
                    "requested_steps": 1,
                    "completed_steps": int(record["native_validation_passed"]),
                    "successful_horizon": h
                    if record["native_validation_passed"]
                    else 0.0,
                    "runtime_s": runtime,
                }
            )

    runner = _load_diffreach_followup()
    adapted = copy.deepcopy(spec)
    rounds = adapted["diffreach"]["frr_rounds"]
    windows = adapted["diffreach"]["symbolic_windows"]
    adapted["diffreach"]["frr_rounds"] = max(rounds) if isinstance(rounds, list) else rounds
    adapted["diffreach"]["symbolic_remainder_window"] = (
        max(windows) if isinstance(windows, list) else windows
    )
    for system_name, configurations in spec["multi_step"].items():
        configs = [spec["smoke"][system_name]] if smoke else configurations
        for configuration in configs:
            h = float(configuration["h"])
            horizon = float(configuration["horizon"])
            if smoke:
                horizon = float(spec["smoke"][system_name]["horizon"])
            affine_rows, affine_summary = runner.run_configuration(
                spec=adapted,
                system_name=system_name,
                h=h,
                horizon=horizon,
                protocol="matched_affine_carry",
            )
            for row in affine_rows:
                row["protocol"] = "common_affine_carry"
                row["interval_kind"] = (
                    "endpoint_raw"
                    if row["interval_kind"] == "endpoint"
                    else row["interval_kind"]
                )
                row["carry_contract"] = "c+A*xi+fresh_independent_interval"
            affine_summary["protocol"] = "common_affine_carry"
            rows.extend(affine_rows)
            summaries.append(affine_summary)

            # Common box carry is intentionally a reset control. Each step
            # starts a fresh upstream one-step solve from the previous raw box.
            box_spec = copy.deepcopy(spec)
            current_box = copy.deepcopy(spec["systems"][system_name]["initial_box"])
            steps = round(horizon / h)
            completed = 0
            timings: list[float] = []
            for step in range(1, steps + 1):
                box_spec["systems"][system_name]["initial_box"] = current_box
                started = time.perf_counter()
                record = export_segment(
                    box_spec,
                    system_name=system_name,
                    h=h,
                    affine=False,
                )
                elapsed = time.perf_counter() - started
                timings.append(elapsed)
                if not record["native_validation_passed"]:
                    break
                current_box = copy.deepcopy(record["raw_endpoint_box"])
                rows.extend(
                    _record_rows(
                        record,
                        protocol="common_box_carry",
                        step=step,
                        time_value=step * h,
                        initial_box=spec["systems"][system_name]["initial_box"],
                        runtime_s=elapsed,
                        carry="componentwise_box",
                    )
                )
                completed = step
            summaries.append(
                {
                    "tool": "diffreach",
                    "variant": "upstream_restricted_quasi_quadratic",
                    "protocol": "common_box_carry",
                    "system": system_name,
                    "h": h,
                    "horizon": horizon,
                    "requested_steps": steps,
                    "completed_steps": completed,
                    "successful_horizon": completed * h,
                    "first_call_time_s": timings[0] if timings else math.nan,
                    "steady_step_time_s": (
                        statistics.median(timings[1:] or timings)
                        if timings
                        else math.nan
                    ),
                }
            )
    write_csv(output / "controlled_diffreach.csv", rows)
    write_json(output / "controlled_diffreach_summary.json", summaries)
    return {"rows": len(rows), "summaries": len(summaries)}


def _load_module(name: str, path: Path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


def _load_diffreach_followup():
    followup = HERE.parent / "first_order_followup"
    baseline = HERE.parent / "first_order_three_way"
    diffreach = Path("/srv/local/shengenli/DiffReach")
    for path in (followup, baseline, diffreach):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    previous = sys.modules.get("common")
    baseline_common = _load_module("common", baseline / "common.py")
    try:
        module = _load_module(
            "_deep_study_diffreach_followup",
            followup / "run_diffreach_followup.py",
        )
    finally:
        if previous is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = previous
    module._baseline_common = baseline_common
    return module


def _load_flowstar_followup():
    followup = HERE.parent / "first_order_followup"
    baseline = HERE.parent / "first_order_three_way"
    for path in (followup, baseline):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    previous = sys.modules.get("common")
    baseline_common = _load_module("common", baseline / "common.py")
    try:
        module = _load_module(
            "_deep_study_flowstar_followup",
            followup / "run_flowstar_followup.py",
        )
    finally:
        if previous is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = previous
    module._baseline_common = baseline_common
    return module


def run_flowstar(
    spec: dict[str, Any], output: Path, *, smoke: bool
) -> dict[str, Any]:
    from export_flowstar_segment import export_segment

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    orders = [2] if smoke else [2, 4]
    for system_name, h in _one_step_cases(spec, smoke):
        for order in orders:
            started = time.perf_counter()
            record = export_segment(
                spec,
                system_name=system_name,
                h=h,
                order=order,
                variant=spec["flowstar"]["primary_variant"],
                work_dir=(
                    output
                    / "logs"
                    / "controlled_flowstar"
                    / f"one_step_{system_name}_h{h:g}_o{order}"
                ),
            )
            _save_common_segment(output, record)
            runtime = time.perf_counter() - started
            rows.extend(
                _record_rows(
                    record,
                    protocol="one_step_common_input",
                    step=1,
                    time_value=h,
                    initial_box=spec["systems"][system_name]["initial_box"],
                    runtime_s=runtime,
                    carry="none_one_step",
                )
            )
            summaries.append(
                {
                    "tool": "flowstar",
                    "variant": record["variant"],
                    "protocol": "one_step_common_input",
                    "system": system_name,
                    "h": h,
                    "order": order,
                    "requested_steps": 1,
                    "completed_steps": 1,
                    "successful_horizon": h,
                    "runtime_s": runtime,
                }
            )

    runner = _load_flowstar_followup()
    runner.FLOWSTAR_ROOT = Path(spec["repositories"]["flowstar_audit"])
    original_render = runner.render_cpp

    def corrected_render(*args: Any, box_carry: bool = False, **kwargs: Any) -> str:
        source = original_render(*args, **kwargs)
        mutation = """
    // Keep the configured remainder candidate that the first Picard inclusion
    // check proved self-mapping.  Do not use the toolbox's un-revalidated
    // refinement image.
    for(unsigned int state = 0; state < next.tmvPre.tms.size(); ++state) {
      next.tmvPre.tms[state].remainder =
          setting.tm_setting.remainder_estimation[state];
    }
"""
        source = source.replace(mutation, "")
        source = source.replace(
            "int main() {",
            'int main() {\n  setenv("FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION", "1", 1);',
        )
        if box_carry:
            source = source.replace(
                "    current = next;",
                "    current = Flowpipe(endpoint_box);",
            )
        return source

    adapted = {
        "systems": spec["systems"],
        "timeout_s": spec["timeout_s"],
        "flowstar": {
            "remainder_estimation": 1e-4,
            "cutoff": spec["flowstar"]["cutoff"],
        },
    }
    for system_name, configurations in spec["multi_step"].items():
        configs = [spec["smoke"][system_name]] if smoke else configurations
        for configuration in configs:
            h = float(configuration["h"])
            horizon = float(configuration["horizon"])
            if smoke:
                horizon = float(spec["smoke"][system_name]["horizon"])
            for protocol, box_carry in (
                ("common_affine_carry", False),
                ("common_box_carry", True),
            ):
                runner.render_cpp = (
                    lambda *args, _box=box_carry, **kwargs: corrected_render(
                        *args, box_carry=_box, **kwargs
                    )
                )
                native_protocol = (
                    "matched_affine_carry"
                    if protocol == "common_affine_carry"
                    else "complete_degree_two_reference"
                )
                case_rows, summary = runner.run_configuration(
                    spec=adapted,
                    system_name=system_name,
                    h=h,
                    horizon=horizon,
                    output=output / "logs" / "controlled_flowstar_carry",
                    protocol=native_protocol,
                )
                for row in case_rows:
                    row["variant"] = "flowstar_root_cause_patch"
                    row["protocol"] = protocol
                    row["validator"] = (
                        "Flowstar_cached_remainder_with_leaf_truncation_patch"
                    )
                    row["interval_kind"] = (
                        "endpoint_raw"
                        if row["interval_kind"] == "endpoint"
                        else row["interval_kind"]
                    )
                    row["carry_contract"] = (
                        "c+A*xi+fresh_independent_interval"
                        if protocol == "common_affine_carry"
                        else "componentwise_box"
                    )
                summary["variant"] = "flowstar_root_cause_patch"
                summary["protocol"] = protocol
                summary["validator"] = (
                    "Flowstar_cached_remainder_with_leaf_truncation_patch"
                )
                source_text = Path(summary["source"]).read_text(encoding="utf-8")
                forbidden_mutation = (
                    "next.tmvPre.tms[state].remainder ="
                    in source_text
                )
                summary["no_post_advance_remainder_mutation"] = (
                    not forbidden_mutation
                )
                summary["no_candidate_reinjection"] = not forbidden_mutation
                if forbidden_mutation:
                    raise RuntimeError(
                        f"forbidden Flow* remainder mutation in {summary['source']}"
                    )
                rows.extend(case_rows)
                summaries.append(summary)
    runner.render_cpp = original_render
    write_csv(output / "controlled_flowstar.csv", rows)
    write_json(output / "controlled_flowstar_summary.json", summaries)
    return {"rows": len(rows), "summaries": len(summaries)}


def collect(output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for tool in ("torch", "diffreach", "flowstar"):
        csv_path = output / f"controlled_{tool}.csv"
        if csv_path.exists():
            import csv

            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
        summary_path = output / f"controlled_{tool}_summary.json"
        if summary_path.exists():
            summaries.extend(json.loads(summary_path.read_text(encoding="utf-8")))
    write_csv(output / "controlled_raw.csv", rows)
    write_json(output / "controlled_summary.json", summaries)
    return {"rows": len(rows), "summaries": len(summaries)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tool", choices=["torch", "diffreach", "flowstar", "collect"], required=True
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.tool == "torch":
        result = run_torch(spec, output, smoke=args.smoke)
    elif args.tool == "diffreach":
        result = run_diffreach(spec, output, smoke=args.smoke)
    elif args.tool == "flowstar":
        result = run_flowstar(spec, output, smoke=args.smoke)
    else:
        result = collect(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
