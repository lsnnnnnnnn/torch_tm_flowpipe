#!/usr/bin/env python3
"""Run genuine native low-order and practical-capability configurations."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common import analytic_contained, load_spec, write_csv, write_json
from run_controlled import _load_module

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _box(intervals: Sequence[Any]) -> list[list[float]]:
    return [
        [
            float(interval.lo.detach().cpu()),
            float(interval.hi.detach().cpu()),
        ]
        for interval in intervals
    ]


def _torch_configs(smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return [
            {"name": "order1_raw_dependency", "order": 1, "mode": "dependency_raw"},
            {"name": "order1_legacy_tightened", "order": 1, "mode": "dependency_tightened"},
            {"name": "order2_affine_reset", "order": 2, "mode": "affine_reset"},
        ]
    return [
        {"name": "order1_raw_dependency", "order": 1, "mode": "dependency_raw"},
        {"name": "order1_legacy_tightened", "order": 1, "mode": "dependency_tightened"},
        {"name": "order1_range_only", "order": 1, "mode": "range_only"},
        {"name": "order2_raw_dependency", "order": 2, "mode": "dependency_raw"},
        {"name": "order2_affine_reset", "order": 2, "mode": "affine_reset"},
        {"name": "order4_affine_reset", "order": 4, "mode": "affine_reset"},
        {"name": "order4_qr_reset", "order": 4, "mode": "qr_reset"},
        {"name": "order6_affine_reset", "order": 6, "mode": "affine_reset"},
    ]


def run_torch(spec: dict[str, Any], output: Path, *, smoke: bool) -> dict[str, Any]:
    src = REPO_ROOT / "src"
    followup = HERE.parent / "first_order_followup"
    for candidate in (src, followup):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from export_torch_segment import rhs_from_spec
    from torch_basis import affine_reset, normalized_initial_tm
    from torch_tm_flowpipe import Interval, TMVector, flowpipe_step, flowpipe_step_from_tm

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for system_name, all_configs in spec["multi_step"].items():
        configuration = (
            spec["smoke"][system_name] if smoke else all_configs[0]
        )
        h = float(configuration["h"])
        horizon = float(configuration["horizon"])
        system = spec["systems"][system_name]
        rhs = rhs_from_spec(system)
        for candidate in _torch_configs(smoke):
            order = int(candidate["order"])
            mode = str(candidate["mode"])
            steps = round(horizon / h)
            if mode in {"affine_reset", "qr_reset"}:
                current = normalized_initial_tm(system["initial_box"], order=order)
            else:
                current = TMVector.identity(
                    [Interval(*bounds) for bounds in system["initial_box"]],
                    order=order,
                )
            current_box = [Interval(*bounds) for bounds in system["initial_box"]]
            timings: list[float] = []
            completed = 0
            analytic_violations = 0
            failure = ""
            for step in range(1, steps + 1):
                started = time.perf_counter()
                diagnostics: list[dict[str, Any]] = []
                if mode == "range_only":
                    segment = flowpipe_step(
                        rhs,
                        current_box,
                        h,
                        order,
                        diagnostics=diagnostics,
                        diagnostics_context={"native_mode": mode},
                    )
                else:
                    segment = flowpipe_step_from_tm(
                        rhs,
                        current,
                        h,
                        order,
                        diagnostics=diagnostics,
                        diagnostics_context={"native_mode": mode},
                    )
                elapsed = time.perf_counter() - started
                timings.append(elapsed)
                if segment.status != "validated" or segment.endpoint_raw_tm is None:
                    failure = segment.message or "native validation failed"
                    break
                raw_endpoint = segment.endpoint_raw_tm
                tightened = segment.endpoint_tightened_tm or raw_endpoint
                carry_endpoint = (
                    tightened if mode == "dependency_tightened" else raw_endpoint
                )
                raw_box = _box(raw_endpoint.range_box())
                exact = analytic_contained(
                    system_name, system["initial_box"], step * h, raw_box
                )
                if exact is False:
                    analytic_violations += 1
                    failure = "analytic raw endpoint containment failed"
                for interval_kind, vector in (
                    ("tube", segment.tm),
                    ("endpoint_raw", raw_endpoint),
                    ("endpoint_tightened_supplemental", tightened),
                ):
                    for state_index, bounds in enumerate(_box(vector.range_box())):
                        model = vector[state_index]
                        poly_box = _box(
                            [model.polynomial.evaluate_interval(model.domain)]
                        )[0]
                        remainder = _box([model.remainder])[0]
                        rows.append(
                            {
                                "tool": "torch_tm_flowpipe",
                                "variant": candidate["name"],
                                "protocol": (
                                    "native_low_order"
                                    if order == 1
                                    else "native_practical"
                                ),
                                "system": system_name,
                                "h": h,
                                "horizon": horizon,
                                "step_index": step,
                                "time": step * h,
                                "state_index": state_index,
                                "interval_kind": interval_kind,
                                "lower": bounds[0],
                                "upper": bounds[1],
                                "width": bounds[1] - bounds[0],
                                "polynomial_width": poly_box[1] - poly_box[0],
                                "remainder_width": remainder[1] - remainder[0],
                                "order": order,
                                "basis": f"complete_total_degree_{order}",
                                "carry": mode,
                                "native_validation_passed": True,
                                "analytic_reference_contained": (
                                    exact if interval_kind == "endpoint_raw" else ""
                                ),
                                "runtime_s": elapsed,
                                "validation_attempts": segment.validation_attempts,
                                "directed_rounding_or_mpfr": "torch_nextafter_outward",
                                "floating_point_enclosure_candidate": True,
                                "message": failure,
                            }
                        )
                if mode == "range_only":
                    current_box = [
                        interval.inflate(1e-12)
                        for interval in raw_endpoint.range_box()
                    ]
                elif mode == "affine_reset":
                    current, _ = affine_reset(carry_endpoint, method="box")
                elif mode == "qr_reset":
                    current, _ = affine_reset(carry_endpoint, method="qr")
                else:
                    current = carry_endpoint
                completed = step
                if exact is False:
                    break
            summaries.append(
                {
                    "tool": "torch_tm_flowpipe",
                    "variant": candidate["name"],
                    "protocol": (
                        "native_low_order"
                        if order == 1
                        else "native_practical"
                    ),
                    "system": system_name,
                    "h": h,
                    "horizon": horizon,
                    "order": order,
                    "basis": f"complete_total_degree_{order}",
                    "carry": mode,
                    "requested_steps": steps,
                    "completed_steps": completed,
                    "successful_horizon": completed * h,
                    "native_validation_passed": completed == steps,
                    "analytic_reference_violations": analytic_violations,
                    "first_call_time_s": timings[0] if timings else math.nan,
                    "steady_step_time_s": (
                        statistics.median(timings[1:] or timings)
                        if timings
                        else math.nan
                    ),
                    "total_runtime_s": sum(timings),
                    "message": failure,
                }
            )
    write_csv(output / "native_torch.csv", rows)
    write_json(output / "native_torch_summary.json", summaries)
    return {"rows": len(rows), "summaries": len(summaries)}


def _diffreach_configs(smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return [
            {"name": "affine_flag", "affine": True, "rounds": 5, "window": 100},
            {
                "name": "restricted_quasiquadratic",
                "affine": False,
                "rounds": 5,
                "window": 100,
            },
        ]
    return [
        {"name": "affine_flag", "affine": True, "rounds": 5, "window": 100},
        {
            "name": "restricted_quasiquadratic",
            "affine": False,
            "rounds": 5,
            "window": 100,
        },
        {
            "name": "quasi_window1_round1",
            "affine": False,
            "rounds": 1,
            "window": 1,
        },
        {
            "name": "quasi_window10_round3",
            "affine": False,
            "rounds": 3,
            "window": 10,
        },
    ]


def run_diffreach(
    spec: dict[str, Any], output: Path, *, smoke: bool
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from export_diffreach_segment import (
        OPTIONAL_IMPORT_SHIM_USED,
        _initial_carry,
        _rhs,
        reachability,
        dr_settings,
    )

    jax.config.update("jax_enable_x64", True)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for system_name, all_configs in spec["multi_step"].items():
        configuration = (
            spec["smoke"][system_name] if smoke else all_configs[0]
        )
        h = float(configuration["h"])
        horizon = float(configuration["horizon"])
        steps = round(horizon / h)
        system = spec["systems"][system_name]
        lower = jnp.asarray(
            [[bounds[0] for bounds in system["initial_box"]]], dtype=jnp.float64
        )
        upper = jnp.asarray(
            [[bounds[1] for bounds in system["initial_box"]]], dtype=jnp.float64
        )
        for candidate in _diffreach_configs(smoke):
            dr_settings.update_config(
                {
                    "TRUNCATE_TO_AFFINE": bool(candidate["affine"]),
                    "BOUND_TIME_STEP": True,
                    "DEBUG_LOG": False,
                }
            )
            core = reachability.CT_Dyn_Reach(
                rhs=_rhs(system),
                state_dim=len(system["state_names"]),
                nn_dyn=False,
                step_size=h,
                init_remainder=float(spec["diffreach"]["init_remainder"]),
                frr_rounds=int(candidate["rounds"]),
                frr_stop_ratio=float(spec["diffreach"]["frr_stop_ratio"]),
                sr_window_size=int(candidate["window"]),
            )
            # The upstream public ``verify`` helper hard-codes float32 in its
            # initial step box, affine parameterization, and symbolic state.
            # Calling it with the required float64 endpoints therefore gives
            # a mixed-dtype lax.scan carry.  Build exactly the same public
            # step_once scan with the upstream constructors' dtype arguments
            # made explicit; no propagation formula is changed.
            core.step_boxes = reachability._make_step_boxes(
                1,
                len(system["state_names"]),
                h,
                dtype=jnp.float64,
            )
            initial_carry = _initial_carry(
                system, min(int(candidate["window"]), steps)
            )
            initial_interval = initial_carry[0].eval_interval(
                core.step_boxes[0], core.step_boxes[1]
            )
            compiled = jax.jit(
                lambda carry: jax.lax.scan(
                    core.step_once, carry, None, length=steps
                )
            )
            started = time.perf_counter()
            result = compiled(initial_carry)
            result = jax.tree.map(
                lambda value: value.block_until_ready()
                if hasattr(value, "block_until_ready")
                else value,
                result,
            )
            first_call = time.perf_counter() - started
            started = time.perf_counter()
            result = jax.tree.map(
                lambda value: value.block_until_ready()
                if hasattr(value, "block_until_ready")
                else value,
                compiled(initial_carry),
            )
            steady_call = time.perf_counter() - started
            (final_carry, scan_output) = result
            final_tm = final_carry[0]
            los, his, contraction = scan_output
            times_np = np.arange(steps + 1, dtype=np.float64) * h
            lowers_np = np.concatenate(
                [
                    np.asarray(initial_interval.lo)[:, None, :],
                    np.asarray(los).transpose((1, 0, 2)),
                ],
                axis=1,
            )[0]
            uppers_np = np.concatenate(
                [
                    np.asarray(initial_interval.hi)[:, None, :],
                    np.asarray(his).transpose((1, 0, 2)),
                ],
                axis=1,
            )[0]
            contraction_np = np.asarray(contraction)
            valid = np.all(
                contraction_np,
                axis=tuple(range(1, contraction_np.ndim)),
            )
            finite = np.all(np.isfinite(lowers_np[1:]), axis=1) & np.all(
                np.isfinite(uppers_np[1:]), axis=1
            )
            valid = valid & finite
            bad = np.flatnonzero(~valid)
            completed = int(bad[0]) if bad.size else steps
            analytic_violations = 0
            for step in range(1, completed + 1):
                endpoint = [
                    [float(lo), float(hi)]
                    for lo, hi in zip(lowers_np[step], uppers_np[step])
                ]
                exact = analytic_contained(
                    system_name,
                    system["initial_box"],
                    float(times_np[step]),
                    endpoint,
                )
                if exact is False:
                    analytic_violations += 1
                for state_index, bounds in enumerate(endpoint):
                    rows.append(
                        {
                            "tool": "diffreach",
                            "variant": candidate["name"],
                            "protocol": "native_low_order",
                            "system": system_name,
                            "h": h,
                            "horizon": horizon,
                            "step_index": step,
                            "time": float(times_np[step]),
                            "state_index": state_index,
                            "interval_kind": "endpoint_raw",
                            "lower": bounds[0],
                            "upper": bounds[1],
                            "width": bounds[1] - bounds[0],
                            "polynomial_width": "",
                            "remainder_width": "",
                            "order": (
                                "affine_flag"
                                if candidate["affine"]
                                else "restricted_quasiquadratic"
                            ),
                            "basis": (
                                "{1,tau,xi}"
                                if candidate["affine"]
                                else "{1,tau,xi,tau^2,tau*xi}"
                            ),
                            "carry": "upstream_symbolic_normalized",
                            "native_validation_passed": True,
                            "analytic_reference_contained": exact,
                            "runtime_s": steady_call / max(steps, 1),
                            "directed_rounding_or_mpfr": False,
                            "floating_point_enclosure_candidate": True,
                            "optional_jax_verify_shim_used": OPTIONAL_IMPORT_SHIM_USED,
                            "message": "",
                        }
                    )
            summaries.append(
                {
                    "tool": "diffreach",
                    "variant": candidate["name"],
                    "protocol": "native_low_order",
                    "system": system_name,
                    "h": h,
                    "horizon": horizon,
                    "affine_flag": candidate["affine"],
                    "basis": (
                        "{1,tau,xi}"
                        if candidate["affine"]
                        else "{1,tau,xi,tau^2,tau*xi}"
                    ),
                    "frr_rounds": candidate["rounds"],
                    "symbolic_window": candidate["window"],
                    "requested_steps": steps,
                    "completed_steps": completed,
                    "successful_horizon": completed * h,
                    "native_validation_passed": completed == steps,
                    "analytic_reference_violations": analytic_violations,
                    "jit_compile_and_first_call_s": first_call,
                    "after_jit_call_s": steady_call,
                    "steady_step_time_s": steady_call / max(steps, 1),
                    "final_nonzero_c": int(np.count_nonzero(np.asarray(final_tm.P.c))),
                    "final_nonzero_L": int(np.count_nonzero(np.asarray(final_tm.P.L))),
                    "final_nonzero_Lt": int(np.count_nonzero(np.asarray(final_tm.P.Lt))),
                }
            )
    write_csv(output / "native_diffreach.csv", rows)
    write_json(output / "native_diffreach_summary.json", summaries)
    return {"rows": len(rows), "summaries": len(summaries)}


def _load_flowstar_repair():
    repair = HERE.parent / "three_way_comparison_repair"
    previous = sys.modules.get("common")
    repair_common = _load_module("common", repair / "common.py")
    try:
        runner = _load_module(
            "_deep_study_native_flowstar",
            repair / "run_flowstar_audit.py",
        )
    finally:
        if previous is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = previous
    runner._repair_common = repair_common
    return runner


def run_flowstar(
    spec: dict[str, Any], output: Path, *, smoke: bool
) -> dict[str, Any]:
    runner = _load_flowstar_repair()
    os.environ["FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION"] = "1"
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    orders = [2] if smoke else list(map(int, spec["flowstar"]["orders"]))
    for system_name, all_configs in spec["multi_step"].items():
        configuration = (
            spec["smoke"][system_name] if smoke else all_configs[0]
        )
        h = float(configuration["h"])
        horizon = float(configuration["horizon"])
        for order in orders:
            case_rows, run, _ = runner.run_fixed_case(
                spec,
                output,
                system_name=system_name,
                protocol=runner.PROTOCOL_NATIVE,
                h=h,
                horizon=horizon,
                order=order,
                candidate=float(
                    spec["flowstar"]["candidate_remainder"][system_name]
                ),
                cutoff=float(spec["flowstar"]["cutoff"]),
                variant="flowstar_root_cause_patch",
            )
            for row in case_rows:
                if row["interval_kind"] == "failure":
                    continue
                rows.append(
                    {
                        "tool": "flowstar",
                        "variant": f"root_cause_fixed_order_{order}",
                        "protocol": (
                            "native_low_order"
                            if order == 2
                            else "native_practical"
                        ),
                        "system": system_name,
                        "h": h,
                        "horizon": horizon,
                        "step_index": row["step_index"],
                        "time": row["absolute_time"],
                        "state_index": row["state_index"],
                        "interval_kind": "endpoint_raw",
                        "lower": row["lower"],
                        "upper": row["upper"],
                        "width": row["width"],
                        "polynomial_width": row["polynomial_width"],
                        "remainder_width": row["remainder_width"],
                        "order": order,
                        "basis": f"complete_total_degree_{order}",
                        "carry": "native_Flowpipe_tmvPre_tmv_composition",
                        "native_validation_passed": True,
                        "analytic_reference_contained": (
                            row["analytic_reference_status"] == "passed"
                            if row["analytic_reference_status"] != "not_available"
                            else ""
                        ),
                        "runtime_s": row["steady_runtime_s"],
                        "directed_rounding_or_mpfr": True,
                        "floating_point_enclosure_candidate": False,
                        "message": "",
                    }
                )
            summaries.append(
                {
                    **run,
                    "variant": f"root_cause_fixed_order_{order}",
                    "protocol": (
                        "native_low_order"
                        if order == 2
                        else "native_practical"
                    ),
                    "basis": f"complete_total_degree_{order}",
                    "successful_horizon": run["completed_steps"] * h,
                    "native_validation_passed": (
                        run["status"] == "success"
                        and run["completed_steps"] == run["requested_steps"]
                    ),
                    "analytic_reference_violations": sum(
                        row.get("analytic_reference_status") == "failed"
                        for row in case_rows
                    ),
                    "symbolic_remainder": False,
                    "adaptive_step": False,
                    "preconditioning": "native_normalized_composition",
                }
            )

    # The original adaptive order-4/symbolic-window-100 Van der Pol
    # configuration is a separate native capability row.
    from flowstar_correctness import run_original_parity_gate

    parity = run_original_parity_gate(
        spec, output / "flowstar_native_original_parity"
    )
    root_log = Path(parity["root_cause_log"])
    parity_row = runner._parse_parity_rows(
        root_log.read_text(encoding="utf-8")
    )
    for item in parity_row:
        rows.append(
            {
                "tool": "flowstar",
                "variant": "adaptive_order4_symbolic100",
                "protocol": "native_practical",
                "system": "van_der_pol",
                "h": "adaptive_0.002_to_0.1",
                "horizon": 10.0,
                "step_index": item["step"],
                "time": item["time"],
                "state_index": item["state"],
                "interval_kind": "endpoint_raw",
                "lower": item["elo"],
                "upper": item["ehi"],
                "width": item["ehi"] - item["elo"],
                "polynomial_width": item["poly"],
                "remainder_width": item["rem"],
                "order": 4,
                "basis": "complete_total_degree_4",
                "carry": "native_symbolic_remainder_window_100",
                "native_validation_passed": True,
                "analytic_reference_contained": "",
                "runtime_s": "",
                "directed_rounding_or_mpfr": True,
                "floating_point_enclosure_candidate": False,
                "message": "",
            }
        )
    summaries.append(
        {
            "tool": "flowstar",
            "variant": "adaptive_order4_symbolic100",
            "protocol": "native_practical",
            "system": "van_der_pol",
            "h": "adaptive_0.002_to_0.1",
            "horizon": 10.0,
            "order": 4,
            "basis": "complete_total_degree_4",
            "requested_steps": "adaptive",
            "completed_steps": parity["root_cause_segments"],
            "successful_horizon": 10.0,
            "native_validation_passed": parity[
                "root_cause_variant_reached_horizon_10"
            ],
            "symbolic_remainder": True,
            "symbolic_window": 100,
            "adaptive_step": True,
            "preconditioning": (
                "native_normalized_composition; public QR off/on toggle "
                "not exposed in this Flow* checkout"
            ),
            "qr_toggle_status": "unavailable_public_api",
            "total_runtime_s": parity["runtimes_s"].get(
                "generated_identical", ""
            ),
        }
    )
    write_csv(output / "native_flowstar.csv", rows)
    write_json(output / "native_flowstar_summary.json", summaries)
    return {"rows": len(rows), "summaries": len(summaries)}


def collect(output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for tool in ("torch", "diffreach", "flowstar"):
        path = output / f"native_{tool}.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
        summary = output / f"native_{tool}_summary.json"
        if summary.exists():
            summaries.extend(json.loads(summary.read_text(encoding="utf-8")))
    write_csv(output / "native_raw.csv", rows)
    write_json(output / "native_summary.json", summaries)
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
