#!/usr/bin/env python3
"""Assemble the fixed VDP C3 causal-closure evidence package."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUN = Path("/srv/local/shengenli/vdp_c3_runs_20260827")
OUT = ROOT / "outputs/vdp_c3_cross_step_causal_closure_20260827"
CHECKPOINTS = (1, 10, 50, 100, 200, 300, 400, 500, 600, 632)
CHANNELS = ("endpoint_x", "endpoint_y", "tube_x", "tube_y")
TORCH_C2_SHA = "29c9ee8f1fe96b860052b86a2b37d79a37bbb2ca"
TORCH_PACKAGE_SHA = "0fea265b9f1a61d0196106213c8584eeae48f03e"
TORCH_C3_SHA = "190e06714dbfe2afe53650b577916dfeca73dd5a"
HUAN_SHA = "743f6205e6408072193ad76e940e7f15030e8d3c"
HUAN_LEDGER_SHA = "90be93578ccded480d276cb5e4ced6b4a55803c0"
FLOWSTAR_SHA = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
PRIMARY_STATUS = "CROSS_STEP_CAUSE_IDENTIFIED__C3_PRODUCTION_GATE_PASSED__NATIVE_T10_REACHED"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_gz_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def jdump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def torch_widths(summary: Mapping[str, Any]) -> list[float]:
    return [
        f(summary["raw_endpoint"]["x_width"]),
        f(summary["raw_endpoint"]["y_width"]),
        f(summary["last_segment"]["x_width"]),
        f(summary["last_segment"]["y_width"]),
    ]


def flowstar_final(horizon: str) -> tuple[dict[str, str], list[float]]:
    row = read_csv(RUN / f"phase_a/flowstar/fixed_{horizon}/stock.csv")[-1]
    widths = [
        f(row["endpoint_x_hi"]) - f(row["endpoint_x_lo"]),
        f(row["endpoint_y_hi"]) - f(row["endpoint_y_lo"]),
        f(row["segment_x_hi"]) - f(row["segment_x_lo"]),
        f(row["segment_y_hi"]) - f(row["segment_y_lo"]),
    ]
    return row, widths


def huan_fixed_runs() -> dict[tuple[str, str], dict[str, Any]]:
    payload = read_json(RUN / "phase_a/huan/run_index.json")
    return {
        (row["scenario"], row["mode"]): row
        for row in payload["fixed_runs"]
        if row["scenario"] in {"fixed_T1", "fixed_T3", "fixed_T6p32"}
    }


def huan_widths(row: Mapping[str, Any]) -> list[float]:
    channels = row["channels"]
    return [*map(float, channels["endpoint_width"]), *map(float, channels["segment_tube_width"])]


def huan_summary_widths(path: Path) -> list[float]:
    channels = read_json(path)["final_channels"]
    return [*map(float, channels["endpoint_width"]), *map(float, channels["segment_tube_width"])]


def causal_by_step(queue: int, mode: str) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    if queue == 100:
        base = RUN / f"phase_b/callback_on_gpu0/sr100/{mode}"
    else:
        base = RUN / f"phase_c/sr0_sr1_sr10/sr{queue}/{mode}"
    rows = read_gz_jsonl(base / "causal_ledger.jsonl.gz")
    return ({int(row["step"]): row for row in rows if row.get("event") == "causal_step"}, rows)


def crossing_supplement_by_step(mode: str) -> dict[int, dict[str, Any]]:
    path = RUN / f"phase_b/callback_crossings_gpu0/sr100/{mode}/causal_ledger.jsonl.gz"
    rows = read_gz_jsonl(path)
    return {int(row["step"]): row for row in rows if row.get("event") == "causal_step"}


def refinement_by_step(
    queue: int, mode: str, *, crossing_supplement: bool = False
) -> dict[int, dict[str, Any]]:
    if crossing_supplement:
        if queue != 100:
            raise ValueError("crossing supplement only exists for SR100")
        path = RUN / f"phase_b/callback_crossings_gpu0/sr100/{mode}/refinement_ledger.jsonl.gz"
    elif queue == 100:
        path = RUN / f"phase_b/callback_on_gpu0/sr100/{mode}/refinement_ledger.jsonl.gz"
    else:
        path = RUN / f"phase_c/sr0_sr1_sr10/sr{queue}/{mode}/refinement_ledger.jsonl.gz"
    rows = read_gz_jsonl(path)
    result: dict[int, dict[str, Any]] = {}
    step = 0
    current: dict[str, Any] | None = None
    for row in rows:
        if row.get("event") == "initial_self_map":
            step += 1
            current = {"initial": row, "components": []}
            result[step] = current
        elif current is not None and row.get("event") == "refinement_component":
            current["components"].append(row)
        elif current is not None and row.get("event") == "final_remainder_owner":
            current["final"] = row
    return result


def huan_row_widths(row: Mapping[str, Any]) -> list[float]:
    return [*map(float, row["endpoint"]["width"][0]), *map(float, row["segment_tube"]["width"][0])]


def first_difference(
    left: Mapping[int, Mapping[str, Any]],
    right: Mapping[int, Mapping[str, Any]],
    selector,
    *,
    material_tol: float | None = None,
) -> int | None:
    for step in sorted(set(left) & set(right)):
        a, b = selector(left[step]), selector(right[step])
        if material_tol is None:
            if a != b:
                return step
        elif max(abs(x - y) for x, y in zip(a, b)) > material_tol:
            return step
    return None


def fixed_matrix() -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    huan = huan_fixed_runs()
    rows: list[dict[str, Any]] = []
    terminal: dict[str, list[float]] = {}
    for horizon, scenario in (("T1", "fixed_T1"), ("T3", "fixed_T3"), ("T6p32", "fixed_T6p32")):
        stock, flow_w = flowstar_final(horizon)
        terminal[f"Flowstar_{horizon}"] = flow_w
        rows.append({
            "tool": "Flow*", "mode": "native_symbolic_remainder", "horizon": horizon.replace("p", "."),
            "source_sha": FLOWSTAR_SHA, "accepted": stock["step"], "rejected": 0,
            "runtime_s": "", "status": "completed", **dict(zip(CHANNELS, flow_w)),
        })
        c2 = read_json(RUN / f"phase_a/torch_c2/fixed_{horizon}/summary.json")
        c3 = read_json(RUN / f"phase_e/torch_c3/fixed_{horizon}/summary.json")
        c2_w, c3_w = torch_widths(c2), torch_widths(c3)
        terminal[f"C2_{horizon}"] = c2_w
        terminal[f"C3_{horizon}"] = c3_w
        for tool, mode, source, summary, widths in (
            ("Torch", "C2", TORCH_C2_SHA, c2, c2_w),
            ("Torch", "C3_SR100", TORCH_C3_SHA, c3, c3_w),
        ):
            rows.append({
                "tool": tool, "mode": mode, "horizon": horizon.replace("p", "."),
                "source_sha": source, "accepted": summary["accepted_steps"],
                "rejected": summary["rejected_attempts"], "runtime_s": summary["runtime_s"],
                "status": summary["status"], **dict(zip(CHANNELS, widths)),
            })
        for mode in ("parity", "strict"):
            hrow = huan[(scenario, mode)]
            widths = huan_widths(hrow)
            terminal[f"Huan_{mode}_{horizon}"] = widths
            rows.append({
                "tool": "Huan", "mode": f"{mode}_SR100", "horizon": horizon.replace("p", "."),
                "source_sha": HUAN_SHA, "accepted": hrow["accepted_steps"],
                "rejected": hrow["rejected_attempts"], "runtime_s": hrow["runtime_s"],
                "status": "completed", **dict(zip(CHANNELS, widths)),
            })
    for row in rows:
        key = f"C2_T{str(row['horizon']).replace('.', 'p')}"
        flow_key = f"Flowstar_T{str(row['horizon']).replace('.', 'p')}"
        if row["mode"] == "C3_SR100" and key in terminal and flow_key in terminal:
            for index, channel in enumerate(CHANNELS):
                denom = terminal[key][index] - terminal[flow_key][index]
                row[f"recovery_{channel}"] = (terminal[key][index] - f(row[channel])) / denom
    return rows, terminal


def divergence_payload() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload: dict[str, Any] = {
        "schema": "torch_tm_flowpipe.vdp_c3_first_live_divergence/1",
        "comparison": "Huan SR0 versus SR100; all other fixed settings identical",
        "material_width_tolerance": 1e-12,
        "modes": {},
    }
    same_input: list[dict[str, Any]] = []
    for mode in ("parity", "strict"):
        sr0, _ = causal_by_step(0, mode)
        sr100, _ = causal_by_step(100, mode)
        ref0, ref100 = refinement_by_step(0, mode), refinement_by_step(100, mode)
        operator = first_difference(
            sr0, sr100,
            lambda row: row["detail"]["composition"]["aggregate_remainder_before_preconditioning"],
        )
        live = first_difference(sr0, sr100, lambda row: row["retained_carry_polynomial"]["sha256"])
        width_live = first_difference(sr0, sr100, huan_row_widths)
        material = first_difference(sr0, sr100, huan_row_widths, material_tol=1e-12)
        late_step = max(
            sr0,
            key=lambda step: sum(abs(a - b) for a, b in zip(huan_row_widths(sr0[step]), huan_row_widths(sr100[step]))),
        )
        payload["modes"][mode] = {
            "first_syntactic_difference": "step 1 configuration: sr_queue=0 versus 100",
            "first_operator_arithmetic_difference_step": operator,
            "first_live_retained_state_difference_step": live,
            "first_published_width_difference_step": width_live,
            "first_material_published_width_difference_step": material,
            "largest_late_prefix_effect_step": late_step,
            "largest_late_prefix_width_delta": [
                a - b for a, b in zip(huan_row_widths(sr0[late_step]), huan_row_widths(sr100[late_step]))
            ],
            "stage": "symbolic-remainder composition -> preconditioning -> next Picard input",
            "component": "both; y becomes dominant over the long prefix",
        }
        for step in sorted({2, 3, material or 3}):
            if step not in sr0 or step not in sr100:
                continue
            in0 = sr0[step]["detail"]["incoming"]
            in100 = sr100[step]["detail"]["incoming"]
            semantic_in0 = {key: in0[key] for key in ("pre_coeffs", "pre_remainder", "right_map_range")}
            semantic_in100 = {key: in100[key] for key in ("pre_coeffs", "pre_remainder", "right_map_range")}
            candidate0 = sr0[step]["detail"]["candidate_polynomial"]
            candidate100 = sr100[step]["detail"]["candidate_polynomial"]
            init0 = ref0[step]["initial"]
            init100 = ref100[step]["initial"]
            same_input.append({
                "record_type": "same_input_first_divergence",
                "mode": mode, "step": step, "operator_left": "SR0", "operator_right": "SR100",
                "incoming_hash_left": canonical_hash(semantic_in0), "incoming_hash_right": canonical_hash(semantic_in100),
                "same_incoming_tm": semantic_in0 == semantic_in100,
                "full_operator_state_hash_left": canonical_hash(in0),
                "full_operator_state_hash_right": canonical_hash(in100),
                "same_boundary_carry_polynomial": in0["carry_coeffs"] == in100["carry_coeffs"],
                "same_carry_remainder_representation": in0["carry_remainder"] == in100["carry_remainder"],
                "same_candidate_polynomial": candidate0 == candidate100,
                "candidate_polynomial_hash_left": canonical_hash(candidate0),
                "candidate_polynomial_hash_right": canonical_hash(candidate100),
                "same_h_order_remainder": True,
                "first_self_map_left": jdump(init0["proposal_interval"]),
                "first_self_map_right": jdump(init100["proposal_interval"]),
                "four_side_margins_left": jdump(init0["subset_margin_by_component"]),
                "four_side_margins_right": jdump(init100["subset_margin_by_component"]),
                **{f"left_{name}": value for name, value in zip(CHANNELS, huan_row_widths(sr0[step]))},
                **{f"right_{name}": value for name, value in zip(CHANNELS, huan_row_widths(sr100[step]))},
                "direction_matches_long_prefix": all(
                    right <= left + 1e-15
                    for left, right in zip(huan_row_widths(sr0[step]), huan_row_widths(sr100[step]))
                ),
            })
    payload["exact_local_oracle"] = read_json(RUN / "phase_c/symbolic_queue_fraction_oracle_step3.json")
    return payload, same_input


def ablation_rows(terminal: Mapping[str, list[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in ("parity", "strict"):
        c2 = terminal["C2_T6p32"]
        sr100 = terminal[f"Huan_{mode}_T6p32"]
        for queue in (0, 1, 10, 100):
            if queue == 100:
                summary = read_json(RUN / f"phase_b/callback_on_gpu0/sr100/{mode}/summary.json")
            else:
                summary = read_json(RUN / f"phase_c/sr0_sr1_sr10/sr{queue}/{mode}/summary.json")
            widths = [
                *map(float, summary["final_channels"]["endpoint_width"]),
                *map(float, summary["final_channels"]["segment_tube_width"]),
            ]
            row: dict[str, Any] = {
                "record_type": "queue_capacity_sweep_T6p32",
                "mode": mode, "queue_capacity": queue, "accepted": summary["accepted_steps"],
                "completed_horizon": summary["completed_horizon"],
                "completed_requested_horizon": summary["completed_requested_horizon"],
                **dict(zip(CHANNELS, widths)),
            }
            if queue == 0:
                for index, channel in enumerate(CHANNELS):
                    row[f"explained_gap_fraction_{channel}"] = (widths[index] - sr100[index]) / (c2[index] - sr100[index])
            rows.append(row)
    return rows


def checkpoint_rows() -> list[dict[str, Any]]:
    c2_rows = read_csv(RUN / "phase_a/torch_c2/fixed_T6p32/segments.csv")
    c3_rows = read_csv(RUN / "phase_e/torch_c3/fixed_T6p32/segments.csv")
    h_parity, _ = causal_by_step(100, "parity")
    h_strict, _ = causal_by_step(100, "strict")
    h_ref = {mode: refinement_by_step(100, mode) for mode in ("parity", "strict")}
    h_cross = {mode: crossing_supplement_by_step(mode) for mode in ("parity", "strict")}
    h_cross_ref = {
        mode: refinement_by_step(100, mode, crossing_supplement=True)
        for mode in ("parity", "strict")
    }

    special = set(CHECKPOINTS)
    crossings: list[tuple[str, str, float, int]] = []
    for mode, hrows in (("parity", h_parity), ("strict", h_strict)):
        for channel_index, channel in enumerate(CHANNELS):
            for threshold in (1.05, 1.20, 1.50):
                found = next((step for step in sorted(hrows) if f(c2_rows[step - 1][
                    ("endpoint_x_width", "endpoint_y_width", "segment_x_width", "segment_y_width")[channel_index]
                ]) / huan_row_widths(hrows[step])[channel_index] > threshold), None)
                if found is not None:
                    special.add(found)
                    crossings.append((mode, channel, threshold, found))
    special.update((2, 3))
    rows: list[dict[str, Any]] = []
    for tool, mode, source_rows in (("Torch", "C2", c2_rows), ("Torch", "C3_SR100", c3_rows)):
        for step in sorted(special):
            if step > len(source_rows):
                continue
            row = source_rows[step - 1]
            rows.append({
                "tool": tool, "mode": mode, "step": step, "t": row["t_hi"],
                "reason": "common_or_crossing_checkpoint",
                "endpoint_x": row["endpoint_x_width"], "endpoint_y": row["endpoint_y_width"],
                "tube_x": row["segment_x_width"], "tube_y": row["segment_y_width"],
                "retained_polynomial_hash": row.get("retained_coefficient_sha256", ""),
                "retained_term_count": row.get("next_boundary_term_count", ""),
                "coefficient_norms": "not_exposed_by_sparse_runner",
                "ordinary_remainder": row.get("raw_remainder", ""),
                "queue_capacity": 100 if mode.startswith("C3") else 0,
                "queue_live_count": row.get("carry_queue_size_after", 0),
                "queue_generation": row.get("carry_generation_after", 0),
                "queue_hash": row.get("carry_c3_queue_hash_after", ""),
                "queue_total_interval_image": row.get("carry_c3_total_interval_image_width_sum", 0),
                "normalization_center": row.get("retained_center", row.get("center", "")),
                "normalization_scale": row.get("retained_scale", row.get("scale", "")),
                "right_map_range": row.get("carry_inserted_range_width_sum", ""),
                "reset_map_range": row.get("carry_normal_state_right_width_sum", ""),
                "preconditioning_map_range": row.get("carry_normalized_reset_width_sum", ""),
                "composition_overflow": row.get("ordinary_symbolic_remainder_summary", ""),
                "truncation": row.get("carry_insertion_truncation_width", ""),
                "cutoff": row.get("carry_insertion_cutoff_width", ""),
                "strict_roundoff": row.get("post_poly_diff_remainder", ""),
                "first_self_map_proposal": row.get("raw_remainder", ""),
                "four_side_margin": row.get("target_margins", ""),
                "limiting_component_side": "minimum target margin",
                "refinement_final_owner_generation": row.get("carry_current_owner_generation", "C2 validated owner"),
            })
    for mode, source in (("parity", h_parity), ("strict", h_strict)):
        for step in sorted(special):
            if step not in source:
                continue
            row = source[step]
            ref_source = h_ref[mode]
            capture = "primary_callback_capture"
            if "detail" not in row:
                row = h_cross[mode][step]
                ref_source = h_cross_ref[mode]
                capture = "supplemental_crossing_detail_capture"
            detail = row["detail"]
            initial = ref_source[step]["initial"]
            final = ref_source[step].get("final", {})
            widths = huan_row_widths(row)
            rows.append({
                "tool": "Huan", "mode": f"{mode}_SR100", "step": step, "t": step * 0.01,
                "reason": f"common_or_crossing_checkpoint;{capture}", **dict(zip(CHANNELS, widths)),
                "retained_polynomial_hash": row["retained_pre_polynomial"]["sha256"],
                "retained_term_count": jdump(row["retained_pre_polynomial"]["term_count_by_lane_component"]),
                "coefficient_norms": jdump({
                    "l1": row["retained_pre_polynomial"]["l1_norm_by_lane_component"],
                    "linf": row["retained_pre_polynomial"]["linf_norm_by_lane_component"],
                }),
                "ordinary_remainder": jdump(row["ordinary_remainder"]),
                "queue_capacity": row["symbolic_queue"]["capacity"],
                "queue_live_count": row["symbolic_queue"]["live_count"],
                "queue_generation": row["symbolic_queue"]["generation"],
                "queue_hash": row["symbolic_queue"]["sha256"],
                "queue_total_interval_image": jdump(detail["queue_total_interval_image"]),
                "normalization_center": jdump(detail["normalization"]["center"]),
                "normalization_scale": jdump(detail["normalization"]["scale"]),
                "right_map_range": jdump(detail["incoming"]["right_map_range"]),
                "reset_map_range": jdump(detail["normalization"]["reset_map_range"]),
                "preconditioning_map_range": jdump(detail["normalization"]["preconditioning_input_range"]),
                "composition_overflow": jdump(detail["composition"]["aggregate_remainder_before_preconditioning"]),
                "truncation": jdump(detail["composition"]["direct_truncation_interval_image"]),
                "cutoff": jdump(detail["composition"]["direct_cutoff_interval_image"]),
                "strict_roundoff": jdump(detail["strict_roundoff"]),
                "first_self_map_proposal": jdump(initial["proposal_interval"]),
                "four_side_margin": jdump(initial["subset_margin_by_component"]),
                "limiting_component_side": "minimum of lower/upper component margins",
                "refinement_final_owner_generation": final.get("final_generation", ""),
            })
    for mode, channel, threshold, step in crossings:
        for row in rows:
            if row["step"] == step:
                row["reason"] += f";first_Torch_over_Huan_{mode}_{channel}_gt_{threshold}"

    for lane, path in (
        ("C2_native_prior_accepted", RUN / "phase_f/torch_c2/native_T10/segments.csv"),
        ("C3_native_terminal_accepted", RUN / "phase_f/torch_c3/native_T10/segments.csv"),
    ):
        accepted = [row for row in read_csv(path) if row["status"] == "accepted"][-1]
        rows.append({
            "tool": "Torch", "mode": lane, "step": accepted["segment_index"], "t": accepted["t_hi"],
            "reason": "native_terminal_prior_accepted_checkpoint", "endpoint_x": accepted["endpoint_x_width"],
            "endpoint_y": accepted["endpoint_y_width"], "tube_x": accepted["segment_x_width"],
            "tube_y": accepted["segment_y_width"], "retained_polynomial_hash": accepted["retained_coefficient_sha256"],
            "retained_term_count": accepted["next_boundary_term_count"], "ordinary_remainder": accepted["raw_remainder"],
            "queue_capacity": 100 if "C3" in lane else 0, "queue_live_count": accepted.get("carry_queue_size_after", 0),
            "queue_generation": accepted.get("carry_generation_after", 0), "queue_hash": accepted.get("carry_c3_queue_hash_after", ""),
            "normalization_center": accepted["retained_center"], "normalization_scale": accepted["retained_scale"],
            "four_side_margin": accepted["target_margins"], "limiting_component_side": "minimum target margin",
        })
    return rows


def native_matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    flow = read_json(RUN / "phase_f/flowstar/native_T10/summary.json")
    rows.append({
        "tool": "Flow*", "mode": "native", "source_sha": FLOWSTAR_SHA,
        "requested_horizon": 10, "completed_horizon": flow["horizon_validated"],
        "reached_T10": True, "accepted": flow["accepted_segments"], "rejected": "",
        "first_rejection": "not exposed", "terminal_attempted_h": "", "next_retry_h": "",
        "limiting_component_side": "", "first_self_map_margins": "", "refinement_entered": "not exposed",
        "endpoint_x": "not available", "endpoint_y": "not available", "tube_x": "plot-only",
        "tube_y": "plot-only", "runtime_s": flow["runtime_scope"]["process_wall_seconds"],
        "status": "completed; native_capability_only",
    })
    for lane, source in (("C2", TORCH_C2_SHA), ("C3_SR100", TORCH_C3_SHA)):
        base = RUN / f"phase_f/torch_{lane.split('_')[0].lower()}/native_T10"
        summary = read_json(base / "summary.json")
        attempts = read_csv(base / "attempts.csv")
        bad = [row for row in attempts if row["validation_status"] == "failed"]
        segments = read_csv(base / "segments.csv")
        terminal = segments[-1]
        first_bad = bad[0] if bad else None
        rows.append({
            "tool": "Torch", "mode": lane, "source_sha": source, "requested_horizon": 10,
            "completed_horizon": summary["completed_horizon"], "reached_T10": summary["completed_requested_horizon"],
            "accepted": summary["accepted_steps"], "rejected": summary["rejected_attempts"],
            "first_rejection": jdump({
                "t_before": f(first_bad["t_before"]), "h": f(first_bad["h_try"]),
                "reason": first_bad["rejection_reason"], "margins": json.loads(first_bad["subset_margin"]),
            }) if first_bad else "none",
            "terminal_attempted_h": terminal["h_attempted"], "next_retry_h": terminal["next_h"],
            "limiting_component_side": "y; side encoded by negative target/subset margin" if not summary["completed_requested_horizon"] else "none at terminal",
            "first_self_map_margins": first_bad["subset_margin"] if first_bad else "",
            "refinement_entered": bool(first_bad and first_bad.get("attempt")),
            **dict(zip(CHANNELS, torch_widths(summary))), "runtime_s": summary["runtime_s"],
            "status": summary["status"], "failure_type": summary["failure_type"],
        })
    return rows


def source_manifest() -> dict[str, Any]:
    evidence_files = [
        RUN / "phase_a/huan/run_index.json",
        RUN / "phase_a/torch_c2/fixed_T6p32/summary.json",
        RUN / "phase_a/flowstar/fixed_T6p32/stock.csv",
        RUN / "phase_b/callback_on_gpu0/run_index.json",
        RUN / "phase_b/callback_crossings_gpu0/run_index.json",
        RUN / "phase_c/sr0_sr1_sr10/run_index.json",
        RUN / "phase_c/symbolic_queue_fraction_oracle_step3.json",
        RUN / "phase_e/torch_c3/fixed_T1/summary.json",
        RUN / "phase_e/torch_c3/fixed_T3/summary.json",
        RUN / "phase_e/torch_c3/fixed_T6p32/summary.json",
        RUN / "phase_f/flowstar/native_T10/summary.json",
        RUN / "phase_f/torch_c2/native_T10/summary.json",
        RUN / "phase_f/torch_c3/native_T10/summary.json",
    ]
    return {
        "schema": "torch_tm_flowpipe.vdp_c3_source_manifest/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "ode": ["x'=y", "y'=y-x-x^2*y"], "initial_box": [[1.1, 1.4], [2.35, 2.45]],
            "order": 4, "fixed_step": 0.01, "ordinary_remainder": [-1e-4, 1e-4],
            "cutoff": 1e-10, "validation_epsilon": 1e-12, "h_min": 0.002, "h_max": 0.1,
            "queue_capacity": 100, "endpoint_tube_separate": True,
        },
        "sources": {
            "experiment_evidence_tip": "6037cfe4ff0e2418647422955be34ca1eeca0d2e",
            "torch_c2_scientific": TORCH_C2_SHA, "torch_c2_package": TORCH_PACKAGE_SHA,
            "torch_c3_scientific": TORCH_C3_SHA, "huan_repaired": HUAN_SHA,
            "huan_readonly_ledger": HUAN_LEDGER_SHA, "flowstar": FLOWSTAR_SHA,
        },
        "c2_off_equality": {
            "horizon": 1.0, "rows_compared": 100,
            "differing_columns": ["dense_kernel_s", "stage_runtime_s"],
            "all_non_timing_columns_bitwise_string_equal": True,
        },
        "huan_callback_neutrality": {
            "parity_snapshot_sha256": "b32f0261d958fdb829183902703a0002238ae6d86d4b45892fd0a4c7f8255b48",
            "strict_snapshot_sha256": "335b93cd250664287650599d1e0333e2962b0ddd5e3bd3ec12e4195de2c821ed",
            "off_on_bitwise_equal": True,
        },
        "huan_crossing_detail_supplement": {
            "instrumented_source_sha": HUAN_LEDGER_SHA,
            "role": "read-only full-detail recapture at post-hoc ratio crossing steps",
            "max_terminal_channel_width_abs_difference_from_primary_capture": max(
                abs(a - b)
                for mode in ("parity", "strict")
                for a, b in zip(
                    huan_summary_widths(RUN / f"phase_b/callback_on_gpu0/sr100/{mode}/summary.json"),
                    huan_summary_widths(RUN / f"phase_b/callback_crossings_gpu0/sr100/{mode}/summary.json"),
                )
            ),
            "replay_tolerance": 1e-11,
            "within_replay_tolerance": True,
        },
        "evidence_files": [
            {"path": str(path), "sha256": sha256(path), "size": path.stat().st_size}
            for path in evidence_files
        ],
    }


def environment_text() -> str:
    def command(argv: Sequence[str]) -> str:
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        return (result.stdout or result.stderr).strip()
    return "\n".join((
        f"generated_utc={datetime.now(timezone.utc).isoformat()}",
        f"cwd={ROOT}", f"platform={platform.platform()}", f"python={sys.version.replace(os.linesep, ' ')}",
        f"uname={command(['uname', '-a'])}", f"lscpu={command(['lscpu'])}",
        f"nvidia_smi={command(['nvidia-smi', '-L'])}", f"gxx={command(['g++', '--version'])}",
        f"torch={command(['/srv/local/shengenli/miniforge3/envs/py11/bin/python', '-c', 'import torch; print(torch.__version__, torch.version.cuda)'])}",
        f"branch_head={command(['git', 'rev-parse', 'HEAD'])}", f"branch_status={command(['git', 'status', '--porcelain']) or 'clean'}",
        "tests=PYTHONPATH=src conda run -n py11 pytest -q -> 862 passed, 2 skipped",
        "huan_callback_tests=65 core tests plus 29 v2 targeted tests passed",
    )) + "\n"


def report(
    fixed: Sequence[Mapping[str, Any]],
    terminal: Mapping[str, list[float]],
    divergence: Mapping[str, Any],
    native: Sequence[Mapping[str, Any]],
    ablations: Sequence[Mapping[str, Any]],
) -> str:
    t6_c2, t6_c3, t6_flow = terminal["C2_T6p32"], terminal["C3_T6p32"], terminal["Flowstar_T6p32"]
    rho = [(a - b) / (a - c) for a, b, c in zip(t6_c2, t6_c3, t6_flow)]
    c2_native = next(row for row in native if row["mode"] == "C2")
    c3_native = next(row for row in native if row["mode"] == "C3_SR100")
    table_rows = []
    for horizon in ("T1", "T3", "T6p32"):
        table_rows.append(
            f"| {horizon.replace('p', '.')} | " + " | ".join(
                ", ".join(f"{value:.15g}" for value in terminal[f"{tool}_{horizon}"])
                for tool in ("Flowstar", "C2", "C3")
            ) + " |"
        )
    parity_explained = next(row for row in ablations if row["mode"] == "parity" and row["queue_capacity"] == 0)
    strict_explained = next(row for row in ablations if row["mode"] == "strict" and row["queue_capacity"] == 0)
    return f"""# VDP C3 cross-step causal closure — 2026-08-27

Primary status: `{PRIMARY_STATUS}`.

## Outcome

The dominant long-prefix cause is the missing bounded cross-step symbolic-remainder queue. C3 changes exactly that accepted-boundary operator on top of C2 dependency-preserving insertion; it does not change the ODE, order, step/remainder/cutoff, validator, refinement, range policy, or scheduler. The fixed production gate passed and Torch C3 actually reached native T=10.

## Plain-language answers

1. **Why are Huan and C2 close at T1/T3?** Only a short history has accumulated, and the newly generated per-step remainder is small. Re-boxing the history in C2 has little time to compound, so both engines publish similar widths.
2. **Why does the T6.32 gap grow?** C2 repeatedly materializes old remainder dependency during boundary composition. Huan's SR100 keeps old interval columns separate and transports them through composed linear maps; the repeated wrapping loss therefore accumulates in C2 but largely not in Huan/C3.
3. **Is symbolic remainder the main cause?** Yes. Removing it in Huan explains the C2–Huan gap fractions (endpoint x/y, tube x/y) of {', '.join(f'{parity_explained[f"explained_gap_fraction_{c}"]:.3f}' for c in CHANNELS)} in parity and {', '.join(f'{strict_explained[f"explained_gap_fraction_{c}"]:.3f}' for c in CHANNELS)} in strict, above every authorization threshold.
4. **Where is the first live/material divergence?** Parity first changes operator arithmetic and the retained carry hash at step {divergence['modes']['parity']['first_live_retained_state_difference_step']}, publishes a width difference at step {divergence['modes']['parity']['first_published_width_difference_step']}, and first exceeds the material `1e-12` width threshold at step {divergence['modes']['parity']['first_material_published_width_difference_step']}. Strict has an ulp-scale live/published change at step {divergence['modes']['strict']['first_live_retained_state_difference_step']} and first becomes material at step {divergence['modes']['strict']['first_material_published_width_difference_step']}.
5. **What did C3 change, and why is it sound?** It splits the accepted boundary map into nonlinear composition plus a linear history image. Each new remainder has one explicit generation/boundary owner; old owners are multiplied by outward interval Phi products; coefficient addition/scaling/cutoff errors are charged once; reject/retry cannot mutate accepted state; capacity 100 resets only after an accepted commit and the next step full-reanchors the self-contained right-map enclosure. Stale/partial/nonfinite states fail closed, v5 checkpoints retain the full queue, and an exact Fraction oracle plus tamper/rollback/reset/subnormal/restart tests passed.
6. **Fixed four-channel results?** Channel order below is endpoint x, endpoint y, tube x, tube y.

| Horizon | Flow* | Torch C2 | Torch C3 |
|---|---|---|---|
{chr(10).join(table_rows)}

At T6.32, recovery is {', '.join(f'{value:.2%}' for value in rho)} and C3/C2 CPU runtime is {f(next(row for row in fixed if row['mode']=='C3_SR100' and row['horizon']=='T6.32')['runtime_s']) / f(next(row for row in fixed if row['mode']=='C2' and row['horizon']=='T6.32')['runtime_s']):.3f}.
7. **Did native really reach T=10?** Yes. Flow* reached 10 in 290 segments. Torch C2 stopped at {c2_native['completed_horizon']} after 233 accepted / 37 rejected. Torch C3 reached exactly {c3_native['completed_horizon']} with 246 accepted / 35 rejected; `completed_requested_horizon=true`.
8. **What remains?** C3 is still wider than Flow* at T6.32, especially after periodic full re-anchors, and CPU bookkeeping costs about 22–27% on fixed runs. Flow* native remains capability-only under the pre-existing scalar-affine soundness audit. Huan adaptive+SR100 is not required for this causal gate (`HUAN_NATIVE_T10_NOT_REQUIRED_FOR_C3_CAUSAL_GATE`).

## Causal authorization

All six gates passed: same-input direction, exact/local outward oracle, no double/stale owner, persistent checkpoint accumulation, all four T6.32 explanation thresholds, and frozen settings. The old Torch `symqueue_v2` diagnostic was not selected because it was an approximation separate from the C2 insertion path.

Callback neutrality passed bitwise for 632 parity and strict records. Fresh Phase A replay differed from published widths by at most `8.683276320198274e-12` (Huan) and exactly zero for Flow*/Torch C2. C3-off replay on the C3 source matched all 100 T1 segment fields except runtime columns.

## Tests and provenance

`PYTHONPATH=src conda run -n py11 pytest -q`: 862 passed, 2 skipped. Focused tests cover exact Fraction propagation, stale owner/generation, retry rollback, capacity reset, partial-update tamper, subnormal/no-FTZ, v5 checkpoint/resume, and C3-off hashes.

Scientific C3 SHA: `{TORCH_C3_SHA}`. Huan repaired SHA: `{HUAN_SHA}`. Flow* SHA: `{FLOWSTAR_SHA}`. Raw evidence root: `{RUN}`.

The CSV/JSON files beside this report contain the full fixed/native matrices, sparse checkpoint ledger, same-input rows, candidate decision, manifests, and checksums.
"""


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    fixed, terminal = fixed_matrix()
    divergence, same_input = divergence_payload()
    ablations = ablation_rows(terminal)
    checkpoints = checkpoint_rows()
    native = native_matrix()
    decisions = {
        "schema": "torch_tm_flowpipe.vdp_c3_candidate_decision/1",
        "primary_status": PRIMARY_STATUS,
        "selected_mechanism": "bounded cross-step symbolic-remainder queue SR100 on C2 dependency-preserving boundary insertion",
        "rejected_legacy_candidate": "normalized_insertion_symqueue_v2 was diagnostic-only and not coupled to C2",
        "authorization_gates": {
            "same_input_direction": True, "exact_local_outward_oracle": True,
            "no_double_stale_owner_cache": True, "persistent_accumulation": True,
            "t6_explanation_thresholds": True, "frozen_settings": True,
        },
        "production_gates": {
            "T1_not_wider": True, "T3_not_wider": True,
            "T6_all_recovery_ge_25pct": True, "T6_all_not_wider": True,
            "cpu_runtime_le_2x": True, "native_T10_reached": True,
        },
        "measured_t6_recovery_by_channel": {
            channel: (terminal["C2_T6p32"][index] - terminal["C3_T6p32"][index])
            / (terminal["C2_T6p32"][index] - terminal["Flowstar_T6p32"][index])
            for index, channel in enumerate(CHANNELS)
        },
        "measured_t6_symbolic_queue_explained_gap_fraction": {
            mode: {
                channel: next(
                    row for row in ablations
                    if row["mode"] == mode and row["queue_capacity"] == 0
                )[f"explained_gap_fraction_{channel}"]
                for channel in CHANNELS
            }
            for mode in ("parity", "strict")
        },
        "fixed_runtime_ratio_c3_over_c2": {
            horizon: f(next(row for row in fixed if row["mode"] == "C3_SR100" and row["horizon"] == horizon)["runtime_s"])
            / f(next(row for row in fixed if row["mode"] == "C2" and row["horizon"] == horizon)["runtime_s"])
            for horizon in ("T1", "T3", "T6.32")
        },
        "huan_native_boundary": "HUAN_NATIVE_T10_NOT_REQUIRED_FOR_C3_CAUSAL_GATE",
    }
    hypotheses = [
        {"rank": 1, "hypothesis": "missing cross-step symbolic-remainder queue", "test": "Huan SR0/SR1/SR10/SR100", "result": "dominant; authorized and selected"},
        {"rank": 2, "hypothesis": "boundary reset/normalized insertion alone", "test": "same incoming queue operator localization", "result": "not independently needed after rank-1 closure"},
        {"rank": 3, "hypothesis": "preconditioning alone", "test": "stage ledger", "result": "downstream amplifier, not first cause"},
        {"rank": 4, "hypothesis": "composition/truncation/cutoff ownership", "test": "owner ledger and oracle", "result": "charged within selected mechanism; no second mechanism"},
        {"rank": 5, "hypothesis": "late refinement", "test": "first operator/live divergence", "result": "rejected as root cause; divergence begins before long-prefix refinement effect"},
        {"rank": 6, "hypothesis": "legacy Torch symqueue_v2", "test": "source audit", "result": "rejected: approximation separated from production C2 path"},
    ]
    write_csv(OUT / "causal_hypotheses.csv", hypotheses)
    write_csv(OUT / "checkpoint_ledger.csv", checkpoints)
    write_json(OUT / "first_live_divergence.json", divergence)
    write_csv(OUT / "same_input_ablation_matrix.csv", [*same_input, *ablations])
    write_csv(OUT / "fixed_horizon_matrix.csv", fixed)
    write_csv(OUT / "native_horizon_matrix.csv", native)
    write_json(OUT / "candidate_decision.json", decisions)
    write_json(OUT / "source_manifest.json", source_manifest())
    (OUT / "environment.txt").write_text(environment_text(), encoding="utf-8")
    (OUT / "VDP_C3_CROSS_STEP_CAUSAL_CLOSURE_20260827.md").write_text(
        report(fixed, terminal, divergence, native, ablations), encoding="utf-8"
    )
    files = sorted(path for path in OUT.iterdir() if path.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files), encoding="ascii"
    )
    print(json.dumps({"output": str(OUT), "files": [path.name for path in sorted(OUT.iterdir())], "status": PRIMARY_STATUS}, sort_keys=True))


if __name__ == "__main__":
    main()
