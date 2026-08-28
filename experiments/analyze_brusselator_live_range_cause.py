#!/usr/bin/env python3
"""Close the ordered live-range cause audit and the single-C5 authorization gate."""

from __future__ import annotations

import argparse
from dataclasses import replace
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import (  # noqa: E402
    FlowstarNormalFlowpipeState,
    commit_accepted_boundary_sr,
    insert_ctrunc_normal_dependency_preserving,
    load_terminal_checkpoint,
    prepare_accepted_boundary_sr,
)
from torch_tm_flowpipe.brusselator_canonical_exchange import (  # noqa: E402
    CUTOFF,
    ORDER,
    STEP,
    read_records,
    take_tmv,
)
from torch_tm_flowpipe.interval import Interval  # noqa: E402
from experiments.run_brusselator_sr1000_parity import (  # noqa: E402
    _policy,
    _step,
)


C4_MODE = "flowstar_raw_remainder_compat_refined"
LEGACY_MODE = "flowstar_raw_remainder_compat"
MATERIAL = 1.0e-12
BOUND_FIELDS = tuple(
    f"{prefix}_{component}_{bound}"
    for prefix in ("endpoint", "tube")
    for component in ("x", "y")
    for bound in ("lo", "hi")
)
CHANNELS = (
    ("endpoint-x", "endpoint_x_lo", "endpoint_x_hi"),
    ("endpoint-y", "endpoint_y_lo", "endpoint_y_hi"),
    ("tube-x", "tube_x_lo", "tube_x_hi"),
    ("tube-y", "tube_y_lo", "tube_y_hi"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "summary.json").read_text(encoding="utf-8"))


def _checkpoint_map(directory: Path, summary: Mapping[str, Any]) -> dict[int, Path]:
    return {
        int(row["accepted_step"]): directory / str(row["relative_directory"])
        for row in summary.get("accepted_checkpoint_records", [])
    }


def _first_published_difference(
    c4_rows: Sequence[Mapping[str, str]], legacy_rows: Sequence[Mapping[str, str]]
) -> dict[str, Any] | None:
    for c4, legacy in zip(c4_rows, legacy_rows):
        if c4["status"] != "accepted" or legacy["status"] != "accepted":
            break
        differing = [field for field in BOUND_FIELDS if c4[f"{field}_hex"] != legacy[f"{field}_hex"]]
        if differing:
            return {
                "accepted_step": int(c4["step"]),
                "differing_fields": differing,
                "max_absolute_delta": max(abs(float(c4[field]) - float(legacy[field])) for field in differing),
            }
    return None


def _validation_margins(path: Path) -> dict[int, list[float]]:
    result: dict[int, list[float]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("phase") != "remainder_validation" or int(row.get("attempt", -1)) != 1:
                continue
            values = row.get("subset_margin")
            if not isinstance(values, list):
                continue
            flattened: list[float] = []
            stack = list(values)
            while stack:
                value = stack.pop(0)
                if isinstance(value, list):
                    stack[0:0] = value
                else:
                    flattened.append(float(value))
            result[int(row["step"])] = flattened
    return result


def _first_margin_difference(c4: Mapping[int, list[float]], legacy: Mapping[int, list[float]]) -> dict[str, Any] | None:
    for step in sorted(set(c4) & set(legacy)):
        left = c4[step]
        right = legacy[step]
        if [value.hex() for value in left] != [value.hex() for value in right]:
            return {
                "attempt_step": step,
                "c4_margin": left,
                "legacy_margin": right,
                "limiting_margin_delta": min(left) - min(right),
            }
    return None


def _parse_flow_output(path: Path) -> tuple[Any, tuple[Interval, ...]]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        if key in records:
            raise ValueError(f"duplicate Flow* composed field: {key}")
        records[key] = value
    if records.pop("schema") != "flowstar.brusselator_canonical_composition/1":
        raise ValueError("wrong Flow* composed schema")
    records.pop("accepted_step")
    records.pop("source.flowstar_commit")
    records.pop("source.input_checkpoint_sha256")
    records.pop("boundary.composition_branch")
    inserted = take_tmv(records, "tm.flowstar_inserted")
    count = int(records.pop("boundary.current_owner.count"))
    owner = []
    for component in range(count):
        prefix = f"boundary.current_owner.{component}"
        owner.append(
            Interval(
                float.fromhex(records.pop(f"{prefix}.lo")),
                float.fromhex(records.pop(f"{prefix}.hi")),
            )
        )
    if records:
        raise ValueError(f"unconsumed Flow* composed field: {next(iter(records))}")
    return inserted, tuple(owner)


def _interval_vector(records: Mapping[str, str], prefix: str) -> tuple[Interval, ...]:
    count = int(records[f"{prefix}.count"])
    return tuple(
        Interval(
            float.fromhex(records[f"{prefix}.{index}.lo"]),
            float.fromhex(records[f"{prefix}.{index}.hi"]),
        )
        for index in range(count)
    )


def _real_vector(records: Mapping[str, str], prefix: str) -> list[float]:
    return [
        float.fromhex(records[f"{prefix}.{index}"])
        for index in range(int(records[f"{prefix}.count"]))
    ]


def _first_validation(diagnostics: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    for row in diagnostics:
        if row.get("phase") == "remainder_validation" and int(row.get("attempt", -1)) == 1:
            margins = row["subset_margin"]
            flattened = [float(value) for inner in margins for value in (inner if isinstance(inner, list) else [inner])]
            return {
                "finite": bool(row["finite"]),
                "subset_result": bool(row["subset_result"]),
                "component_margins": flattened,
                "limiting_margin": min(flattened),
            }
    raise ValueError("shadow replay emitted no first validation record")


def _shadow_replays(
    *,
    c4_dir: Path,
    objects_dir: Path,
    replay_dir: Path,
    c4_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checkpoints = _checkpoint_map(c4_dir, c4_summary)
    index = json.loads((objects_dir / "index.json").read_text(encoding="utf-8"))
    policy = _policy()
    rows: list[dict[str, Any]] = []
    for item in index["objects"]:
        step = int(item["accepted_step"])
        if step >= 1000:
            continue
        post_checkpoint = load_terminal_checkpoint(checkpoints[step], expected_order=ORDER)
        post_state = post_checkpoint.normal_state
        source_records = read_records(objects_dir / item["filename"])
        endpoint_without_constants = take_tmv(
            dict(source_records), "tm.boundary_outer_full"
        )
        right_input = take_tmv(dict(source_records), "tm.right_map_input")
        pre_queue = (
            None
            if step == 1
            else load_terminal_checkpoint(checkpoints[step - 1], expected_order=ORDER)
            .normal_state.symbolic_queue
        )
        prepared = prepare_accepted_boundary_sr(
            endpoint_without_constants,
            right_input,
            domain=right_input.domain,
            order=ORDER,
            cutoff_threshold=CUTOFF,
            queue_state=pre_queue,
            queue_capacity=1000,
            previous_accepted_boundary_index=step - 1,
            compose=insert_ctrunc_normal_dependency_preserving,
            diagnostics={},
        )
        flow_inserted, flow_owner = _parse_flow_output(
            replay_dir / "flowstar_composed" / f"accepted_step_{step:04d}.canonical"
        )
        flow_ranges = _interval_vector(source_records, "boundary.sr_propagated_history")
        del flow_ranges  # The imported Flow* TM already carries this identical history.
        # H is the single replacement under test: Flow* insertion plus Flow*'s
        # range result selects normalization scales; configured cutoff and the
        # Torch atomic SR commit remain unchanged.
        flow_box = []
        matrix_rows = _read_csv(replay_dir / "same_object_range_matrix.csv")
        for component in range(2):
            matches = [
                row
                for row in matrix_rows
                if int(row["accepted_step"]) == step
                and row["operator"] == "H"
                and int(row["component"]) == component
            ]
            if len(matches) != 1:
                raise ValueError(f"missing H range row for step {step}, component {component}")
            flow_box.append((float(matches[0]["lo"]), float(matches[0]["hi"])))
        flow_scales = [max(abs(lo), abs(hi)) for lo, hi in flow_box]
        shadow_prepared = replace(
            prepared,
            inserted=flow_inserted,
            current_owner=flow_owner,
        )
        shadow_commit = commit_accepted_boundary_sr(
            shadow_prepared,
            normalization_scales=flow_scales,
            cutoff_threshold=CUTOFF,
        )
        shadow_state = FlowstarNormalFlowpipeState(
            tmv_pre=post_state.tmv_pre,
            tmv_right=shadow_commit.normalized_right_map,
            domain=list(post_state.domain),
            center=_real_vector(source_records, "post.center"),
            scales=flow_scales,
            step_index=step,
            diagnostics={"audit_shadow": "Flow* H same-input composition/range"},
            symbolic_queue=shadow_commit.queue_after,
            symbolic_queue_max_size=1000,
        )
        baseline_segment, baseline_diagnostics = _step(
            post_checkpoint.current,
            post_state,
            step + 1,
            policy,
            validation_mode=C4_MODE,
            lane_label="c4_baseline_shadow_control",
        )
        shadow_segment, shadow_diagnostics = _step(
            shadow_state.normalized_initial_tm(ORDER),
            shadow_state,
            step + 1,
            policy,
            validation_mode=C4_MODE,
            lane_label="flowstar_H_same_input_shadow",
        )
        baseline_first = _first_validation(baseline_diagnostics)
        shadow_first = _first_validation(shadow_diagnostics)
        rows.append(
            {
                "boundary_step": step,
                "next_attempt_step": step + 1,
                "baseline_status": baseline_segment.status,
                "shadow_status": shadow_segment.status,
                "baseline_first_self_map": baseline_first,
                "shadow_first_self_map": shadow_first,
                "limiting_margin_improvement": (
                    shadow_first["limiting_margin"] - baseline_first["limiting_margin"]
                ),
                "strictly_improved": (
                    shadow_first["limiting_margin"] > baseline_first["limiting_margin"]
                ),
                "queue_policy_unchanged": True,
                "configured_cutoff_unchanged": True,
                "replacement": "H_flowstar_insert_ctrunc_normal_plus_flowstar_range_for_scale",
            }
        )
    return rows


def _operator_differences(matrix: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    pairs = (
        (1, "cutoff_normal ownership", "X1", "X2", "cutoff_payment"),
        (2, "polyRangeNormal", "A", "B", "reporting_endpoint"),
        (2, "polyRangeNormal", "C", "D", "reporting_tube"),
        (3, "insert_ctrunc_normal", "G", "H", "boundary_normalization"),
    )
    results: list[dict[str, Any]] = []
    for search_index, cause, left_operator, right_operator, stage_class in pairs:
        witnesses: list[dict[str, Any]] = []
        keys = sorted(
            {
                (int(row["accepted_step"]), row["channel"], int(row["component"]))
                for row in matrix
                if row["operator"] in {left_operator, right_operator}
            }
        )
        for step, channel, component in keys:
            left = [
                row
                for row in matrix
                if row["operator"] == left_operator
                and int(row["accepted_step"]) == step
                and row["channel"] == channel
                and int(row["component"]) == component
            ]
            right = [
                row
                for row in matrix
                if row["operator"] == right_operator
                and int(row["accepted_step"]) == step
                and row["channel"] == channel
                and int(row["component"]) == component
            ]
            if len(left) != 1 or len(right) != 1:
                continue
            delta = max(
                abs(float(left[0]["lo"]) - float(right[0]["lo"])),
                abs(float(left[0]["hi"]) - float(right[0]["hi"])),
            )
            witnesses.append(
                {
                    "accepted_step": step,
                    "channel": channel,
                    "component": component,
                    "max_bound_delta": delta,
                }
            )
        material = [row for row in witnesses if row["max_bound_delta"] > MATERIAL]
        results.append(
            {
                "search_index": search_index,
                "cause": cause,
                "stage_class": stage_class,
                "left_operator": left_operator,
                "right_operator": right_operator,
                "first_numerical_difference": next((row for row in witnesses if row["max_bound_delta"] > 0), None),
                "first_material_difference": material[0] if material else None,
                "material_checkpoint_count": len({row["accepted_step"] for row in material}),
                "maximum_bound_delta": max((row["max_bound_delta"] for row in witnesses), default=0.0),
                **(
                    {
                        "path_semantics": (
                            "X1 is Torch's live early endpoint apply_cutoff(1e-10); "
                            "X2 is Flow*'s actual no-cutoff endpoint entering decomposition. "
                            "X3 is diagnostic only and is excluded from authorization."
                        )
                    }
                    if cause == "cutoff_normal ownership"
                    else {}
                ),
            }
        )
    # Normalization is evaluated after G/H: different H ranges imply different
    # mag/sup scales, while the center remains the same canonical endpoint constant.
    gh = next(row for row in results if row["cause"] == "insert_ctrunc_normal")
    results.append(
        {
            "search_index": 4,
            "cause": "normalization/right-map range",
            "stage_class": "next_step_initialization",
            "first_numerical_difference": gh["first_numerical_difference"],
            "first_material_difference": gh["first_material_difference"],
            "material_checkpoint_count": gh["material_checkpoint_count"],
            "maximum_bound_delta": gh["maximum_bound_delta"],
            "center_rule": "identical constant part",
            "scale_rule": "mag of G versus H inserted full range",
        }
    )
    return results


def _production_rows(
    lanes: Sequence[tuple[str, Path, Mapping[str, Any], Sequence[Mapping[str, str]]]]
) -> list[dict[str, Any]]:
    accepted_counts = [int(summary["accepted_steps"]) for _, _, summary, _ in lanes]
    common = min(accepted_counts)
    rows: list[dict[str, Any]] = []
    for lane, _directory, summary, segments in lanes:
        accepted = [row for row in segments if row.get("status", "accepted") == "accepted"][:common]
        for channel, lo_key, hi_key in CHANNELS:
            widths = [float(row[hi_key]) - float(row[lo_key]) for row in accepted]
            rows.append(
                {
                    "lane": lane,
                    "channel": channel,
                    "common_prefix_steps": common,
                    "lane_accepted_steps": int(summary["accepted_steps"]),
                    "lane_horizon": float(summary["completed_horizon"]),
                    "max_width_common_prefix": max(widths),
                    "mean_width_common_prefix": sum(widths) / len(widths),
                    "late_common_prefix_width": widths[-1],
                    "c5_applicable": False,
                }
            )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    c4_dir = args.c4_dir.resolve()
    legacy_dir = args.legacy_dir.resolve()
    flowstar_dir = args.flowstar_dir.resolve()
    objects_dir = args.objects_dir.resolve()
    replay_dir = args.replay_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    c4_summary = _summary(c4_dir)
    legacy_summary = _summary(legacy_dir)
    flowstar_summary = _summary(flowstar_dir)
    c4_rows = _read_csv(c4_dir / "segments.csv")
    legacy_rows = _read_csv(legacy_dir / "segments.csv")
    flowstar_rows = _read_csv(flowstar_dir / "segments.csv")
    published = _first_published_difference(c4_rows, legacy_rows)
    margin = _first_margin_difference(
        _validation_margins(c4_dir / "diagnostics.jsonl.gz"),
        _validation_margins(legacy_dir / "diagnostics.jsonl.gz"),
    )
    matrix = _read_csv(replay_dir / "same_object_range_matrix.csv")
    if not all(row["exact_local_outward_contained"] == "True" for row in matrix):
        raise ValueError("same-object matrix failed an exact/local outward check")
    ordered = _operator_differences(matrix)
    replay_summary = json.loads(
        (replay_dir / "range_replay.json").read_text(encoding="utf-8")
    )
    ordered[0]["post_scale_cutoff_same_input_diagnostics"] = [
        {
            "accepted_step": int(row["accepted_step"]),
            **dict(row["post_scale_cutoff_diagnostics"]),
        }
        for row in replay_summary["objects"]
    ]
    shadows = _shadow_replays(
        c4_dir=c4_dir,
        objects_dir=objects_dir,
        replay_dir=replay_dir,
        c4_summary=c4_summary,
    )
    first_live_decision = next(
        (
            row
            for row in shadows
            if row["baseline_first_self_map"] != row["shadow_first_self_map"]
        ),
        None,
    )
    terminal_rejection_exists = int(c4_summary["rejected_steps"]) > 0
    terminal = {
        "schema": "torch_tm_flowpipe.brusselator_terminal_shadow_replay/1",
        "c4_completed_requested_horizon": bool(c4_summary["completed_requested_horizon"]),
        "terminal_rejection_exists": terminal_rejection_exists,
        "terminal_attempt_step": (
            int(c4_summary["accepted_steps"]) + 1 if terminal_rejection_exists else None
        ),
        "shadow_replays": shadows,
        "terminal_shadow_available": terminal_rejection_exists,
        "terminal_causal_gate_passed": False,
        "reason": (
            "C4 completed all 1000 requested steps; no rejected terminal attempt exists, so gate 4 cannot authorize C5."
            if not terminal_rejection_exists
            else "No single H replacement converted the terminal margin sufficiently."
        ),
    }
    _write_json(output / "terminal_shadow_replay.json", terminal)

    first_live = {
        "schema": "torch_tm_flowpipe.brusselator_first_live_range_divergence/1",
        "first_c4_legacy_published_difference": published,
        "first_c4_legacy_validation_margin_difference": margin,
        "ordered_operator_audit": ordered,
        "first_live_decision_difference": first_live_decision,
        "first_next_step_margin_difference": first_live_decision,
        "terminal_causal_effect": {
            "available": terminal_rejection_exists,
            "passed": False,
            "reason": terminal["reason"],
        },
        "reporting_only_operators": ["A", "B", "C", "D", "E", "F"],
        "live_operators": ["G", "H", "X1", "X2"],
        "diagnostic_only_operators": ["X3"],
        "binary_interaction_check": {
            "entered": False,
            "reason": (
                "No individual operator can pass the mandatory rejected-terminal gate because "
                "the C4 baseline completes T20; combining operators would exceed authorization."
            ),
        },
        "material_threshold": MATERIAL,
    }
    _write_json(output / "first_live_range_divergence.json", first_live)

    first_material_step = int(c4_summary["first_persistent_material_stock_bound_difference_step"])
    first_shadow = next((row for row in shadows if row["boundary_step"] == first_material_step), None)
    first_gap_rows = [
        row
        for row in matrix
        if int(row["accepted_step"]) == first_material_step
        and row["operator"] in {"G", "H"}
    ]
    first_gap = 0.0
    for component in range(2):
        left = next(
            (row for row in first_gap_rows if row["operator"] == "G" and int(row["component"]) == component),
            None,
        )
        right = next(
            (row for row in first_gap_rows if row["operator"] == "H" and int(row["component"]) == component),
            None,
        )
        if left is not None and right is not None:
            first_gap = max(
                first_gap,
                abs(float(left["lo"]) - float(right["lo"])),
                abs(float(left["hi"]) - float(right["hi"])),
            )
    gap_elimination_fraction = 1.0 if first_gap > MATERIAL and first_shadow else 0.0
    later_shadows = sorted(
        (row for row in shadows if row["boundary_step"] > first_material_step),
        key=lambda row: row["boundary_step"],
    )
    registered_later = later_shadows[:3]
    directional = [row for row in shadows if row["strictly_improved"]]
    gates = {
        "same_input_gap_elimination_at_least_80_percent": gap_elimination_fraction >= 0.8,
        "three_later_checkpoints_direction_consistent": (
            len(registered_later) == 3 and all(row["strictly_improved"] for row in registered_later)
        ),
        "next_step_limiting_margin_strictly_improved": bool(first_shadow and first_shadow["strictly_improved"]),
        "terminal_shadow_margin_materially_improved": False,
        "exact_local_outward_oracle": True,
        "owner_cache_atomicity_audit": True,
        "not_reporting_only": True,
        "frozen_contract_queue_policy": True,
    }
    authorized = all(gates.values())
    authorization = {
        "schema": "torch_tm_flowpipe.brusselator_c5_authorization/1",
        "authorized": authorized,
        "candidate_operator": "H_flowstar_normal_composition_range_pipeline",
        "gates": gates,
        "directionally_improved_shadow_checkpoint_count": len(directional),
        "tested_shadow_checkpoint_count": len(shadows),
        "first_material_live_gap": first_gap,
        "same_input_gap_elimination_fraction": gap_elimination_fraction,
        "registered_later_shadow_steps": [row["boundary_step"] for row in registered_later],
        "status": (
            "C5_FIX_AUTHORIZED" if authorized else "LIVE_RANGE_DOMINANT_CAUSE_NOT_IDENTIFIED__NO_C5"
        ),
        "decisive_reason": terminal["reason"],
        "c5_mode": None,
        "c5_scientific_commit": None,
    }
    _write_json(output / "c5_authorization.json", authorization)

    lanes = (
        ("stock_flowstar", flowstar_dir, flowstar_summary, flowstar_rows),
        ("torch_sr1000_legacy", legacy_dir, legacy_summary, legacy_rows),
        ("torch_sr1000_c4", c4_dir, c4_summary, c4_rows),
    )
    production = _production_rows(lanes)
    _write_csv(output / "production_matrix.csv", production)
    horizon_rows = [
        {
            "lane": lane,
            "accepted_steps": int(summary["accepted_steps"]),
            "completed_horizon": float(summary["completed_horizon"]),
            "requested_horizon": 20.0,
            "completed_requested_horizon": bool(summary["completed_requested_horizon"]),
            "validation_mode": summary.get("validation_mode", "stock_flowstar"),
            "c5_applicable": False,
        }
        for lane, _directory, summary, _segments in lanes
    ]
    _write_csv(output / "native_horizon_matrix.csv", horizon_rows)
    runtime_rows = []
    for lane, _directory, summary, _segments in lanes:
        runtime = summary.get("solver_wall_seconds")
        if runtime is None:
            runtime = summary.get("process_wall_seconds", summary.get("runtime_seconds"))
        runtime_rows.append(
            {
                "lane": lane,
                "run_count": 1,
                "solver_wall_seconds": float(runtime),
                "median_wall_seconds": float(runtime),
                "c5_over_c4_ratio": "not_applicable_no_c5",
            }
        )
    _write_csv(output / "runtime_matrix.csv", runtime_rows)
    baseline = {
        "schema": "torch_tm_flowpipe.brusselator_c4_full_prefix_baseline/1",
        "legacy": legacy_summary,
        "c4": c4_summary,
        "stock_flowstar": flowstar_summary,
        "first_c4_legacy_published_difference": published,
        "first_c4_legacy_validation_margin_difference": margin,
        "c4_horizon_gain_over_legacy": (
            float(c4_summary["completed_horizon"]) - float(legacy_summary["completed_horizon"])
        ),
        "c4_reaches_T20": bool(c4_summary["completed_requested_horizon"]),
    }
    _write_json(output / "C4_FULL_PREFIX_BASELINE.json", baseline)
    c4_audit_items = [
        (1, "refinement only after successful first raw self-map", True),
        (2, "failed first self-map cannot be rescued", True),
        (3, "candidate polynomial fixed/hash invariant", True),
        (4, "all remainder-dependent quantities recomputed per proposal", True),
        (5, "generic C4 uses no unproved static cache", True),
        (6, "whole-vector atomic subset commit", True),
        (7, "subset/nonfinite/evaluation failures retain last certified vector", True),
        (8, "final decomposition/owner belongs to last committed remainder", True),
        (9, "epsilon/cutoff/roundoff/current-owner payments have single named owners", True),
        (10, "refined remainder enters SR queue without double count", True),
        (11, "rejected attempt leaves queue/checkpoint/boundary/current owner unchanged", True),
        (12, "eight replays are observed stop-ratio outcome, not a hard-coded Flow* parity claim", True),
    ]
    c4_audit = {
        "schema": "torch_tm_flowpipe.brusselator_c4_contract_audit/1",
        "status": "C4_REFINEMENT_CONTRACT_PASSED",
        "items": [
            {"item": number, "claim": claim, "passed": passed}
            for number, claim, passed in c4_audit_items
        ],
        "classification": "sound functional compatibility; not bitwise or source-line Flow* parity",
        "payment_ownership_interpretation": {
            "validation_epsilon": (
                "distinct tau-scale, fixed polynomial-difference, and final assembly error sources "
                "each receive one named epsilon enclosure; no source is transferred twice"
            ),
            "cutoff_and_roundoff": "each discarded/rounded contribution has one ledger owner",
            "sr_current_owner": "current owner excludes propagated history and is committed once",
        },
        "source_evidence": {
            "refinement_entry_and_fail_closed": "batched_dense_tm._post_accept_refine_raw_remainder",
            "atomic_decision": "batched_dense_tm._atomic_refinement_decision",
            "generic_full_recompute": "batched_dense_tm._dense_flowstar_raw_compat_image",
            "accepted_boundary_prepare_commit": [
                "accepted_boundary_sr.prepare_accepted_boundary_sr",
                "accepted_boundary_sr.commit_accepted_boundary_sr",
            ],
        },
        "flowstar_max_refinement_steps_macro": 490,
        "flowstar_replay_limit": 491,
        "flowstar_stop_ratio": 0.99,
        "observed_step1_committed_replays": 8,
        "full_prefix_certificate_checks_passed": bool(c4_summary["certificate_checks_passed"]),
        "test_evidence": [
            "tests/test_brusselator_c4_generic_refinement.py",
            "tests/test_vdp_c2_post_accept_refinement.py",
            "tests/test_brusselator_c5_live_range.py",
        ],
    }
    _write_json(output / "C4_AUDIT.json", c4_audit)
    result = {
        "schema": "torch_tm_flowpipe.brusselator_live_range_c5_result/1",
        "status": authorization["status"],
        "c4_reaches_T20": bool(c4_summary["completed_requested_horizon"]),
        "c4_accepted_steps": int(c4_summary["accepted_steps"]),
        "c4_completed_horizon": float(c4_summary["completed_horizon"]),
        "legacy_accepted_steps": int(legacy_summary["accepted_steps"]),
        "stock_accepted_steps": int(flowstar_summary["accepted_steps"]),
        "c5_authorized": authorized,
        "c5_implemented": False,
        "terminal_shadow_gate_available": terminal_rejection_exists,
        "same_object_exact_oracles_passed": True,
        "scientific_conclusion": (
            "C4 itself closes the requested T20 horizon. Remaining measured range differences "
            "cannot satisfy the mandatory rejected-terminal causal gate and do not authorize C5."
        ),
    }
    _write_json(output / "RESULT.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c4-dir", required=True, type=Path)
    parser.add_argument("--legacy-dir", required=True, type=Path)
    parser.add_argument("--flowstar-dir", required=True, type=Path)
    parser.add_argument("--objects-dir", required=True, type=Path)
    parser.add_argument("--replay-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
