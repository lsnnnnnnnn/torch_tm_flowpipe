#!/usr/bin/env python3
"""Build the public evidence tables and figures for the S1 prefix result."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_OUTCOME = "S1_PREFIX_REJECTS_BEFORE_TERMINAL"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field), sort_keys=True, separators=(",", ":"))
                    if isinstance(row.get(field), (dict, list, tuple))
                    else row.get(field, "")
                    for field in fields
                }
            )


def _jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _width(interval: Mapping[str, Any] | None) -> float | None:
    if not interval:
        return None
    lo = interval.get("lo")
    hi = interval.get("hi")
    if not isinstance(lo, list) or not isinstance(hi, list):
        return None
    while lo and isinstance(lo[0], list):
        lo = lo[0]
        hi = hi[0]
    return float(sum(float(right) - float(left) for left, right in zip(lo, hi)))


def _max_component_width(interval: Mapping[str, Any] | None) -> float:
    if not interval:
        return 0.0
    lo = interval["lo"]
    hi = interval["hi"]
    while lo and isinstance(lo[0], list):
        lo = lo[0]
        hi = hi[0]
    return max((float(right) - float(left) for left, right in zip(lo, hi)), default=0.0)


def _copy_checkpoint(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("terminal_state.json", "terminal_state_manifest.json"):
        shutil.copy2(source / name, destination / name)


def _not_run_row(stage: str, stop: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "not_run_after_stop",
        "primary_outcome": PRIMARY_OUTCOME,
        "stop_boundary": stop["accepted_boundaries"],
        "stop_time": stop["stop_time"],
        "reason": "L2 rejected a frozen step that the historical baseline accepted",
    }


def package(run_root: Path) -> dict[str, Any]:
    prefix = run_root / "04_frozen_schedule_prefix"
    lane_paths = {
        "L0": prefix / "L0_historical_baseline",
        "L1": prefix / "L1_materialize_every_boundary_final",
        "L2": prefix / "L2_structured_k16_final_checkpointed",
    }
    summaries = {lane: _read_json(path / "summary.json") for lane, path in lane_paths.items()}
    schedule = _read_json(prefix / "frozen_schedule.json")
    divergence = _read_json(lane_paths["L2"] / "divergence_replay.json")
    stop_boundary = int(summaries["L2"]["accepted_boundaries"])
    stop_time = float(schedule["rows"][stop_boundary - 1]["t_after"]["value"])
    stop = {"accepted_boundaries": stop_boundary, "stop_time": stop_time}

    for directory in (
        "05_prefix_checkpoints",
        "06_terminal_same_pre_state_ab",
        "07_capacity_attribution",
        "08_fresh_horizon_ladder",
        "09_second_system",
        "10_regression_tests",
        "figures",
    ):
        (run_root / directory).mkdir(parents=True, exist_ok=True)

    checkpoint_source = lane_paths["L2"] / "final_common_prefix_checkpoint_v2"
    roundtrip_source = lane_paths["L2"] / "final_common_prefix_checkpoint_v2_roundtrip"
    _copy_checkpoint(checkpoint_source, run_root / "05_prefix_checkpoints" / "boundary_164_v2")
    _copy_checkpoint(roundtrip_source, run_root / "05_prefix_checkpoints" / "boundary_164_v2_roundtrip")
    shutil.copy2(lane_paths["L2"] / "checkpoint_roundtrip.json", run_root / "05_prefix_checkpoints" / "checkpoint_roundtrip.json")
    shutil.copy2(lane_paths["L2"] / "divergence_replay.json", run_root / "05_prefix_checkpoints" / "divergence_replay.json")

    conservation_rows: list[dict[str, Any]] = []
    hash_rows: list[dict[str, Any]] = []
    lane_raw: dict[str, list[dict[str, Any]]] = {}
    for lane, lane_path in lane_paths.items():
        raw = _rows(lane_path / "prefix_conservation.jsonl")
        lane_raw[lane] = raw
        for row in raw:
            conservation_rows.append(
                {
                    "lane": lane,
                    "attempt_index": row["attempt_index"],
                    "accepted_boundary_index_before": row["accepted_boundary_index_before"],
                    "accepted_boundary_index_after": row.get("accepted_boundary_index_after", ""),
                    "t_before_hex": row["t_before_hex"],
                    "h_attempted_hex": row["h_attempted_hex"],
                    "h_actual_hex": row["h_actual_hex"],
                    "expected_status": row["expected_status"],
                    "actual_status": row["actual_status"],
                    "actual_rejections": row["actual_rejections"],
                    "committed_to_frozen_prefix": row.get("committed_to_frozen_prefix", False),
                    "schedule_match": row["schedule_match"],
                    "ordinary_width": _width(row.get("ordinary_post")),
                    "materialized_total_width": _width(row.get("materialized_structured_post")),
                    "structured_incremental_width": (
                        max(0.0, (_width(row.get("materialized_structured_post")) or 0.0) - (_width(row.get("ordinary_post")) or 0.0))
                        if row.get("ordinary_post")
                        else None
                    ),
                    "nonlinear_residual_width": _width(row.get("nonlinear_structured_residual")),
                    "evicted_width": _width(row.get("evicted_contribution")),
                    "endpoint_total_width": _width(row.get("published_endpoint_total")),
                    "tube_total_width": _width(row.get("published_tube_total")),
                    "active_columns_after": row.get("active_columns_after", ""),
                    "event_count_after": row.get("event_count_after", ""),
                    "conservation_mask": row.get("conservation_mask", ""),
                    "source_decomposition_mask": row.get("source_decomposition_mask", ""),
                    "no_double_count_mask": row.get("no_double_count_mask", ""),
                    "finite_mask": row.get("finite_mask", ""),
                    "endpoint_publication_mask": row.get("endpoint_publication_mask", ""),
                    "tube_publication_mask": row.get("tube_publication_mask", ""),
                    "accepted_mask": row.get("accepted_mask", ""),
                }
            )
            hash_rows.append(
                {
                    "lane": lane,
                    "attempt_index": row["attempt_index"],
                    "committed_to_frozen_prefix": row.get("committed_to_frozen_prefix", False),
                    "prestate_sha256": row["prestate_sha256"],
                    "poststate_sha256": row["poststate_sha256"],
                }
            )
    _csv(run_root / "prefix_conservation.csv", conservation_rows)
    _csv(run_root / "prefix_state_hashes.csv", hash_rows)

    source_events = _rows(lane_paths["L2"] / "prefix_source_events.jsonl")
    _jsonl(run_root / "prefix_source_events.jsonl", source_events)
    capacity_rows: list[dict[str, Any]] = []
    cumulative_width = 0.0
    for event_index, event in enumerate(source_events):
        if event["reason"] != "capacity_eviction" or not any(event["active_mask"]):
            continue
        width = _max_component_width(event["materialized"])
        cumulative_width += width
        capacity_rows.append(
            {
                "record_type": "capacity_eviction",
                "event_index": event_index,
                "boundary": event["boundary"],
                "slot": event["slot"],
                "source_category": event["source_category"],
                "source_category_id": event["source_category_id"],
                "source_boundary_index": event["source_boundary_index"],
                "source_occurrence_index": event["source_occurrence_index"],
                "age": event["age"],
                "materialized_width_max_component": width,
                "cumulative_materialized_width_max_components": cumulative_width,
                "status": "observed_on_sound_common_prefix",
            }
        )
    capacity_rows.append(
        {
            "record_type": "terminal_attribution",
            "event_index": "",
            "boundary": "",
            "materialized_width_max_component": "",
            "cumulative_materialized_width_max_components": cumulative_width,
            "status": "not_run_after_stop",
        }
    )
    _csv(run_root / "capacity_attribution.csv", capacity_rows)
    shutil.copy2(run_root / "capacity_attribution.csv", run_root / "07_capacity_attribution" / "capacity_attribution.csv")
    _json(
        run_root / "07_capacity_attribution" / "decision.json",
        {
            "supplemental_outcome": "K32_NOT_AUTHORIZED_BY_EVICTION_ATTRIBUTION",
            "reason": "terminal GO gates were not reached, independent of observed eviction fraction",
            "first_full_k16_boundary": summaries["L2"]["first_full_k16_boundary"],
            "first_eviction_boundary": summaries["L2"]["first_eviction_boundary"],
            "largest_eviction": summaries["L2"]["largest_eviction"],
            "eviction_event_count": len(capacity_rows) - 1,
        },
    )

    terminal_ab = {
        **_not_run_row("terminal_same_pre_state_ab", stop),
        "historical_terminal_time": 6.397083942944808,
        "historical_terminal_h": 0.003623635847674574,
        "checkpoint_available_at_historical_terminal": False,
    }
    terminal_gate = {
        **_not_run_row("terminal_gate", stop),
        "authorized": False,
        "passed": False,
        "primary_outcome": PRIMARY_OUTCOME,
    }
    _json(run_root / "terminal_ab.json", terminal_ab)
    _json(run_root / "terminal_gate.json", terminal_gate)
    _json(run_root / "06_terminal_same_pre_state_ab" / "terminal_ab.json", terminal_ab)
    _json(run_root / "06_terminal_same_pre_state_ab" / "terminal_gate.json", terminal_gate)

    horizon_rows = [
        {
            "lane": "L0_historical_baseline",
            "experiment_scope": "frozen_schedule",
            "status": "completed",
            "validated_horizon": schedule["rows"][307]["t_before"]["value"],
            "accepted_boundaries": summaries["L0"]["accepted_boundaries"],
            "primary_outcome": summaries["L0"]["outcome"],
        },
        {
            "lane": "L1_materialize_every_boundary",
            "experiment_scope": "frozen_schedule_common_prefix",
            "status": "stopped_on_first_divergence",
            "validated_horizon": stop_time,
            "accepted_boundaries": summaries["L1"]["accepted_boundaries"],
            "primary_outcome": PRIMARY_OUTCOME,
        },
        {
            "lane": "L2_structured_k16",
            "experiment_scope": "frozen_schedule_common_prefix",
            "status": "stopped_on_first_divergence",
            "validated_horizon": stop_time,
            "accepted_boundaries": summaries["L2"]["accepted_boundaries"],
            "primary_outcome": PRIMARY_OUTCOME,
        },
        {**_not_run_row("fresh_adaptive_horizon_ladder", stop), "lane": "baseline_vs_s1", "experiment_scope": "fresh_horizon", "validated_horizon": "", "accepted_boundaries": ""},
    ]
    _csv(run_root / "horizon_ladder.csv", horizon_rows)
    shutil.copy2(run_root / "horizon_ladder.csv", run_root / "08_fresh_horizon_ladder" / "horizon_ladder.csv")
    common_time_rows = [{**_not_run_row("fresh_common_time_tightness", stop), "time": "", "endpoint_width": "", "tube_width": ""}]
    _csv(run_root / "common_time_tightness.csv", common_time_rows)
    shutil.copy2(run_root / "common_time_tightness.csv", run_root / "08_fresh_horizon_ladder" / "common_time_tightness.csv")
    second_rows = [{**_not_run_row("integrated_s1_second_system", stop), "system": "not_selected", "validated_horizon": ""}]
    _csv(run_root / "second_system.csv", second_rows)
    shutil.copy2(run_root / "second_system.csv", run_root / "09_second_system" / "second_system.csv")

    failure = {
        "primary_outcome": PRIMARY_OUTCOME,
        "longest_frozen_schedule_common_prefix_boundary": stop_boundary,
        "longest_frozen_schedule_common_prefix_time": stop_time,
        "first_failed_obligation": "accept the same frozen proposed step as the historical complete-O4 baseline",
        "attempt_index": 164,
        "historical_baseline": {
            "decision": "accepted",
            "h": divergence["historical_h_accepted"],
            "rejections": divergence["historical_rejections"],
        },
        "L2": {
            "decision_at_frozen_h": divergence["frozen_proposed_step_decision"],
            "returned_h_after_internal_shrink": divergence["s1_returned_h"],
            "rejections": divergence["s1_rejections"],
            "first_failed_diagnostic": divergence["first_failed_diagnostic"],
            "prestate_checkpoint_full_sha256": divergence["checkpoint_full_sha256"],
            "active_columns": divergence["active_columns"],
            "event_count": divergence["event_count"],
            "off_schedule_poststate_published": False,
        },
        "fresh_horizon_authorized": False,
        "terminal_ab_authorized": False,
        "next_action": "retain the sound common-prefix result and revisit the S1 representation decision; do not tune horizons or K",
    }
    _json(run_root / "failure_attribution.json", failure)

    claims = [
        ("previous evidence package", True, True, True, True, True, False, False, "mixed_historical", "checksum-verified previous package", "historical", "01_previous_package_validation/baseline_results.txt"),
        ("typed additive ledger", True, True, True, True, True, False, False, "outward_binary64", "canonical tensor-native additive source schema", "passed", "03_typed_ledger_fixtures/focused_results.json"),
        ("complete-O4 sensitivity oracle", True, True, True, True, True, False, False, "independent_fraction_oracle", "degree<=4 endpoint and tube", "passed", "02_coupling_contract_oracles/oracle_results.json"),
        ("S1 prefix conservation", False, True, True, True, True, False, False, "outward_binary64", f"sound frozen common prefix through boundary {stop_boundary}", PRIMARY_OUTCOME, "prefix_conservation.csv"),
        ("S1 checkpoint v2", True, True, True, True, True, False, False, "exact_binary64_hex", f"boundary {stop_boundary} complete prefix state", "passed", "05_prefix_checkpoints/checkpoint_roundtrip.json"),
        ("terminal same-pre-state A/B", False, False, True, False, False, False, False, "not_run_after_stop", "historical terminal", PRIMARY_OUTCOME, "terminal_ab.json"),
        ("fresh S1 horizon", False, False, True, False, False, False, False, "not_run_after_stop", "fresh adaptive horizon", PRIMARY_OUTCOME, "horizon_ladder.csv"),
        ("integrated S1 second system", False, False, True, False, False, False, False, "not_run_after_stop", "integrated second system", PRIMARY_OUTCOME, "second_system.csv"),
        ("fixed compiled historical", True, False, False, True, False, False, False, "semantics_changed_historical", "fixed-support compiled observation", "FIXED_SUPPORT_COMPILE_SEMANTICS_CHANGED_UNCHANGED", "00_provenance/previous_claim_registry.csv"),
        ("fixed outward historical", False, False, True, True, False, False, False, "fail_closed_historical", "fixed-support outward multi-step", "FIXED_SUPPORT_FORMAL_SOUNDNESS_NOT_CLOSED_UNCHANGED", "00_provenance/previous_claim_registry.csv"),
    ]
    claim_rows = [
        {
            "claim": row[0],
            "completed": row[1],
            "certificate_semantics_passed": row[2],
            "mathematical_contract_known": row[3],
            "finite_outputs": row[4],
            "formal_claim_eligible": row[5],
            "performance_measurement_eligible": row[6],
            "cross_tool_ranking_eligible": row[7],
            "numerical_soundness_class": row[8],
            "numerical_soundness_scope": row[9],
            "requested_horizon_completed": row[1] if row[0] not in {"S1 prefix conservation"} else False,
            "outcome": row[10],
            "evidence_path": row[11],
        }
        for row in claims
    ]
    _csv(run_root / "claim_registry.csv", claim_rows)

    provenance = {
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "HEAD"),
        "start_sha": "3b7b6ef97d9a33dea8498b7595131ffc6095bc1f",
        "primary_outcome": PRIMARY_OUTCOME,
        "authoritative_schedule_source": schedule["source_artifact"],
        "authoritative_schedule_source_sha256": schedule["source_artifact_sha256"],
        "frozen_schedule_sha256": hashlib.sha256((prefix / "frozen_schedule.json").read_bytes()).hexdigest(),
        "final_common_prefix_checkpoint_full_sha256": divergence["checkpoint_full_sha256"],
        "torch_dtype": "float64",
        "device": "cpu",
    }
    _json(run_root / "provenance.json", provenance)
    verification = {
        "primary_outcome": PRIMARY_OUTCOME,
        "L0_schedule_exact": summaries["L0"]["first_schedule_or_decision_divergence"] is None,
        "L1_final_common_prefix_boundary": summaries["L1"]["final_common_prefix_boundary"],
        "L2_final_common_prefix_boundary": summaries["L2"]["final_common_prefix_boundary"],
        "all_committed_L2_conservation_gates": all(
            all(bool(row.get(field)) for field in ("conservation_mask", "source_decomposition_mask", "no_double_count_mask", "finite_mask", "endpoint_publication_mask", "tube_publication_mask", "accepted_mask"))
            for row in lane_raw["L2"]
            if row.get("committed_to_frozen_prefix")
        ),
        "checkpoint_v2_byte_stable": _read_json(lane_paths["L2"] / "checkpoint_roundtrip.json")["byte_stable"],
        "terminal_ab_status": "not_run_after_stop",
        "fresh_horizon_status": "not_run_after_stop",
        "second_system_status": "not_run_after_stop",
        "focused_tests": "pending_final_run",
        "full_tests": "pending_final_run",
        "fresh_clone": "pending",
        "checksums": "pending",
    }
    _json(run_root / "verification.json", verification)

    _figures(run_root, conservation_rows, capacity_rows, terminal_gate, common_time_rows, horizon_rows)

    required = [
        "provenance.json", "coupling_contract.json", "source_schema.json",
        "prefix_conservation.csv", "prefix_source_events.jsonl", "prefix_state_hashes.csv",
        "terminal_ab.json", "terminal_gate.json", "capacity_attribution.csv", "horizon_ladder.csv",
        "common_time_tightness.csv", "second_system.csv", "claim_registry.csv",
        "failure_attribution.json", "verification.json",
    ]
    figures = [
        "ordinary_vs_structured_width_over_prefix.png",
        "active_columns_and_evictions_over_prefix.png",
        "nonlinear_residual_over_prefix.png",
        "terminal_same_pre_state_margins.png",
        "common_time_endpoint_widths.png",
        "common_time_tube_widths.png",
        "validated_horizon_ladder.png",
    ]
    manifest = {
        "schema": "s1_prefix_integrated_complete_o4_evidence_v1",
        "primary_outcome": PRIMARY_OUTCOME,
        "required_files": required,
        "required_figures": [f"figures/{name}" for name in figures],
        "required_directories": [f"{index:02d}_{name}" for index, name in enumerate((
            "provenance", "previous_package_validation", "coupling_contract_oracles", "typed_ledger_fixtures",
            "frozen_schedule_prefix", "prefix_checkpoints", "terminal_same_pre_state_ab", "capacity_attribution",
            "fresh_horizon_ladder", "second_system", "regression_tests",
        ))],
        "report_paths": [
            "docs/S1_COMPLETE_O4_COUPLING_CONTRACT_20260810.md",
            "docs/S1_PREFIX_INTEGRATION_RESULT_20260810.md",
            "docs/S1_TERMINAL_CAUSAL_GATE_20260810.md",
            "docs/S1_FRESH_HORIZON_RESULT_20260810.md",
            "handoff.md",
        ],
    }
    _json(run_root / "manifest.json", manifest)
    return {"run_root": str(run_root.relative_to(ROOT)), **verification}


def _empty_figure(path: Path, title: str, status: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, status, ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _figures(
    run_root: Path,
    conservation_rows: Sequence[Mapping[str, Any]],
    capacity_rows: Sequence[Mapping[str, Any]],
    terminal_gate: Mapping[str, Any],
    common_time_rows: Sequence[Mapping[str, Any]],
    horizon_rows: Sequence[Mapping[str, Any]],
) -> None:
    directory = run_root / "figures"
    l2 = [row for row in conservation_rows if row["lane"] == "L2" and row["committed_to_frozen_prefix"]]
    x = [int(row["accepted_boundary_index_after"]) for row in l2]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(x, [row["ordinary_width"] for row in l2], label="ordinary normalized width")
    ax.plot(x, [row["structured_incremental_width"] for row in l2], label="live structured incremental width")
    ax.set(xlabel="accepted boundary", ylabel="sum of normalized interval widths", title="Ordinary and structured width on frozen common prefix")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "ordinary_vs_structured_width_over_prefix.png", dpi=160)
    plt.close(fig)

    evictions_by_boundary: dict[int, int] = {}
    for row in capacity_rows:
        if row.get("record_type") == "capacity_eviction":
            boundary = int(row["boundary"])
            evictions_by_boundary[boundary] = evictions_by_boundary.get(boundary, 0) + 1
    cumulative = 0
    cumulative_evictions = []
    for boundary in x:
        cumulative += evictions_by_boundary.get(boundary, 0)
        cumulative_evictions.append(cumulative)
    fig, first = plt.subplots(figsize=(7.5, 4.4))
    first.plot(x, [int(row["active_columns_after"]) for row in l2], color="tab:blue", label="active columns")
    first.set(xlabel="accepted boundary", ylabel="active K16 columns", title="K16 occupancy and eviction events")
    second = first.twinx()
    second.plot(x, cumulative_evictions, color="tab:red", label="cumulative evictions")
    second.set_ylabel("cumulative eviction events")
    fig.tight_layout()
    fig.savefig(directory / "active_columns_and_evictions_over_prefix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(x, [row["nonlinear_residual_width"] for row in l2])
    ax.set(xlabel="accepted boundary", ylabel="sum of normalized interval widths", title="Complete-O4 structured nonlinear residual")
    fig.tight_layout()
    fig.savefig(directory / "nonlinear_residual_over_prefix.png", dpi=160)
    plt.close(fig)

    _empty_figure(
        directory / "terminal_same_pre_state_margins.png",
        "Terminal same-pre-state margins",
        str(terminal_gate["status"]),
    )
    common_status = str(common_time_rows[0]["status"])
    _empty_figure(directory / "common_time_endpoint_widths.png", "Fresh common-time endpoint widths", common_status)
    _empty_figure(directory / "common_time_tube_widths.png", "Fresh common-time tube widths", common_status)

    observed = [row for row in horizon_rows if isinstance(row.get("validated_horizon"), (int, float))]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar([row["lane"] for row in observed], [row["validated_horizon"] for row in observed])
    ax.set(ylabel="validated time", title="Frozen-schedule validated horizons (fresh ladder not run)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(directory / "validated_horizon_ladder.png", dpi=160)
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = package(args.run_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
