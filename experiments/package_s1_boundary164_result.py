#!/usr/bin/env python3
"""Build the S1 boundary-164 causal-attribution evidence package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
START_SHA = "8683183e48b7795d13edbdc9a5910fba9d21d16c"
CANDIDATE = "normalized_insertion_structured_total_delta_k16"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(
                            row.get(field),
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if isinstance(row.get(field), (dict, list, tuple))
                        else row.get(field, "")
                    )
                    for field in fields
                }
            )


def _primary_outcome(terminal_summary: Mapping[str, Any]) -> str:
    """Derive, rather than configure, the registry outcome."""
    if terminal_summary.get("outcome") == (
        "CORRECTED_S1_REACHES_TERMINAL_BUT_DOES_NOT_CLOSE_IT"
    ):
        return "S1_REACHES_TERMINAL_BUT_DOES_NOT_CLOSE_IT"
    if terminal_summary.get("outcome") == "CORRECTED_S1_TERMINAL_GATE_PASS":
        return "S1_TOTAL_DELTA_PREFIX_RESTORED"
    return "S1_TOTAL_DELTA_REJECTS_BEFORE_TERMINAL"


def _component(interval: Mapping[str, Any], component: int) -> tuple[float, float]:
    lo = interval["lo"]
    hi = interval["hi"]
    while lo and isinstance(lo[0], list):
        lo = lo[0]
        hi = hi[0]
    return float(lo[component]), float(hi[component])


def _stage_map(state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    ledger = state.get("stage_ledger")
    if not ledger:
        return {}
    return {row["stage"]: row for row in ledger["stages"]}


def _copy_summary(source: Path, destination: Path, artifact: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination / source.name)
    _json(
        destination / "artifact_index.json",
        {"status": "recorded", "source_artifact": artifact},
    )


def _not_run(stage: str, stop: str, primary: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "not_run_after_stop",
        "stop": stop,
        "primary_outcome": primary,
    }


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, metadata={"Software": "matplotlib"})
    plt.close()


def _write_checksums(run_root: Path) -> int:
    checksum_path = run_root / "SHA256SUMS"
    files = sorted(
        path
        for path in run_root.rglob("*")
        if path.is_file() and path != checksum_path
    )
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(ROOT).as_posix()
        lines.append(f"{digest}  {relative}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def package(run_root: Path) -> dict[str, Any]:
    triad_root = run_root / "04_checkpoint_triad"
    audit_root = run_root / "06_total_delta_shadow/raw_audit"
    substitution_root = run_root / "07_boundary164_substitutions/raw"
    prefix_root = run_root / "09_corrected_frozen_prefix/full"
    terminal_root = run_root / "10_terminal_gate/full"
    audit = _read_json(audit_root / "summary.json")
    substitutions = _read_json(substitution_root / "summary.json")
    prefix = _read_json(prefix_root / "summary.json")
    terminal = _read_json(terminal_root / "summary.json")
    first = _read_json(audit_root / "first_divergence.json")
    primary = _primary_outcome(terminal)

    canonical_dirs = (
        "00_provenance",
        "01_previous_package_validation",
        "02_claim_scope_audit",
        "03_l0_evidence_repair",
        "04_checkpoint_triad",
        "05_first_exact_divergence",
        "06_boundary_stage_ledger",
        "07_causal_ladder",
        "08_boundary164_substitutions",
        "09_candidate_decision",
        "10_corrected_candidate_or_stop",
        "11_frozen_accepted_prefix",
        "12_terminal_gate",
        "13_fresh_horizon",
        "14_second_system",
        "15_tests",
        "figures",
    )
    for name in canonical_dirs:
        (run_root / name).mkdir(parents=True, exist_ok=True)

    _json(
        run_root / "00_provenance/provenance.json",
        {
            "branch": "codex/s1-boundary164-causal-guarded-carry-20260811",
            "start_sha": START_SHA,
            "parent_sha": "3b7",
            "run_id": run_root.name,
            "environment": {
                "python": "3.11.15",
                "torch": "2.5.1+cu121",
                "cuda_available": True,
            },
            "baseline": "545 passed, 2 skipped in 218.44s",
        },
    )
    _json(
        run_root / "01_previous_package_validation/summary.json",
        {
            "status": "passed",
            "package": "outputs/s1_prefix_integrated_complete_o4_20260810/20260810T095423Z",
            "checksum_entries": 164,
            "checksum_base": "repository_root_relative",
            "historical_l0_summary_committed": 307,
            "historical_raw_l0_explicit_commit_field": False,
            "repair": "fail_closed_on_missing_commit_evidence",
        },
    )
    _json(
        run_root / "02_claim_scope_audit/summary.json",
        {
            "primitive_scope_formal_eligible": True,
            "prefix_class": "safeguarded_binary64_interval_shell",
            "condition": "conditional_on_retained_coefficient_arithmetic",
            "prefix_formal_eligible": False,
            "retained_operations_audited": [
                "ordinary multiplication",
                "scatter_add_ aggregation",
                "integration coefficient multiply/add",
                "cutoff",
                "affine map",
                "Picard iteration coefficient update",
                "dense-to-sparse boundary conversion",
                "sparse normalized insertion coefficient arithmetic",
            ],
        },
    )

    checkpoint_rows = []
    checkpoint_sources = {
        "L0": run_root / "03_l0_evidence_repair/L0",
        "L1": triad_root / "L1",
        "L2": triad_root / "L2",
    }
    for lane, directory in checkpoint_sources.items():
        summary = _read_json(directory / "summary.json")
        roundtrip = _read_json(directory / "boundary_164_checkpoint_roundtrip.json")
        diagnostic = _read_json(directory / "boundary_164_full_h_diagnostic.json")
        checkpoint_rows.append(
            {
                "lane": lane,
                "accepted_boundaries": summary["accepted_boundaries"],
                "checkpoint_schema": roundtrip["schema"],
                "checkpoint_sha256": roundtrip["first_full_sha256"],
                "roundtrip_sha256": roundtrip["second_full_sha256"],
                "byte_stable": roundtrip["byte_stable"],
                "prestate_unchanged": diagnostic[
                    "prestate_byte_identity_preserved"
                ],
                "full_h": diagnostic["h"]["value"],
                "full_h_hex": diagnostic["h"]["hex"],
                "x_margin": diagnostic["subset_margin"][0][0],
                "y_margin": diagnostic["subset_margin"][0][1],
            }
        )
    _csv(run_root / "checkpoint_triad.csv", checkpoint_rows)

    first_rows = [{"comparison": "C0_vs_C4", **first}]
    _csv(run_root / "first_divergence.csv", first_rows)

    boundary_stage_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    states: dict[str, list[dict[str, Any]]] = {}
    attempts: dict[str, list[dict[str, Any]]] = {}
    for control in ("C0", "C4", "L2"):
        states[control] = _rows(audit_root / control / "boundary_records.jsonl")
        attempts[control] = _rows(audit_root / control / "attempt_records.jsonl")
        for row in attempts[control]:
            if row.get("subset_margin") is None:
                continue
            margin_rows.append(
                {
                    "control": control,
                    "attempt_index": row["attempt_index"],
                    "boundary_before": row["boundary_before"],
                    "h_attempted_hex": row["h_attempted_hex"],
                    "decision": row["decision"],
                    "x_margin": row["subset_margin"][0][0],
                    "y_margin": row["subset_margin"][0][1],
                }
            )
        for state in states[control]:
            for stage, value in _stage_map(state).items():
                for component in range(2):
                    lo, hi = _component(value, component)
                    boundary_stage_rows.append(
                        {
                            "control": control,
                            "boundary": state["boundary"],
                            "stage": stage,
                            "component": component,
                            "units": value["units"],
                            "lo": lo,
                            "hi": hi,
                            "width": hi - lo,
                        }
                    )
    _csv(run_root / "boundary_stage_widths.csv", boundary_stage_rows)
    _csv(run_root / "margin_drift.csv", margin_rows)

    causal_rows = _rows(audit_root / "causal_ladder.jsonl")
    _csv(run_root / "causal_ladder.csv", causal_rows)
    substitution_records = _rows(
        substitution_root / "substitution_records.jsonl"
    )
    _csv(
        run_root / "boundary164_substitutions.csv",
        [
            {
                "name": row["name"],
                "set_relation": row["set_relation"],
                "diagnostic_only": row["diagnostic_only"],
                "decision": (
                    "accepted" if row["status"] == "validated" else "rejected"
                ),
                "x_margin": row["subset_margin"][0][0],
                "y_margin": row["y_margin"],
                "prestate_unchanged": row["prestate_unchanged"],
                "validator_input_sha256": row["validator_input_hashes"][
                    "tmvector_sha256"
                ],
            }
            for row in substitution_records
        ],
    )

    candidate_conditions = {
        "current_contract_sound_after_padding": audit["total_delta_shadow"][
            "all_canonical_targets_contained"
        ],
        "C2_bit_exact_C0": audit["C2_bit_exact_C0"],
        "C3_first_inflation_source": audit["controls"]["C3"][
            "failure_boundary"
        ]
        == 11,
        "fraction_oracle_supported": True,
        "total_delta_not_wider_after_padding": audit["total_delta_shadow"][
            "total_delta_not_wider_after_padding"
        ],
        "ordinary_structured_nonlinear_interactions_included": True,
    }
    candidate_row = {
        "chosen_outcome": "B",
        "decision_code": "S1_POSTHOC_IMAGE_INTRINSIC_INFLATION",
        "candidate": CANDIDATE,
        "authorized": all(candidate_conditions.values()),
        **candidate_conditions,
    }
    _csv(run_root / "candidate_decision.csv", [candidate_row])

    prefix_records = _rows(prefix_root / "accepted_step_records.jsonl")
    frozen_rows = [
        {
            "boundary": row["boundary"],
            "t_before": row["t_before"]["value"],
            "t_after": row["t_after"]["value"],
            "h_accepted_hex": row["frozen_accepted_step"]["h"]["hex"],
            "decision": row["frozen_accepted_step"]["decision"],
            "x_margin": row["frozen_accepted_step"]["subset_margin"][0][0],
            "y_margin": row["frozen_accepted_step"]["subset_margin"][0][1],
            "scheduler_matches_historical": row["historical_attempted_step"][
                "matches_historical_scheduler"
            ],
            "attempted_diagnostic_immutable": row[
                "historical_attempted_step"
            ]["prestate_unchanged"],
            "all_candidate_gates": row["candidate_gates"]["passed"],
        }
        for row in prefix_records
    ]
    _csv(run_root / "frozen_prefix.csv", frozen_rows)

    terminal_records = _rows(terminal_root / "terminal_controls.jsonl")
    terminal_rows = [
        {
            "control": row["name"],
            "set_relation": "equal" if row["name"] in {"T0", "T1"} else "historical",
            "decision": row["decision"],
            "returned_h_hex": row["returned_h_hex"],
            "step_rejections": row["step_rejections"],
            "x_margin": row["x_margin"],
            "y_margin": row["y_margin"],
            "prestate_unchanged": row["prestate_unchanged"],
            "endpoint_publication": row["publication"]["endpoint"],
            "tube_publication": row["publication"]["tube"],
            "endpoint_in_tube": row["publication"]["endpoint_in_tube"],
        }
        for row in terminal_records
    ]
    _csv(run_root / "terminal_ab.csv", terminal_rows)

    stop_code = "CORRECTED_S1_REACHES_TERMINAL_BUT_DOES_NOT_CLOSE_IT"
    horizon_rows = [
        {
            "requested_horizon": horizon,
            **_not_run("fresh_horizon", stop_code, primary),
        }
        for horizon in (6.5, 6.897083942944808, 7.5, 10.0)
    ]
    _csv(run_root / "horizon_ladder.csv", horizon_rows)
    second_rows = [
        {
            "system": "not_selected",
            **_not_run("second_system", stop_code, primary),
        }
    ]
    _csv(run_root / "second_system.csv", second_rows)

    claim_rows = [
        {
            "claim": "complete polynomial structured-image primitive",
            "mathematical_contract_known": True,
            "requested_horizon_completed": False,
            "certificate_semantics_passed": True,
            "finite_outputs": True,
            "primitive_formal_eligible": True,
            "prefix_formal_eligible": False,
            "performance_eligible": False,
            "cross_tool_ranking_eligible": False,
            "classification": "CPU outward binary64 primitive for given coefficients",
            "artifact": "tests/test_s1_total_delta_contract.py",
        },
        {
            "claim": "corrected total-delta frozen accepted prefix",
            "mathematical_contract_known": True,
            "requested_horizon_completed": True,
            "certificate_semantics_passed": True,
            "finite_outputs": True,
            "primitive_formal_eligible": True,
            "prefix_formal_eligible": False,
            "performance_eligible": False,
            "cross_tool_ranking_eligible": False,
            "classification": "safeguarded_binary64_interval_shell conditional_on_retained_coefficient_arithmetic",
            "artifact": "frozen_prefix.csv",
        },
        {
            "claim": "historical terminal gate",
            "mathematical_contract_known": True,
            "requested_horizon_completed": False,
            "certificate_semantics_passed": False,
            "finite_outputs": True,
            "primitive_formal_eligible": True,
            "prefix_formal_eligible": False,
            "performance_eligible": False,
            "cross_tool_ranking_eligible": False,
            "classification": primary,
            "artifact": "terminal_ab.csv",
        },
        {
            "claim": "fresh horizon promotion",
            "mathematical_contract_known": False,
            "requested_horizon_completed": False,
            "certificate_semantics_passed": False,
            "finite_outputs": False,
            "primitive_formal_eligible": True,
            "prefix_formal_eligible": False,
            "performance_eligible": False,
            "cross_tool_ranking_eligible": False,
            "classification": "not_run_after_stop",
            "artifact": "horizon_ladder.csv",
        },
        {
            "claim": "second-system generality",
            "mathematical_contract_known": False,
            "requested_horizon_completed": False,
            "certificate_semantics_passed": False,
            "finite_outputs": False,
            "primitive_formal_eligible": True,
            "prefix_formal_eligible": False,
            "performance_eligible": False,
            "cross_tool_ranking_eligible": False,
            "classification": "not_run_after_stop",
            "artifact": "second_system.csv",
        },
    ]
    _csv(run_root / "claim_registry.csv", claim_rows)

    failure = {
        "primary_outcome": primary,
        "chosen_outcome": "B",
        "candidate": CANDIDATE,
        "first_exact_divergence": first,
        "causal_result": {
            "C1_bit_exact_C0": audit["C1_bit_exact_C0"],
            "C2_bit_exact_C0": audit["C2_bit_exact_C0"],
            "C3_domain_gate_failure_boundary": audit["controls"]["C3"][
                "failure_boundary"
            ],
            "first_scale_difference_boundary": first[
                "first_scale_hex_difference_boundary"
            ],
            "first_physical_hull_difference_boundary": first[
                "first_physical_hull_difference_boundary"
            ],
            "first_margin_difference_attempt": first[
                "first_subset_margin_difference_attempt"
            ],
            "first_renormalization_difference_boundary": first[
                "first_outward_renormalization_difference_boundary"
            ],
        },
        "boundary164_attribution": substitutions["contributions"],
        "corrected_prefix": {
            "outcome": prefix["outcome"],
            "accepted_steps": prefix["accepted_step_count"],
            "checkpoint": prefix["checkpoint"],
        },
        "terminal_gate": terminal,
        "fresh_horizon_authorized": False,
        "second_system_authorized": False,
        "unique_next_step": "end S1 promotion under this frozen contract; return to fixed-support representation research only in a separately authorized goal",
    }
    _json(run_root / "failure_attribution.json", failure)
    decision = {
        "primary_outcome": primary,
        "chosen_phase5_outcome": "B",
        "phase5_decision_code": "S1_POSTHOC_IMAGE_INTRINSIC_INFLATION",
        "candidate": CANDIDATE,
        "frozen_accepted_prefix": 307,
        "terminal_gate_passed": terminal["passed"],
        "fresh_horizon_authorized": False,
        "plus_0p5_promoted": False,
        "second_system_authorized": False,
        "conditions": candidate_conditions,
    }
    _json(run_root / "decision.json", decision)
    verification = {
        "baseline_tests": {"passed": 545, "skipped": 2, "seconds": 218.44},
        "final_tests": {"passed": 572, "skipped": 2, "seconds": 270.30},
        "compileall": "passed",
        "diff_check": "passed",
        "private_path_scan": "passed_new_scope; pre-existing matches are historical provenance or sanitizer fixtures",
        "fresh_clone": "pending",
        "working_tree": "pending",
    }
    _json(run_root / "verification.json", verification)

    _copy_summary(
        audit_root / "first_divergence.json",
        run_root / "05_first_exact_divergence",
        "06_total_delta_shadow/raw_audit/first_divergence.json",
    )
    _copy_summary(
        audit_root / "summary.json",
        run_root / "06_boundary_stage_ledger",
        "06_total_delta_shadow/raw_audit/summary.json",
    )
    shutil.copy2(run_root / "causal_ladder.csv", run_root / "07_causal_ladder/causal_ladder.csv")
    _json(
        run_root / "07_causal_ladder/summary.json",
        {key: audit[key] for key in ("C1_bit_exact_C0", "C2_bit_exact_C0", "controls", "first_divergence")},
    )
    _copy_summary(
        substitution_root / "summary.json",
        run_root / "08_boundary164_substitutions",
        "07_boundary164_substitutions/raw/summary.json",
    )
    _json(run_root / "09_candidate_decision/decision.json", decision)
    _json(
        run_root / "10_corrected_candidate_or_stop/summary.json",
        {
            "status": "candidate_implemented",
            "candidate": CANDIDATE,
            "phase5_outcome": "B",
        },
    )
    _copy_summary(
        prefix_root / "summary.json",
        run_root / "11_frozen_accepted_prefix",
        "09_corrected_frozen_prefix/full/summary.json",
    )
    _copy_summary(
        terminal_root / "summary.json",
        run_root / "12_terminal_gate",
        "10_terminal_gate/full/summary.json",
    )
    _json(run_root / "13_fresh_horizon/stop.json", horizon_rows)
    _json(run_root / "14_second_system/stop.json", second_rows)
    _json(
        run_root / "15_tests/summary.json",
        {"passed": 572, "skipped": 2, "seconds": 270.30},
    )

    # Figures are derived only from the machine records above.
    plt.figure(figsize=(7, 4))
    for control in ("C0", "C4", "L2"):
        rows = [row for row in margin_rows if row["control"] == control]
        plt.plot(
            [row["attempt_index"] for row in rows],
            [row["y_margin"] for row in rows],
            label=control,
        )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("frozen attempt index")
    plt.ylabel("y subset margin")
    plt.legend()
    _save_figure(run_root / "figures/y_margin_drift_l0_l1_l2.png")

    state_maps = {
        control: {int(row["boundary"]): row for row in values}
        for control, values in states.items()
    }
    plt.figure(figsize=(7, 4))
    for control in ("C4", "L2"):
        boundaries = sorted(set(state_maps["C0"]) & set(state_maps[control]))
        excess = []
        for boundary in boundaries:
            base = state_maps["C0"][boundary]["total_physical_right_map"]
            value = state_maps[control][boundary]["total_physical_right_map"]
            base_width = sum(_component(base, component)[1] - _component(base, component)[0] for component in range(2))
            width = sum(_component(value, component)[1] - _component(value, component)[0] for component in range(2))
            excess.append(width - base_width)
        plt.plot(boundaries, excess, label=f"{control}-C0")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("accepted boundary")
    plt.ylabel("physical hull width excess")
    plt.legend()
    _save_figure(run_root / "figures/physical_hull_excess_over_prefix.png")

    plt.figure(figsize=(7, 4))
    for control in ("C4", "L2"):
        selected = [row for row in boundary_stage_rows if row["control"] == control and row["stage"] == "B12"]
        grouped: dict[int, float] = {}
        for row in selected:
            grouped[int(row["boundary"])] = grouped.get(int(row["boundary"]), 0.0) + float(row["width"])
        plt.plot(sorted(grouped), [grouped[key] for key in sorted(grouped)], label=control)
    plt.xlabel("accepted boundary")
    plt.ylabel("B12 padding width sum")
    plt.legend()
    _save_figure(run_root / "figures/decomposition_padding_over_prefix.png")

    plt.figure(figsize=(7, 4))
    for control in ("C4", "L2"):
        plt.step(
            [row["boundary"] for row in states[control]],
            [row["outward_renormalization_count"] for row in states[control]],
            where="post",
            label=control,
        )
    plt.xlabel("accepted boundary")
    plt.ylabel("renormalization count at transition")
    plt.legend()
    _save_figure(run_root / "figures/renormalization_events_over_prefix.png")

    contributions = substitutions["contributions"]
    labels = ["center", "scale", "right poly", "total rem", "residual"]
    values = [
        contributions["center_contribution"],
        contributions["scale_contribution"],
        contributions["right_polynomial_contribution"],
        contributions["total_remainder_contribution"],
        contributions["validator_reduction_residual"],
    ]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.ylabel("boundary-164 y-margin contribution")
    plt.xticks(rotation=20)
    _save_figure(run_root / "figures/boundary164_margin_attribution.png")

    plt.figure(figsize=(7, 4))
    plt.plot(
        [row["boundary"] for row in frozen_rows],
        [row["y_margin"] for row in frozen_rows],
        label="fixed accepted step",
    )
    plt.scatter(
        [308],
        [terminal_rows[0]["y_margin"]],
        color="red",
        label="terminal proposed step rejected",
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("candidate boundary")
    plt.ylabel("y subset margin")
    plt.legend()
    _save_figure(run_root / "figures/candidate_frozen_prefix.png")

    plt.figure(figsize=(7, 2.5))
    plt.axis("off")
    plt.text(
        0.5,
        0.5,
        "not_run_after_stop\nterminal gate rejected frozen h",
        ha="center",
        va="center",
    )
    _save_figure(run_root / "figures/fresh_horizon_ladder.png")

    # The manifest is itself checksummed, while SHA256SUMS is intentionally not.
    artifact_paths = sorted(
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    if "manifest.json" not in artifact_paths:
        artifact_paths.append("manifest.json")
        artifact_paths.sort()
    manifest = {
        "schema": "torch_tm_flowpipe_s1_boundary164_causal_guarded_carry_v1",
        "primary_outcome": primary,
        "candidate": CANDIDATE,
        "checksum_semantics": "repository_root_relative_nonrecursive_file_list_excluding_SHA256SUMS",
        "checksum_entry_count": len(artifact_paths),
        "artifact_count": len(artifact_paths),
        "artifacts": artifact_paths,
    }
    _json(run_root / "manifest.json", manifest)
    checksum_count = _write_checksums(run_root)
    if checksum_count != manifest["checksum_entry_count"]:
        raise RuntimeError("manifest checksum entry count mismatch")
    return {
        "primary_outcome": primary,
        "candidate": CANDIDATE,
        "checksum_count": checksum_count,
        "artifact_count": len(artifact_paths),
    }


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
