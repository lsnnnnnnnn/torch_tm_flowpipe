#!/usr/bin/env python3
"""Finalize the fail-closed Huan proof/VDP evidence package.

This tool does not run a scientific kernel.  It consolidates already captured
Phase-D and Phase-E evidence, records the mandatory portability stop, and
builds provenance/checksum files for the artifact-only verifier.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence


HUAN_HEAD = "743f6205e6408072193ad76e940e7f15030e8d3c"
FLOWSTAR_HEAD = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
TORCH_C2_SCIENTIFIC = "29c9ee8f1fe96b860052b86a2b37d79a37bbb2ca"
TORCH_C2_PACKAGE = "0fea2657b30aea5f8cfe326dbcd06d659b8dd26c"
PRIMARY = "HUAN_PROOF_CONTRACT_CLOSED__VDP_CONTRACT_NOT_PORTABLE"
STOP = "NOT_RUN_AFTER_CONTRACT_PORTABILITY_STOP"
THROUGHPUT = "NOT_RUN_THIS_ROUND_AFTER_PROOF_AND_VDP_SCOPE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for raw in rows:
            writer.writerow(
                {
                    key: json.dumps(raw.get(key), sort_keys=True)
                    if isinstance(raw.get(key), (dict, list))
                    else raw.get(key, "")
                    for key in fields
                }
            )


def _command(command: Sequence[str], cwd: Path | None = None) -> dict[str, Any]:
    result = subprocess.run(
        list(command), cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    return {
        "command": list(command),
        "cwd": str(cwd.resolve()) if cwd else None,
        "returncode": result.returncode,
        "stdout": result.stdout,
    }


def _git_state(root: Path, expected: str, role: str) -> dict[str, Any]:
    head = _command(("git", "rev-parse", "HEAD"), root)
    status = _command(("git", "status", "--porcelain", "--untracked-files=no"), root)
    branch = _command(("git", "branch", "--show-current"), root)
    if head["returncode"] or status["returncode"]:
        raise RuntimeError(f"cannot inspect {role} repository")
    actual = head["stdout"].strip()
    if actual != expected:
        raise RuntimeError(f"{role} SHA mismatch: {actual} != {expected}")
    if status["stdout"]:
        raise RuntimeError(f"{role} tracked source is dirty")
    return {
        "role": role,
        "root": str(root.resolve()),
        "head": actual,
        "expected_head": expected,
        "branch": branch["stdout"].strip(),
        "tracked_clean": True,
    }


def _contract_rows() -> list[dict[str, str]]:
    sources = {
        "stock_flowstar": FLOWSTAR_HEAD,
        "torch_c2": TORCH_C2_SCIENTIFIC,
        "huan_parity": HUAN_HEAD,
        "huan_strict": HUAN_HEAD,
    }
    common = (
        ("ode", "x'=y; y'=y-x-x^2*y", "IDENTICAL"),
        ("initial_set", "x=[1.1,1.4]; y=[2.35,2.45]", "SEMANTICALLY_EQUIVALENT_BINARY64_ENDPOINTS"),
        ("taylor_order", "complete total-degree O4", "IDENTICAL"),
        ("fixed_step", "0.01", "IDENTICAL"),
        ("ordinary_remainder", "[-1e-4,1e-4] per component", "IDENTICAL"),
        ("cutoff", "1e-10", "IDENTICAL"),
        ("validation_epsilon", "1e-12", "SEMANTICALLY_EQUIVALENT_LOOP_THRESHOLD"),
        ("native_h_min", "0.002", "IDENTICAL_WHEN_CONFIGURABLE"),
        ("native_h_max", "0.1", "IDENTICAL_WHEN_CONFIGURABLE"),
        ("symbolic_remainder_queue", "100", "IDENTICAL"),
    )
    rows: list[dict[str, str]] = []
    for lane, source in sources.items():
        for setting, required, comparison in common:
            executed = lane.startswith("huan_")
            status = "EXECUTED_FIXED_CONTRACT"
            note = "fresh CUDA fixed-step run"
            if not executed:
                status = STOP
                note = "source inspected; run forbidden after Huan portability stop"
            if lane.startswith("huan_") and setting in {"native_h_min", "native_h_max"}:
                comparison = "UNREPRESENTABLE_WITH_SYMBOLIC_QUEUE_100"
                status = "NOT_RUN_CONTRACT_NOT_PORTABLE"
                note = "Settings rejects adaptive step with symbolic remainder"
            rows.append(
                {
                    "lane": lane,
                    "source_sha": source,
                    "setting": setting,
                    "required_value": required,
                    "actual_or_declared_value": required,
                    "comparison": comparison,
                    "execution_status": status,
                    "evidence_note": note,
                }
            )
    return rows


def _compact_fixed(row: Mapping[str, Any]) -> dict[str, Any]:
    channels = row["channels"]
    return {
        "lane": "huan_" + str(row["mode"]),
        "source_sha": row["source_sha"],
        "scenario": row["scenario"],
        "execution_status": "EXECUTED",
        "completed_requested_horizon": row["completed_requested_horizon"],
        "requested_horizon": row["requested_horizon"],
        "completed_horizon": row["completed_horizon"],
        "accepted_steps": row["accepted_steps"],
        "rejected_attempts": row["rejected_attempts"],
        "refinement_iterations": row["refinement_iterations"],
        "status_code": row["status_code"],
        "endpoint_x": channels["endpoint"][0],
        "endpoint_y": channels["endpoint"][1],
        "endpoint_x_width": channels["endpoint_width"][0],
        "endpoint_y_width": channels["endpoint_width"][1],
        "segment_tube_x": channels["segment_tube"][0],
        "segment_tube_y": channels["segment_tube"][1],
        "segment_tube_x_width": channels["segment_tube_width"][0],
        "segment_tube_y_width": channels["segment_tube_width"][1],
        "runtime_s": row["runtime_s"],
        "peak_gpu_memory_bytes": row["peak_gpu_memory_bytes"],
        "retained_candidate_polynomial_sha256": row[
            "retained_candidate_polynomial_sha256"
        ],
        "retained_candidate_coefficients": None,
        "retained_candidate_coefficients_status": (
            "HASH_CAPTURED__COEFFICIENTS_NOT_SERIALIZED_BEFORE_PORTABILITY_STOP"
        ),
        "first_self_map": row["first_self_map"],
        "refinement_ledger_path": "vdp/refinement_ledgers.jsonl.gz",
        "ordinary_remainder_final": row["step1_remainder"],
        "ordinary_remainder_decomposition_status": (
            "AGGREGATE_FINAL_REMAINDER_ONLY__CATEGORY_TOTALS_NOT_EXPOSED"
        ),
        "symbolic_queue_capacity": row["settings"]["symbolic_remainder_queue"],
        "symbolic_queue_state": None,
        "symbolic_queue_state_status": (
            "ENABLED__INTERNAL_QUEUE_SNAPSHOT_NOT_EXPOSED"
        ),
        "cutoff_threshold": row["settings"]["cutoff"],
        "cutoff_contribution": None,
        "cutoff_contribution_status": "NOT_SEPARATELY_EXPOSED",
        "roundoff_contribution_ledger": row["roundoff_contribution_ledger"],
        "final_accepted_remainder": row["step1_remainder"],
        "not_run_reason": "",
    }


def _stopped_fixed(lane: str, sha: str, scenario: str, horizon: float) -> dict[str, Any]:
    return {
        "lane": lane,
        "source_sha": sha,
        "scenario": scenario,
        "execution_status": STOP,
        "completed_requested_horizon": False,
        "requested_horizon": horizon,
        "completed_horizon": None,
        "accepted_steps": None,
        "rejected_attempts": None,
        "refinement_iterations": None,
        "status_code": None,
        "endpoint_x": None,
        "endpoint_y": None,
        "endpoint_x_width": None,
        "endpoint_y_width": None,
        "segment_tube_x": None,
        "segment_tube_y": None,
        "segment_tube_x_width": None,
        "segment_tube_y_width": None,
        "runtime_s": None,
        "peak_gpu_memory_bytes": None,
        "retained_candidate_polynomial_sha256": None,
        "retained_candidate_coefficients": None,
        "retained_candidate_coefficients_status": STOP,
        "first_self_map": None,
        "refinement_ledger_path": None,
        "ordinary_remainder_final": None,
        "ordinary_remainder_decomposition_status": STOP,
        "symbolic_queue_capacity": None,
        "symbolic_queue_state": None,
        "symbolic_queue_state_status": STOP,
        "cutoff_threshold": None,
        "cutoff_contribution": None,
        "cutoff_contribution_status": STOP,
        "roundoff_contribution_ledger": None,
        "final_accepted_remainder": None,
        "not_run_reason": "mandatory Huan adaptive+SR100 contract-portability stop occurred before this lane",
    }


def _write_phase_e(output: Path) -> None:
    huan = _json(output / "vdp/huan_final/run_index.json")
    if huan.get("engine_head") != HUAN_HEAD or huan.get("primary_status") != PRIMARY:
        raise RuntimeError("Huan Phase-E index does not carry the required stop result")
    fixed = [_compact_fixed(row) for row in huan["fixed_runs"]]
    horizons = {"step1": 0.01, "fixed_T1": 1.0, "fixed_T3": 3.0, "fixed_T6p32": 6.32}
    for lane, sha in (("stock_flowstar", FLOWSTAR_HEAD), ("torch_c2", TORCH_C2_SCIENTIFIC)):
        fixed.extend(_stopped_fixed(lane, sha, scenario, horizon) for scenario, horizon in horizons.items())
    fixed.sort(key=lambda row: (row["scenario"], row["lane"]))
    fields = (
        "lane", "source_sha", "scenario", "execution_status",
        "completed_requested_horizon", "requested_horizon", "completed_horizon",
        "accepted_steps", "rejected_attempts", "refinement_iterations", "status_code",
        "endpoint_x", "endpoint_y", "endpoint_x_width", "endpoint_y_width",
        "segment_tube_x", "segment_tube_y", "segment_tube_x_width",
        "segment_tube_y_width", "runtime_s", "peak_gpu_memory_bytes", "not_run_reason",
        "retained_candidate_polynomial_sha256", "retained_candidate_coefficients",
        "retained_candidate_coefficients_status", "first_self_map",
        "refinement_ledger_path", "ordinary_remainder_final",
        "ordinary_remainder_decomposition_status", "symbolic_queue_capacity",
        "symbolic_queue_state", "symbolic_queue_state_status", "cutoff_threshold",
        "cutoff_contribution", "cutoff_contribution_status",
        "roundoff_contribution_ledger", "final_accepted_remainder",
    )
    _write_csv(output / "vdp/fixed_horizon_matrix.csv", fixed, fields)
    _write_csv(
        output / "vdp/step1_common_input.csv",
        [row for row in fixed if row["scenario"] == "step1"],
        fields,
    )
    shutil.copyfile(
        output / "vdp/huan_final/refinement_ledgers.jsonl.gz",
        output / "vdp/refinement_ledgers.jsonl.gz",
    )
    native = {
        "schema": "torch_tm_flowpipe.huan_vdp_native_terminal/2",
        "primary_status": PRIMARY,
        "stock_flowstar": {
            "source_sha": FLOWSTAR_HEAD,
            "status": STOP,
            "reason": "not launched after mandatory Huan contract-portability stop",
        },
        "torch_c2": {
            "scientific_sha": TORCH_C2_SCIENTIFIC,
            "package_tip": TORCH_C2_PACKAGE,
            "status": STOP,
            "reason": "not launched after mandatory Huan contract-portability stop",
        },
        "huan_parity": huan["native"]["parity"],
        "huan_strict": huan["native"]["strict"],
    }
    _write_json(output / "vdp/native_terminal.json", native)
    _write_json(
        output / "vdp/first_divergence.json",
        {
            "schema": "torch_tm_flowpipe.huan_vdp_first_divergence/1",
            "status": "NOT_ADJUDICATED_CONTRACT_PORTABILITY_STOP",
            "huan_vs_flowstar": None,
            "huan_vs_torch_c2": None,
            "torch_terminal_y_upper_cause_in_huan": "NOT_ADJUDICATED",
            "reason": (
                "The exact Huan native contract is unrepresentable with symbolic remainder "
                "queue 100; cross-tool ranking and first-divergence claims stopped before "
                "fresh Flow*/Torch lanes, as required by the preregistered stop-loss rule."
            ),
            "within_huan_first_observable_strict_parity_difference": {
                "scenario": "step1",
                "stage": "accepted endpoint/tube enclosure",
                "interpretation": "strict roundoff charging widens the observable at step 1",
            },
        },
    )
    _write_csv(
        output / "vdp/contract_matrix.csv",
        _contract_rows(),
        (
            "lane", "source_sha", "setting", "required_value",
            "actual_or_declared_value", "comparison", "execution_status", "evidence_note",
        ),
    )
    _write_json(
        output / "vdp/phase_e_decision.json",
        {
            "schema": "torch_tm_flowpipe.huan_vdp_phase_e_decision/1",
            "primary_status": PRIMARY,
            "phase_d_passed": True,
            "huan_fixed_runs_executed": 8,
            "huan_fixed_runs_completed": 8,
            "native_parity": "NOT_RUN_CONTRACT_NOT_PORTABLE",
            "native_strict": "NOT_RUN_CONTRACT_NOT_PORTABLE",
            "flowstar_fresh_lane": STOP,
            "torch_c2_fresh_lane": STOP,
            "throughput_phase": THROUGHPUT,
            "contract_was_changed": False,
        },
    )
    _write_json(
        output / "vdp/superseded_runs.json",
        {
            "schema": "torch_tm_flowpipe.huan_vdp_superseded_runs/1",
            "authoritative_path": "vdp/huan_final",
            "authoritative_huan_sha": HUAN_HEAD,
            "superseded": [
                {
                    "path": "vdp/huan",
                    "huan_sha": "b0ff55745d69205f3afb4dc8077b9ac1310bfff3",
                    "reason": (
                        "pre-final full-suite run; later source commit removed two "
                        "coverage-only pragma annotations without numerical changes"
                    ),
                    "eligible_for_final_claims": False,
                }
            ],
        },
    )


def _write_completion_audit(output: Path) -> None:
    """Record a requirement-by-requirement closure, including mandatory stops."""

    proven = "PROVEN_COMPLETE"
    stopped = "STOPPED_BY_MANDATORY_CONTRACT_PORTABILITY_RULE"
    partial = "PARTIAL_EVIDENCE_BEFORE_MANDATORY_STOP"
    excluded = "NOT_APPLICABLE_SCOPE_EXCLUSION"
    superseded = "SUPERSEDED_BY_FOLLOWUP_GOAL"
    requirements = [
        {
            "id": "0_ordered_objective",
            "goal_section": "0",
            "status": proven,
            "evidence": ["phase_d_gate_v2.json", "vdp/phase_e_decision.json"],
            "note": "Audit correction preceded proof repair; Phase E began only after D1-D6 passed.",
        },
        {
            "id": "1_torch_c2_frozen",
            "goal_section": "1.1",
            "status": proven,
            "evidence": ["commands/torch_c2_source_freeze.log", "source_manifest.json"],
            "note": "The required source diff is empty at scientific SHA 29c9ee8.",
        },
        {
            "id": "1_huan_isolated",
            "goal_section": "1.2",
            "status": proven,
            "evidence": ["commands/phase0_huan_worktrees.log", "commands/phase0_huan_branch_reflog.log"],
            "note": "Original checkout remained unmodified; repair branch was pushed.",
        },
        {
            "id": "1_scope_exclusions",
            "goal_section": "1.3",
            "status": excluded,
            "evidence": ["vdp/phase_e_decision.json"],
            "note": "No controller/coupling/transcendental substitute or throughput campaign ran.",
        },
        {
            "id": "2_phase0_provenance",
            "goal_section": "2",
            "status": proven,
            "evidence": [
                "source_manifest.json", "environment.txt",
                "commands/phase0_torch_fetch.log", "commands/phase0_torch_audit_base.log",
                "commands/phase0_torch_worktrees.log", "commands/phase0_huan_status.log",
                "commands/phase0_huan_head.log", "commands/phase0_huan_worktrees.log",
                "commands/phase0_huan_uv_lock_diff.log",
            ],
            "note": "Verbatim command captures and the audit-only ninja/PyYAML overlay are preserved.",
        },
        {
            "id": "3_corrected_d2",
            "goal_section": "3",
            "status": proven,
            "evidence": [
                "d2_schedule_inventory.csv", "d2_host_order_oracle.json",
                "d2_actual_cpu.json", "d2_actual_cuda.json",
                "commands/d1_d2_cpu.log", "commands/d1_d2_cuda.log",
            ],
            "note": "Host, Torch CPU, Torch CUDA, and custom CUDA routes are separately counted with invocation evidence.",
        },
        {
            "id": "4_dirty_patch_search",
            "goal_section": "4",
            "status": proven,
            "evidence": ["dirty_patch_recovery.json", "dirty_patch_candidates.csv"],
            "note": "Bounded search conclusion is HISTORICAL_DIRTY_PATCHES_NOT_FOUND_AFTER_BOUNDED_SEARCH.",
        },
        {
            "id": "5_d6_map_oracles_repair",
            "goal_section": "5",
            "status": proven,
            "evidence": [
                "strict_roundoff_sources.csv", "witnesses/d6_minimal_witness.json",
                "witnesses/d6_repaired_minimal_witness.json", "commands/d6_repaired_witness.log",
            ],
            "note": "A concrete symbolic-Phi under-enclosure was minimized and strict accounting was repaired without changing parity semantics.",
        },
        {
            "id": "6_no_ftz_nonfinite",
            "goal_section": "6",
            "status": proven,
            "evidence": ["raw_logs/proof_kernel_cpu.json", "raw_logs/proof_kernel_cuda.json", "phase_d_gate_v2.json"],
            "note": "Production startup and certificate boundaries fail closed.",
        },
        {
            "id": "7_refinement_cache",
            "goal_section": "7",
            "status": proven,
            "evidence": [
                "refinement_boundary_cpu_v2.json", "refinement_boundary_cuda_v2.json",
                "commands/d5_refinement_cpu_audit.log", "commands/d5_refinement_cuda_audit.log",
            ],
            "note": "Sequential partial commits, owner generations, stale-cache tamper, and final ownership are machine-verifiable.",
        },
        {
            "id": "8_two_verifiers_d1_d6",
            "goal_section": "8",
            "status": proven,
            "evidence": ["phase_d_gate_v2.json", "commands/phase_d_scientific_gate.log", "proof_to_code_map_v2.csv"],
            "note": "Scientific rerun and artifact-only verifier have distinct scopes.",
        },
        {
            "id": "9_frozen_contract",
            "goal_section": "9.1",
            "status": stopped,
            "evidence": ["vdp/contract_matrix.csv", "vdp/native_terminal.json"],
            "note": "Fixed settings were represented; adaptive 0.002..0.1 plus SR100 is rejected without changing the contract.",
        },
        {
            "id": "9_four_lanes",
            "goal_section": "9.2",
            "status": stopped,
            "evidence": ["vdp/fixed_horizon_matrix.csv", "vdp/phase_e_decision.json"],
            "note": "Eight Huan fixed runs completed; fresh Flow*/Torch runs were forbidden after the portability stop.",
        },
        {
            "id": "9_step1_detail",
            "goal_section": "9.3",
            "status": partial,
            "evidence": ["vdp/step1_common_input.csv", "vdp/refinement_ledgers.jsonl.gz", "vdp/huan_final/run_index.json"],
            "note": "Hash, self-map, refinement, final remainder, endpoint/tube and aggregate roundoff are recorded. Candidate coefficients, an internal queue snapshot, and category-separated cutoff/remainder totals were not serialized before the mandatory stop and are explicitly labeled unavailable.",
        },
        {
            "id": "9_native_and_adjudication",
            "goal_section": "9.3-9.4",
            "status": stopped,
            "evidence": ["vdp/native_terminal.json", "vdp/first_divergence.json"],
            "note": "No T=10, cross-tool ranking, first-divergence, or Torch-terminal-cause claim is made.",
        },
        {
            "id": "9_primary_and_throughput",
            "goal_section": "9.5",
            "status": proven,
            "evidence": ["vdp/phase_e_decision.json"],
            "note": "Primary portability-stop label and explicit no-throughput status are present.",
        },
        {
            "id": "10_regressions_tamper",
            "goal_section": "10",
            "status": proven,
            "evidence": [
                "commands/huan_plant_full_final.log", "commands/torch_full_final.log",
                "commands/previous_package_verifier.log",
            ],
            "note": "Full suites and focused artifact/cache/kernel/status/fabrication tamper checks passed.",
        },
        {
            "id": "11_reports_artifacts",
            "goal_section": "11",
            "status": proven,
            "evidence": ["source_manifest.json", "SHA256SUMS"],
            "note": "Required reports, maps, gates, scripts, outputs, and fallback patch series are present.",
        },
        {
            "id": "12_commit_push",
            "goal_section": "12",
            "status": proven,
            "evidence": ["commands/phase0_torch_branch_reflog.log", "commands/phase0_huan_branch_reflog.log"],
            "note": "Both repair branches have committed, pushed histories; final tip equality is rechecked outside this self-hashed artifact.",
        },
        {
            "id": "13_stop_loss",
            "goal_section": "13",
            "status": proven,
            "evidence": ["vdp/native_terminal.json", "vdp/phase_e_decision.json", "vdp/first_divergence.json"],
            "note": "The portability stop was obeyed; stopped rows contain no fabricated science.",
        },
        {
            "id": "14_external_material",
            "goal_section": "14",
            "status": proven,
            "evidence": ["dirty_patch_recovery.json"],
            "note": "Missing historical patches and external review inputs are listed as non-blocking requests in the report.",
        },
        {
            "id": "original_goal_phase_f",
            "goal_section": "original goal Phase F",
            "status": superseded,
            "evidence": ["vdp/phase_e_decision.json"],
            "note": "The follow-up goal explicitly excludes throughput Phase F this round.",
        },
    ]
    _write_json(
        output / "completion_audit.json",
        {
            "schema": "torch_tm_flowpipe.huan_goal_completion_audit/1",
            "primary_status": PRIMARY,
            "overall": "GOAL_EXECUTED_WITH_MANDATORY_VDP_PORTABILITY_STOP",
            "allowed_partial_completion": True,
            "unsupported_green_claims": False,
            "status_vocabulary": [proven, stopped, partial, excluded, superseded],
            "requirements": requirements,
            "remaining_unresolved_scientific_claims": [
                "Huan parity native T=10 reachability",
                "Huan strict native T=10 reachability",
                "fresh four-lane cross-tool ranking and first divergence",
                "Huan adjudication of Torch C2 terminal y-upper failure",
                "category-separated step-1 cutoff/roundoff decomposition and queue snapshot",
                "historical 450-run dirty-source reproduction",
                "throughput Phase F",
            ],
        },
    )


def _write_environment(output: Path) -> None:
    commands = [
        ("uname", "uname", "-a"),
        ("lscpu", "lscpu"),
        ("nvidia-smi", "nvidia-smi", "-q"),
        ("nvcc", "/usr/local/cuda-12.6/bin/nvcc", "--version"),
        ("gcc", "/srv/local/shengenli/.huan-audit-gxx13/bin/x86_64-conda-linux-gnu-gcc", "--version"),
        ("g++", "/srv/local/shengenli/.huan-audit-gxx13/bin/x86_64-conda-linux-gnu-g++", "--version"),
        ("python", "/srv/local/shengenli/flowstar-gpu/.venv/bin/python", "--version"),
        (
            "torch",
            "/srv/local/shengenli/flowstar-gpu/.venv/bin/python",
            "-c",
            "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))",
        ),
    ]
    chunks = [
        "schema=torch_tm_flowpipe.huan_environment/1",
        f"generated_utc={datetime.now(timezone.utc).isoformat()}",
        f"platform={platform.platform()}",
        "audit_overlay=ninja and PyYAML installed outside uv.lock; uv.lock intentionally unchanged",
    ]
    for label, *command in commands:
        result = _command(command)
        chunks.extend(
            (
                f"\n[{label}]",
                "command=" + json.dumps(command),
                f"returncode={result['returncode']}",
                result["stdout"].rstrip(),
            )
        )
    (output / "environment.txt").write_text("\n".join(chunks) + "\n", encoding="utf-8")


def _write_manifest(
    repo: Path, engine: Path, torch_c2: Path, flowstar: Path, output: Path
) -> None:
    source_files = [
        repo / "scripts/run_huan_scientific_gate.py",
        repo / "scripts/run_huan_vdp_phase_e.py",
        repo / "scripts/verify_huan_proof_closure_package.py",
        repo / "docs/HUAN_STRICT_PROOF_CONTRACT_CLOSURE_20260826.md",
        repo / "docs/HUAN_FROZEN_VDP_PHASE_E_20260826.md",
        repo / "docs/strict_roundoff_accounting_graph.md",
        engine / "uv.lock",
    ]
    manifest = {
        "schema": "torch_tm_flowpipe.huan_proof_closure_source_manifest/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repositories": {
            "audit": _git_state(repo, _command(("git", "rev-parse", "HEAD"), repo)["stdout"].strip(), "audit_package_code"),
            "huan": _git_state(engine, HUAN_HEAD, "repaired_huan_engine"),
            "torch_c2_scientific": _git_state(torch_c2, TORCH_C2_SCIENTIFIC, "frozen_torch_c2_science"),
            "flowstar": _git_state(flowstar, FLOWSTAR_HEAD, "pinned_stock_flowstar"),
        },
        "frozen_torch_c2_package_tip": TORCH_C2_PACKAGE,
        "original_huan_base": "d5f0b68fcd36ba5f582733624f074728fe9720d8",
        "historical_dirty_patch_recovery": _json(output / "dirty_patch_recovery.json")["conclusion"],
        "source_files": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in source_files
        ],
        "uv_lock_intentionally_unchanged": True,
        "audit_overlay": ["ninja", "PyYAML"],
    }
    _write_json(output / "source_manifest.json", manifest)


def _write_checksums(output: Path) -> None:
    checksum = output / "SHA256SUMS"
    paths = sorted(path for path in output.rglob("*") if path.is_file() and path != checksum)
    checksum.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(output).as_posix()}\n" for path in paths),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    output = args.output_root.resolve()
    gate = _json(output / "phase_d_gate_v2.json")
    if not gate.get("overall_gate_passed") or gate.get("engine_head") != HUAN_HEAD:
        raise RuntimeError("final Phase-D scientific gate is not closed at the repaired SHA")
    # Inspect all repositories before touching any tracked package output.
    # Otherwise this finalizer makes its own audit worktree dirty and then
    # rejects that self-created state while building the manifest.
    _write_manifest(
        args.repo_root.resolve(), args.engine_root.resolve(),
        args.torch_c2_root.resolve(), args.flowstar_root.resolve(), output,
    )
    _write_phase_e(output)
    _write_completion_audit(output)
    _write_environment(output)
    _write_checksums(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--torch-c2-root", type=Path, required=True)
    parser.add_argument("--flowstar-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args)
    print(json.dumps({"primary_status": PRIMARY, "output_root": str(args.output_root.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
