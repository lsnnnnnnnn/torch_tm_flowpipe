#!/usr/bin/env python3
"""Build a compact, hash-complete package from full numerical run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Iterable, Mapping


PACKAGE_SCHEMA = "three_tool_full_horizon_pairwise_carry_package_v3"


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite token {token} in {path}")
        ),
    )
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique(
    root: Path,
    pattern: str,
    predicate: Callable[[Path], bool] | None = None,
) -> Path:
    candidates = [path for path in sorted(root.glob(pattern)) if path.is_file()]
    if predicate is not None:
        candidates = [path for path in candidates if predicate(path)]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one {pattern}, found {[str(path) for path in candidates]}")
    return candidates[0]


def _json_field(field: str, expected: Any) -> Callable[[Path], bool]:
    return lambda path: _load(path).get(field) == expected


def _copy(source: Path, root: Path, relative: str) -> Path:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _copy_names(source_dir: Path, root: Path, prefix: str, names: Iterable[str]) -> None:
    for name in names:
        source = source_dir / name
        if source.is_file():
            _copy(source, root, f"{prefix}/{name}")


def _head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_run_root.resolve()
    output = args.output_root.resolve()
    repo = args.repo_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    if _head(repo) != args.tested_source_sha:
        raise RuntimeError("tested source SHA is not the builder checkout")
    output.mkdir(parents=True)

    historical = repo / "outputs/three_tool_matched_divergence_fixed_support_20260811/20260811T100304Z/manifest.json"
    if not historical.is_file():
        raise RuntimeError("tracked historical package is missing")
    _copy(historical, output, "02_evidence_architecture/historical_package_manifest.json")

    provenance = source / "00_provenance"
    if provenance.is_dir():
        for path in sorted(provenance.rglob("*")):
            if path.is_file() and path.name != "artifact_index.json" and path.stat().st_size <= 2_000_000:
                _copy(path, output, f"00_provenance/{path.relative_to(provenance).as_posix()}")

    for path in sorted(args.h1_tests_dir.rglob("*")):
        if path.is_file() and path.stat().st_size <= 5_000_000:
            _copy(path, output, f"14_tests_at_tested_source/{path.relative_to(args.h1_tests_dir).as_posix()}")

    flow_summary = _unique(
        source,
        "03_flowstar_torch_fixed_schedule/**/common_prefix/artifacts/run/summary.json",
        _json_field("outcome", "FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_ONLY"),
    )
    flow_run = flow_summary.parent
    flow_runner = flow_run.parents[1]
    flow_lab = flow_run.parents[2]
    _copy_names(flow_run, output, "04_flowstar_torch_fixed_schedule/common_prefix", ("summary.json", "common_prefix.csv", "report.md"))
    _copy_names(flow_runner, output, "04_flowstar_torch_fixed_schedule/runner", ("config.json", "summary.json", "stdout.log", "stderr.log", "command.txt", "exit_code.txt", "started_at.txt", "finished_at.txt", "timing.json"))
    flowstar_trace = flow_lab / "flowstar_cold/artifacts/flowstar_trace.csv"
    if not flowstar_trace.is_file():
        raise RuntimeError("Flow* trace from the selected comparison lab is missing")
    metadata = flowstar_trace.with_name("flowstar_trace_metadata.csv")
    _copy(metadata, output, "04_flowstar_torch_fixed_schedule/flowstar/flowstar_trace_metadata.csv")
    flowstar_runner = flowstar_trace.parents[1]
    _copy_names(flowstar_runner, output, "04_flowstar_torch_fixed_schedule/flowstar", ("config.json", "summary.json", "stdout.log", "stderr.log", "command.txt", "exit_code.txt", "timing.json"))
    expected_torch_summary_sha = str(_load(flow_summary)["source_sha256"]["torch_summary.json"])
    torch_summary = _unique(
        source,
        "03_flowstar_torch_fixed_schedule/**/torch_cold/artifacts/run/summary.json",
        lambda path: _sha(path) == expected_torch_summary_sha,
    )
    torch_run = torch_summary.parent
    _copy_names(torch_run, output, "04_flowstar_torch_fixed_schedule/torch", ("summary.json", "decision.json", "segments.csv", "checkpoints.csv", "profile.csv", "command.json", "config_snapshot.yaml"))
    torch_runner = torch_run.parents[1]
    _copy_names(torch_runner, output, "04_flowstar_torch_fixed_schedule/torch", ("config.json", "summary.json", "stdout.log", "stderr.log", "command.txt", "exit_code.txt", "timing.json"))

    diff_comparison = _unique(
        source,
        "04_diffreach_torch_full_horizon/**/full_trace_v1/comparison/comparison.json",
        _json_field("outcome", "DIFFREACH_TORCH_DR7_FULL_HORIZON_DIVERGED"),
    )
    diff_root = diff_comparison.parents[1]
    _copy_names(diff_comparison.parent, output, "05_diffreach_torch_full_horizon/cross_tool_comparison", ("comparison.json", "endpoint_tube_delta_by_step.csv"))
    stock = _unique(source, "04_diffreach_torch_full_horizon/stock_diffreach/summary.json")
    _copy(stock, output, "05_diffreach_torch_full_horizon/stock_diffreach/summary.json")
    for tool in ("diffreach", "torch"):
        lane = diff_root / tool
        _copy_names(lane, output, f"05_diffreach_torch_full_horizon/{tool}", ("summary.json", "command.json", "artifact_manifest.json"))
        for step in ("step_0001.npz", "step_0002.npz"):
            _copy(lane / "captures" / step, output, f"05_diffreach_torch_full_horizon/{tool}/captures/{step}")
    repeat = diff_root / "diffreach_sequential_repeat"
    _copy_names(repeat, output, "05_diffreach_torch_full_horizon/diffreach_sequential_repeat", ("summary.json", "command.json", "artifact_manifest.json"))

    reproduction: dict[tuple[str, int], Path] = {}
    for path in sorted(source.glob("06_carry_reproduction/**/summary.json")):
        summary = _load(path)
        if summary.get("schema") == "torch_r35_a3_a4_carry_trace_v1":
            key = (str(summary["cell"]), int(summary["batch"]))
            if key in reproduction:
                raise RuntimeError(f"duplicate carry reproduction {key}")
            reproduction[key] = path
    if set(reproduction) != {("A3", 1), ("A3", 64), ("A4", 1), ("A4", 64)}:
        raise RuntimeError("carry reproduction matrix is incomplete")
    for (cell, batch), summary_path in sorted(reproduction.items()):
        label = f"{cell.lower()}_b{batch}"
        _copy(summary_path, output, f"06_carry_reproduction/{label}/summary.json")
        _copy(summary_path.with_name("metrics.csv"), output, f"06_carry_reproduction/{label}/metrics.csv")
    a4_b1 = reproduction[("A4", 1)].parent / "prestates"
    for step in ("before_step_0001.npz", "before_step_0002.npz", "before_step_0101.npz", "before_step_0320.npz"):
        _copy(a4_b1 / step, output, f"06_carry_reproduction/a4_b1/prestates/{step}")

    divergence = _unique(source, "07_carry_state_traces/**/divergence_ledger.json")
    _copy(divergence, output, "07_carry_state_traces/divergence_ledger.json")
    substitutions = _unique(source, "08_same_prestate_substitutions/**/summary.json", _json_field("schema", "torch_r35_a3_a4_same_prestate_substitution_v1"))
    _copy_names(substitutions.parent, output, "08_same_prestate_substitutions", ("summary.json", "substitutions.csv"))
    dense = _unique(source, "09_dense_cni_parity/**/parity.json", _json_field("dense_cni_parity_outcome", "DENSE_CNI_PARITY_NOT_EXPRESSIBLE"))
    _copy(dense, output, "09_dense_cni_parity/parity.json")

    def accounting_for(step: str) -> Path:
        return _unique(
            source,
            f"10_root_cause/composition_accounting_lab_*/{step}/composition_accounting.json",
            lambda path: "endpoint_remainder_times_parameterization_polynomial"
            in _load(path)["checkpoints"][0]["source_intervals"],
        )

    first_accounting = accounting_for("before_step_0002")
    failure_accounting = accounting_for("before_step_0320")
    _copy(first_accounting, output, "10_root_cause/composition_before_step_0002.json")
    _copy(failure_accounting, output, "10_root_cause/composition_before_step_0320.json")
    root_cause = _unique(source, "10_root_cause/root_cause_lab_*/root_cause.json", _json_field("root_cause_class", "C4"))
    _copy(root_cause, output, "10_root_cause/root_cause.json")
    root_report = _load(root_cause)
    _write(
        output / "11_single_fix_if_authorized/no_fix_authorized.json",
        {
            "schema": "single_fix_authorization_v1",
            "outcome": root_report["single_fix_authorization"],
            "root_cause_class": root_report["root_cause_class"],
            "next_authorized_action": root_report["next_authorized_action"],
        },
    )

    tables = _unique(source, "12_pairwise_tables/**/summary.json", _json_field("outcome", "PAIRWISE_TABLES_BUILT_WITHOUT_UNIVERSAL_RANKING"))
    for path in sorted(tables.parent.iterdir()):
        if path.is_file():
            _copy(path, output, f"12_pairwise_tables/{path.name}")
    figures = _unique(source, "13_figures/**/summary.json", _json_field("outcome", "CAUSAL_FIGURES_BUILT"))
    for path in sorted(figures.parent.iterdir()):
        if path.is_file():
            _copy(path, output, f"13_figures/{path.name}")

    outcome_registry = {
        "evidence_package_status": "EVIDENCE_PACKAGE_REBUILT_PENDING_TRACKING",
        "flowstar_torch_fixed_schedule_status": _load(flow_summary)["outcome"],
        "diffreach_torch_full_horizon_status": _load(diff_comparison)["outcome"],
        "carry_semantics_status": root_report["outcome"],
        "dense_cni_parity_status": _load(dense)["dense_cni_parity_outcome"],
        "single_fix_status": root_report["single_fix_authorization"],
        "matched_workload_timing": "UNAVAILABLE_SEMANTICS_GATE_NOT_CLOSED",
        "formal_scope": "NO_NEW_FORMAL_CROSS_TOOL_CLAIM",
    }
    _write(
        output / "03_claim_registry_before/registry.json",
        _load(historical).get("outcome_registry_at_recovery", {}),
    )
    _write(output / "16_claim_registry_after/registry.json", outcome_registry)

    payload = [
        path for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "verification.json", "SHA256SUMS"}
    ]
    artifacts = [
        {"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": _sha(path)}
        for path in payload
    ]
    required = (
        "04_flowstar_torch_fixed_schedule/common_prefix/summary.json",
        "04_flowstar_torch_fixed_schedule/common_prefix/common_prefix.csv",
        "05_diffreach_torch_full_horizon/cross_tool_comparison/comparison.json",
        "06_carry_reproduction/a4_b1/prestates/before_step_0320.npz",
        "09_dense_cni_parity/parity.json",
        "10_root_cause/root_cause.json",
        "11_single_fix_if_authorized/no_fix_authorized.json",
        "12_pairwise_tables/summary.json",
        "13_figures/summary.json",
        "16_claim_registry_after/registry.json",
    )
    manifest = {
        "schema": PACKAGE_SCHEMA,
        "run_id": output.name,
        "tested_source_sha": args.tested_source_sha,
        "package_commit_sha": None,
        "delivery_audit_sha": None,
        "package_root": ".",
        "outcome_registry": outcome_registry,
        "required_paths": list(required),
        "artifacts": artifacts,
        "limitations": [
            "large full traces and compiled binaries are intentionally excluded",
            "raw trace hashes remain recorded in the copied scientific summaries",
            "package commit identity is established by Git ancestry at H2, not embedded circularly",
        ],
    }
    _write(output / "manifest.json", manifest)
    verification = {
        "schema": "three_tool_full_horizon_pairwise_carry_verification_v1",
        "status": "pass",
        "tested_source_sha": args.tested_source_sha,
        "checks": {
            "required_paths_present": all((output / relative).is_file() for relative in required),
            "flowstar_torch_outcome_derived": outcome_registry["flowstar_torch_fixed_schedule_status"] == "FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_ONLY",
            "diffreach_torch_outcome_derived": outcome_registry["diffreach_torch_full_horizon_status"] == "DIFFREACH_TORCH_DR7_FULL_HORIZON_DIVERGED",
            "carry_class_derived": outcome_registry["carry_semantics_status"] == "CARRY_MISSING_SYMBOLIC_SEMANTICS",
            "no_fix_derived": outcome_registry["single_fix_status"] == "NO_FIX_AUTHORIZED",
            "every_figure_has_source_csv": _load(figures)["constraints"]["every_figure_has_source_csv"],
            "universal_ranking_absent": not _load(tables)["universal_ranking_emitted"],
        },
    }
    if not all(verification["checks"].values()):
        raise RuntimeError("package scientific verification failed")
    _write(output / "verification.json", verification)
    checksum = output / "SHA256SUMS"
    checksum.write_text(
        "".join(
            f"{_sha(path)}  {path.relative_to(output).as_posix()}\n"
            for path in sorted(output.rglob("*")) if path.is_file() and path != checksum
        ),
        encoding="utf-8",
    )
    try:
        from .verify_full_horizon_pairwise_package import verify
    except ImportError:
        from verify_full_horizon_pairwise_package import verify

    verify(output, expected_source_sha=args.tested_source_sha, require_tracked=False, repo_root=repo)
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--h1-tests-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--tested-source-sha", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(_args())
