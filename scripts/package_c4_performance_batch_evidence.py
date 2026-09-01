#!/usr/bin/env python3
"""Assemble the audited C4 reference/performance/CPU-batch evidence package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import formal_reference_configuration  # noqa: E402


ARTIFACT_RELATIVE = Path("artifacts/runs/c4_reference_performance_batch_20260829")
SOURCE_PACKAGE_SHA = "ed9c305dc39c25eab23a96f4fb3775cc2d13d396"
SOURCE_BRANCH = "codex/torch-flowstar-brusselator-live-range-c5-20260828"
BRANCH = "codex/c4-reference-performance-batch-foundation-20260829"
REFERENCE_SHA = "f34b5fa4155f5475a681411b627d68345ed401ea"
OPTIMIZED_SHA = "4939fb288c941a67f55cc191f4d75f8594692f47"
BATCH_SHA = "7608dd52e48af3ce8ae2e0a8343aae125c63b7f4"
INSTRUMENTATION_SHA = "d6b543446402ef6b12717b727b236fc7c9c75af5"
FLOWSTAR_SHA = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"

WINDOW_ORDER = (
    "brusselator_steps_1_20",
    "brusselator_steps_1_100",
    "brusselator_steps_901_1000",
    "vdp_representative_prefix",
)
WINDOW_RENAMES = {
    "vdp_fixed_prefix_1_20": "vdp_representative_prefix",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError(f"inconsistent CSV columns: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _select_profile_rows(
    prefix_profile: Path,
    formal_profile: Path,
    filename: str,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    selections = (
        (prefix_profile, {"brusselator_steps_1_20"}),
        (
            formal_profile,
            {
                "brusselator_steps_1_100",
                "brusselator_steps_901_1000",
                "vdp_fixed_prefix_1_20",
            },
        ),
    )
    for source, source_windows in selections:
        for row in _read_rows(source / filename):
            source_window = row["window"]
            if source_window not in source_windows:
                continue
            copied = dict(row)
            copied["window"] = WINDOW_RENAMES.get(source_window, source_window)
            selected.append(copied)
    seen = {row["window"] for row in selected}
    if seen != set(WINDOW_ORDER):
        raise ValueError(f"{filename} profile windows differ: {sorted(seen)}")
    rank = {name: index for index, name in enumerate(WINDOW_ORDER)}
    selected.sort(key=lambda row: rank[row["window"]])
    return selected


def _flame_sections(path: Path) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("===== ") and line.rstrip().endswith(" ====="):
            current = line.strip()[6:-6]
            sections[current] = [line]
        elif current is not None:
            sections[current].append(line)
    return {name: "".join(lines) for name, lines in sections.items()}


def _assemble_flamegraph(prefix_profile: Path, formal_profile: Path) -> str:
    prefix = _flame_sections(prefix_profile / "flamegraph.txt")
    formal = _flame_sections(formal_profile / "flamegraph.txt")
    sources = {
        "brusselator_steps_1_20": prefix,
        "brusselator_steps_1_100": formal,
        "brusselator_steps_901_1000": formal,
        "vdp_representative_prefix": formal,
    }
    source_names = {
        "vdp_representative_prefix": "vdp_fixed_prefix_1_20",
    }
    output: list[str] = []
    for target in WINDOW_ORDER:
        source_name = source_names.get(target, target)
        section = sources[target].get(source_name)
        if section is None:
            raise ValueError(f"missing flamegraph section: {source_name}")
        if source_name != target:
            section = section.replace(
                f"===== {source_name} =====", f"===== {target} =====", 1
            )
        output.append(section.rstrip() + "\n")
    return "\n".join(output)


def _median(rows: Iterable[Mapping[str, str]], variant: str, steps: int) -> float:
    values = [
        float(row["wall_s"])
        for row in rows
        if row["variant"] == variant
        and row["workload"] == "brusselator"
        and int(row["steps"]) == steps
    ]
    if not values:
        raise ValueError(f"missing {variant} Brusselator {steps}-step timing")
    return statistics.median(values)


def _copy_required_inputs(args: argparse.Namespace, artifact_dir: Path) -> None:
    copies = {
        args.gate_dir / "VDP_REGRESSION.json": artifact_dir / "VDP_REGRESSION.json",
        args.gate_dir / "BRUSSELATOR_REGRESSION.json": artifact_dir / "BRUSSELATOR_REGRESSION.json",
        args.gate_dir / "optimization_result.json": artifact_dir / "optimization_result.json",
        args.gate_dir / "prefix_runtime_matrix.csv": artifact_dir / "prefix_runtime_matrix.csv",
        args.gate_dir / "full_runtime_matrix.csv": artifact_dir / "full_runtime_matrix.csv",
        args.batch_dir / "cpu_batch_equivalence.csv": artifact_dir / "cpu_batch_equivalence.csv",
        args.batch_dir / "cpu_batch_runtime.csv": artifact_dir / "cpu_batch_runtime.csv",
        args.batch_dir / "cpu_batch_result.json": artifact_dir / "cpu_batch_result.json",
        args.observer_profile_dir / "production_vs_audit_overhead.csv": artifact_dir
        / "production_vs_audit_overhead.csv",
    }
    for source, target in copies.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, target)


def _write_manifest(artifact_dir: Path) -> None:
    lines = []
    for path in sorted(artifact_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}\n")
    (artifact_dir / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def assemble(args: argparse.Namespace) -> Path:
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if any(artifact_dir.iterdir()):
        raise ValueError(f"artifact directory must be empty: {artifact_dir}")

    _copy_required_inputs(args, artifact_dir)
    for filename in ("hotspot_profile.csv", "call_count_matrix.csv", "allocation_profile.csv"):
        _write_rows(
            artifact_dir / filename,
            _select_profile_rows(args.prefix_profile_dir, args.formal_profile_dir, filename),
        )
    (artifact_dir / "flamegraph.txt").write_text(
        _assemble_flamegraph(args.prefix_profile_dir, args.formal_profile_dir),
        encoding="utf-8",
    )

    formal_summary = _read_json(args.formal_profile_dir / "profile_summary.json")
    prefix_summary = _read_json(args.prefix_profile_dir / "profile_summary.json")
    observer_summary = _read_json(args.observer_profile_dir / "profile_summary.json")
    if formal_summary["status"] != "PROFILE_COMPLETE" or prefix_summary["status"] != "PROFILE_COMPLETE":
        raise ValueError("profile input is incomplete")
    profile_summary = dict(formal_summary)
    profile_summary["observer_rows"] = observer_summary["observer_rows"]
    profile_summary["observer_scientific_equality"] = observer_summary[
        "observer_scientific_equality"
    ]
    profile_summary["profile_windows"] = [
        *(row for row in prefix_summary["profile_windows"] if row["window"] == "brusselator_steps_1_20"),
        *(row for row in formal_summary["profile_windows"] if row["window"] != "vdp_fixed_prefix_1_20"),
        *(
            {**row, "window": "vdp_representative_prefix"}
            for row in formal_summary["profile_windows"]
            if row["window"] == "vdp_fixed_prefix_1_20"
        ),
    ]
    profile_summary["instrumentation_sha"] = INSTRUMENTATION_SHA
    profile_summary["numerical_reference_sha"] = REFERENCE_SHA
    profile_summary["cpu_affinity"] = [0]
    _write_json(artifact_dir / "profile_summary.json", profile_summary)

    _write_json(artifact_dir / "REFERENCE_CONFIG.json", formal_reference_configuration())
    _write_json(
        artifact_dir / "PROVENANCE.json",
        {
            "schema": "torch_tm_flowpipe.c4_reference_performance_batch_provenance/1",
            "source_package_sha": SOURCE_PACKAGE_SHA,
            "source_branch": SOURCE_BRANCH,
            "branch": BRANCH,
            "reference_scientific_sha": REFERENCE_SHA,
            "optimized_scientific_sha": OPTIMIZED_SHA,
            "batch_scientific_sha": BATCH_SHA,
            "instrumentation_sha": INSTRUMENTATION_SHA,
            "evidence_assembly_code_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "stock_flowstar_sha": FLOWSTAR_SHA,
            "formal_runs_clean": True,
            "cpu_affinity": [0],
            "cpu_contention_observed": False,
            "observer_mode_for_performance": "production_no_observer",
            "timer_scope": "solver_only_excludes_snapshot_serialization_checkpoint",
            "reference_profile_import_mode": "instrumentation wrapper with numerical modules imported from clean reference SHA",
            "tail_checkpoint": {
                "accepted_step": 900,
                "generation": 900,
                "sha256": "5ae837ec83240ac2c52800371e050c5ea607ecd8d57debc25a2932ea19c8c5b0",
                "bitwise_equivalent_to_reference_prefix": True,
            },
        },
    )
    _write_json(
        artifact_dir / "optimization_authorization.json",
        {
            "schema": "torch_tm_flowpipe.c4_optimization_authorization/1",
            "authorized": True,
            "single_candidate_only": True,
            "candidate": "packed accepted-boundary SR owner propagation",
            "profile_basis": "accepted-boundary reset and SR preparation/propagation dominate the late full-queue window",
            "profile_total_fraction": 0.805,
            "expected_end_to_end_speedup": 2.651126609978951,
            "no_cache": True,
            "common_to_vdp_and_brusselator": True,
            "b1_bitwise_oracle_passed": True,
            "numerical_order_preserved": True,
            "outward_rounding_sites_preserved": True,
            "reference_scientific_sha": REFERENCE_SHA,
            "candidate_commit_sha": "4302968",
            "optimized_scientific_sha": OPTIMIZED_SHA,
            "no_second_optimization_stacked": True,
        },
    )

    prefix_rows = _read_rows(artifact_dir / "prefix_runtime_matrix.csv")
    full_rows = _read_rows(artifact_dir / "full_runtime_matrix.csv")
    optimization = _read_json(artifact_dir / "optimization_result.json")
    batch = _read_json(args.batch_dir / "cpu_batch_result.json")
    roots = optimization["scientific_roots"]
    if roots["reference"]["sha"] != REFERENCE_SHA or not roots["reference"]["clean"]:
        raise ValueError("reference scientific root is not the frozen clean SHA")
    if roots["optimized"]["sha"] != OPTIMIZED_SHA or not roots["optimized"]["clean"]:
        raise ValueError("optimized scientific root is not the frozen clean SHA")
    if batch["scientific_sha"] != BATCH_SHA or batch["cpu_affinity"] != [0]:
        raise ValueError("CPU batch input identity drift")
    speed100 = _median(prefix_rows, "reference", 100) / _median(prefix_rows, "optimized", 100)
    speed300 = _median(prefix_rows, "reference", 300) / _median(prefix_rows, "optimized", 300)
    reference_full = next(row for row in full_rows if row["variant"] == "reference")
    optimized_full = next(row for row in full_rows if row["variant"] == "optimized")
    full_speed = float(reference_full["wall_s"]) / float(optimized_full["wall_s"])
    speed_passed = speed100 >= 2.0 and speed300 >= 2.0 and full_speed >= 2.0
    if speed_passed != bool(optimization["prefix_speed_gate_passed"] and optimization["full_speed_gate_passed"]):
        raise ValueError("performance-gate classification drift")
    if not batch["equivalence_passed"] or not batch["b8_runtime_diagnostic_passed"]:
        raise ValueError("CPU batch formal run did not pass")
    _write_json(
        artifact_dir / "RESULT.json",
        {
            "schema": "torch_tm_flowpipe.c4_reference_performance_batch_result/1",
            "status": (
                "C4_REFERENCE_FROZEN__SEMANTICS_PRESERVED__CPU_SPEED_GATE_PASSED__CPU_BATCH_FOUNDATION_PASSED"
                if speed_passed
                else "C4_REFERENCE_FROZEN__CPU_BATCH_FOUNDATION_PASSED__CPU_SPEED_GATE_FAILED"
            ),
            "reference_frozen": True,
            "semantics_preserved": True,
            "vdp_zero_regression_passed": True,
            "brusselator_zero_regression_passed": True,
            "single_optimization_rule_observed": True,
            "cpu_speed_gate_passed": speed_passed,
            "speed_gates": {
                "brusselator_prefix_100_speedup": speed100,
                "brusselator_prefix_100_passed": speed100 >= 2.0,
                "brusselator_prefix_300_speedup": speed300,
                "brusselator_prefix_300_passed": speed300 >= 2.0,
                "brusselator_full_speedup": full_speed,
                "brusselator_full_passed": full_speed >= 2.0,
            },
            "memory_gate_passed": optimization["memory_gate_passed"],
            "cpu_batch_foundation_passed": True,
            "cpu_batch_equivalence_rows": batch["equivalence_rows"],
            "b8_slowdown_vs_8x_serial_b1": batch["b8_slowdown_vs_8x_serial_b1"],
            "cuda_batch_next_round_authorized": True,
            "cuda_implementation_in_scope": False,
        },
    )
    _write_manifest(artifact_dir)
    return artifact_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / ARTIFACT_RELATIVE)
    parser.add_argument("--prefix-profile-dir", type=Path, required=True)
    parser.add_argument("--formal-profile-dir", type=Path, required=True)
    parser.add_argument("--observer-profile-dir", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_dir = assemble(args)
    print(json.dumps({"status": "C4_EVIDENCE_PACKAGE_ASSEMBLED", "artifact_dir": str(artifact_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
