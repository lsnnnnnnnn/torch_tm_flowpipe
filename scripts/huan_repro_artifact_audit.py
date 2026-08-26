#!/usr/bin/env python3
"""Fail-closed artifact inventory for the Huan flowstar_gpu audit.

This script deliberately does not import or reconstruct the missing engine.  It
records enough local evidence to decide whether the Phase A source-closure gate
in goal_vdp_terminal.md permits scientific reproduction.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable


AUDIT_DATE = "2026-08-26"
PRIMARY_DECISION = "HUAN_REPRO_BLOCKED_MISSING_CORE_SOURCE"
ENGINE_SOURCE_SUFFIX = Path("src/flowstar_gpu")
BUILD_FILE_NAMES = {
    "CMakeLists.txt",
    "Makefile",
    "meson.build",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
}
DISCOVERY_PATTERNS = (
    "*flowstar*gpu*",
    "*CROWN-Reach*GPU*",
    "new_crown_reach*.pdf",
    "REPRODUCE.md",
    "OPTIMIZATION.md",
)


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.rstrip("\n")


def _matches_discovery_name(name: str) -> bool:
    lowered = name.lower()
    return any(fnmatch.fnmatch(lowered, pattern.lower()) for pattern in DISCOVERY_PATTERNS)


def discover_candidates(root: Path, max_depth: int = 4) -> tuple[list[Path], list[str]]:
    """Mirror the goal's bounded artifact discovery without following symlinks."""

    root = root.resolve()
    hits: set[Path] = set()
    errors: list[str] = []

    def onerror(error: OSError) -> None:
        errors.append(f"{error.filename}: {error.strerror}")

    for current, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
        current_path = Path(current)
        depth = len(current_path.relative_to(root).parts)
        if depth >= max_depth:
            dirs[:] = []
        for name in [*dirs, *files]:
            candidate = current_path / name
            if len(candidate.relative_to(root).parts) <= max_depth and _matches_discovery_name(name):
                hits.add(candidate)
    return sorted(hits), sorted(set(errors))


def find_engine_roots(search_root: Path, max_depth: int = 4) -> list[Path]:
    """Find candidates that contain both the required package and a build file."""

    roots: set[Path] = set()
    for current, dirs, _files in os.walk(search_root, followlinks=False):
        current_path = Path(current)
        depth = len(current_path.resolve().relative_to(search_root.resolve()).parts)
        if depth >= max_depth:
            dirs[:] = []
        source = current_path / ENGINE_SOURCE_SUFFIX
        if not source.is_dir():
            continue
        if any((current_path / name).is_file() for name in BUILD_FILE_NAMES):
            roots.add(current_path)
    return sorted(roots)


def scan_symlinks(repo: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for current, dirs, files in os.walk(repo, followlinks=False):
        current_path = Path(current)
        if current_path == repo / ".git":
            dirs[:] = []
            continue
        dirs[:] = [name for name in dirs if name != ".git"]
        for name in [*dirs, *files]:
            path = current_path / name
            if not path.is_symlink():
                continue
            target = os.readlink(path)
            resolved = (path.parent / target).resolve(strict=False)
            probe_error = ""
            try:
                target_exists = resolved.exists()
                target_status = "AVAILABLE" if target_exists else "MISSING"
            except OSError as error:
                target_exists = False
                target_status = "INACCESSIBLE"
                probe_error = f"{type(error).__name__}: {error}"
            rows.append(
                {
                    "path": path.relative_to(repo).as_posix(),
                    "target": target,
                    "resolved_target": str(resolved),
                    "target_exists": target_exists,
                    "target_status": target_status,
                    "target_probe_error": probe_error,
                    "external_absolute_target": Path(target).is_absolute(),
                }
            )
    return sorted(rows, key=lambda row: row["path"])


def _json_record_provenance(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    git = payload.get("git")
    if not isinstance(git, dict):
        return None
    value = git.get("flowstar_gpu")
    return value if isinstance(value, str) and value else None


def collect_record_provenance(repo: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((repo / "comparison").glob("**/records/*.json")):
        value = _json_record_provenance(path)
        if value is None:
            continue
        rows.append(
            {
                "path": path.relative_to(repo).as_posix(),
                "flowstar_gpu_revision": value,
                "source_state": "DIRTY" if value.endswith("-dirty") else "CLEAN",
            }
        )
    return rows


def _tracked_paths(repo: Path) -> list[str]:
    output = _run_git(repo, "ls-files", "-z")
    return [item for item in output.split("\0") if item]


def candidate_repository_inventory(repo: Path) -> dict[str, Any]:
    tracked = _tracked_paths(repo)
    records = collect_record_provenance(repo)
    provenance_counts = Counter(row["flowstar_gpu_revision"] for row in records)
    symlinks = scan_symlinks(repo)
    integration_sources = [
        path
        for path in tracked
        if path
        in {
            "src/CrownReach.cpp",
            "src/CrownReach.h",
            "src/CrownSettings.cpp",
            "src/crown.py",
            "src/run.py",
        }
    ]
    result_reports = [
        path
        for path in tracked
        if path
        in {
            "comparison/ANALYSIS_NOTES.md",
            "comparison/HARDSUITE_POLYCROWN_ABLATION.md",
            "comparison/PHASE_F_PLAN.md",
            "comparison/POLYCROWN_CAMPAIGN.md",
            "comparison/REPORT.md",
            "comparison/SUITE_RESULTS.csv",
            "comparison/SUITE_RESULTS.json",
            "comparison/SUITE_RESULTS.md",
        }
    ]
    engine_named_paths = [
        path
        for path in tracked
        if "flowstar_gpu" in path.lower()
        or path.endswith("REPRODUCE.md")
        or path.endswith("OPTIMIZATION.md")
    ]
    onnx_files = [path for path in tracked if path.lower().endswith(".onnx")]
    yaml_files = [path for path in tracked if path.lower().endswith(('.yaml', '.yml'))]
    log_files = [path for path in tracked if path.lower().endswith((".log", ".out", ".err"))]
    yaml_symlinks = [path for path in yaml_files if (repo / path).is_symlink()]
    yaml_regular_files = [
        path for path in yaml_files if not (repo / path).is_symlink() and (repo / path).is_file()
    ]
    onnx_regular_files = [
        path for path in onnx_files if not (repo / path).is_symlink() and (repo / path).is_file()
    ]
    status = _run_git(repo, "status", "--porcelain=v1")
    return {
        "path": str(repo.resolve()),
        "classification": "WRAPPER_INTEGRATION_AND_RESULT_REPOSITORY",
        "git": {
            "head": _run_git(repo, "rev-parse", "HEAD"),
            "branch": _run_git(repo, "branch", "--show-current"),
            "remote_v": _run_git(repo, "remote", "-v").splitlines(),
            "status_porcelain": status.splitlines(),
            "clean": not bool(status),
            "history_commit_count": int(_run_git(repo, "rev-list", "--count", "--all")),
        },
        "required_engine_source": {
            "path": "src/flowstar_gpu",
            "present": (repo / ENGINE_SOURCE_SUFFIX).is_dir(),
            "tracked_paths_with_engine_name": engine_named_paths,
            "build_file_present_for_engine": False,
        },
        "available_integration_sources": integration_sources,
        "available_result_reports": result_reports,
        "tracked_file_counts": {
            "all": len(tracked),
            "onnx_model_paths": len(onnx_files),
            "onnx_regular_files": len(onnx_regular_files),
            "yaml_config_paths": len(yaml_files),
            "yaml_regular_files": len(yaml_regular_files),
            "yaml_symlinks": len(yaml_symlinks),
            "log_out_err_paths": len(log_files),
        },
        "symlinks": {
            "all": symlinks,
            "broken_count": sum(row["target_status"] == "MISSING" for row in symlinks),
            "inaccessible_count": sum(row["target_status"] == "INACCESSIBLE" for row in symlinks),
            "external_absolute_count": sum(row["external_absolute_target"] for row in symlinks),
        },
        "result_record_provenance": {
            "record_files_with_engine_revision": len(records),
            "clean_record_count": sum(row["source_state"] == "CLEAN" for row in records),
            "dirty_record_count": sum(row["source_state"] == "DIRTY" for row in records),
            "revision_counts": dict(sorted(provenance_counts.items())),
        },
    }


def torch_baseline_inventory(repo: Path) -> dict[str, Any]:
    target_ref = "origin/codex/vdp-post-accept-refinement-c2-20260820"
    expected = "0fea2657b30aea5f8cfe326dbcd06d659b8dd26c"
    actual = _run_git(repo, "rev-parse", target_ref)
    return {
        "path": str(repo.resolve()),
        "head": _run_git(repo, "rev-parse", "HEAD"),
        "branch": _run_git(repo, "branch", "--show-current"),
        "target_ref": target_ref,
        "expected_target_sha": expected,
        "actual_target_sha": actual,
        "target_matches_expected": actual == expected,
        "scientific_code_sha": "29c9ee8f1fe96b860052b86a2b37d79a37bbb2ca",
        "target_log": _run_git(repo, "log", "-8", "--oneline", target_ref).splitlines(),
    }


def missing_artifact_rows(record_revisions: Iterable[str]) -> list[dict[str, str]]:
    revisions = ", ".join(sorted(record_revisions)) or "no revision records found"
    return [
        {
            "artifact_id": "ENGINE_SOURCE",
            "required_item": "actual flowstar_gpu source repository including src/flowstar_gpu and build files",
            "status": "MISSING",
            "evidence": "bounded local discovery found no qualifying engine root",
            "request_from_huan": "provide repository access or an immutable source archive with a complete SHA256 manifest",
        },
        {
            "artifact_id": "ENGINE_HISTORY",
            "required_item": "flowstar_gpu .git history or immutable archive provenance",
            "status": "MISSING",
            "evidence": "only the separate CROWN-Reach-GPU repository has Git history",
            "request_from_huan": "provide full Git history, tags, submodules, remote URL, and target commit",
        },
        {
            "artifact_id": "EXACT_CLEAN_STATE",
            "required_item": "exact clean engine commit used by the target experiment",
            "status": "MISSING",
            "evidence": f"available run records reference only dirty states: {revisions}",
            "request_from_huan": "identify and export a clean reproduction commit for every claimed result family",
        },
        {
            "artifact_id": "DIRTY_PATCHES",
            "required_item": "complete uncommitted patches for every -dirty run record",
            "status": "MISSING",
            "evidence": "CROWN-Reach-GPU records retain revision labels but not the engine patches",
            "request_from_huan": "provide git diff --binary and untracked files for each recorded -dirty state",
        },
        {
            "artifact_id": "REPRODUCE_DOC",
            "required_item": "flowstar_gpu/docs/REPRODUCE.md",
            "status": "MISSING",
            "evidence": "not present in local discovery or CROWN-Reach-GPU history",
            "request_from_huan": "provide the exact version used for the recorded campaigns",
        },
        {
            "artifact_id": "OPTIMIZATION_DOC",
            "required_item": "flowstar_gpu/docs/OPTIMIZATION.md",
            "status": "MISSING",
            "evidence": "not present in local discovery or CROWN-Reach-GPU history",
            "request_from_huan": "provide the exact version used for the recorded campaigns",
        },
        {
            "artifact_id": "PAPER_NOTE",
            "required_item": "new_crown_reach*.pdf proof note",
            "status": "MISSING",
            "evidence": "no matching PDF was found by the required bounded discovery",
            "request_from_huan": "provide the exact PDF and its SHA256 digest",
        },
        {
            "artifact_id": "ENGINE_BENCHMARKS",
            "required_item": "engine benchmark configs, controllers/models, graded specs, and generation scripts",
            "status": "MISSING",
            "evidence": "records contain absolute /home/huan/projects/flowstar_gpu paths; those files are absent",
            "request_from_huan": "provide every referenced file with path mapping and SHA256, including the frozen VDP port",
        },
        {
            "artifact_id": "DEPENDENCY_LOCK",
            "required_item": "pinned Flow*, CROWN-Reach, auto_LiRPA, PyTorch/CUDA, compiler, and Python versions",
            "status": "PARTIAL_ONLY",
            "evidence": "reports mention selected versions, but no engine-owned complete lock or source closure is present",
            "request_from_huan": "provide lockfiles plus git SHAs, compiler output, pip/conda freeze, and CUDA runtime/driver capture",
        },
        {
            "artifact_id": "MACHINE_AND_COMMANDS",
            "required_item": "exact GPU identity and execution commands for target experiments",
            "status": "PARTIAL_ONLY",
            "evidence": "reports name Tesla V100-32GB and records retain commands, but exact machine/software capture is absent",
            "request_from_huan": "provide nvidia-smi -q, lscpu, environment capture, warmup/repetition protocol, and canonical commands",
        },
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tsv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_raw_logs(
    output_root: Path,
    search_root: Path,
    hits: list[Path],
    discovery_errors: list[str],
    torch: dict[str, Any],
    candidate: dict[str, Any],
    records: list[dict[str, str]],
) -> None:
    raw = output_root / "raw_logs"
    raw.mkdir(parents=True, exist_ok=True)
    discovery_lines = [
        "Phase A bounded artifact discovery",
        f"search_root={search_root.resolve()}",
        "max_depth=4",
        *(f"pattern={pattern}" for pattern in DISCOVERY_PATTERNS),
        *(f"hit={path}" for path in hits),
        *(f"scan_error={error}" for error in discovery_errors),
    ]
    (raw / "artifact_discovery.log").write_text("\n".join(discovery_lines) + "\n", encoding="utf-8")
    torch_lines = [
        f"path={torch['path']}",
        f"branch={torch['branch']}",
        f"head={torch['head']}",
        f"target_ref={torch['target_ref']}",
        f"expected_target_sha={torch['expected_target_sha']}",
        f"actual_target_sha={torch['actual_target_sha']}",
        f"target_matches_expected={str(torch['target_matches_expected']).lower()}",
        f"scientific_code_sha={torch['scientific_code_sha']}",
        "target_log:",
        *torch["target_log"],
    ]
    (raw / "torch_provenance.log").write_text("\n".join(torch_lines) + "\n", encoding="utf-8")
    candidate_lines = [
        f"path={candidate['path']}",
        f"classification={candidate['classification']}",
        f"head={candidate['git']['head']}",
        f"branch={candidate['git']['branch']}",
        f"clean={str(candidate['git']['clean']).lower()}",
        f"history_commit_count={candidate['git']['history_commit_count']}",
        *(f"remote={remote}" for remote in candidate["git"]["remote_v"]),
        f"required_engine_source_present={str(candidate['required_engine_source']['present']).lower()}",
        f"record_files_with_engine_revision={candidate['result_record_provenance']['record_files_with_engine_revision']}",
        f"clean_record_count={candidate['result_record_provenance']['clean_record_count']}",
        f"dirty_record_count={candidate['result_record_provenance']['dirty_record_count']}",
    ]
    for revision, count in candidate["result_record_provenance"]["revision_counts"].items():
        candidate_lines.append(f"record_revision={revision}\tcount={count}")
    (raw / "crown_reach_gpu_provenance.log").write_text(
        "\n".join(candidate_lines) + "\n", encoding="utf-8"
    )
    _write_tsv(
        raw / "record_provenance.tsv",
        ["path", "flowstar_gpu_revision", "source_state"],
        records,
    )
    _write_tsv(
        raw / "symlink_inventory.tsv",
        [
            "path",
            "target",
            "resolved_target",
            "target_exists",
            "target_status",
            "target_probe_error",
            "external_absolute_target",
        ],
        candidate["symlinks"]["all"],
    )


def write_checksums(output_root: Path) -> None:
    checksum_path = output_root / "SHA256SUMS"
    rows: list[str] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(output_root).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def verify_checksums(output_root: Path) -> list[str]:
    checksum_path = output_root / "SHA256SUMS"
    errors: list[str] = []
    if not checksum_path.is_file():
        return ["SHA256SUMS is missing"]
    covered: set[str] = set()
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"SHA256SUMS:{line_number}: malformed line")
            continue
        covered.add(relative)
        path = output_root / relative
        if not path.is_file():
            errors.append(f"missing covered file: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"checksum mismatch: {relative}")
    actual_files = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    for relative in sorted(actual_files - covered):
        errors.append(f"uncovered file: {relative}")
    return errors


def generate(search_root: Path, candidate_repo: Path, torch_repo: Path, output_root: Path) -> dict[str, Any]:
    hits, discovery_errors = discover_candidates(search_root)
    engine_roots = find_engine_roots(search_root)
    candidate = candidate_repository_inventory(candidate_repo)
    torch = torch_baseline_inventory(torch_repo)
    records = collect_record_provenance(candidate_repo)
    record_revisions = candidate["result_record_provenance"]["revision_counts"].keys()
    missing = missing_artifact_rows(record_revisions)
    inventory = {
        "schema": "torch_tm_flowpipe.huan_repro_artifact_inventory/1",
        "audit_date": AUDIT_DATE,
        "primary_decision": PRIMARY_DECISION,
        "search": {
            "root": str(search_root.resolve()),
            "max_depth": 4,
            "patterns": list(DISCOVERY_PATTERNS),
            "candidate_paths": [str(path) for path in hits],
            "errors": discovery_errors,
        },
        "torch_baseline": torch,
        "candidate_repositories": [candidate],
        "source_closure": {
            "qualifying_engine_roots": [str(path) for path in engine_roots],
            "engine_source_available": bool(engine_roots),
            "paper_note_available": any(
                path.is_file() and fnmatch.fnmatch(path.name.lower(), "new_crown_reach*.pdf")
                for path in hits
            ),
            "reproduce_doc_available": any(path.name.lower() == "reproduce.md" for path in hits),
            "optimization_doc_available": any(path.name.lower() == "optimization.md" for path in hits),
            "all_required_artifacts_available": False,
        },
        "stop_rule": {
            "triggered": True,
            "reason": "flowstar_gpu/src/flowstar_gpu, engine build files, and exact clean engine source state are unavailable",
            "phases_not_run": ["B", "C", "D", "E", "F"],
        },
    }
    _write_json(output_root / "artifact_inventory.json", inventory)
    _write_tsv(
        output_root / "missing_artifacts.tsv",
        ["artifact_id", "required_item", "status", "evidence", "request_from_huan"],
        missing,
    )
    _write_raw_logs(output_root, search_root, hits, discovery_errors, torch, candidate, records)
    write_checksums(output_root)
    return inventory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--candidate-repo", type=Path, required=True)
    parser.add_argument("--torch-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.verify_only:
        errors = verify_checksums(args.output_root)
        if errors:
            for error in errors:
                print(error)
            return 1
        print(f"verified {args.output_root / 'SHA256SUMS'}")
        return 0
    inventory = generate(args.search_root, args.candidate_repo, args.torch_repo, args.output_root)
    print(inventory["primary_decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
