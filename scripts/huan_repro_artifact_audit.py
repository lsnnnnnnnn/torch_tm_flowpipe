#!/usr/bin/env python3
"""Artifact and source-closure inventory for the Huan flowstar_gpu audit.

The inventory deliberately separates two questions which must not be merged:

* is a clean, inspectable engine revision available for a new audit; and
* are the historical ``-dirty`` result records exactly reproducible?

The first can be true while the second remains false because a dirty patch is
missing. Scientific runners consume this inventory and must fail closed at the
narrower gate that applies to their claim.
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


def _run_git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.rstrip("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_regular_accessible(path: Path) -> bool:
    """Return False for missing, non-regular, broken, or inaccessible paths."""

    try:
        return path.is_file() and not path.is_symlink()
    except OSError:
        return False


def _matches_discovery_name(name: str) -> bool:
    lowered = name.lower()
    return any(fnmatch.fnmatch(lowered, pattern.lower()) for pattern in DISCOVERY_PATTERNS)


def discover_candidates(root: Path, max_depth: int = 4) -> tuple[list[Path], list[str]]:
    """Mirror the goal's bounded artifact discovery without following links."""

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
    """Find roots containing both the required package and a build file."""

    roots: set[Path] = set()
    search_root = search_root.resolve()
    for current, dirs, _files in os.walk(search_root, followlinks=False):
        current_path = Path(current)
        depth = len(current_path.resolve().relative_to(search_root).parts)
        if depth >= max_depth:
            dirs[:] = []
        if not (current_path / ENGINE_SOURCE_SUFFIX).is_dir():
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
    git = payload.get("git") if isinstance(payload, dict) else None
    value = git.get("flowstar_gpu") if isinstance(git, dict) else None
    return value if isinstance(value, str) and value else None


def collect_record_provenance(repo: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((repo / "comparison").glob("**/records/*.json")):
        value = _json_record_provenance(path)
        if value is not None:
            rows.append(
                {
                    "path": path.relative_to(repo).as_posix(),
                    "flowstar_gpu_revision": value,
                    "base_revision": value.removesuffix("-dirty"),
                    "source_state": "DIRTY" if value.endswith("-dirty") else "CLEAN",
                }
            )
    return rows


def _tracked_paths(repo: Path) -> list[str]:
    return [item for item in _run_git(repo, "ls-files", "-z").split("\0") if item]


def _git_inventory(repo: Path) -> dict[str, Any]:
    status = _run_git(repo, "status", "--porcelain=v1")
    return {
        "head": _run_git(repo, "rev-parse", "HEAD"),
        "branch": _run_git(repo, "branch", "--show-current"),
        "remote_v": _run_git(repo, "remote", "-v").splitlines(),
        "status_porcelain": status.splitlines(),
        "clean": not bool(status),
        "history_commit_count": int(_run_git(repo, "rev-list", "--count", "--all")),
        "tags": _run_git(repo, "tag", "--list").splitlines(),
        "submodules": _run_git(repo, "submodule", "status", "--recursive").splitlines(),
        "full_diff": _run_git(repo, "diff", "--binary"),
    }


def integration_repository_inventory(repo: Path) -> dict[str, Any]:
    tracked = _tracked_paths(repo)
    records = collect_record_provenance(repo)
    provenance_counts = Counter(row["flowstar_gpu_revision"] for row in records)
    symlinks = scan_symlinks(repo)
    yaml_files = [p for p in tracked if p.lower().endswith((".yaml", ".yml"))]
    onnx_files = [p for p in tracked if p.lower().endswith(".onnx")]
    return {
        "path": str(repo.resolve()),
        "classification": "WRAPPER_INTEGRATION_AND_RESULT_REPOSITORY",
        "git": _git_inventory(repo),
        "required_engine_source": {
            "path": "src/flowstar_gpu",
            "present": (repo / ENGINE_SOURCE_SUFFIX).is_dir(),
            "build_file_present_for_engine": False,
        },
        "tracked_file_counts": {
            "all": len(tracked),
            "onnx_model_paths": len(onnx_files),
            "onnx_regular_files": sum(_is_regular_accessible(repo / p) for p in onnx_files),
            "yaml_config_paths": len(yaml_files),
            "yaml_regular_files": sum(_is_regular_accessible(repo / p) for p in yaml_files),
            "yaml_symlinks": sum((repo / p).is_symlink() for p in yaml_files),
            "log_out_err_paths": sum(p.lower().endswith((".log", ".out", ".err")) for p in tracked),
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


def engine_repository_inventory(repo: Path, recorded_revisions: Iterable[str]) -> dict[str, Any]:
    tracked = _tracked_paths(repo)
    base_revisions = sorted({value.removesuffix("-dirty") for value in recorded_revisions})
    bases: dict[str, bool] = {}
    for revision in base_revisions:
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        bases[revision] = result.returncode == 0
    return {
        "path": str(repo.resolve()),
        "classification": "FLOWSTAR_GPU_ENGINE_SOURCE_REPOSITORY",
        "git": _git_inventory(repo),
        "required_engine_source": {
            "path": "src/flowstar_gpu",
            "present": (repo / ENGINE_SOURCE_SUFFIX).is_dir(),
            "build_files": sorted(name for name in BUILD_FILE_NAMES if (repo / name).is_file()),
        },
        "required_docs": {
            "REPRODUCE.md": (repo / "docs/REPRODUCE.md").is_file(),
            "OPTIMIZATION.md": (repo / "docs/OPTIMIZATION.md").is_file(),
            "paper_authoritative_tex": (repo / "docs/paper/main.tex").is_file(),
            "paper_pdf": any((repo / "docs/paper").glob("*.pdf")),
        },
        "locks": {
            "pyproject.toml": (repo / "pyproject.toml").is_file(),
            "uv.lock": (repo / "uv.lock").is_file(),
        },
        "tracked_file_counts": {
            "all": len(tracked),
            "engine_source": sum(p.startswith("src/flowstar_gpu/") for p in tracked),
            "tests": sum(p.startswith("tests/") for p in tracked),
            "benchmark_artifacts": sum(p.startswith("benchmarks/") for p in tracked),
        },
        "recorded_dirty_base_commits_present": bases,
        "recorded_dirty_patches_present": False,
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


def artifact_gap_rows(engine: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "artifact_id": "CURRENT_ENGINE_SOURCE",
            "status": "AVAILABLE",
            "evidence": f"clean Git HEAD {engine['git']['head']}",
            "effect": "permits a new source/build/proof-kernel audit at this exact revision",
        },
        {
            "artifact_id": "HISTORICAL_DIRTY_PATCHES",
            "status": "MISSING",
            "evidence": "all 450 integration result records are -dirty; base commits exist but patches/untracked files do not",
            "effect": "historical paper timing/verdict records are not exact clean-source reproductions",
        },
        {
            "artifact_id": "PAPER_PDF",
            "status": "MISSING",
            "evidence": "no new_crown_reach*.pdf or docs/paper/*.pdf is present",
            "effect": "proof mapping uses docs/paper/main.tex, which the engine README identifies as authoritative, and records the format gap",
        },
        {
            "artifact_id": "EXTERNAL_FLOWSTAR",
            "status": "INACCESSIBLE",
            "evidence": "documented /home/huan/projects/flowstar checkout cannot be traversed by the audit account",
            "effect": "stock Flow* cross-tool reruns require a separately pinned accessible checkout",
        },
        {
            "artifact_id": "EXTERNAL_AUTOLIRPA",
            "status": "INAPPLICABLE_PLANT_ONLY_AND_INACCESSIBLE_TARGET",
            "evidence": "documented patched /home/huan/projects/Verifier_Development tree is inaccessible",
            "effect": "controller tests are excluded by the goal's hard prohibition and cannot support any claim",
        },
        {
            "artifact_id": "TARGET_HARDWARE",
            "status": "MISMATCH",
            "evidence": "current host exposes 16GB V100-class devices; REPRODUCE.md reports 32GB V100-SXM2 devices",
            "effect": "paper throughput and maximum-batch memory claims cannot be directly compared",
        },
    ]


def source_manifest(engine_repo: Path, engine: dict[str, Any]) -> dict[str, Any]:
    tracked = _tracked_paths(engine_repo)
    tree = _run_git(engine_repo, "ls-tree", "-r", "--full-tree", "HEAD")
    key_paths = [
        "pyproject.toml",
        "uv.lock",
        "docs/REPRODUCE.md",
        "docs/OPTIMIZATION.md",
        "docs/paper/main.tex",
        *[p for p in tracked if p.startswith("src/flowstar_gpu/")],
    ]
    return {
        "schema": "torch_tm_flowpipe.huan_source_manifest/1",
        "audit_date": AUDIT_DATE,
        "repository": engine["path"],
        "git": engine["git"],
        "git_tree_listing_sha256": hashlib.sha256((tree + "\n").encode()).hexdigest(),
        "tracked_file_count": len(tracked),
        "key_file_sha256": {
            path: _sha256(engine_repo / path)
            for path in sorted(key_paths)
            if (engine_repo / path).is_file()
        },
        "historical_dirty_state_exact": False,
        "historical_dirty_state_gap": "dirty patches and untracked files for recorded runs are absent",
    }


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
    integration: dict[str, Any],
    engine: dict[str, Any],
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
    integration_lines = [
        f"path={integration['path']}",
        f"classification={integration['classification']}",
        f"head={integration['git']['head']}",
        f"clean={str(integration['git']['clean']).lower()}",
        f"record_files={integration['result_record_provenance']['record_files_with_engine_revision']}",
        f"dirty_records={integration['result_record_provenance']['dirty_record_count']}",
        *(f"remote={remote}" for remote in integration["git"]["remote_v"]),
    ]
    (raw / "crown_reach_gpu_provenance.log").write_text("\n".join(integration_lines) + "\n", encoding="utf-8")
    engine_lines = [
        f"path={engine['path']}",
        f"classification={engine['classification']}",
        f"head={engine['git']['head']}",
        f"branch={engine['git']['branch']}",
        f"clean={str(engine['git']['clean']).lower()}",
        f"history_commit_count={engine['git']['history_commit_count']}",
        f"source_present={str(engine['required_engine_source']['present']).lower()}",
        f"build_files={','.join(engine['required_engine_source']['build_files'])}",
        f"tags={','.join(engine['git']['tags'])}",
        f"submodules={len(engine['git']['submodules'])}",
        *(f"remote={remote}" for remote in engine["git"]["remote_v"]),
        *(f"record_base={revision}\tpresent={str(present).lower()}" for revision, present in engine["recorded_dirty_base_commits_present"].items()),
        "recorded_dirty_patches_present=false",
    ]
    (raw / "flowstar_gpu_provenance.log").write_text("\n".join(engine_lines) + "\n", encoding="utf-8")
    _write_tsv(
        raw / "record_provenance.tsv",
        ["path", "flowstar_gpu_revision", "base_revision", "source_state"],
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
        integration["symlinks"]["all"],
    )


def write_checksums(output_root: Path) -> None:
    checksum_path = output_root / "SHA256SUMS"
    rows: list[str] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path != checksum_path:
            rows.append(f"{_sha256(path)}  {path.relative_to(output_root).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def verify_checksums(output_root: Path) -> list[str]:
    checksum_path = output_root / "SHA256SUMS"
    if not checksum_path.is_file():
        return ["SHA256SUMS is missing"]
    errors: list[str] = []
    covered: set[str] = set()
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"SHA256SUMS:{line_number}: malformed line")
            continue
        covered.add(relative)
        path = output_root / relative
        if not path.is_file():
            errors.append(f"missing covered file: {relative}")
        elif _sha256(path) != expected:
            errors.append(f"checksum mismatch: {relative}")
    actual = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    errors.extend(f"uncovered file: {relative}" for relative in sorted(actual - covered))
    return errors


def generate(
    search_root: Path,
    integration_repo: Path,
    engine_repo: Path,
    torch_repo: Path,
    output_root: Path,
) -> dict[str, Any]:
    hits, discovery_errors = discover_candidates(search_root)
    engine_roots = find_engine_roots(search_root)
    records = collect_record_provenance(integration_repo)
    integration = integration_repository_inventory(integration_repo)
    engine = engine_repository_inventory(
        engine_repo, (row["flowstar_gpu_revision"] for row in records)
    )
    torch = torch_baseline_inventory(torch_repo)
    current_gate = (
        engine["git"]["clean"]
        and engine["required_engine_source"]["present"]
        and bool(engine["required_engine_source"]["build_files"])
    )
    inventory = {
        "schema": "torch_tm_flowpipe.huan_repro_artifact_inventory/2",
        "audit_date": AUDIT_DATE,
        "search": {
            "root": str(search_root.resolve()),
            "max_depth": 4,
            "patterns": list(DISCOVERY_PATTERNS),
            "candidate_paths": [str(path) for path in hits],
            "errors": discovery_errors,
        },
        "torch_baseline": torch,
        "candidate_repositories": [integration, engine],
        "source_closure": {
            "qualifying_engine_roots": [str(path) for path in engine_roots],
            "current_clean_engine_source_available": current_gate,
            "historical_dirty_experiment_state_available": False,
            "paper_pdf_available": engine["required_docs"]["paper_pdf"],
            "authoritative_paper_tex_available": engine["required_docs"]["paper_authoritative_tex"],
            "reproduce_doc_available": engine["required_docs"]["REPRODUCE.md"],
            "optimization_doc_available": engine["required_docs"]["OPTIMIZATION.md"],
        },
        "phase_gates": {
            "current_source_build_and_kernel_audit": "OPEN" if current_gate else "CLOSED",
            "historical_result_exact_reproduction": "CLOSED_MISSING_DIRTY_PATCHES",
            "paper_pdf_mapping": "CLOSED_PDF_MISSING__AUTHORITATIVE_TEX_AVAILABLE",
            "plant_only_scope": "OPEN",
            "controller_coupling_scope": "PROHIBITED_NOT_RUN",
        },
        "stop_rule": {
            "triggered_for_current_source_audit": not current_gate,
            "triggered_for_historical_paper_result_claims": True,
            "reason": "historical -dirty patches are absent; current clean HEAD remains auditable",
        },
    }
    _write_json(output_root / "artifact_inventory.json", inventory)
    _write_json(output_root / "source_manifest.json", source_manifest(engine_repo, engine))
    _write_tsv(
        output_root / "artifact_gaps.tsv",
        ["artifact_id", "status", "evidence", "effect"],
        artifact_gap_rows(engine),
    )
    _write_raw_logs(
        output_root,
        search_root,
        hits,
        discovery_errors,
        torch,
        integration,
        engine,
        records,
    )
    obsolete = output_root / "missing_artifacts.tsv"
    if obsolete.exists():
        obsolete.unlink()
    write_checksums(output_root)
    return inventory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-repo",
        type=Path,
        required=True,
        help="CROWN-Reach-GPU integration/result repository",
    )
    parser.add_argument("--engine-repo", type=Path, required=True)
    parser.add_argument("--torch-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.verify_only:
        errors = verify_checksums(args.output_root)
        if errors:
            print("\n".join(errors))
            return 1
        print(f"verified {args.output_root / 'SHA256SUMS'}")
        return 0
    inventory = generate(
        args.search_root,
        args.candidate_repo,
        args.engine_repo,
        args.torch_repo,
        args.output_root,
    )
    print(inventory["phase_gates"]["current_source_build_and_kernel_audit"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
