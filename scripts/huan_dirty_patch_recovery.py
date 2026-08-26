#!/usr/bin/env python3
"""Bounded, deterministic recovery search for recorded flowstar_gpu dirty states."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Iterable


DIRTY_RE = re.compile(r"\b([0-9a-f]{40})-dirty\b")
SKIP_DIRS = {
    ".git", ".venv", "node_modules", "docker-rootless-data", "miniforge3",
    ".cache", "native_envs", "__pycache__",
}
SEARCH_NAME_RE = re.compile(
    r"(?:\.patch$|\.diff$|source.*\.(?:tar|tgz|tar\.gz|tar\.xz)$|"
    r"flowstar.*manifest|manifest.*flowstar|(?:slurm|tmux).*(?:log|out)$)",
    re.IGNORECASE,
)


def _run(command: list[str], cwd: Path) -> dict[str, object]:
    result = subprocess.run(
        command, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "command": command,
        "cwd": str(cwd.resolve()),
        "returncode": result.returncode,
        "output": result.stdout,
    }


def dirty_bases(record_path: Path) -> list[str]:
    return sorted(set(DIRTY_RE.findall(record_path.read_text(encoding="utf-8"))))


def bounded_files(root: Path, max_depth: int = 6) -> Iterable[Path]:
    root = root.resolve()
    for current, dirs, files in os.walk(root, onerror=lambda _error: None):
        current_path = Path(current)
        depth = len(current_path.relative_to(root).parts)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and depth < max_depth)
        for name in sorted(files):
            yield current_path / name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_repositories(search_root: Path, bases: list[str]) -> list[Path]:
    candidates: set[Path] = set()
    for path in bounded_files(search_root, max_depth=4):
        if path.name == ".git":
            candidates.add(path.parent)
    # os.walk does not yield .git directories after pruning, so include
    # top-level project-shaped directories explicitly.
    for path in search_root.glob("*"):
        if path.is_dir() and (path / ".git").exists():
            candidates.add(path)
    selected: list[Path] = []
    for repo in sorted(candidates):
        remote = _run(["git", "remote", "-v"], repo)
        has_base = any(
            _run(["git", "cat-file", "-e", f"{base}^{{commit}}"], repo)["returncode"] == 0
            for base in bases
        )
        if "flowstar-gpu" in str(remote["output"]).lower() or has_base:
            selected.append(repo)
    return selected


def recover(search_root: Path, record_path: Path) -> tuple[dict[str, object], list[dict[str, str]]]:
    bases = dirty_bases(record_path)
    repositories = discover_repositories(search_root, bases)
    candidates: list[dict[str, str]] = []
    git_evidence: list[dict[str, object]] = []

    for repo in repositories:
        commands = [
            ["git", "stash", "list", "--date=iso"],
            ["git", "reflog", "--all", "--date=iso"],
            ["git", "fsck", "--full", "--no-reflogs", "--unreachable"],
            ["git", "worktree", "list", "--porcelain"],
        ]
        records = [_run(command, repo) for command in commands]
        git_evidence.append({"repository": str(repo.resolve()), "commands": records})
        combined = "\n".join(str(record["output"]) for record in records)
        for base in bases:
            if base in combined:
                candidates.append({
                    "base_revision": base,
                    "candidate_type": "git_metadata_reference",
                    "path_or_object": str(repo.resolve()),
                    "exact_match": "NO",
                    "reason": "base appears in stash/reflog/fsck/worktree output; no dirty patch identity established",
                })
        unreachable = re.findall(r"unreachable commit ([0-9a-f]{40})", combined)
        for obj in unreachable:
            parents = _run(["git", "show", "-s", "--format=%P", obj], repo)
            parent_set = str(parents["output"]).split()
            for base in bases:
                if base in parent_set:
                    candidates.append({
                        "base_revision": base,
                        "candidate_type": "unreachable_child_commit",
                        "path_or_object": f"{repo}:{obj}",
                        "exact_match": "CANDIDATE",
                        "reason": "unreachable commit directly descends from dirty base; requires tree/diff comparison",
                    })

    searched_files = 0
    for path in bounded_files(search_root, max_depth=6):
        searched_files += 1
        rel = path.relative_to(search_root).as_posix()
        name_match = SEARCH_NAME_RE.search(path.name)
        history_like = path.name in {".bash_history", ".zsh_history"}
        if not name_match and not history_like:
            continue
        try:
            if path.stat().st_size > 50 * 1024 * 1024:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        text = data.decode("utf-8", errors="ignore")
        mentioned = [base for base in bases if base in text]
        if mentioned or name_match:
            for base in mentioned or ["ALL_RECORDED_BASES"]:
                candidates.append({
                    "base_revision": base,
                    "candidate_type": "archive_or_log_file",
                    "path_or_object": rel,
                    "exact_match": "CANDIDATE" if base in mentioned else "NO",
                    "reason": (
                        "file mentions the base revision; content does not include a verified tracked+untracked dirty snapshot"
                        if base in mentioned else
                        "name matched bounded patch/archive/log search but mentioned none of the three recorded bases"
                    ),
                    "sha256": _sha256(path),
                })

    exact = [row for row in candidates if row.get("exact_match") == "YES"]
    partial = [row for row in candidates if row.get("exact_match") == "CANDIDATE"]
    conclusion = (
        "HISTORICAL_DIRTY_PATCHES_RECOVERED_EXACTLY" if exact and len({row["base_revision"] for row in exact}) == len(bases)
        else "HISTORICAL_DIRTY_PATCHES_PARTIALLY_RECOVERED" if exact
        else "HISTORICAL_DIRTY_PATCHES_NOT_FOUND_AFTER_BOUNDED_SEARCH"
    )
    payload: dict[str, object] = {
        "schema": "torch_tm_flowpipe.huan_dirty_patch_recovery/1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "search_root": str(search_root.resolve()),
        "record_path": str(record_path.resolve()),
        "dirty_base_revisions": bases,
        "repositories_searched": [str(repo.resolve()) for repo in repositories],
        "git_evidence": git_evidence,
        "bounded_file_count": searched_files,
        "candidate_count": len(candidates),
        "unresolved_candidate_count": len(partial),
        "conclusion": conclusion,
        "interpretation": "A base commit or a log mentioning '-dirty' is not the missing tracked diff plus untracked-file snapshot.",
    }
    return payload, candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--record-provenance", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    payload, candidates = recover(args.search_root, args.record_provenance)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ("base_revision", "candidate_type", "path_or_object", "exact_match", "reason", "sha256")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)
    print(json.dumps({"conclusion": payload["conclusion"], "candidates": len(candidates)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
