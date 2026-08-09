#!/usr/bin/env python3
"""Scan the complete clean worktree and its reachable history.

Exact paths, matching lines, and candidate inventories are private.  The
public output contains only aggregate counts, policy metadata, and hashes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Iterable


SENSITIVE_SUFFIXES = {
    ".onnx",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
}
LARGE_FILE_BYTES = 1_000_000
HIGH_ENTROPY_MIN_BYTES = 65_536
HIGH_ENTROPY_BITS_PER_BYTE = 7.5
ABSOLUTE_OR_CREDENTIAL_PATTERN = re.compile(rb"/(?:srv/local|home/[A-Za-z0-9_.-]+|root|mnt)/|[A-Za-z]:\\\\Users\\\\|BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|(?:password|passwd|api_key|secret_key)[ \t]*=[ \t]*[^ \t\r\n]+", re.IGNORECASE)
HERMETIC_MOUNT_PATTERN = re.compile(rb"/(?:workspace|workspaces|repo|src)/|/tmp/(?:inputs|observation)")

# Pattern definitions necessarily contain the strings that they detect.  This
# narrow self-signature exception is reported publicly and is not a path
# allowlist.  It applies only to the constant-definition lines in this file.
SCANNER_SIGNATURE_ALLOWLIST = {
    "path": "scripts/run_tora_q3_secret_scan.py",
    "line_prefixes": (
        b"ABSOLUTE_OR_CREDENTIAL_PATTERN = re.compile(",
        b"HERMETIC_MOUNT_PATTERN = re.compile(",
        b'"literal":',
    ),
    "reason": "scanner pattern definitions must contain their detection signatures",
}

# Intentional hermetic absolute mount paths must be enumerated here with an
# exact public-relative file, exact byte string, and a reviewable reason.  The
# clean branch currently has no such usage; an empty allowlist is deliberate.
HERMETIC_MOUNT_ALLOWLIST: tuple[dict[str, str], ...] = (
    {
        "path": "scripts/run_xiangru_q3_observation.py",
        "literal": "/tmp/inputs",
        "reason": "fixed read-only input mount inside the bwrap observation sandbox",
    },
    {
        "path": "scripts/run_xiangru_q3_observation.py",
        "literal": "/tmp/observation",
        "reason": "fixed writable output mount inside the bwrap observation sandbox",
    },
    {
        "path": "outputs/tora_q3_perf_closure_20260806/provenance/public_artifact_scan_summary.json",
        "literal": "/tmp/inputs",
        "reason": "sanitized public policy summary repeats the exact reviewed input-mount allowlist",
    },
    {
        "path": "outputs/tora_q3_perf_closure_20260806/provenance/public_artifact_scan_summary.json",
        "literal": "/tmp/observation",
        "reason": "sanitized public policy summary repeats the exact reviewed output-mount allowlist",
    },
    {
        "path": "outputs/tora_q3_stage_parity_fused_20260809/provenance/checkpoint1_publication_scan.json",
        "literal": "/tmp/inputs",
        "reason": "stage-parity sanitized policy summary repeats the exact reviewed input-mount allowlist",
    },
    {
        "path": "outputs/tora_q3_stage_parity_fused_20260809/provenance/checkpoint1_publication_scan.json",
        "literal": "/tmp/observation",
        "reason": "stage-parity sanitized policy summary repeats the exact reviewed output-mount allowlist",
    },
    {
        "path": "outputs/tora_q3_stage_parity_fused_20260809/provenance/checkpoint2_publication_scan.json",
        "literal": "/tmp/inputs",
        "reason": "checkpoint-2 sanitized policy summary repeats the exact reviewed input-mount allowlist",
    },
    {
        "path": "outputs/tora_q3_stage_parity_fused_20260809/provenance/checkpoint2_publication_scan.json",
        "literal": "/tmp/observation",
        "reason": "checkpoint-2 sanitized policy summary repeats the exact reviewed output-mount allowlist",
    },
    {
        "path": "outputs/tora_q3_stage_parity_fused_20260809/provenance/checkpoint3_publication_scan.json",
        "literal": "/tmp/inputs",
        "reason": "checkpoint-3 sanitized policy summary repeats the exact reviewed input-mount allowlist",
    },
    {
        "path": "outputs/tora_q3_stage_parity_fused_20260809/provenance/checkpoint3_publication_scan.json",
        "literal": "/tmp/observation",
        "reason": "checkpoint-3 sanitized policy summary repeats the exact reviewed output-mount allowlist",
    },
    {
        "path": "outputs/tora_q3_stage_parity_fused_20260809/provenance/checkpoint4_publication_scan.json",
        "literal": "/tmp/inputs",
        "reason": "checkpoint-4 sanitized policy summary repeats the exact reviewed input-mount allowlist",
    },
    {
        "path": "outputs/tora_q3_stage_parity_fused_20260809/provenance/checkpoint4_publication_scan.json",
        "literal": "/tmp/observation",
        "reason": "checkpoint-4 sanitized policy summary repeats the exact reviewed output-mount allowlist",
    },
)


def _run_bytes(root: Path, command: list[str], *, check: bool = True) -> bytes:
    completed = subprocess.run(command, cwd=root, capture_output=True)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command!r}\n"
            f"{completed.stderr.decode(errors='replace')}"
        )
    return completed.stdout


def _nul_paths(payload: bytes) -> list[str]:
    return [part.decode("utf-8", errors="surrogateescape") for part in payload.split(b"\0") if part]


def _tracked_paths(root: Path) -> list[str]:
    return _nul_paths(_run_bytes(root, ["git", "ls-files", "-z"]))


def _untracked_paths(root: Path) -> list[str]:
    return _nul_paths(
        _run_bytes(root, ["git", "ls-files", "--others", "--exclude-standard", "-z"])
    )


def _history_entries(root: Path) -> tuple[list[str], list[dict[str, str]]]:
    commits = _run_bytes(root, ["git", "rev-list", "HEAD"]).decode().splitlines()
    entries: list[dict[str, str]] = []
    for commit in commits:
        tree = _run_bytes(root, ["git", "ls-tree", "-r", "-z", commit])
        for record in tree.split(b"\0"):
            if not record:
                continue
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode().split()
            if object_type != "blob":
                continue
            entries.append(
                {
                    "commit": commit,
                    "mode": mode,
                    "object_id": object_id,
                    "path": raw_path.decode("utf-8", errors="surrogateescape"),
                }
            )
    return commits, entries


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    length = len(data)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def _is_scanner_signature(path: str, line: bytes) -> bool:
    if path != SCANNER_SIGNATURE_ALLOWLIST["path"]:
        return False
    stripped = line.lstrip()
    return any(
        stripped.startswith(prefix)
        for prefix in SCANNER_SIGNATURE_ALLOWLIST["line_prefixes"]
    )


def _hermetic_allowlist_reason(path: str, matched: bytes) -> str | None:
    decoded = matched.decode("utf-8", errors="replace")
    for entry in HERMETIC_MOUNT_ALLOWLIST:
        if entry["path"] == path and entry["literal"] == decoded:
            return entry["reason"]
    return None


def _text_matches(scope: str, path: str, data: bytes) -> list[dict[str, object]]:
    if b"\0" in data:
        return []
    matches: list[dict[str, object]] = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        for kind, pattern in (
            ("absolute_path_or_credential", ABSOLUTE_OR_CREDENTIAL_PATTERN),
            ("hermetic_mount", HERMETIC_MOUNT_PATTERN),
        ):
            for found in pattern.finditer(line):
                matched = found.group(0)
                allow_reason: str | None = None
                if _is_scanner_signature(path, line):
                    allow_reason = str(SCANNER_SIGNATURE_ALLOWLIST["reason"])
                elif kind == "hermetic_mount":
                    allow_reason = _hermetic_allowlist_reason(path, matched)
                matches.append(
                    {
                        "scope": scope,
                        "path": path,
                        "line_number": line_number,
                        "kind": kind,
                        "match": matched.decode("utf-8", errors="replace"),
                        "line": line.decode("utf-8", errors="replace"),
                        "allowlisted": allow_reason is not None,
                        "allowlist_reason": allow_reason,
                    }
                )
    return matches


def _file_record(scope: str, path: str, data: bytes) -> dict[str, object]:
    entropy = _entropy(data)
    return {
        "scope": scope,
        "path": path,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "suffix": PurePosixPath(path).suffix.lower(),
        "entropy_bits_per_byte": entropy,
        "sensitive_suffix": PurePosixPath(path).suffix.lower() in SENSITIVE_SUFFIXES,
        "large_file_candidate": len(data) >= LARGE_FILE_BYTES,
        "high_entropy_candidate": (
            len(data) >= HIGH_ENTROPY_MIN_BYTES
            and entropy >= HIGH_ENTROPY_BITS_PER_BYTE
        ),
    }


def _read_working_files(root: Path, paths: Iterable[str], scope: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in paths:
        target = root / path
        if not target.is_file() or target.is_symlink():
            continue
        data = target.read_bytes()
        records.append(
            {
                "file": _file_record(scope, path, data),
                "matches": _text_matches(scope, path, data),
            }
        )
    return records


def scan_repository(root: Path) -> dict[str, object]:
    tracked_paths = _tracked_paths(root)
    untracked_paths = _untracked_paths(root)
    working = _read_working_files(root, tracked_paths, "working_tracked_tree")
    untracked = _read_working_files(root, untracked_paths, "working_untracked_tree")
    commits, history_entries = _history_entries(root)

    blob_cache: dict[str, bytes] = {}
    history: list[dict[str, object]] = []
    for entry in history_entries:
        object_id = entry["object_id"]
        if object_id not in blob_cache:
            blob_cache[object_id] = _run_bytes(root, ["git", "cat-file", "blob", object_id])
        data = blob_cache[object_id]
        scope = f"reachable_history:{entry['commit']}"
        history.append(
            {
                **entry,
                "file": _file_record(scope, entry["path"], data),
                "matches": _text_matches(scope, entry["path"], data),
            }
        )

    all_records = [*working, *untracked, *history]
    file_records = [record["file"] for record in all_records]
    matches = [match for record in all_records for match in record["matches"]]
    unallowlisted = [match for match in matches if not match["allowlisted"]]
    sensitive = [record for record in file_records if record["sensitive_suffix"]]
    large = [record for record in file_records if record["large_file_candidate"]]
    high_entropy = [record for record in file_records if record["high_entropy_candidate"]]
    return {
        "schema": "tora_q3_private_complete_artifact_scan_v3",
        "root": str(root),
        "tracked_paths": tracked_paths,
        "untracked_paths": untracked_paths,
        "reachable_commits": commits,
        "reachable_history_entries": history_entries,
        "working_records": working,
        "untracked_records": untracked,
        "history_records": history,
        "matches": matches,
        "unallowlisted_matches": unallowlisted,
        "sensitive_suffix_candidates": sensitive,
        "large_file_candidates": large,
        "high_entropy_candidates": high_entropy,
        "unique_reachable_blob_count": len(blob_cache),
    }


def _public_summary(raw: bytes, scan: dict[str, object]) -> dict[str, object]:
    working = scan["working_records"]
    untracked = scan["untracked_records"]
    history = scan["history_records"]
    matches = scan["matches"]
    unallowlisted = scan["unallowlisted_matches"]
    sensitive = scan["sensitive_suffix_candidates"]
    large = scan["large_file_candidates"]
    high_entropy = scan["high_entropy_candidates"]
    current_sensitive = [
        record for record in sensitive
        if record["scope"] in {"working_tracked_tree", "working_untracked_tree"}
    ]
    return {
        "schema": "tora_q3_public_artifact_scan_summary_v3",
        "scanner_scope": "whole working tracked tree, whole untracked tree, and every blob/path in git rev-list HEAD",
        "reachable_history_ref": "HEAD only (the isolated clean lineage); local unrelated refs are intentionally out of scope",
        "raw_private_log_sha256": hashlib.sha256(raw).hexdigest(),
        "working_tracked_file_count": len(working),
        "working_untracked_file_count": len(untracked),
        "reachable_commit_count": len(scan["reachable_commits"]),
        "reachable_history_path_entry_count": len(history),
        "unique_reachable_blob_count": scan["unique_reachable_blob_count"],
        "pattern_match_count": len(matches),
        "allowlisted_scanner_signature_match_count": sum(
            bool(match["allowlisted"]) for match in matches
        ),
        "unallowlisted_path_or_credential_match_count": len(unallowlisted),
        "sensitive_suffix_candidate_count_all_scopes": len(sensitive),
        "current_tree_sensitive_suffix_candidate_count": len(current_sensitive),
        "large_file_candidate_count_all_scopes": len(large),
        "high_entropy_candidate_count_all_scopes": len(high_entropy),
        "intentional_hermetic_mount_allowlist": [
            {"path": entry["path"], "literal": entry["literal"], "reason": entry["reason"]}
            for entry in HERMETIC_MOUNT_ALLOWLIST
        ],
        "intentional_hermetic_mount_allowlist_reason": (
            "No intentional hermetic absolute mount path is present in this clean branch."
            if not HERMETIC_MOUNT_ALLOWLIST
            else "Every intentional hermetic mount is enumerated exactly; no pattern construction bypass is used."
        ),
        "large_file_threshold_bytes": LARGE_FILE_BYTES,
        "high_entropy_threshold": {
            "minimum_bytes": HIGH_ENTROPY_MIN_BYTES,
            "bits_per_byte": HIGH_ENTROPY_BITS_PER_BYTE,
        },
        "credential_claim": "Pattern scanning is a fail-closed publication gate, not proof that credentials can never exist.",
        "governance_status": (
            "PASS_CLEAN_LINEAGE"
            if not unallowlisted and not current_sensitive
            else "FAIL_PUBLICATION_GATE"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--private-log", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    scan = scan_repository(root)
    raw = (json.dumps(scan, indent=2, sort_keys=True) + "\n").encode()
    summary = _public_summary(raw, scan)
    args.private_log.parent.mkdir(parents=True, exist_ok=True)
    args.private_log.write_bytes(raw)
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    args.public_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["governance_status"] == "PASS_CLEAN_LINEAGE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
