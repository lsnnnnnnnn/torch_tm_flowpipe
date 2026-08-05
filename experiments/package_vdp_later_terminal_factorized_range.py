#!/usr/bin/env python3
"""Create the deterministic tracked evidence package for the later terminal."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "vdp_later_terminal_factorized_range"
PACKAGE = OUTPUT / "evidence_package"
IMPLEMENTATION_SHA = "a1fb3527bb7c12ce23aa2fb49d66f6380c463c90"
BASELINE_SHA = "cdb54bd3d2ffb49a0b58245055932756ebc3aa47"
CHECKPOINT_SHA = "dcb8f646d45c9742e0cff23fea596c12e53d8ccd00d1544f70564a44a7463420"
COMPRESS_THRESHOLD = 256_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b""))


def _copy_or_compress(source: Path, destination: Path, records: list[dict[str, Any]]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_size >= COMPRESS_THRESHOLD:
        stored = destination.with_name(destination.name + ".gz")
        stored.write_bytes(gzip.compress(source.read_bytes(), compresslevel=9, mtime=0))
        records.append(
            {
                "source": str(source.relative_to(OUTPUT)),
                "stored": str(stored.relative_to(OUTPUT)),
                "source_bytes": source.stat().st_size,
                "source_sha256": _sha256(source),
                "source_lines": _line_count(source),
                "stored_bytes": stored.stat().st_size,
                "stored_sha256": _sha256(stored),
                "compression": "gzip-9-mtime-0",
                "decompress": f"gzip -dc {stored.relative_to(OUTPUT)}",
            }
        )
        return stored
    shutil.copy2(source, destination)
    records.append(
        {
            "source": str(source.relative_to(OUTPUT)),
            "stored": str(destination.relative_to(OUTPUT)),
            "source_bytes": source.stat().st_size,
            "source_sha256": _sha256(source),
            "source_lines": _line_count(source) if source.suffix in {".csv", ".jsonl", ".txt", ".yaml"} else None,
            "stored_bytes": destination.stat().st_size,
            "stored_sha256": _sha256(destination),
            "compression": "none",
        }
    )
    return destination


def _verify_stage_hashes(lane_dir: Path) -> None:
    ranges = _jsonl(lane_dir / "range_context_trace.jsonl")
    stages = _jsonl(lane_dir / "horner_stage_trace.jsonl")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for stage in stages:
        compact = dict(stage)
        call = int(compact.pop("range_call_index"))
        compact.pop("context", None)
        grouped.setdefault(call, []).append(compact)
    empty_hash = hashlib.sha256(b"[]").hexdigest()
    for row in ranges:
        call = int(row["range_call_index"])
        expected_count = int(row["horner_stage_count"])
        actual = grouped.get(call, [])
        if len(actual) != expected_count:
            raise RuntimeError(f"Horner stage count mismatch: {lane_dir.name} call={call}")
        encoded = json.dumps(actual, sort_keys=True, separators=(",", ":")).encode("utf-8")
        actual_hash = hashlib.sha256(encoded).hexdigest() if actual else empty_hash
        if actual_hash != row["horner_stage_sha256"]:
            raise RuntimeError(f"Horner stage SHA mismatch: {lane_dir.name} call={call}")


def _verify_formal_runs() -> dict[str, Any]:
    commands: list[dict[str, Any]] = []
    for phase in ("attribution", "terminal_ab"):
        root = OUTPUT / phase / "formal"
        for lane in sorted(path for path in root.iterdir() if path.is_dir()):
            _verify_stage_hashes(lane)
            command = _json(lane / "command.json")
            commands.append({"phase": phase, "lane": lane.name, **command})
    commits = {row["commit"] for row in commands}
    dirty = [row["lane"] for row in commands if row["worktree_status"]]
    if commits != {IMPLEMENTATION_SHA} or dirty:
        raise RuntimeError(f"formal run provenance mismatch: commits={commits}, dirty={dirty}")
    terminal = _json(OUTPUT / "terminal_decision.json")
    if terminal["stop_go_gate"] != "STOP" or terminal["fresh_horizons_authorized"]:
        raise RuntimeError("terminal STOP/GO evidence is inconsistent")
    with (OUTPUT / "fresh_horizons.csv").open(encoding="utf-8") as handle:
        horizons = list(csv.DictReader(handle))
    if any(row["status"] != "not_run_stop_go_gate_failed" for row in horizons):
        raise RuntimeError("fresh horizon evidence violates STOP gate")
    return {
        "passed": True,
        "formal_run_count": len(commands),
        "formal_run_commit": IMPLEMENTATION_SHA,
        "all_formal_worktrees_clean": not dirty,
        "stage_trace_hashes_verified": True,
        "stop_gate_verified": True,
        "fresh_runs_absent_by_gate": True,
        "commands": commands,
    }


def _copy_raw_package(records: list[dict[str, Any]]) -> None:
    raw = PACKAGE / "raw"
    for phase in ("attribution", "terminal_ab"):
        source_root = OUTPUT / phase / "formal"
        for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
            _copy_or_compress(source, raw / phase / source.relative_to(source_root), records)


def _write_metadata(verification: Mapping[str, Any]) -> None:
    _write_json(
        OUTPUT / "test_results.json",
        {
            "baseline": {
                "command": "pytest -q",
                "git_sha": BASELINE_SHA,
                "result": "401 passed, 1 skipped in 45.49s",
            },
            "focused": {
                "command": "pytest -q tests/test_dense_horner_range.py tests/test_vdp_later_terminal_factorized_range.py tests/test_dense_range_subdivision.py tests/test_dense_range_policy.py tests/test_batched_dense_runner_contract.py",
                "result": "85 passed in 13.35s",
            },
            "final_full": {
                "command": "pytest -q",
                "result": "434 passed, 1 skipped in 73.56s",
            },
            "cuda": {
                "command": "pytest -q tests/test_dense_horner_range.py -k cuda",
                "result": "3 passed, 28 deselected in 1.83s",
                "devices_available": 4,
            },
        },
    )
    _write_json(
        OUTPUT / "policy_preregistration.json",
        {
            "phase_e_status": "not_entered",
            "implementation_commit": IMPLEMENTATION_SHA,
            "selected_range_policy": None,
            "registered_variable_orders": [[0, 1, 2], [1, 0, 2], [2, 0, 1]],
            "deterministic_order_selection_rule": "minimum width then lexicographic variable order",
            "acceptance_predicate": "flowstar_raw_remainder_compat",
            "fresh_sequence": [0.1, 1.0, 6.5, 7.5, 10.0, 10.0],
            "fresh_policy_preregistered": False,
            "reason": "D2 and D3 both rejected the frozen terminal; STOP gate forbids Phase E and fresh runs",
            "policy_changed_after_viewing_results": False,
        },
    )
    _write_json(OUTPUT / "provenance" / "formal_verification.json", verification)
    _write_json(
        OUTPUT / "provenance" / "git_start_state.json",
        {
            "authoritative_branch": "codex/vdp-terminal-range-closure-20260805",
            "expected_and_verified_remote_sha": BASELINE_SHA,
            "ls_remote_sha": BASELINE_SHA,
            "starting_primary_worktree_branch": "codex/flowstar-raw-remainder-compat",
            "starting_primary_worktree_sha": "26a254ef585a9dee394b7e41922c06bf8799f501",
            "starting_primary_worktree_user_changes_preserved": [
                "README.md",
                "experiments/flowstar_probe/flowstar_vdp_step_trace_probe.cpp",
                "experiments/flowstar_step_trace_compare.py",
                "tests/test_flowstar_step_trace_compare_alignment.py",
                "docs/flowstar_order2_vanderpol_failure.md",
            ],
            "isolated_worktree": str(ROOT),
            "isolated_branch": "codex/vdp-later-terminal-factorized-range-20260805",
            "fetch_origin_prune_completed": True,
            "main_modified": False,
        },
    )
    source_paths = (
        ROOT / "src" / "torch_tm_flowpipe" / "batched_dense_tm.py",
        ROOT / "src" / "torch_tm_flowpipe" / "flowpipe.py",
        ROOT / "src" / "torch_tm_flowpipe" / "terminal_checkpoint.py",
        ROOT / "experiments" / "replay_vdp_terminal_range.py",
        ROOT / "experiments" / "run_vdp_dense_backend.py",
        ROOT / "experiments" / "run_vdp_later_terminal_factorized_range.py",
        ROOT / "experiments" / "package_vdp_later_terminal_factorized_range.py",
        ROOT / "tests" / "test_dense_horner_range.py",
        ROOT / "tests" / "test_vdp_later_terminal_factorized_range.py",
        ROOT / "benchmarks" / "canonical.yaml",
        ROOT / "benchmarks" / "three_tool_matched_contract.yaml",
    )
    _write_json(
        OUTPUT / "provenance" / "source_hashes.json",
        {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths},
    )
    _write_json(
        OUTPUT / "provenance" / "packaging_environment.json",
        {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "platform": platform.platform(),
            "git_sha": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "git_status": _git("status", "--short"),
            "implementation_sha_used_by_formal_runs": IMPLEMENTATION_SHA,
            "safeguard_claim": "safeguarded float64 enclosure; not a hardware-independent directed-rounding formal proof",
        },
    )


def _aggregate_compression(records: list[dict[str, Any]]) -> None:
    for name in ("horner_stage_trace.jsonl", "horner_stage_trace.csv"):
        source = OUTPUT / name
        _copy_or_compress(source, OUTPUT / name, records)
    _write_json(OUTPUT / "compression_manifest.json", records)


def _manifest(records: Sequence[Mapping[str, Any]], verification: Mapping[str, Any]) -> None:
    selected_names = {
        "attribution.csv",
        "attribution.json",
        "attribution_decision.json",
        "terminal_ab.csv",
        "terminal_ab.json",
        "terminal_decision.json",
        "summary.csv",
        "range_context_trace.csv",
        "range_context_trace.jsonl",
        "horner_stage_trace.csv.gz",
        "horner_stage_trace.jsonl.gz",
        "fresh_horizons.csv",
        "environment.json",
        "policy_preregistration.json",
        "test_results.json",
        "compression_manifest.json",
    }
    paths = [OUTPUT / name for name in sorted(selected_names)]
    paths.extend(sorted(path for path in PACKAGE.rglob("*") if path.is_file()))
    paths.extend(sorted(path for path in (OUTPUT / "provenance").rglob("*") if path.is_file()))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    files = [
        {
            "path": str(path.relative_to(OUTPUT)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "lines": _line_count(path) if path.suffix in {".csv", ".json", ".jsonl", ".txt"} else None,
        }
        for path in paths
    ]
    _write_json(
        OUTPUT / "manifest.json",
        {
            "schema": "torch_tm_flowpipe_vdp_later_terminal_factorized_range_v1",
            "baseline_sha": BASELINE_SHA,
            "implementation_sha_used_by_formal_runs": IMPLEMENTATION_SHA,
            "packaging_sha": _git("rev-parse", "HEAD"),
            "checkpoint_full_sha256": CHECKPOINT_SHA,
            "state": "H1_factorized_range_correctness_complete_with_R4_global_horizon",
            "fresh_runs_performed": False,
            "highest_validated_horizon": 6.397083942944808,
            "formal_verification": dict(verification),
            "compressed_sources": list(records),
            "files": files,
        },
    )
    checksum_paths = [*paths, OUTPUT / "manifest.json"]
    _write_text(
        OUTPUT / "SHA256SUMS",
        "".join(f"{_sha256(path)}  {path.relative_to(OUTPUT)}\n" for path in checksum_paths),
    )
def main() -> int:
    try:
        if PACKAGE.exists() and any(PACKAGE.iterdir()):
            raise FileExistsError(f"refusing non-empty evidence package: {PACKAGE}")
        if _git("status", "--short"):
            raise RuntimeError("packaging requires a clean worktree")
        verification = _verify_formal_runs()
        records: list[dict[str, Any]] = []
        _copy_raw_package(records)
        _write_metadata(verification)
        _aggregate_compression(records)
        _manifest(records, verification)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(OUTPUT), "files": len(list(PACKAGE.rglob('*')))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
