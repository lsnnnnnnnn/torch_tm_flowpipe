#!/usr/bin/env python3
"""Write an unsanitized Git artifact/path scan privately and a public digest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def run(root: Path, command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--private-log", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    sensitive_pattern = (
        r"(/" + r"srv/local/|/" + r"home/[A-Za-z0-9_.-]+/|"
        r"BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|AKIA[0-9A-Z]{16})"
    )
    records = [
        run(root, ["git", "ls-files", "*.onnx", "*.pt", "*.pth", "*.ckpt", "*.safetensors"]),
        run(root, ["git", "log", "--all", "--format=%H %ad %s", "--date=iso-strict", "--", "*.onnx", "*.pt", "*.pth", "*.ckpt", "*.safetensors"]),
        run(root, ["git", "grep", "-I", "-n", "-E", sensitive_pattern, "HEAD"]),
        run(root, ["git", "rev-list", "--objects", "--all"]),
        run(root, ["git", "ls-files", "--others", "--exclude-standard"]),
        run(
            root,
            [
                "rg", "-n", "-I", "--hidden", "--glob", "!.git/**",
                sensitive_pattern,
                "outputs/tora_q3_native_matched_20260806",
                "TORA_Q3_NATIVE_TORCH_IMPLEMENTATION_REPORT.md",
                "TORA_Q3_COMMON_CONTROL_COMPARISON_REPORT.md",
                "TORA_Q3_FULL_CLOSED_LOOP_COMPARISON_REPORT.md",
                "TORA_Q3_RUNTIME_REPORT.md",
                "SINE_TM_SOUNDNESS_REPORT.md",
                "PUBLIC_ARTIFACT_GOVERNANCE_AUDIT.md",
                "handoff.md",
                "src/torch_tm_flowpipe/tora_q3.py",
                "src/torch_tm_flowpipe/tora_controller.py",
                "src/torch_tm_flowpipe/batched_dense_tm.py",
                "src/torch_tm_flowpipe/__init__.py",
                "experiments/benchmark_tora_q3_backend.py",
                "experiments/benchmark_tora_q3_common_control_runtime.py",
                "experiments/benchmark_tora_controller_runtime.py",
                "experiments/run_tora_q3_common_control_replay.py",
                "experiments/run_tora_q3_full_closed_loop.py",
                "scripts/analyze_tora_q3_full_closed_loop.py",
                "scripts/analyze_tora_q3_plant_gates.py",
                "scripts/analyze_xiangru_observation_equivalence.py",
                "scripts/audit_tora_q3_final_requirements.py",
                "scripts/build_tora_q3_contracts.py",
                "scripts/build_tora_q3_public_manifest.py",
                "scripts/build_tora_q3_root_cause_summary.py",
                "scripts/capture_private_observer_patch.py",
                "scripts/collect_tora_q3_final_state.py",
                "scripts/collect_tora_q3_phase0.py",
                "scripts/compare_tora_q3_common_control.py",
                "scripts/export_sine_tm_cases.py",
                "scripts/run_tora_q3_final_quality_gates.py",
                "scripts/run_tora_q3_one_step_diagnostic.py",
                "scripts/run_tora_q3_secret_scan.py",
                "scripts/run_xiangru_q3_observation.py",
                "scripts/summarize_tora_q3_runtime.py",
                "tests/test_sine_tm.py",
                "tests/test_tora_comparator.py",
                "tests/test_tora_controller.py",
                "tests/test_tora_q3.py",
                "tests/test_tora_runtime_scope.py",
                "tests/test_xiangru_q3_matched_audit.py",
                "pyproject.toml",
            ],
        ),
    ]
    raw = (json.dumps(records, indent=2, sort_keys=True) + "\n").encode()
    args.private_log.parent.mkdir(parents=True, exist_ok=True)
    args.private_log.write_bytes(raw)
    tracked_binary_count = len(records[0]["stdout"].splitlines())
    path_match_count = len(records[2]["stdout"].splitlines())
    untracked = records[4]["stdout"].splitlines()
    sensitive_suffixes = {".onnx", ".pt", ".pth", ".ckpt", ".safetensors"}
    new_sensitive_binary_count = sum(
        Path(path).suffix.lower() in sensitive_suffixes for path in untracked
    )
    new_deliverable_match_count = len(records[5]["stdout"].splitlines())
    summary = {
        "schema": "tora_q3_public_artifact_scan_summary_v2",
        "raw_private_log_sha256": hashlib.sha256(raw).hexdigest(),
        "tracked_sensitive_binary_count": tracked_binary_count,
        "tracked_path_or_secret_pattern_match_count": path_match_count,
        "new_untracked_sensitive_binary_count": new_sensitive_binary_count,
        "new_deliverable_path_or_secret_pattern_match_count": new_deliverable_match_count,
        "scanner": "git history/index/object inventory + untracked inventory + scoped ripgrep of new deliverables",
        "credential_claim": "No credential conclusion is made from pattern scanning alone.",
        "governance_status": "BLOCKED_UNKNOWN_AUTHORIZATION",
    }
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    args.public_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
