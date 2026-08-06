#!/usr/bin/env python3
"""Build the reviewed public working-tree artifact manifest fail closed."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SENSITIVE_SUFFIXES = {".onnx", ".pt", ".pth", ".ckpt", ".safetensors"}
ROOT_FILES = (
    "CLEAN_REVIEW_PUBLICATION.md",
    "TORA_Q3_NATIVE_TORCH_IMPLEMENTATION_REPORT.md",
    "TORA_Q3_COMMON_CONTROL_COMPARISON_REPORT.md",
    "TORA_Q3_FULL_CLOSED_LOOP_COMPARISON_REPORT.md",
    "TORA_Q3_RUNTIME_REPORT.md",
    "SINE_TM_SOUNDNESS_REPORT.md",
    "PUBLIC_ARTIFACT_GOVERNANCE_AUDIT.md",
    "handoff.md",
    "public_artifact_inventory.csv",
    "license_and_origin_map.csv",
)
IMPLEMENTATION_FILES = (
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
    "scripts/build_tora_q3_contracts.py",
    "scripts/audit_tora_q3_final_requirements.py",
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
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    output = args.output.resolve()
    files = []
    for relative in (*ROOT_FILES, *IMPLEMENTATION_FILES):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(path)
    artifact_root = root / "outputs/tora_q3_native_matched_20260806"
    for path in sorted(artifact_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed in public manifest: {path}")
        if path.is_file() and path.resolve() != output:
            files.append(path)
    unique = sorted(set(files))
    sensitive = [
        path for path in unique if path.suffix.lower() in SENSITIVE_SUFFIXES
    ]
    if sensitive:
        raise ValueError(f"sensitive binary in manifest scope: {sensitive}")
    lines = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in unique
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"manifested {len(lines)} reviewed public working-tree files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
