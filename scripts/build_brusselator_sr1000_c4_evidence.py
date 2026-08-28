#!/usr/bin/env python3
"""Build the frozen SR1000/operator-ledger/C4 closure package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_brusselator_sr1000_c4_evidence import (  # noqa: E402
    BASELINE_COMMIT,
    C4_COMMIT,
    CORE_PATHS,
    DEFAULT_PACKAGE,
    recompute,
    sha256,
)


DEFAULT_SR1000 = Path("/srv/local/shengenli/brusselator_sr1000_parity_20260828")
DEFAULT_FLOWSTAR = Path("/srv/local/shengenli/brusselator_step1_operator_trace_20260828")
DEFAULT_C4 = Path("/srv/local/shengenli/brusselator_c4_same_input_20260828_v4")
FROZEN_STOCK = ROOT / "artifacts/runs/brusselator_generic_core_validation_20260827/raw/flowstar/segments.csv"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _copy(
    source: Path,
    destination: Path,
    output: Path,
    manifest: list[dict[str, Any]],
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    manifest.append(
        {
            "path": destination.relative_to(output).as_posix(),
            "source": str(source),
            "size": destination.stat().st_size,
            "sha256": sha256(destination),
            "encoding": "identity",
        }
    )


def _gzip_copy(
    source: Path,
    destination: Path,
    output: Path,
    manifest: list[dict[str, Any]],
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=destination_handle, mtime=0) as compressed:
            shutil.copyfileobj(source_handle, compressed)
    manifest.append(
        {
            "path": destination.relative_to(output).as_posix(),
            "source": str(source),
            "size": destination.stat().st_size,
            "sha256": sha256(destination),
            "uncompressed_size": source.stat().st_size,
            "uncompressed_sha256": sha256(source),
            "encoding": "gzip-mtime-0",
        }
    )


def build(sr1000: Path, flowstar: Path, c4: Path, output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    copies = (
        (ROOT / "benchmarks/brusselator_terminal_sr1000_contract.json", "raw/contract.json"),
        (sr1000 / "command.json", "raw/sr1000/command.json"),
        (sr1000 / "summary.json", "raw/sr1000/summary.json"),
        (sr1000 / "diagnostics.jsonl.gz", "raw/sr1000/diagnostics.jsonl.gz"),
        (
            sr1000 / "material_candidate_step_1_prestate/terminal_state.json",
            "raw/same_input_prestate/terminal_state.json",
        ),
        (
            sr1000 / "material_candidate_step_1_prestate/terminal_state_manifest.json",
            "raw/same_input_prestate/terminal_state_manifest.json",
        ),
        (
            sr1000 / "terminal_checkpoint_before/terminal_state.json",
            "raw/sr1000/terminal_checkpoint_before/terminal_state.json",
        ),
        (
            sr1000 / "terminal_checkpoint_before/terminal_state_manifest.json",
            "raw/sr1000/terminal_checkpoint_before/terminal_state_manifest.json",
        ),
        (
            sr1000 / "terminal_checkpoint_after/terminal_state.json",
            "raw/sr1000/terminal_checkpoint_after/terminal_state.json",
        ),
        (
            sr1000 / "terminal_checkpoint_after/terminal_state_manifest.json",
            "raw/sr1000/terminal_checkpoint_after/terminal_state_manifest.json",
        ),
        (flowstar / "compose_probe_result.json", "raw/flowstar/compose_probe_result.json"),
        (flowstar / "observed_summary.json", "raw/flowstar/observed_summary.json"),
        (flowstar / "binary.sha256", "raw/flowstar/binary.sha256"),
        (
            ROOT / "experiments/flowstar_step1_stage_observer_20260813.patch",
            "raw/flowstar/observer.patch",
        ),
        (
            ROOT / "experiments/flowstar_probe/flowstar_brusselator_step1_compose_probe.cpp",
            "raw/flowstar/flowstar_brusselator_step1_compose_probe.cpp",
        ),
        (c4 / "RESULT.json", "raw/c4/RESULT.json"),
        (c4 / "operator_ledger.json", "raw/c4/operator_ledger.json"),
        (c4 / "same_input_gate.json", "raw/c4/same_input_gate.json"),
        (c4 / "provenance.json", "raw/c4/provenance.json"),
        (c4 / "MANIFEST.json", "raw/c4/MANIFEST.json"),
    )
    for source, relative in copies:
        _copy(source, output / relative, output, manifest)

    compressed_copies = (
        (sr1000 / "segments.csv", "raw/sr1000/segments.csv.gz"),
        (FROZEN_STOCK, "raw/flowstar/frozen_stock_segments.csv.gz"),
        (flowstar / "observed.csv", "raw/flowstar/observed.csv.gz"),
        (flowstar / "unobserved.csv", "raw/flowstar/unobserved.csv.gz"),
        (flowstar / "observed_trace.jsonl", "raw/flowstar/observed_trace.jsonl.gz"),
        (flowstar / "compose_probe_trace.jsonl", "raw/flowstar/compose_probe_trace.jsonl.gz"),
    )
    for source, relative in compressed_copies:
        _gzip_copy(source, output / relative, output, manifest)

    core_patch = subprocess.run(
        ["git", "diff", "--binary", BASELINE_COMMIT, C4_COMMIT, "--", *CORE_PATHS],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    patch_path = output / "raw/c4/core.patch"
    patch_path.write_bytes(core_patch)
    manifest.append(
        {
            "path": patch_path.relative_to(output).as_posix(),
            "source": f"git diff --binary {BASELINE_COMMIT} {C4_COMMIT} -- {' '.join(CORE_PATHS)}",
            "size": patch_path.stat().st_size,
            "sha256": sha256(patch_path),
            "encoding": "identity",
        }
    )

    result = recompute(output)
    _write_json(output / "CLOSURE_RESULT.json", result)
    _write_json(
        output / "MANIFEST.json",
        {
            "schema": "torch_tm_flowpipe.brusselator_sr1000_c4_manifest/1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_commit": BASELINE_COMMIT,
            "c4_commit": C4_COMMIT,
            "raw_files": sorted(manifest, key=lambda row: row["path"]),
        },
    )
    (output / "README.md").write_text(
        "# Brusselator SR1000 and C4 closure\n\n"
        "This package preserves the frozen SR1000 baseline, output-equivalent Flow*\n"
        "operator traces, the mechanically selected step-one checkpoint, and the one\n"
        "same-input C4 gate. It derives the non-capacity verdict and the first material\n"
        "operator divergence from raw records. No C4 native-prefix rerun was performed.\n\n"
        "```bash\npython scripts/verify_brusselator_sr1000_c4_evidence.py\n```\n\n"
        f"Recomputed status: `{result['status']}`.\n",
        encoding="utf-8",
    )
    files = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output).as_posix()}\n" for path in files),
        encoding="ascii",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr1000", type=Path, default=DEFAULT_SR1000)
    parser.add_argument("--flowstar", type=Path, default=DEFAULT_FLOWSTAR)
    parser.add_argument("--c4", type=Path, default=DEFAULT_C4)
    parser.add_argument("--output", type=Path, default=DEFAULT_PACKAGE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build(
        args.sr1000.resolve(),
        args.flowstar.resolve(),
        args.c4.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "BRUSSELATOR_SR1000_OPERATOR_C4_CLOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
