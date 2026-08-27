#!/usr/bin/env python3
"""Build the compact VDP generic-refactor zero-regression evidence package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_vdp_generic_refactor_regression import recompute, sha256  # noqa: E402


DEFAULT_REFERENCE_ROOT = Path("/srv/local/shengenli/vdp_c3_runs_20260827")
DEFAULT_CANDIDATE_ROOT = Path("/srv/local/shengenli/vdp_c3_phase2_regression_20260827")
DEFAULT_OUTPUT = ROOT / "artifacts/runs/vdp_generic_refactor_vdp_zero_regression_20260827"
EMPTY_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()

CONTRACT: dict[str, Any] = {
    "schema": "torch_tm_flowpipe.vdp_generic_refactor_regression_contract/1",
    "fixed_horizons": {"T1": 1.0, "T3": 3.0, "T6p32": 6.32},
    "comparison": {
        "c3_numeric_tolerance": 1e-12,
        "segment_ignored_fields": ["dense_kernel_s", "stage_runtime_s"],
        "summary_ignored_fields": [
            "branch",
            "commit",
            "dense_kernel_s",
            "nonkernel_nontransfer_solver_s",
            "peak_rss_bytes",
            "runtime_s",
            "trace_io_s",
        ],
    },
    "empty_diff_sha256": EMPTY_DIFF_SHA256,
    "source_commits": {
        "reference": {
            "torch_c2": "29c9ee8f1fe96b860052b86a2b37d79a37bbb2ca",
            "torch_c3": "190e06714dbfe2afe53650b577916dfeca73dd5a",
        },
        "candidate": {
            "torch_c2": "29c9ee8f1fe96b860052b86a2b37d79a37bbb2ca",
            "torch_c3": "b88888691eaeefac1fb2e48d5ab0f82ad50c58ac",
        },
    },
    "native_expectations": {
        "torch_c2": {
            "status": "failed",
            "completed_horizon": 6.714914669607182,
            "completed_requested_horizon": False,
            "accepted_steps": 233,
            "rejected_attempts": 37,
        },
        "torch_c3": {
            "status": "completed",
            "completed_horizon": 10.0,
            "completed_requested_horizon": True,
            "accepted_steps": 246,
            "rejected_attempts": 35,
        },
    },
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_dir(root: Path, side: str, schedule: str, lane: str, label: str) -> Path:
    if side == "candidate":
        return root / lane / ("native_T10" if schedule == "native" else f"fixed_{label}")
    if schedule == "native":
        return root / "phase_f" / lane / "native_T10"
    phase = "phase_a" if lane == "torch_c2" else "phase_e"
    return root / phase / lane / f"fixed_{label}"


def _gzip_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            shutil.copyfileobj(src, zipped)


def build(reference_root: Path, candidate_root: Path, output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    schedules = [("fixed", label) for label in CONTRACT["fixed_horizons"]] + [("native", "T10")]
    for side, root in (("reference", reference_root), ("candidate", candidate_root)):
        for schedule, label in schedules:
            for lane in ("torch_c2", "torch_c3"):
                source = _source_dir(root, side, schedule, lane, label)
                destination = output / "raw" / side / schedule / lane / label
                destination.mkdir(parents=True, exist_ok=True)
                names = ["command.json", "summary.json", "segments.csv"]
                if schedule == "native":
                    names.append("attempts.csv")
                for name in names:
                    source_path = source / name
                    if not source_path.is_file():
                        raise FileNotFoundError(f"required regression evidence missing: {source_path}")
                    if name.endswith(".csv"):
                        destination_path = destination / f"{name}.gz"
                        _gzip_copy(source_path, destination_path)
                    else:
                        destination_path = destination / name
                        shutil.copyfile(source_path, destination_path)
                    manifest.append(
                        {
                            "path": destination_path.relative_to(output).as_posix(),
                            "source": str(source_path),
                            "sha256": sha256(destination_path),
                            "size": destination_path.stat().st_size,
                        }
                    )
    write_json(output / "EVIDENCE_CONTRACT.json", CONTRACT)
    write_json(
        output / "MANIFEST.json",
        {
            "schema": "torch_tm_flowpipe.vdp_generic_refactor_regression_manifest/1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "raw_files": sorted(manifest, key=lambda row: row["path"]),
        },
    )
    result = recompute(output)
    if not result["passed"]:
        raise RuntimeError(f"refusing failed regression package: {result['gates']}")
    write_json(output / "RESULT.json", result)
    (output / "README.md").write_text(
        "# VDP generic-refactor zero regression\n\n"
        "This package contains deterministic gzip copies of both the frozen reference and\n"
        "post-refactor candidate CSVs. The verifier recomputes all hashes, tolerance checks,\n"
        "source provenance, horizons, and accepted/rejected counts from package-local raw data.\n\n"
        "```bash\npython scripts/verify_vdp_generic_refactor_regression.py\n```\n\n"
        f"Maximum observed C3 numeric delta: `{result['maximum_c3_numeric_delta']}`.\n",
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
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build(args.reference_root.resolve(), args.candidate_root.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
