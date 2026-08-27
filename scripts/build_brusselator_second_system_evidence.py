#!/usr/bin/env python3
"""Build the self-verifying pre-registered Brusselator evidence package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_brusselator_second_system import analyze, sha256  # noqa: E402


DEFAULT_RAW_ROOT = Path("/srv/local/shengenli/brusselator_second_system_20260827")
DEFAULT_OUTPUT = ROOT / "artifacts/runs/brusselator_generic_core_validation_20260827"
EMPTY_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()

RAW_FILES = {
    "flowstar": (
        "command.json",
        "summary.json",
        "native_summary.json",
        "segments.csv",
        "clone.stdout.log",
        "clone.stderr.log",
        "checkout.stdout.log",
        "checkout.stderr.log",
        "build_toolbox.stdout.log",
        "build_toolbox.stderr.log",
        "build_driver.stdout.log",
        "build_driver.stderr.log",
        "run.stdout.log",
        "run.stderr.log",
    ),
    "torch_generic_no_queue": ("command.json", "summary.json", "segments.csv"),
    "torch_generic_sr100": ("command.json", "summary.json", "segments.csv"),
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build(raw_root: Path, output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for lane, names in RAW_FILES.items():
        for name in names:
            source = raw_root / lane / name
            if not source.is_file():
                raise FileNotFoundError(f"required second-system evidence missing: {source}")
            destination = output / "raw" / lane / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            manifest.append(
                {
                    "path": destination.relative_to(output).as_posix(),
                    "source": str(source),
                    "sha256": sha256(destination),
                    "size": destination.stat().st_size,
                }
            )
    junit_source = raw_root / "exact_fraction_2d.xml"
    if not junit_source.is_file():
        raise FileNotFoundError(f"required exact-test JUnit missing: {junit_source}")
    shutil.copyfile(junit_source, output / "exact_fraction_2d.xml")
    shutil.copyfile(ROOT / "SECOND_SYSTEM_CONTRACT.md", output / "SECOND_SYSTEM_CONTRACT.md")

    result = analyze(output / "raw", output / "exact_fraction_2d.xml")
    write_json(output / "RESULT.json", result)
    canonical_result = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    evidence_contract = {
        "schema": "torch_tm_flowpipe.brusselator_second_system_evidence_contract/1",
        "preregistered_contract_sha256": sha256(output / "SECOND_SYSTEM_CONTRACT.md"),
        "exact_test_nodeid": (
            "tests/test_accepted_boundary_sr.py::"
            "test_generic_accepted_boundary_operator_contains_exact_fraction_image[2]"
        ),
        "lanes": list(RAW_FILES),
        "torch_run_commit": "33ea600d01143177d02784b204cafabb4343711d",
        "generic_core_commit": "b88888691eaeefac1fb2e48d5ab0f82ad50c58ac",
        "flowstar_commit": "b85a3211748cb77b736fe4ad42ee02d8d2b81148",
        "empty_tracked_diff_sha256": EMPTY_DIFF_SHA256,
        "result_sha256": hashlib.sha256(canonical_result.encode()).hexdigest(),
    }
    write_json(output / "EVIDENCE_CONTRACT.json", evidence_contract)
    write_json(
        output / "MANIFEST.json",
        {
            "schema": "torch_tm_flowpipe.brusselator_second_system_manifest/1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "raw_files": sorted(manifest, key=lambda row: row["path"]),
        },
    )
    (output / "README.md").write_text(
        "# Pre-registered Brusselator generic-core validation\n\n"
        "This package contains the only three numerical lanes allowed by\n"
        "`SECOND_SYSTEM_CONTRACT.md`, the exact 2D Fraction JUnit record, and the\n"
        "raw inputs needed to recompute every soundness, horizon, divergence,\n"
        "late-prefix, owner-accounting, and terminal-status field.\n\n"
        "```bash\npython scripts/verify_brusselator_second_system_evidence.py\n```\n\n"
        f"Observed terminal status: `{result['status']}`.\n",
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
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build(args.raw_root.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
