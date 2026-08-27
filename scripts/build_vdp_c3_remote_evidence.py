#!/usr/bin/env python3
"""Build the compact, self-contained VDP C3 remote evidence package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_vdp_c3_remote_evidence import CHANNELS, recompute, sha256  # noqa: E402


DEFAULT_RAW_ROOT = Path("/srv/local/shengenli/vdp_c3_runs_20260827")
DEFAULT_OUTPUT = ROOT / "artifacts/runs/vdp_c3_cross_step_causal_closure_20260827"

CONTRACT: dict[str, Any] = {
    "schema": "torch_tm_flowpipe.vdp_c3_remote_evidence_contract/1",
    "fixed_horizons": {"T1": 1.0, "T3": 3.0, "T6p32": 6.32},
    "gates": {
        "width_tolerance": 1e-12,
        "horizon_tolerance": 1e-12,
        "no_regression_horizons": ["T1", "T3"],
        "recovery_horizon": "T6p32",
        "t6_recovery_min": 0.25,
        "runtime_ratio_max": 2.0,
    },
    "native_expectations": {
        "flowstar": {"completed_horizon": 10.0, "accepted": 290, "status": "completed"},
        "c2": {
            "completed_horizon": 6.714914669607182,
            "completed_requested_horizon": False,
            "accepted": 233,
            "rejected": 37,
            "status": "failed",
        },
        "c3": {
            "completed_horizon": 10.0,
            "completed_requested_horizon": True,
            "accepted": 246,
            "rejected": 35,
            "status": "completed",
        },
    },
    "source_shas": {
        "c3_scientific": "190e06714dbfe2afe53650b577916dfeca73dd5a",
        "c3_assembly": "73e5484764df96847daf3dbbd90b637dbe2d6e06",
        "c2_scientific": "29c9ee8f1fe96b860052b86a2b37d79a37bbb2ca",
        "c2_package": "0fea2657b30aea5f8cfe326dbcd06d659b8dd26c",
        "huan_repaired": "743f6205e6408072193ad76e940e7f15030e8d3c",
        "flowstar": "b85a3211748cb77b736fe4ad42ee02d8d2b81148",
    },
    "statuses": {
        "failed_stop": "C3_REMOTE_EVIDENCE_CLOSURE_FAILED_STOP",
        "passed": "CROSS_STEP_CAUSE_IDENTIFIED__C3_PRODUCTION_GATE_PASSED__NATIVE_T10_REACHED",
    },
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def raw_file_map(raw_root: Path, junit: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for label in CONTRACT["fixed_horizons"]:
        flowstar = raw_root / f"phase_a/flowstar/fixed_{label}"
        files.extend(
            (flowstar / name, f"raw/fixed/flowstar/{label}/{name}")
            for name in ("stock.csv", "summary.json")
        )
        for lane, phase in (("c2", "phase_a"), ("c3", "phase_e")):
            source = raw_root / phase / f"torch_{lane}" / f"fixed_{label}"
            files.extend(
                (source / name, f"raw/fixed/torch_{lane}/{label}/{name}")
                for name in ("command.json", "summary.json")
            )
    files.append(
        (raw_root / "phase_a/huan/run_index.json", "raw/source/huan_run_index.json")
    )
    files.append(
        (raw_root / "phase_f/flowstar/native_T10/summary.json", "raw/native/flowstar/summary.json")
    )
    for lane in ("c2", "c3"):
        source = raw_root / "phase_f" / f"torch_{lane}" / "native_T10"
        files.extend(
            (source / name, f"raw/native/torch_{lane}/{name}")
            for name in ("command.json", "summary.json", "attempts.csv", "segments.csv")
        )
    files.append((junit, "raw/tests/pytest.xml"))
    return files


def report(result: dict[str, Any]) -> str:
    lines = [
        "# VDP C3 remote evidence closure",
        "",
        f"Highest recomputed status: `{result['highest_status']}`.",
        "",
        "All values below are recomputed by `scripts/verify_vdp_c3_remote_evidence.py` from",
        "the package-local raw CSV/JSON/JUnit XML. No external raw root is required.",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- `{name}`: `{passed}`" for name, passed in result["gates"].items())
    lines.extend(
        [
            "",
            "## T6.32 recovery",
            "",
            *(
                f"- {channel}: {result['t6_recovery_by_channel'][channel]:.15g}"
                for channel in CHANNELS
            ),
            "",
            "## Native outcome",
            "",
            f"- C2: T={result['native']['c2']['completed_horizon']:.15g}, "
            f"{result['native']['c2']['accepted']} accepted / "
            f"{result['native']['c2']['rejected']} rejected",
            f"- C3 SR100: T={result['native']['c3']['completed_horizon']:.15g}, "
            f"{result['native']['c3']['accepted']} accepted / "
            f"{result['native']['c3']['rejected']} rejected",
            "",
            "## Independent verification",
            "",
            "```bash",
            "python scripts/verify_vdp_c3_remote_evidence.py",
            "python scripts/verify_vdp_c3_remote_evidence.py --run-tests",
            "```",
            "",
            "`--run-tests` additionally reruns pytest and compares testcase identities and counts",
            "with the committed JUnit XML. The ordinary verifier never trusts `RESULT.json`; it",
            "recomputes that file and the highest status from raw evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def build(raw_root: Path, junit: Path, output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    if not junit.is_file():
        raise FileNotFoundError(f"JUnit XML missing: {junit}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for source, relative in raw_file_map(raw_root, junit):
        if not source.is_file():
            raise FileNotFoundError(f"required raw evidence missing: {source}")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest_rows.append(
            {
                "path": relative,
                "source_relative": (
                    source.relative_to(raw_root).as_posix()
                    if source.is_relative_to(raw_root)
                    else "pytest.xml"
                ),
                "sha256": sha256(destination),
                "size": destination.stat().st_size,
            }
        )
    write_json(output / "EVIDENCE_CONTRACT.json", CONTRACT)
    write_json(
        output / "MANIFEST.json",
        {
            "schema": "torch_tm_flowpipe.vdp_c3_remote_evidence_manifest/1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "raw_root_name": raw_root.name,
            "raw_files": sorted(manifest_rows, key=lambda row: row["path"]),
        },
    )
    result = recompute(output)
    if not all(result["gates"].values()):
        raise RuntimeError(
            f"refusing failed evidence package: {json.dumps(result['gates'], sort_keys=True)}"
        )
    write_json(output / "RESULT.json", result)
    (output / "README.md").write_text(report(result), encoding="utf-8")
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
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build(args.raw_root.resolve(), args.junit.resolve(), args.output.resolve())
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "highest_status": result["highest_status"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
