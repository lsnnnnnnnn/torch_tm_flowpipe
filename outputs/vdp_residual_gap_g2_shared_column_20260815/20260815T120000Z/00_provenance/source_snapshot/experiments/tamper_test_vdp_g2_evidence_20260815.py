#!/usr/bin/env python3
"""Exercise checksum, recomputed-semantics, and required-file rejection."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "experiments/verify_vdp_g2_evidence_20260815.py"


def load_verifier() -> Any:
    spec = importlib.util.spec_from_file_location("vdp_g2_verifier", VERIFY)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejected(module: Any, package: Path) -> str:
    try:
        module.verify(package)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    raise AssertionError("tampered package passed verification")


def update_integrity(package: Path, relative: str | None, *, remove: bool = False) -> None:
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if relative is not None:
        if remove:
            manifest["files"].pop(relative, None)
        else:
            path = package / relative
            manifest["files"][relative] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    manifest["file_count"] = len(manifest["files"])
    manifest["total_bytes_excluding_manifest_and_sums"] = sum(
        row["bytes"] for row in manifest["files"].values()
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (package / "SHA256SUMS").write_text(
        "".join(f"{row['sha256']}  {name}\n" for name, row in manifest["files"].items()),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verifier = load_verifier()
    cases = []
    with tempfile.TemporaryDirectory(prefix="vdp-g2-tamper-") as temporary:
        temporary_root = Path(temporary)

        checksum_case = temporary_root / "checksum"
        shutil.copytree(args.package, checksum_case)
        curve = checksum_case / "04_matrix/fixed_curve.csv.gz"
        raw = gzip.decompress(curve.read_bytes()).replace(b"0.01,", b"0.0100000000001,", 1)
        curve.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
        cases.append({
            "case": "raw_numeric_checksum_mutation",
            "rejected": True,
            "reason": rejected(verifier, checksum_case),
        })

        semantic_case = temporary_root / "semantic"
        shutil.copytree(args.package, semantic_case)
        summary_path = semantic_case / "04_matrix/scientific_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["conclusion"] = "G2_VDP_T10_VALIDATED"
        summary["gates"]["production_success"] = True
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        update_integrity(semantic_case, "04_matrix/scientific_summary.json")
        cases.append({
            "case": "refinalized_semantic_conclusion_mutation",
            "rejected": True,
            "reason": rejected(verifier, semantic_case),
        })

        required_case = temporary_root / "required"
        shutil.copytree(args.package, required_case)
        relative = "03_oracle/independent_oracle.json"
        (required_case / relative).unlink()
        update_integrity(required_case, relative, remove=True)
        cases.append({
            "case": "refinalized_required_oracle_deletion",
            "rejected": True,
            "reason": rejected(verifier, required_case),
        })
    result = {
        "schema": "vdp_g2_evidence_tamper_tests_v1",
        "passed": all(row["rejected"] for row in cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
