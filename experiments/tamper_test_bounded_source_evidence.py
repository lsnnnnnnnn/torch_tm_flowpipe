#!/usr/bin/env python3
"""Show that checksum-only and checksum-refinalized semantic tampering fail."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "experiments/verify_bounded_source_evidence.py"


def load_verifier():
    spec = importlib.util.spec_from_file_location("bounded_source_package_verifier", VERIFY)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load evidence verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejection(module: Any, package: Path) -> str:
    try:
        module.verify(package)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    raise AssertionError("tampered package passed verification")


def rewrite_integrity(package: Path, relative: str) -> None:
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = package / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["files"][relative] = {"bytes": path.stat().st_size, "sha256": digest}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = []
    for line in (package / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        _, name = line.split("  ", 1)
        lines.append(f"{digest if name == relative else manifest['files'][name]['sha256']}  {name}\n")
    (package / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verifier = load_verifier()
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="vdp-g1-tamper-") as temporary:
        root = Path(temporary)
        checksum_case = root / "checksum_case"
        shutil.copytree(args.package, checksum_case)
        path = checksum_case / "04_causal_runs/scientific_summary.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["native_candidate_terminal_time"] = 10.0
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        cases.append({"case": "raw_scientific_mutation", "rejected": True, "reason": rejection(verifier, checksum_case)})

        semantic_case = root / "semantic_case"
        shutil.copytree(args.package, semantic_case)
        path = semantic_case / "04_causal_runs/scientific_summary.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["conclusion"] = "T1_T3_WIDTH_CAUSE_CLOSED__SOURCE_LEDGER_CARRY_ACCEPTED"
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rewrite_integrity(semantic_case, "04_causal_runs/scientific_summary.json")
        cases.append({"case": "refinalized_semantic_conclusion_mutation", "rejected": True, "reason": rejection(verifier, semantic_case)})

        file_case = root / "file_case"
        shutil.copytree(args.package, file_case)
        (file_case / "02_contract_oracles/independent_oracle.json").unlink()
        cases.append({"case": "required_oracle_file_deleted", "rejected": True, "reason": rejection(verifier, file_case)})
    result = {"schema": "vdp_t1_t3_bounded_source_tamper_tests_v1", "passed": True, "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
