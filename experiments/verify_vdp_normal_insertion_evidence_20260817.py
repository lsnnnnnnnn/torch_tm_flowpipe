#!/usr/bin/env python3
"""Fail closed on integrity, contract, causal gates, and test evidence."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence
import xml.etree.ElementTree as ET


BASE_SHA = "e47ce68c61e73fc38f17fab3037d6cfe1877f3fd"
CHANNELS = ("endpoint_x", "endpoint_y", "segment_x", "segment_y")


class VerificationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(package: Path) -> dict[str, Any]:
    package = package.resolve()
    manifest = _load(package / "manifest.json")
    _require(manifest["schema"] == "vdp_normal_insertion_root_cause_evidence_v1", "schema")
    _require(manifest["base_sha"] == BASE_SHA, "base SHA")
    expected = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}
    }
    records = manifest["files"]
    _require(set(records) == expected, "manifest coverage")
    for relative, record in records.items():
        path = package / relative
        _require(path.stat().st_size == int(record["bytes"]), f"size: {relative}")
        _require(_sha(path) == record["sha256"], f"sha256: {relative}")

    checksums: dict[str, str] = {}
    for line in (package / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        _require(relative not in checksums and len(digest) == 64, "checksum syntax")
        checksums[relative] = digest
    checksum_expected = {
        path.relative_to(package).as_posix(): _sha(path)
        for path in package.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    _require(checksums == checksum_expected, "SHA256SUMS coverage or digest")

    for row in manifest["compressed_sources"]:
        stored = package / row["stored"]
        decompressed = gzip.decompress(stored.read_bytes())
        _require(len(decompressed) == int(row["source_bytes"]), f"decompressed bytes: {stored}")
        _require(
            hashlib.sha256(decompressed).hexdigest() == row["source_sha256"],
            f"decompressed digest: {stored}",
        )

    provenance = _load(package / "00_provenance/provenance.json")
    contract = provenance["contract"]
    _require(contract["ode"] == ["y", "y-x-x^2*y"], "ODE")
    _require(
        contract["initial_box_exact_decimal"] == [["1.1", "1.4"], ["2.35", "2.45"]],
        "initial box",
    )
    _require(
        contract["order"] == 4
        and contract["cutoff"] == "1e-10"
        and contract["fixed_h"] == "0.01"
        and contract["target_remainder_radius"] == "1e-4",
        "numerical contract",
    )

    gate = _load(package / "01_gate_a/summary.json")
    _require(gate["schema"] == "vdp_normal_insertion_gate_a_v2", "Gate A schema")
    _require(gate["checkpoint_count"] == 6, "Gate A checkpoint count")
    for key in (
        "all_cells_byte_identical_inputs",
        "D_and_H_exact_rational_bernstein_containment_all_checkpoints",
        "step_1_to_2_zero_inner_remainder_negative_control",
        "first_nonzero_repeated_insertion_at_step_2_to_3",
        "direct_repeated_nonlinear_remainder_consumption_after_first_nonzero",
        "H1_gate_a_mechanism_pass",
    ):
        _require(gate[key] is True, f"Gate A: {key}")
    for boundary in gate["boundaries"]:
        raw = _load(package / "01_gate_a" / boundary["relative_evidence"])
        _require(raw["same_prestate_sha256"] == boundary["same_prestate_sha256"], "boundary hash")
        _require(
            len({cell["same_input_sha256"] for cell in raw["cells"].values()}) == 1,
            "cell input identity",
        )
        for cell in ("D", "H"):
            _require(
                all(
                    component["oracle"][
                        "production_remainder_contains_exact_bernstein_enclosure"
                    ]
                    for component in raw["cells"][cell]["insertion_output"]
                ),
                f"exact oracle: {boundary['label']}/{cell}",
            )

    matrix = _load(package / "02_scientific_matrix/matrix.json")
    gates = matrix["gates"]
    _require(gates["H1_factorization_explains_at_least_one_T3_or_T6p32_channel_10pct"], "H1 threshold")
    _require(not gates["T1_T3_all_four_channels_remove_10pct_legacy_excess"], "T1/T3 failure accounting")
    _require(gates["T6p32_no_channel_regression"], "T6.32")
    _require(gates["native_at_least_6p397083942944808"], "native floor")
    _require(gates["runtime_at_most_2x_legacy"], "runtime")
    _require(not gates["reaches_T10"], "T10 failure accounting")
    for mode in ("legacy", "candidate"):
        consistency = matrix["cpu_cuda_consistency_T0p1"][mode]
        _require(consistency["consistent_at_1e_12"], f"CPU/CUDA: {mode}")
        _require(set(consistency["width_abs_deltas"]) == set(CHANNELS), "CUDA channels")

    test_counts: dict[str, int] = {}
    for path in sorted((package / "03_tests").glob("*.xml")):
        root = ET.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
        errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
        tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
        _require(failures == 0 and errors == 0 and tests > 0, f"tests: {path.name}")
        test_counts[path.name] = tests

    result = {
        "status": "verified",
        "files": len(expected),
        "bytes": sum((package / relative).stat().st_size for relative in expected),
        "gate_a_checkpoints": gate["checkpoint_count"],
        "decision": matrix["decision"],
        "tests": test_counts,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    verify(parse_args(argv).package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
