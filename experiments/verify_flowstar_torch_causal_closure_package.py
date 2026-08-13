#!/usr/bin/env python3
"""Fail-closed verifier that re-derives the causal-closure package conclusions."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, TextIO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.audit_flowstar_stock_probe_equivalence import (
    PUBLISHED_MAP,
    observer_full_state,
    observer_retained,
)
from torch_tm_flowpipe.lossless_state_queue_schema import (
    decode_binary64_exact,
    iter_canonical_dyadics,
    parse_file,
)
from torch_tm_flowpipe.source_carry_audit import (
    accepted_flowstar_rows,
    accepted_torch_rows,
    checkpoint_reproduction,
    derive_width_minima,
)


REQUIRED_DIRECTORIES = (
    "00_identity_provenance",
    "01_baseline_reproduction",
    "02_evidence_label_corrections",
    "03_stock_clean_outputs",
    "04_stock_instrumented_outputs",
    "05_copied_probe_equivalence",
    "06_flowstar_queue_factorial",
    "07_torch_horner_queue_factorial",
    "08_step1_step2_attribution",
    "09_lossless_schema_roundtrip",
    "10_same_prestate_2x2",
    "11_source_ledger_micro_oracles",
    "12_candidate_l1_l2_l3",
    "13_tests",
    "14_final_clone",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Mapping[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON token {value}: {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_csv(path: Path) -> list[dict[str, str]]:
    with open_text(path) as handle:
        return list(csv.DictReader(handle))


def verify_checksums(root: Path) -> int:
    rows = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        expected, separator, relative = row.partition("  ")
        if not separator or len(expected) != 64 or relative in seen:
            raise ValueError(f"invalid checksum line {line_number}")
        seen.add(relative)
        target = root / relative
        if not target.is_file():
            raise ValueError(f"missing checksummed file: {relative}")
        if sha256(target) != expected:
            raise ValueError(f"checksum mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if actual != seen:
        raise ValueError(
            f"checksum inventory mismatch: missing={sorted(actual-seen)}, extra={sorted(seen-actual)}"
        )
    return len(rows)


def verify_artifact_indices(root: Path) -> int:
    count = 0
    for index_path in sorted(root.rglob("artifact_index.json")):
        index = strict_json(index_path)
        if index.get("schema") != "torch_tm_flowpipe_evidence_artifact_index_v1":
            raise ValueError(f"artifact index schema mismatch: {index_path}")
        runner = index_path.parent
        published = index.get("files")
        if not isinstance(published, list):
            raise ValueError(f"artifact index rows missing: {index_path}")
        actual = [
            {
                "path": path.relative_to(runner).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(runner.rglob("*"))
            if path.is_file() and path.name != "artifact_index.json"
        ]
        if published != actual:
            raise ValueError(f"artifact index content mismatch: {index_path}")
        count += 1
    return count


def derive_scientific_statuses(*, baseline: bool, stock: bool, factor: str,
                               bridge: bool, operator_closed: bool,
                               oracle_executed: bool, candidate_implemented: bool) -> list[str]:
    return [
        "BASELINE_CONCLUSIONS_REPRODUCED" if baseline else "BASELINE_NOT_REPRODUCIBLE_STOP",
        "FLOWSTAR_WIDTH_MINIMUM_POSITIVE_NOT_NUMERICALLY_NEAR_ZERO",
        "STOCK_COPIED_PROBE_EQUIVALENCE_CLOSED" if stock else "SOURCE_TRACE_NOT_STOCK_EQUIVALENT",
        factor,
        "SAME_PRESTATE_LOSSLESS_BRIDGE_AVAILABLE" if bridge else "SAME_PRESTATE_LOSSLESS_BRIDGE_NOT_AVAILABLE",
        "CAUSAL_SOURCE_DELTA_CLOSED" if operator_closed else "SOURCE_MECHANISM_CANDIDATES_LOCALIZED_CAUSAL_SPLIT_OPEN",
        "SOURCE_LEDGER_ORACLE_CLOSED" if oracle_executed else "SOURCE_LEDGER_ORACLE_INCOMPLETE",
        "SOUND_CARRY_CANDIDATE_L1" if candidate_implemented else "NO_FIX_AUTHORIZED",
    ]


def verify_claims(derived: list[str], verification: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if verification.get("scientific_statuses") != derived:
        raise ValueError("verification status does not match raw-derived statuses")
    if manifest.get("scientific_statuses") != derived:
        raise ValueError("manifest status does not match raw-derived statuses")
    if verification.get("scientific_outcome_uses_process_exit_code") is not False:
        raise ValueError("scientific outcome improperly depends on process exit code")


def verify_exact_claim(label: str, published: Any, derived: Any) -> None:
    if published != derived:
        raise ValueError(f"{label} is not raw-derived")


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not all((root / directory).is_dir() for directory in REQUIRED_DIRECTORIES):
        raise ValueError("required evidence directory missing")
    checksum_count = verify_checksums(root)
    json_paths = sorted(root.rglob("*.json"))
    loaded = {path.relative_to(root).as_posix(): strict_json(path) for path in json_paths}
    artifact_index_count = verify_artifact_indices(root)
    manifest = loaded["manifest.json"]
    verification = loaded["verification.json"]
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) + 1 != checksum_count:
        raise ValueError("manifest/checksum inventory count mismatch")
    expected_manifest = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}
    ]
    published_manifest = [
        {key: row[key] for key in ("path", "bytes", "sha256")}
        for row in outputs
    ]
    if published_manifest != expected_manifest:
        raise ValueError("manifest output inventory mismatch")

    fresh_flow_all = read_csv(
        root / "01_baseline_reproduction/copied_probe_fresh/artifacts/flowstar_trace.csv.gz"
    )
    fresh_torch_all = read_csv(
        root / "01_baseline_reproduction/torch_legacy_fresh/artifacts/run/segments.csv.gz"
    )
    fresh_flow = accepted_flowstar_rows(fresh_flow_all)
    fresh_torch = accepted_torch_rows(fresh_torch_all)
    checkpoints, baseline_verdict = checkpoint_reproduction(fresh_flow, fresh_torch)
    minima, _ = derive_width_minima(fresh_flow)
    baseline = (
        len(fresh_flow) == 1000
        and len(fresh_torch) == 632
        and baseline_verdict["status"] == "BASELINE_CONCLUSIONS_REPRODUCED"
        and all(float(row["width"]) >= 1e-9 for row in minima)
    )
    baseline_summary = loaded[
        "02_evidence_label_corrections/baseline_audit/artifacts/audit/summary.json"
    ]
    verify_exact_claim(
        "Flow* accepted-step claim", baseline_summary.get("flowstar_accepted_steps"), len(fresh_flow)
    )
    verify_exact_claim(
        "Torch accepted-step claim", baseline_summary.get("torch_accepted_steps"), len(fresh_torch)
    )
    verify_exact_claim(
        "baseline checkpoint verdict", baseline_summary.get("checkpoint_verdict"), baseline_verdict
    )
    verify_exact_claim("Flow* minima", baseline_summary.get("minima"), minima)
    rejected = [row for row in fresh_torch_all if row.get("status") == "rejected"]
    if len(rejected) != 1 or json.loads(rejected[0]["target_margins"])[0][1] != -8.441898798404161e-06:
        raise ValueError("candidate 633 margin mismatch")

    clean_path = root / "03_stock_clean_outputs/oneshot_q100/artifacts/stock.csv"
    instrumented_path = root / "04_stock_instrumented_outputs/oneshot_q100/artifacts/stock.csv"
    clean = read_csv(clean_path)
    copied = read_csv(
        root / "05_copied_probe_equivalence/copied_probe_exact/artifacts/flowstar_trace.csv.gz"
    )
    observer = read_csv(
        root / "04_stock_instrumented_outputs/oneshot_q100/artifacts/observer.csv.gz"
    )
    stock = clean_path.read_bytes() == instrumented_path.read_bytes() and len(clean) == 1000
    for index, (stock_row, copied_row) in enumerate(zip(clean, copied, strict=True)):
        stock &= all(stock_row[left] == copied_row[right] for left, right in PUBLISHED_MAP.items())
        stock &= observer_retained(observer[2 * index]) == copied_row["retained_coefficients_binary_canonical"]
        if index + 1 < len(copied):
            stock &= observer_full_state(observer[2 * index + 1]) == copied[index + 1]["prestate_state_binary_canonical"]
    equivalence = loaded[
        "05_copied_probe_equivalence/three_way_audit/artifacts/audit/summary.json"
    ]
    if equivalence.get("status") != (
        "STOCK_COPIED_PROBE_EQUIVALENCE_CLOSED" if stock else "SOURCE_TRACE_NOT_STOCK_EQUIVALENT"
    ):
        raise ValueError("three-way equivalence status not raw-derived")

    q_expected = {1: 620, 2: 640, 10: 685}
    for q, expected in q_expected.items():
        q_summary = loaded[
            f"06_flowstar_queue_factorial/q{q}_qualified/artifacts/summary.json"
        ]
        if q_summary.get("accepted_steps") != expected or q_summary.get("result_status_code") != 4:
            raise ValueError(f"Flow* queue ablation mismatch: Q{q}")
    cells = {
        "T-D0": (632, "normalized_insertion"),
        "T-H0": (636, "normalized_insertion_horner"),
        "T-DQ": (632, "normalized_insertion_symqueue_v2"),
        "T-HQ": (636, "normalized_insertion_horner_symqueue_v2"),
    }
    cell_rows: dict[str, list[dict[str, str]]] = {}
    for cell, (steps, mode) in cells.items():
        summary = loaded[f"07_torch_horner_queue_factorial/{cell}/artifacts/run/summary.json"]
        if summary.get("accepted_steps") != steps or summary.get("reset_mode") != mode:
            raise ValueError(f"Torch factorial cell mismatch: {cell}")
        cell_rows[cell] = read_csv(
            root / f"07_torch_horner_queue_factorial/{cell}/artifacts/run/segments.csv.gz"
        )
    for index in range(632):
        if cell_rows["T-D0"][index]["endpoint_x_width"] != cell_rows["T-DQ"][index]["endpoint_x_width"]:
            raise ValueError("diagnostic queue unexpectedly changes endpoint x")
        if cell_rows["T-D0"][index]["endpoint_y_width"] != cell_rows["T-DQ"][index]["endpoint_y_width"]:
            raise ValueError("diagnostic queue unexpectedly changes endpoint y")
    factor_summary = loaded[
        "08_step1_step2_attribution/causal_analysis/artifacts/analysis/summary.json"
    ]
    factor = str(factor_summary.get("status"))
    if factor != "CAUSAL_FACTOR_SPLIT_PARTIAL":
        raise ValueError("factor split status mismatch")

    fixture_root = root / "09_lossless_schema_roundtrip/flowstar_fixtures_retry/artifacts/fixtures"
    state_files = sorted(path for path in fixture_root.glob("*.state") if not path.name.endswith(".roundtrip"))
    exact_roundtrips = 0
    exact_dyadics = 0
    for state in state_files:
        paired = state.with_name(state.name + ".roundtrip")
        if state.read_bytes() != paired.read_bytes():
            raise ValueError(f"lossless fixture mismatch: {state.name}")
        exact_roundtrips += 1
        for _, encoded in iter_canonical_dyadics(parse_file(state)):
            decode_binary64_exact(encoded)
            exact_dyadics += 1
    bridge_summary = loaded[
        "09_lossless_schema_roundtrip/flowstar_fixtures_retry/artifacts/fixtures/summary.json"
    ]
    cross_summary = loaded[
        "09_lossless_schema_roundtrip/cross_language_retry2/artifacts/audit/summary.json"
    ]
    bridge = (
        exact_roundtrips == 24
        and bridge_summary.get("canonical_byte_roundtrips_exact") == 24
        and bridge_summary.get("next_step_roundtrips_exact") == 24
        and cross_summary.get("torch_flowstar_roundtrip_byte_exact") is True
        and cross_summary.get("negative_tests_all_rejected") is True
    )

    matrix = loaded["10_same_prestate_2x2/operator_matrix/artifacts/matrix/summary.json"]
    for before, after in ((1, 2), (99, 100), (100, 101)):
        actual = root / f"10_same_prestate_2x2/operator_matrix/artifacts/matrix/flowstar_step_{before}_to_{after}.state"
        expected = fixture_root / f"step_{after}_pre_reset.state"
        if actual.read_bytes() != expected.read_bytes():
            raise ValueError(f"same-producer continuation mismatch: {before}->{after}")
    operator_closed = bool(matrix.get("operator_attribution_closed"))
    if (
        matrix.get("full_two_by_two_same_prestate_executed") is not False
        or matrix.get("queue_dropped") is not False
        or operator_closed
    ):
        raise ValueError("same-prestate operator mismatch was not fail-closed")

    oracle = loaded["11_source_ledger_micro_oracles/not_run_reason.json"]
    candidate = loaded["12_candidate_l1_l2_l3/not_run_reason.json"]
    oracle_executed = bool(oracle.get("executed"))
    candidate_implemented = bool(candidate.get("candidate_implemented"))
    if oracle_executed or candidate_implemented or candidate.get("legacy_default_changed") is not False:
        raise ValueError("unauthorized oracle/candidate execution")

    derived = derive_scientific_statuses(
        baseline=baseline,
        stock=stock,
        factor=factor,
        bridge=bridge,
        operator_closed=operator_closed,
        oracle_executed=oracle_executed,
        candidate_implemented=candidate_implemented,
    )
    verify_claims(derived, verification, manifest)
    publication = loaded["14_final_clone/status.json"]
    publication_status = publication.get("status")
    if publication_status not in {
        "PENDING_SCIENTIFIC_COMMIT_AND_FRESH_CLONE",
        "SCIENTIFIC_SHA_FRESH_CLONE_VERIFIED",
        "FINAL_PUBLICATION_TIP_FRESH_CLONE_VERIFIED",
    }:
        raise ValueError("unknown publication attestation status")
    expected_verification_status = (
        "PASS_SCIENTIFIC_PREPUBLICATION"
        if publication_status == "PENDING_SCIENTIFIC_COMMIT_AND_FRESH_CLONE"
        else "PASS"
    )
    if verification.get("status") != expected_verification_status:
        raise ValueError("package/publication verification phase mismatch")

    return {
        "schema": "flowstar_torch_causal_closure_rederived_verification_v1",
        "status": expected_verification_status,
        "checksum_files": checksum_count,
        "json_files_loaded": len(json_paths),
        "artifact_indices_verified": artifact_index_count,
        "flowstar_accepted_steps": len(fresh_flow),
        "torch_accepted_steps": len(fresh_torch),
        "checkpoint_rows_rederived": len(checkpoints),
        "flowstar_minima": [
            {"channel": row["channel"], "step": row["step"], "width": row["width"]}
            for row in minima
        ],
        "stock_equivalence": stock,
        "flowstar_lossless_fixture_roundtrips": exact_roundtrips,
        "flowstar_fixture_dyadics_checked": exact_dyadics,
        "scientific_statuses": derived,
        "publication_status": publication_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.package)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
