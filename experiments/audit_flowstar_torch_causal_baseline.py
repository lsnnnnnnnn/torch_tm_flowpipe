#!/usr/bin/env python3
"""Re-derive Gate-A conclusions from committed and independently fresh traces."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence, TextIO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.source_carry_audit import (
    accepted_flowstar_rows,
    accepted_torch_rows,
    checkpoint_reproduction,
    derive_width_minima,
)


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite(raw: str) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def compare_traces(
    committed_path: Path,
    fresh_path: Path,
    *,
    runtime_fields: set[str],
) -> dict[str, Any]:
    old_fields, old_rows = read_csv(committed_path)
    new_fields, new_rows = read_csv(fresh_path)
    if old_fields != new_fields or len(old_rows) != len(new_rows):
        raise ValueError("committed/fresh trace schema or row count mismatch")
    field_counts = {field: 0 for field in old_fields}
    scientific_text_mismatches = 0
    binary64_values = 0
    binary64_mismatches = 0
    decimal17_values = 0
    decimal17_mismatches = 0
    tolerance_values = 0
    tolerance_mismatches = 0
    tolerance = 1e-15
    for old, new in zip(old_rows, new_rows, strict=True):
        for field in old_fields:
            if old[field] != new[field]:
                field_counts[field] += 1
                if field not in runtime_fields:
                    scientific_text_mismatches += 1
            if field in runtime_fields:
                continue
            old_float = finite(old[field])
            new_float = finite(new[field])
            if old_float is None or new_float is None:
                continue
            binary64_values += 1
            binary64_mismatches += old_float.hex() != new_float.hex()
            decimal17_values += 1
            decimal17_mismatches += format(old_float, ".17g") != format(new_float, ".17g")
            tolerance_values += 1
            tolerance_mismatches += not math.isclose(
                old_float, new_float, rel_tol=tolerance, abs_tol=tolerance
            )
    return {
        "committed": {"path": str(committed_path), "sha256": sha256(committed_path)},
        "fresh": {"path": str(fresh_path), "sha256": sha256(fresh_path)},
        "rows": len(old_rows),
        "fields": len(old_fields),
        "runtime_fields_excluded_from_science": sorted(runtime_fields),
        "field_text_mismatch_counts": {
            field: count for field, count in field_counts.items() if count
        },
        "scientific_decimal_text": {
            "comparisons": len(old_rows) * (len(old_fields) - len(runtime_fields)),
            "mismatches": scientific_text_mismatches,
            "exact": scientific_text_mismatches == 0,
        },
        "binary64_hex_after_parse": {
            "comparisons": binary64_values,
            "mismatches": binary64_mismatches,
            "exact": binary64_mismatches == 0,
        },
        "decimal_17_digit_after_parse": {
            "comparisons": decimal17_values,
            "mismatches": decimal17_mismatches,
            "exact": decimal17_mismatches == 0,
        },
        "tolerance_equality": {
            "relative_and_absolute_tolerance": tolerance,
            "comparisons": tolerance_values,
            "mismatches": tolerance_mismatches,
            "all_equal": tolerance_mismatches == 0,
        },
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    _, committed_flow_all = read_csv(args.committed_flow.resolve())
    _, committed_torch_all = read_csv(args.committed_torch.resolve())
    _, fresh_flow_all = read_csv(args.fresh_flow.resolve())
    _, fresh_torch_all = read_csv(args.fresh_torch.resolve())
    committed_flow = accepted_flowstar_rows(committed_flow_all)
    committed_torch = accepted_torch_rows(committed_torch_all)
    fresh_flow = accepted_flowstar_rows(fresh_flow_all)
    fresh_torch = accepted_torch_rows(fresh_torch_all)
    fresh_torch_summary = read_json(args.fresh_torch_summary.resolve())
    if (
        len(committed_flow) != len(fresh_flow)
        or len(committed_torch) != len(fresh_torch)
    ):
        raise ValueError("accepted-prefix count mismatch")
    if len(fresh_flow) != 1000 or len(fresh_torch) != 632:
        raise ValueError("fresh baseline did not reproduce 1000/632")
    if (
        fresh_torch_summary.get("accepted_steps") != 632
        or fresh_torch_summary.get("failure_type") != "minimum_step_reached"
    ):
        raise ValueError("fresh Torch natural failure changed")

    checkpoints, checkpoint_verdict = checkpoint_reproduction(fresh_flow, fresh_torch)
    if checkpoint_verdict["status"] != "BASELINE_CONCLUSIONS_REPRODUCED":
        raise ValueError("frozen width ratios did not reproduce")
    write_csv(output / "checkpoint_ratios.csv", checkpoints)
    minima, contexts = derive_width_minima(fresh_flow)
    write_csv(output / "flowstar_width_minima.csv", minima)
    write_csv(output / "flowstar_width_minima_context.csv", contexts)
    expected_minima = {
        "endpoint_x": (397, 0.00861211181140531),
        "endpoint_y": (474, 0.026272600935460244),
        "segment_tube_x": (397, 0.008888711363604695),
        "segment_tube_y": (474, 0.030888053869117083),
    }
    for row in minima:
        step, width = expected_minima[str(row["channel"])]
        if int(row["step"]) != step or float(row["width"]) != width:
            raise ValueError(f"Flow* minimum changed: {row['channel']}")
        if float(row["width"]) <= 1e-9:
            raise ValueError("Flow* minimum is unexpectedly numerically near zero")

    rejected = [row for row in fresh_torch_all if row.get("status") == "rejected"]
    if len(rejected) != 1:
        raise ValueError("Torch candidate rejection count mismatch")
    candidate = rejected[0]
    margins = json.loads(candidate["target_margins"])
    candidate_result = {
        "candidate_step": int(candidate["segment_index"]) + 1,
        "pre_time": float(candidate["t_lo"]),
        "h": float(fresh_torch_summary["schedule"]["h_decimal"]),
        "status": candidate["status"],
        "y_subset_margin": float(margins[0][1]),
    }
    if candidate_result != {
        "candidate_step": 633,
        "pre_time": 6.32,
        "h": 0.01,
        "status": "rejected",
        "y_subset_margin": -8.441898798404161e-06,
    }:
        raise ValueError("Torch candidate-633 contract changed")

    # The copied Flow* trace records the scale consumed by each step.  Thus
    # row 2 carries the state produced at the accepted step-1 boundary.
    flow_step2 = fresh_flow[1]
    torch_step1 = fresh_torch[0]
    scale = {
        "flowstar": [
            float(flow_step2["extracted_scale_x"]),
            float(flow_step2["extracted_scale_y"]),
        ],
        "torch": list(json.loads(torch_step1["scale"])),
    }
    if scale != {
        "flowstar": [0.15044966009214522, 0.060913584414125518],
        "torch": [0.15045059849388548, 0.06092414958140347],
    }:
        raise ValueError("step-2 input scales did not reproduce")

    comparisons = {
        "flowstar": compare_traces(
            args.committed_flow.resolve(),
            args.fresh_flow.resolve(),
            runtime_fields={"stage_runtime_seconds"},
        ),
        "torch": compare_traces(
            args.committed_torch.resolve(),
            args.fresh_torch.resolve(),
            runtime_fields={"stage_runtime_s"},
        ),
    }
    write_json(output / "committed_vs_fresh_trace_comparison.json", comparisons)
    labels = {
        "width_status": "FLOWSTAR_WIDTH_MINIMUM_POSITIVE_NOT_NUMERICALLY_NEAR_ZERO",
        "historical_width_alias": "FLOWSTAR_WIDTH_IS_POSITIVE_NEAR_ZERO",
        "historical_width_alias_eligible_for_current_conclusion": False,
        "source_headline": "SOURCE_MECHANISM_CANDIDATES_LOCALIZED_CAUSAL_SPLIT_OPEN",
        "source_map_evidence_class": "HUMAN_AUTHORED_SOURCE_CANDIDATE_MAP",
        "source_map_is_causal_proof": False,
        "runtime_feature_hardcoded_claims_are_observations": False,
        "copied_probe_pre_gate_b_classification": "SOURCE_OBSERVATION_TRACE",
        "candidate_status": "NO_FIX_AUTHORIZED",
    }
    write_json(output / "evidence_label_corrections.json", labels)

    stale_attestation = {
        "canonical_package_root": str(args.old_package.resolve()),
        "stale_command_root": "20260813T025448Z",
        "correct_command_root": "20260813T030338Z",
        "stored_checksum_count": 55,
        "actual_checksum_count_before_correction": sum(
            1 for line in (args.old_package / "SHA256SUMS").read_text().splitlines() if line
        ),
        "stored_json_count": 27,
        "actual_json_count_before_correction": len(list(args.old_package.rglob("*.json"))),
        "scientific_tested_sha": "adb985e703b61a384703bfa724021472caa3f870",
        "prior_publication_tip": "cdda27bf2c0e7f72e135edbfd2b2ba10a8c5f96d",
        "source_test_tree_diff_between_tested_and_prior_tip": False,
        "correction_required": True,
    }
    write_json(output / "stale_attestation_audit.json", stale_attestation)

    result = {
        "schema": "flowstar_torch_causal_baseline_audit_v1",
        "flowstar_accepted_steps": len(fresh_flow),
        "torch_accepted_steps": len(fresh_torch),
        "torch_candidate_633": candidate_result,
        "step1_to_step2_scale": scale,
        "checkpoint_verdict": checkpoint_verdict,
        "minima": minima,
        "trace_comparisons": comparisons,
        "labels": labels,
        "stale_attestation": stale_attestation,
        "statuses": [
            "BASELINE_CONCLUSIONS_REPRODUCED",
            "FLOWSTAR_WIDTH_MINIMUM_POSITIVE_NOT_NUMERICALLY_NEAR_ZERO",
            "SOURCE_MECHANISM_CANDIDATES_LOCALIZED_CAUSAL_SPLIT_OPEN",
            "NO_FIX_AUTHORIZED",
        ],
    }
    write_json(output / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--committed-flow", type=Path, required=True)
    parser.add_argument("--committed-torch", type=Path, required=True)
    parser.add_argument("--fresh-flow", type=Path, required=True)
    parser.add_argument("--fresh-torch", type=Path, required=True)
    parser.add_argument("--fresh-torch-summary", type=Path, required=True)
    parser.add_argument("--old-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), sort_keys=True, allow_nan=False))
