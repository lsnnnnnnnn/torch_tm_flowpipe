#!/usr/bin/env python3
"""Normalize the read-only Flow* step-1 trace into the common stage ledger."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import struct
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence


FLOWSTAR_SHA = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
SCHEMA = "common_step_operator_stage_ledger_v1"
CLASSIFICATIONS = {
    "BITWISE_EQUAL",
    "EXACT_VALUE_EQUAL_DIFFERENT_ENCODING",
    "ROUNDING_ONLY_BOTH_SOUND",
    "ENCLOSURE_DIFFERENT_BOTH_SOUND",
    "FIRST_SEMANTIC_DELTA",
    "UNDER_ENCLOSURE_WITNESS",
    "UNRESOLVED",
}
BASIS = {
    "id": "canonical_tau_ux_uy_o4",
    "canonical_variables": ["tau", "ux", "uy"],
    "flowstar_actual_variables": ["tau", "ux", "uy", "ut"],
    "actual_to_canonical": [0, 1, 2, None],
    "eliminated_variable_proof": "ut has zero exponent in physical x/y rows; t is a separate deterministic clock state",
    "domain": [["0", "1/100"], ["-1", "1"], ["-1", "1"]],
    "support": "complete total-degree <= 4",
}


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_path(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _number_from_pair(value: Mapping[str, str]) -> dict[str, Any]:
    if set(value) != {"decimal", "hex"}:
        raise ValueError(f"noncanonical Flow* number record: {value}")
    decimal = float(value["decimal"])
    hexadecimal = float.fromhex(value["hex"])
    if not math.isfinite(decimal) or decimal != hexadecimal:
        raise ValueError(f"Flow* decimal/hex mismatch: {value}")
    numerator, denominator = decimal.as_integer_ratio()
    return {
        "decimal": format(decimal, ".17g"),
        "hex": decimal.hex(),
        "binary64_bits": f"0x{struct.unpack('>Q', struct.pack('>d', decimal))[0]:016x}",
        "exact_rational": {"numerator": str(numerator), "denominator": str(denominator)},
    }


def _number(value: float) -> dict[str, Any]:
    return _number_from_pair({"decimal": format(float(value), ".17g"), "hex": float(value).hex()})


def _enrich_numbers(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"decimal", "hex"}:
            return _number_from_pair(value)
        return {str(key): _enrich_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_enrich_numbers(item) for item in value]
    return value


def _canonical_exponent(component: int, exponent: Sequence[int]) -> list[int] | None:
    if len(exponent) != 4:
        raise ValueError(f"Flow* exponent dimension is not four: {exponent}")
    if component in (0, 1):
        if int(exponent[3]) != 0:
            raise ValueError(f"physical state depends on deterministic ut variable: c={component}, e={exponent}")
        return [int(exponent[0]), int(exponent[1]), int(exponent[2])]
    return None


def _ownership(stage: str) -> dict[str, str]:
    if "degree_truncation" in stage:
        return {"ordinary": "unchanged", "truncation": "emitted interval owns discarded degree", "cutoff": "none", "source_ledger": "none at step 1"}
    if "cutoff" in stage:
        return {"ordinary": "unchanged", "truncation": "prior ledger", "cutoff": "emitted interval owns discarded small terms", "source_ledger": "none at step 1"}
    if "refinement" in stage or "candidate" in stage or "remainder" in stage:
        return {"ordinary": "Flow* self-map/refinement interval", "truncation": "included in ordinary image", "cutoff": "included in ordinary image", "source_ledger": "symbolic queue separately emitted"}
    if "symbolic_queue" in stage or "phi_" in stage or "propagated_symbolic" in stage:
        return {"ordinary": "separate", "truncation": "separate", "cutoff": "separate", "source_ledger": "J/Phi_L/scalars payload"}
    return {"ordinary": "Taylor-model remainder field", "truncation": "operation-local if emitted", "cutoff": "operation-local if emitted", "source_ledger": "separately emitted"}


def _source_map(flowstar_root: Path) -> dict[str, dict[str, Any]]:
    candidates = [
        flowstar_root / "flowstar-toolbox" / "Continuous.cpp",
        flowstar_root / "flowstar-toolbox" / "Continuous.h",
        flowstar_root / "flowstar-toolbox" / "TaylorModel.h",
    ]
    result: dict[str, dict[str, Any]] = {}
    for path in candidates:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if "flowstar_causal::emit" not in line:
                continue
            for quote in ('"',):
                parts = line.split(quote)
                if len(parts) >= 3:
                    stage = parts[1]
                    result.setdefault(
                        stage,
                        {
                            "file": str(path.relative_to(flowstar_root)),
                            "function": (
                                "TaylorModel::{ctrunc_normal,mul_insert_ctrunc_normal,cutoff_normal}"
                                if path.name == "TaylorModel.h"
                                else "Flowpipe::advance actual fixed symbolic-remainder path"
                            ),
                            "line_start": line_number,
                            "line_end": line_number,
                        },
                    )
    return result


def _source_for(stage: str, mapping: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    base = stage.split(":", 1)[0]
    if base in mapping:
        return dict(mapping[base])
    if stage in mapping:
        return dict(mapping[stage])
    return {
        "file": "flowstar-toolbox/CausalTrace.h",
        "function": "flowstar_causal read-only serializer; caller unavailable",
        "line_start": 1,
        "line_end": 1,
    }


def _range_interval(row: Mapping[str, str], prefix: str) -> dict[str, Any]:
    lo = float(row[prefix + "_lo"])
    hi = float(row[prefix + "_hi"])
    if float.fromhex(row[prefix + "_lo_hex"]) != lo or float.fromhex(row[prefix + "_hi_hex"]) != hi:
        raise ValueError(f"CSV decimal/hex mismatch for {prefix}")
    return {
        "lower": _number(lo),
        "upper": _number(hi),
        "precision_bits": 53,
        "directed_rounding_provenance": "Flow* Interval lower RNDD / upper RNDU converted outward to binary64",
    }


def _raw_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"empty Flow* trace record at line {line_number}")
            row = json.loads(line)
            if row.get("schema") != "flowstar_causal_observer_v1":
                raise ValueError(f"unknown Flow* trace schema at line {line_number}")
            if row.get("source_commit") != FLOWSTAR_SHA:
                raise ValueError(f"Flow* source SHA mismatch at line {line_number}")
            if int(row.get("accepted_step_index", -1)) == 0:
                rows.append(row)
    if not rows:
        raise ValueError("Flow* trace has no step-1 rows")
    return rows


def process(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    flowstar_root = args.flowstar_root.resolve()
    if subprocess_sha(flowstar_root) != FLOWSTAR_SHA:
        raise ValueError("Flow* worktree is not at the pinned SHA")
    source_mapping = _source_map(flowstar_root)
    raw = _raw_rows(args.raw_trace)
    patch_sha = _sha_path(args.observer_patch)
    source_patch = args.observer_patch.read_bytes()

    rows: list[dict[str, Any]] = []
    previous: list[str] = []
    stage_counts: dict[str, int] = {}
    for index, raw_row in enumerate(raw):
        payload = _enrich_numbers(raw_row)
        component = int(raw_row.get("component", -1))
        if raw_row.get("record_type") == "polynomial_term":
            payload["canonical_exponents"] = _canonical_exponent(component, raw_row["exponents"])
        stage = str(raw_row["stage"])
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        output_hash = _sha_bytes(_json_bytes(payload))
        row = {
            "schema": SCHEMA,
            "ledger_row_index": len(rows),
            "tool": "flowstar_pinned_actual",
            "actual_source": _source_for(stage, source_mapping),
            "flowstar_sha": FLOWSTAR_SHA,
            "observer_patch_sha256": patch_sha,
            "stage_id": stage,
            "iteration": int(raw_row.get("picard_iteration", -1)),
            "refinement_iteration": int(raw_row.get("refinement_iteration", -1)),
            "component": component,
            "record_type": raw_row["record_type"],
            "basis_id": BASIS["id"],
            "basis": BASIS,
            "support": payload.get("canonical_exponents"),
            "binary_exact_coefficient_encoding": "IEEE-754 binary64 bits + hexfloat, with RNDD/RNDU endpoint extraction",
            "exact_rational_value": "decoded per finite binary64 endpoint",
            "interval_precision_bits": 53,
            "directed_rounding_provenance": "Flow* Real.getValue_RNDD/RNDU or Interval.inf/sup",
            "remainder_ownership": _ownership(stage),
            "input_artifact_hashes": list(previous),
            "raw_record_sha256": _sha_bytes(_json_bytes(raw_row)),
            "output_artifact_hash": output_hash,
            "classification": "UNRESOLVED",
            "payload": payload,
        }
        rows.append(row)
        previous = [output_hash]

    csv_rows = list(csv.DictReader(args.result_csv.open(encoding="utf-8")))
    step1 = [row for row in csv_rows if int(row["step"]) == 1]
    if len(step1) != 1:
        raise ValueError(f"expected one Flow* step-1 CSV row, found {len(step1)}")
    result = step1[0]
    driver_lines = args.driver_source.read_text(encoding="utf-8").splitlines()
    driver_line = next(i for i, line in enumerate(driver_lines, 1) if "void write_flowpipe_row" in line)
    driver_source = {
        "file": str(args.driver_source.resolve()),
        "function": "write_flowpipe_row -> Flowpipe::intEvalNormal",
        "line_start": driver_line,
        "line_end": driver_line + 100,
    }
    for stage, prefixes in (
        ("segment_polynomial_and_final_range", ("segment_polynomial_x", "segment_polynomial_y", "segment_x", "segment_y")),
        ("endpoint_polynomial_and_final_range", ("endpoint_polynomial_x", "endpoint_polynomial_y", "endpoint_x", "endpoint_y")),
    ):
        payload = {
            "polynomial": [_range_interval(result, prefixes[0]), _range_interval(result, prefixes[1])],
            "final": [_range_interval(result, prefixes[2]), _range_interval(result, prefixes[3])],
            "evaluation_contract": "tau in [0,h]" if stage.startswith("segment") else "tau=h endpoint table",
            "range_method": "Flowpipe::intEvalNormal; polynomial lane zeros both tmvPre/tmv remainders on a copy",
        }
        output_hash = _sha_bytes(_json_bytes(payload))
        rows.append(
            {
                "schema": SCHEMA,
                "ledger_row_index": len(rows),
                "tool": "flowstar_pinned_actual",
                "actual_source": driver_source,
                "flowstar_sha": FLOWSTAR_SHA,
                "observer_patch_sha256": patch_sha,
                "stage_id": stage,
                "iteration": 4,
                "refinement_iteration": -1,
                "component": -1,
                "record_type": "range",
                "basis_id": BASIS["id"],
                "basis": BASIS,
                "support": None,
                "binary_exact_coefficient_encoding": "IEEE-754 binary64 bits + hexfloat",
                "exact_rational_value": "decoded per finite binary64 endpoint",
                "interval_precision_bits": 53,
                "directed_rounding_provenance": "Flow* MPFR interval arithmetic with outward binary64 endpoint extraction",
                "remainder_ownership": {"ordinary": "included only in final lane", "truncation": "included only in final lane", "cutoff": "included only in final lane", "source_ledger": "included only in final lane"},
                "input_artifact_hashes": list(previous),
                "raw_record_sha256": _sha_bytes(_json_bytes(result)),
                "output_artifact_hash": output_hash,
                "classification": "UNRESOLVED",
                "payload": payload,
            }
        )
        previous = [output_hash]

    required = {
        "picard_initial_polynomial",
        "picard_polynomial_iteration",
        "operator_degree_truncation",
        "operator_cutoff",
        "candidate_raw_picard",
        "candidate_subset",
        "refinement_raw_image",
        "refinement_subset",
        "accepted_remainder",
        "next_step_pre_map",
        "normalization_center",
        "normalization_scale",
    }
    missing = required - stage_counts.keys()
    if missing:
        raise ValueError(f"Flow* actual trace misses required stages: {sorted(missing)}")
    classifications = {row["classification"] for row in rows}
    if not classifications <= CLASSIFICATIONS:
        raise AssertionError("internal classification vocabulary error")

    ledger = {"schema": SCHEMA, "rows": rows}
    _write_json(output_dir / "stage_ledger.json", ledger)
    fields = [
        "tool", "flowstar_sha", "stage_id", "iteration", "refinement_iteration",
        "component", "record_type", "basis_id", "classification", "source_file",
        "source_function", "source_line_start", "source_line_end", "input_artifact_hashes",
        "raw_record_sha256", "output_artifact_hash",
    ]
    with (output_dir / "stage_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            source = row["actual_source"]
            writer.writerow(
                {
                    **{name: row.get(name, "") for name in fields},
                    "source_file": source["file"],
                    "source_function": source["function"],
                    "source_line_start": source["line_start"],
                    "source_line_end": source["line_end"],
                    "input_artifact_hashes": json.dumps(row["input_artifact_hashes"], separators=(",", ":")),
                }
            )
    with args.raw_trace.open("rb") as source, gzip.GzipFile(
        filename="", mode="wb", fileobj=(output_dir / "raw_actual_trace.jsonl.gz").open("wb"), mtime=0
    ) as destination:
        destination.write(source.read())
    (output_dir / "observer.patch").write_bytes(source_patch)
    summary = {
        "schema": "flowstar_step1_stage_ledger_summary_v1",
        "flowstar_sha": FLOWSTAR_SHA,
        "observer_patch_sha256": patch_sha,
        "raw_trace_sha256": _sha_path(args.raw_trace),
        "result_csv_sha256": _sha_path(args.result_csv),
        "raw_step1_record_count": len(raw),
        "ledger_row_count": len(rows),
        "stage_counts": stage_counts,
        "required_stages_present": True,
        "classification_status": "UNRESOLVED until neutral comparison",
    }
    _write_json(output_dir / "summary.json", summary)
    manifest = {
        "schema": "flowstar_step1_stage_ledger_manifest_v1",
        "files": {
            path.name: {"sha256": _sha_path(path), "bytes": path.stat().st_size}
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
        "summary": summary,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def subprocess_sha(root: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-trace", type=Path, required=True)
    parser.add_argument("--result-csv", type=Path, required=True)
    parser.add_argument("--flowstar-root", type=Path, required=True)
    parser.add_argument("--observer-patch", type=Path, required=True)
    parser.add_argument("--driver-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = process(parse_args(argv))
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
