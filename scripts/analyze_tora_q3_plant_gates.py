#!/usr/bin/env python3
"""Derive hierarchical plant-gate evidence from complete private replay traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        header = json.loads(next(handle))
        return header, [json.loads(line) for line in handle]


def scalar_detail(a: float, b: float) -> dict[str, Any]:
    return {
        "xiangru_decimal": float(a),
        "torch_decimal": float(b),
        "xiangru_hex": float(a).hex(),
        "torch_hex": float(b).hex(),
        "absolute_difference": abs(float(a) - float(b)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xiangru-jsonl", type=Path, required=True)
    parser.add_argument("--torch-jsonl", type=Path, required=True)
    parser.add_argument("--private-detail", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    args = parser.parse_args()
    x_header, x_rows = load(args.xiangru_jsonl)
    t_header, t_rows = load(args.torch_jsonl)
    if len(x_rows) != len(t_rows) or len(x_rows) != 200:
        raise ValueError("complete T20 traces are required")
    if x_header["basis_exponents"] != t_header["basis_exponents"]:
        raise ValueError("basis order differs")
    for x_row, t_row in zip(x_rows, t_rows, strict=True):
        for field in ("segment_index", "physical_time", "controller_period", "local_segment", "leaf_id"):
            if x_row[field] != t_row[field]:
                raise ValueError(f"alignment mismatch: {field}")

    first_x, first_t = x_rows[0], t_rows[0]
    detail = {
        "schema": "tora_q3_one_leaf_one_step_hex_detail_v1",
        "leaf_id": 0,
        "segment_index": 1,
        "physical_time": 0.1,
        "basis_exponents": x_header["basis_exponents"],
        "coefficient": [],
        "remainder": [],
        "endpoint": [],
        "tube": [],
    }
    coefficient_max = 0.0
    for state in range(5):
        state_rows = []
        for slot in range(84):
            row = scalar_detail(
                first_x["polynomial_coefficient_vector"][0][state][slot],
                first_t["polynomial_coefficient_vector"][0][state][slot],
            )
            coefficient_max = max(coefficient_max, row["absolute_difference"])
            state_rows.append(row)
        detail["coefficient"].append(state_rows)
        detail["remainder"].append({
            "lower": scalar_detail(first_x["interval_remainder"]["lower"][0][state], first_t["interval_remainder"]["lower"][0][state]),
            "upper": scalar_detail(first_x["interval_remainder"]["upper"][0][state], first_t["interval_remainder"]["upper"][0][state]),
        })
        detail["endpoint"].append({
            "lower": scalar_detail(first_x["endpoint"]["lower"][0][state], first_t["endpoint"]["lower"][0][state]),
            "upper": scalar_detail(first_x["endpoint"]["upper"][0][state], first_t["endpoint"]["upper"][0][state]),
        })
        detail["tube"].append({
            "lower": scalar_detail(first_x["tube"]["lower"][0][state], first_t["tube"]["lower"][0][state]),
            "upper": scalar_detail(first_x["tube"]["upper"][0][state], first_t["tube"]["upper"][0][state]),
        })
    args.private_detail.parent.mkdir(parents=True, exist_ok=True)
    args.private_detail.write_text(json.dumps(detail, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gate_specs = [
        ("one_leaf_one_step", 1, 1),
        ("b48_one_step", 1, 48),
        ("b48_t1", 10, 48),
        ("b48_t5", 50, 48),
        ("b48_t10", 100, 48),
        ("b48_t20", 200, 48),
    ]
    gates = []
    for name, segment_count, expected_leaves in gate_specs:
        selected = t_rows[:segment_count]
        accepted = all(all(row["accepted"][:expected_leaves]) for row in selected)
        gates.append({
            "gate": name,
            "status": "PASS" if accepted else "FAIL",
            "completed_segments": segment_count if accepted else next((index for index, row in enumerate(selected) if not all(row["accepted"][:expected_leaves])), 0),
            "certified_horizon": segment_count * 0.1 if accepted else "N/A",
            "expected_leaf_count": expected_leaves,
        })
    public = {
        "schema": "tora_q3_native_plant_gates_v1",
        "lane": "common_control_plant_replay",
        "period_local_observation_restart": True,
        "not_independent_closed_loop": True,
        "gates": gates,
        "basis_gate": {
            "status": "PASS",
            "variables": x_header["basis_variables"],
            "slot_count": len(x_header["basis_exponents"]),
            "slot_permutation": list(range(84)),
            "torch_fingerprint": t_header["basis_fingerprint"],
        },
        "one_leaf_one_step": {
            "leaf_id": 0,
            "accepted_by_xiangru": first_x["accepted"][0],
            "accepted_by_torch": first_t["accepted"][0],
            "maximum_coefficient_absolute_difference": coefficient_max,
            "coefficient_comparison_scope": "legal for segment 1 only: identical diagonal initial normalization and identity slot permutation",
            "private_decimal_hex_detail_sha256": sha256(args.private_detail),
        },
        "source_hashes": {
            "xiangru_trace": sha256(args.xiangru_jsonl),
            "torch_trace": sha256(args.torch_jsonl),
            "controller_trace": x_header["controller_trace_sha256"],
        },
    }
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    args.public_summary.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS" if all(row["status"] == "PASS" for row in gates) else "FAIL", "gates": gates, "maximum_coefficient_absolute_difference": coefficient_max}))
    return 0 if all(row["status"] == "PASS" for row in gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
