#!/usr/bin/env python3
"""Derive the three-way Gate-B actual-path/copied-probe equivalence result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PUBLISHED_MAP = {
    "endpoint_x_lo": "flowstar_tau_h_endpoint_x_lo",
    "endpoint_x_hi": "flowstar_tau_h_endpoint_x_hi",
    "endpoint_y_lo": "flowstar_tau_h_endpoint_y_lo",
    "endpoint_y_hi": "flowstar_tau_h_endpoint_y_hi",
    "segment_x_lo": "flowstar_full_step_tube_x_lo",
    "segment_x_hi": "flowstar_full_step_tube_x_hi",
    "segment_y_lo": "flowstar_full_step_tube_y_lo",
    "segment_y_hi": "flowstar_full_step_tube_y_hi",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(path)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def observer_full_state(row: Mapping[str, str]) -> str:
    domain = row["domain_canonical"].replace("|", ";")
    return (
        f"tmvPre{{{row['tmvPre_canonical']}}}"
        f"|tmv{{{row['tmv_canonical']}}}"
        f"|domain={domain}"
        f"|queue{{{row['queue_canonical']}}}"
    )


def observer_retained(row: Mapping[str, str]) -> str:
    return f"tmvPre{{{row['tmvPre_canonical']}}}|tmv{{{row['tmv_canonical']}}}"


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    clean_path = args.clean_stock.resolve()
    instrumented_path = args.instrumented_stock.resolve()
    copied_path = args.copied_probe.resolve()
    observer_path = args.observer.resolve()
    clean = read_csv(clean_path)
    instrumented = read_csv(instrumented_path)
    copied = read_csv(copied_path)
    observer = read_csv(observer_path)
    clean_summary = read_json(args.clean_summary.resolve())
    instrumented_summary = read_json(args.instrumented_summary.resolve())

    if len(clean) != len(instrumented) or len(clean) != len(copied) or len(clean) != 1000:
        raise ValueError("three-way accepted-prefix length mismatch")
    if len(observer) != 2000:
        raise ValueError("actual-path observer did not record pre/post for every step")
    if clean_path.read_bytes() != instrumented_path.read_bytes():
        raise ValueError("clean and instrumented actual-path outputs differ")
    if clean_summary != instrumented_summary:
        raise ValueError("clean and instrumented actual-path summaries differ")

    rows: list[dict[str, Any]] = []
    copied_published_mismatches = 0
    observer_retained_mismatches = 0
    post_to_next_mismatches = 0
    for index, (clean_row, copied_row) in enumerate(zip(clean, copied, strict=True)):
        pre = observer[2 * index]
        post = observer[2 * index + 1]
        if pre["phase"] != "pre_reset" or post["phase"] != "post_reset":
            raise ValueError(f"observer phase order mismatch at step {index + 1}")
        published = {
            field: clean_row[field] == copied_row[copied_field]
            for field, copied_field in PUBLISHED_MAP.items()
        }
        schedule = {
            "step": int(clean_row["step"]) == int(copied_row["accepted_step_index"]) + 1,
            "t_after": clean_row["t_after"] == copied_row["t_after"],
            "h": clean_row["h"] == copied_row["h"],
            "t_after_hex_binary64": (
                float.fromhex(clean_row["t_after_hex"]).hex()
                == float.fromhex(copied_row["t_after_hex"]).hex()
            ),
            "h_hex_binary64": (
                float.fromhex(clean_row["h_hex"]).hex()
                == float.fromhex(copied_row["h_hex"]).hex()
            ),
        }
        retained_exact = (
            observer_retained(pre) == copied_row["retained_coefficients_binary_canonical"]
        )
        next_exact: bool | None = None
        if index + 1 < len(copied):
            next_exact = (
                observer_full_state(post)
                == copied[index + 1]["prestate_state_binary_canonical"]
            )
        copied_published_mismatches += not (all(published.values()) and all(schedule.values()))
        observer_retained_mismatches += not retained_exact
        post_to_next_mismatches += next_exact is False
        rows.append(
            {
                "step": index + 1,
                "published_and_schedule_exact": all(published.values()) and all(schedule.values()),
                "actual_pre_reset_retained_state_exact": retained_exact,
                "actual_post_reset_equals_next_copied_prestate": next_exact,
                "queue_J_size_pre_reset": pre["queue_J_size"],
                "queue_Phi_L_size_pre_reset": pre["queue_Phi_L_size"],
            }
        )
    write_csv(output / "stepwise_equivalence.csv", rows)

    result = {
        "schema": "flowstar_actual_copied_probe_equivalence_v1",
        "flowstar_sha": "b85a3211748cb77b736fe4ad42ee02d8d2b81148",
        "actual_public_entry": "ODE::reach -> ODE::reach_symbolic_remainder -> Flowpipe::advance",
        "copied_entry": "probe-local traced_advance_adaptive_symbolic",
        "clean_instrumented": {
            "clean_sha256": sha256(clean_path),
            "instrumented_sha256": sha256(instrumented_path),
            "byte_exact": True,
            "accepted_decisions_exact": True,
            "published_endpoint_segment_prefix_exact": True,
        },
        "actual_copied": {
            "compared_steps": len(clean),
            "published_schedule_mismatches": copied_published_mismatches,
            "actual_pre_reset_retained_state_mismatches": observer_retained_mismatches,
            "actual_post_reset_to_next_copied_prestate_mismatches": post_to_next_mismatches,
            "actual_post_reset_to_next_copied_prestate_comparisons": len(clean) - 1,
            "retained_state_includes": [
                "term support",
                "binary-exact coefficients",
                "ordinary remainders",
                "tmv right map",
            ],
            "full_prestate_includes": [
                "tmvPre",
                "tmv",
                "domain",
                "center/scale/scalars",
                "queue J/Phi_L/max_size",
            ],
        },
        "first_bitwise_difference": None,
        "first_semantic_difference": None,
        "first_decision_relevant_difference": None,
        "equivalence_scope": (
            "pinned Flow* b85a321, frozen VDP B1 complete-O4 h=.01 T=10, "
            "symbolic-remainder max_size=100"
        ),
        "stepwise_driver_warning": (
            "Repeated one-step ode.reach calls are not an equivalent driver because Flow* "
            "shortens a call horizon near THRESHOLD_HIGH; only the one-shot actual path is eligible."
        ),
        "status": "STOCK_COPIED_PROBE_EQUIVALENCE_CLOSED",
    }
    if (
        copied_published_mismatches
        or observer_retained_mismatches
        or post_to_next_mismatches
    ):
        result["status"] = "SOURCE_TRACE_NOT_STOCK_EQUIVALENT"
        raise RuntimeError(json.dumps(result, sort_keys=True))
    write_json(output / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-stock", type=Path, required=True)
    parser.add_argument("--instrumented-stock", type=Path, required=True)
    parser.add_argument("--clean-summary", type=Path, required=True)
    parser.add_argument("--instrumented-summary", type=Path, required=True)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--copied-probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), sort_keys=True, allow_nan=False))
