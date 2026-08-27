#!/usr/bin/env python3
"""Apply the pre-registered Brusselator soundness and material-gain gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "SECOND_SYSTEM_CONTRACT.md"
GENERIC_CORE_COMMIT = "b88888691eaeefac1fb2e48d5ab0f82ad50c58ac"
FLOWSTAR_COMMIT = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
FLOWSTAR_BENCHMARK_SHA = "b982f7c6f737e4b5e070942dc5fe01fa9d60e17a419a146d42444c71b5bf4f3b"
STEP = 0.02
HORIZON = 20.0
TOLERANCE = 1e-12
CORE_PATHS = (
    "src/torch_tm_flowpipe/accepted_boundary_sr.py",
    "src/torch_tm_flowpipe/symbolic_remainder.py",
    "src/torch_tm_flowpipe/flowpipe.py",
    "src/torch_tm_flowpipe/terminal_checkpoint.py",
    "src/torch_tm_flowpipe/state_equality.py",
)
BOUND_PREFIXES = ("endpoint_x", "endpoint_y", "tube_x", "tube_y")


class SecondSystemEvidenceError(ValueError):
    """Raised when the three-lane evidence is incomplete or malformed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SecondSystemEvidenceError(f"cannot read JSON {path}: {exc}") from exc


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise SecondSystemEvidenceError(f"cannot read CSV {path}: {exc}") from exc


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SecondSystemEvidenceError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise SecondSystemEvidenceError(f"{label} is nonfinite: {value!r}")
    return result


def integer(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SecondSystemEvidenceError(f"{label} is not an integer: {value!r}") from exc


def boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1"}:
        return True
    if lowered in {"false", "0", ""}:
        return False
    raise SecondSystemEvidenceError(f"not a boolean value: {value!r}")


def _width_rows_valid(rows: Sequence[Mapping[str, str]], lane: str) -> bool:
    valid = bool(rows)
    for row_index, row in enumerate(rows):
        if lane != "flowstar" and row.get("status") != "accepted":
            continue
        for prefix in BOUND_PREFIXES:
            lo = finite_float(row.get(f"{prefix}_lo"), f"{lane}[{row_index}].{prefix}_lo")
            hi = finite_float(row.get(f"{prefix}_hi"), f"{lane}[{row_index}].{prefix}_hi")
            width = finite_float(
                row.get(f"{prefix}_width"), f"{lane}[{row_index}].{prefix}_width"
            )
            valid &= lo <= hi and width >= 0.0 and abs(width - (hi - lo)) <= TOLERANCE
    return valid


def _junit_exact_2d_passed(path: Path) -> bool:
    root = ET.parse(path).getroot()
    cases = list(root.findall(".//testcase"))
    if root.tag == "testsuite":
        cases = list(root.findall(".//testcase"))
    matches = [
        case
        for case in cases
        if "test_generic_accepted_boundary_operator_contains_exact_fraction_image[2]"
        in case.get("name", "")
    ]
    return (
        len(matches) == 1
        and matches[0].find("failure") is None
        and matches[0].find("error") is None
        and matches[0].find("skipped") is None
    )


def _core_unchanged(run_commit: str) -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", GENERIC_CORE_COMMIT, run_commit, "--", *CORE_PATHS],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def _torch_lane_checks(
    lane: str,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
) -> dict[str, bool]:
    accepted = [row for row in rows if row.get("status") == "accepted"]
    rejected = [row for row in rows if row.get("status") == "rejected"]
    checks = {
        "nonempty": bool(accepted),
        "row_count_matches": len(accepted) == integer(summary["accepted_steps"], f"{lane}.accepted"),
        "at_most_one_terminal_rejection": len(rejected) <= 1 and (
            not rejected or rows[-1] is rejected[-1]
        ),
        "validation_passed": all(boolean(row["validation_passed"]) for row in accepted),
        "endpoint_published_only_on_accept": all(
            boolean(row.get("endpoint_published", "false")) for row in accepted
        ) and all(not boolean(row.get("endpoint_published", "false")) for row in rejected),
        "finite_consistent_boxes": _width_rows_valid(rows, lane),
        "local_sample_sanity": (
            boolean(summary["sample_solver_ok"])
            and integer(summary["sample_endpoint_violations"], f"{lane}.sample_endpoint") == 0
            and integer(summary["sample_tube_violations"], f"{lane}.sample_tube") == 0
        ),
        "native_horizon_consistent": abs(
            finite_float(summary["completed_horizon"], f"{lane}.horizon")
            - len(accepted) * STEP
        ) <= TOLERANCE,
        "summary_certificate": boolean(summary["certificate_checks_passed"]),
        "rollback": not rejected or boolean(rejected[-1].get("rollback_queue_unchanged", "false")),
    }
    if lane == "torch_generic_no_queue":
        checks["no_queue_created"] = all(not boolean(row["queue_present"]) for row in accepted)
    else:
        checks["owner_accounting"] = all(
            boolean(row["queue_present"])
            and boolean(row["queue_accounting_ok"])
            and boolean(row["owner_widths_nonnegative_finite"])
            for row in accepted
        )
        for row in accepted:
            step = integer(row["step"], "sr100.step")
            remainder = step % 100
            expected_owners = list(range(step - remainder + 1, step + 1)) if remainder else []
            owners = json.loads(row["queue_owner_generations"])
            boundaries = json.loads(row["queue_owner_boundaries"])
            checks["owner_accounting"] &= (
                integer(row["queue_generation"], "sr100.generation") == step
                and integer(row["queue_accepted_boundary"], "sr100.boundary") == step
                and integer(row["queue_size"], "sr100.size") == remainder
                and integer(row["queue_reset_count"], "sr100.reset") == step // 100
                and owners == expected_owners
                and boundaries == expected_owners
            )
    return checks


def _accepted(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    return [row for row in rows if row.get("status") == "accepted"]


def _first_divergence(
    no_queue: Sequence[Mapping[str, str]],
    sr100: Sequence[Mapping[str, str]],
) -> dict[str, Any] | None:
    fields = tuple(f"{prefix}_{bound}_hex" for prefix in BOUND_PREFIXES for bound in ("lo", "hi"))
    for left, right in zip(_accepted(no_queue), _accepted(sr100)):
        differing = [field for field in fields if left.get(field) != right.get(field)]
        if differing:
            return {
                "step": integer(left["step"], "divergence.step"),
                "time": finite_float(left["t_after"], "divergence.time"),
                "fields": differing,
                "maximum_absolute_bound_delta": max(
                    abs(
                        finite_float(left[field.removesuffix("_hex")], f"left.{field}")
                        - finite_float(right[field.removesuffix("_hex")], f"right.{field}")
                    )
                    for field in differing
                ),
            }
    return None


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise SecondSystemEvidenceError("cannot compute percentile of an empty sequence")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _late_metrics(
    no_queue: Sequence[Mapping[str, str]],
    sr100: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    left = _accepted(no_queue)
    right = _accepted(sr100)
    common_steps = min(len(left), len(right))
    common_horizon = common_steps * STEP
    first_late_step = max(1, math.ceil(0.8 * common_steps))
    pairs = list(zip(left[first_late_step - 1 : common_steps], right[first_late_step - 1 : common_steps]))
    result: dict[str, Any] = {
        "common_steps": common_steps,
        "common_horizon": common_horizon,
        "late_first_step": first_late_step,
        "late_last_step": common_steps,
        "late_rows": len(pairs),
    }
    for metric in ("endpoint", "tube"):
        no_over_sr: list[float] = []
        sr_over_no: list[float] = []
        for no_row, sr_row in pairs:
            no_width = finite_float(no_row[f"{metric}_width_sum"], f"no.{metric}")
            sr_width = finite_float(sr_row[f"{metric}_width_sum"], f"sr.{metric}")
            no_over_sr.append(math.inf if sr_width == 0.0 and no_width > 0.0 else no_width / max(sr_width, 1e-300))
            sr_over_no.append(math.inf if no_width == 0.0 and sr_width > 0.0 else sr_width / max(no_width, 1e-300))
        result[f"median_no_queue_over_sr100_{metric}"] = statistics.median(no_over_sr) if no_over_sr else None
        result[f"p95_sr100_over_no_queue_{metric}"] = _percentile(sr_over_no, 0.95) if sr_over_no else None
    return result


def analyze(raw_root: Path, junit: Path) -> dict[str, Any]:
    contract_sha = sha256(CONTRACT)
    summaries = {
        lane: read_json(raw_root / lane / "summary.json")
        for lane in ("flowstar", "torch_generic_no_queue", "torch_generic_sr100")
    }
    rows = {
        lane: read_csv(raw_root / lane / "segments.csv")
        for lane in ("flowstar", "torch_generic_no_queue", "torch_generic_sr100")
    }
    commands = {
        lane: read_json(raw_root / lane / "command.json")
        for lane in ("flowstar", "torch_generic_no_queue", "torch_generic_sr100")
    }
    torch_commit = commands["torch_generic_no_queue"]["commit"]
    source_checks = {
        "contract_hash_matches": all(
            summaries[lane].get("contract_sha256") == contract_sha
            for lane in summaries
        ),
        "torch_same_clean_commit": (
            torch_commit == commands["torch_generic_sr100"].get("commit")
            and not commands["torch_generic_no_queue"].get("worktree_status")
            and not commands["torch_generic_sr100"].get("worktree_status")
            and not summaries["torch_generic_no_queue"].get("worktree_dirty")
            and not summaries["torch_generic_sr100"].get("worktree_dirty")
        ),
        "generic_core_unchanged": _core_unchanged(torch_commit),
        "flowstar_source_pinned": (
            summaries["flowstar"].get("source_commit") == FLOWSTAR_COMMIT
            and summaries["flowstar"].get("benchmark_sha256") == FLOWSTAR_BENCHMARK_SHA
            and not summaries["flowstar"].get("source_tree_tracked_changes_after_build")
        ),
    }
    lane_checks = {
        lane: _torch_lane_checks(lane, summaries[lane], rows[lane])
        for lane in ("torch_generic_no_queue", "torch_generic_sr100")
    }
    flowstar_checks = {
        "finite_consistent_boxes": _width_rows_valid(rows["flowstar"], "flowstar"),
        "row_count_matches": len(rows["flowstar"])
        == integer(summaries["flowstar"]["accepted_steps"], "flowstar.accepted"),
        "native_horizon_consistent": abs(
            finite_float(summaries["flowstar"]["completed_horizon"], "flowstar.horizon")
            - len(rows["flowstar"]) * STEP
        ) <= TOLERANCE,
    }
    exact_test = _junit_exact_2d_passed(junit)
    soundness = (
        exact_test
        and all(source_checks.values())
        and all(all(checks.values()) for checks in lane_checks.values())
        and all(flowstar_checks.values())
    )
    divergence = _first_divergence(
        rows["torch_generic_no_queue"], rows["torch_generic_sr100"]
    )
    late = _late_metrics(rows["torch_generic_no_queue"], rows["torch_generic_sr100"])
    no_horizon = finite_float(
        summaries["torch_generic_no_queue"]["completed_horizon"], "no_queue.horizon"
    )
    sr_horizon = finite_float(
        summaries["torch_generic_sr100"]["completed_horizon"], "sr100.horizon"
    )
    flowstar_completed = (
        bool(summaries["flowstar"].get("completed_requested_horizon"))
        and abs(finite_float(summaries["flowstar"]["completed_horizon"], "flowstar.horizon") - HORIZON)
        <= TOLERANCE
    )
    horizon_gain = sr_horizon >= 18.0 and sr_horizon + TOLERANCE >= no_horizon and sr_horizon - no_horizon >= 2.0 - TOLERANCE
    both_t20 = no_horizon >= HORIZON - TOLERANCE and sr_horizon >= HORIZON - TOLERANCE
    tightness_gain = bool(
        both_t20
        and late["median_no_queue_over_sr100_endpoint"] >= 1.10
        and late["median_no_queue_over_sr100_tube"] >= 1.10
        and late["p95_sr100_over_no_queue_endpoint"] <= 1.05
        and late["p95_sr100_over_no_queue_tube"] <= 1.05
    )
    material_gain = horizon_gain or tightness_gain
    eligible = late["common_horizon"] >= 2.0 - TOLERANCE and flowstar_completed
    if not soundness:
        status = "C3_GENERICITY_SOUNDNESS_GATE_FAILED_STOP"
    elif not eligible:
        status = "C3_GENERIC_POLYNOMIAL_PLANT_CORE_VALIDATED"
    elif material_gain:
        status = "C3_GENERIC_CORE_VALIDATED__SECOND_SYSTEM_PRODUCTION_USEFUL"
    else:
        status = "C3_GENERIC_CORE_VALIDATED__SECOND_SYSTEM_NO_MATERIAL_GAIN"
    return {
        "schema": "torch_tm_flowpipe.brusselator_second_system_result/1",
        "status": status,
        "soundness_gate_passed": soundness,
        "source_checks": source_checks,
        "lane_checks": lane_checks,
        "flowstar_checks": flowstar_checks,
        "exact_fraction_2d_test_passed": exact_test,
        "flowstar_completed_t20": flowstar_completed,
        "native_horizons": {
            lane: summaries[lane]["completed_horizon"] for lane in summaries
        },
        "accepted_steps": {lane: summaries[lane]["accepted_steps"] for lane in summaries},
        "first_live_divergence": divergence,
        "late_common_prefix": late,
        "material_gain": {
            "horizon_criterion": horizon_gain,
            "late_tightness_criterion": tightness_gain,
            "passed": material_gain,
            "eligible": eligible,
        },
        "runtime_seconds": {
            lane: summaries[lane].get("solver_wall_seconds") for lane in summaries
        },
        "contract_sha256": contract_sha,
        "torch_run_commit": torch_commit,
        "junit_sha256": sha256(junit),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(args.raw_root.resolve(), args.junit.resolve())
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 1 if result["status"] == "C3_GENERICITY_SOUNDNESS_GATE_FAILED_STOP" else 0


if __name__ == "__main__":
    raise SystemExit(main())
