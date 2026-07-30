#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = {
    "RUN_MANIFEST.json",
    "CONFIG_MANIFEST.json",
    "ENVIRONMENT.json",
    "PROVENANCE.json",
    "RAW_OBSERVATIONS.csv",
    "SUMMARY.csv",
    "FAILURES.csv",
    "ELIGIBILITY.csv",
    "PRIMARY_PARETO.csv",
    "EXPLORATORY.csv",
    "FIGURE_MANIFEST.csv",
    "REPORT.md",
    "COMPLETE_TEST.log",
    "SHA256SUMS",
}
IDENTITY_FIELDS = (
    "tool",
    "variant",
    "system",
    "h",
    "requested_horizon",
    "requested_order",
    "effective_order",
    "basis_id",
    "remainder_policy",
    "step_policy",
    "bound_semantics",
    "runtime_boundary_version",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _explicit_true(value: Any) -> bool:
    return str(value).strip() == "True"


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _identity_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = {field: row.get(field, "") for field in IDENTITY_FIELDS}
    for field in ("h", "requested_horizon"):
        payload[field] = float(payload[field])
    for field in ("requested_order", "effective_order"):
        text = str(payload[field])
        if text.isdigit():
            payload[field] = int(text)
    return payload


def _config_id(row: Mapping[str, Any]) -> str:
    payload = _identity_payload(row)
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _observation_key(row: Mapping[str, Any]) -> tuple[str, str, str, float, float]:
    return (
        str(row.get("tool", "")),
        str(row.get("variant", "")),
        str(row.get("system", "")),
        round(float(row.get("h", math.nan)), 12),
        round(float(row.get("requested_horizon", math.nan)), 12),
    )


def _pareto_ids(rows: Iterable[Mapping[str, str]]) -> set[str]:
    groups: dict[tuple[str, float], list[Mapping[str, str]]] = (
        defaultdict(list)
    )
    for row in rows:
        groups[
            (
                row["system"],
                round(float(row["evaluation_time"]), 12),
            )
        ].append(row)
    frontier: set[str] = set()
    for group in groups.values():
        for candidate in group:
            width = float(candidate["width_at_evaluation_time"])
            runtime = float(
                candidate["steady_total_configuration_time_s"]
            )
            dominated = any(
                other is not candidate
                and float(other["width_at_evaluation_time"]) <= width
                and float(
                    other["steady_total_configuration_time_s"]
                )
                <= runtime
                and (
                    float(other["width_at_evaluation_time"]) < width
                    or float(
                        other["steady_total_configuration_time_s"]
                    )
                    < runtime
                )
                for other in group
            )
            if not dominated:
                frontier.add(candidate["config_id"])
    return frontier


def _sha_manifest(output: Path) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for line in (output / "SHA256SUMS").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, relative = line.split("  ", 1)
        path = output / relative
        if not path.is_file():
            failures.append(f"missing checksum target: {relative}")
        elif _sha256(path) != expected:
            failures.append(f"checksum mismatch: {relative}")
    return not failures, failures


def _worktree_changes_outside(output: Path) -> list[str]:
    process = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    outside: list[str] = []
    for line in process.stdout.splitlines():
        relative = line[3:]
        candidate = (REPO_ROOT / relative).resolve()
        try:
            candidate.relative_to(output)
        except ValueError:
            outside.append(line)
    return outside


def audit(output: Path) -> dict[str, Any]:
    output = output.resolve()
    failures: list[str] = []
    checks: dict[str, Any] = {}

    missing = sorted(
        name for name in REQUIRED_FILES if not (output / name).is_file()
    )
    checks["required_files"] = {"passed": not missing, "missing": missing}
    failures.extend(f"missing required file: {name}" for name in missing)
    if missing:
        return _finish(output, checks, failures, authoritative=False)

    run = _read_json(output / "RUN_MANIFEST.json")
    config = _read_json(output / "CONFIG_MANIFEST.json")
    summary = _read_csv(output / "SUMMARY.csv")
    raw = _read_csv(output / "RAW_OBSERVATIONS.csv")
    failures_table = _read_csv(output / "FAILURES.csv")
    eligibility = _read_csv(output / "ELIGIBILITY.csv")
    primary = _read_csv(output / "PRIMARY_PARETO.csv")
    figure_manifest = _read_csv(output / "FIGURE_MANIFEST.csv")
    required_repetitions = int(config["required_repetitions"])
    expected = config["expected_configurations"]
    expected_ids = {row["config_id"] for row in expected}
    observed_ids = [_config_id(row) for row in summary]
    delivered_ids = [row.get("config_id", "") for row in summary]

    exact_configs = (
        set(observed_ids) == expected_ids
        and observed_ids == delivered_ids
        and len(observed_ids) == len(expected_ids)
    )
    checks["expected_configurations"] = {
        "passed": exact_configs,
        "expected": len(expected_ids),
        "observed": len(observed_ids),
        "missing": sorted(expected_ids - set(observed_ids)),
        "unexpected": sorted(set(observed_ids) - expected_ids),
    }
    if not exact_configs:
        failures.append("expected configuration identities do not match")

    duplicates = sorted(
        identity
        for identity, count in Counter(observed_ids).items()
        if count != 1
    )
    checks["configuration_identity_uniqueness"] = {
        "passed": not duplicates,
        "duplicates": duplicates,
    }
    if duplicates:
        failures.append("duplicate configuration identity")

    repetition_failures = []
    for row in summary:
        if (
            int(row["runtime_repetitions"]) != required_repetitions
            or not _explicit_true(
                row["all_required_repetitions_present"]
            )
        ):
            repetition_failures.append(row["config_id"])
    checks["formal_repetitions"] = {
        "passed": not repetition_failures,
        "required": required_repetitions,
        "failed_config_ids": repetition_failures,
    }
    if repetition_failures:
        failures.append("required repetitions are incomplete")

    expected_by_observation_key = {
        _observation_key(row): row["config_id"] for row in expected
    }
    raw_groups: dict[tuple[str, str, str, float, float], list[dict[str, str]]] = (
        defaultdict(list)
    )
    unexpected_raw: list[str] = []
    for row in raw:
        key = _observation_key(row)
        if key not in expected_by_observation_key:
            unexpected_raw.append("|".join(map(str, key)))
        raw_groups[key].append(row)
    raw_repetition_failures: list[dict[str, Any]] = []
    for key, config_id in expected_by_observation_key.items():
        observations = raw_groups.get(key, [])
        cold = [
            row
            for row in observations
            if row.get("measurement_phase") == "cold"
        ]
        steady = [
            row
            for row in observations
            if row.get("measurement_phase") == "steady"
        ]
        cold_indices = {
            int(float(row.get("repetition", -1))) for row in cold
        }
        steady_indices = {
            int(float(row.get("repetition", -1))) for row in steady
        }
        if (
            len(cold) != 1
            or cold_indices != {0}
            or len(steady) != required_repetitions
            or steady_indices
            != set(range(1, required_repetitions + 1))
        ):
            raw_repetition_failures.append(
                {
                    "config_id": config_id,
                    "cold_count": len(cold),
                    "cold_indices": sorted(cold_indices),
                    "steady_count": len(steady),
                    "steady_indices": sorted(steady_indices),
                }
            )
    raw_repetitions_passed = (
        not raw_repetition_failures
        and not unexpected_raw
        and len(raw_groups) == len(expected_by_observation_key)
    )
    checks["raw_observation_repetitions"] = {
        "passed": raw_repetitions_passed,
        "required_steady_repetitions": required_repetitions,
        "expected_configurations": len(expected_by_observation_key),
        "observed_groups": len(raw_groups),
        "unexpected_groups": sorted(set(unexpected_raw)),
        "failures": raw_repetition_failures,
    }
    if not raw_repetitions_passed:
        failures.append("raw observations do not contain exact repetitions")

    runtime_failures = [
        row["config_id"]
        for row in summary
        if not (
            math.isfinite(
                _float(row["steady_total_configuration_time_s"])
            )
            and _float(row["steady_total_configuration_time_s"]) > 0
            and row["runtime_boundary_version"]
            == run["runtime_boundary_version"]
        )
    ]
    checks["runtime_boundary"] = {
        "passed": not runtime_failures,
        "failed_config_ids": runtime_failures,
    }
    if runtime_failures:
        failures.append("runtime boundary or finite-runtime check failed")

    validation_failures = []
    for row in primary:
        for field in (
            "native_validation_passed",
            "trajectory_sanity_passed",
            "analytic_containment_passed",
        ):
            applicability = row.get(
                field.replace("_passed", "_applicability"),
                "required",
            )
            if applicability == "not_applicable":
                continue
            if applicability != "required" or not _explicit_true(
                row.get(field)
            ):
                validation_failures.append(
                    f"{row['config_id']}:{field}"
                )
    invalid_marked_eligible = []
    for row in eligibility:
        if not _explicit_true(row.get("primary_numerical_eligible")):
            continue
        for field in (
            "native_validation_passed",
            "trajectory_sanity_passed",
            "analytic_containment_passed",
        ):
            applicability = row.get(
                field.replace("_passed", "_applicability"),
                "required",
            )
            if applicability == "not_applicable":
                continue
            if applicability != "required" or not _explicit_true(
                row.get(field)
            ):
                invalid_marked_eligible.append(
                    f"{row['config_id']}:{field}"
                )
    validation_failures.extend(invalid_marked_eligible)
    checks["fail_closed_validations"] = {
        "passed": not validation_failures,
        "failures": validation_failures,
    }
    if validation_failures:
        failures.append("required validation is not explicit True")

    horizon_failures = [
        row["config_id"]
        for row in summary
        if _explicit_true(row["completed_requested_horizon"])
        and not math.isclose(
            float(row["requested_horizon"]),
            float(row["successful_horizon"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ]
    checks["horizon_semantics"] = {
        "passed": not horizon_failures,
        "failed_config_ids": horizon_failures,
    }
    if horizon_failures:
        failures.append("requested and successful horizons differ")

    incomplete_ids = {
        row["config_id"]
        for row in summary
        if not _explicit_true(row["completed_requested_horizon"])
        or row["failure_category"] != "completed"
    }
    failure_ids = {row["config_id"] for row in failures_table}
    checks["failure_completeness"] = {
        "passed": incomplete_ids == failure_ids,
        "expected_failure_ids": sorted(incomplete_ids),
        "delivered_failure_ids": sorted(failure_ids),
    }
    if incomplete_ids != failure_ids:
        failures.append("failure table does not match incomplete configs")

    partition_failures = [
        row["config_id"]
        for row in eligibility
        if row["bound_semantics"] != "raw_endpoint"
        or not _explicit_true(row["primary_comparable"])
    ]
    checks["bound_partition"] = {
        "passed": not partition_failures,
        "failed_config_ids": partition_failures,
    }
    if partition_failures:
        failures.append("raw/tightened comparison partition failed")

    flowstar_keys = [
        (
            row["config_id"],
            row["requested_order"],
            row["effective_order"],
            row["basis_id"],
        )
        for row in summary
        if row["tool"] == "flowstar"
    ]
    flowstar_collisions = (
        len({item[0] for item in flowstar_keys}) != len(flowstar_keys)
        or any(not item[1] or not item[2] or not item[3] for item in flowstar_keys)
    )
    checks["flowstar_order_basis_identity"] = {
        "passed": not flowstar_collisions,
        "rows": flowstar_keys,
    }
    if flowstar_collisions:
        failures.append("Flowstar order/basis identity collision")

    primary_ids = {row["config_id"] for row in primary}
    eligible_primary_ids = {
        row["config_id"]
        for row in eligibility
        if _explicit_true(row["primary_numerical_eligible"])
    }
    excluded_marked_frontier = [
        row["config_id"]
        for row in eligibility
        if not _explicit_true(row["primary_numerical_eligible"])
        and _explicit_true(row["width_runtime_pareto"])
    ]
    ordering_passed = (
        primary_ids == eligible_primary_ids
        and not excluded_marked_frontier
    )
    checks["eligibility_before_pareto"] = {
        "passed": ordering_passed,
        "primary_ids": sorted(primary_ids),
        "eligible_ids": sorted(eligible_primary_ids),
        "excluded_marked_frontier": excluded_marked_frontier,
    }
    if not ordering_passed:
        failures.append("eligibility/Pareto ordering check failed")

    recomputed = _pareto_ids(primary)
    delivered_frontier = {
        row["config_id"]
        for row in primary
        if _explicit_true(row["width_runtime_pareto"])
    }
    checks["independent_pareto"] = {
        "passed": recomputed == delivered_frontier,
        "recomputed": sorted(recomputed),
        "delivered": sorted(delivered_frontier),
    }
    if recomputed != delivered_frontier:
        failures.append("independent Pareto frontier differs")

    figure_failures = []
    for row in figure_manifest:
        source = output / row["source_files"]
        figure = output / row["figure"]
        if (
            not source.is_file()
            or not figure.is_file()
            or _sha256(source) != row["source_sha256"]
        ):
            figure_failures.append(row.get("figure", "unknown"))
    checks["figure_sources"] = {
        "passed": not figure_failures,
        "failures": figure_failures,
    }
    if figure_failures:
        failures.append("figure source hash check failed")

    report = (output / "REPORT.md").read_text(encoding="utf-8")
    report_requirements = [
        f"- raw_observations: {len(raw)}",
        f"- summary_rows: {len(summary)}",
        f"- eligible_primary_rows: {len(primary)}",
        _sha256(output / "SUMMARY.csv"),
        _sha256(output / "PRIMARY_PARETO.csv"),
    ]
    report_missing = [
        value for value in report_requirements if value not in report
    ]
    checks["report_traceability"] = {
        "passed": not report_missing,
        "missing": report_missing,
    }
    if report_missing:
        failures.append("report values are not traceable")

    checksums_passed, checksum_failures = _sha_manifest(output)
    checks["sha256_manifest"] = {
        "passed": checksums_passed,
        "failures": checksum_failures,
    }
    failures.extend(checksum_failures)

    test_log = (output / "COMPLETE_TEST.log").read_text(
        encoding="utf-8"
    )
    test_exit_codes = re.findall(r"^exit_code=(\d+)$", test_log, re.M)
    test_passed = bool(test_exit_codes) and all(
        value == "0" for value in test_exit_codes
    )
    checks["complete_test_log"] = {
        "passed": test_passed,
        "exit_codes": test_exit_codes,
    }
    if not test_passed:
        failures.append("complete test log has no successful exit code")

    actual_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    code_frozen = (
        run["code_sha"] == run["code_freeze_sha"] == actual_sha
    )
    checks["code_freeze"] = {
        "passed": code_frozen,
        "run_sha": run["code_sha"],
        "actual_sha": actual_sha,
    }
    if not code_frozen:
        failures.append("run code SHA does not equal code-freeze SHA")

    fresh = (
        run.get("output_preexisted") is False
        and run.get("output_directory_empty_at_start") is True
    )
    checks["fresh_output_no_stale_files"] = {"passed": fresh}
    if not fresh:
        failures.append("output was not created from a fresh directory")

    profile_authoritative = bool(
        config["profile"].get("authoritative", False)
    )
    outside = (
        _worktree_changes_outside(output)
        if profile_authoritative
        else []
    )
    checks["worktree_changes_confined_to_run"] = {
        "passed": not outside,
        "outside_changes": outside,
        "applicability": (
            "required" if profile_authoritative else "not_applicable_smoke"
        ),
    }
    if outside:
        failures.append("worktree has changes outside the run directory")

    return _finish(
        output,
        checks,
        failures,
        authoritative=profile_authoritative and not failures,
    )


def _finish(
    output: Path,
    checks: Mapping[str, Any],
    failures: list[str],
    *,
    authoritative: bool,
) -> dict[str, Any]:
    result = {
        "passed": not failures,
        "authoritative": authoritative,
        "status": (
            "accepted_authoritative"
            if authoritative and not failures
            else (
                "accepted_smoke_non_authoritative"
                if not failures
                else "failed_acceptance"
            )
        ),
        "checks": checks,
        "failures": failures,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (output / "INDEPENDENT_AUDIT.json").write_text(
        text, encoding="utf-8"
    )
    (output / "final_acceptance.json").write_text(
        text, encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
