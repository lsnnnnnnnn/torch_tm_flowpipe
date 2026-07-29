#!/usr/bin/env python3
"""Audit completed result tables for parseability and semantic hygiene."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from common import write_json


NONFINITE_LITERALS = {"nan", "+nan", "-nan", "inf", "+inf", "-inf",
                      "infinity", "+infinity", "-infinity"}
NUMERIC_TEXT = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)


def _number(value: Any) -> float | None:
    try:
        if value in ("", None, "unavailable", "reference"):
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _json_nonfinite(value: Any, location: str = "$") -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [location]
    if isinstance(value, list):
        return [
            item
            for index, child in enumerate(value)
            for item in _json_nonfinite(child, f"{location}[{index}]")
        ]
    if isinstance(value, dict):
        return [
            item
            for key, child in value.items()
            for item in _json_nonfinite(child, f"{location}.{key}")
        ]
    return []


def audit(output: Path) -> dict[str, Any]:
    failures: list[str] = []
    csv_rows: list[dict[str, Any]] = []
    total_rows = 0
    nonfinite_cells: list[dict[str, Any]] = []
    horizon_violations: list[dict[str, Any]] = []
    for path in sorted(output.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if not fieldnames:
                failures.append(f"{path.name}: missing CSV header")
                continue
            if len(fieldnames) != len(set(fieldnames)):
                failures.append(f"{path.name}: duplicate CSV column")
            rows = list(reader)
        total_rows += len(rows)
        for row_index, row in enumerate(rows, start=2):
            for field, value in row.items():
                text = str(value).strip()
                lower = text.lower()
                numeric_overflow = (
                    bool(NUMERIC_TEXT.fullmatch(text))
                    and not math.isfinite(float(text))
                )
                if lower in NONFINITE_LITERALS or numeric_overflow:
                    nonfinite_cells.append(
                        {
                            "path": path.name,
                            "row": row_index,
                            "field": field,
                            "value": value,
                        }
                    )
            requested = _number(
                row.get("requested_horizon", row.get("horizon"))
            )
            successful = _number(
                row.get(
                    "successful_horizon",
                    row.get("evaluation_time", row.get("time")),
                )
            )
            if (
                requested is not None
                and successful is not None
                and successful > requested + 1e-10 * max(1.0, abs(requested))
            ):
                horizon_violations.append(
                    {
                        "path": path.name,
                        "row": row_index,
                        "requested_horizon": requested,
                        "successful_or_evaluation_horizon": successful,
                    }
                )
            requested_step = _number(
                row.get("requested_step", row.get("h"))
            )
            accepted_step = _number(row.get("accepted_step"))
            if (
                requested_step is not None
                and accepted_step is not None
                and accepted_step
                > requested_step
                + 1e-12 * max(1.0, abs(requested_step))
            ):
                horizon_violations.append(
                    {
                        "path": path.name,
                        "row": row_index,
                        "requested_step": requested_step,
                        "accepted_step": accepted_step,
                    }
                )
        csv_rows.append(
            {
                "path": path.name,
                "rows": len(rows),
                "columns": len(fieldnames),
                "runtime_unit_columns": sorted(
                    field
                    for field in fieldnames
                    if field.endswith("_time_s")
                    or field.endswith("_runtime_s")
                    or field in {
                        "runtime_s",
                        "total_runtime_s",
                        "execution_time_s",
                    }
                ),
                "horizon_columns": sorted(
                    field
                    for field in fieldnames
                    if "horizon" in field or field == "time"
                ),
            }
        )

    json_rows: list[dict[str, Any]] = []
    for path in sorted(output.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle, parse_constant=_reject_constant)
            nonfinite_locations = _json_nonfinite(payload)
            if nonfinite_locations:
                failures.append(
                    f"{path.name}: non-finite JSON number(s) at "
                    + ", ".join(nonfinite_locations[:10])
                )
            json_rows.append(
                {
                    "path": path.name,
                    "top_level_type": type(payload).__name__,
                    "entries": (
                        len(payload)
                        if isinstance(payload, (dict, list))
                        else 1
                    ),
                }
            )
        except Exception as exc:
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")

    if nonfinite_cells:
        failures.append(
            f"{len(nonfinite_cells)} non-finite CSV literal(s) found"
        )
    if horizon_violations:
        failures.append(
            f"{len(horizon_violations)} step/horizon semantic violation(s)"
        )
    required = {
        "raw_results.csv",
        "correctness_checks.json",
        "final_acceptance.json",
        "bern_feasibility.json",
        "three_tool_deep_study_report.md",
    }
    missing = sorted(name for name in required if not (output / name).is_file())
    if missing:
        failures.append(f"missing required artifacts: {missing}")
    final_acceptance = (
        json.loads(
            (output / "final_acceptance.json").read_text(encoding="utf-8")
        )
        if (output / "final_acceptance.json").exists()
        else {}
    )
    if final_acceptance and not final_acceptance.get("passed", False):
        failures.append("final_acceptance.json is not passed")

    result = {
        "passed": not failures,
        "failures": failures,
        "csv_files": csv_rows,
        "csv_file_count": len(csv_rows),
        "csv_total_rows": total_rows,
        "json_files": json_rows,
        "json_file_count": len(json_rows),
        "nonfinite_csv_cells": nonfinite_cells,
        "horizon_or_step_violations": horizon_violations,
        "unit_policy": (
            "seconds use *_s fields; widths are in native state units; "
            "memory fields name KiB or bytes; unavailable is never zero"
        ),
        "sampling_policy": "deterministic trajectory sampling is non-proof",
    }
    write_json(output / "artifact_quality_audit.json", result)
    if failures:
        raise SystemExit("; ".join(failures))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = audit(Path(args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
