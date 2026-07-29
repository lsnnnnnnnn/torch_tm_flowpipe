#!/usr/bin/env python3
"""Generate the minimal Flow* Riccati root-cause evidence bundle."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from common import (
    analytic_endpoint,
    load_spec,
    validate_record,
    write_csv,
    write_json,
)
from export_flowstar_segment import export_segment

HERE = Path(__file__).resolve().parent


def generate(output: Path) -> dict[str, Any]:
    spec = load_spec()
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("flowstar_stock", 53),
        ("flowstar_stock", 256),
        ("flowstar_full_picard_revalidated", 53),
        ("flowstar_root_cause_patch", 53),
    ]
    exact = analytic_endpoint(
        "riccati", spec["systems"]["riccati"]["initial_box"], 0.01
    )
    if exact is None:
        raise RuntimeError("Riccati analytic endpoint is unavailable")
    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for variant, precision_bits in cases:
        tag = f"{variant}_p{precision_bits}"
        record = export_segment(
            spec,
            system_name="riccati",
            h=0.01,
            order=2,
            variant=variant,
            precision_bits=precision_bits,
            work_dir=output / "logs" / tag,
        )
        checks = validate_record(record)
        endpoint = record["raw_endpoint_box"][0]
        tube = record["whole_tube_box"][0]
        contained = endpoint[0] <= exact[0][0] and endpoint[1] >= exact[0][1]
        summary_rows.append(
            {
                "variant": variant,
                "precision_bits": precision_bits,
                "h": 0.01,
                "order": 2,
                "candidate_remainder_radius": spec["flowstar"][
                    "candidate_remainder"
                ]["riccati"],
                "exact_lower": exact[0][0],
                "exact_upper": exact[0][1],
                "endpoint_lower": endpoint[0],
                "endpoint_upper": endpoint[1],
                "upper_margin_vs_exact": endpoint[1] - exact[0][1],
                "tube_lower": tube[0],
                "tube_upper": tube[1],
                "analytic_contained": contained,
                "endpoint_contained_in_tube": (
                    tube[0] <= endpoint[0] and tube[1] >= endpoint[1]
                ),
                "native_validation_passed": record[
                    "native_validation_passed"
                ],
                "cir_validation_passed": checks["passed"],
            }
        )
        for index, row in enumerate(record["validation_trace"]):
            trace_rows.append(
                {
                    "variant": variant,
                    "precision_bits": precision_bits,
                    "trace_index": index,
                    **row,
                }
            )
        write_json(output / f"{tag}_segment.json", record)
        records.append(record)
    write_csv(output / "flowstar_riccati_summary.csv", summary_rows)
    write_csv(output / "flowstar_riccati_stage_trace.csv", trace_rows)
    evidence = {
        "system": "x'=x^2",
        "initial_interval": [0.0, 0.1],
        "h": 0.01,
        "order": 2,
        "exact_endpoint": exact,
        "rows": summary_rows,
        "stock_precision_upper_difference": (
            summary_rows[1]["endpoint_upper"]
            - summary_rows[0]["endpoint_upper"]
        ),
        "all_corrected_variants_contain_analytic_endpoint": all(
            bool(row["analytic_contained"])
            for row in summary_rows
            if row["variant"] != "flowstar_stock"
        ),
        "stock_miss_persists_at_256_bits": (
            not summary_rows[0]["analytic_contained"]
            and not summary_rows[1]["analytic_contained"]
            and math.isclose(
                summary_rows[0]["endpoint_upper"],
                summary_rows[1]["endpoint_upper"],
                rel_tol=0.0,
                abs_tol=5e-15,
            )
        ),
    }
    write_json(output / "flowstar_riccati_evidence.json", evidence)
    write_json(output / "flowstar_riccati_records.json", records)
    if not evidence["all_corrected_variants_contain_analytic_endpoint"]:
        raise SystemExit("a corrected Flow* variant missed the analytic endpoint")
    if not evidence["stock_miss_persists_at_256_bits"]:
        raise SystemExit("the high-precision stock reproduction changed")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    generate(Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
