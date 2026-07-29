#!/usr/bin/env python3
"""Run component attribution and the common-engine matched-basis experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from common import classify_exponent, load_spec, write_csv, write_json

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _scalar(value: Any, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _tm_interval(value: Any) -> list[float]:
    return [
        float(value.lo.detach().cpu()),
        float(value.hi.detach().cpu()),
    ]


def run_matched_basis(
    spec: Mapping[str, Any], output: Path, *, smoke: bool
) -> dict[str, Any]:
    src = REPO_ROOT / "src"
    followup = HERE.parent / "first_order_followup"
    for candidate in (src, followup):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    import torch

    from export_torch_segment import rhs_from_spec
    from torch_basis import finite_basis_step_from_tm, normalized_initial_tm

    torch.set_default_dtype(torch.float64)
    system_name = "coupled_quadratic"
    system = spec["systems"][system_name]
    hs = [0.005] if smoke else list(map(float, spec["one_step"][system_name]))
    summary_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    discarded_rows: list[dict[str, Any]] = []
    for h in hs:
        for basis in ("B1", "B_DR", "B2", "B3"):
            initial = normalized_initial_tm(system["initial_box"], order=3)
            diagnostics: list[dict[str, Any]] = []
            started = time.perf_counter()
            segment, discarded = finite_basis_step_from_tm(
                rhs_from_spec(system),
                initial,
                h,
                basis,
                picard_iterations=2,
                arithmetic_order=3,
                diagnostics=diagnostics,
            )
            elapsed = time.perf_counter() - started
            if segment.status != "validated":
                raise RuntimeError(
                    f"matched basis {basis} h={h:g} failed: {segment.message}"
                )
            endpoint = segment.final_tm
            for state_index, model in enumerate(endpoint):
                polynomial = _tm_interval(
                    model.polynomial.evaluate_interval(model.domain)
                )
                remainder = _tm_interval(model.remainder)
                total = _tm_interval(model.range_box())
                terms_by_family: dict[str, int] = defaultdict(int)
                for exponent, coefficient in sorted(
                    model.polynomial.terms.items()
                ):
                    family = classify_exponent(exponent, None)
                    terms_by_family[family] += 1
                    support_rows.append(
                        {
                            "basis": basis,
                            "system": system_name,
                            "h": h,
                            "state_index": state_index,
                            "exponent": json.dumps(list(exponent)),
                            "coefficient": float(coefficient.detach().cpu()),
                            "family": family,
                            "retained": True,
                            "support_scope": "raw_endpoint",
                        }
                    )
                state_discarded = [
                    row for row in discarded if row.state_index == state_index
                ]
                summary_rows.append(
                    {
                        "tool": "torch_common_engine",
                        "variant": basis,
                        "protocol": "matched_basis_common_engine",
                        "system": system_name,
                        "h": h,
                        "state_index": state_index,
                        "basis": basis,
                        "arithmetic_order": 3,
                        "picard_iterations": 2,
                        "validator": "torch_growth_picard_common",
                        "range_backend": "torch_float64_nextafter_interval",
                        "reset": "none_one_step",
                        "dtype": "float64",
                        "lower": total[0],
                        "upper": total[1],
                        "width": total[1] - total[0],
                        "polynomial_width": polynomial[1] - polynomial[0],
                        "independent_remainder_width": (
                            remainder[1] - remainder[0]
                        ),
                        "discarded_term_count": len(state_discarded),
                        "discarded_range_width_sum": sum(
                            row.range_width for row in state_discarded
                        ),
                        "monomial_families": json.dumps(
                            dict(sorted(terms_by_family.items())),
                            sort_keys=True,
                        ),
                        "validation_attempts": segment.validation_attempts,
                        "native_validation_passed": True,
                        "runtime_s": elapsed,
                    }
                )
            local_families: set[str] = set()
            for state_index, model in enumerate(segment.tm):
                for exponent, coefficient in sorted(
                    model.polynomial.terms.items()
                ):
                    family = classify_exponent(
                        exponent, segment.tau_index
                    )
                    local_families.add(family)
                    support_rows.append(
                        {
                            "basis": basis,
                            "system": system_name,
                            "h": h,
                            "state_index": state_index,
                            "exponent": json.dumps(list(exponent)),
                            "coefficient": float(coefficient.detach().cpu()),
                            "family": family,
                            "retained": True,
                            "support_scope": "validated_local_segment",
                        }
                    )
            if basis == "B3" and "time_state_higher" not in local_families:
                raise RuntimeError(
                    "coupled quadratic failed to activate a time-lifted "
                    "quadratic state cross term in B3"
                )
            for record in discarded:
                discarded_rows.append(
                    {
                        "basis": basis,
                        "system": system_name,
                        "h": h,
                        "stage": record.stage,
                        "iteration": record.iteration,
                        "state_index": record.state_index,
                        "exponent": json.dumps(list(record.exponent)),
                        "coefficient": record.coefficient,
                        "range_lower": record.range_lower,
                        "range_upper": record.range_upper,
                        "range_width": record.range_width,
                        "destination": "fresh_independent_interval_remainder",
                    }
                )
    write_csv(output / "matched_basis_summary.csv", summary_rows)
    write_csv(output / "matched_basis_support.csv", support_rows)
    write_csv(output / "matched_basis_discarded_terms.csv", discarded_rows)
    capability_rows = [
        {
            "tool": tool,
            "basis": basis,
            "status": status,
            "mapping": mapping,
            "reason": reason,
        }
        for tool, entries in {
            "torch_common_engine": {
                "B1": (
                    "supported_experiment_adapter",
                    "sound finite-dictionary projection",
                    "",
                ),
                "B_DR": (
                    "supported_experiment_adapter",
                    "sound finite-dictionary projection",
                    "",
                ),
                "B2": (
                    "supported_experiment_adapter",
                    "sound finite-dictionary projection",
                    "",
                ),
                "B3": (
                    "supported_experiment_adapter",
                    "sound quadratic-dependency/time-lift projection",
                    "",
                ),
            },
            "diffreach": {
                "B1": (
                    "supported_native",
                    "TRUNCATE_TO_AFFINE=true",
                    "",
                ),
                "B_DR": (
                    "supported_native",
                    "c/L/Lt restricted quasi-quadratic dictionary",
                    "",
                ),
                "B2": (
                    "capability_gap",
                    "unavailable",
                    "no complete total-degree-2 native dictionary",
                ),
                "B3": (
                    "capability_gap",
                    "unavailable",
                    "no quadratic state-cross dictionary with tau lift",
                ),
            },
            "flowstar": {
                "B1": (
                    "capability_gap",
                    "unavailable",
                    "minimum legal fixed order is 2; no exact B1 selector",
                ),
                "B_DR": (
                    "capability_gap",
                    "unavailable",
                    "no exact restricted c/L/Lt dictionary selector",
                ),
                "B2": (
                    "supported_native",
                    "fixed complete order 2",
                    "",
                ),
                "B3": (
                    "capability_gap",
                    "unavailable",
                    "order 3 is a strict cubic superset, not exact B3",
                ),
            },
        }.items()
        for basis, (status, mapping, reason) in entries.items()
    ]
    write_csv(output / "matched_basis_capabilities.csv", capability_rows)
    return {
        "summaries": len(summary_rows),
        "retained_terms": len(support_rows),
        "discarded_terms": len(discarded_rows),
        "capability_rows": len(capability_rows),
    }


def run_flowstar_ablation(
    spec: Mapping[str, Any], output: Path, *, smoke: bool
) -> dict[str, Any]:
    from run_native import _load_flowstar_repair

    runner = _load_flowstar_repair()
    candidates = [1e-4] if smoke else [1e-6, 1e-4, 1e-2]
    modes = [
        ("refinement_disabled", True, False, True),
        ("stock_cached_refinement", False, False, False),
        ("full_picard_revalidated", False, True, False),
        ("root_cause_leaf_cache_patch", False, False, True),
    ]
    rows: list[dict[str, Any]] = []
    old_leaf = os.environ.get("FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION")
    try:
        for candidate in candidates:
            for label, no_refinement, full_revalidate, leaf_patch in modes:
                os.environ["FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION"] = (
                    "1" if leaf_patch else "0"
                )
                case_rows, run, traces = runner.run_fixed_case(
                    spec,
                    output,
                    system_name="riccati",
                    protocol=runner.PROTOCOL_NATIVE,
                    h=0.01,
                    horizon=0.01,
                    order=2,
                    candidate=candidate,
                    cutoff=float(spec["flowstar"]["cutoff"]),
                    variant=label,
                    no_refinement=no_refinement,
                    revalidate_refinement=full_revalidate,
                    sensitivity_label=f"candidate_{candidate:g}",
                )
                endpoint = [
                    row
                    for row in case_rows
                    if row.get("interval_kind") == "endpoint_raw"
                ]
                widths = [_scalar(row.get("width")) for row in endpoint]
                rows.append(
                    {
                        **run,
                        "ablation_family": "flowstar_refinement_candidate",
                        "leaf_cache_patch": leaf_patch,
                        "endpoint_max_width": max(widths, default=math.nan),
                        "polynomial_max_width": max(
                            (
                                _scalar(row.get("polynomial_width"))
                                for row in endpoint
                            ),
                            default=math.nan,
                        ),
                        "remainder_max_width": max(
                            (
                                _scalar(row.get("remainder_width"))
                                for row in endpoint
                            ),
                            default=math.nan,
                        ),
                        "analytic_reference_violations": sum(
                            row.get("analytic_reference_status") == "failed"
                            for row in endpoint
                        ),
                        "trace_events": len(traces),
                    }
                )
    finally:
        if old_leaf is None:
            os.environ.pop("FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION", None)
        else:
            os.environ["FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION"] = old_leaf
    write_csv(output / "flowstar_component_ablation.csv", rows)
    return {"rows": len(rows)}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _last_endpoint_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        kind = str(row.get("interval_kind", ""))
        if kind not in {
            "endpoint_raw",
            "endpoint_tightened_supplemental",
            "tube",
        }:
            continue
        key = tuple(
            str(row.get(field, ""))
            for field in (
                "tool",
                "variant",
                "protocol",
                "system",
                "h",
                "state_index",
                "interval_kind",
            )
        )
        grouped[key].append(row)
    selected: list[dict[str, Any]] = []
    for values in grouped.values():
        maximum = max(_scalar(row.get("step_index"), 0.0) for row in values)
        selected.extend(
            dict(row)
            for row in values
            if _scalar(row.get("step_index"), 0.0) == maximum
        )
    return selected


def collect_components(output: Path) -> dict[str, Any]:
    raw: list[dict[str, Any]] = []
    for name in (
        "controlled_torch.csv",
        "controlled_diffreach.csv",
        "controlled_flowstar.csv",
        "native_torch.csv",
        "native_diffreach.csv",
        "native_flowstar.csv",
    ):
        raw.extend(_read_csv(output / name))
    rows: list[dict[str, Any]] = []
    for row in _last_endpoint_rows(raw):
        total = _scalar(row.get("width"))
        polynomial = _scalar(row.get("polynomial_width"), 0.0)
        remainder = _scalar(row.get("remainder_width"), 0.0)
        if not math.isfinite(polynomial):
            polynomial = 0.0
        if not math.isfinite(remainder):
            remainder = 0.0
        rows.append(
            {
                "tool": row.get("tool", ""),
                "variant": row.get("variant", ""),
                "protocol": row.get("protocol", ""),
                "system": row.get("system", ""),
                "h": row.get("h", ""),
                "horizon": row.get("horizon", row.get("time", "")),
                "state_index": row.get("state_index", ""),
                "interval_kind": row.get("interval_kind", ""),
                "basis": row.get("basis", row.get("local_basis", "")),
                "carry_or_reset": row.get(
                    "carry",
                    row.get(
                        "carry_contract",
                        row.get("carried_representation", ""),
                    ),
                ),
                "total_width": total,
                "polynomial_range_width": polynomial,
                "truncation_overflow_width": remainder,
                "independent_interval_remainder_width": remainder,
                "structured_symbolic_remainder_width": (
                    "" if row.get("tool") == "diffreach" else 0.0
                ),
                "unattributed_dependency_or_reset_width": max(
                    0.0, total - polynomial - remainder
                )
                if math.isfinite(total)
                else "",
                "monomial_families": row.get("monomial_families", ""),
                "runtime_s": row.get(
                    "runtime_s", row.get("steady_runtime_s", "")
                ),
                "native_validation_passed": row.get(
                    "native_validation_passed",
                    row.get("native_validation_status", ""),
                ),
            }
        )
    rows.extend(_read_csv(output / "flowstar_component_ablation.csv"))
    write_csv(output / "component_ablation.csv", rows)
    summary = {
        "rows": len(rows),
        "tools": sorted({str(row.get("tool", "")) for row in rows}),
        "note": (
            "A nonzero unattributed field is interval dependency/reset "
            "inflation, not an additional native remainder object."
        ),
    }
    write_json(output / "component_ablation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["matched", "flowstar", "collect", "all"],
        required=True,
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    if args.mode in {"matched", "all"}:
        result["matched"] = run_matched_basis(
            spec, output, smoke=args.smoke
        )
    if args.mode in {"flowstar", "all"}:
        result["flowstar"] = run_flowstar_ablation(
            spec, output, smoke=args.smoke
        )
    if args.mode in {"collect", "all"}:
        result["collect"] = collect_components(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
