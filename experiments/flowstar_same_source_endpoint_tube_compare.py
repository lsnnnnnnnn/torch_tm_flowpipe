#!/usr/bin/env python3
"""Compare same-source Flow*/PyTorch full-step tubes and tau=h endpoints.

This is diagnostic-only. It reads committed/generated trace artifacts and does
not change solver behavior, rerun h10, add queue variants, or claim Flow* parity.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = ROOT / "outputs" / "flowstar_step_trace_compare"
DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_same_source_endpoint_tube_compare"

LEDGER_FIELDS = [
    "comparison_kind",
    "flowstar_source_object",
    "torch_source_object",
    "source_objects_match",
    "domain_semantics_match",
    "semantic_comparison_valid",
    "flowstar_x_lo",
    "flowstar_x_hi",
    "flowstar_y_lo",
    "flowstar_y_hi",
    "torch_noqueue_x_lo",
    "torch_noqueue_x_hi",
    "torch_noqueue_y_lo",
    "torch_noqueue_y_hi",
    "torch_v2_x_lo",
    "torch_v2_x_hi",
    "torch_v2_y_lo",
    "torch_v2_y_hi",
    "x_lo_delta",
    "x_hi_delta",
    "y_lo_delta",
    "y_hi_delta",
    "flowstar_width_sum",
    "torch_noqueue_width_sum",
    "torch_v2_width_sum",
    "width_ratio_noqueue_over_flowstar",
    "width_ratio_v2_over_flowstar",
    "includes_target_remainder_match",
    "includes_ordinary_remainder_match",
    "includes_cutoff_poly_diff_match",
    "includes_symbolic_output_width_match",
    "verdict",
    "notes",
]

COMPARISONS = {
    "full_step_tube": {
        "flow_prefix": "flowstar_full_step_tube",
        "torch_prefix": "torch_full_step_validation_candidate",
    },
    "tau_h_endpoint": {
        "flow_prefix": "flowstar_tau_h_endpoint",
        "torch_prefix": "torch_tau_h_endpoint",
    },
}

FLAG_SUFFIXES = (
    "includes_target_remainder",
    "includes_ordinary_remainder",
    "includes_cutoff_poly_diff",
    "includes_symbolic_output_width",
)

BOUNDS = (
    ("x", "lo"),
    ("x", "hi"),
    ("y", "lo"),
    ("y", "hi"),
)

TOLERANCE = 1e-12


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field, "")) for field in LEDGER_FIELDS})


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.17g}"
    return str(value)


def finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _first_present(row: Mapping[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return ""


def _h_try(row: Mapping[str, Any]) -> float | None:
    return finite_float(_first_present(row, "h_try", "h_forced", "h"))


def _find_attempt(rows: Iterable[Mapping[str, Any]], *, t: float, h: float, tolerance: float = 1e-9) -> Mapping[str, Any] | None:
    for row in rows:
        t_before = finite_float(row.get("t_before"))
        h_try = _h_try(row)
        if t_before is None or h_try is None:
            continue
        if abs(t_before - t) <= tolerance and abs(h_try - h) <= tolerance:
            return row
    return None


def _bound(row: Mapping[str, Any], prefix: str, dim: str, side: str) -> Any:
    return _first_present(row, f"{prefix}_{dim}_{side}", f"{prefix}_{side}_{dim}")


def _width_sum(row: Mapping[str, Any], prefix: str) -> float | None:
    parts: list[float] = []
    for dim in ("x", "y"):
        lo = finite_float(_bound(row, prefix, dim, "lo"))
        hi = finite_float(_bound(row, prefix, dim, "hi"))
        if lo is None or hi is None:
            return None
        parts.append(hi - lo)
    return sum(parts)


def _delta(candidate: Any, reference: Any) -> float | None:
    cand = finite_float(candidate)
    ref = finite_float(reference)
    if cand is None or ref is None:
        return None
    return cand - ref


def _ratio(candidate: Any, reference: Any) -> float | None:
    cand = finite_float(candidate)
    ref = finite_float(reference)
    if cand is None or ref is None or abs(ref) <= 0.0:
        return None
    return cand / ref


def _known(value: Any) -> bool:
    return value not in (None, "", "unknown")


def _match_values(values: Iterable[Any]) -> str:
    vals = [str(value) for value in values]
    if any(not _known(value) for value in vals):
        return "unknown"
    return "true" if len(set(vals)) == 1 else "false"


def _match_bool(*matches: str) -> str:
    if any(match == "false" for match in matches):
        return "false"
    if any(match == "unknown" for match in matches):
        return "unknown"
    return "true"


def _missing_fields(kind: str, flow: Mapping[str, Any], noqueue: Mapping[str, Any], v2: Mapping[str, Any]) -> list[str]:
    spec = COMPARISONS[kind]
    flow_prefix = spec["flow_prefix"]
    torch_prefix = spec["torch_prefix"]
    checks: list[tuple[str, Mapping[str, Any], str, Any]] = []
    for source, row, prefix in (
        ("flowstar", flow, flow_prefix),
        ("torch_noqueue", noqueue, torch_prefix),
        ("torch_v2", v2, torch_prefix),
    ):
        checks.append((source, row, f"{prefix}_source_object", row.get(f"{prefix}_source_object")))
        checks.append((source, row, f"{prefix}_domain_semantics", row.get(f"{prefix}_domain_semantics")))
        for suffix in FLAG_SUFFIXES:
            checks.append((source, row, f"{prefix}_{suffix}", row.get(f"{prefix}_{suffix}")))
        for dim, side in BOUNDS:
            checks.append((source, row, f"{prefix}_{dim}_{side}", _bound(row, prefix, dim, side)))
    return [f"{source}.{field}" for source, _row, field, value in checks if not _known(value)]


def _first_box_divergence(row: Mapping[str, Any], *, tolerance: float = TOLERANCE) -> str:
    for field in ("x_lo_delta", "x_hi_delta", "y_lo_delta", "y_hi_delta"):
        value = finite_float(row.get(field))
        if value is not None and abs(value) > tolerance:
            return field.removesuffix("_delta")
    return ""


def _v2_divergence(flow_values: Mapping[str, Any], v2_values: Mapping[str, Any], *, tolerance: float = TOLERANCE) -> str:
    for dim, side in BOUNDS:
        delta = _delta(v2_values.get(f"{dim}_{side}"), flow_values.get(f"{dim}_{side}"))
        if delta is not None and abs(delta) > tolerance:
            return f"{dim}_{side}"
    return ""


def _ledger_row(kind: str, flow: Mapping[str, Any], noqueue: Mapping[str, Any], v2: Mapping[str, Any]) -> dict[str, Any]:
    spec = COMPARISONS[kind]
    flow_prefix = spec["flow_prefix"]
    torch_prefix = spec["torch_prefix"]

    flow_values = {f"{dim}_{side}": _bound(flow, flow_prefix, dim, side) for dim, side in BOUNDS}
    noqueue_values = {f"{dim}_{side}": _bound(noqueue, torch_prefix, dim, side) for dim, side in BOUNDS}
    v2_values = {f"{dim}_{side}": _bound(v2, torch_prefix, dim, side) for dim, side in BOUNDS}

    source_match = _match_values(
        [
            flow.get(f"{flow_prefix}_source_object"),
            noqueue.get(f"{torch_prefix}_source_object"),
            v2.get(f"{torch_prefix}_source_object"),
        ]
    )
    domain_match = _match_values(
        [
            flow.get(f"{flow_prefix}_domain_semantics"),
            noqueue.get(f"{torch_prefix}_domain_semantics"),
            v2.get(f"{torch_prefix}_domain_semantics"),
        ]
    )
    flag_matches = {
        suffix: _match_values(
            [
                flow.get(f"{flow_prefix}_{suffix}"),
                noqueue.get(f"{torch_prefix}_{suffix}"),
                v2.get(f"{torch_prefix}_{suffix}"),
            ]
        )
        for suffix in FLAG_SUFFIXES
    }
    missing = _missing_fields(kind, flow, noqueue, v2)
    semantic_valid = _match_bool(source_match, domain_match, *flag_matches.values())
    if missing:
        semantic_valid = "unknown" if semantic_valid == "true" else semantic_valid

    flow_width = _width_sum(flow, flow_prefix)
    noqueue_width = _width_sum(noqueue, torch_prefix)
    v2_width = _width_sum(v2, torch_prefix)
    row: dict[str, Any] = {
        "comparison_kind": kind,
        "flowstar_source_object": flow.get(f"{flow_prefix}_source_object", ""),
        "torch_source_object": noqueue.get(f"{torch_prefix}_source_object", ""),
        "source_objects_match": source_match,
        "domain_semantics_match": domain_match,
        "semantic_comparison_valid": semantic_valid,
        "flowstar_x_lo": flow_values["x_lo"],
        "flowstar_x_hi": flow_values["x_hi"],
        "flowstar_y_lo": flow_values["y_lo"],
        "flowstar_y_hi": flow_values["y_hi"],
        "torch_noqueue_x_lo": noqueue_values["x_lo"],
        "torch_noqueue_x_hi": noqueue_values["x_hi"],
        "torch_noqueue_y_lo": noqueue_values["y_lo"],
        "torch_noqueue_y_hi": noqueue_values["y_hi"],
        "torch_v2_x_lo": v2_values["x_lo"],
        "torch_v2_x_hi": v2_values["x_hi"],
        "torch_v2_y_lo": v2_values["y_lo"],
        "torch_v2_y_hi": v2_values["y_hi"],
        "x_lo_delta": _delta(noqueue_values["x_lo"], flow_values["x_lo"]),
        "x_hi_delta": _delta(noqueue_values["x_hi"], flow_values["x_hi"]),
        "y_lo_delta": _delta(noqueue_values["y_lo"], flow_values["y_lo"]),
        "y_hi_delta": _delta(noqueue_values["y_hi"], flow_values["y_hi"]),
        "flowstar_width_sum": flow_width,
        "torch_noqueue_width_sum": noqueue_width,
        "torch_v2_width_sum": v2_width,
        "width_ratio_noqueue_over_flowstar": _ratio(noqueue_width, flow_width),
        "width_ratio_v2_over_flowstar": _ratio(v2_width, flow_width),
        "includes_target_remainder_match": flag_matches["includes_target_remainder"],
        "includes_ordinary_remainder_match": flag_matches["includes_ordinary_remainder"],
        "includes_cutoff_poly_diff_match": flag_matches["includes_cutoff_poly_diff"],
        "includes_symbolic_output_width_match": flag_matches["includes_symbolic_output_width"],
    }

    noqueue_divergence = _first_box_divergence(row)
    v2_divergence = _v2_divergence(flow_values, v2_values)
    notes: list[str] = []
    if missing:
        notes.append("missing fields: " + ";".join(missing))
    if v2_divergence and v2_divergence != noqueue_divergence:
        notes.append(f"torch_v2 first divergent endpoint: {v2_divergence}")

    if semantic_valid == "unknown":
        verdict = "unknown_missing_fields"
    elif semantic_valid == "false":
        verdict = "semantic_mismatch"
    elif noqueue_divergence:
        verdict = f"same_source_{noqueue_divergence}_divergence"
    elif v2_divergence:
        verdict = f"same_source_torch_v2_{v2_divergence}_divergence"
    else:
        verdict = "same_source_boxes_match"
    row["verdict"] = verdict
    row["notes"] = "; ".join(notes)
    return row


def _previous_endpoint_y_hi_delta(flow: Mapping[str, Any], row: Mapping[str, Any]) -> float | None:
    return _delta(
        _bound(row, "endpoint_box_before_center", "y", "hi"),
        _bound(flow, "endpoint_box_before_center", "y", "hi"),
    )


def summarize(rows: list[dict[str, Any]], flow: Mapping[str, Any], noqueue: Mapping[str, Any], v2: Mapping[str, Any]) -> dict[str, Any]:
    by_kind = {row["comparison_kind"]: row for row in rows}
    first = "none"
    for kind in ("full_step_tube", "tau_h_endpoint"):
        verdict = str(by_kind[kind].get("verdict", ""))
        if verdict not in {"same_source_boxes_match", ""}:
            first = f"{kind}:{verdict}"
            break

    valid_values = {kind: by_kind[kind].get("semantic_comparison_valid", "unknown") for kind in by_kind}
    close_values = {}
    for kind, row in by_kind.items():
        verdict = str(row.get("verdict", ""))
        if row.get("semantic_comparison_valid") != "true":
            close_values[kind] = "unknown"
        elif verdict == "same_source_boxes_match":
            close_values[kind] = "true"
        else:
            close_values[kind] = "false"

    same_source_y_deltas = [
        finite_float(row.get("y_hi_delta"))
        for row in rows
        if row.get("semantic_comparison_valid") == "true"
    ]
    same_source_y_deltas = [value for value in same_source_y_deltas if value is not None]
    old_deltas = [
        _previous_endpoint_y_hi_delta(flow, noqueue),
        _previous_endpoint_y_hi_delta(flow, v2),
    ]
    old_deltas = [value for value in old_deltas if value is not None]
    if not old_deltas or not same_source_y_deltas:
        previous_gap_remains = "unknown"
    elif max(abs(value) for value in same_source_y_deltas) <= TOLERANCE:
        previous_gap_remains = "false"
    else:
        previous_gap_remains = "true"

    missing: list[str] = []
    for row in rows:
        notes = str(row.get("notes", ""))
        if notes.startswith("missing fields: "):
            missing.extend(notes[len("missing fields: "):].split(";"))
        elif "missing fields: " in notes:
            missing.extend(notes.split("missing fields: ", 1)[1].split(";", 1)[0].split(";"))
    missing = sorted({field for field in missing if field})

    if any(row.get("semantic_comparison_valid") == "false" for row in rows):
        likely_component = "source/domain/inclusion semantics mismatch"
    elif any(row.get("semantic_comparison_valid") == "unknown" for row in rows):
        likely_component = "unknown: missing same-source trace fields"
    elif by_kind["full_step_tube"].get("verdict") != "same_source_boxes_match":
        likely_component = "full-step validation candidate tube construction or range evaluation"
    elif by_kind["tau_h_endpoint"].get("verdict") != "same_source_boxes_match":
        likely_component = "tau=h substitution/drop endpoint evaluation"
    else:
        likely_component = "previous endpoint gap was source-stage mismatch; same-source boxes match within tolerance"

    return {
        "full_step_semantic_valid": valid_values.get("full_step_tube", "unknown"),
        "tau_h_semantic_valid": valid_values.get("tau_h_endpoint", "unknown"),
        "full_step_close": close_values.get("full_step_tube", "unknown"),
        "tau_h_close": close_values.get("tau_h_endpoint", "unknown"),
        "first_same_source_divergence": first,
        "previous_y_hi_gap_remains_under_same_source": previous_gap_remains,
        "likely_component_if_mismatch_remains": likely_component,
        "missing_fields": missing,
        "previous_endpoint_y_hi_delta_noqueue": old_deltas[0] if old_deltas else None,
    }


def build_same_source_ledger(
    flowstar_rows: list[Mapping[str, Any]],
    noqueue_rows: list[Mapping[str, Any]],
    v2_rows: list[Mapping[str, Any]],
    *,
    t: float,
    h: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    flow = _find_attempt(flowstar_rows, t=t, h=h)
    noqueue = _find_attempt(noqueue_rows, t=t, h=h)
    v2 = _find_attempt(v2_rows, t=t, h=h)
    missing = [name for name, row in (("flowstar", flow), ("torch_noqueue", noqueue), ("torch_v2", v2)) if row is None]
    if missing:
        raise ValueError(f"missing same-t/h trace rows for: {', '.join(missing)}")
    assert flow is not None and noqueue is not None and v2 is not None
    rows = [_ledger_row(kind, flow, noqueue, v2) for kind in ("full_step_tube", "tau_h_endpoint")]
    return rows, summarize(rows, flow, noqueue, v2)


def _fmt_box(row: Mapping[str, Any], source: str) -> str:
    return (
        f"x=[{_format(row.get(f'{source}_x_lo'))}, {_format(row.get(f'{source}_x_hi'))}], "
        f"y=[{_format(row.get(f'{source}_y_lo'))}, {_format(row.get(f'{source}_y_hi'))}]"
    )


def _report(out_dir: Path, rows: list[dict[str, Any]], summary: Mapping[str, Any], *, t: float, h: float) -> str:
    full = next(row for row in rows if row.get("comparison_kind") == "full_step_tube")
    tau = next(row for row in rows if row.get("comparison_kind") == "tau_h_endpoint")
    missing_fields = summary.get("missing_fields") or []
    missing_text = "; ".join(str(field) for field in missing_fields) if missing_fields else "none"
    lines = [
        "# Flow* Same-Source Endpoint/Tube Comparison",
        "",
        "This is diagnostic-only and makes no solver change. It does not rerun h10, add queue variants, or claim Flow* parity.",
        "",
        "## Scope",
        "",
        f"- t_before requested: `{t:.17g}`",
        f"- h_try: `{h:.17g}`",
        "- Input traces: `outputs/flowstar_step_trace_compare/*.csv`",
        "- Output ledger: `outputs/flowstar_same_source_endpoint_tube_compare/same_source_endpoint_tube_ledger.csv`",
        "",
        "## Answers",
        "",
        f"- Full-step tube comparison semantically valid: `{summary.get('full_step_semantic_valid', 'unknown')}`.",
        f"- Tau=h endpoint comparison semantically valid: `{summary.get('tau_h_semantic_valid', 'unknown')}`.",
        f"- Which same-source object differs first: `{summary.get('first_same_source_divergence', 'unknown')}`.",
        f"- Is the tau=h endpoint close: `{summary.get('tau_h_close', 'unknown')}`.",
        f"- Is the full-step tube close: `{summary.get('full_step_close', 'unknown')}`.",
        f"- Does the previous y_hi gap remain once source semantics match: `{summary.get('previous_y_hi_gap_remains_under_same_source', 'unknown')}`.",
        f"- Likely component if mismatch remains: {summary.get('likely_component_if_mismatch_remains', 'unknown')}.",
        f"- Missing fields: {missing_text}.",
        "",
        "## Boxes",
        "",
        "| comparison | semantic valid | verdict | Flow* box | torch no_queue box | torch v2 box | y_hi delta no_queue-flowstar |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in (full, tau):
        lines.append(
            "| {kind} | {valid} | {verdict} | {flow} | {noq} | {v2} | {delta} |".format(
                kind=row.get("comparison_kind", ""),
                valid=row.get("semantic_comparison_valid", ""),
                verdict=row.get("verdict", ""),
                flow=_fmt_box(row, "flowstar"),
                noq=_fmt_box(row, "torch_noqueue"),
                v2=_fmt_box(row, "torch_v2"),
                delta=_format(row.get("y_hi_delta", "")),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "Blank endpoint fields are reported as unknown and are not treated as zero.",
            "The older endpoint-before-center fields remain a separate source-stage diagnostic; this report compares the explicitly labeled full-step tube and tau=h endpoint objects.",
        ]
    )
    text = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "same_source_endpoint_tube_report.md").write_text(text, encoding="utf-8")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--t", type=float, default=0.0)
    parser.add_argument("--h", type=float, default=0.025)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_dir = args.trace_dir.resolve()
    out_dir = args.out_dir.resolve()
    flowstar_rows = _read_rows(trace_dir / "flowstar_trace.csv")
    noqueue_rows = _read_rows(trace_dir / "torch_noqueue_trace.csv")
    v2_rows = _read_rows(trace_dir / "torch_v2_trace.csv")
    rows, summary = build_same_source_ledger(flowstar_rows, noqueue_rows, v2_rows, t=args.t, h=args.h)
    _write_rows(out_dir / "same_source_endpoint_tube_ledger.csv", rows)
    _report(out_dir, rows, summary, t=args.t, h=args.h)
    print(f"wrote same-source endpoint/tube comparison to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
