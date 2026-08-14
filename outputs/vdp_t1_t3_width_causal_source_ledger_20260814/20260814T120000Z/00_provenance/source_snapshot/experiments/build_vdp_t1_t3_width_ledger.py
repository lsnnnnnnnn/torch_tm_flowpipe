#!/usr/bin/env python3
"""Rebuild the fixed-schedule VDP width/source ledger from raw artifacts.

The output keeps endpoint, last-segment tube, and prefix semantics separate.
It never zero-fills unavailable internal fields: an unavailable field carries
an explicit availability label.  The fixed-schedule ledger is also kept
separate from the native adaptive terminal record at t=6.397083942944808.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = (
    ROOT
    / "outputs"
    / "flowstar_torch_source_carry_root_cause_20260813"
    / "20260813T030338Z"
    / "06_native_stage_traces"
)
DEFAULT_EQUIVALENCE = (
    ROOT
    / "outputs"
    / "flowstar_torch_causal_mechanism_closure_20260813"
    / "20260813T060020Z"
    / "05_copied_probe_equivalence"
    / "three_way_audit"
    / "artifacts"
    / "audit"
    / "summary.json"
)
DEFAULT_NATIVE = (
    ROOT
    / "outputs"
    / "mainline_realignment_20260810"
    / "20260810T025910Z"
    / "01_native_baselines"
    / "torch_complete_o4_authoritative_t6p5"
)

CHANNELS = (
    "endpoint_x",
    "endpoint_y",
    "segment_tube_x",
    "segment_tube_y",
)
INITIAL_WIDTH = {"x": 0.3, "y": 0.1}
THRESHOLDS = (1.1, 1.5, 2.0, 5.0)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _float(row: Mapping[str, Any], name: str) -> float | None:
    value = row.get(name)
    if value in (None, ""):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name}: {value!r}")
    return result


def _json_field(row: Mapping[str, Any], name: str, default: Any = None) -> Any:
    value = row.get(name)
    if value in (None, ""):
        return default
    return json.loads(str(value))


def _interval(lo: float | None, hi: float | None, *, availability: str = "available") -> dict[str, Any]:
    if lo is None or hi is None:
        return {"lo": None, "hi": None, "width": None, "availability": availability}
    if lo > hi:
        raise ValueError(f"reversed interval [{lo}, {hi}]")
    return {"lo": lo, "hi": hi, "width": hi - lo, "availability": availability}


def _row_interval(row: Mapping[str, Any], lo: str, hi: str) -> dict[str, Any]:
    return _interval(_float(row, lo), _float(row, hi))


def _matrix_interval(value: Mapping[str, Any] | None, component: int) -> dict[str, Any]:
    if value is None:
        return _interval(None, None, availability="UNAVAILABLE_IN_RAW_ARTIFACT")
    lo = value.get("lo", [[None, None]])[0][component]
    hi = value.get("hi", [[None, None]])[0][component]
    return _interval(None if lo is None else float(lo), None if hi is None else float(hi))


def _attempt_interval(attempt: Mapping[str, Any], lo_name: str, hi_name: str, component: int) -> dict[str, Any]:
    lo = _json_field(attempt, lo_name)
    hi = _json_field(attempt, hi_name)
    if lo is None or hi is None:
        return _interval(None, None, availability="UNAVAILABLE_IN_RAW_ARTIFACT")
    return _interval(float(lo[0][component]), float(hi[0][component]))


def _flow_interval(row: Mapping[str, Any], stem: str, component: str) -> dict[str, Any]:
    return _row_interval(row, f"{stem}_{component}_lo", f"{stem}_{component}_hi")


def _torch_interval(row: Mapping[str, Any], stem: str, component: str) -> dict[str, Any]:
    return _row_interval(row, f"{stem}_{component}_lo", f"{stem}_{component}_hi")


def _source_lines() -> dict[str, Any]:
    return {
        "flowstar_boundary": {
            "file": "flowstar-toolbox/Continuous.cpp",
            "function": "Flowpipe::advance(..., Symbolic_Remainder&)",
            "line": 2123,
            "identity": "pinned Flow* b85a321",
        },
        "torch_boundary": {
            "file": "src/torch_tm_flowpipe/flowpipe.py",
            "function": "_flowstar_normalized_insertion_transition",
            "line": 1510,
        },
        "torch_picard_consumer": {
            "file": "src/torch_tm_flowpipe/batched_dense_tm.py",
            "function": "dense_picard_validate_step",
            "line": 3240,
        },
    }


def _flow_details(row: Mapping[str, Any], component: str, equivalence: Mapping[str, Any]) -> dict[str, Any]:
    component_index = 0 if component == "x" else 1
    queue_size = int(row.get("queue_size") or row.get("symbolic_J_size") or 0)
    fingerprint_payload = {
        "prestate": row.get("prestate_coefficients_canonical", ""),
        "queue_size": queue_size,
        "j": int(row.get("symbolic_J_size") or 0),
        "phi_l": int(row.get("symbolic_Phi_L_size") or 0),
    }
    target = _flow_interval(row, "target_remainder", component)
    refined = _flow_interval(row, "residual", component)
    margin = None
    if target["lo"] is not None and refined["lo"] is not None:
        margin = min(refined["lo"] - target["lo"], target["hi"] - refined["hi"])
    return {
        "prestate": {
            "box": _flow_interval(row, "pre_step_box", component),
            "center": _float(row, f"extracted_center_{component}"),
            "scale": _float(row, f"extracted_scale_{component}"),
        },
        "retained_polynomial_natural_range": _flow_interval(row, "polynomial_range", component),
        "remainders": {
            "polynomial_truncation": _flow_interval(row, "int_trunc_dropped_terms", component),
            "integration_overflow": _flow_interval(row, "raw_remainder_integration_remainder", component),
            "cutoff": _flow_interval(row, "cutoff_polynomial_difference", component),
            "ordinary": _flow_interval(row, "post_cutoff_residual", component),
            "right_map_parameterization": {
                "interval": _interval(
                    None,
                    None,
                    availability="WIDTH_ONLY_IN_AUTHORITATIVE_RAW_ARTIFACT",
                ),
                "width": _float(row, f"right_map_range_width_{component}"),
            },
        },
        "picard": {
            "raw_image": _flow_interval(row, "raw_ctrunc_residual", component),
            "refinement_image": refined,
            "target": target,
            "subset_margin": margin,
            "accepted": str(row.get("accepted", "")).lower() == "true",
        },
        "endpoint_tau_h_substitution_merge_range": _flow_interval(
            row, "flowstar_tau_h_endpoint", component
        ),
        "sources": {
            "live_source_count": queue_size,
            "source_fingerprint": _canonical_sha(fingerprint_payload),
            "structured_width_mass": _float(row, f"symbolic_propagated_width_{component}"),
            "ordinary_width_mass": refined["width"],
            "lineage_available": True,
        },
        "actual_source": _source_lines()["flowstar_boundary"],
        "observer_parity": {
            "status": equivalence["status"],
            "clean_instrumented_byte_exact": equivalence["clean_instrumented"]["byte_exact"],
            "actual_copied_compared_steps": equivalence["actual_copied"]["compared_steps"],
            "actual_copied_mismatches": equivalence["actual_copied"]["actual_pre_reset_retained_state_mismatches"],
        },
        "component_index": component_index,
    }


def _torch_details(
    segment: Mapping[str, Any],
    attempt: Mapping[str, Any],
    ledger: Mapping[str, Any],
    component: str,
) -> dict[str, Any]:
    component_index = 0 if component == "x" else 1
    validated = ledger.get("validated_remainder_ledger_intervals", {})
    centers = _json_field(segment, "prestate_center", [None, None])
    scales = _json_field(segment, "prestate_scale", [None, None])
    retained_width = _float(segment, f"carry_composed_poly_range_width_{component}")
    target_margin = _json_field(attempt, "subset_margin", [[None, None]])[0][component_index]
    return {
        "prestate": {
            "center": None if centers[component_index] is None else float(centers[component_index]),
            "scale": None if scales[component_index] is None else float(scales[component_index]),
            "fingerprint": segment.get("prestate_sha256"),
        },
        "retained_polynomial_natural_range": {
            "lo": None,
            "hi": None,
            "width": retained_width,
            "availability": "WIDTH_ONLY_IN_AUTHORITATIVE_RAW_ARTIFACT",
        },
        "remainders": {
            "polynomial_truncation": _matrix_interval(validated.get("polynomial_truncation"), component_index),
            "integration_overflow": _matrix_interval(validated.get("integration_overflow"), component_index),
            "cutoff": _matrix_interval(validated.get("cutoff"), component_index),
            "ordinary": _attempt_interval(attempt, "ordinary_residual_lo", "ordinary_residual_hi", component_index),
            "right_map_parameterization": {
                "interval": _interval(
                    None,
                    None,
                    availability="WIDTH_ONLY_IN_AUTHORITATIVE_RAW_ARTIFACT",
                ),
                "width": _float(segment, f"carry_output_remainder_width_{component}"),
            },
        },
        "picard": {
            "raw_image": _attempt_interval(
                attempt, "picard_image_remainder_lo", "picard_image_remainder_hi", component_index
            ),
            "refinement_image": _attempt_interval(
                attempt, "validated_remainder_decomposition_lo", "validated_remainder_decomposition_hi", component_index
            ),
            "subset_margin": None if target_margin is None else float(target_margin),
            "accepted": str(attempt.get("subset_result", "")).lower() == "true",
        },
        "endpoint_tau_h_substitution_merge_range": _torch_interval(segment, "endpoint", component),
        "sources": {
            "live_source_count": 0,
            "source_fingerprint": _canonical_sha({"legacy_prestate": segment.get("prestate_sha256")}),
            "structured_width_mass": 0.0,
            "ordinary_width_mass": _float(segment, f"carry_output_remainder_width_{component}"),
            "lineage_available": False,
            "reason": "legacy normalized insertion has no persistent source identities",
        },
        "actual_source": {
            "boundary": _source_lines()["torch_boundary"],
            "picard_consumer": _source_lines()["torch_picard_consumer"],
        },
        "observer_parity": {
            "status": "read_only trace fields are generated by the actual dense production call",
            "default_off": True,
        },
        "component_index": component_index,
    }


def _output_object(tool: str, row: Mapping[str, Any], object_name: str, component: str) -> dict[str, Any]:
    if tool == "flowstar":
        stem = "flowstar_tau_h_endpoint" if object_name == "endpoint" else "flowstar_full_step_tube"
        return _flow_interval(row, stem, component)
    stem = "endpoint" if object_name == "endpoint" else "segment"
    return _torch_interval(row, stem, component)


def _native_terminal(native_dir: Path) -> dict[str, Any]:
    segments = _read_csv(native_dir / "segments.csv")
    attempts = _read_csv(native_dir / "attempts.csv")
    summary = json.loads((native_dir / "summary.json").read_text(encoding="utf-8"))
    accepted = [row for row in segments if row.get("status") == "accepted"]
    rejected = [row for row in segments if row.get("status") == "rejected"]
    if len(accepted) != 307 or not rejected:
        raise ValueError("native terminal artifact no longer has 307 accepted plus rejection")
    last = accepted[-1]
    reject = rejected[-1]
    terminal_attempts = [row for row in attempts if abs(float(row["t_before"]) - 6.397083942944808) < 1e-15]
    return {
        "contract": "native_adaptive_complete_o4_separate_from_fixed_schedule_ratios",
        "accepted_segments": len(accepted),
        "highest_continuously_validated_time": float(last["t_hi"]),
        "last_accepted": last,
        "rejected_segment_record": reject,
        "rejected_attempts": terminal_attempts,
        "summary": summary,
        "source_files": {
            "segments": str(native_dir / "segments.csv"),
            "attempts": str(native_dir / "attempts.csv"),
            "summary": str(native_dir / "summary.json"),
        },
    }


def _polyline(points: Iterable[tuple[float, float]], *, width: int, height: int, bounds: tuple[float, float, float, float]) -> str:
    x0, x1, y0, y1 = bounds
    span_x = max(x1 - x0, 1e-300)
    span_y = max(y1 - y0, 1e-300)
    return " ".join(
        f"{20 + (x - x0) / span_x * (width - 40):.2f},{10 + (y1 - y) / span_y * (height - 30):.2f}"
        for x, y in points
    )


def _write_svg(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    panel_w, panel_h = 760, 230
    colors = {
        "flowstar_lo": "#1f77b4", "flowstar_hi": "#1f77b4", "flowstar_width": "#17becf",
        "torch_lo": "#d62728", "torch_hi": "#d62728", "torch_width": "#ff7f0e",
    }
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{panel_w}" height="{panel_h * 4}">',
             '<rect width="100%" height="100%" fill="white"/>']
    for panel, channel in enumerate(CHANNELS):
        object_name, component = channel.rsplit("_", 1)
        values: dict[str, list[tuple[float, float]]] = {name: [] for name in colors}
        for row in rows:
            time = float(row["time"])
            for tool in ("flowstar", "torch"):
                interval = row["outputs"][object_name][component][tool]
                values[f"{tool}_lo"].append((time, interval["lo"]))
                values[f"{tool}_hi"].append((time, interval["hi"]))
                values[f"{tool}_width"].append((time, interval["width"]))
        all_y = [value for series in values.values() for _, value in series]
        bounds = (rows[0]["time"], rows[-1]["time"], min(all_y), max(all_y))
        offset = panel * panel_h
        parts.append(f'<g transform="translate(0,{offset})">')
        parts.append(f'<text x="20" y="18" font-size="13">{html.escape(channel)}: raw lower / upper / width</text>')
        for name, series in values.items():
            dash = ' stroke-dasharray="5,4"' if name.endswith("width") else ""
            parts.append(
                f'<polyline fill="none" stroke="{colors[name]}" stroke-width="1"{dash} points="{_polyline(series, width=panel_w, height=panel_h, bounds=bounds)}"/>'
            )
        parts.append('</g>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build(raw_dir: Path, equivalence_path: Path, native_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "flowstar": raw_dir / "flowstar_trace.csv.gz",
        "torch_segments": raw_dir / "torch_segments.csv.gz",
        "torch_attempts": raw_dir / "torch_attempts.csv.gz",
        "torch_ledger": raw_dir / "torch_remainder_ledger.jsonl.gz",
        "equivalence": equivalence_path,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
    if equivalence.get("status") != "STOCK_COPIED_PROBE_EQUIVALENCE_CLOSED":
        raise ValueError("Flow* actual/copied observer parity is not closed")

    flow_rows = [row for row in _read_csv(paths["flowstar"]) if row.get("accepted", "").lower() == "true"]
    torch_rows = [row for row in _read_csv(paths["torch_segments"]) if row.get("status") == "accepted"]
    attempts = {int(row["segment_index"]): row for row in _read_csv(paths["torch_attempts"])}
    ledgers = {int(row["segment_index"]): row for row in _read_jsonl(paths["torch_ledger"])}
    if len(flow_rows) < 632 or len(torch_rows) != 632:
        raise ValueError(f"expected at least/ exactly 632 fixed rows, got {len(flow_rows)} and {len(torch_rows)}")
    flow_rows = flow_rows[:632]
    joined: list[dict[str, Any]] = []
    previous_widths = {tool: {channel: INITIAL_WIDTH[channel[-1]] for channel in CHANNELS} for tool in ("flowstar", "torch")}
    previous_excess = {channel: 0.0 for channel in CHANNELS}

    for index, (flow, torch) in enumerate(zip(flow_rows, torch_rows, strict=True)):
        step = index + 1
        time = float(torch["t_hi"])
        flow_time = float(flow["t_after"])
        if abs(time - flow_time) > 2e-15 or int(flow["accepted_step_index"]) != index or int(torch["segment_index"]) != index:
            raise ValueError(f"fixed common-prefix alignment failed at {step}")
        attempt = attempts[index]
        ledger = ledgers[index]
        outputs: dict[str, Any] = {"endpoint": {}, "segment_tube": {}}
        comparisons: dict[str, Any] = {}
        details = {"flowstar": {}, "torch": {}}
        for object_name in ("endpoint", "segment_tube"):
            for component in ("x", "y"):
                channel = f"{object_name}_{component}"
                flow_output = _output_object("flowstar", flow, object_name, component)
                torch_output = _output_object("torch", torch, object_name, component)
                flow_width = float(flow_output["width"])
                torch_width = float(torch_output["width"])
                excess = torch_width - flow_width
                outputs[object_name][component] = {"flowstar": flow_output, "torch": torch_output}
                comparisons[channel] = {
                    "absolute_excess": excess,
                    "relative_ratio": torch_width / flow_width,
                    "flowstar_width_increment": flow_width - previous_widths["flowstar"][channel],
                    "torch_width_increment": torch_width - previous_widths["torch"][channel],
                    "excess_increment": excess - previous_excess[channel],
                }
                previous_widths["flowstar"][channel] = flow_width
                previous_widths["torch"][channel] = torch_width
                previous_excess[channel] = excess
        for component in ("x", "y"):
            details["flowstar"][component] = _flow_details(flow, component, equivalence)
            details["torch"][component] = _torch_details(torch, attempt, ledger, component)
        joined.append(
            {
                "schema": "vdp_fixed_common_prefix_width_source_ledger_v1",
                "contract": "binary64_literal_matched_contract",
                "schedule_semantics": "fixed_schedule_common_accepted_prefix",
                "step": step,
                "time": time,
                "time_hex": time.hex(),
                "h": float(torch["h_accepted"]),
                "h_hex": torch["h_accepted_hex"],
                "order": int(torch["requested_order"]),
                "status": {"flowstar": flow["status"], "torch": torch["status"]},
                "retry": {
                    "flowstar_attempt_index": int(flow["attempt_index_within_step"]),
                    "torch_step_rejections": int(torch["step_rejections"]),
                    "torch_validation_attempts": int(torch["validation_attempts"]),
                },
                "outputs": outputs,
                "comparison": comparisons,
                "stage_details": details,
            }
        )

    with gzip.open(output_dir / "joined_ledger.jsonl.gz", "wt", encoding="utf-8", newline="\n") as handle:
        for row in joined:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    long_rows: list[dict[str, Any]] = []
    for row in joined:
        for channel in CHANNELS:
            object_name, component = channel.rsplit("_", 1)
            f = row["outputs"][object_name][component]["flowstar"]
            t = row["outputs"][object_name][component]["torch"]
            c = row["comparison"][channel]
            long_rows.append({
                "step": row["step"], "time": row["time"], "channel": channel,
                "flowstar_lo": f["lo"], "flowstar_hi": f["hi"], "flowstar_width": f["width"],
                "torch_lo": t["lo"], "torch_hi": t["hi"], "torch_width": t["width"],
                **c,
            })
    with (output_dir / "width_ledger.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(long_rows)

    crossings: list[dict[str, Any]] = []
    for channel in CHANNELS:
        for threshold in THRESHOLDS:
            match = next((row for row in joined if row["comparison"][channel]["relative_ratio"] > threshold), None)
            crossings.append({
                "channel": channel,
                "threshold": threshold,
                "step": None if match is None else match["step"],
                "time": None if match is None else match["time"],
                "ratio": None if match is None else match["comparison"][channel]["relative_ratio"],
            })
    with (output_dir / "ratio_crossings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(crossings[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(crossings)

    reasons: dict[int, set[str]] = {1: {"mandatory_step_1"}, 2: {"mandatory_step_2"}}
    for target in (0.5, 1.0, 2.0, 3.0, 4.205867, 6.225303, 6.32):
        nearest = min(joined, key=lambda row: abs(row["time"] - target))
        reasons.setdefault(nearest["step"], set()).add(f"nearest_to_{target}")
    for crossing in crossings:
        if crossing["step"] is not None:
            reasons.setdefault(int(crossing["step"]), set()).add(
                f"first_{crossing['channel']}_ratio_gt_{crossing['threshold']}"
            )
    checkpoint_rows = []
    for step in sorted(reasons):
        row = joined[step - 1]
        checkpoint_rows.append({
            "step": step,
            "time": row["time"],
            "reasons": ";".join(sorted(reasons[step])),
            **{
                f"{channel}_{field}": row["comparison"][channel][field]
                for channel in CHANNELS
                for field in ("absolute_excess", "relative_ratio", "excess_increment")
            },
        })
    with (output_dir / "checkpoint_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checkpoint_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(checkpoint_rows)

    minima = []
    for channel in CHANNELS:
        object_name, component = channel.rsplit("_", 1)
        row = min(joined, key=lambda item: item["outputs"][object_name][component]["flowstar"]["width"])
        interval = row["outputs"][object_name][component]["flowstar"]
        minima.append({"channel": channel, "step": row["step"], "time": row["time"], **interval})
    if not all(float(row["width"]) > 0.0086 for row in minima):
        raise ValueError("Flow* raw lo/hi regression found a non-positive or <=0.0086 minimum")
    with (output_dir / "flowstar_width_minima.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(minima[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(minima)
    _write_svg(output_dir / "lower_upper_width.svg", joined)

    t1 = joined[99]["comparison"]
    t3 = joined[299]["comparison"]
    t632 = joined[631]["comparison"]
    t1_excess = [t1[channel]["absolute_excess"] for channel in CHANNELS]
    t3_excess = [t3[channel]["absolute_excess"] for channel in CHANNELS]
    t632_excess = [t632[channel]["absolute_excess"] for channel in CHANNELS]
    if not (0.0027 < min(t1_excess) < max(t1_excess) < 0.0090):
        raise ValueError("T=1 excess regression moved")
    if not (0.0469 < min(t3_excess) < max(t3_excess) < 0.0490):
        raise ValueError("T=3 excess regression moved")
    if not (0.763 < min(t632_excess) and max(t632_excess) < 1.469):
        raise ValueError("T=6.32 excess regression moved")

    terminal = _native_terminal(native_dir)
    _write_json(output_dir / "native_terminal.json", terminal)
    provenance = {
        "schema": "vdp_width_source_ledger_provenance_v1",
        "inputs": {name: {"path": str(path), "sha256": _sha(path)} for name, path in paths.items()},
        "native_inputs": {
            name: {"path": str(native_dir / name), "sha256": _sha(native_dir / name)}
            for name in ("segments.csv", "attempts.csv", "summary.json")
        },
        "source_locations": _source_lines(),
    }
    _write_json(output_dir / "provenance.json", provenance)
    summary = {
        "schema": "vdp_t1_t3_width_source_ledger_summary_v1",
        "fixed_common_prefix_rows": len(joined),
        "long_width_rows": len(long_rows),
        "contract": "binary64_literal_matched_contract",
        "t1_excess_range": [min(t1_excess), max(t1_excess)],
        "t3_excess_range": [min(t3_excess), max(t3_excess)],
        "t6p32_excess_range": [min(t632_excess), max(t632_excess)],
        "flowstar_minima_all_gt_0p0086": True,
        "flowstar_minima": minima,
        "native_terminal_time": terminal["highest_continuously_validated_time"],
        "native_terminal_accepted_segments": terminal["accepted_segments"],
        "native_and_fixed_schedule_ratios_mixed": False,
        "observer_parity": equivalence["status"],
    }
    _write_json(output_dir / "summary.json", summary)
    manifest = {
        "schema": "vdp_t1_t3_width_source_ledger_manifest_v1",
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha(path)}
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
        "summary": summary,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--equivalence", type=Path, default=DEFAULT_EQUIVALENCE)
    parser.add_argument("--native-dir", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build(args.raw_dir, args.equivalence, args.native_dir, args.output_dir)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
