#!/usr/bin/env python3
"""Build the fixed-h=0.01 Flow*/Torch complete-O4 common-prefix table."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.comparison_contract import vdp_identity_hashes


H = 0.01
REQUESTED_STEPS = 1000
TARGET = 1e-4
EXPECTED_FLOWSTAR_METADATA = {
    "ode_dxdt": "y",
    "ode_dydt": "y - x - x^2*y",
    "initial_x": "[1.1,1.4]",
    "initial_y": "[2.35,2.45]",
    "horizon": "10",
    "step_min": format(H, ".17g"),
    "step_max": format(H, ".17g"),
    "starting_attempted_h": format(H, ".17g"),
    "fixed_h_hex": H.hex(),
    "schedule_kind": "fixed",
    "adaptive_fallback_allowed": "false",
    "order": "4",
    "cutoff": "[-1e-10,1e-10]",
    "remainder_estimation": "[-1e-4,1e-4]",
    "symbolic_remainder_enabled": "true",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {token} in {path}")
        ),
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: Mapping[str, str], field: str) -> float:
    raw = row.get(field, "")
    if raw == "":
        raise ValueError(f"required numeric field is missing: {field}")
    value = float(raw)
    if not (float("-inf") < value < float("inf")):
        raise ValueError(f"non-finite field: {field}")
    return value


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_cell(raw: str, *, label: str) -> Any:
    return json.loads(
        raw,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {token} in {label}")
        ),
    )


def _margin(lo: float, hi: float) -> float:
    return min(lo + TARGET, TARGET - hi)


def _parse_elapsed(value: str) -> float:
    parts = value.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60.0 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)
    raise ValueError(f"unrecognized GNU time elapsed value: {value}")


def _parse_gnu_time(path: Path) -> dict[str, float | int]:
    peak_rss: int | None = None
    wall_s: float | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("Maximum resident set size (kbytes):"):
            peak_rss = int(stripped.rsplit(":", 1)[1].strip()) * 1024
        elif stripped.startswith("Elapsed (wall clock) time"):
            wall_s = _parse_elapsed(stripped.split("):", 1)[1].strip())
    if peak_rss is None or peak_rss <= 0:
        raise ValueError(f"GNU time peak RSS field is missing or invalid: {path}")
    if wall_s is None or wall_s < 0.0:
        raise ValueError(f"GNU time wall field is missing or invalid: {path}")
    return {"peak_rss_bytes": peak_rss, "process_wall_s": wall_s}


def _assert_bounds(row: Mapping[str, str], prefixes: Sequence[str], *, label: str) -> None:
    for prefix in prefixes:
        for component in ("x", "y"):
            lo = _float(row, f"{prefix}_{component}_lo")
            hi = _float(row, f"{prefix}_{component}_hi")
            if lo > hi:
                raise ValueError(f"inverted {label} bounds: {prefix}_{component}")


def _assert_interval_pair_json(row: Mapping[str, str], field: str, *, label: str) -> None:
    value = _json_cell(row.get(field, "null"), label=f"{label} {field}")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(side, list) or len(side) != 2 for side in value)
    ):
        raise ValueError(f"{label} {field} must be a 2x2 lo/hi array")
    for component in range(2):
        lo = float(value[0][component])
        hi = float(value[1][component])
        if not (float("-inf") < lo <= hi < float("inf")):
            raise ValueError(f"invalid {label} {field} component {component}")


def _vector_json(row: Mapping[str, str], field: str, *, label: str) -> list[float]:
    value = _json_cell(row.get(field, "null"), label=f"{label} {field}")
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} {field} must contain two components")
    result = [float(item) for item in value]
    if any(not (float("-inf") < item < float("inf")) for item in result):
        raise ValueError(f"{label} {field} contains a non-finite component")
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _first(rows: Sequence[Mapping[str, Any]], predicate: Any) -> Mapping[str, Any] | None:
    return next((row for row in rows if predicate(row)), None)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    flow_path = args.flowstar_trace.resolve()
    flow_metadata_path = args.flowstar_metadata.resolve()
    torch_path = args.torch_segments.resolve()
    torch_summary_path = args.torch_summary.resolve()
    flow_rows_all = _csv(flow_path)
    torch_rows_all = _csv(torch_path)
    torch_summary = _json(torch_summary_path)
    flow_metadata_rows = _csv(flow_metadata_path)
    flow_metadata = {row["key"]: row["value"] for row in flow_metadata_rows}

    for key, expected in EXPECTED_FLOWSTAR_METADATA.items():
        if flow_metadata.get(key) != expected:
            raise ValueError(f"Flow* metadata mismatch for {key}")
    torch_schedule = torch_summary.get("schedule")
    if not isinstance(torch_schedule, Mapping):
        raise ValueError("Torch schedule summary is missing")
    if torch_schedule.get("kind") != "fixed" or torch_schedule.get("h_hex") != H.hex():
        raise ValueError("Torch fixed schedule mismatch")
    if torch_schedule.get("adaptive_fallback_allowed") is not False:
        raise ValueError("Torch fixed schedule permits adaptive fallback")
    if torch_schedule.get("requested_steps") != REQUESTED_STEPS:
        raise ValueError("Torch fixed requested-step count mismatch")
    if torch_summary.get("requested_horizon") != 10.0:
        raise ValueError("Torch requested horizon mismatch")
    if torch_summary.get("requested_order") != 4 or torch_summary.get("support") != "complete_total_degree_O4":
        raise ValueError("Torch complete-O4 support mismatch")
    if torch_summary.get("partition") != "B1" or torch_summary.get("partition_count") != 1:
        raise ValueError("Torch B1 partition mismatch")
    if torch_summary.get("cutoff") != 1e-10 or torch_summary.get("target_remainder_radius") != TARGET:
        raise ValueError("Torch cutoff/target mismatch")
    if torch_summary.get("contract_identity") != vdp_identity_hashes():
        raise ValueError("Torch ODE/initial-set identity mismatch")
    if torch_summary.get("fallback_count") != 0:
        raise ValueError("Torch fixed run used a backend fallback")
    if torch_summary.get("endpoint_repair_used") is not False:
        raise ValueError("Torch fixed run used endpoint repair")

    flow_accepted = [
        row
        for row in flow_rows_all
        if row.get("accepted") == "true" and row.get("status") == "accepted"
    ]
    torch_accepted = [row for row in torch_rows_all if row.get("status") == "accepted"]
    if len(flow_accepted) > REQUESTED_STEPS or len(torch_accepted) > REQUESTED_STEPS:
        raise ValueError("accepted fixed-step count exceeds the requested horizon")
    flow_first_failure_index = next(
        (index for index, row in enumerate(flow_rows_all) if row.get("status") in {"rejected", "failed"}),
        None,
    )
    torch_first_failure_index = next(
        (index for index, row in enumerate(torch_rows_all) if row.get("status") == "rejected"),
        None,
    )
    if flow_first_failure_index is not None and any(
        row.get("status") == "accepted" for row in flow_rows_all[flow_first_failure_index + 1 :]
    ):
        raise ValueError("Flow* accepted a step after its first fixed-schedule failure")
    if torch_first_failure_index is not None and any(
        row.get("status") == "accepted" for row in torch_rows_all[torch_first_failure_index + 1 :]
    ):
        raise ValueError("Torch accepted a step after its first fixed-schedule failure")
    for index, row in enumerate(flow_accepted):
        if _float(row, "h") != H:
            raise ValueError(f"Flow* accepted a nonfixed step at {index}")
        if row.get("h_hex") != H.hex():
            raise ValueError(f"Flow* fixed-step hex mismatch at {index}")
        if _float(row, "t_before") != index * H or row.get("t_before_hex") != (index * H).hex():
            raise ValueError(f"Flow* logical time mismatch at {index}")
        if _float(row, "t_after") != (index + 1) * H or row.get("t_after_hex") != ((index + 1) * H).hex():
            raise ValueError(f"Flow* logical end time mismatch at {index}")
        if not row.get("prestate_state_canonical") or not row.get("retained_coefficients_canonical"):
            raise ValueError(f"Flow* state coefficients are missing at {index}")
        if row.get("flowstar_full_step_tube_source_object") != "accepted_result_Flowpipe_composition_after_remainder_refinement":
            raise ValueError(f"Flow* published tube object mismatch at {index}")
        if row.get("flowstar_tau_h_endpoint_source_object") != "accepted_result_Flowpipe_composition_at_tau_h_after_remainder_refinement":
            raise ValueError(f"Flow* published endpoint object mismatch at {index}")
        if any(
            row.get(field) != "true"
            for field in (
                "flowstar_full_step_tube_includes_ordinary_remainder",
                "flowstar_full_step_tube_includes_symbolic_output_width",
                "flowstar_tau_h_endpoint_includes_ordinary_remainder",
                "flowstar_tau_h_endpoint_includes_symbolic_output_width",
            )
        ):
            raise ValueError(f"Flow* published range omits a retained remainder channel at {index}")
        _assert_bounds(
            row,
            ("flowstar_tau_h_endpoint", "flowstar_full_step_tube", "raw_remainder_after_poly_diff"),
            label="Flow*",
        )
    for index, row in enumerate(torch_accepted):
        if (
            _float(row, "h_attempted") != H
            or _float(row, "h_accepted") != H
            or row.get("h_attempted_hex") != H.hex()
            or row.get("h_accepted_hex") != H.hex()
        ):
            raise ValueError(f"Torch accepted a nonfixed step at {index}")
        if _float(row, "t_lo") != index * H or row.get("t_lo_hex") != (index * H).hex():
            raise ValueError(f"Torch logical time mismatch at {index}")
        if _float(row, "t_hi") != (index + 1) * H or row.get("t_hi_hex") != ((index + 1) * H).hex():
            raise ValueError(f"Torch logical end time mismatch at {index}")
        if not row.get("prestate_sha256") or not row.get("retained_coefficient_sha256"):
            raise ValueError(f"Torch state hashes are missing at {index}")
        if row.get("schedule_kind") != "fixed":
            raise ValueError(f"Torch row schedule mismatch at {index}")
        if row.get("raw_endpoint_published") not in {"True", "true"}:
            raise ValueError(f"Torch accepted raw endpoint is unpublished at {index}")
        if row.get("endpoint_tightening_applied") not in {"False", "false"}:
            raise ValueError(f"Torch accepted endpoint was repaired/tightened at {index}")
        _assert_bounds(row, ("endpoint", "segment"), label="Torch")
        _assert_interval_pair_json(row, "raw_remainder", label="Torch")
        _assert_interval_pair_json(row, "post_poly_diff_remainder", label="Torch")
        _vector_json(row, "prestate_center", label="Torch")
        _vector_json(row, "prestate_scale", label="Torch")

    if int(torch_summary.get("accepted_steps", -1)) != len(torch_accepted):
        raise ValueError("Torch summary/trace accepted-step mismatch")
    if any(
        row.get("raw_endpoint_published") not in {"False", "false", ""}
        or any(row.get(f"endpoint_{component}_{bound}", "") for component in ("x", "y") for bound in ("lo", "hi"))
        for row in torch_rows_all
        if row.get("status") == "rejected"
    ):
        raise ValueError("Torch rejected candidate was published as an endpoint")

    flow_resource = (
        _parse_gnu_time(args.flowstar_resource.resolve())
        if args.flowstar_resource is not None
        else None
    )
    flow_peak_rss = None if flow_resource is None else flow_resource["peak_rss_bytes"]
    torch_peak_rss = torch_summary.get("peak_rss_bytes")
    if torch_peak_rss is not None and (not isinstance(torch_peak_rss, int) or torch_peak_rss <= 0):
        raise ValueError("Torch peak RSS is invalid")

    shared_steps = min(len(flow_accepted), len(torch_accepted))
    prefix = {
        "flowstar": {"x_lo": float("inf"), "x_hi": float("-inf"), "y_lo": float("inf"), "y_hi": float("-inf")},
        "torch": {"x_lo": float("inf"), "x_hi": float("-inf"), "y_lo": float("inf"), "y_hi": float("-inf")},
    }
    common: list[dict[str, Any]] = []
    cumulative_flow = 0.0
    cumulative_torch = 0.0
    for index in range(shared_steps):
        flow = flow_accepted[index]
        torch = torch_accepted[index]
        torch_center = _vector_json(torch, "prestate_center", label="Torch")
        torch_scale = _vector_json(torch, "prestate_scale", label="Torch")
        cumulative_flow += _float(flow, "stage_runtime_seconds")
        cumulative_torch += _float(torch, "stage_runtime_s")
        row: dict[str, Any] = {
            "step": index + 1,
            "time": (index + 1) * H,
            "time_hex": ((index + 1) * H).hex(),
            "both_completed": True,
            "flowstar_prestate_sha256": _hash_text(flow["prestate_state_canonical"]),
            "flowstar_retained_coefficient_sha256": _hash_text(flow["retained_coefficients_canonical"]),
            "torch_prestate_sha256": torch["prestate_sha256"],
            "torch_retained_coefficient_sha256": torch["retained_coefficient_sha256"],
            "flowstar_prestate_center_x": _float(flow, "extracted_center_x"),
            "flowstar_prestate_center_y": _float(flow, "extracted_center_y"),
            "flowstar_prestate_scale_x": _float(flow, "extracted_scale_x"),
            "flowstar_prestate_scale_y": _float(flow, "extracted_scale_y"),
            "torch_prestate_center_x": torch_center[0],
            "torch_prestate_center_y": torch_center[1],
            "torch_prestate_scale_x": torch_scale[0],
            "torch_prestate_scale_y": torch_scale[1],
            "flowstar_cumulative_runtime_s": cumulative_flow,
            "torch_cumulative_runtime_s": cumulative_torch,
            "flowstar_peak_rss_bytes": flow_peak_rss if flow_peak_rss is not None else "unavailable",
            "torch_peak_rss_bytes": torch_peak_rss if torch_peak_rss is not None else "unavailable",
            "qualification": "empirical_or_build_qualified_schedule_controlled",
        }
        for component in ("x", "y"):
            row[f"prestate_center_{component}_delta_torch_minus_flowstar"] = (
                row[f"torch_prestate_center_{component}"]
                - row[f"flowstar_prestate_center_{component}"]
            )
            row[f"prestate_scale_{component}_delta_torch_minus_flowstar"] = (
                row[f"torch_prestate_scale_{component}"]
                - row[f"flowstar_prestate_scale_{component}"]
            )
        for tool, source, endpoint_prefix, tube_prefix in (
            ("flowstar", flow, "flowstar_tau_h_endpoint", "flowstar_full_step_tube"),
            ("torch", torch, "endpoint", "segment"),
        ):
            for component in ("x", "y"):
                lo = _float(source, f"{endpoint_prefix}_{component}_lo")
                hi = _float(source, f"{endpoint_prefix}_{component}_hi")
                tube_lo = _float(source, f"{tube_prefix}_{component}_lo")
                tube_hi = _float(source, f"{tube_prefix}_{component}_hi")
                prefix[tool][f"{component}_lo"] = min(
                    prefix[tool][f"{component}_lo"], tube_lo
                )
                prefix[tool][f"{component}_hi"] = max(
                    prefix[tool][f"{component}_hi"], tube_hi
                )
                row[f"{tool}_endpoint_{component}_lo"] = lo
                row[f"{tool}_endpoint_{component}_hi"] = hi
                row[f"{tool}_endpoint_{component}_width"] = hi - lo
                row[f"{tool}_segment_tube_{component}_lo"] = tube_lo
                row[f"{tool}_segment_tube_{component}_hi"] = tube_hi
                row[f"{tool}_segment_tube_{component}_width"] = tube_hi - tube_lo
                row[f"{tool}_prefix_tube_{component}_lo"] = prefix[tool][f"{component}_lo"]
                row[f"{tool}_prefix_tube_{component}_hi"] = prefix[tool][f"{component}_hi"]
                row[f"{tool}_prefix_tube_{component}_width"] = (
                    prefix[tool][f"{component}_hi"] - prefix[tool][f"{component}_lo"]
                )
        for component in ("x", "y"):
            flow_margin = _margin(
                _float(flow, f"raw_remainder_after_poly_diff_{component}_lo"),
                _float(flow, f"raw_remainder_after_poly_diff_{component}_hi"),
            )
            row[f"flowstar_margin_{component}"] = flow_margin
        torch_margins = _json_cell(
            torch.get("target_margins", "null"), label="Torch target_margins"
        )
        if (
            isinstance(torch_margins, list)
            and len(torch_margins) == 1
            and isinstance(torch_margins[0], list)
        ):
            torch_margins = torch_margins[0]
        if not isinstance(torch_margins, list) or len(torch_margins) != 2:
            raise ValueError(f"Torch target margins missing at {index}")
        parsed_margins = [float(item) for item in torch_margins]
        if any(not (float("-inf") < item < float("inf")) for item in parsed_margins):
            raise ValueError(f"Torch target margins are non-finite at {index}")
        row["torch_margin_x"] = parsed_margins[0]
        row["torch_margin_y"] = parsed_margins[1]
        for object_name in ("endpoint", "segment_tube", "prefix_tube"):
            for component in ("x", "y"):
                flow_width = row[f"flowstar_{object_name}_{component}_width"]
                torch_width = row[f"torch_{object_name}_{component}_width"]
                row[f"{object_name}_{component}_width_delta_torch_minus_flowstar"] = (
                    torch_width - flow_width
                )
                row[f"{object_name}_{component}_width_ratio_torch_over_flowstar"] = (
                    torch_width / flow_width if flow_width != 0.0 else "unavailable"
                )
        common.append(row)

    width_keys = [
        f"{object_name}_{component}_width_delta_torch_minus_flowstar"
        for object_name in ("endpoint", "segment_tube", "prefix_tube")
        for component in ("x", "y")
    ]
    first_width_divergence = _first(
        common, lambda row: any(row[key] != 0.0 for key in width_keys)
    )
    first_torch_wider = _first(
        common, lambda row: any(row[key] > 0.0 for key in width_keys)
    )
    first_flowstar_wider = _first(
        common, lambda row: any(row[key] < 0.0 for key in width_keys)
    )
    first_margin_ordering_change: Mapping[str, Any] | None = None
    if common:
        initial_signs = {
            component: (difference > 0.0) - (difference < 0.0)
            for component in ("x", "y")
            for difference in [
                common[0][f"torch_margin_{component}"]
                - common[0][f"flowstar_margin_{component}"]
            ]
        }
        first_margin_ordering_change = _first(
            common[1:],
            lambda row: any(
                (
                    (row[f"torch_margin_{component}"] - row[f"flowstar_margin_{component}"] > 0.0)
                    - (row[f"torch_margin_{component}"] - row[f"flowstar_margin_{component}"] < 0.0)
                )
                != initial_signs[component]
                for component in ("x", "y")
            ),
        )

    flow_failure = next(
        (row for row in flow_rows_all if row.get("status") in {"rejected", "failed"}),
        None,
    )
    torch_failure = next(
        (row for row in torch_rows_all if row.get("status") == "rejected"), None
    )
    if len(flow_accepted) < REQUESTED_STEPS and flow_failure is None:
        raise ValueError("Flow* trace is truncated before T10 without a failure row")
    torch_environment_blocked = (
        torch_summary.get("status") == "timeout"
        or torch_summary.get("failure_type") == "timeout_resource_exhaustion"
    )
    if (
        len(torch_accepted) < REQUESTED_STEPS
        and torch_failure is None
        and not torch_environment_blocked
    ):
        raise ValueError("Torch trace is truncated before T10 without a failure row")
    flow_complete = len(flow_accepted) == REQUESTED_STEPS and flow_failure is None
    torch_complete = (
        len(torch_accepted) == REQUESTED_STEPS
        and torch_summary.get("completed_requested_horizon") is True
        and torch_failure is None
    )
    outcome = (
        "FLOWSTAR_TORCH_FIXED_SCHEDULE_T10_BOTH_COMPLETE"
        if flow_complete and torch_complete
        else "FLOWSTAR_TORCH_FIXED_SCHEDULE_ENVIRONMENT_BLOCKED"
        if torch_environment_blocked
        else "FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_ONLY"
    )
    source_sha256 = {
        "flowstar_trace.csv": _sha(flow_path),
        "flowstar_trace_metadata.csv": _sha(flow_metadata_path),
        "torch_segments.csv": _sha(torch_path),
        "torch_summary.json": _sha(torch_summary_path),
    }
    if args.flowstar_resource is not None:
        source_sha256["flowstar_resource.txt"] = _sha(args.flowstar_resource.resolve())
    summary = {
        "schema": "flowstar_torch_fixed_schedule_common_prefix_v1",
        "outcome": outcome,
        "scope": "full_horizon_or_first_failure",
        "contract": {
            "rhs_sha256": vdp_identity_hashes()["rhs_sha256"],
            "initial_set_sha256": vdp_identity_hashes()["initial_set_sha256"],
            "partition_sha256": vdp_identity_hashes()["partition_b1_sha256"],
            "partition_count": 1,
            "support": "complete_total_degree_O4",
            "h_decimal": format(H, ".17g"),
            "h_hex": H.hex(),
            "requested_steps": REQUESTED_STEPS,
            "requested_horizon": 10.0,
            "target_remainder": [-TARGET, TARGET],
            "cutoff": [-1e-10, 1e-10],
            "adaptive_fallback": False,
            "endpoint_repair": False,
        },
        "flowstar": {
            "accepted_steps": len(flow_accepted),
            "validated_horizon": len(flow_accepted) * H,
            "completed_t10": flow_complete,
            "first_failure_step": None if flow_failure is None else len(flow_accepted) + 1,
            "failure_reason": None if flow_failure is None else flow_failure.get("message"),
            "coefficient_type": "Flowstar Real/Interval",
            "build_qualification": "scalar_affine_under_enclosure_open",
            "peak_rss_bytes": flow_peak_rss,
            "process_wall_s": None if flow_resource is None else flow_resource["process_wall_s"],
        },
        "torch": {
            "accepted_steps": len(torch_accepted),
            "validated_horizon": len(torch_accepted) * H,
            "completed_t10": torch_complete,
            "first_failure_step": None if torch_failure is None else len(torch_accepted) + 1,
            "failure_reason": None if torch_failure is None else torch_failure.get("message"),
            "coefficient_type": "torch.float64",
            "fallback_count": torch_summary.get("fallback_count"),
            "endpoint_repair_used": torch_summary.get("endpoint_repair_used"),
            "peak_rss_bytes": torch_peak_rss,
            "process_wall_s": torch_summary.get("runtime_s"),
        },
        "shared": {
            "accepted_steps": shared_steps,
            "validated_horizon": shared_steps * H,
            "first_width_divergence_step": None if first_width_divergence is None else first_width_divergence["step"],
            "first_margin_ordering_change_step": None if first_margin_ordering_change is None else first_margin_ordering_change["step"],
            "first_torch_wider_step": None if first_torch_wider is None else first_torch_wider["step"],
            "first_flowstar_wider_step": None if first_flowstar_wider is None else first_flowstar_wider["step"],
            "tightness_eligibility": "empirical_or_build_qualified_common_prefix",
            "same_prestate_scope": "initial_state_only",
            "later_scope": "schedule_controlled_comparative_trace",
            "first_failure": [
                item
                for item in (
                    None
                    if flow_failure is None
                    else {"tool": "flowstar", "step": len(flow_accepted) + 1, "time": len(flow_accepted) * H},
                    None
                    if torch_failure is None
                    else {"tool": "torch", "step": len(torch_accepted) + 1, "time": len(torch_accepted) * H},
                )
                if item is not None
            ],
        },
        "environment_blocked": (
            None
            if not torch_environment_blocked
            else {
                "tool": "torch",
                "reason": torch_summary.get("failure_type") or torch_summary.get("message"),
            }
        ),
        "source_sha256": source_sha256,
    }
    _write_csv(output / "common_prefix.csv", common)
    summary["common_prefix_sha256"] = _sha(output / "common_prefix.csv")
    _write_json(output / "summary.json", summary)
    (output / "report.md").write_text(
        "# Flow*/Torch fixed-schedule common prefix\n\n"
        f"Outcome: `{outcome}`\n\n"
        f"Shared validated horizon: {shared_steps * H:.17g}\n\n"
        f"Flow* validated horizon: {len(flow_accepted) * H:.17g}\n\n"
        f"Torch validated horizon: {len(torch_accepted) * H:.17g}\n\n"
        "Qualification: empirical/build-qualified only; the stock Flow* "
        "scalar-affine correctness gate remains open.\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flowstar-trace", type=Path, required=True)
    parser.add_argument("--flowstar-metadata", type=Path, required=True)
    parser.add_argument("--torch-segments", type=Path, required=True)
    parser.add_argument("--torch-summary", type=Path, required=True)
    parser.add_argument("--flowstar-resource", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(compare(parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
