"""Independently recompute saved flowpipe plot boxes and artifact hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

import torch


DOCKING_COORDINATES = ["sx", "sy", "sx_dot", "sy_dot"]
DOCKING_INITIAL_BOUNDS = {
    "sx": [70.0, 106.0],
    "sy": [70.0, 106.0],
    "sx_dot": [-0.28, 0.28],
    "sy_dot": [-0.28, 0.28],
}


def _affine_interval(
    bounds: Any,
    lo_column: int,
    hi_column: int,
    transform: dict[str, Any],
) -> tuple[float, float]:
    """Independent binary64-outward affine image for artifact verification."""
    lower = float(transform["offset"])
    upper = float(transform["offset"])
    for term in transform["terms"]:
        coefficient = float(term["coefficient"])
        index = int(term["index"])
        raw_lo = float(bounds[index][lo_column])
        raw_hi = float(bounds[index][hi_column])
        selected_lo, selected_hi = (
            (raw_lo, raw_hi) if coefficient > 0.0 else (raw_hi, raw_lo)
        )
        lower = math.nextafter(
            lower + math.nextafter(coefficient * selected_lo, -math.inf),
            -math.inf,
        )
        upper = math.nextafter(
            upper + math.nextafter(coefficient * selected_hi, math.inf),
            math.inf,
        )
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("affine derived-coordinate interval is nonfinite")
    return lower, upper


def _finite_outward(value: float, direction: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} is nonfinite")
    widened = math.nextafter(value, direction)
    if not math.isfinite(widened):
        raise ValueError(f"{label} overflows binary64")
    return widened


def _radial_interval(
    bounds: Any,
    lo_column: int,
    hi_column: int,
    terms: list[dict[str, Any]],
) -> tuple[float, float]:
    if len(terms) != 2:
        raise ValueError("radial interval requires exactly two coordinates")
    absolute_bounds: list[tuple[float, float]] = []
    for term in terms:
        lo = float(bounds[int(term["index"])][lo_column])
        hi = float(bounds[int(term["index"])][hi_column])
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise ValueError("radial coordinate bounds must be finite and ordered")
        absolute_bounds.append((
            0.0 if lo <= 0.0 <= hi else min(abs(lo), abs(hi)),
            max(abs(lo), abs(hi)),
        ))

    square_lowers: list[float] = []
    square_uppers: list[float] = []
    for absolute_lo, absolute_hi in absolute_bounds:
        raw_lower = absolute_lo * absolute_lo
        raw_upper = absolute_hi * absolute_hi
        if not math.isfinite(raw_lower) or not math.isfinite(raw_upper):
            raise ValueError("radial square overflows binary64")
        square_lowers.append(
            0.0
            if absolute_lo == 0.0
            else max(0.0, _finite_outward(raw_lower, -math.inf, "radial square lower"))
        )
        square_uppers.append(
            0.0
            if absolute_hi == 0.0
            else _finite_outward(raw_upper, math.inf, "radial square upper")
        )

    raw_sum_lower = square_lowers[0] + square_lowers[1]
    raw_sum_upper = square_uppers[0] + square_uppers[1]
    if not math.isfinite(raw_sum_lower) or not math.isfinite(raw_sum_upper):
        raise ValueError("radial square sum overflows binary64")
    sum_lower = (
        0.0
        if raw_sum_lower == 0.0
        else max(0.0, _finite_outward(raw_sum_lower, -math.inf, "radial sum lower"))
    )
    sum_upper = (
        0.0
        if raw_sum_upper == 0.0
        else _finite_outward(raw_sum_upper, math.inf, "radial sum upper")
    )
    raw_lower = math.sqrt(sum_lower)
    raw_upper = math.sqrt(sum_upper)
    return (
        0.0
        if raw_lower == 0.0
        else max(0.0, _finite_outward(raw_lower, -math.inf, "radial sqrt lower")),
        0.0
        if raw_upper == 0.0
        else _finite_outward(raw_upper, math.inf, "radial sqrt upper"),
    )


def _radial_speed_margin_interval(
    bounds: Any,
    lo_column: int,
    hi_column: int,
    transform: dict[str, Any],
) -> tuple[float, float]:
    position_lo, position_hi = _radial_interval(
        bounds, lo_column, hi_column, transform["position_terms"]
    )
    velocity_lo, velocity_hi = _radial_interval(
        bounds, lo_column, hi_column, transform["velocity_terms"]
    )
    offset = float(transform["offset"])
    gain = float(transform["radial_gain"])
    lower_product = (
        0.0
        if position_lo == 0.0
        else _finite_outward(gain * position_lo, -math.inf, "radial gain lower")
    )
    upper_product = (
        0.0
        if position_hi == 0.0
        else _finite_outward(gain * position_hi, math.inf, "radial gain upper")
    )
    lower_sum = _finite_outward(offset + lower_product, -math.inf, "margin sum lower")
    upper_sum = _finite_outward(offset + upper_product, math.inf, "margin sum upper")
    return (
        _finite_outward(lower_sum - velocity_hi, -math.inf, "margin lower"),
        _finite_outward(upper_sum - velocity_lo, math.inf, "margin upper"),
    )


def _derived_interval(
    bounds: Any,
    lo_column: int,
    hi_column: int,
    transform: dict[str, Any],
) -> tuple[float, float]:
    if transform.get("kind") == "affine":
        return _affine_interval(bounds, lo_column, hi_column, transform)
    if transform.get("kind") == "radial_speed_margin":
        return _radial_speed_margin_interval(bounds, lo_column, hi_column, transform)
    raise ValueError("unsupported derived projection kind")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_affine_projection_contract(geometry: dict[str, Any]) -> None:
    projection = geometry["projection"]
    transform = projection.get("y_transform")
    if transform is None:
        return
    spec = geometry.get("spec", {})
    name = projection.get("y")
    declared = spec.get("derived_coordinates", {}).get(name)
    coordinate_names = geometry["coordinate_names"]
    schema = spec.get("schema")
    if schema not in {
        "torch-tm-flowpipe-plot-spec-v3",
        "torch-tm-flowpipe-plot-spec-v4",
    } or not isinstance(declared, dict):
        raise ValueError("derived projection is absent from the plot spec")
    if schema == "torch-tm-flowpipe-plot-spec-v3":
        expected = {
            "kind": "affine",
            "name": name,
            "offset": float(declared["offset"]),
            "terms": [
                {
                    "coordinate": coordinate,
                    "index": coordinate_names.index(coordinate),
                    "coefficient": float(declared["coefficients"][coordinate]),
                }
                for coordinate in coordinate_names
                if coordinate in declared["coefficients"]
            ],
        }
    else:
        regions = spec.get("regions")
        if (
            geometry.get("benchmark") != "Docking"
            or geometry.get("instance_id") != "docking-constraint"
            or coordinate_names != DOCKING_COORDINATES
            or name != "docking_safety_margin"
            or set(declared) != {
                "kind", "offset", "radial_gain",
                "position_coordinates", "velocity_coordinates",
            }
            or declared.get("kind") != "radial_speed_margin"
            or not isinstance(declared.get("offset"), (int, float))
            or isinstance(declared.get("offset"), bool)
            or float(declared["offset"]).hex() != float(0.2).hex()
            or not isinstance(declared.get("radial_gain"), (int, float))
            or isinstance(declared.get("radial_gain"), bool)
            or float(declared["radial_gain"]).hex() != float(0.002054).hex()
            or declared.get("position_coordinates") != DOCKING_COORDINATES[:2]
            or declared.get("velocity_coordinates") != DOCKING_COORDINATES[2:]
            or spec.get("horizon") != {
                "kind": "continuous_time", "start": 0.0, "end": 40.0
            }
            or spec.get("property_quantifier") != "all_times"
            or spec.get("initial_set", {}).get("bounds") != DOCKING_INITIAL_BOUNDS
            or not isinstance(regions, list)
            or len(regions) != 1
            or regions[0].get("role") != "safe"
            or regions[0].get("constraint") != {
                "kind": "threshold", "coordinate": "docking_safety_margin",
                "operator": ">=", "value": 0.0,
            }
            or regions[0].get("time") != {"kind": "all"}
        ):
            raise ValueError("v4 projection is not the canonical Docking contract")

        def indexed(coordinates: list[str]) -> list[dict[str, Any]]:
            return [
                {"coordinate": coordinate,
                 "index": coordinate_names.index(coordinate)}
                for coordinate in coordinates
            ]

        expected = {
            "kind": "radial_speed_margin",
            "name": name,
            "offset": float(declared["offset"]),
            "radial_gain": float(declared["radial_gain"]),
            "position_terms": indexed(declared["position_coordinates"]),
            "velocity_terms": indexed(declared["velocity_coordinates"]),
        }
    if transform != expected:
        raise ValueError("derived projection metadata does not match the plot spec")


def _observer_boxes(
    geometry: dict[str, Any], series: dict[str, Any], frame: dict[str, Any]
) -> tuple[list[list[float]], list[int] | None]:
    path = Path(series["source_path"]) / frame["source_file"]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    chosen = payload["bounds"][payload["accepted"]]
    lo_col, hi_col = (0, 1) if geometry["view"] == "tube" else (2, 3)
    projection = geometry["projection"]
    if projection["kind"] == "time-state":
        boxes = []
        if chosen.shape[0]:
            if geometry["view"] == "tube":
                xlo = (frame["step"] - 1) * geometry["step_size"]
                xhi = frame["step"] * geometry["step_size"]
            else:
                xlo = xhi = frame["step"] * geometry["step_size"]
            if "y_transform" in projection:
                intervals = [
                    _derived_interval(row, lo_col, hi_col, projection["y_transform"])
                    for row in chosen
                ]
                ylo = min(interval[0] for interval in intervals)
                yhi = max(interval[1] for interval in intervals)
            else:
                index = projection["y_index"]
                ylo = float(chosen[:, index, lo_col].amin())
                yhi = float(chosen[:, index, hi_col].amax())
            boxes = [[xlo, xhi, ylo, yhi]]
        lane_ids = None
    else:
        x_index, y_index = projection["x_index"], projection["y_index"]
        boxes = [[float(row[x_index, lo_col]), float(row[x_index, hi_col]),
                  float(row[y_index, lo_col]), float(row[y_index, hi_col])]
                 for row in chosen]
        lane_ids = payload["accepted"].nonzero(as_tuple=False).flatten().tolist()
    if frame["source_sha256"] != _sha256(path):
        raise ValueError(f"{path}: observer hash mismatch")
    return boxes, lane_ids


def _verify_observer(
    geometry: dict[str, Any], series: dict[str, Any]
) -> None:
    root = Path(series["source_path"])
    files = series["source_identity"]["files"]
    if len(files) != len(series["observed_steps"]):
        raise ValueError(f"{root}: observer hash manifest is incomplete")
    for item in files:
        path = root / item["source_file"]
        if item["source_sha256"] != _sha256(path):
            raise ValueError(f"{path}: source hash mismatch")
        if item["sidecar"] is not None:
            sidecar = root / item["sidecar"]["source_file"]
            if item["sidecar"]["source_sha256"] != _sha256(sidecar):
                raise ValueError(f"{sidecar}: source hash mismatch")
    for frame in series["frames"]:
        boxes, lane_ids = _observer_boxes(geometry, series, frame)
        if frame["boxes"] != boxes or frame["lane_ids"] != lane_ids:
            raise ValueError(f"{root}: frame {frame['step']} geometry mismatch")


def _verify_native(geometry: dict[str, Any], series: dict[str, Any]) -> None:
    path = Path(series["source_path"])
    if series["source_sha256"] != _sha256(path):
        raise ValueError(f"{path}: source hash mismatch")
    state_count = len(geometry["coordinate_names"])
    record = struct.Struct(f"<QQd{4 * state_count}d")
    if path.stat().st_size % record.size:
        raise ValueError(f"{path}: truncated native record")
    by_step: dict[int, list[tuple[int, list[list[float]], int]]] = {}
    with path.open("rb") as handle:
        for index in range(path.stat().st_size // record.size):
            lane, step, recorded_h, *flat = record.unpack(handle.read(record.size))
            if recorded_h != geometry["step_size"]:
                raise ValueError(f"{path}: record {index} step size mismatch")
            bounds = [flat[offset:offset + 4]
                      for offset in range(0, len(flat), 4)]
            by_step.setdefault(step, []).append((lane, bounds, index))
    projection = geometry["projection"]
    lo_col, hi_col = (0, 1) if geometry["view"] == "tube" else (2, 3)
    for frame in series["frames"]:
        rows = by_step[frame["step"]]
        if projection["kind"] == "time-state":
            if "y_transform" in projection:
                intervals = [
                    _derived_interval(
                        bounds, lo_col, hi_col, projection["y_transform"]
                    )
                    for _, bounds, _ in rows
                ]
                ylo = min(interval[0] for interval in intervals)
                yhi = max(interval[1] for interval in intervals)
            else:
                state = projection["y_index"]
                ylo = min(bounds[state][lo_col] for _, bounds, _ in rows)
                yhi = max(bounds[state][hi_col] for _, bounds, _ in rows)
            end = frame["step"] * geometry["step_size"]
            start = (
                (frame["step"] - 1) * geometry["step_size"]
                if geometry["view"] == "tube"
                else end
            )
            boxes = [[start, end, ylo, yhi]]
        else:
            x_index, y_index = projection["x_index"], projection["y_index"]
            boxes = [[bounds[x_index][lo_col], bounds[x_index][hi_col],
                      bounds[y_index][lo_col], bounds[y_index][hi_col]]
                     for _, bounds, _ in rows]
        if frame["boxes"] != boxes:
            raise ValueError(f"{path}: frame {frame['step']} geometry mismatch")


def verify_artifact(root: Path, stem: str) -> dict[str, Any]:
    geometry = json.loads((root / f"{stem}.geometry.json").read_text())
    receipt = json.loads((root / f"{stem}.render.json").read_text())
    _verify_affine_projection_contract(geometry)
    for name, entry in receipt["artifacts"].items():
        if entry is not None:
            path = Path(entry["path"])
            if entry["sha256"] != _sha256(path) or entry["bytes"] != path.stat().st_size:
                raise ValueError(f"{stem}: {name} receipt mismatch")
    for series in geometry["series"]:
        if series["source_kind"] == "torch-observer-pt-v1":
            _verify_observer(geometry, series)
        elif series["source_kind"] == "native-ranges-bin-v1":
            _verify_native(geometry, series)
        else:
            raise ValueError(f"{stem}: unsupported source kind")
    return {
        "stem": stem,
        "frames": sum(len(item["frames"]) for item in geometry["series"]),
        "boxes": sum(len(frame["boxes"]) for item in geometry["series"]
                     for frame in item["frames"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    args = parser.parse_args()
    stems = sorted(path.name.removesuffix(".geometry.json")
                   for path in args.artifact_root.glob("*.geometry.json"))
    if not stems:
        raise ValueError("artifact root contains no geometry JSON files")
    results = [verify_artifact(args.artifact_root, stem) for stem in stems]
    print(json.dumps({"status": "passed", "artifacts": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
