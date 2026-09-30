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
    if (
        spec.get("schema") != "torch-tm-flowpipe-plot-spec-v3"
        or not isinstance(declared, dict)
    ):
        raise ValueError("derived projection is absent from the plot spec")
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
                    _affine_interval(row, lo_col, hi_col, projection["y_transform"])
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
                    _affine_interval(
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
