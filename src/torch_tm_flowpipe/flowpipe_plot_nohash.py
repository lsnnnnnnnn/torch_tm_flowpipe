"""Render saved native range boxes or geometry without computing content digests.

Native ranges.bin records show presence, not solver acceptance or a certificate.
The plot spec and optional run configuration are declarations, unbound to the
binary unless a separate verification establishes that relationship.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from . import flowpipe_plot as plot


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _saved_geometry(path: Path) -> dict[str, Any]:
    value = _read_object(path)
    spec = value.get("spec")
    if not isinstance(spec, dict) or spec.get("schema") != plot.PLOT_SPEC_SCHEMA:
        raise ValueError(f"{path}: no-hash redraw requires a v1 informational plot spec")
    return plot.validate_geometry(value)


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size}


def _native_geometry(args: argparse.Namespace) -> dict[str, Any]:
    if not args.spec:
        raise ValueError("--spec is required for native ranges so initial and property regions are explicit")
    if args.step_size is None or not math.isfinite(args.step_size) or args.step_size <= 0:
        raise ValueError("--step-size must be finite and positive")
    if args.expected_steps is None or args.expected_steps < 1:
        raise ValueError("--expected-steps must be positive")
    if args.expected_lanes is None or args.expected_lanes < 1:
        raise ValueError("--expected-lanes must be positive")
    names = [name.strip() for name in args.coordinate_names.split(",")]
    if not args.coordinate_names or any(not name for name in names):
        raise ValueError("--coordinate-names must list the physical states in binary record order")
    spec = _read_object(args.spec)
    if spec.get("schema") != plot.PLOT_SPEC_SCHEMA:
        raise ValueError("no-hash native export requires a v1 informational plot spec without source hashes")
    if spec.get("coordinate_names") != names:
        raise ValueError("plot spec coordinate_names must match --coordinate-names")
    if spec.get("instance_id") is not None and spec["instance_id"] != args.instance_id:
        raise ValueError("plot spec instance_id must match --instance-id")
    horizon = spec.get("horizon")
    if spec.get("instance_id") is not None and not isinstance(horizon, dict):
        raise ValueError("instance-specific no-hash spec must declare its full numerical horizon")
    if horizon is not None and (
        not isinstance(horizon, dict)
        or horizon.get("kind") != "continuous_time"
        or horizon.get("start") != 0
        or not isinstance(horizon.get("end"), (int, float))
        or isinstance(horizon.get("end"), bool)
        or not math.isfinite(horizon["end"])
        or not math.isclose(
            horizon["end"], args.expected_steps * args.step_size,
            rel_tol=0.0, abs_tol=1e-12,
        )
    ):
        raise ValueError("plot spec horizon must match --expected-steps times --step-size")
    if not spec.get("initial_set", {}).get("bounds") or not spec.get("regions"):
        raise ValueError("plot spec must declare initial bounds and property regions")
    spec_binding = plot._validate_plot_spec(
        spec, args.benchmark, names, instance_id=args.instance_id,
        numerical_horizon=args.expected_steps * args.step_size,
    )
    projection = plot._parse_projection(args.projection, names)
    selected = plot._parse_display_steps(args.display_steps, args.expected_steps)
    frames, observed, partial = plot._native_frames(
        args.ranges,
        projection=projection,
        view=args.view,
        step_size=args.step_size,
        state_count=len(names),
        expected_lanes=args.expected_lanes,
        expected_steps=args.expected_steps,
        display_steps=selected,
        partial_policy=args.partial_policy,
    )
    displayed = [frame["step"] for frame in frames]
    omitted = sorted(set(observed) - set(displayed))
    configuration = (
        {"path": str(args.run_config.resolve()), "content": _read_object(args.run_config),
         "binding": "declared file; not identity-bound to ranges.bin"}
        if args.run_config else None
    )
    result_path = args.ranges.with_name("RESULT.json")
    evidence = {
        "binding": {"series": "adjacent run evidence unbound to ranges.bin"},
        "plot_contract_check": {
            "status": "explicit plotting declarations unbound to native records",
            "step_size": "matched to each binary record h",
            "expected_steps": "explicit_cli_unverified",
            "lane_count": "explicit_cli_unverified",
        },
        "run_configuration": configuration,
    }
    if result_path.is_file():
        evidence["RESULT.json"] = _read_object(result_path)
        evidence["RESULT_path"] = str(result_path.resolve())
        evidence["RESULT_binding"] = "adjacent file; unbound to ranges.bin"
    source = {
        "label": args.label,
        "source_kind": "native-ranges-bin-v1",
        "source_path": str(args.ranges.resolve()),
        "declared_origin_path": args.origin_path,
        "origin_binding": (
            "user-declared copy origin; byte identity not established by this plot"
            if args.origin_path else "local source path only"
        ),
        "source_bytes": args.ranges.stat().st_size,
        "record_size": plot._native_range_record(len(names)).size,
        "record_schema": (
            f"little-endian <lane:uint64,step:uint64,h:float64,{4 * len(names)}xfloat64>; "
            "tube_lo,tube_hi,endpoint_lo,endpoint_hi per physical state"
        ),
        "physical_state_count_from_explicit_coordinates": len(names),
        "lane_universe_binding": "explicit_cli_expected_lanes; not verified against initial partition",
        "solver_run_evidence": evidence,
        "observed_steps": observed,
        "displayed_steps": displayed,
        "expected_steps": args.expected_steps,
        "display_selection": {
            "policy": "all-observed" if selected is None else "explicit",
            "requested_steps": None if selected is None else sorted(selected),
        },
        "projection_unobserved_step_count": args.expected_steps - len(observed),
        "projection_unobserved_step_ranges": plot._missing_ranges(observed, args.expected_steps),
        "projection_unobserved_reason": "no native range record; solver status is not inferred",
        "display_omitted_observed_step_count": len(omitted),
        "display_omitted_observed_step_ranges": plot._compact_ranges(omitted),
        "projection_coverage_semantics": "record presence only; acceptance and certification unknown",
        "projection_partial_step_ranges": plot._compact_ranges(partial),
        "projection_partial_coverage_semantics": "known record coverage for every observed step",
        "displayed_partial_step_ranges": plot._compact_ranges(
            frame["step"] for frame in frames if not frame["complete"]
        ),
        "frames": frames,
    }
    geometry = {
        "schema": plot.SCHEMA,
        "benchmark": args.benchmark,
        "coordinate_names": names,
        "projection": projection,
        "view": args.view,
        "observer_columns": ["tube_lo", "tube_hi", "endpoint_lo", "endpoint_hi"],
        "step_size": args.step_size,
        "step_size_hex": args.step_size.hex(),
        "expected_steps": args.expected_steps,
        "partial_policy": args.partial_policy,
        "interpolation": "none",
        "geometry_class": plot._projection_disclosure(projection),
        "semantics": (
            "tube bounds cover one local integration step" if args.view == "tube"
            else "endpoint bounds are evaluated at the end of one integration step"
        ),
        "series": [source],
        "spec": spec,
        "spec_binding": spec_binding,
    }
    if args.instance_id:
        geometry["instance_id"] = args.instance_id
    return plot.validate_geometry(geometry)


def _overlay_geometry(paths: list[Path]) -> dict[str, Any]:
    geometries = [_saved_geometry(path) for path in paths]
    first = geometries[0]
    common = (
        "schema", "benchmark", "instance_id", "coordinate_names", "projection",
        "view", "observer_columns", "step_size", "step_size_hex",
        "expected_steps", "partial_policy", "interpolation", "geometry_class",
        "semantics", "spec", "spec_binding",
    )
    labels: set[str] = set()
    series: list[dict[str, Any]] = []
    for path, geometry in zip(paths, geometries):
        for key in common:
            if geometry.get(key) != first.get(key):
                raise ValueError(f"{path}: geometry {key} differs from the first input")
        for item in geometry["series"]:
            label = item["label"]
            if label in labels:
                raise ValueError(f"{path}: duplicate series label {label!r}")
            labels.add(label)
            series.append(item)
    return plot.validate_geometry({**first, "series": series})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ranges", type=Path, help="saved native ranges.bin")
    source.add_argument(
        "--geometry", type=Path, action="append",
        help="existing geometry JSON; repeat to overlay compatible saved series",
    )
    parser.add_argument("--output", type=Path, required=True, help="output path prefix")
    parser.add_argument("--label", default="native")
    parser.add_argument("--benchmark", default="unknown")
    parser.add_argument("--instance-id")
    parser.add_argument("--coordinate-names", default="")
    parser.add_argument("--projection", default="t,theta1")
    parser.add_argument("--view", choices=("tube", "endpoint"), default="tube")
    parser.add_argument("--step-size", type=float)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--expected-lanes", type=int)
    parser.add_argument("--display-steps")
    parser.add_argument("--partial-policy", choices=("reject", "mark"), default="reject")
    parser.add_argument("--spec", type=Path, help="v1 informational plot spec; no source digests")
    parser.add_argument("--run-config", type=Path, help="saved JSON run configuration, copied as an unbound declaration")
    parser.add_argument("--origin-path", help="original source path when ranges.bin was copied; declaration only")
    return parser


def main(argv: list[str] | None = None) -> int:
    total_started = time.perf_counter()
    args = _parser().parse_args(argv)
    source_started = time.perf_counter()
    if args.ranges:
        if args.ranges.name != "ranges.bin" or not args.ranges.is_file():
            raise ValueError("--ranges must name an existing ranges.bin file")
        if args.origin_path:
            plot._plain_text(args.origin_path, "origin path")
        geometry = _native_geometry(args)
        mode = "native-source"
        geometry_path = args.output.with_suffix(".geometry.json")
        plot._json_dump(geometry_path, geometry)
    else:
        if args.spec or args.run_config or args.origin_path:
            raise ValueError("geometry redraw uses the plot spec and run configuration saved in geometry")
        if len(args.geometry) == 1:
            geometry = _saved_geometry(args.geometry[0])
            geometry_path = args.geometry[0]
            mode = "geometry-redraw"
        else:
            geometry_path = args.output.with_suffix(".geometry.json")
            if geometry_path.resolve() in {path.resolve() for path in args.geometry}:
                raise ValueError("overlay output geometry must not replace an input geometry")
            geometry = _overlay_geometry(args.geometry)
            plot._json_dump(geometry_path, geometry)
            mode = "geometry-overlay"
    source_seconds = time.perf_counter() - source_started
    matlab = args.output.with_suffix(".m")
    matlab_started = time.perf_counter()
    plot.write_matlab(geometry, matlab)
    matlab_seconds = time.perf_counter() - matlab_started
    render_started = time.perf_counter()
    png, pdf = plot.render_matplotlib(geometry, args.output)
    render_seconds = time.perf_counter() - render_started
    receipt = {
        "schema": "torch-tm-flowpipe-nohash-render-v1",
        "mode": mode,
        "content_digest_policy": "none computed by this entry point",
        "source_geometry": str(geometry_path.resolve()),
        "input_geometries": (
            [str(path.resolve()) for path in args.geometry] if args.geometry else []
        ),
        "source_ranges": (
            str(args.ranges.resolve()) if args.ranges else
            [item.get("source_path") for item in geometry["series"]]
        ),
        "declared_origin_path": args.origin_path if args.ranges else None,
        "plot_configuration": {
            "benchmark": geometry["benchmark"],
            "instance_id": geometry.get("instance_id"),
            "projection": geometry["projection"],
            "view": geometry["view"],
            "step_size": geometry["step_size"],
            "expected_steps": geometry["expected_steps"],
            "expected_lanes": args.expected_lanes if args.ranges else None,
            "partial_policy": geometry["partial_policy"],
            "spec_path": str(args.spec.resolve()) if args.spec else None,
            "run_config_path": str(args.run_config.resolve()) if args.run_config else None,
        },
        "artifacts": {name: _artifact(path) for name, path in (
            ("geometry", geometry_path), ("matlab", matlab), ("png", png), ("pdf", pdf)
        )},
        "timings_seconds": {
            "source_read_and_projection_export_or_geometry_validation": source_seconds,
            "matlab_script_generation": matlab_seconds,
            "matplotlib_png_pdf_render": render_seconds,
            "total_before_receipt_write": time.perf_counter() - total_started,
        },
        "solver_timing": "not measured by this plotting entry point",
        "run_identity_check": "not performed; source paths and adjacent run files are unbound",
        "native_record_warning": (
            "ranges.bin has no accepted/status field; complete means record coverage only, "
            "not solver acceptance or certification"
        ),
        "matlab_runtime_check": "not performed by generator",
    }
    plot._json_dump(args.output.with_suffix(".render.json"), receipt)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"flowpipe-plot-nohash: {error}") from error
