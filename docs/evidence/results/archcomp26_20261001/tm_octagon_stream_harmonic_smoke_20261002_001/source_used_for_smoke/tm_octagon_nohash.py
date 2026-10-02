"""Eight-direction state-state plots from accepted scalar Taylor models.

The interval supports are the data. Polygon vertices are display-only floating
point intersections of their half-planes and are not a numerical certificate.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from .tm_vector import TMVector


def directional_intervals(tm: TMVector, x: int, y: int) -> dict[str, list[float]]:
    """Range x, y, x+y, x-y as TMs before losing shared symbols."""
    if x == y or min(x, y) < 0 or max(x, y) >= len(tm):
        raise ValueError("projection needs two different state indices")
    expressions = {"x": tm[x], "y": tm[y], "x_plus_y": tm[x] + tm[y],
                   "x_minus_y": tm[x] - tm[y]}
    out = {name: list(model.range_box().to_tuple()) for name, model in expressions.items()}
    if any(not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi)
           for lo, hi in out.values()):
        raise FloatingPointError("Taylor-model directional range is not finite and ordered")
    return out


def display_polygon(bounds: dict[str, list[float]]) -> list[list[float]]:
    """Clip the support box for drawing; supports, not vertices, are authoritative."""
    xlo, xhi = bounds["x"]
    ylo, yhi = bounds["y"]
    vertices = [[xlo, ylo], [xhi, ylo], [xhi, yhi], [xlo, yhi]]
    for name, nx, ny in (("x_plus_y", 1, 1), ("x_minus_y", 1, -1)):
        lo, hi = bounds[name]
        for a, b, limit in ((nx, ny, hi), (-nx, -ny, -lo)):
            if not vertices:
                raise ValueError("inconsistent directional intervals")
            clipped = []
            for start, end in zip(vertices, vertices[1:] + vertices[:1]):
                v0 = a * start[0] + b * start[1] - limit
                v1 = a * end[0] + b * end[1] - limit
                if (v0 <= 0) != (v1 <= 0):
                    weight = v0 / (v0 - v1)
                    clipped.append([start[0] + weight * (end[0] - start[0]),
                                    start[1] + weight * (end[1] - start[1])])
                if v1 <= 0:
                    clipped.append(end)
            vertices = clipped
    if not vertices:
        raise ValueError("inconsistent directional intervals")
    return vertices


def _segment_frame(segment, step: int, t_start: float, x: int, y: int) -> dict:
    if segment.status != "validated" or not math.isfinite(segment.h) or segment.h <= 0:
        raise ValueError("eight-direction observer requires an accepted positive-length segment")
    if segment.tm is None or segment.final_tm is None:
        raise ValueError("accepted segment lacks tube or propagated endpoint Taylor models")
    tube = directional_intervals(segment.tm, x, y)
    endpoint = directional_intervals(segment.final_tm, x, y)
    return {"step": step, "t_start": t_start, "t_end": t_start + segment.h,
            "status": "validated", "tube_supports": tube,
            "endpoint_supports": endpoint,
            "tube_polygon_display_only": display_polygon(tube),
            "endpoint_polygon_display_only": display_polygon(endpoint)}


class OctagonJSONLObserver:
    """Write one accepted CPU Taylor-model segment immediately after each step.

    Pass an instance as ``flowpipe_multi_step(accepted_segment_observer=...)``.
    A failed segment produces no row; any observation error stops the solver.
    """

    def __init__(self, path: Path, initial_box, *, x: int = 0, y: int = 1,
                 names: tuple[str, str] = ("x1", "x2"), source_label: str = "CPU Taylor-model flowpipe"):
        self.path = Path(path)
        initial = [list(map(float, item)) for item in initial_box]
        if x == y or min(x, y) < 0 or len(initial) <= max(x, y) or any(
            len(item) != 2 or item[0] > item[1] or not all(map(math.isfinite, item))
            for item in initial
        ):
            raise ValueError("initial box must give finite ordered bounds for two different states")
        self.x, self.y = x, y
        self.step, self.time = 0, 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8")
        header = {"schema": "torch-tm-flowpipe-octagon-stream-nohash-v1",
                  "source_label": source_label,
                  "projection": {"x": names[0], "y": names[1], "x_index": x, "y_index": y},
                  "initial_box": initial,
                  "endpoint_tm_semantics": "accepted_final_tm_used_for_propagation",
                  "content_digest_policy": "none computed"}
        self.stream.write(json.dumps(header, allow_nan=False) + "\n")
        self.stream.flush()

    def __call__(self, step: int, segment) -> None:
        if self.stream.closed or step != self.step + 1:
            raise ValueError("observer is closed or accepted step sequence is not contiguous")
        frame = _segment_frame(segment, step, self.time, self.x, self.y)
        line = json.dumps(frame, allow_nan=False) + "\n"
        offset = self.stream.tell()
        try:
            if self.stream.write(line) != len(line):
                raise OSError("short octagon stream write")
            self.stream.flush()
        except BaseException:
            self.stream.seek(offset)
            self.stream.truncate()
            self.stream.flush()
            raise
        self.step, self.time = step, frame["t_end"]

    def close(self) -> None:
        self.stream.close()

    def __enter__(self) -> "OctagonJSONLObserver":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def geometry_from_stream(path: Path) -> dict:
    """Load complete JSONL rows as an accepted-prefix geometry for rendering."""
    with Path(path).open(encoding="utf-8") as stream:
        lines = stream.readlines()
    if not lines or any(not line.endswith("\n") for line in lines):
        raise ValueError("empty or truncated octagon stream")
    header, *frames = [json.loads(line) for line in lines]
    if header.get("schema") != "torch-tm-flowpipe-octagon-stream-nohash-v1" or not frames:
        raise ValueError("unsupported or header-only octagon stream")
    t = 0.0
    for step, frame in enumerate(frames, 1):
        if (frame.get("status") != "validated" or frame.get("step") != step
                or frame.get("t_start") != t or not isinstance(frame.get("t_end"), (int, float))
                or not math.isfinite(frame["t_end"]) or frame["t_end"] <= t):
            raise ValueError("octagon stream has a missing or invalid accepted step")
        for view in ("tube", "endpoint"):
            bounds = frame[f"{view}_supports"]
            if set(bounds) != {"x", "y", "x_plus_y", "x_minus_y"} or any(
                len(pair) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v)
                                         for v in pair) or pair[0] > pair[1]
                for pair in bounds.values()
            ):
                raise ValueError("octagon stream has invalid directional supports")
            frame[f"{view}_polygon_display_only"] = display_polygon(bounds)
        t = frame["t_end"]
    return {"schema": "torch-tm-flowpipe-octagon-nohash-v1",
            "source_label": header["source_label"], "projection": header["projection"],
            "geometry_kind": "eight directional interval supports from shared Taylor models",
            "endpoint_tm_semantics": header["endpoint_tm_semantics"],
            "initial_box": header["initial_box"], "frames": frames,
            "numerical_horizon": t, "accepted_steps": len(frames),
            "property_status": "not_checked_by_plot_export",
            "certificate_scope": "accepted prefix only; source solver validation, no independent NNCS certificate",
            "display_vertices": "floating-point half-plane intersections for drawing only",
            "content_digest_policy": "none computed"}


def export_segments(segments, initial_box, output: Path, *, x: int = 0, y: int = 1,
                    names: tuple[str, str] = ("x1", "x2"),
                    source_label: str = "CPU Taylor-model flowpipe",
                    view: str = "tube") -> dict:
    """Export a validated flowpipe's tube/accepted-endpoint octagons and plots.

    Call directly after solving, while each accepted segment still owns its TM.
    No saved axis-aligned observer can be promoted to an octagon by this entry.
    """
    output = Path(output)
    if output.with_suffix(".geometry.json").exists():
        raise FileExistsError(output.with_suffix(".geometry.json"))
    initial = [list(map(float, item)) for item in initial_box]
    if len(initial) <= max(x, y) or any(len(item) != 2 or item[0] > item[1]
                                         or not all(map(math.isfinite, item)) for item in initial):
        raise ValueError("initial box must provide finite ordered bounds for both states")
    segments = list(segments)
    if not segments or any(seg.status != "validated" or not math.isfinite(seg.h) or seg.h <= 0
                           for seg in segments):
        raise ValueError("only a nonempty sequence of validated segments may be exported")
    observation_started = time.perf_counter()
    frames = []
    t = 0.0
    for index, segment in enumerate(segments, 1):
        frame = _segment_frame(segment, index, t, x, y)
        frames.append(frame)
        t = frame["t_end"]
    observation_s = time.perf_counter() - observation_started
    geometry = {
        "schema": "torch-tm-flowpipe-octagon-nohash-v1",
        "source_label": source_label,
        "projection": {"x": names[0], "y": names[1], "x_index": x, "y_index": y},
        "geometry_kind": "eight directional interval supports from shared Taylor models",
        "endpoint_tm_semantics": "accepted_final_tm_used_for_propagation",
        "initial_box": initial,
        "frames": frames,
        "numerical_horizon": t,
        "accepted_steps": len(frames),
        "property_status": "not_checked_by_plot_export",
        "certificate_scope": "source solver validation only; no independent end-to-end NNCS certificate",
        "display_vertices": "floating-point half-plane intersections for drawing only",
        "timings_seconds": {"directional_observation_and_geometry": observation_s},
        "content_digest_policy": "none computed",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".geometry.json").write_text(json.dumps(geometry, indent=2, allow_nan=False) + "\n")
    render_geometry(geometry, output, view=view,
                    geometry_path=output.with_suffix(".geometry.json"))
    return geometry


def render_geometry(geometry: dict, output: Path, *, view: str = "tube",
                    geometry_path: Path | None = None) -> None:
    """Redraw saved directional supports without running the solver again."""
    if view not in {"tube", "endpoint"}:
        raise ValueError("view must be tube or endpoint")
    if geometry.get("schema") != "torch-tm-flowpipe-octagon-nohash-v1":
        raise ValueError("unsupported directional geometry")
    output = Path(output)
    if any(output.with_suffix(suffix).exists() for suffix in (".m", ".png", ".pdf", ".render.json")):
        raise FileExistsError("plot output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    timings = {}
    started = time.perf_counter()
    _write_matlab(geometry, output.with_suffix(".m"), view)
    timings["matlab_script_generation"] = time.perf_counter() - started
    started = time.perf_counter()
    _render_matplotlib(geometry, output, view)
    timings["matplotlib_png_pdf_render"] = time.perf_counter() - started
    output.with_suffix(".render.json").write_text(json.dumps({
        "geometry": str(geometry_path) if geometry_path else "in-memory accepted segment TMs",
        "view": view,
        "matlab": str(output.with_suffix(".m")),
        "png": str(output.with_suffix(".png")),
        "pdf": str(output.with_suffix(".pdf")),
        "timings_seconds": {**geometry.get("timings_seconds", {}), **timings},
        "matlab_runtime_check": "not_performed",
        "content_digest_policy": "none computed",
    }, indent=2) + "\n")


def _write_matlab(geometry: dict, path: Path, view: str) -> None:
    lines = ["% Eight-direction Taylor-model supports; vertices are display-only.",
             "% Generated script; MATLAB execution not established by this file.",
             "figure('Color','w'); hold on; box on; grid on;"]
    x, y = geometry["projection"]["x"], geometry["projection"]["y"]
    for frame in geometry["frames"]:
        vertices = frame[f"{view}_polygon_display_only"]
        xs = " ".join(f"{point[0]:.17g}" for point in vertices)
        ys = " ".join(f"{point[1]:.17g}" for point in vertices)
        lines.append(f"patch([{xs}], [{ys}], [0.12 0.47 0.71], 'FaceAlpha', 0.12, "
                     "'EdgeColor', [0.12 0.47 0.71]);")
    ix, iy = geometry["projection"]["x_index"], geometry["projection"]["y_index"]
    xlo, xhi = geometry["initial_box"][ix]
    ylo, yhi = geometry["initial_box"][iy]
    lines.append(f"plot([{xlo:.17g} {xhi:.17g} {xhi:.17g} {xlo:.17g} {xlo:.17g}], "
                 f"[{ylo:.17g} {ylo:.17g} {yhi:.17g} {yhi:.17g} {ylo:.17g}], "
                 "'k-', 'LineWidth', 1.3);")
    lines.extend([f"xlabel('{x}');", f"ylabel('{y}');",
                  f"title('Taylor-model eight-direction {view} projection');",
                  "axis tight;"])
    path.write_text("\n".join(lines) + "\n")


def _render_matplotlib(geometry: dict, output: Path, view: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    fig, axis = plt.subplots(figsize=(8, 6))
    for frame in geometry["frames"]:
        axis.add_patch(Polygon(frame[f"{view}_polygon_display_only"], closed=True,
                               facecolor="#1f77b4", edgecolor="#1f77b4", alpha=.18))
        xlo, xhi = frame[f"{view}_supports"]["x"]
        ylo, yhi = frame[f"{view}_supports"]["y"]
        axis.add_patch(Rectangle((xlo, ylo), xhi - xlo, yhi - ylo,
                                 fill=False, edgecolor="#888888", linestyle=":", linewidth=.7))
    ix, iy = geometry["projection"]["x_index"], geometry["projection"]["y_index"]
    xlo, xhi = geometry["initial_box"][ix]
    ylo, yhi = geometry["initial_box"][iy]
    axis.add_patch(Rectangle((xlo, ylo), xhi - xlo, yhi - ylo,
                             fill=False, edgecolor="black", linewidth=1.3))
    axis.set_xlabel(geometry["projection"]["x"])
    axis.set_ylabel(geometry["projection"]["y"])
    axis.set_title(geometry["source_label"] + f": TM octagon {view}")
    axis.grid(alpha=.2)
    axis.autoscale()
    fig.text(.01, .01, f"Blue: 8-direction TM {view}; dotted: interval box; black: initial box. "
             "Vertices are for display; supports are the saved bounds.", fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smoke-harmonic", action="store_true")
    source.add_argument("--geometry", type=Path)
    source.add_argument("--stream", type=Path)
    parser.add_argument("--view", choices=("tube", "endpoint"), default="tube")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stream:
        geometry = geometry_from_stream(args.stream)
        geometry_path = args.output.with_suffix(".geometry.json")
        if geometry_path.exists():
            raise FileExistsError(geometry_path)
        geometry_path.parent.mkdir(parents=True, exist_ok=True)
        geometry_path.write_text(json.dumps(geometry, indent=2, allow_nan=False) + "\n")
        render_geometry(geometry, args.output, view=args.view, geometry_path=geometry_path)
        return 0
    if args.geometry:
        render_geometry(json.loads(args.geometry.read_text()), args.output,
                        view=args.view, geometry_path=args.geometry)
        return 0
    from . import Interval, flowpipe_multi_step
    from .ode_examples import harmonic_oscillator_ode

    solver_start = time.perf_counter()
    result = flowpipe_multi_step(harmonic_oscillator_ode,
                                 [Interval(1.0, 1.2), Interval(0.0, 0.0)],
                                 h=.05, steps=3, order=4, mode="dependency_preserving")
    solver_s = time.perf_counter() - solver_start
    if result.status != "validated" or len(result.segments) != 3:
        raise RuntimeError(f"short harmonic smoke stopped at {len(result.segments)}: {result.status}")
    geometry = export_segments(result.segments, [[1, 1.2], [0, 0]], args.output,
                               source_label="Harmonic oscillator plant-only 3-step smoke",
                               view=args.view)
    render_receipt = json.loads(args.output.with_suffix(".render.json").read_text())
    args.output.with_suffix(".run.json").write_text(json.dumps({
        "status": result.status, "accepted_steps": len(result.segments),
        "h": .05, "order": 4, "mode": "dependency_preserving",
        "ode": ["x1'=x2", "x2'=-x1"], "initial_box": [[1, 1.2], [0, 0]],
        "solver_wall_s": solver_s,
        "observation_wall_s": geometry["timings_seconds"]["directional_observation_and_geometry"],
        "render_wall_s": render_receipt["timings_seconds"]["matplotlib_png_pdf_render"],
        "scope": "plant-only drawing smoke; not an ARCH-COMP26 NNCS comparison",
        "content_digest_policy": "none computed",
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
