#!/usr/bin/env python3
"""Independent numerical falsification checks at the Flow* width minima.

These point and variational integrations are deliberately not advertised as an
enclosure proof.  They can produce a concrete containment counterexample, or
explain why a narrow projection is dynamically plausible, but cannot certify a
flowpipe for every point in the initial box.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import mpmath as mp
import numpy as np
import scipy
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.source_carry_audit import FLOWSTAR_CHANNELS, accepted_flowstar_rows


CHECKPOINTS = (3.97, 4.74, 6.32)
INITIAL_X = (1.1, 1.4)
INITIAL_Y = (2.35, 2.45)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rhs(_time: float, state: Sequence[float]) -> list[float]:
    x, y = state
    return [y, y - x - x * x * y]


def rhs_variational(_time: float, augmented: Sequence[float]) -> list[float]:
    x, y = augmented[:2]
    phi = np.asarray(augmented[2:], dtype=float).reshape(2, 2)
    jacobian = np.asarray([[0.0, 1.0], [-1.0 - 2.0 * x * y, 1.0 - x * x]])
    return [y, y - x - x * x * y, *(jacobian @ phi).ravel()]


def checkpoint_boxes(flow: Sequence[Mapping[str, str]]) -> dict[float, dict[str, str]]:
    rows = {round(float(row["t_after"]), 12): row for row in flow}
    output: dict[float, dict[str, str]] = {}
    for checkpoint in CHECKPOINTS:
        row = rows[checkpoint]
        x_lo, x_hi = FLOWSTAR_CHANNELS["endpoint_x"]
        y_lo, y_hi = FLOWSTAR_CHANNELS["endpoint_y"]
        output[checkpoint] = {
            "x_lo": row[x_lo],
            "x_hi": row[x_hi],
            "y_lo": row[y_lo],
            "y_hi": row[y_hi],
        }
    return output


def containment(value: float, lo: float, hi: float) -> tuple[bool, float]:
    return lo <= value <= hi, min(value - lo, hi - value)


def grid_values(lo: float, hi: float, count: int) -> Iterable[tuple[int, float]]:
    for index, value in enumerate(np.linspace(lo, hi, count)):
        yield index, float(value)


def scipy_grid_replay(
    boxes: Mapping[float, Mapping[str, str]], grid_size: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for ix, x0 in grid_values(*INITIAL_X, grid_size):
        for iy, y0 in grid_values(*INITIAL_Y, grid_size):
            result = solve_ivp(
                rhs,
                (0.0, CHECKPOINTS[-1]),
                (x0, y0),
                method="DOP853",
                t_eval=CHECKPOINTS,
                rtol=1e-12,
                atol=1e-14,
            )
            if not result.success or len(result.t) != len(CHECKPOINTS):
                raise RuntimeError(result.message)
            for checkpoint, state in zip(CHECKPOINTS, result.y.T, strict=True):
                box = boxes[checkpoint]
                for component, value in zip(("x", "y"), state, strict=True):
                    lo = float(box[f"{component}_lo"])
                    hi = float(box[f"{component}_hi"])
                    contained, signed_slack = containment(float(value), lo, hi)
                    output.append(
                        {
                            "solver": "scipy_solve_ivp_DOP853",
                            "grid_ix": ix,
                            "grid_iy": iy,
                            "initial_x": format(x0, ".17g"),
                            "initial_y": format(y0, ".17g"),
                            "time": format(checkpoint, ".17g"),
                            "component": component,
                            "value": format(float(value), ".17g"),
                            "flowstar_lo": box[f"{component}_lo"],
                            "flowstar_hi": box[f"{component}_hi"],
                            "contained": contained,
                            "signed_slack": format(signed_slack, ".17g"),
                        }
                    )
    return output


def variational_replay() -> list[dict[str, Any]]:
    initial = np.asarray(
        [(INITIAL_X[0] + INITIAL_X[1]) / 2, (INITIAL_Y[0] + INITIAL_Y[1]) / 2, 1, 0, 0, 1],
        dtype=float,
    )
    result = solve_ivp(
        rhs_variational,
        (0.0, CHECKPOINTS[-1]),
        initial,
        method="DOP853",
        t_eval=CHECKPOINTS,
        rtol=1e-12,
        atol=1e-14,
    )
    if not result.success:
        raise RuntimeError(result.message)
    rows: list[dict[str, Any]] = []
    for checkpoint, augmented in zip(CHECKPOINTS, result.y.T, strict=True):
        phi = augmented[2:].reshape(2, 2)
        singular_values = np.linalg.svd(phi, compute_uv=False)
        x_radius_bound = float(np.sum(np.abs(phi[0]) * np.asarray([0.15, 0.05])))
        y_radius_bound = float(np.sum(np.abs(phi[1]) * np.asarray([0.15, 0.05])))
        rows.append(
            {
                "time": format(checkpoint, ".17g"),
                "center_x": format(float(augmented[0]), ".17g"),
                "center_y": format(float(augmented[1]), ".17g"),
                "phi_00": format(float(phi[0, 0]), ".17g"),
                "phi_01": format(float(phi[0, 1]), ".17g"),
                "phi_10": format(float(phi[1, 0]), ".17g"),
                "phi_11": format(float(phi[1, 1]), ".17g"),
                "singular_value_max": format(float(singular_values[0]), ".17g"),
                "singular_value_min": format(float(singular_values[-1]), ".17g"),
                "determinant": format(float(np.linalg.det(phi)), ".17g"),
                "linearized_x_projection_radius": format(x_radius_bound, ".17g"),
                "linearized_y_projection_radius": format(y_radius_bound, ".17g"),
            }
        )
    return rows


def representative_initial_points() -> list[tuple[str, str, str]]:
    x_lo, x_hi = map(mp.mpf, ("1.1", "1.4"))
    y_lo, y_hi = map(mp.mpf, ("2.35", "2.45"))
    x_mid = (x_lo + x_hi) / 2
    y_mid = (y_lo + y_hi) / 2
    return [
        ("corner_ll", str(x_lo), str(y_lo)),
        ("edge_xmid_ylo", str(x_mid), str(y_lo)),
        ("corner_hl", str(x_hi), str(y_lo)),
        ("edge_xlo_ymid", str(x_lo), str(y_mid)),
        ("center", str(x_mid), str(y_mid)),
        ("edge_xhi_ymid", str(x_hi), str(y_mid)),
        ("corner_lh", str(x_lo), str(y_hi)),
        ("edge_xmid_yhi", str(x_mid), str(y_hi)),
        ("corner_hh", str(x_hi), str(y_hi)),
    ]


def mpmath_replay(boxes: Mapping[float, Mapping[str, str]]) -> list[dict[str, Any]]:
    mp.mp.dps = 70

    def mp_rhs(_time: mp.mpf, state: Sequence[mp.mpf]) -> tuple[mp.mpf, mp.mpf]:
        x, y = state
        return y, y - x - x * x * y

    output: list[dict[str, Any]] = []
    for name, x0_raw, y0_raw in representative_initial_points():
        solution = mp.odefun(
            mp_rhs,
            mp.mpf("0"),
            (mp.mpf(x0_raw), mp.mpf(y0_raw)),
            tol=mp.mpf("1e-45"),
            degree=40,
        )
        for checkpoint in CHECKPOINTS:
            state = solution(mp.mpf(str(checkpoint)))
            box = boxes[checkpoint]
            for component, value in zip(("x", "y"), state, strict=True):
                lo = mp.mpf(box[f"{component}_lo"])
                hi = mp.mpf(box[f"{component}_hi"])
                contained = lo <= value <= hi
                signed_slack = min(value - lo, hi - value)
                output.append(
                    {
                        "solver": "mpmath_taylor_odefun",
                        "precision_decimal_digits": mp.mp.dps,
                        "tolerance": "1e-45",
                        "degree": 40,
                        "point": name,
                        "initial_x": x0_raw,
                        "initial_y": y0_raw,
                        "time": str(checkpoint),
                        "component": component,
                        "value": mp.nstr(value, 60),
                        "flowstar_lo": box[f"{component}_lo"],
                        "flowstar_hi": box[f"{component}_hi"],
                        "contained": bool(contained),
                        "signed_slack": mp.nstr(signed_slack, 60),
                    }
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flowstar-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=9)
    args = parser.parse_args()
    if args.grid_size < 2:
        raise ValueError("grid size must be at least two")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = args.flowstar_trace.resolve()
    flow = accepted_flowstar_rows(read_csv(trace_path))
    boxes = checkpoint_boxes(flow)
    scipy_rows = scipy_grid_replay(boxes, args.grid_size)
    variational_rows = variational_replay()
    mp_rows = mpmath_replay(boxes)
    write_csv(args.output_dir / "sample_replay.csv", scipy_rows)
    write_csv(args.output_dir / "variational_replay.csv", variational_rows)
    write_csv(args.output_dir / "high_precision_replay.csv", mp_rows)

    scipy_violations = sum(row["contained"] is False for row in scipy_rows)
    mp_violations = sum(row["contained"] is False for row in mp_rows)
    summary = {
        "schema": "flowstar_width_numerical_falsification_v1",
        "scope": {
            "checkpoints": list(CHECKPOINTS),
            "reason": "endpoint-x minimum, endpoint-y minimum, and Torch terminal accepted prefix",
            "initial_box": {"x": list(INITIAL_X), "y": list(INITIAL_Y)},
        },
        "flowstar_trace": {"path": str(trace_path), "sha256": sha256(trace_path)},
        "scipy": {
            "version": scipy.__version__,
            "method": "DOP853",
            "rtol": 1e-12,
            "atol": 1e-14,
            "grid_size_per_axis": args.grid_size,
            "initial_points": args.grid_size * args.grid_size,
            "component_observations": len(scipy_rows),
            "containment_violations": scipy_violations,
        },
        "high_precision": {
            "library": "mpmath",
            "version": mp.__version__,
            "method": "Taylor-series odefun",
            "precision_decimal_digits": 70,
            "tolerance": "1e-45",
            "degree": 40,
            "representative_initial_points": len(representative_initial_points()),
            "component_observations": len(mp_rows),
            "containment_violations": mp_violations,
        },
        "variational": {
            "method": "DOP853 center trajectory plus 2x2 fundamental matrix",
            "observations": len(variational_rows),
            "interpretation": "Projection contraction in a row of the flow-map Jacobian can make a positive coordinate width visually narrow while the full map remains nonsingular.",
        },
        "falsification_result": (
            "NUMERICAL_CONTAINMENT_WITNESS_FOUND"
            if scipy_violations or mp_violations
            else "NO_NUMERICAL_CONTAINMENT_WITNESS_IN_TESTED_POINTS"
        ),
        "proof_status": "NOT_AN_ENCLOSURE_PROOF",
        "limitation": "Point samples and a center variational equation neither cover the continuum of initial states nor validate Flow* interval arithmetic.",
        "python": platform.python_version(),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
