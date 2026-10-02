#!/usr/bin/env python3
"""Independent sampled-solution and geometry check for the three-step smoke."""
import json
import math
from pathlib import Path


ROOT = (Path(__file__).resolve().parents[1] / "docs/evidence/results/archcomp26_20261001"
        / "tm_octagon_harmonic_smoke_20261002_001")


def area(vertices):
    return abs(sum(a[0] * b[1] - b[0] * a[1]
                   for a, b in zip(vertices, vertices[1:] + vertices[:1]))) / 2


def contains(bounds, x, y, tol=1e-12):
    for key, value in (("x", x), ("y", y), ("x_plus_y", x + y), ("x_minus_y", x - y)):
        lo, hi = bounds[key]
        if not lo - tol <= value <= hi + tol:
            raise AssertionError(f"{key}: {value} outside [{lo}, {hi}]")


def main():
    geometry = json.loads((ROOT / "harmonic_3step.geometry.json").read_text())
    run = json.loads((ROOT / "harmonic_3step.run.json").read_text())
    assert run["status"] == "validated" and geometry["accepted_steps"] == 3
    assert geometry["content_digest_policy"] == "none computed"
    sample_cases = 0
    rows = []
    for step, frame in enumerate(geometry["frames"], 1):
        assert frame["step"] == step and frame["status"] == "validated"
        t0, t1 = frame["t_start"], frame["t_end"]
        assert abs(t0 - (step - 1) * .05) < 1e-14
        assert abs(t1 - step * .05) < 1e-14
        entry = {"step": step}
        for view in ("tube", "endpoint"):
            bounds = frame[f"{view}_supports"]
            vertices = frame[f"{view}_polygon_display_only"]
            assert len(vertices) >= 3
            assert all(math.isfinite(v) for pair in bounds.values() for v in pair)
            assert all(lo <= hi for lo, hi in bounds.values())
            for x, y in vertices:
                contains(bounds, x, y)
            box_area = (bounds["x"][1] - bounds["x"][0]) * (bounds["y"][1] - bounds["y"][0])
            polygon_area = area(vertices)
            assert 0 <= polygon_area <= box_area + 1e-12
            entry[view] = {"box_area": box_area, "display_polygon_area": polygon_area,
                           "display_area_ratio": polygon_area / box_area}
            times = [t0 + (t1 - t0) * j / 10 for j in range(11)] if view == "tube" else [t1]
            for t in times:
                for x0 in (1.0, 1.1, 1.2):
                    contains(bounds, x0 * math.cos(t), -x0 * math.sin(t))
                    sample_cases += 1
        rows.append(entry)
    assert rows[-1]["endpoint"]["display_area_ratio"] < .9
    report = {"status": "pass", "sampled_exact_solution_cases": sample_cases,
              "checks": "all four TM directional intervals contain exact oscillator samples; "
                        "display polygon vertices obey all saved half-planes",
              "qualification": "finite sampled check and floating-point drawing audit; not proof of all trajectories",
              "steps": rows, "content_digest_policy": "none computed"}
    (ROOT / "AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": sample_cases,
                      "last_endpoint_area_ratio": rows[-1]["endpoint"]["display_area_ratio"]}))


if __name__ == "__main__":
    main()
