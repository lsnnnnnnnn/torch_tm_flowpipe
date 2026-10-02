"""Finite exact-solution and file-structure audit of the native octagon smoke."""
import csv
import json
import math
import sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("usage: check_native_flowstar_octagon_probe_nohash.py run-directory")
here = Path(sys.argv[1]).resolve()
with (here / "directional.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 24:
    raise AssertionError(f"expected 24 direction rows, got {len(rows)}")
bounds = {}
for row in rows:
    key = (int(row["step"]), int(row["view"]), int(row["form"]))
    lo, hi = float(row["lo"]), float(row["hi"])
    if key in bounds or not (1 <= key[0] <= 3 and key[1] in (0, 1)
                             and 0 <= key[2] <= 3 and math.isfinite(lo)
                             and math.isfinite(hi) and lo <= hi):
        raise AssertionError(f"invalid or repeated directional row: {row}")
    bounds[key] = (lo, hi)

checks = 0
min_margin = math.inf
first_failure = None
failure_count = 0
failures_by_view = {"tube": 0, "propagated_endpoint": 0}
first_failure_by_step_view = {}
for step in range(1, 4):
    for view in (0, 1):
        times = [step * 0.05] if view else [(step - 1 + k / 4) * 0.05 for k in range(5)]
        for t in times:
            for i in range(21):
                x0 = 1 + i / 100
                x, y = x0 * math.cos(t), -x0 * math.sin(t)
                for form, value in enumerate((x, y, x + y, x - y)):
                    lo, hi = bounds[(step, view, form)]
                    if not lo <= value <= hi:
                        failure_count += 1
                        failures_by_view["tube" if view == 0 else "propagated_endpoint"] += 1
                        if first_failure is None:
                            first_failure = {"step": step, "view": "tube" if view == 0 else "propagated_endpoint",
                                             "form": form, "x0": x0, "t": t,
                                             "exact_value": value, "saved_lo": lo, "saved_hi": hi}
                        first_failure_by_step_view.setdefault(f"{step}:{view}",
                            {"step": step, "view": "tube" if view == 0 else "propagated_endpoint",
                             "form": form, "x0": x0, "t": t,
                             "exact_value": value, "saved_lo": lo, "saved_hi": hi})
                    min_margin = min(min_margin, value - lo, hi - value)
                    checks += 1

diagonal_ratios = []
for step in range(1, 4):
    for view in (0, 1):
        xlo, xhi = bounds[(step, view, 0)]
        ylo, yhi = bounds[(step, view, 1)]
        for form in (2, 3):
            lo, hi = bounds[(step, view, form)]
            naive_width = (xhi - xlo) + (yhi - ylo)
            ratio = (hi - lo) / naive_width
            diagonal_ratios.append({"step": step, "view": "tube" if view == 0 else "propagated_endpoint",
                                    "form": "x+y" if form == 2 else "x-y", "width_ratio_to_axis_sum": ratio})

native_plot_count = (here / "native_reference.m").read_text().count("plot( ")
if native_plot_count != 3:
    raise AssertionError(f"expected three native Flow* MATLAB plot calls, got {native_plot_count}")
result = {"status": "pass" if not failure_count else "fail", "accepted_steps": 3, "direction_rows": len(rows),
          "native_plot_segments": native_plot_count, "exact_solution_direction_checks": checks,
          "minimum_sample_margin": min_margin, "sample_failures": failure_count,
          "sample_failures_by_view": failures_by_view,
          "first_sample_failure": first_failure,
          "first_failure_by_step_view": first_failure_by_step_view,
          "diagonal_ratios": diagonal_ratios,
          "qualification": "finite samples and structural cross-check only; no neural controller or independent certificate",
          "content_digest_policy": "none computed"}
(here / "AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"status": result["status"], "checks": checks,
                  "failures": failure_count, "failures_by_view": failures_by_view,
                  "first_failure": first_failure,
                  "third_endpoint_xy_ratio": diagonal_ratios[-2]["width_ratio_to_axis_sum"]}))
if failure_count:
    raise SystemExit(1)
