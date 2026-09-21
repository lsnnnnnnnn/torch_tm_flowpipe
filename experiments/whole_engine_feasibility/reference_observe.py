"""Re-observe the four frozen B1 model archives; never rerun a solver."""
import argparse
import csv
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import time

from .baseline import ROOT, frozen_case
from .observe import observe_step


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def numeric(value):
    return float.fromhex(value) if isinstance(value, str) else float(value)


def layout(model):
    return {"variables": model["variables"], "domain_dimensions": len(model["domain"]),
            "domain_hex": [[numeric(v).hex() for v in pair] for pair in model["domain"]],
            "state_components": len(model["components"])}


def sources():
    runs = ROOT / "artifacts/runs"
    for plant, short in (("van_der_pol", "vdp"), ("brusselator", "brusselator")):
        yield f"cpu_{short}", plant, (runs / "endpoint_roundoff_repair_20260908/raw_minimal" / f"{short}_full")
        yield f"flowstar_{short}", plant, (runs / "xiangru_adoption_20260907T032448Z/raw_minimal" / f"native_{short}")


def comparison_widths(bounds_path):
    """Common 1..999 comparison and explicit step-1000 tail, using NEW bounds."""
    maxima, tail, prefix_rows = {}, [], 0
    with Path(bounds_path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["step"]) == 1000:
                tail.append({key: row[key] for key in ("view", "coordinate", "lo", "hi", "width",
                                                     "source_tube_covers_nominal_step",
                                                     "source_tube_time_lo_hex", "source_tube_time_hi_hex")})
            else:
                if row["source_tube_covers_nominal_step"] != "True":
                    raise ValueError("common first-999-step scope is not fully covered")
                prefix_rows += 1
                channel = f'{row["view"]}_{row["coordinate"]}'
                maxima[channel] = max(maxima.get(channel, 0.0), float(row["width"]))
    if prefix_rows != 3996 or len(tail) != 4:
        raise ValueError("expected 999 complete common steps plus one separately reported tail")
    return {"common_comparison_prefix_steps": 999, "common_prefix_max_widths": maxima,
            "separate_tail_step": 1000, "separate_tail_bounds": tail,
            "tail_comparison_rule": "Flow* tail tube has a shorter domain; its nominal endpoint is outside that tube domain and is not a validated matching-domain baseline"}


def reobserve(name, plant, source, output):
    case = frozen_case(plant, 1)
    h, steps = case["h"], case["requested_steps"]
    model_file, source_summary = source / "models.jsonl.gz", source / "summary.json"
    original = json.loads(source_summary.read_text())
    if original["accepted_steps"] != steps:
        raise ValueError(f"incomplete reference: {source}")
    source_h = original.get("fixed_step_hex", original.get("h"))
    if numeric(source_h) != h:
        raise ValueError(f"reference fixed step differs from frozen case: {source}")
    source_hash = digest(model_file)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    layouts, final_layouts, maxima, terminal = {}, {}, {}, {}
    domain_changes, scope_exceptions = [], []
    count = 0
    with gzip.open(model_file, "rt") as src, (output / "bounds.csv").open("w", newline="") as dst:
        writer = None
        for count, line in enumerate(src, start=1):
            record = json.loads(line)
            if record["step"] != count or count > steps:
                raise ValueError("reference model sequence is not the frozen contiguous horizon")
            if "h" in record and numeric(record["h"]) != h:
                raise ValueError("reference changed step size")
            for key, expected in (("t_start_exact", (count - 1) * Fraction(h)),
                                  ("t_end_exact", count * Fraction(h))):
                if key in record and Fraction(record[key]) != expected:
                    raise ValueError("reference model clock does not match the fixed binary64 clock")
            models = record["models"]
            for view in ("endpoint", "tube"):
                current_layout = layout(models[view])
                if view in layouts:
                    if (layouts[view]["variables"] != current_layout["variables"] or
                            layouts[view]["domain_dimensions"] != current_layout["domain_dimensions"]):
                        raise ValueError("reference changed variable order or dimension")
                    if final_layouts[view] != current_layout:
                        domain_changes.append({"step": count, "view": view, **current_layout})
                else:
                    layouts[view] = current_layout
                final_layouts[view] = current_layout
            time_index = models["tube"]["variables"].index("tau")
            tube_time = list(map(numeric, models["tube"]["domain"][time_index]))
            tube_covers_nominal = tube_time[0] <= 0.0 and tube_time[1] >= h
            if not tube_covers_nominal:
                scope_exceptions.append({"step": count, "source_tube_local_time_hex": [v.hex() for v in tube_time],
                                         "reason": "saved tube domain does not cover the complete nominal step; nominal endpoint also lacks matching tube-domain coverage"})
            rows = observe_step(models, plant=plant, step=count, h=h, lane=0)
            for row in rows:
                row["source_tube_covers_nominal_step"] = tube_covers_nominal
                row["source_tube_time_lo_hex"], row["source_tube_time_hi_hex"] = [v.hex() for v in tube_time]
            if writer is None:
                writer = csv.DictWriter(dst, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            for row in rows:
                channel = f'{row["view"]}_{row["coordinate"]}'
                maxima[channel] = max(maxima.get(channel, 0.0), row["width"])
                terminal[channel] = {key: row[key] for key in ("lo", "hi", "width")}
            if count % 100 == 0:
                print(json.dumps({"reference": name, "observed_steps": count,
                                  "elapsed_s": time.perf_counter() - start}), flush=True)
    if count != steps:
        raise ValueError(f"only {count} of {steps} reference models present")
    summary = {"schema": "whole-engine-reference-exact-observer/1", "reference": name,
               "plant": plant, "batch": 1, "original_unpartitioned_box": True,
               "completed": True, "accepted_steps": count, "row_count": count * 4,
               "h_hex": h.hex(), "final_time_exact": str(count * Fraction(h)),
               "observer": "experiments.xiangru_adoption.common.measure via observe_step",
               "scope": "physical endpoint and saved tube domain, x and y, every saved accepted step; exact rational evaluation of each complete polynomial plus interval remainder on its own legal coordinates; consult scope_exceptions before treating a row as a full nominal fixed-step comparison",
               "coordinate_comparison": "compare physical interval bounds only; variable order/domain dimension need not match across engines",
               "time_labels": "nominal fixed binary64-step clock; actual saved local tube domain is recorded per row",
               "full_nominal_step_scope_all_rows": not scope_exceptions,
               "full_nominal_step_scope_prefix": min((e["step"] - 1 for e in scope_exceptions), default=count),
               "scope_exceptions": scope_exceptions,
               "source_models": str(model_file), "source_models_sha256": source_hash,
               "source_summary": str(source_summary), "source_summary_sha256": digest(source_summary),
               "source_model_layouts_initial": layouts, "source_model_layouts_final": final_layouts,
               "source_domain_changes": domain_changes, "initial_exact_decimal_box": case["exact_boxes"][0],
               "max_widths": maxima, "terminal_bounds": terminal,
               "observation_seconds": time.perf_counter() - start,
               "solver_rerun": False, "legacy_bounds_csv_used": False,
               "observer_source_sha256": digest(ROOT / "experiments/xiangru_adoption/common.py"),
               "observer_wrapper_sha256": digest(Path(__file__).with_name("observe.py")),
               "script_sha256": digest(__file__), "bounds_sha256": digest(output / "bounds.csv")}
    summary.update(comparison_widths(output / "bounds.csv"))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("derived observation artifacts must be outside the frozen source checkout")
    output.mkdir(parents=True, exist_ok=False)
    summaries = [reobserve(name, plant, source, output / name) for name, plant, source in sources()]
    (output / "index.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"completed": True, "references": len(summaries),
                      "observed_steps": sum(s["accepted_steps"] for s in summaries),
                      "rows": sum(s["row_count"] for s in summaries), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
