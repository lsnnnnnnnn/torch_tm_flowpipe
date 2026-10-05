#!/usr/bin/env python3
"""New official u=11f TORA tanh run: only lower plant cutoff to 1e-8.

Reuse the frozen expansion wrapper's private/Horner/fused1 installation, full
500-step observer, strict path and first refusal. Compare saved ranges; never
execute a historical checker. Target observation is not a property verdict.
"""
import argparse
import csv
from fractions import Fraction as F
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys


TARGET = {"x1": [-0.1, 0.2], "x2": [-0.9, -0.6]}
TARGET_SOURCE = "docs/ARCHCOMP26_TORA_REACH_EXECUTION_GATE_20261002.md"
RECORD = struct.Struct("<QQd16d")


def range_widths(reference, data):
    """Read both saved objects; classify exact binary64 endpoint differences."""
    raw = [(p / "ranges.bin").read_bytes() for p in (reference, data)]
    if any(len(value) != 500*RECORD.size for value in raw):
        raise RuntimeError("complete 500 x 152-byte four-state records required")
    records = [list(RECORD.iter_unpack(value)) for value in raw]
    summaries, widths = {}, {}
    for state in range(1, 5):
        for geometry in ("tube", "endpoint"):
            key = (state, geometry)
            summaries[key] = dict(state=f"x{state}", geometry=geometry, steps=500,
                                  narrower=0, equal=0, wider=0, subset=0)
            widths[key] = ([], [])
    target_steps, terminal = [], None
    fields = ["step", "nominal_t", "state", "geometry", "old_lower", "old_upper",
              "new_lower", "new_upper", "old_width", "new_width", "width_difference",
              "width_difference_rational", "classification", "candidate_subset_reference"]
    destination = data.parent / "ABSOLUTE_WIDTHS.csv"
    with destination.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step, (old, new) in enumerate(zip(*records), 1):
            if old[:3] != (0, step, .01) or new[:3] != old[:3]:
                raise RuntimeError(f"saved lane/time index differs at step {step}")
            if not all(math.isfinite(v) for v in (*old[3:], *new[3:])):
                raise RuntimeError(f"nonfinite saved bound at step {step}")
            for state in range(1, 5):
                for geometry, offset in (("tube", 0), ("endpoint", 2)):
                    j = 3+4*(state-1)+offset
                    al, ah, bl, bh = map(F, (*old[j:j+2], *new[j:j+2]))
                    if al > ah or bl > bh:
                        raise RuntimeError(f"unordered {geometry} at step {step}")
                    aw, bw = ah-al, bh-bl
                    relation = "narrower" if bw < aw else "wider" if bw > aw else "equal"
                    subset = al <= bl <= bh <= ah
                    key = state, geometry
                    summaries[key][relation] += 1
                    summaries[key]["subset"] += int(subset)
                    widths[key][0].append(aw)
                    widths[key][1].append(bw)
                    writer.writerow(dict(zip(fields, [step, step*.01, f"x{state}", geometry,
                        float(al), float(ah), float(bl), float(bh), float(aw), float(bw),
                        float(bw-aw), str(bw-aw), relation, subset])))
            terminal = {f"x{i+1}": list(new[5+4*i:7+4*i]) for i in range(4)}
            if all(lo <= terminal[key][0] <= terminal[key][1] <= hi
                   for key, (lo, hi) in TARGET.items()):
                target_steps.append(step)
    for key, row in summaries.items():
        old, new = widths[key]
        row.update(old_final_width=float(old[-1]), new_final_width=float(new[-1]),
                   final_width_ratio=float(new[-1]/old[-1]) if old[-1] else None,
                   old_max_width=float(max(old)), new_max_width=float(max(new)),
                   old_mean_width=float(sum(old)/500), new_mean_width=float(sum(new)/500))
    rows = list(summaries.values())
    return dict(full_state_full_step_widths=rows, absolute_widths_csv=str(destination),
                absolute_width_rows=4000, all_widths_nonincreasing=all(r["wider"] == 0 for r in rows),
                strictly_narrower_observations=sum(r["narrower"] for r in rows),
                subset_observations=sum(r["subset"] for r in rows),
                property_record=dict(payload_property_evaluated=False, new_property_checker_executed=False,
                    terminal_all_states=terminal, target_limits=TARGET, target_source=TARGET_SOURCE,
                    target_contained_saved_endpoint_steps=target_steps,
                    terminal_box_in_target=500 in target_steps,
                    basis="direct candidate endpoint-box observations; sufficient endpoint evidence only, no inherited label or independent NNCS proof"))


def compare_widths(reference, data, source_files):
    import yaml
    old_cfg, new_cfg = [yaml.safe_load((p / "config.yaml").read_text()) for p in (reference, data)]
    old_cutoff, new_cutoff = old_cfg.pop("cut_off_threshold"), new_cfg.pop("cut_off_threshold")
    old_model, new_model = old_cfg.pop("model_dir"), new_cfg.pop("model_dir")
    if old_cfg != new_cfg or (old_cutoff, new_cutoff) != (1e-6, 1e-8):
        raise RuntimeError("contract differs beyond the declared cutoff and model path")
    if Path(new_model) != data / "controller_plant_u.onnx":
        raise RuntimeError("new tanh model location differs")
    for name in source_files:
        if (reference / name).read_bytes() != (data / name).read_bytes():
            raise RuntimeError(f"frozen tanh source/model differs: {name}")
    old_receipt, new_receipt = [json.loads((p / "controller_plant_u.onnx.json").read_text())
                              for p in (reference, data)]
    old_root = str(Path(old_receipt["model_path"]).parent)
    if (new_receipt["activations"] != ["relu"]*3+["tanh"]
            or (new_receipt["scale"], new_receipt["offset"]) != (11, 0)
            or new_cfg["output_scale"] != 1 or new_cfg["output_offset"] != 0
            or old_model != old_receipt["model_path"]
            or old_receipt != json.loads((data / "controller_plant_u.onnx.json").read_text().replace(str(data), old_root))):
        raise RuntimeError("official tanh u=11f export or external scale differs")
    old_start, new_start = [json.loads((p / "START.json").read_text()) for p in (reference, data)]
    old_data = str(Path(old_start["driver_argv"][1]).parent)
    if old_start["driver_argv"] != [v.replace(str(data), old_data) for v in new_start["driver_argv"]]:
        raise RuntimeError("driver route changed beyond relocated output paths")
    observations = [(p / "observations.jsonl").read_bytes() for p in (reference, data)]
    if observations[0] != observations[1]:
        raise RuntimeError("acceptance/interval-valid observations changed")
    rows = [json.loads(line) for line in observations[1].splitlines()]
    if len(rows) != 500 or any(r != {"substep": i, "accepted": True, "interval_valid": True}
                               for i, r in enumerate(rows, 1)):
        raise RuntimeError("all 500 numerical observations must be accepted and valid")
    old_metrics, metrics = [json.loads((p / "metrics.json").read_text()) for p in (reference, data)]
    allowed = {"config", "elapsed_s", "peak_allocated_bytes", "peak_reserved_bytes",
               "cuda_graph_captures", "cuda_graph_hits", "ctrl_steps", "final_hull", "final_hull_width_sum_mean"}
    if any(old_metrics.get(k) != metrics.get(k) for k in (old_metrics.keys() | metrics.keys())-allowed):
        raise RuntimeError("method setting changed")
    if tuple(metrics[k] for k in ("B", "order", "steps", "substeps", "broken")) != (1, 3, 10, 50, 0):
        raise RuntimeError("complete working-order-three horizon differs")
    old_result, result = [json.loads((p / "RESULT.json").read_text()) for p in (reference, data)]
    old_result.pop("wall_s")
    result.pop("wall_s")
    if old_result != result or (result["status"] != "completed_full_numerical_horizon"
            or result["accepted_substeps"] != 500 or result["property_evaluated"] is not False):
        raise RuntimeError("full numerical result or property boundary changed")
    return dict(reference=str(reference), candidate=str(data), B=1, completed_substeps=500,
                changed_parameter={"cut_off_threshold": [1e-6, 1e-8]},
                contract_equal_except_cutoff_and_artifact_path=True,
                comparison="exact rational widths of saved binary64 bounds, all states and both geometries; no global containment or NNCS claim",
                old_checker_executed=False, digest_operations=0,
                end_to_end_floating_point_nn_certificate=False, **range_widths(reference, data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-wrapper", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not sys.dont_write_bytecode:
        raise RuntimeError("launch with python -B")
    base_path = args.base_wrapper.resolve()
    if not base_path.is_file() or base_path.name != "run_b1_candidate.py":
        parser.error("--base-wrapper must name the frozen expansion/run_b1_candidate.py")
    sys.path.insert(0, str(args.adapters.resolve()))
    import run_quad_candidate as utilities
    utilities.guard_digests()
    spec = importlib.util.spec_from_file_location("tanh_cutoff_expansion_wrapper", base_path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    original_stage, original_compare, original_write = baseline.stage, baseline.compare_saved, utilities.write_new
    outer_argv = list(sys.argv)

    def stage(instance, source, data):
        import yaml
        if instance != "tora-tanh":
            raise RuntimeError("this candidate requires official tanh")
        receipt = original_stage(instance, source, data)
        cfg = yaml.safe_load((data / "config.yaml").read_text())
        if cfg["cut_off_threshold"] != 1e-6:
            raise RuntimeError("expected frozen cutoff 1e-6")
        cfg["cut_off_threshold"] = 1e-8
        (data / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        receipt["new_numerical_parameter"] = {"cut_off_threshold": [1e-6, 1e-8]}
        return receipt

    def compare(instance, reference, data):
        if instance != "tora-tanh":
            raise RuntimeError("only official tanh source can be compared")
        return compare_widths(reference, data, baseline.SOURCE_FILES[instance])

    def write(path, value):
        if path.name in ("START.json", "RESULT.json"):
            value = dict(value, numerical_change={"cut_off_threshold": [1e-6, 1e-8]},
                         cutoff_outer_argv=outer_argv, base_wrapper=str(base_path))
        if path.name == "START.json":
            value["changes"] = value["changes"] + ["plant cutoff 1e-6 to 1e-8; order remains 3"]
            value["comparison_policy"] = "new complete tanh ranges versus saved original; no equality claim"
        elif path.name == "RESULT.json" and value.get("status") == "COMPLETED_SAVED_OUTPUT_EQUIVALENT":
            value.update(status="COMPLETED_NUMERICAL_WIDTH_CANDIDATE", saved_outputs_equal_claim=False)
        elif path.name == "SAVED_COMPARISON.json":
            path = path.with_name("WIDTH_COMPARISON.json")
        return original_write(path, value)

    baseline.stage, baseline.compare_saved, utilities.write_new = stage, compare, write
    sys.argv = [str(base_path), "--instance", "tora-tanh", "--weighted-mode", "fused",
                "--adapters", str(args.adapters.resolve()), "--gate", str(args.gate.resolve()),
                "--output", str(args.output.resolve())]
    for name in ("source_snapshot", "reference"):
        if getattr(args, name) is not None:
            sys.argv += ["--"+name.replace("_", "-"), str(getattr(args, name).resolve())]
    try:
        return baseline.main()
    finally:
        sys.argv = outer_argv
        baseline.stage, baseline.compare_saved, utilities.write_new = original_stage, original_compare, original_write


if __name__ == "__main__":
    raise SystemExit(main())
