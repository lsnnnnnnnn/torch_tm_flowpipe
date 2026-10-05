#!/usr/bin/env python3
"""New full sigmoid experiment: lower plant cutoff after qualified speed bindings.

Reuse the Oct 5 single-run wrapper. The only numerical-method change is the
cutoff; controller, initial set, step size, order, strict validation, observers
and refusal handling are unchanged. This never reruns a baseline or checker.
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys


def compare_widths(reference, data):
    import numpy as np
    import yaml
    old_cfg = yaml.safe_load((reference / "config.yaml").read_text())
    new_cfg = yaml.safe_load((data / "config.yaml").read_text())
    old_cutoff, new_cutoff = old_cfg.pop("cut_off_threshold"), new_cfg.pop("cut_off_threshold")
    old_model, new_model = old_cfg.pop("model_dir"), new_cfg.pop("model_dir")
    if old_cfg != new_cfg or old_cutoff != 1e-6 or new_cutoff not in (1e-8, 1e-10):
        raise RuntimeError("numerical contract changed beyond the declared cutoff")
    if Path(new_model) != data / "controller_plant_u.onnx":
        raise RuntimeError("new model location differs")
    if (reference / "controller_plant_u.onnx").read_bytes() != Path(new_model).read_bytes():
        raise RuntimeError("controller model changed")
    if (reference / "observations.jsonl").read_bytes() != (data / "observations.jsonl").read_bytes():
        raise RuntimeError("acceptance/interval-valid observation sequence changed")
    observations = [json.loads(line) for line in (data / "observations.jsonl").read_text().splitlines()]
    if len(observations) != 500 or any(not r["accepted"] or not r["interval_valid"] for r in observations):
        raise RuntimeError("500 valid accepted substeps are required")
    dtype = np.dtype([("lane", "<u8"), ("step", "<u8"), ("h", "<f8"), ("bounds", "<f8", (4, 4))])
    old, new = (np.fromfile(p / "ranges.bin", dtype=dtype) for p in (reference, data))
    if len(old) != 500 or len(new) != 500:
        raise RuntimeError("missing full saved range sequence")
    for name in ("lane", "step", "h"):
        if old[name].tobytes() != new[name].tobytes():
            raise RuntimeError(f"changed saved range index: {name}")
    if not np.isfinite(new["bounds"]).all():
        raise RuntimeError("nonfinite range")
    metrics = json.loads((data / "metrics.json").read_text())
    old_metrics = json.loads((reference / "metrics.json").read_text())
    changed_fields = {"config", "elapsed_s", "peak_allocated_bytes", "peak_reserved_bytes",
                      "cuda_graph_captures", "cuda_graph_hits", "ctrl_steps", "final_hull",
                      "final_hull_width_sum_mean"}
    for key in (set(metrics) | set(old_metrics)) - changed_fields:
        if metrics.get(key) != old_metrics.get(key):
            raise RuntimeError(f"method setting changed: {key}")
    payload = json.loads((data / "RESULT.json").read_text())
    if (metrics["B"], metrics["order"], metrics["steps"], metrics["substeps"], metrics["broken"]) != (1, 3, 10, 50, 0):
        raise RuntimeError("full working-order-three contract differs")
    if payload.get("status") != "completed_full_numerical_horizon" or payload.get("accepted_substeps") != 500:
        raise RuntimeError("incomplete numerical result")
    rows = []
    for state in range(4):
        for geom, j in (("tube", 0), ("endpoint", 2)):
            counts = dict(narrower=0, equal=0, wider=0, subset=0)
            widths = [[], []]
            for number, (a, b) in enumerate(zip(old["bounds"][:, state, j:j+2], new["bounds"][:, state, j:j+2]), 1):
                al, ah, bl, bh = map(lambda v: Fraction(float(v)), (*a, *b))
                if al > ah or bl > bh:
                    raise RuntimeError(f"unordered {geom} bound at step {number}")
                aw, bw = ah-al, bh-bl
                widths[0].append(aw)
                widths[1].append(bw)
                counts["narrower" if bw < aw else "wider" if bw > aw else "equal"] += 1
                counts["subset"] += al <= bl <= bh <= ah
            rows.append({"state": f"x{state+1}", "geometry": geom, "steps": 500, **counts,
                         "old_final_width": float(widths[0][-1]), "new_final_width": float(widths[1][-1]),
                         "final_width_ratio": float(widths[1][-1] / widths[0][-1]),
                         "old_max_width": float(max(widths[0])), "new_max_width": float(max(widths[1])),
                         "old_mean_width": float(sum(widths[0]) / 500), "new_mean_width": float(sum(widths[1]) / 500)})
    final_target = {f"x{i+1}": [float(v) for v in new["bounds"][-1, i, 2:4]] for i in range(2)}
    target_limits = {"x1": [-0.1, 0.2], "x2": [-0.9, -0.6]}
    target_box = all(target_limits[key][0] <= lo <= hi <= target_limits[key][1]
                     for key, (lo, hi) in final_target.items())
    return {"reference": str(reference), "candidate": str(data), "completed_substeps": 500,
            "B": 1, "changed_parameter": {"cut_off_threshold": [old_cutoff, new_cutoff]},
            "full_state_full_step_widths": rows, "all_widths_nonincreasing": all(r["wider"] == 0 for r in rows),
            "strictly_narrower_observations": sum(r["narrower"] for r in rows),
            "comparison": "exact rational differences of saved binary64 endpoints; comparisons remain per state and geometry",
            "contract_equal_except_cutoff_and_artifact_path": True,
            "property_record": {"payload_property_evaluated": False, "new_property_checker_executed": False,
                                "terminal_x1_x2": final_target, "target_limits": target_limits,
                                "terminal_box_in_target": target_box,
                                "basis": "direct saved endpoint box observation; no inherited old result or independent NNCS certificate"},
            "old_checker_executed": False, "digest_operations": 0,
            "end_to_end_floating_point_nn_certificate": False}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--plant-cutoff", type=float, choices=(1e-8, 1e-10), required=True)
    options, forwarded = parser.parse_known_args()
    if not sys.dont_write_bytecode:
        raise RuntimeError("launch with python -B")
    sys.path.insert(0, str(options.adapters.resolve()))
    import run_small1_candidate as baseline
    original_stage, original_write = baseline.stage_tora, baseline.write_new

    def stage(source, data):
        import yaml
        receipt = original_stage(source, data)
        cfg = yaml.safe_load((data / "config.yaml").read_text())
        if cfg["cut_off_threshold"] != 1e-6:
            raise RuntimeError("expected saved cutoff 1e-6")
        cfg["cut_off_threshold"] = options.plant_cutoff
        (data / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        receipt["new_numerical_parameter"] = {"cut_off_threshold": [1e-6, options.plant_cutoff]}
        return receipt

    def compare(instance, reference, data):
        if instance != "tora-sigmoid":
            raise RuntimeError("this candidate is only official TORA sigmoid u=11f")
        return compare_widths(reference, data)

    def write(path, value):
        if path.name == "START.json":
            value = dict(value, numerical_change={"cut_off_threshold": [1e-6, options.plant_cutoff]},
                         optimization_source="plant cutoff loses small coefficients; retain more at unchanged order/step/validation",
                         comparison_policy="new full numerical trajectory compared to saved old ranges; no equality claim")
        elif path.name == "RESULT.json" and value.get("status") == "COMPLETED_SAVED_OUTPUT_EQUIVALENT":
            value = dict(value, status="COMPLETED_NUMERICAL_WIDTH_CANDIDATE", saved_outputs_equal_claim=False)
        elif path.name == "SAVED_COMPARISON.json":
            path = path.with_name("WIDTH_COMPARISON.json")
        return original_write(path, value)

    baseline.stage_tora, baseline.compare_saved, baseline.write_new = stage, compare, write
    sys.argv = [sys.argv[0], "--instance", "tora-sigmoid", "--weighted-mode", "fused", *forwarded]
    return baseline.main()


if __name__ == "__main__":
    raise SystemExit(main())
