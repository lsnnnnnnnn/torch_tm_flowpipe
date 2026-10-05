#!/usr/bin/env python3
"""New sigmoid working-order4 / point3 / validation5, cutoff1e-8 candidate.

Only a new staged copy is changed. The original benchmark, controller u=11f,
strict injection/endpoint, K+1 validation, 500-step observer, SR and fused1
implementation remain. This is a numerical-method change, not equivalence.
"""
import argparse
import ast
import json
from pathlib import Path
import sys

# Both official TORA reach contracts share this saved-endpoint target. This
# helper only parses geometry; it does not import the tanh model/comparator.
from run_tanh_cutoff_candidate import range_widths


EDITS = {
    "run_full.py": (
        ('Working-P3 full-box TORA reach-sigmoid', 'Working-order4 full-box TORA reach-sigmoid'),
        ('"ode_order": 3, "output_scale": 1', '"ode_order": 4, "output_scale": 1'),
        ('"method": "working-P3 plant engine,', '"method": "working-order4 plant engine,'),
        ('working_order=3, validation_order=4', 'working_order=4, validation_order=5'),
        ('"--strict", "--order", "3",', '"--strict", "--order", "4",'),
    ),
    "archcomp26_tora_remain_p3_nohash.py": (
        ('(engine.tables.n, engine.tables.k) != (6, 3)', '(engine.tables.n, engine.tables.k) != (6, 4)'),
        ('TORA P3 needs 6 variables and working order 3', 'TORA candidate needs 6 variables and working order 4'),
    ),
}


def revised(name, text):
    """Fail on source drift; change only the declared active-route literals."""
    for old, new in EDITS.get(name, ()):
        if text.count(old) != 1:
            raise RuntimeError(f"expected one declared source replacement in {name}: {old}")
        text = text.replace(old, new)
    ast.parse(text, filename=name)
    return text


def compare_widths(reference, data, source_files):
    import yaml
    old_cfg, new_cfg = [yaml.safe_load((p/"config.yaml").read_text()) for p in (reference,data)]
    changes = {key: [old_cfg.pop(key),new_cfg.pop(key)] for key in ("ode_order","cut_off_threshold")}
    if changes != {"ode_order": [3,4], "cut_off_threshold": [1e-6,1e-8]}:
        raise RuntimeError("expected only working3 to4 and cutoff1e-6 to1e-8")
    old_model, new_model = old_cfg.pop("model_dir"), new_cfg.pop("model_dir")
    if old_cfg != new_cfg or Path(new_model) != data/"controller_plant_u.onnx":
        raise RuntimeError("benchmark contract changed")
    for name in source_files:
        before, after = (reference/name).read_bytes(), (data/name).read_bytes()
        expected = revised(name,before.decode()).encode() if name in EDITS else before
        if after != expected:
            raise RuntimeError(f"staged source/model differs beyond declared edits: {name}")
    old_receipt, receipt = [json.loads((p/"controller_plant_u.onnx.json").read_text()) for p in (reference,data)]
    old_root = str(Path(old_receipt["model_path"]).parent)
    if (receipt["activations"] != ["sigmoid"]*4 or (receipt["scale"],receipt["offset"]) != (11,0)
            or (new_cfg["output_scale"],new_cfg["output_offset"]) != (1,0)
            or old_model != old_receipt["model_path"]
            or old_receipt != json.loads((data/"controller_plant_u.onnx.json").read_text().replace(str(data),old_root))):
        raise RuntimeError("official sigmoid u=11f model/receipt differs")
    old_start, start = [json.loads((p/"START.json").read_text()) for p in (reference,data)]
    old_argv = old_start["driver_argv"]
    new_argv = [v.replace(str(data),str(Path(old_argv[1]).parent)) for v in start["driver_argv"]]
    if old_argv.count("--order") != 1 or new_argv.count("--order") != 1:
        raise RuntimeError("exactly one working order argument required")
    index = new_argv.index("--order")+1
    if new_argv[index] != "4" or old_argv[old_argv.index("--order")+1] != "3":
        raise RuntimeError("driver working-order transition differs")
    new_argv[index] = "3"
    if old_argv != new_argv or (start["working_order"],start["validation_order"]) != (4,5):
        raise RuntimeError("driver route or declared K+1 order differs")
    observed = [(p/"observations.jsonl").read_bytes() for p in (reference,data)]
    if observed[0] != observed[1]:
        raise RuntimeError("accepted/valid observation sequence changed")
    rows = [json.loads(line) for line in observed[1].splitlines()]
    if len(rows) != 500 or any(r != dict(substep=i,accepted=True,interval_valid=True) for i,r in enumerate(rows,1)):
        raise RuntimeError("all 500 observations must be accepted and valid")
    old_metrics, metrics = [json.loads((p/"metrics.json").read_text()) for p in (reference,data)]
    allowed = {"config","order","elapsed_s","peak_allocated_bytes","peak_reserved_bytes",
               "cuda_graph_captures","cuda_graph_hits","ctrl_steps","final_hull","final_hull_width_sum_mean"}
    if any(old_metrics.get(k) != metrics.get(k) for k in (old_metrics.keys()|metrics.keys())-allowed):
        raise RuntimeError("unexpected method-setting change")
    if tuple(metrics[k] for k in ("B","order","steps","substeps","broken")) != (1,4,10,50,0):
        raise RuntimeError("complete working-order4 horizon differs")
    old_result, result = [json.loads((p/"RESULT.json").read_text()) for p in (reference,data)]
    old_result.pop("wall_s")
    result.pop("wall_s")
    if old_result != result or (result["status"] != "completed_full_numerical_horizon"
            or result["accepted_substeps"] != 500 or result["property_evaluated"] is not False):
        raise RuntimeError("numerical completion or property boundary differs")
    return dict(reference=str(reference),candidate=str(data),B=1,completed_substeps=500,
                changed_parameters=changes,working_order=4,point_order=3,validation_order=5,
                contract_equal_except_method_parameters_and_artifact_paths=True,
                comparison="exact rational widths of saved binary64 bounds; no old property label inherited",
                old_checker_executed=False,digest_operations=0,end_to_end_floating_point_nn_certificate=False,
                **range_widths(reference,data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters",type=Path,required=True)
    parser.add_argument("--source-snapshot",type=Path)
    parser.add_argument("--reference",type=Path)
    parser.add_argument("--gate",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if not sys.dont_write_bytecode:
        raise RuntimeError("launch with python -B")
    sys.path.insert(0,str(args.adapters.resolve()))
    import run_small1_candidate as baseline
    baseline.guard_digests()
    original_stage, original_compare, original_write = baseline.stage_tora,baseline.compare_saved,baseline.write_new
    outer_argv = list(sys.argv)

    def stage(source,data):
        import yaml
        receipt = original_stage(source,data)
        cfg = yaml.safe_load((data/"config.yaml").read_text())
        if (cfg["ode_order"],cfg["cut_off_threshold"]) != (3,1e-6):
            raise RuntimeError("stage from the original order3/cutoff1e-6 source")
        cfg.update(ode_order=4,cut_off_threshold=1e-8)
        (data/"config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
        for name in EDITS:
            path = data/name
            path.write_text(revised(name,path.read_text()))
        receipt["initial_byte_equal_copies"] = receipt.pop("byte_equal_copies")
        receipt["initial_path_only_relocation"] = receipt.pop("path_only_relocation")
        receipt["declared_staged_source_edits"] = {name:list(pairs) for name,pairs in EDITS.items()}
        receipt["new_numerical_parameters"] = {"ode_order":[3,4],"cut_off_threshold":[1e-6,1e-8]}
        receipt["orders"] = dict(working=4,point=3,validation=5,policy="solution_plus_one")
        return receipt

    def compare(instance,reference,data):
        if instance != "tora-sigmoid":
            raise RuntimeError("order4 candidate is only official sigmoid")
        return compare_widths(reference,data,baseline.TORA_FILES)

    def write(path,value):
        if path.name in ("START.json","RESULT.json"):
            value = dict(value,working_order=4,point_order=3,validation_order=5,
                         numerical_changes={"ode_order":[3,4],"cut_off_threshold":[1e-6,1e-8]},
                         order4_outer_argv=outer_argv,method_parameter_change=True,benchmark_dynamics_changed=False)
        if path.name == "START.json":
            value["changes"] += ["working order3 to4, point2 to3, strict K+1 validation4 to5", "cutoff1e-6 to1e-8"]
            value["comparison_policy"] = "new complete state/tube/endpoint widths; no equivalence or inherited property label"
        elif path.name == "RESULT.json" and value.get("status") == "COMPLETED_SAVED_OUTPUT_EQUIVALENT":
            value.update(status="COMPLETED_NUMERICAL_WIDTH_CANDIDATE",saved_outputs_equal_claim=False)
        elif path.name == "SAVED_COMPARISON.json":
            path = path.with_name("WIDTH_COMPARISON.json")
        return original_write(path,value)

    baseline.stage_tora,baseline.compare_saved,baseline.write_new = stage,compare,write
    sys.argv = [str(Path(baseline.__file__).resolve()),"--instance","tora-sigmoid","--weighted-mode","fused",
                "--gate",str(args.gate.resolve()),"--output",str(args.output.resolve())]
    for name in ("source_snapshot","reference"):
        if getattr(args,name) is not None:
            sys.argv += ["--"+name.replace("_","-"),str(getattr(args,name).resolve())]
    try:
        return baseline.main()
    finally:
        sys.argv = outer_argv
        baseline.stage_tora,baseline.compare_saved,baseline.write_new = original_stage,original_compare,original_write


if __name__ == "__main__":
    raise SystemExit(main())
