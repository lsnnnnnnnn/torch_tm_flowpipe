#!/usr/bin/env python3
"""CPU-only static preflight for the fixed 2026 Airplane continuous box."""

import hashlib
import json
from pathlib import Path
import sys


def prohibited(*_args, **_kwargs):
    raise RuntimeError("content digest prohibited for this campaign")


hashlib.sha256 = prohibited
original_new = hashlib.new


def guarded_new(name, *args, **kwargs):
    if str(name).lower().replace("-", "") == "sha256":
        return prohibited()
    return original_new(name, *args, **kwargs)


hashlib.new = guarded_new

import onnx  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402


HERE = Path(__file__).resolve().parent
N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
PREP = N / "runs/archcomp26_20261001/airplane_prep_001"
ENGINE = Path("/srv/local/shengenli/flowstar-gpu/src")


def main():
    config = yaml.safe_load((HERE / "config.yaml").read_text())
    source = yaml.safe_load((HERE / "source_point_config.yaml").read_text())
    spec = (PREP / "specifications.txt").read_text()
    matlab = (PREP / "dynamics.m").read_text()
    assert "t = 20 control steps = 2 seconds" in spec
    assert "control step = 0.1" in spec
    assert "x2 should be in [-1, 1] and x7,x8,x9 should be in [-1.0,1.0]" in spec
    assert "dr = Mz;" in matlab and "dp = Mx;" in matlab and "dq = My;" in matlab
    assert [config[x] for x in ("num_vars", "num_nn_input", "num_nn_output",
                                "steps", "step_size", "ode_step_size", "ode_order")] == [
                                    19, 12, 6, 20, .1, .01, 6]
    assert config["dynamics_expressions"] == source["dynamics_expressions"]
    assert config["constraints_safe"] == source["constraints_safe"]
    assert len(config["dynamics_expressions"]) == 19
    assert config["output_scale"] == 1 and config["output_offset"] == 0
    assert config["input_shape"] == [-1, 12]
    assert config["output_T_shape"] == [-1, 6, 12]
    assert config["output_c_shape"] == [-1, 6]
    assert not config.get("split_vars")
    assert not any(config.get(key) for key in (
        "constraints_safe_from", "constraints_safe_until",
        "constraints_unsafe", "constraints_target"))
    names = [item["name"] for item in config["initial_set"]]
    assert names == ["x", "y", "z", "u", "v", "w", "phi", "theta", "psi",
                     "r", "p", "q", "t", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    intervals = [item["interval"] for item in config["initial_set"]]
    assert intervals == ([[0, 0]] * 3 + [[0.0, 1.0]] * 6 + [[0, 0]] * 10)
    assert all(not item.get("splits") for item in config["initial_set"])
    assert [(item["name"], item["interval"]) for item in source["initial_set"]][3:9] != [
        (item["name"], item["interval"]) for item in config["initial_set"]][3:9]
    assert config["constraints_safe"] == ["-y - 1", "y - 1", "-phi - 1", "phi - 1",
                                           "-theta - 1", "theta - 1", "-psi - 1", "psi - 1"]

    sys.path.insert(0, str(ENGINE))
    from flowstar_gpu.expr_parser import parse
    from flowstar_gpu.ode_compiler import compile_ode

    ode = compile_ode(config["dynamics_expressions"], names,
                      order=config["ode_order"] - 1)
    constraints = [parse(value, names) for value in config["constraints_safe"]]
    assert len(constraints) == 8 and len(ode.out_slots) == 19

    model_path = Path(config["model_dir"])
    assert model_path == PREP / "controller_airplane.onnx"
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    initialized = {x.name for x in model.graph.initializer}
    inputs = [x for x in model.graph.input if x.name not in initialized]
    outputs = list(model.graph.output)
    assert len(inputs) == len(outputs) == 1
    tensor_shape = lambda info: [d.dim_value if d.HasField("dim_value") else d.dim_param
                                 for d in info.type.tensor_type.shape.dim]
    assert (inputs[0].name, tensor_shape(inputs[0])) == ("sequential_1_input", ["N", 12])
    assert (outputs[0].name, tensor_shape(outputs[0])) == ("dense_4", ["N", 6])
    ops = [node.op_type for node in model.graph.node]
    assert ops[0] == "MatMul" and ops.count("MatMul") == 4 and ops.count("Add") == 4
    assert ops.count("Relu") == 3 and len(ops) == 11
    assert not torch.cuda.is_initialized()
    report = {
        "schema": "archcomp26-airplane-continuous-fullbox-cpu-preflight-v1",
        "source_spec": str(PREP / "specifications.txt"),
        "source_dynamics": str(PREP / "dynamics.m"),
        "source_controller": str(model_path),
        "generated_config": str(HERE / "config.yaml"),
        "source_point_config_used_only_for_ode_and_checker_comparison":
            str(HERE / "source_point_config.yaml"),
        "physical_state_order": names[:12], "physical_initial_box": intervals[:12],
        "auxiliary_initial_intervals": intervals[12:],
        "full_initial_box_count": 1, "control_period_s": .1, "control_periods": 20,
        "ode_substep_s": .01, "ode_substeps": 200,
        "controller_input": {"name": inputs[0].name, "shape": tensor_shape(inputs[0])},
        "controller_output": {"name": outputs[0].name, "shape": tensor_shape(outputs[0])},
        "ode_components_parsed": len(ode.out_slots), "safe_constraints_parsed": len(constraints),
        "initial_safe_check": "y=0 and phi/theta/psi in [0,1], inside closed [-1,1]",
        "checked_property": "y,phi,theta,psi in [-1,1] for all t in [0,2]",
        "gpu_initialized": torch.cuda.is_initialized(),
        "numerical_run_started": False, "content_digest_performed": False,
    }
    with (HERE / "PREFLIGHT.json").open("x") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    print("Airplane continuous full-box CPU preflight passed")


if __name__ == "__main__":
    main()
