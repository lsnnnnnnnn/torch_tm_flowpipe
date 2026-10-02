#!/usr/bin/env python3
"""One isolated replay of the saved first QUAD CROWN batch, recording uA.

No plant propagation, old experiment loop, or digest is performed.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import time


def disabled_sha256(*_args, **_kwargs):
    raise RuntimeError("SHA-256 disabled for this diagnostic")


original_hash_new = hashlib.new


def guarded_hash_new(name, *args, **kwargs):
    if name.lower().replace("-", "") == "sha256":
        return disabled_sha256()
    return original_hash_new(name, *args, **kwargs)


hashlib.sha256 = disabled_sha256
hashlib.new = guarded_hash_new

# Match the frozen QUAD server's imports and model construction.
sys.path.append("../../Verifier_Development/complete_verifier")
import torch  # noqa: E402
import onnx  # noqa: E402
import onnx2pytorch  # noqa: E402
from collections import defaultdict  # noqa: E402
from auto_LiRPA import BoundedModule, BoundedTensor  # noqa: E402
from auto_LiRPA.perturbations import PerturbationLpNorm  # noqa: E402


RUN = Path(os.environ["QUAD_PROBE_RUN"])
RPC = Path(os.environ["QUAD_PROBE_OLD_RPC"])
MODEL = Path(os.environ["QUAD_MODEL"])


def equal_report(new, old):
    mismatches = 0
    first = None
    max_abs_difference = 0.0
    for lane in range(1024):
        for output in range(3):
            values = new[lane][output]
            expected = old[lane][output]
            if not isinstance(values, list):
                values = [values]
                expected = [expected]
            for coordinate, (a, b) in enumerate(zip(values, expected)):
                if a != b:
                    mismatches += 1
                    max_abs_difference = max(max_abs_difference, abs(a - b))
                    if first is None:
                        first = {"lane": lane, "output": output + 1,
                                 "coordinate": coordinate + 1, "new": a, "old": b}
    return {"mismatches": mismatches, "first_mismatch": first,
            "max_abs_difference": max_abs_difference}


def main():
    started = time.monotonic()
    rpc = json.loads(RPC.read_text())
    saved = rpc["coefficients"]
    lb = rpc["params"]["input_lb"]
    ub = rpc["params"]["input_ub"]
    assert len(lb) == len(ub) == 1024 * 12
    assert all(len(saved[k]) == 1024 for k in ("T", "u_min", "u_max"))
    assert MODEL.is_file()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert device.type == "cuda", "this replay requires its isolated GPU assignment"
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_ori = onnx2pytorch.ConvertModel(onnx.load(str(MODEL)))
    model_ori.to(torch.get_default_dtype())
    lirpa_model = BoundedModule(model_ori, torch.zeros(1, 12), device=device,
                                bound_opts={"activation_bound_option": "same-slope"})
    input_lb = torch.tensor(lb).view(-1, 12).to(dtype=torch.get_default_dtype(), device=device)
    input_ub = torch.tensor(ub).view(-1, 12).to(dtype=torch.get_default_dtype(), device=device)
    bounded = BoundedTensor(input_lb, PerturbationLpNorm(x_L=input_lb, x_U=input_ub))
    required_A = defaultdict(set)
    required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
    _, _, A_dict = lirpa_model.compute_bounds(x=(bounded,), method="backward",
                                                return_A=True, needed_A_dict=required_A)
    A = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]
    tensors = {name: A[name] for name in ("lA", "uA", "lbias", "ubias")}
    expected_shape = {"lA": (1024, 3, 12), "uA": (1024, 3, 12),
                      "lbias": (1024, 3), "ubias": (1024, 3)}
    shapes = {}
    raw = {}
    for name, value in tensors.items():
        assert isinstance(value, torch.Tensor), f"{name} is not a tensor"
        shapes[name] = list(value.shape)
        assert value.numel() == 1024 * 3 * (12 if name.endswith("A") else 1), (name, shapes[name])
        assert torch.isfinite(value).all().item(), f"{name} contains a nonfinite value"
        raw[name] = value.reshape(expected_shape[name]).cpu().tolist()
    (RUN / "RAW_COEFFICIENTS.json").write_text(json.dumps(raw, allow_nan=False) + "\n")
    comparison = {
        "lA_vs_saved_T": equal_report(raw["lA"], saved["T"]),
        "lbias_vs_saved_u_min": equal_report(raw["lbias"], saved["u_min"]),
        "ubias_vs_saved_u_max": equal_report(raw["ubias"], saved["u_max"]),
        "uA_vs_lA": equal_report(raw["uA"], raw["lA"]),
    }
    old_match = all(comparison[key]["mismatches"] == 0 for key in
                    ("lA_vs_saved_T", "lbias_vs_saved_u_min", "ubias_vs_saved_u_max"))
    result = {
        "scope": "one controller-only 1024-box replay of the saved first QUAD RPC batch",
        "status": "RECORDED",
        "old_coefficient_arrays_exactly_match": old_match,
        "uA_equals_lA_exactly": comparison["uA_vs_lA"]["mismatches"] == 0,
        "new_uA_attachable_to_saved_rpc": old_match,
        "tensor_shapes": shapes,
        "all_tensors_finite": True,
        "comparison": comparison,
        "elapsed_seconds": time.monotonic() - started,
        "torch_version": torch.__version__,
        "device": str(device),
        "limits": "No independent CROWN soundness, NN residual, float32 transport, plant, or full-horizon proof.",
    }
    (RUN / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "old_coefficient_arrays_exactly_match",
                    "uA_equals_lA_exactly", "new_uA_attachable_to_saved_rpc", "elapsed_seconds")}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        (RUN / "RESULT.json").write_text(json.dumps({"status": "ERROR",
            "error_type": type(error).__name__, "error": str(error),
            "limits": "No coefficient conclusion from a failed diagnostic."}, indent=2) + "\n")
        raise
