#!/usr/bin/env python3
"""One full-box paper Unicycle control period on the saved working-P3 core."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import archcomp26_dp_p3_nohash as helper
import archcomp26_unicycle_paper_author_nohash as unicycle


def prepare_p3(backend):
    if backend != "p3" or os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("Unicycle P3 diagnostic requires physical GPU3")
    for key, value in helper.ENV.items():
        os.environ[key] = value
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path.insert(0, str(helper.ENGINE / "src"))
    helper.apply_guards()
    import torch

    if torch.__version__ != "2.5.1+cu121" or not torch.cuda.is_available():
        raise RuntimeError("saved PyTorch/CUDA runtime unavailable")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    total = torch.cuda.get_device_properties(0).total_memory
    cap = min(11 * 2**30, int(total * 0.9))
    torch.cuda.set_per_process_memory_fraction(cap / total)

    import flowstar_gpu
    from flowstar_gpu import cuda_kernels as ck, elementary as elem

    ck.load_cuda_extension = helper.prohibit
    ck._ext = helper.load_binary("flowstar_seg_kernels", helper.EXTENSIONS["flowstar_seg_kernels"])
    ck._tried = True
    core = helper.load_python("unicycle_p3_reciprocal_core", helper.RECIP_CORE)
    for name in ("rec_series_valid_g", "rec_series_valid", "rec_series_replay"):
        setattr(elem, name, getattr(core, name))
    tape = helper.load_python("flowstar_gpu.tape_kernels", helper.RECIP_SOURCE)
    tape.load_cuda_extension = helper.prohibit
    tape._ext = helper.load_binary("flowstar_recip_geom_replay_1f9efda6325c",
                                   helper.EXTENSIONS["flowstar_recip_geom_replay_1f9efda6325c"])
    tape._vext = helper.load_binary("flowstar_recip_geom_valid_1f9efda6325c",
                                    helper.EXTENSIONS["flowstar_recip_geom_valid_1f9efda6325c"])
    tape._tried = tape._vtried = True
    flowstar_gpu.tape_kernels = tape
    for module_name, extension_name in (
        ("sr_kernels", "flowstar_sr_interval_matmul"),
        ("sr_sum_kernels", "flowstar_sr_history_sum"),
        ("injective_index", "flowstar_injective_index_v2"),
        ("private_output_kernels", "flowstar_seg_private_output_v1"),
        ("horner_edge_kernels", "horner_edge_1312fa8b2aed"),
    ):
        module = __import__("flowstar_gpu." + module_name, fromlist=[module_name])
        module._ext = helper.load_binary(extension_name, helper.EXTENSIONS[extension_name])
        if hasattr(module, "_tried"):
            module._tried = True
    if not (ck.available() and tape.available() and tape.valid_available()):
        raise RuntimeError("saved P3 CUDA libraries unavailable")

    driver = helper.load_python("archcomp26_unicycle_p3_driver", helper.DRIVER)
    from flowstar_gpu import sparse_exec as se

    if not hasattr(se, "COMPOSE_PARENT_ASSEMBLY"):
        se.COMPOSE_PARENT_ASSEMBLY = None
    endpoint = helper.load_python("archcomp26_unicycle_p3_strict_endpoint", helper.ENDPOINT)
    driver.end_of_time_s = endpoint.end_of_time_s
    holder = {}
    original_engine = driver.SparseEngine

    def capture_engine(*args, **kwargs):
        if holder:
            raise RuntimeError("expected one Unicycle P3 sparse engine")
        engine = original_engine(*args, **kwargs)
        if (engine.tables.n, engine.tables.k) != (8, 3):
            raise ValueError("Unicycle P3 requires eight variables and working order three")
        holder["engine"] = engine
        return engine

    def strict_inject(state, matrix, lower, upper, control_ids, nn_input):
        if "engine" not in holder or tuple(control_ids) != (6, 7) or nn_input != 4:
            raise RuntimeError("unexpected Unicycle P3 two-control injection layout")
        with torch.no_grad():
            return helper.strict_injection(state, matrix, lower, upper,
                                           control_ids, nn_input, holder["engine"])

    driver.SparseEngine = capture_engine
    driver.inject_controls_s = strict_inject
    driver.SR_QUEUE = 1000
    return torch, driver, helper.ENGINE, helper.CACHE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke1", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    unicycle.common.prepare = prepare_p3
    try:
        return unicycle.run(argparse.Namespace(backend="p3", mode=args.mode, output=args.output))
    finally:
        if args.output.is_dir():
            (args.output / "P3_METHOD.json").write_text(json.dumps({
                "schema": "archcomp26-unicycle-paper-speed-w-constant-p3-method-nohash-v1",
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
                "method": "working P3 order 3 / validation order 4, strict endpoint and two-control injection, box same-slope CROWN, rpc-float32",
                "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "engine": str(helper.ENGINE), "shared_driver": str(helper.DRIVER),
                "source_files": [str(Path(__file__).resolve()),
                                 str(Path(unicycle.__file__).resolve()),
                                 str(Path(helper.__file__).resolve())],
                "scope": ("full original initial box, all 50 control periods to T=10"
                          if args.mode == "full" else
                          "full original initial box, first of 50 control periods; no T=10 property verdict"),
                "qualification": "saved P3 numerical diagnostic; no independent end-to-end floating-point NNCS certificate",
            }, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
