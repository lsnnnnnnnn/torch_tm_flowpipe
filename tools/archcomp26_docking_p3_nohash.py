#!/usr/bin/env python3
"""Docking full-box run on the saved working-P3 engine and radial checker."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import archcomp26_docking_author_nohash as docking
import archcomp26_dp_p3_nohash as helper


def prepare_p3(backend):
    if backend != "p3" or os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("Docking P3 requires physical GPU3")
    for key, value in helper.ENV.items():
        os.environ[key] = value
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path.insert(0, str(helper.ENGINE / "src"))
    helper.apply_guards()
    import torch

    if torch.__version__ != "2.5.1+cu121" or not torch.cuda.is_available():
        raise RuntimeError("saved PyTorch/CUDA environment unavailable")
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
    core = helper.load_python("docking_p3_reciprocal_core", helper.RECIP_CORE)
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

    driver = helper.load_python("archcomp26_docking_p3_driver", helper.DRIVER)
    from flowstar_gpu import sparse_exec as se
    if not hasattr(se, "COMPOSE_PARENT_ASSEMBLY"):
        se.COMPOSE_PARENT_ASSEMBLY = None
    endpoint = helper.load_python("archcomp26_docking_p3_strict_endpoint", helper.ENDPOINT)
    driver.end_of_time_s = endpoint.end_of_time_s
    owner = {}
    original_engine = driver.SparseEngine

    def capture_engine(*args, **kwargs):
        if owner:
            raise RuntimeError("one Docking run must create one P3 sparse engine")
        engine = original_engine(*args, **kwargs)
        if (engine.tables.n, engine.tables.k) != (7, 3):
            raise ValueError("Docking P3 requires seven variables and order three")
        owner["engine"] = engine
        return engine

    def strict_inject(state, matrix, lower, upper, control_ids, nn_input):
        if "engine" not in owner:
            raise RuntimeError("P3 control injection before engine creation")
        with torch.no_grad():
            return helper.strict_injection(state, matrix, lower, upper,
                                           control_ids, nn_input, owner["engine"])

    driver.SparseEngine = capture_engine
    driver.inject_controls_s = strict_inject
    driver.SR_QUEUE = 1000
    return torch, driver, helper.ENGINE, helper.CACHE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke1", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    docking.common.ENGINES["p3"] = helper.ENGINE
    docking.common.prepare = prepare_p3
    run_args = argparse.Namespace(backend="p3", mode=args.mode, output=args.output)
    try:
        return docking.run(run_args)
    finally:
        if args.output.is_dir():
            (args.output / "P3_METHOD.json").write_text(json.dumps({
                "schema": "archcomp26-docking-p3-method-nohash-v1",
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
                "method": "working-P3 core, strict endpoint and control injection, box same-slope CROWN, rpc-float32",
                "engine": str(helper.ENGINE), "shared_driver": str(helper.DRIVER),
                "strict_injection": str(Path(helper.__file__).resolve()),
                "radial_checker": str(Path(docking.__file__).resolve()),
                "prebuilt_libraries": {name: {"path": str(path), "bytes": path.stat().st_size,
                                               "mtime_ns": path.stat().st_mtime_ns}
                                       for name, path in helper.EXTENSIONS.items()},
                "qualification": "new Docking P3 diagnostic; no independent end-to-end floating-point NN certificate",
            }, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
