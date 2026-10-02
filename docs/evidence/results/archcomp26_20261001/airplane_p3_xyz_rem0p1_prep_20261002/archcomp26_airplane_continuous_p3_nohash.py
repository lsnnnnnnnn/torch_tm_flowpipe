#!/usr/bin/env python3
"""Isolated one-box 2026 Airplane continuous P3 diagnostic."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import archcomp26_dp_p3_nohash as helper


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
AUTHOR = N / "runs/archcomp26_20261001/airplane_continuous_order3_fullbox_20261002/run.py"
COMMON = N / "runs/archcomp26_20261001/author_tora_remain_v1/archcomp26_tora_remain_author_nohash.py"


def prepare_p3(backend):
    if backend != "p3" or os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("Airplane P3 requires physical GPU3")
    for key, value in helper.ENV.items():
        os.environ[key] = value
    # n=19 validation order 4 cannot use the saved int64 radix table (9^20).
    # The engine's supported strict solution-order policy validates at order 3.
    os.environ["FLOWSTAR_VALIDATION_POLICY"] = "solution_order"
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
    core = helper.load_python("airplane_p3_reciprocal_core", helper.RECIP_CORE)
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

    driver = helper.load_python("archcomp26_airplane_p3_driver", helper.DRIVER)
    from flowstar_gpu import sparse_exec as se
    if not hasattr(se, "COMPOSE_PARENT_ASSEMBLY"):
        se.COMPOSE_PARENT_ASSEMBLY = None
    endpoint = helper.load_python("archcomp26_airplane_p3_strict_endpoint", helper.ENDPOINT)
    driver.end_of_time_s = endpoint.end_of_time_s
    owner = {}
    original_engine = driver.SparseEngine

    def capture_engine(*args, **kwargs):
        if owner:
            raise RuntimeError("one Airplane run must create one P3 sparse engine")
        engine = original_engine(*args, **kwargs)
        if (engine.tables.n, engine.tables.k) != (19, 3):
            raise ValueError("Airplane P3 requires 19 variables and order three")
        owner["engine"] = engine
        return engine

    def strict_inject(state, matrix, lower, upper, control_ids, nn_input):
        if "engine" not in owner or tuple(control_ids) != (13, 14, 15, 16, 17, 18) or nn_input != 12:
            raise RuntimeError("unexpected Airplane six-control injection layout")
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
    airplane = helper.load_python("archcomp26_airplane_p3_entry", AUTHOR)
    common = helper.load_python("archcomp26_airplane_p3_common", COMMON)
    airplane.import_helper = lambda: SimpleNamespace(prepare=prepare_p3,
                                                      DRIVER=helper.DRIVER, Tee=common.Tee)
    run_args = argparse.Namespace(backend="p3", mode=args.mode, output=args.output)
    try:
        return airplane.run(run_args)
    finally:
        if args.output.is_dir():
            (args.output / "P3_METHOD.json").write_text(json.dumps({
                "schema": "archcomp26-airplane-continuous-p3-method-nohash-v1",
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
                "method": "working-P3/validation-P3 solution_order, strict endpoint and six-control injection, box same-slope CROWN, rpc-float32",
                "validation_policy": "solution_order",
                "engine": str(helper.ENGINE), "shared_driver": str(helper.DRIVER),
                "airplane_entry": str(AUTHOR), "strict_injection": str(Path(helper.__file__).resolve()),
                "prebuilt_libraries": {name: {"path": str(path), "bytes": path.stat().st_size,
                                               "mtime_ns": path.stat().st_mtime_ns}
                                       for name, path in helper.EXTENSIONS.items()},
                "qualification": "new complete-box P3 diagnostic; no independent end-to-end floating-point NN certificate",
            }, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
