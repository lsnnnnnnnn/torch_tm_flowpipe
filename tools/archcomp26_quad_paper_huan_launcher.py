#!/usr/bin/env python3
"""Run one new paper-contract Huan QUAD arm with existing CUDA modules."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("mode", choices=("parity", "strict"))
p.add_argument("run_root", type=Path)
a = p.parse_args()

def no_sha256(*args, **kwargs):
    raise RuntimeError("SHA-256 is disabled for this run")

hashlib.sha256 = no_sha256
n = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
h = n / "engine_huan_sr_chunk"
cache = n / "cache_four_way_v1/huan"
config = a.run_root / "quad_paper.yaml"
arm = a.run_root / a.mode
assert config.is_file() and arm.is_dir()
assert not (arm / "metrics.json").exists()
with (arm / "CLAIM").open("x") as claim:
    claim.write(datetime.now(timezone.utc).isoformat() + "\n")
for key in list(os.environ):
    if key.startswith("FLOWSTAR_"):
        del os.environ[key]
os.environ["TORCH_EXTENSIONS_DIR"] = str(cache)
sys.path.insert(0, str(h / "src"))
import torch
import torch.utils.cpp_extension as cpp_extension

def no_jit(*args, **kwargs):
    raise RuntimeError("JIT extension build is disabled for this run")

cpp_extension.load_inline = no_jit
cpp_extension.load = no_jit
from flowstar_gpu import cuda_kernels as ck, tape_kernels as tk
ck.load_cuda_extension = no_jit
tk.load_cuda_extension = no_jit

modules = {}
for owner, attr, tried, name, exports in (
    (ck, "_ext", "_tried", "flowstar_seg_kernels", ("seg_mul_iv", "seg_mul_pt", "seg_dot_pt_iv")),
    (tk, "_ext", "_tried", "flowstar_tape_kernels", ("refine_tape", "transcendental_probe")),
    (tk, "_vext", "_vtried", "flowstar_valid_kernels", ("valid_tape", "point_tape")),
):
    binary = cache / name / (name + ".so")
    assert binary.is_file(), binary
    spec = importlib.util.spec_from_file_location(name, binary)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for export in exports:
        assert callable(getattr(module, export, None)), (name, export)
    setattr(owner, attr, module)
    setattr(owner, tried, True)
    stat = binary.stat()
    modules[name] = {"path": str(binary), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
assert ck.available() and tk.available() and tk.valid_available()

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.cuda.set_per_process_memory_fraction(
    11 * 2**30 / torch.cuda.get_device_properties(0).total_memory
)
driver_path = h / "integrations/crown_reach/gpu_driver.py"
spec = importlib.util.spec_from_file_location("huan_quad_stage_a_driver", driver_path)
assert spec is not None and spec.loader is not None
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)
argv = [str(driver_path), str(config), "--device", "cuda:0", "--engine", "sparse",
        "--crown-domain", "box", "--crown-relax", "same-slope", "--print-final-hull",
        "--metrics-json", str(arm / "metrics.json")]
if a.mode == "strict":
    argv.append("--strict")
sys.argv = argv
receipt = {"mode": a.mode, "config": str(config), "driver": str(driver_path),
           "argv": argv, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "cpu_affinity": sorted(os.sched_getaffinity(0)), "modules": modules,
           "start_utc": datetime.now(timezone.utc).isoformat()}
print("RUN_START", json.dumps(receipt, sort_keys=True), flush=True)
t0 = time.perf_counter()
rc = None
try:
    rc = driver.main()
finally:
    receipt["end_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["driver_wall_s"] = time.perf_counter() - t0
    receipt["returncode"] = rc
    if torch.cuda.is_initialized():
        receipt["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        receipt["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    (arm / "launcher_result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("RUN_END", json.dumps({k: receipt.get(k) for k in (
        "mode", "end_utc", "driver_wall_s", "returncode",
        "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")}), flush=True)
raise SystemExit(rc)
