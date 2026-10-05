"""Run only the new trig-power gate on the frozen engine's CPU arithmetic."""
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
sys.path.insert(0, str(RESEARCH / "p3_speed_tightness_20261005"))
from run_quad_candidate import guard_digests, reject_digest
guard_digests()
import torch
torch.jit.script = lambda f=None, **kw: f if f is not None else lambda value: value
torch.jit.trace = torch.compile = reject_digest
sys.path.insert(0, str(RESEARCH / "gpu_verified_20260930/source/engine/src"))
from flowstar_gpu import sparse_exec as se
from trig_power_reuse import install
from trig_power_gate import check


if __name__ == "__main__":
    path = RESEARCH / "gpu_verified_20260930/source/adapters/quad_fullbatch_p3_20260928/trig_direct_reuse.py"
    spec = importlib.util.spec_from_file_location("cpu_existing_direct_trig", path)
    direct = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = direct
    spec.loader.exec_module(direct)
    previous = direct._bind(se)  # Binding only; never the historical checker/install.
    with patch.object(torch, "__version__", "2.5.1+CPU-admission-fixture"):
        binding = install(torch, se, previous)
    try:
        receipt = check(torch, se, binding, device="cpu")
        try:
            with patch.object(torch, "__version__", "2.5.1+CPU-admission-fixture"):
                install(torch, se, previous)
        except RuntimeError:
            receipt["duplicate_install_refused"] = True
        else:
            raise RuntimeError("duplicate binding accepted")
    finally:
        binding.restore()
        previous.restore()
    receipt["both_bindings_restored"] = True
    receipt["runtime_admission_fixture"] = "only the install version string was mocked; arithmetic used real CPU Torch"
    print(json.dumps(receipt, indent=2))
