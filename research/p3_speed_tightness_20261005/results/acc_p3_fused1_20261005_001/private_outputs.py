"""Select an already loaded private-allocation extension before graph capture.

This adapter does not load/build extensions, call availability helpers, change
deterministic settings, or alter arithmetic. Existing captured graphs must not
be reused across installation/restoration: they retain their captured launches.
Module-name and object-identity checks are not a binary provenance certificate.
"""

from dataclasses import dataclass
import sys
from types import ModuleType


EXPORTS = (
    "seg_mul_iv", "seg_mul_pt", "seg_dot_pt_iv", "iv_mul", "iv_addsub",
    "iv_neg", "iv_mul_point", "iv_sum", "iv_dot_point_iv",
)
BASE_NAME = "flowstar_seg_kernels"
PRIVATE_NAME = "flowstar_seg_private_output_v1"


def require_module(module, name):
    if not isinstance(module, ModuleType) or module.__name__ != name:
        raise RuntimeError(f"expected module {name}")
    if sys.modules.get(name) is not module:
        raise RuntimeError(f"{name} is not the registered preloaded module")


@dataclass
class Binding:
    ck: ModuleType
    original: ModuleType
    selected: ModuleType
    receipt: dict
    restored: bool = False

    def restore(self):
        expected = self.original if self.restored else self.selected
        if self.ck._ext is not expected or self.ck._tried is not True:
            raise RuntimeError("private-output binding no longer owned; refusing restore")
        self.ck._ext = self.original
        self.restored = True


def install(torch, ck, private_module):
    """Bind only preloaded, named extensions; return a reversible receipt."""
    if not isinstance(torch, ModuleType) or torch.__name__ != "torch":
        raise RuntimeError("expected the torch module")
    version = getattr(torch, "__version__", "")
    if not isinstance(version, str) or version.split("+")[0] != "2.5.1":
        raise RuntimeError("private-output extension requires PyTorch 2.5.1")
    require_module(ck, "flowstar_gpu.cuda_kernels")
    require_module(private_module, "flowstar_gpu.private_output_kernels")
    if (getattr(ck, "_tried", None) is not True
            or getattr(private_module, "_tried", None) is not True):
        raise RuntimeError("both extensions must already be loaded")
    if (getattr(private_module, "base", None) is not ck
            or getattr(private_module, "NAME", None) != PRIVATE_NAME
            or getattr(private_module, "SUPPORTED", None) is not True):
        raise RuntimeError("unexpected private-output module configuration")
    original = getattr(ck, "_ext", None)
    selected = getattr(private_module, "_ext", None)
    require_module(original, BASE_NAME)
    require_module(selected, PRIVATE_NAME)
    for module in (original, selected):
        missing = [name for name in EXPORTS if not callable(getattr(module, name, None))]
        if missing:
            raise RuntimeError(f"{module.__name__} missing callable exports: {missing}")
    receipt = {
        "schema": "p3-private-output-binding-v1",
        "torch_version": version,
        "original_extension": BASE_NAME,
        "selected_extension": PRIVATE_NAME,
        "exports_checked": list(EXPORTS),
        "changed_field": "flowstar_gpu.cuda_kernels._ext",
        "scope": "preloaded output allocation only; arithmetic and settings unchanged",
        "hash_or_jit_performed": False,
        "qualification": "configuration checks only; numerical GPU gate is separate",
    }
    binding = Binding(ck, original, selected, receipt)
    ck._ext = selected
    return binding
