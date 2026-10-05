"""Bind the preloaded Horner remainder kernel to a fresh CUDA engine.

No imports of the numerical engine, extension loading, compilation, source
checks or arithmetic changes occur here. The engine's existing composition
dispatcher keeps its input guards and fallback. Install before any graph
capture; restore after the run and discard that run's engine/graphs.
Module and object checks are not a binary provenance certificate.
"""

from dataclasses import dataclass
import sys
from types import ModuleType


EXTENSION_NAME = "horner_edge_1312fa8b2aed"
FIELD = "horner_edge_kernel"
_MISSING = object()


def require_module(module, name):
    if (not isinstance(module, ModuleType) or module.__name__ != name
            or sys.modules.get(name) is not module):
        raise RuntimeError(f"expected registered preloaded module {name}")


@dataclass
class Binding:
    engine: object
    original: object
    selected: object
    receipt: dict
    restored: bool = False

    def restore(self):
        expected = self.original if self.restored else self.selected
        if vars(self.engine).get(FIELD, _MISSING) is not expected:
            raise RuntimeError("Horner binding no longer owned; refusing restore")
        if self.original is _MISSING:
            if not self.restored:
                delattr(self.engine, FIELD)
        else:
            setattr(self.engine, FIELD, self.original)
        self.restored = True


def install(torch, engine, sparse_exec, edge_module):
    """Attach only the qualified preloaded export; return a reversible binding."""
    require_module(torch, "torch")
    version = getattr(torch, "__version__", "")
    if not isinstance(version, str) or version.split("+")[0] != "2.5.1":
        raise RuntimeError("Horner edge requires PyTorch 2.5.1")
    require_module(sparse_exec, "flowstar_gpu.sparse_exec")
    require_module(edge_module, "flowstar_gpu.horner_edge_kernels")
    support = sys.modules.get("flowstar_gpu.support")
    require_module(support, "flowstar_gpu.support")
    engine_type = getattr(support, "SparseEngine", None)
    if not isinstance(engine_type, type) or not isinstance(engine, engine_type):
        raise RuntimeError("expected an engine from the registered sparse support")
    if getattr(sparse_exec, "COMPOSITION_MODE", None) != "horner":
        raise RuntimeError("Horner composition must already be selected")
    zero = getattr(engine, "_cutoff_zero", None)
    if (not isinstance(zero, torch.Tensor) or not zero.is_cuda
            or zero.dtype != torch.float64
            or zero.device != torch.device(engine.device)):
        raise RuntimeError("expected the engine's CUDA float64 cutoff tensor")
    if any(getattr(engine, name, None) is not None
           for name in ("_graphs", "_weighted_graphs")):
        raise RuntimeError("install Horner edge immediately after engine construction")
    original = vars(engine).get(FIELD, _MISSING)
    if getattr(engine, FIELD, None) is not None:
        raise RuntimeError("refusing to replace an existing Horner binding")
    if getattr(edge_module, "NAME", None) != EXTENSION_NAME:
        raise RuntimeError("unexpected Horner extension configuration")
    extension = getattr(edge_module, "_ext", None)
    require_module(extension, EXTENSION_NAME)
    kernel = getattr(extension, "horner_edge", None)
    if not callable(kernel):
        raise RuntimeError("preloaded extension has no callable horner_edge")
    receipt = {
        "schema": "p3-horner-edge-binding-v1",
        "torch_version": version,
        "extension": EXTENSION_NAME,
        "changed_field": "engine.horner_edge_kernel",
        "original_field": "absent" if original is _MISSING else "None",
        "scope": "existing directed remainder operations in their original order",
        "hash_or_jit_performed": False,
        "qualification": "configuration checks only; numerical GPU gate is separate",
    }
    binding = Binding(engine, original, kernel, receipt)
    setattr(engine, FIELD, kernel)
    return binding
