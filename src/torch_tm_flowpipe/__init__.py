"""PyTorch-native Taylor-model flowpipe research prototype.

Public solver objects are loaded lazily so metadata and protocol-only consumers
do not need PyTorch installed merely to import ``torch_tm_flowpipe.protocol``.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any


_PUBLIC_MODULES = {
    "FlowpipeResult": ".flowpipe",
    "FlowpipeSegment": ".flowpipe",
    "FlowstarNormalFlowpipeState": ".flowpipe",
    "HornerInsertionDiagnosticResult": ".flowpipe",
    "flowpipe_multi_step": ".flowpipe",
    "flowpipe_step": ".flowpipe",
    "flowpipe_step_flowstar_style_adaptive": ".flowpipe",
    "flowpipe_step_from_tm": ".flowpipe",
    "insert_ctrunc_normal_horner_diagnostic": ".flowpipe",
    "insert_ctrunc_normal_like": ".flowpipe",
    "Interval": ".interval",
    "Polynomial": ".polynomial",
    "evaluate_interval_normal": ".polynomial",
    "FlowstarSymbolicRemainderQueue": ".symbolic_remainder",
    "SymbolicNoiseSymbol": ".symbolic_remainder",
    "SymbolicRemainderState": ".symbolic_remainder",
    "SymbolicTaylorModel": ".symbolic_remainder",
    "introduce_symbolic_remainders": ".symbolic_remainder",
    "materialize_all_symbols": ".symbolic_remainder",
    "materialize_non_symbolic_variables": ".symbolic_remainder",
    "materialize_oldest_symbols": ".symbolic_remainder",
    "symbolic_noise_domain": ".symbolic_remainder",
    "TaylorModel": ".taylor_model",
    "taylor_model_mul_breakdown": ".taylor_model",
    "TMVector": ".tm_vector",
}

__all__ = list(_PUBLIC_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _PUBLIC_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
