"""PyTorch-native Taylor-model flowpipe research prototype."""
from .flowpipe import FlowpipeResult, FlowpipeSegment, flowpipe_multi_step, flowpipe_step, flowpipe_step_from_tm
from .interval import Interval
from .polynomial import Polynomial
from .symbolic_remainder import (
    SymbolicNoiseSymbol,
    SymbolicRemainderState,
    SymbolicTaylorModel,
    introduce_symbolic_remainders,
    materialize_all_symbols,
    materialize_non_symbolic_variables,
    materialize_oldest_symbols,
    symbolic_noise_domain,
)
from .taylor_model import TaylorModel, taylor_model_mul_breakdown
from .tm_vector import TMVector

__all__ = [
    "FlowpipeResult",
    "FlowpipeSegment",
    "Interval",
    "Polynomial",
    "SymbolicNoiseSymbol",
    "SymbolicRemainderState",
    "SymbolicTaylorModel",
    "introduce_symbolic_remainders",
    "materialize_all_symbols",
    "materialize_non_symbolic_variables",
    "materialize_oldest_symbols",
    "symbolic_noise_domain",
    "TaylorModel",
    "taylor_model_mul_breakdown",
    "TMVector",
    "flowpipe_step",
    "flowpipe_step_from_tm",
    "flowpipe_multi_step",
]
