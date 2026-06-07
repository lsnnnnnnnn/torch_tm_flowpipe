"""PyTorch-native Taylor-model flowpipe research prototype."""
from .flowpipe import FlowpipeResult, FlowpipeSegment, FlowstarNormalFlowpipeState, HornerInsertionDiagnosticResult, flowpipe_multi_step, flowpipe_step, flowpipe_step_flowstar_style_adaptive, flowpipe_step_from_tm, insert_ctrunc_normal_horner_diagnostic, insert_ctrunc_normal_like
from .interval import Interval
from .polynomial import Polynomial, evaluate_interval_normal
from .symbolic_remainder import (
    FlowstarSymbolicRemainderQueue,
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
    "FlowstarSymbolicRemainderQueue",
    "FlowstarNormalFlowpipeState",
    "HornerInsertionDiagnosticResult",
    "Interval",
    "Polynomial",
    "evaluate_interval_normal",
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
    "flowpipe_step_flowstar_style_adaptive",
    "flowpipe_multi_step",
    "insert_ctrunc_normal_horner_diagnostic",
    "insert_ctrunc_normal_like",
]
