"""PyTorch-native Taylor-model flowpipe research prototype.

Public solver objects are loaded lazily so metadata and protocol-only consumers
do not need PyTorch installed merely to import ``torch_tm_flowpipe.protocol``.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any


_PUBLIC_MODULES = {
    "BatchedMonomialBasis": ".batched_dense_tm",
    "BatchedPolynomial": ".batched_dense_tm",
    "BatchedTaylorModel": ".batched_dense_tm",
    "DenseCanonicalPolynomial": ".batched_dense_tm",
    "DenseHornerOrderResult": ".batched_dense_tm",
    "DensePolynomialRangeResult": ".batched_dense_tm",
    "DenseRegisteredHornerResult": ".batched_dense_tm",
    "DenseRangePolicy": ".batched_dense_tm",
    "DenseSubdivisionCover": ".batched_dense_tm",
    "DenseExecutionCounters": ".batched_dense_tm",
    "DenseRemainderLedger": ".batched_dense_tm",
    "DenseTMContract": ".batched_dense_tm",
    "DenseValidatedStep": ".batched_dense_tm",
    "dense_picard_validate_step": ".batched_dense_tm",
    "dense_polynomial_picard": ".batched_dense_tm",
    "dense_to_sparse_tmvector": ".batched_dense_tm",
    "build_dense_subdivision_cover": ".batched_dense_tm",
    "canonicalize_dense_polynomial": ".batched_dense_tm",
    "evaluate_dense_horner_range": ".batched_dense_tm",
    "evaluate_dense_registered_horner_range": ".batched_dense_tm",
    "registered_dense_horner_orders": ".batched_dense_tm",
    "sparse_tmvector_to_dense": ".batched_dense_tm",
    "validate_dense_subdivision_cover": ".batched_dense_tm",
    "AffineCoordinateBasis": ".common_basis",
    "CommonBasisTransformResult": ".common_basis",
    "IntervalPolynomialBatch": ".common_basis",
    "affine_common_basis_transform": ".common_basis",
    "evaluate_common_basis_point": ".common_basis",
    "FixedSupportDescriptor": ".fixed_support",
    "FixedSupportDRPicardResult": ".fixed_support",
    "FixedSupportIntegrationRoute": ".fixed_support",
    "FixedSupportInterval": ".fixed_support",
    "FixedSupportLedger": ".fixed_support",
    "FixedSupportPolynomial": ".fixed_support",
    "FixedSupportReachResult": ".fixed_support",
    "FixedSupportReachability": ".fixed_support",
    "FixedSupportRoute": ".fixed_support",
    "FixedSupportTaylorModel": ".fixed_support",
    "FixedSupportSymbolicRemainderState": ".fixed_support",
    "FixedSupportSymbolicStepResult": ".fixed_support",
    "FixedSupportValidatedStep": ".fixed_support",
    "diffreach_vdp_polynomial_rhs": ".fixed_support",
    "diffreach_vdp_tm_rhs": ".fixed_support",
    "fixed_support_dr_remainder_picard": ".fixed_support",
    "fixed_support_build_linear_tm": ".fixed_support",
    "fixed_support_identity_parameterization": ".fixed_support",
    "fixed_support_polynomial_picard": ".fixed_support",
    "fixed_support_step_boxes": ".fixed_support",
    "fixed_support_symbolic_step_linear": ".fixed_support",
    "FlowpipeResult": ".flowpipe",
    "FlowpipeSegment": ".flowpipe",
    "FlowstarNormalFlowpipeState": ".flowpipe",
    "HornerInsertionDiagnosticResult": ".flowpipe",
    "flowpipe_multi_step": ".flowpipe",
    "flowpipe_step": ".flowpipe",
    "flowpipe_step_flowstar_style_adaptive": ".flowpipe",
    "flowpipe_step_from_tm": ".flowpipe",
    "preserve_complete_polynomial_carry": ".flowpipe",
    "insert_ctrunc_normal_horner_diagnostic": ".flowpipe",
    "insert_ctrunc_normal_like": ".flowpipe",
    "Interval": ".interval",
    "Polynomial": ".polynomial",
    "PolynomialODE": ".polynomial_ode",
    "PolynomialODETerm": ".polynomial_ode",
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
    "TerminalCheckpoint": ".terminal_checkpoint",
    "load_terminal_checkpoint": ".terminal_checkpoint",
    "save_terminal_checkpoint": ".terminal_checkpoint",
    "tmvector_hashes": ".terminal_checkpoint",
    "TMVector": ".tm_vector",
}

__all__ = list(_PUBLIC_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _PUBLIC_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    attribute_name = "evaluate_point" if name == "evaluate_common_basis_point" else name
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
