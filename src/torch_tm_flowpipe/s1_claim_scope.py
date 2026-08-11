"""Decisive claim-scope audit for the complete-O4 S1 prefix.

The exact-rational structured-image oracle qualifies one post-hoc interval
primitive.  It does not qualify the retained polynomial coefficient arithmetic
that constructs and transports the polynomial consumed by that primitive.
This module makes that distinction machine-readable and deliberately contains
no heuristic inference from passing prefix tests to a stronger formal claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SAFEGUARDED_PREFIX_CLASS = "safeguarded_binary64_interval_shell"
SAFEGUARDED_PREFIX_CONDITION = "conditional_on_retained_coefficient_arithmetic"


@dataclass(frozen=True)
class RetainedCoefficientOperationAudit:
    operation: str
    implementation_paths: tuple[str, ...]
    coefficient_roundoff_added_to_remainder: bool
    independent_exact_or_outward_decisive_replay: bool
    finding: str

    @property
    def closed(self) -> bool:
        return bool(
            self.coefficient_roundoff_added_to_remainder
            or self.independent_exact_or_outward_decisive_replay
        )


RETAINED_COEFFICIENT_AUDIT = (
    RetainedCoefficientOperationAudit(
        "ordinary_multiplication",
        (
            "torch_tm_flowpipe.batched_dense_tm.BatchedPolynomial.mul_trunc",
            "torch_tm_flowpipe.batched_dense_tm.BatchedTaylorModel.mul_trunc",
        ),
        False,
        False,
        "retained binary64 products are accumulated as point coefficients; interval safeguards enclose dropped ranges, not retained coefficient error",
    ),
    RetainedCoefficientOperationAudit(
        "scatter_add_aggregation",
        (
            "torch_tm_flowpipe.batched_dense_tm._merge_coefficients_by_index",
            "torch_tm_flowpipe.batched_dense_tm.BatchedPolynomial.mul_trunc",
            "torch_tm_flowpipe.batched_dense_tm.BatchedPolynomial.integrate",
            "torch_tm_flowpipe.batched_dense_tm.BatchedPolynomial.substitute_const_and_drop",
        ),
        False,
        False,
        "scatter_add_ produces retained point coefficients without an operation-level directed-rounding or exact replay certificate",
    ),
    RetainedCoefficientOperationAudit(
        "integration_coefficient_multiply_add",
        ("torch_tm_flowpipe.batched_dense_tm.BatchedPolynomial.integrate",),
        False,
        False,
        "integration factors and retained aggregation use ordinary binary64 arithmetic",
    ),
    RetainedCoefficientOperationAudit(
        "cutoff",
        (
            "torch_tm_flowpipe.batched_dense_tm.BatchedPolynomial.apply_cutoff",
            "torch_tm_flowpipe.flowpipe._cutoff_polynomial_normal",
        ),
        False,
        False,
        "removed terms are interval-ranged, but rounding already present in retained coefficients is not separately owned",
    ),
    RetainedCoefficientOperationAudit(
        "affine_map",
        ("torch_tm_flowpipe.batched_dense_tm.BatchedPolynomial.affine_map",),
        False,
        False,
        "einsum and constant updates produce point retained coefficients without a coefficient-error remainder",
    ),
    RetainedCoefficientOperationAudit(
        "picard_iteration_coefficient_update",
        (
            "torch_tm_flowpipe.batched_dense_tm.dense_polynomial_picard",
            "torch_tm_flowpipe.flowpipe._picard_polynomial",
        ),
        False,
        False,
        "the Picard polynomial iterate discards arithmetic remainder from the retained coefficient update by design",
    ),
    RetainedCoefficientOperationAudit(
        "dense_to_sparse_boundary_conversion",
        ("torch_tm_flowpipe.batched_dense_tm.dense_to_sparse_tmvector",),
        False,
        False,
        "binary64 coefficients are copied exactly between representations, but the arithmetic that created them remains unqualified",
    ),
    RetainedCoefficientOperationAudit(
        "sparse_normalized_insertion_coefficient_arithmetic",
        (
            "torch_tm_flowpipe.flowpipe._compose_term_with_inner",
            "torch_tm_flowpipe.flowpipe._insert_ctrunc_normal_like_scalar",
        ),
        False,
        False,
        "sparse polynomial add/multiply constructs retained point coefficients without decisive exact/outward replay of the prefix workload",
    ),
)


def complete_o4_claim_scope_audit() -> dict[str, Any]:
    """Return the only claim classification supported by current evidence."""
    operations = [
        {**asdict(item), "closed": item.closed}
        for item in RETAINED_COEFFICIENT_AUDIT
    ]
    retained_closed = all(item.closed for item in RETAINED_COEFFICIENT_AUDIT)
    return {
        "primitive": "complete_polynomial_structured_image",
        "primitive_scope": "given finite CPU binary64 coefficients, degree <= 4",
        "primitive_scope_formal_eligible": True,
        "typed_ledger_additive_containment": True,
        "checkpoint_byte_identity": True,
        "retained_coefficient_arithmetic_closed": retained_closed,
        "prefix_soundness_class": SAFEGUARDED_PREFIX_CLASS,
        "prefix_condition": SAFEGUARDED_PREFIX_CONDITION,
        "prefix_formal_eligible": bool(retained_closed),
        "real_ode_164_boundary_formal_eligible": bool(retained_closed),
        "operations": operations,
    }


__all__ = [
    "RETAINED_COEFFICIENT_AUDIT",
    "SAFEGUARDED_PREFIX_CLASS",
    "SAFEGUARDED_PREFIX_CONDITION",
    "RetainedCoefficientOperationAudit",
    "complete_o4_claim_scope_audit",
]
