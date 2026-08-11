from torch_tm_flowpipe.s1_claim_scope import (
    SAFEGUARDED_PREFIX_CLASS,
    SAFEGUARDED_PREFIX_CONDITION,
    complete_o4_claim_scope_audit,
)


def test_primitive_eligibility_cannot_be_extrapolated_to_prefix():
    audit = complete_o4_claim_scope_audit()
    assert audit["primitive_scope_formal_eligible"] is True
    assert audit["retained_coefficient_arithmetic_closed"] is False
    assert audit["prefix_formal_eligible"] is False
    assert audit["real_ode_164_boundary_formal_eligible"] is False
    assert audit["prefix_soundness_class"] == SAFEGUARDED_PREFIX_CLASS
    assert audit["prefix_condition"] == SAFEGUARDED_PREFIX_CONDITION


def test_all_required_retained_coefficient_operations_are_decisively_classified():
    audit = complete_o4_claim_scope_audit()
    required = {
        "ordinary_multiplication",
        "scatter_add_aggregation",
        "integration_coefficient_multiply_add",
        "cutoff",
        "affine_map",
        "picard_iteration_coefficient_update",
        "dense_to_sparse_boundary_conversion",
        "sparse_normalized_insertion_coefficient_arithmetic",
    }
    operations = {row["operation"]: row for row in audit["operations"]}
    assert set(operations) == required
    for row in operations.values():
        assert row["implementation_paths"]
        assert row["finding"]
        assert row["closed"] is False
