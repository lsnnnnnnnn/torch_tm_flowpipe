from __future__ import annotations

import pytest
import torch

from torch_tm_flowpipe.batched_dense_tm import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    BatchedTaylorModel,
)
from torch_tm_flowpipe.raw_remainder_trace import (
    NODE_FIELDS,
    dropped_product_support_sha,
    interval_record,
    validate_expression_dag,
)


def _node(node_id: str, parents: list[str]) -> dict[str, object]:
    node = {field: None for field in NODE_FIELDS}
    node["expression_node_id"] = node_id
    node["parent_node_ids"] = parents
    return node


def test_interval_record_has_decimal_and_binary64_hex() -> None:
    record = interval_record(-0.1, 0.2)
    assert record["lo"]["hex"] == (-0.1).hex()
    assert record["hi"]["hex"] == (0.2).hex()


def test_expression_dag_requires_prior_parents_and_unique_ids() -> None:
    validate_expression_dag([_node("input", []), _node("output", ["input"])])
    with pytest.raises(ValueError, match="non-prior"):
        validate_expression_dag([_node("output", ["input"]), _node("input", [])])
    with pytest.raises(ValueError, match="duplicate"):
        validate_expression_dag([_node("same", []), _node("same", [])])


def test_expression_dag_rejects_missing_common_field() -> None:
    node = _node("input", [])
    del node["decision"]
    with pytest.raises(ValueError, match="missing fields"):
        validate_expression_dag([node])


def test_dropped_product_support_sha_tracks_active_overflow_exponents() -> None:
    basis = BatchedMonomialBasis.build(2, 4, "cpu")
    left_coeffs = torch.zeros((1, 1, basis.num_terms), dtype=torch.float64)
    right_coeffs = torch.zeros_like(left_coeffs)
    left_coeffs[..., basis.term_index((3, 0))] = 0.1
    right_coeffs[..., basis.term_index((2, 0))] = 0.3
    domain_lo = torch.tensor([[-1.0, -1.0]], dtype=torch.float64)
    domain_hi = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    zero = torch.zeros((1, 1), dtype=torch.float64)
    left = BatchedTaylorModel(
        BatchedPolynomial(left_coeffs, basis), zero, zero, domain_lo, domain_hi
    )
    right = BatchedTaylorModel(
        BatchedPolynomial(right_coeffs, basis), zero, zero, domain_lo, domain_hi
    )
    active = dropped_product_support_sha(left, right, max_degree=4)
    inactive = dropped_product_support_sha(left, right, max_degree=5)
    assert len(active) == 64
    assert len(inactive) == 64
    assert active != inactive
