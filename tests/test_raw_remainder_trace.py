from __future__ import annotations

import pytest

from torch_tm_flowpipe.raw_remainder_trace import (
    NODE_FIELDS,
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
