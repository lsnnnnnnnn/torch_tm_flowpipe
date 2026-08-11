from __future__ import annotations

import copy

import pytest

from torch_tm_flowpipe.comparison_contract import (
    binary64_record,
    canonical_sha256,
    validate_binary64_record,
    validate_comparison_row,
    validate_pairwise_object_identity,
    vdp_identity_hashes,
    vdp_initial_set_identity,
    vdp_partition_identity,
    vdp_rhs_identity,
)


def _row(**updates):
    row = {
        "tool": "torch",
        "execution_kind": "matched",
        "track": "M-F",
        "partition_count": 1,
        "partition_sha256": vdp_identity_hashes()["partition_b1_sha256"],
        "output_object": "endpoint",
        "requested_horizon": 0.1,
        "validated_horizon": 0.1,
        "completion_status": "completed",
        "formal_ranking_eligible": False,
    }
    row.update(updates)
    return row


def test_vdp_rhs_initial_set_and_partitions_have_actual_stable_hashes():
    hashes = vdp_identity_hashes()
    assert hashes["rhs_sha256"] == canonical_sha256(vdp_rhs_identity())
    assert hashes["initial_set_sha256"] == canonical_sha256(
        vdp_initial_set_identity()
    )
    assert hashes["partition_b1_sha256"] == canonical_sha256(
        vdp_partition_identity(1)
    )
    assert hashes["partition_b64_sha256"] == canonical_sha256(
        vdp_partition_identity(64)
    )
    assert all(len(value) == 64 and value != "pending_manifest_generation" for value in hashes.values())


@pytest.mark.parametrize("value", [0.0, -0.0, 1e-10, 1e-4, 0.01, 10.0])
def test_binary64_decimal_and_hex_round_trip(value):
    record = binary64_record(value)
    assert validate_binary64_record(record).hex() == float(value).hex()
    broken = dict(record)
    broken["hex"] = (value + 1.0).hex()
    with pytest.raises(ValueError, match="disagree"):
        validate_binary64_record(broken)


def test_endpoint_and_tube_labels_cannot_mix():
    endpoint = _row(tool="flowstar")
    tube = _row(tool="torch", output_object="segment_tube")
    with pytest.raises(ValueError, match="same track"):
        validate_pairwise_object_identity([endpoint, tube])


def test_incomplete_horizon_cannot_rank():
    with pytest.raises(ValueError, match="incomplete horizon"):
        validate_comparison_row(
            _row(
                validated_horizon=0.05,
                completion_status="partial",
                formal_ranking_eligible=True,
            )
        )


def test_b1_and_b64_cannot_be_compared_as_same_object():
    b1 = _row(tool="flowstar")
    b64 = _row(
        tool="torch",
        partition_count=64,
        partition_sha256=vdp_identity_hashes()["partition_b64_sha256"],
    )
    with pytest.raises(ValueError, match="same track"):
        validate_pairwise_object_identity([b1, b64])


def test_native_matched_diagnostic_labels_are_mandatory():
    missing = _row()
    del missing["execution_kind"]
    with pytest.raises(ValueError, match="missing labels"):
        validate_comparison_row(missing)
    with pytest.raises(ValueError, match="diagnostic_only"):
        validate_comparison_row(_row(execution_kind="diagnostic"))
    valid = _row(
        execution_kind="diagnostic",
        diagnostic_only=True,
        formal_ranking_eligible=False,
    )
    assert validate_comparison_row(valid)["diagnostic_only"]
