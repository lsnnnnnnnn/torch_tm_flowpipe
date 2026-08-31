from __future__ import annotations

import pytest
import torch

from experiments.profile_c4_reference_solver import (
    PROFILE_BUCKETS,
    _TensorResultCounter,
    _bucket,
)


@pytest.mark.unit
def test_profile_declares_every_required_exclusive_bucket() -> None:
    assert PROFILE_BUCKETS == (
        "polynomial Picard construction",
        "initial raw-remainder image",
        "post-accept remainder replays",
        "range bounding/subdivision",
        "polynomial multiplication/truncation/cutoff",
        "SR history propagation",
        "normalization/right-map/reset",
        "outward interval/roundoff accounting",
        "Python orchestration/allocation",
        "audit/serialization",
        "other",
    )


@pytest.mark.unit
def test_profile_classification_prioritizes_reference_stages() -> None:
    assert _bucket("accepted_boundary_sr.py", "prepare_accepted_boundary_sr") == (
        "SR history propagation"
    )
    assert _bucket("batched_dense_tm.py", "_post_accept_refine_raw_remainder") == (
        "post-accept remainder replays"
    )
    assert _bucket("batched_dense_tm.py", "_range_for_terms_with_policy") == (
        "range bounding/subdivision"
    )


@pytest.mark.unit
def test_profile_counts_logical_temporary_tensor_results_and_bytes() -> None:
    counter = _TensorResultCounter()
    with counter:
        value = torch.ones((2, 3), dtype=torch.float64)
        result = value + value
    assert result.shape == (2, 3)
    assert counter.count >= 2
    assert counter.logical_bytes >= 2 * 2 * 3 * 8
