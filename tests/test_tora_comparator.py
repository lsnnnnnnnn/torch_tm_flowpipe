from __future__ import annotations

import pytest

from scripts.compare_tora_q3_common_control import (
    assert_aligned_segment_keys,
)


@pytest.mark.unit
def test_tora_comparator_rejects_time_or_leaf_misalignment() -> None:
    reference = {
        "segment_index": 1,
        "physical_time": 0.1,
        "controller_period": 1,
        "local_segment": 1,
        "leaf_id": list(range(48)),
    }
    assert_aligned_segment_keys(reference, dict(reference))
    for field, value in (
        ("physical_time", 0.10000000000000002),
        ("leaf_id", list(reversed(range(48)))),
    ):
        candidate = dict(reference)
        candidate[field] = value
        with pytest.raises(ValueError, match="exact alignment failed"):
            assert_aligned_segment_keys(reference, candidate)
