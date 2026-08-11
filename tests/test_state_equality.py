from __future__ import annotations

from dataclasses import replace

import torch

from torch_tm_flowpipe import FlowpipeSegment, Interval, TMVector
from torch_tm_flowpipe.state_equality import compare_accepted_segment_states


def _segment(candidate_hi: float = 0.01) -> FlowpipeSegment:
    domain = [Interval(-1.0, 1.0)]
    vector = TMVector.constants([1.25], domain, order=4)
    return FlowpipeSegment(
        tm=vector,
        final_tm=vector,
        reset_tm=vector,
        status="validated",
        h=0.01,
        order=4,
        validation_attempts=1,
        candidate_remainder=[[-0.01], [candidate_hi]],
        picard_image_remainder=[[-0.005], [0.005]],
        endpoint_publication_mask=torch.tensor([True]),
        tube_publication_mask=torch.tensor([True]),
    )


def test_identical_accepted_states_are_bit_exact_with_full_records():
    result = compare_accepted_segment_states(_segment(), _segment())
    assert result["status"] == "pass"
    assert result["natural_state_sha256"] == result["fixed_state_sha256"]
    assert result["field_count"] == len(result["comparisons"])
    coefficient = next(
        row for row in result["comparisons"] if ".terms[" in row["path"]
    )
    assert coefficient["natural"]["shape"] == []
    assert coefficient["natural"]["dtype"] == "torch.float64"
    assert coefficient["natural"]["device"] == "cpu"
    assert coefficient["natural"]["values_hex"]
    assert coefficient["natural"]["raw_bytes_hex"]


def test_first_accepted_state_mismatch_fails_closed():
    result = compare_accepted_segment_states(_segment(), _segment(0.02))
    assert result["status"] == "fail"
    assert result["first_mismatch"] is not None
    assert "candidate_remainder" in result["first_mismatch"]["path"]
    assert result["outward_containment_relations"] == []
