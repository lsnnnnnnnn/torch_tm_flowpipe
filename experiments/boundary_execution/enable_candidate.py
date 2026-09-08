"""Pytest-only opt-in for the inherited endpoint/carry regression files."""
import pytest
from torch_tm_flowpipe.prepared_remainder_replay import prepared_remainder_replay
from torch_tm_flowpipe.packed_boundary_range import packed_boundary_execution


@pytest.fixture(autouse=True)
def actual_candidate_entry():
    with prepared_remainder_replay(True), packed_boundary_execution(True):
        yield
