from __future__ import annotations

from experiments.replay_diffreach_explicit_f64_fixture import _compare


def test_explicit_f64_fixture_comparison_names_exact_mismatch_fields() -> None:
    fields = (
        "poly1_slots",
        "poly2_slots",
        "initial_inclusion_mask",
        "round_masks",
        "round_accepted_lo",
        "round_accepted_hi",
        "tube_lo",
        "tube_hi",
        "endpoint_lo",
        "endpoint_hi",
    )
    expected = {field: [field] for field in fields}
    actual = dict(expected)
    actual["tube_hi"] = ["changed"]
    assert _compare(expected, actual) == ["tube_hi"]
