from __future__ import annotations

import json

import numpy as np

from bern_feasibility import (
    power_to_bernstein_coefficients,
    run,
)


def test_asymmetric_power_to_bernstein_conversion() -> None:
    coefficients = power_to_bernstein_coefficients(
        {(2,): 1.0, (1,): -2.0, (0,): 1.0},
        ((-1.0, 2.0),),
    )
    np.testing.assert_allclose(coefficients, [4.0, -2.0, 1.0])


def test_cross_term_coefficients_preserve_corner_values() -> None:
    coefficients = power_to_bernstein_coefficients(
        {(1, 1): 1.0},
        ((0.08, 0.12), (0.18, 0.22)),
    )
    np.testing.assert_allclose(
        coefficients,
        [
            [0.08 * 0.18, 0.08 * 0.22],
            [0.12 * 0.18, 0.12 * 0.22],
        ],
    )


def test_feasibility_run_is_explicitly_range_only(tmp_path) -> None:
    result = run(tmp_path, repetitions=2)
    assert result["cases"] == 5
    assert result["all_bernstein_exact_ranges_contained"]
    assert result["strictly_tighter_cases"] >= 1
    assert not result["decision"]["fourth_comparable_reachability_tool"]
    payload = json.loads(
        (tmp_path / "bern_feasibility.json").read_text(encoding="utf-8")
    )
    assert payload["prototype_scope"] == "polynomial_range_query_only"
    header = (tmp_path / "bern_feasibility.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "sampling_semantics" in header
