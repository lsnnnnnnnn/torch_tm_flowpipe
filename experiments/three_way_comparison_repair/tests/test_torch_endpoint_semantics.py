from __future__ import annotations

import math

import numpy as np
import torch
from scipy.integrate import solve_ivp

from torch_tm_flowpipe import Interval, TMVector, flowpipe_step

from common import PROTOCOL_BOX, PROTOCOL_RAW, load_spec
from run_torch_audit import run_case

torch.set_default_dtype(torch.float64)


def _harmonic(state: TMVector, control: TMVector | None = None) -> TMVector:
    del control
    return TMVector([state[1], -state[0]])


def _riccati(state: TMVector, control: TMVector | None = None) -> TMVector:
    del control
    return TMVector([state[0] * state[0]])


def _contains(outer: Interval, lower: float, upper: float, tol: float = 1e-11) -> None:
    assert outer.lower.item() <= lower + tol
    assert outer.upper.item() >= upper - tol


def test_raw_and_tightened_endpoints_are_distinct_and_raw_is_in_tube() -> None:
    segment = flowpipe_step(
        _riccati,
        [Interval(0.0, 0.1)],
        h=0.01,
        order=1,
        validation_mode="growth",
    )
    assert segment.status == "validated"
    assert segment.endpoint_raw_tm is not None
    assert segment.endpoint_tightened_tm is not None
    assert segment.endpoint_raw_tm is not segment.endpoint_tightened_tm
    assert segment.final_tm is segment.endpoint_tightened_tm
    assert segment.endpoint_tightening_applied
    assert (
        segment.endpoint_tightening_validation_method
        == "fixed_time_picard_residual_interval_evaluation"
    )
    tube = segment.tm.range_box()[0]
    raw = segment.endpoint_raw_tm.range_box()[0]
    assert tube.contains_interval(raw, tol=1e-12)


def test_analytic_riccati_raw_and_tightened_containment() -> None:
    h = 0.01
    segment = flowpipe_step(
        _riccati,
        [Interval(0.0, 0.1)],
        h=h,
        order=1,
        validation_mode="growth",
    )
    exact = 0.1 / (1.0 - 0.1 * h)
    assert segment.endpoint_raw_tm is not None
    assert segment.endpoint_tightened_tm is not None
    _contains(segment.endpoint_raw_tm.range_box()[0], 0.0, exact)
    _contains(segment.endpoint_tightened_tm.range_box()[0], 0.0, exact)


def test_analytic_harmonic_raw_and_tightened_containment() -> None:
    h = 0.01
    segment = flowpipe_step(
        _harmonic,
        [Interval(-0.1, 0.1), Interval(-0.1, 0.1)],
        h=h,
        order=1,
        validation_mode="growth",
    )
    exact_radius = 0.1 * (abs(math.cos(h)) + abs(math.sin(h)))
    assert segment.endpoint_raw_tm is not None
    assert segment.endpoint_tightened_tm is not None
    for endpoint in (segment.endpoint_raw_tm, segment.endpoint_tightened_tm):
        for interval in endpoint.range_box():
            _contains(interval, -exact_radius, exact_radius)


def test_seeded_random_polynomial_odes_contain_sampled_endpoints() -> None:
    rng = np.random.default_rng(20260728)
    for _ in range(8):
        coefficients = rng.uniform(-0.4, 0.4, size=4)
        lower = float(rng.uniform(-0.15, -0.02))
        upper = float(rng.uniform(0.02, 0.15))
        h = 0.002

        def rhs(state: TMVector, control: TMVector | None = None) -> TMVector:
            del control
            x = state[0]
            return TMVector(
                [
                    float(coefficients[0])
                    + float(coefficients[1]) * x
                    + float(coefficients[2]) * x * x
                    + float(coefficients[3]) * x * x * x
                ]
            )

        segment = flowpipe_step(
            rhs,
            [Interval(lower, upper)],
            h=h,
            order=3,
            validation_mode="growth",
        )
        assert segment.status == "validated"
        assert segment.endpoint_raw_tm is not None
        assert segment.endpoint_tightened_tm is not None
        raw = segment.endpoint_raw_tm.range_box()[0]
        tightened = segment.endpoint_tightened_tm.range_box()[0]

        def numeric_rhs(_: float, state: np.ndarray) -> np.ndarray:
            x = state[0]
            return np.asarray(
                [
                    coefficients[0]
                    + coefficients[1] * x
                    + coefficients[2] * x**2
                    + coefficients[3] * x**3
                ]
            )

        for initial in np.linspace(lower, upper, 9):
            exact = solve_ivp(
                numeric_rhs,
                (0.0, h),
                np.asarray([initial]),
                method="DOP853",
                rtol=1e-13,
                atol=1e-15,
            ).y[0, -1]
            assert raw.contains(float(exact), tol=1e-11)
            assert tightened.contains(float(exact), tol=1e-11)


def test_primary_and_common_box_protocols_use_raw_endpoint() -> None:
    spec = load_spec()
    raw_rows, _ = run_case(
        spec,
        system_name="riccati",
        protocol=PROTOCOL_RAW,
        h=0.01,
        horizon=0.01,
    )
    box_rows, _ = run_case(
        spec,
        system_name="riccati",
        protocol=PROTOCOL_BOX,
        h=0.01,
        horizon=0.02,
    )
    for rows in (raw_rows, box_rows):
        primary = [row for row in rows if row["interval_kind"] == "endpoint_raw"]
        assert primary
        assert all(not row["endpoint_tightening_applied"] for row in primary)
        assert all(
            row["endpoint_semantics"] == "raw_substitution_tau_equals_h"
            for row in primary
        )
