from __future__ import annotations

import itertools
import copy
import json
import math
from pathlib import Path

from common import (
    affine_project_state,
    analytic_endpoint,
    evaluate_polynomial_point,
    load_spec,
    validate_record,
)
from export_torch_segment import export_segment
from export_flowstar_segment import render_cpp


def test_torch_export_round_trip_and_endpoint_tube_contract() -> None:
    record = export_segment(
        load_spec(), system_name="coupled_quadratic", h=0.01, order=3
    )
    checks = validate_record(record)
    assert checks["passed"], checks
    assert record["native_validation_passed"]
    assert record["schema_version"] == 2
    assert record["system_definition"]["equations"]
    assert record["step_semantics"]["requested_step"] == 0.01
    assert record["step_semantics"]["accepted_step"] == 0.01
    assert record["execution"]["repository_commit"]
    assert record["execution"]["runtime"]["propagation_s"] >= 0.0
    assert record["outcome"]["requested_horizon_reached"]
    for sample in record["native_metadata"]["native_point_samples"]:
        for state_index, expected in enumerate(sample["polynomial_values"]):
            actual = evaluate_polynomial_point(
                record["states"][state_index]["polynomial_terms"],
                sample["point"],
            )
            assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=2e-14)


def test_native_round_trip_mismatch_is_a_hard_record_failure() -> None:
    record = export_segment(
        load_spec(), system_name="riccati", h=0.01, order=2
    )
    broken = copy.deepcopy(record)
    broken["native_metadata"]["native_point_samples"][0][
        "polynomial_values"
    ][0] += 1e-4
    checks = validate_record(broken)
    assert checks["native_point_evaluation_violations"] == 1
    assert not checks["passed"]


def test_coupled_quadratic_activates_cross_terms() -> None:
    record = export_segment(
        load_spec(), system_name="coupled_quadratic", h=0.01, order=3
    )
    tube_exponents = {
        tuple(term["exponents"])
        for state in record["states"]
        for term in state["polynomial_terms"]
    }
    endpoint_exponents = {
        tuple(term["exponents"])
        for state in record["raw_endpoint"]
        for term in state["polynomial_terms"]
    }
    # Variable order is (x1 generator, x2 generator, local time).
    assert (1, 1, 1) in tube_exponents
    assert (1, 1) in endpoint_exponents


def test_affine_projection_contains_original_polynomial() -> None:
    state = {
        "polynomial_terms": [
            {"exponents": [0, 0], "coefficient": 0.25},
            {"exponents": [1, 0], "coefficient": 0.5},
            {"exponents": [1, 1], "coefficient": -0.2},
            {"exponents": [0, 2], "coefficient": 0.1},
        ],
        "independent_interval_remainder": [-0.01, 0.02],
    }
    domains = [[-1.0, 1.0], [-0.5, 0.75]]
    projected, discarded = affine_project_state(state, domains)
    assert {tuple(item["exponents"]) for item in discarded} == {(1, 1), (0, 2)}
    for point in itertools.product(*[(domain[0], domain[1]) for domain in domains]):
        original = evaluate_polynomial_point(state["polynomial_terms"], point)
        affine = evaluate_polynomial_point(projected["polynomial_terms"], point)
        residual = original - affine
        rem = projected["independent_interval_remainder"]
        assert rem[0] <= residual + state["independent_interval_remainder"][0]
        assert rem[1] >= residual + state["independent_interval_remainder"][1]


def test_analytic_references() -> None:
    riccati = analytic_endpoint("riccati", [[0.0, 0.1]], 0.01)
    assert riccati is not None
    assert riccati[0][0] == 0.0
    assert math.isclose(riccati[0][1], 0.1 / 0.999)
    harmonic = analytic_endpoint(
        "harmonic", [[-0.1, 0.1], [-0.1, 0.1]], math.pi / 2
    )
    assert harmonic is not None
    assert math.isclose(harmonic[0][0], -0.1, abs_tol=1e-15)
    assert math.isclose(harmonic[1][1], 0.1, abs_tol=1e-15)


def test_flowstar_exporter_sets_requested_mpfr_precision() -> None:
    spec = load_spec()
    source = render_cpp(
        spec["systems"]["riccati"],
        h=0.01,
        order=2,
        candidate=1e-4,
        cutoff=1e-15,
        variant="flowstar_stock",
        precision_bits=256,
    )
    assert "intervalNumPrecision = 256;" in source


def test_vanderpol_state_semantics_are_explicit() -> None:
    spec = load_spec()
    assert spec["systems"]["van_der_pol"]["state_names"] == [
        "position",
        "velocity",
    ]


def test_machine_readable_cir_schema_tracks_validator_contract() -> None:
    schema = json.loads(
        (Path(__file__).parents[1] / "cir_schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["properties"]["schema_version"]["const"] == 2
    assert {
        "system_definition",
        "time_semantics",
        "step_semantics",
        "variable_semantics",
        "dependency_semantics",
        "polynomial_representation",
        "enclosures",
        "reset_carry_policy",
        "outcome",
        "execution",
    }.issubset(schema["required"])
