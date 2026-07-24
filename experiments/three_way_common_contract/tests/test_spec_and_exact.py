from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from common import (
    PROTOCOL_A,
    PROTOCOL_B,
    PROTOCOL_C,
    exact_endpoint,
    exact_tube,
    iter_configurations,
    load_spec,
)


def test_configuration_matrix_matches_contract() -> None:
    spec = load_spec()
    full = list(iter_configurations(spec))
    smoke = list(iter_configurations(spec, smoke=True))
    assert len(full) == 20
    assert len(smoke) == 9
    assert sum(item["protocol"] == PROTOCOL_A for item in full) == 12
    assert sum(item["protocol"] == PROTOCOL_B for item in full) == 4
    assert sum(item["protocol"] == PROTOCOL_C for item in full) == 4
    vdp_a = [
        item["h"]
        for item in full
        if item["protocol"] == PROTOCOL_A
        and item["system"] == "van_der_pol"
    ]
    assert vdp_a == [0.0025, 0.005, 0.01, 0.02]


def test_riccati_exact_reference() -> None:
    box = [[0.0, 0.1]]
    endpoint = exact_endpoint("riccati", 1.0, box)
    tube = exact_tube("riccati", 0.5, 1.0, box)
    assert endpoint is not None and tube is not None
    assert endpoint[0][0] == 0.0
    assert tube[0][0] == 0.0
    assert math.isclose(endpoint[0][1], 1.0 / 9.0, abs_tol=1e-15)
    assert math.isclose(tube[0][1], 1.0 / 9.0, abs_tol=1e-15)


def test_harmonic_exact_reference_and_tube() -> None:
    box = [[-0.1, 0.1], [-0.1, 0.1]]
    endpoint = exact_endpoint("harmonic", math.pi / 2, box)
    assert endpoint is not None
    assert math.isclose(endpoint[0][0], -0.1, abs_tol=1e-15)
    assert math.isclose(endpoint[0][1], 0.1, abs_tol=1e-15)
    tube = exact_tube("harmonic", 0.0, math.pi / 4, box)
    assert tube is not None
    expected = math.sqrt(2) * 0.1
    assert math.isclose(tube[0][0], -expected, abs_tol=1e-15)
    assert math.isclose(tube[0][1], expected, abs_tol=1e-15)
