from __future__ import annotations

import math

from common import load_spec
from export_flowstar_segment import render_cpp
from flowstar_adaptive_trajectory_audit import (
    _reference_solutions,
    _trajectory_failures,
)


def test_first_adaptive_vanderpol_failure_and_native_repair() -> None:
    spec = load_spec()
    references = _reference_solutions(
        spec["systems"]["van_der_pol"], 0.041375
    )
    row = {
        "step": 3,
        "time": 0.04137500000000001,
        "state": 0,
        "export_lower": 1.195701727252073,
        "export_upper": 1.4980826940976244,
        "direct_lower": 1.1946195451854615,
        "direct_upper": 1.4991648761642355,
    }
    collapsed = _trajectory_failures(
        [row],
        references,
        lower_field="export_lower",
        upper_field="export_upper",
        tolerance=float(spec["trajectory_tolerance"]),
    )
    assert len(collapsed) == 1
    failure = collapsed[0]
    assert failure["segment_index"] == 3
    assert failure["state_index"] == 0
    assert failure["reference_lower_initial_point"] == [1.1, 2.35]
    assert math.isclose(
        failure["lower_under_enclosure_gap"],
        8.314335673276219e-7,
        rel_tol=0.0,
        abs_tol=2e-14,
    )

    native = _trajectory_failures(
        [row],
        references,
        lower_field="direct_lower",
        upper_field="direct_upper",
        tolerance=float(spec["trajectory_tolerance"]),
    )
    assert native == []


def test_flowstar_exporter_keeps_collapsed_native_and_repaired_paths_separate() -> None:
    spec = load_spec()
    source = render_cpp(
        spec["systems"]["riccati"],
        h=0.01,
        order=2,
        candidate=1e-4,
        cutoff=1e-15,
        variant="flowstar_root_cause_patch",
    )
    assert "composed.evaluate_time(" in source
    assert "composed.intEval(native_endpoint_box, native_endpoint_domain)" in source
    assert "endpoint.tms[state].remainder +=" not in source
    assert 'print_terms("collapsed", endpoint);' in source
    assert "FS_ENDPOINT_PATH" in source
    assert "repaired Flow* endpoint" not in source
