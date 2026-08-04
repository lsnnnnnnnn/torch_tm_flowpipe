from __future__ import annotations

import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = ROOT / "experiments" / "three_tool_deep_study" / "export_torch_segment.py"
ANALYZE_SCRIPT = ROOT / "experiments" / "three_tool_reaudit" / "analyze_one_step.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPORT = _load("export_torch_segment_for_test", EXPORT_SCRIPT)
ANALYZE = _load("analyze_one_step_for_test", ANALYZE_SCRIPT)


def _spec():
    return EXPORT.load_spec(ROOT / "benchmarks" / "three_tool_one_step.yaml")


def test_normalized_source_prevents_coordinate_dependent_false_rejection() -> None:
    physical = EXPORT.export_segment(
        _spec(),
        system_name="scalar_quadratic",
        h=0.01,
        order=4,
        source_coordinates="physical_identity",
        validation_mode="flowstar_raw_remainder_compat",
        candidate_remainder=1.0e-4,
        cutoff=1.0e-15,
    )
    normalized = EXPORT.export_segment(
        _spec(),
        system_name="scalar_quadratic",
        h=0.01,
        order=4,
        source_coordinates="flowstar_normalized",
        validation_mode="flowstar_raw_remainder_compat",
        candidate_remainder=1.0e-4,
        cutoff=1.0e-15,
    )
    assert physical["outcome"]["status"] == "failure"
    assert normalized["outcome"]["status"] == "success"
    assert normalized["domains"] == [[-1.0, 1.0], [0.0, 0.01]]


def test_normalized_scalar_affine_endpoint_contains_analytic_range() -> None:
    record = EXPORT.export_segment(
        _spec(),
        system_name="scalar_affine",
        h=0.01,
        order=4,
        source_coordinates="flowstar_normalized",
        validation_mode="flowstar_raw_remainder_compat",
        candidate_remainder=1.0e-4,
        cutoff=1.0e-15,
    )
    lower = (0.0 + 0.5) * math.exp(0.02) - 0.5
    upper = (0.1 + 0.5) * math.exp(0.02) - 0.5
    exported = record["raw_endpoint_box"][0]
    assert exported[0] <= lower
    assert exported[1] >= upper


def test_canonical_exponent_moves_local_time_last() -> None:
    assert ANALYZE.canonical_exponent([3, 1, 2], ["local_time", "state_generator", "state_generator"]) == (1, 2, 3)


def test_box_containment_is_directional() -> None:
    assert ANALYZE.box_contains([[0.0, 2.0]], [[0.5, 1.5]])
    assert not ANALYZE.box_contains([[0.5, 1.5]], [[0.0, 2.0]])


def test_current_run_trajectory_sanity_is_fail_closed() -> None:
    evidence = __import__("json").loads(
        (
            ROOT
            / "outputs"
            / "three_tool_reaudit"
            / "20260804T060058Z"
            / "gate_evidence"
            / "one_step_parity.json"
        ).read_text(encoding="utf-8")
    )
    scalar = evidence["cases"]["scalar_affine_o4_h001"]["trajectory_sanity"]
    assert scalar["flowstar"]["passed"] is False
    assert scalar["flowstar"]["endpoint_violations"]
    assert evidence["gate_passed"] is False
