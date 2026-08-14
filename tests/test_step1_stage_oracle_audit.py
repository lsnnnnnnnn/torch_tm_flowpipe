from __future__ import annotations

import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from torch_tm_flowpipe import DenseRangePolicy, FlowstarNormalFlowpipeState, Interval, PolynomialODE
from torch_tm_flowpipe.batched_dense_tm import (
    DenseExecutionCounters,
    dense_picard_validate_step,
    sparse_tmvector_to_dense,
)
from torch_tm_flowpipe.step1_oracle import formal_true_solution_enclosure


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load("step1_soundness_audit_for_test", "experiments/audit_step1_soundness_and_swaps.py")
EXPORT = _load("step1_torch_export_for_test", "experiments/export_torch_step1_stage_ledger.py")


def _torch_terms(x_radius: float, y_center: float, y_radius: float):
    return {
        "x": {
            (0, 0, 0): EXPORT._number(1.25),
            (0, 1, 0): EXPORT._number(x_radius),
        },
        "y": {
            (0, 0, 0): EXPORT._number(y_center),
            (0, 0, 1): EXPORT._number(y_radius),
        },
    }


def _flowstar_terms(x_radius: float, y_center: float, y_radius: float):
    result = _torch_terms(x_radius, y_center, y_radius)
    return {
        component: {
            exponent: {"coefficient_lower": value, "coefficient_upper": value}
            for exponent, value in terms.items()
        }
        for component, terms in result.items()
    }


def _valid_ledger_row() -> dict:
    return {
        "schema": "common_step_operator_stage_ledger_v1",
        "ledger_row_index": 0,
        "tool": "flowstar_pinned_actual",
        "actual_source": {"file": "x", "function": "f", "line_start": 1, "line_end": 1},
        "stage_id": "picard_polynomial_iteration",
        "iteration": 1,
        "basis_id": "canonical_tau_ux_uy_o4",
        "classification": "UNRESOLVED",
        "input_artifact_hashes": [],
        "output_artifact_hash": "0" * 64,
        "record_type": "polynomial_term",
        "payload": {"canonical_exponents": [0, 0, 0]},
    }


def _write_ledger(path: Path, rows: list[dict]) -> None:
    path.write_text(
        json.dumps({"schema": "common_step_operator_stage_ledger_v1", "rows": rows}),
        encoding="utf-8",
    )


@pytest.mark.unit
def test_binary_runtime_initial_tms_fail_exact_rational_containment() -> None:
    flowstar = AUDIT._initial_witness(
        "flowstar",
        _flowstar_terms(
            float.fromhex("0x1.3333333333330p-3"),
            float.fromhex("0x1.3333333333334p+1"),
            float.fromhex("0x1.9999999999980p-5"),
        ),
        flowstar=True,
    )
    torch_result = AUDIT._initial_witness(
        "torch",
        _torch_terms(
            float.fromhex("0x1.3333333333331p-3"),
            float.fromhex("0x1.3333333333334p+1"),
            float.fromhex("0x1.99999999999a1p-5"),
        ),
        flowstar=False,
    )
    assert flowstar["classification"] == "UNDER_ENCLOSURE_WITNESS"
    assert torch_result["classification"] == "UNDER_ENCLOSURE_WITNESS"
    assert flowstar["components"]["x"]["missing_lower_gap"] == "1/11258999068426240"
    assert torch_result["components"]["x"]["missing_upper_gap"] == "11/180143985094819840"


@pytest.mark.unit
def test_formal_true_solution_is_inside_both_published_step1_boxes() -> None:
    exact = formal_true_solution_enclosure(series_degree=100)
    boxes = {
        "flowstar_endpoint": ((1.1233097182976843, 1.4244225119339453), (2.311664291358734, 2.434298058178654)),
        "torch_endpoint": ((1.1234168579141637, 1.4243153549019569), (2.3123133911465477, 2.4336487312925974)),
        "flowstar_segment": ((1.0993097056396248, 1.4245550681839452), (2.3116590217343096, 2.4610649331786534)),
        "torch_segment": ((1.0993087672378616, 1.4245559891701993), (2.3116476026705368, 2.4610761251441828)),
    }
    expected = {
        "endpoint": (exact.endpoint_x, exact.endpoint_y),
        "segment": (exact.segment_x, exact.segment_y),
    }
    for name, components in boxes.items():
        kind = "endpoint" if name.endswith("endpoint") else "segment"
        for actual, target in zip(components, expected[kind], strict=True):
            outer = tuple(Fraction.from_float(value) for value in actual)
            assert outer[0] <= target.lo and target.hi <= outer[1]


@pytest.mark.unit
@pytest.mark.parametrize("mutation,match", [
    ("missing", "misses"),
    ("duplicate", "duplicate or non-sequential"),
    ("unknown", "unknown classification"),
    ("wrong_dimension", "wrong canonical polynomial dimension"),
])
def test_stage_schema_rejects_missing_duplicate_unknown_and_wrong_dimension(
    tmp_path: Path, mutation: str, match: str
) -> None:
    row = _valid_ledger_row()
    rows = [row]
    if mutation == "missing":
        del row["actual_source"]
    elif mutation == "duplicate":
        rows.append({**row, "ledger_row_index": 0})
    elif mutation == "unknown":
        row["classification"] = "SOUNDS_FINE"
    elif mutation == "wrong_dimension":
        row["payload"]["canonical_exponents"] = [0, 0]
    path = tmp_path / "ledger.json"
    _write_ledger(path, rows)
    with pytest.raises(ValueError, match=match):
        AUDIT._load_ledger(path)


@pytest.mark.unit
def test_torch_production_picard_observer_is_read_only() -> None:
    state = FlowstarNormalFlowpipeState.from_initial_box(
        [Interval(1.1, 1.4), Interval(2.35, 2.45)], 4
    )
    base = sparse_tmvector_to_dense(
        state.normalized_initial_tm(4).extend_domain(Interval(0.0, 0.01)),
        order=4,
        device="cpu",
        dtype=torch.float64,
        counters=DenseExecutionCounters(),
        segment_boundary=True,
        range_policy=DenseRangePolicy(),
        range_trace=[],
    )
    ode = PolynomialODE.from_system_spec(
        {
            "state_names": ["x", "y"],
            "initial_box": [[1.1, 1.4], [2.35, 2.45]],
            "rhs": [
                {"terms": [{"coefficient": 1.0, "powers": [0, 1]}]},
                {"terms": [
                    {"coefficient": 1.0, "powers": [0, 1]},
                    {"coefficient": -1.0, "powers": [1, 0]},
                    {"coefficient": -1.0, "powers": [2, 1]},
                ]},
            ],
        }
    )
    observed_iterations: list[int] = []
    common = dict(
        h=0.01,
        order=4,
        tau_index=2,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        max_validation_attempts=2,
        validation_eps=1e-12,
        validation_mode="flowstar_raw_remainder_compat",
    )
    observed = dense_picard_validate_step(
        ode,
        base,
        polynomial_observer=lambda iteration, _before, _after: observed_iterations.append(iteration),
        **common,
    )
    baseline = dense_picard_validate_step(ode, base, **common)
    assert observed_iterations == [1, 2, 3, 4]
    assert observed.status == baseline.status
    assert observed.trace == baseline.trace
    assert torch.equal(observed.segment_tm.poly.coeffs, baseline.segment_tm.poly.coeffs)
    assert torch.equal(observed.segment_tm.rem_lo, baseline.segment_tm.rem_lo)
    assert torch.equal(observed.segment_tm.rem_hi, baseline.segment_tm.rem_hi)
    assert torch.equal(observed.subset_margin, baseline.subset_margin)
