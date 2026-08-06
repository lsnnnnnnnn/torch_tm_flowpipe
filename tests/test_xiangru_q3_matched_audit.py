from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

from torch_tm_flowpipe.polynomial import Polynomial
from torch_tm_flowpipe.protocol.q3_audit import (
    REQUIRED_CONTRACT_FIELDS,
    align_exact_endpoint_times,
    complete_total_degree_exponents,
    deterministic_json_bytes,
    formal_match_decision,
    horizon_row,
    map_coordinates,
    parse_xiangru_runtime,
    reject_formal_interpolation,
    tagged_enclosure,
    total_degree_retained,
    validate_contract,
    width_ratio,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/xiangru_q3_matched_audit_20260806"
XIANGRU = Path("/srv/local/shengenli/CROWN-Reach_Development_native_27d2905")


def load_upstream_fixed_basis():
    module_dir = XIANGRU / "experiments/remainder_ablation"
    sys.path.insert(0, str(module_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "audit_upstream_generic_fixed_basis", module_dir / "generic_fixed_basis.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(module_dir))


@pytest.mark.unit
@pytest.mark.protocol
def test_frozen_q3_config_and_entrypoint_extract_exact_controls() -> None:
    config = json.loads(
        (OUT / "xiangru_reproduction/fresh_q3_b48_t20/config_resolved.json").read_text()
    )
    assert config["n_total_steps"] == 200
    assert config["n_steps_per_control"] == 10
    assert config["step_size"] == 0.1
    assert config["frr_rounds"] == 10
    assert config["initial_set"] == [
        [0.6, 0.7], [-0.7, -0.6], [-0.4, -0.3], [0.5, 0.6], [0.0, 0.0]
    ]
    source = (XIANGRU / "experiments/remainder_ablation/run_s0_tora_static_partition_sweep.py").read_text()
    assert 'elif method == "complete_q3":' in source
    assert "support = complete_total_degree_support(3)" in source


@pytest.mark.unit
@pytest.mark.protocol
def test_upstream_q3_and_torch_order3_retain_the_same_checked_monomials() -> None:
    upstream = load_upstream_fixed_basis()
    support = upstream.complete_total_degree_support(3, variables=2)
    assert len(upstream.complete_total_degree_support(3).exponents) == math.comb(9, 3) == 84
    exponents = [
        (0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0),
        (0, 3), (1, 2), (2, 1), (3, 0), (0, 4),
    ]
    polynomial = Polynomial(
        {exponent: torch.tensor(float(index + 1), dtype=torch.float64) for index, exponent in enumerate(exponents)},
        n_vars=2,
    )
    kept, dropped = polynomial.truncate(3)
    for exponent in exponents:
        upstream_retained = exponent in support.exponents
        assert upstream_retained == total_degree_retained(exponent, 3)
        assert upstream_retained == (exponent in kept.terms)
        assert (not upstream_retained) == (exponent in dropped.terms)
    assert set(support.exponents) == set(complete_total_degree_exponents(3, 2))


@pytest.mark.unit
@pytest.mark.protocol
def test_contract_schema_has_every_required_evidenced_field() -> None:
    for name in (
        "xiangru_native_contract.json", "torch_candidate_contract.json",
        "flowstar_candidate_contract.json",
    ):
        contract = json.loads((OUT / "contract" / name).read_text())
        assert not validate_contract(contract)
        assert set(REQUIRED_CONTRACT_FIELDS).issubset(contract["fields"])


@pytest.mark.unit
@pytest.mark.protocol
def test_unknown_or_mismatched_contract_fails_closed() -> None:
    contract = json.loads((OUT / "contract/xiangru_native_contract.json").read_text())
    contract["tool"] = "mutated"
    contract["fields"]["dynamics"]["matched"] = "unknown"
    decision = formal_match_decision([contract])
    assert not decision["formal_comparison_authorized"]
    assert any(row["field"] == "dynamics" for row in decision["blockers"])
    del contract["fields"]["initial_set"]
    decision = formal_match_decision([contract])
    assert not decision["formal_comparison_authorized"]
    assert any(row["field"] == "schema" for row in decision["blockers"])


@pytest.mark.unit
@pytest.mark.protocol
def test_endpoint_and_tube_types_cannot_collide() -> None:
    assert tagged_enclosure("endpoint", 0.1, 0.1, [[-1, 1]])["kind"] == "endpoint"
    assert tagged_enclosure("tube", 0.0, 0.1, [[-1, 1]])["kind"] == "tube"
    with pytest.raises(ValueError, match="single physical time"):
        tagged_enclosure("endpoint", 0.0, 0.1, [[-1, 1]])
    with pytest.raises(ValueError, match="nonempty"):
        tagged_enclosure("tube", 0.1, 0.1, [[-1, 1]])


@pytest.mark.unit
@pytest.mark.protocol
def test_coordinate_mapping_requires_a_named_bijection() -> None:
    assert map_coordinates([[1, 2], [3, 4]], ["x", "y"], ["y", "x"]) == [[3.0, 4.0], [1.0, 2.0]]
    with pytest.raises(ValueError, match="differ"):
        map_coordinates([[1, 2], [3, 4]], ["x", "y"], ["x", "z"])


@pytest.mark.unit
@pytest.mark.protocol
def test_common_time_alignment_is_exact_and_interpolation_is_prohibited() -> None:
    left = [{"time": 0.1}, {"time": 0.2}, {"time": 0.3}]
    right = [{"time": 0.1}, {"time": 0.20000000000000004}, {"time": 0.3}]
    assert [pair[0]["time"] for pair in align_exact_endpoint_times(left, right)] == [0.1, 0.3]
    with pytest.raises(ValueError, match="interpolation is prohibited"):
        reject_formal_interpolation(left, right, 0.2)


@pytest.mark.unit
@pytest.mark.protocol
def test_width_ratios_and_zero_denominators() -> None:
    assert width_ratio([1.0, 3.0], [-1.0, 3.0]) == 0.5
    assert width_ratio([1.0, 3.0], [2.0, 2.0]) is None
    with pytest.raises(ValueError, match="invalid interval"):
        width_ratio([3.0, 1.0], [0.0, 1.0])


@pytest.mark.unit
@pytest.mark.protocol
def test_failed_horizon_rows_are_retained_as_na() -> None:
    failed = horizon_row("torch", 10.0, 6.39, "validation_rejected")
    assert not failed["completed_requested_horizon"]
    assert failed["target_horizon_tightness"] == "N/A"
    assert failed["reached_horizon"] == 6.39


@pytest.mark.unit
@pytest.mark.protocol
def test_runtime_parser_preserves_declared_stage_boundaries() -> None:
    payload = json.loads(
        (OUT / "xiangru_reproduction/fresh_q3_b48_t20/raw_outputs/s3r_q3_b48_rep1.json").read_text()
    )
    runtime = parse_xiangru_runtime(payload, "b48_static", "complete_q3")
    assert runtime["device"] == "cuda"
    assert runtime["dtype"] == "float64"
    assert runtime["compile_or_graph_construction_seconds"] > 100
    assert runtime["total_end_to_end_seconds_including_validation"] > runtime["solver_seconds_excluding_validation"]
    assert runtime["serialization_io_seconds"] == "unavailable"


@pytest.mark.unit
@pytest.mark.protocol
def test_audit_serialization_is_deterministic() -> None:
    value = {"z": [3, 2, 1], "a": {"q": 3, "matched": False}}
    first = deterministic_json_bytes(value)
    second = deterministic_json_bytes(json.loads(first))
    assert first == second
    assert first.startswith(b'{\n  "a"')
