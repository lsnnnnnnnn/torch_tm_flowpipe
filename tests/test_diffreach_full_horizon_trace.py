from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from torch_tm_flowpipe.comparison_contract import (
    canonical_sha256,
    vdp_partition_identity,
)


ROOT = Path(__file__).parents[1]
EXPERIMENTS = ROOT / "experiments"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dependency_light_partition_is_exactly_the_frozen_torch_b64_contract(monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENTS))
    common = _load(
        "diffreach_torch_full_horizon_common_test",
        EXPERIMENTS / "diffreach_torch_full_horizon_common.py",
    )
    assert common.PARTITION_SHA256 == canonical_sha256(vdp_partition_identity(64))
    lower, upper = common.partition_arrays()
    contract = vdp_partition_identity(64)
    expected_lower = np.asarray(
        [[float.fromhex(value["hex"]) for value in box["lo"]] for box in contract["boxes"]]
    )
    expected_upper = np.asarray(
        [[float.fromhex(value["hex"]) for value in box["hi"]] for box in contract["boxes"]]
    )
    assert np.array_equal(lower, expected_lower)
    assert np.array_equal(upper, expected_upper)


def test_first_divergence_reports_field_index_absolute_relative_and_ulp(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENTS))
    compare = _load(
        "compare_diffreach_torch_full_horizon_test",
        EXPERIMENTS / "compare_diffreach_torch_full_horizon.py",
    )
    left = tmp_path / "left" / "captures"
    right = tmp_path / "right" / "captures"
    left.mkdir(parents=True)
    right.mkdir(parents=True)
    value = np.asarray([[1.0, 2.0]], dtype=np.float64)
    changed = value.copy()
    changed[0, 1] = np.nextafter(changed[0, 1], np.inf)
    np.savez(left / "step_0007.npz", poly1_L=value)
    np.savez(right / "step_0007.npz", poly1_L=changed)
    detail = compare._first_numeric_difference(
        tmp_path / "left", tmp_path / "right", 7, ["poly1_L"]
    )
    assert detail is not None
    assert detail["step"] == 7
    assert detail["field"] == "poly1_L"
    assert detail["index"] == [0, 1]
    assert detail["absolute_delta"] > 0.0
    assert detail["relative_delta"] > 0.0
    assert detail["ulp_delta"] == 1


def test_full_horizon_runner_is_native_explicit_f64_and_observer_is_inertness_gated():
    runner = (EXPERIMENTS / "run_diffreach_explicit_f64_full_trace.py").read_text()
    patch = (
        EXPERIMENTS / "diffreach_patches" / "dd628_explicit_f64_full_trace.patch"
    ).read_text()
    assert "DiffReach source commit mismatch" in runner
    assert "actual_patch_sha != expected_patch_sha" in runner
    assert "reachability.CT_Dyn_Reach" in runner
    assert "core.step_once_observed" in runner
    assert "observer changed native step outputs" in runner
    assert "class DiffReachPlantCore" not in runner
    assert "dtype=X0_lo.dtype" in patch
    assert "dtype=center.dtype" in patch
    assert "init_remainder.astype(base_poly.c.dtype)" in patch
    assert "init_symbolic_state" in patch and "dtype=X0_lo.dtype" in patch


def test_comparator_requires_full_horizon_masks_j_phi_and_endpoint_tube_sources():
    source = (EXPERIMENTS / "compare_diffreach_torch_full_horizon.py").read_text()
    assert '"steps": 1000' in source
    assert '"observer_inertness_bit_exact"' in source
    assert "MASK_FIELDS" in source and '"round_masks"' in source
    assert "J_PHI_FIELDS" in source and '"post_Phi"' in source
    assert "ENDPOINT_TUBE_FIELDS" in source and '"tube_hi"' in source
    assert "mask_equality and j_phi_equality and endpoint_tube_equality" in source
    assert "one_step" not in source


def test_stock_diffreach_lane_remains_separate_mixed_builder_native_capability():
    source = (EXPERIMENTS / "run_stock_diffreach_vdp_reproduction.py").read_text()
    assert '"classification": "mixed_builder_dtype"' in source
    assert '"eligibility_status": "native_capability_only"' in source
    assert '"segment_tube_available": False' in source
    assert '"prefix_tube_available": False' in source

