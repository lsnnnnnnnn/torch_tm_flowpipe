import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from torch_tm_flowpipe import (
    DenseRangePolicy,
    PolynomialODE,
    flowpipe_step_from_tm,
    load_terminal_checkpoint,
    save_terminal_checkpoint,
    tmvector_hashes,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    ROOT
    / "evidence"
    / "vdp_terminal_range_closure"
    / "20260805T055556Z"
    / "05_fresh_horizons"
    / "t6p5_proactive_d1_truncation"
    / "terminal_checkpoint"
)
ORDERS = ((0, 1, 2), (1, 0, 2), (2, 0, 1))


def _runner():
    experiments = ROOT / "experiments"
    if str(experiments) not in sys.path:
        sys.path.insert(0, str(experiments))
    spec = importlib.util.spec_from_file_location("later_terminal_runner", experiments / "run_vdp_dense_backend.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _candidate_hashes(trace):
    row = [item for item in trace if item.get("phase") == "polynomial_picard"][-1]
    return {
        "coefficient_sha256": row["coefficient_sha256"],
        "exponent_support_sha256": row["exponent_support_sha256"],
        "basis_hash": row["basis_hash"],
        "effective_degree": row["effective_degree"],
        "picard_iterations": row["iteration"],
    }


def _replay(policy):
    runner = _runner()
    contract = runner.load_contract()
    checkpoint = load_terminal_checkpoint(
        CHECKPOINT,
        expected_contract=contract,
        expected_order=4,
        expected_dtype="float64",
    )
    current = checkpoint.normal_state.normalized_initial_tm(4)
    return flowpipe_step_from_tm(
        PolynomialODE.from_system_spec(contract["canonical_system_spec"]),
        current,
        float(checkpoint.scheduler["h_attempted"]),
        4,
        max_validation_attempts=2,
        validation_eps=1e-12,
        validation_mode="flowstar_raw_remainder_compat",
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        tm_backend="dense",
        dense_device="cpu",
        dense_range_policy=policy,
    )


def test_later_checkpoint_exact_contract_hashes_and_byte_roundtrip(tmp_path):
    runner = _runner()
    contract = runner.load_contract()
    checkpoint = load_terminal_checkpoint(
        CHECKPOINT,
        expected_contract=contract,
        expected_order=4,
        expected_dtype="float64",
    )
    assert checkpoint.scheduler["current_time"] == 6.397083942944808
    assert checkpoint.scheduler["h_attempted"] == 0.003623635847674574
    assert checkpoint.manifest["full_checkpoint_sha256"] == "dcb8f646d45c9742e0cff23fea596c12e53d8ccd00d1544f70564a44a7463420"
    assert checkpoint.manifest["contract_sha256"] == "8c29a59ac574dc463e3cb73909ba489a911fdeabb3d2b9a4d48507184d909547"
    assert checkpoint.manifest["dtype"] == "float64"
    assert tmvector_hashes(checkpoint.current) == tmvector_hashes(checkpoint.normal_state.normalized_initial_tm(4))
    for relative, expected in checkpoint.provenance["source_hashes"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected

    manifest = save_terminal_checkpoint(
        tmp_path / "roundtrip",
        current=checkpoint.current,
        normal_state=checkpoint.normal_state,
        scheduler=checkpoint.scheduler,
        contract=checkpoint.contract,
        provenance=checkpoint.provenance,
    )
    assert manifest == checkpoint.manifest
    for filename in ("terminal_state.json", "terminal_state_manifest.json"):
        assert (tmp_path / "roundtrip" / filename).read_bytes() == (CHECKPOINT / filename).read_bytes()


def test_later_terminal_natural_subdivision_horner_and_combined_keep_candidate_invariant():
    natural = _replay(DenseRangePolicy())
    subdivision = _replay(
        DenseRangePolicy(
            method="subdivision",
            max_depth=1,
            max_leaves=4,
            split_vars=(0, 1),
            named_contexts=("polynomial_truncation",),
        )
    )
    horner = _replay(
        DenseRangePolicy(
            method="horner_registered_best",
            named_contexts=("polynomial_truncation",),
            variable_orders=ORDERS,
        )
    )
    combined = _replay(
        DenseRangePolicy(
            method="subdivision_then_horner",
            max_depth=1,
            max_leaves=4,
            split_vars=(0, 1),
            named_contexts=("polynomial_truncation",),
            variable_orders=ORDERS,
        )
    )
    rows = (natural, subdivision, horner, combined)
    assert {row.h for row in rows} == {0.003623635847674574}
    assert {row.status for row in rows} == {"failed"}
    assert {_candidate_hashes(row.backend_trace)["coefficient_sha256"] for row in rows} == {
        "bc1433d0d3c89339fca6091e41c0a6667d70c92d2dd4e35ae8b14236d131863c"
    }
    assert {_candidate_hashes(row.backend_trace)["exponent_support_sha256"] for row in rows} == {
        "d0aa354b9057267556d5bb3bc09a36ed4162b36fb44588b0b930dd9e935041e9"
    }
    assert {row.validation_attempts for row in rows} == {1}
    assert all(row.endpoint_raw_tm is None for row in rows)
    assert all(row.backend_counters["sparse_fallback_count"] == 0 for row in rows)
    assert natural.subset_margin[0][1] == pytest.approx(-7.584392650575044e-05, abs=1e-15)
    assert subdivision.subset_margin[0][1] == pytest.approx(-1.99995911680722e-05, abs=1e-15)
    assert horner.subset_margin[0][1] == pytest.approx(-7.584392650574142e-05, abs=1e-15)
    assert combined.subset_margin[0][1] == pytest.approx(-1.5859969428028492e-05, abs=1e-15)
    validation_modes = {
        item["validation_mode"]
        for row in rows
        for item in row.backend_trace
        if item.get("phase") == "remainder_validation"
    }
    assert validation_modes == {"flowstar_raw_remainder_compat"}
    combined_ranges = [item for item in combined.backend_trace if item.get("phase") == "polynomial_range"]
    assert any(item.get("horner_validated") for item in combined_ranges)
    assert all(not item.get("fallback_reason", "").startswith("silent") for item in combined_ranges)
