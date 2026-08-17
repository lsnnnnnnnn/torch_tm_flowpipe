from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from torch_tm_flowpipe import (
    Interval,
    PolynomialODE,
    TMVector,
    flowpipe_step_flowstar_style_adaptive,
    load_terminal_checkpoint,
    save_terminal_checkpoint,
)
from torch_tm_flowpipe.g2_shared_column import (
    G2_SHARED_COLUMN_CANDIDATE,
    G2SharedColumnState,
    accepted_successor,
    commit_or_preserve,
    partition_source_terms,
    polynomial_payload_sha256,
    rotate_current_to_retained,
)
from torch_tm_flowpipe.polynomial import Polynomial


UNIT = Interval(-1.0, 1.0)
ROOT = Path(__file__).resolve().parents[1]


def _ode() -> PolynomialODE:
    return PolynomialODE.from_system_spec(
        {
            "state_names": ["x", "y"],
            "rhs": [
                {"terms": [{"coefficient": 1.0, "powers": [0, 1]}]},
                {"terms": [
                    {"coefficient": 1.0, "powers": [0, 1]},
                    {"coefficient": -1.0, "powers": [2, 1]},
                    {"coefficient": -1.0, "powers": [1, 0]},
                ]},
            ],
        }
    )


def _step(current, normal_state=None, *, h: float = 0.01):
    return flowpipe_step_flowstar_style_adaptive(
        _ode(),
        current,
        h=h,
        h_min=h,
        h_max=h,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode=G2_SHARED_COLUMN_CANDIDATE,
        flowstar_normal_state=normal_state,
        tm_backend="dense",
    )


def _digest(tmv: TMVector) -> str:
    rows = [
        [(tuple(exp), float(coef.detach().cpu()).hex()) for exp, coef in sorted(model.polynomial.terms.items())]
        for model in tmv
    ]
    return hashlib.sha256(repr(rows).encode("utf-8")).hexdigest()


def test_g2_state_has_exactly_three_fixed_banks() -> None:
    state = G2SharedColumnState.initial(2)
    assert state.variable_count == 6
    assert state.oldest_indices == (2, 3)
    assert state.current_indices == (4, 5)
    assert state.generations_retained == 2
    assert state.live_source_count == 0


def test_oldest_current_mixed_terms_retire_and_current_rotates() -> None:
    # Variables are base=(0,1), oldest=(2,3), current=(4,5).
    u = Polynomial.variable(0, 6)
    oldest = Polynomial.variable(2, 6)
    current = Polynomial.variable(4, 6)
    poly = u + current * 2.0 + u * current + oldest * current * 3.0
    after_oldest_partition = partition_source_terms(poly, (2, 3))
    assert (after_oldest_partition.source_bearing - oldest * current * 3.0).terms == {}
    surviving = after_oldest_partition.source_free
    current_partition = partition_source_terms(surviving, (4, 5))
    rotated = rotate_current_to_retained(current_partition.source_bearing, 2)
    assert all(exp[4] == exp[5] == 0 for exp in rotated.terms)
    assert any(exp[2] for exp in rotated.terms)


def test_rejected_rotation_preserves_state_object_and_fingerprint() -> None:
    initial = G2SharedColumnState.initial(2)
    proposed = accepted_successor(
        initial,
        torch.tensor([[0.1, 0.2]], dtype=torch.float64),
        ("picard_residual",),
        retained_payload_sha256=initial.retained_payload_sha256,
        retained_active=(False, False),
    )
    rejected = commit_or_preserve(initial, proposed, accepted=False)
    assert rejected is initial
    assert rejected.fingerprint == initial.fingerprint
    assert commit_or_preserve(initial, proposed, accepted=True).generation == 1


def test_real_dense_consumer_rotates_two_generations_with_six_variables() -> None:
    first = _step([Interval(1.1, 1.4), Interval(2.35, 2.45)])
    assert first.status == "validated" and first.reset_tm is not None
    assert first.validated_remainder_decomposition is not None
    assert bool(torch.all(first.validated_remainder_decomposition.contains_image))
    assert first.flowstar_normal_state is not None
    first_state = first.flowstar_normal_state.g2_shared_column_state
    assert first_state is not None and first_state.live_source_count == 2
    assert first.reset_tm.n_vars == 6
    assert {4, 5}.issubset(first.reset_tm.active_variables())

    second = _step(first.reset_tm, first.flowstar_normal_state)
    assert second.status == "validated" and second.reset_tm is not None
    assert second.flowstar_normal_state is not None
    second_state = second.flowstar_normal_state.g2_shared_column_state
    retained = second.flowstar_normal_state.g2_retained_source_tm
    assert second_state is not None and retained is not None
    assert second_state.retained_source_ids == first_state.fresh_source_ids
    assert second_state.live_source_count == 4
    assert second.reset_tm.n_vars == 6
    assert set(second.reset_tm.active_variables()) <= set(range(6))
    assert polynomial_payload_sha256([model.polynomial for model in retained]) == second_state.retained_payload_sha256
    assert all(
        exponent[4] == exponent[5] == 0
        for model in retained
        for exponent in model.polynomial.terms
    )
    # The x remainder source has propagated into both components, proving that
    # the retained bank is a shared column rather than per-occurrence boxes.
    assert all(any(exponent[2] for exponent in model.polynomial.terms) for model in retained)

    third = _step(second.reset_tm, second.flowstar_normal_state)
    assert third.status == "validated" and third.flowstar_normal_state is not None
    third_state = third.flowstar_normal_state.g2_shared_column_state
    assert third_state is not None
    assert third_state.collapse_count == 1
    assert third_state.retired_source_count == 2
    assert third_state.retained_source_ids == second_state.fresh_source_ids

    diagnostics = third.flowstar_normal_state.diagnostics
    assert len(diagnostics["g2_dense_owner_rows"]) == 22
    assert len(diagnostics["g2_fresh_structured_owner_rows"]) == 2
    assert len(diagnostics["g2_rebox_owner_rows"]) == 2
    for owner in (
        diagnostics["g2_dense_owner_rows"]
        + diagnostics["g2_fresh_structured_owner_rows"]
        + diagnostics["g2_rebox_owner_rows"]
    ):
        assert len(owner["canonical_support_sha256"]) == 64
        assert "outward_lo_hex" in owner and "outward_hi_hex" in owner
        assert owner["width"] >= 0.0
        assert owner["containment_witness"]


def test_g2_retained_and_fresh_payloads_are_not_double_counted() -> None:
    first = _step([Interval(1.1, 1.4), Interval(2.35, 2.45)])
    second = _step(first.reset_tm, first.flowstar_normal_state)
    state = second.flowstar_normal_state
    assert state is not None and state.g2_retained_source_tm is not None
    # The normalized right map owns only base/rebox mass. Retained polynomials
    # occupy bank 1 and the affine fresh payload occupies bank 2 in reset_tm.
    assert all(
        all(exp[index] == 0 for index in range(2, 6))
        for model in state.tmv_right
        for exp in model.polynomial.terms
    )
    assert all(
        all(exp[index] == 0 for index in (4, 5))
        for model in state.g2_retained_source_tm
        for exp in model.polynomial.terms
    )
    assert second.reset_tm is not None
    assert {2, 3, 4, 5}.issubset(second.reset_tm.active_variables())


def test_g2_checkpoint_v4_resume_is_canonical_equal(tmp_path) -> None:
    first = _step([Interval(1.1, 1.4), Interval(2.35, 2.45)])
    second = _step(first.reset_tm, first.flowstar_normal_state)
    assert second.reset_tm is not None and second.flowstar_normal_state is not None
    contract = {"order": 4, "candidate": G2_SHARED_COLUMN_CANDIDATE}
    manifest = save_terminal_checkpoint(
        tmp_path / "first",
        current=second.reset_tm,
        normal_state=second.flowstar_normal_state,
        scheduler={"t": 0.02, "h": 0.01},
        contract=contract,
        provenance={"test": True},
    )
    assert manifest["schema"] == "torch_tm_flowpipe_terminal_checkpoint_v4"
    loaded = load_terminal_checkpoint(
        tmp_path / "first",
        expected_contract=contract,
        expected_order=4,
        expected_dtype="float64",
    )
    assert _digest(loaded.current) == _digest(second.reset_tm)
    assert loaded.normal_state.g2_shared_column_state.as_dict() == second.flowstar_normal_state.g2_shared_column_state.as_dict()
    assert _digest(loaded.normal_state.g2_retained_source_tm) == _digest(second.flowstar_normal_state.g2_retained_source_tm)
    assert loaded.normal_state.diagnostics == second.flowstar_normal_state.diagnostics

    save_terminal_checkpoint(
        tmp_path / "second",
        current=loaded.current,
        normal_state=loaded.normal_state,
        scheduler=loaded.scheduler,
        contract=loaded.contract,
        provenance=loaded.provenance,
    )
    assert (tmp_path / "first" / "terminal_state.json").read_bytes() == (tmp_path / "second" / "terminal_state.json").read_bytes()
    assert (tmp_path / "first" / "terminal_state_manifest.json").read_bytes() == (tmp_path / "second" / "terminal_state_manifest.json").read_bytes()

    native_next = _step(second.reset_tm, second.flowstar_normal_state)
    resumed_next = _step(loaded.current, loaded.normal_state)
    assert native_next.status == resumed_next.status == "validated"
    assert _digest(native_next.reset_tm) == _digest(resumed_next.reset_tm)
    assert native_next.flowstar_normal_state.g2_shared_column_state.as_dict() == resumed_next.flowstar_normal_state.g2_shared_column_state.as_dict()

    changed_models = list(second.reset_tm)
    changed_models[0] = changed_models[0] + 0.125
    with pytest.raises(ValueError, match="canonical materialization"):
        save_terminal_checkpoint(
            tmp_path / "invalid",
            current=TMVector(changed_models),
            normal_state=second.flowstar_normal_state,
            scheduler={"t": 0.02, "h": 0.01},
            contract=contract,
            provenance={"test": True},
        )


def test_rejected_retry_does_not_mutate_full_g2_prestate() -> None:
    first = _step([Interval(1.1, 1.4), Interval(2.35, 2.45)])
    assert first.reset_tm is not None and first.flowstar_normal_state is not None
    before = first.flowstar_normal_state.g2_shared_column_state
    retained_before = _digest(first.flowstar_normal_state.g2_retained_source_tm)
    reset_before = _digest(first.reset_tm)
    rejected = _step(first.reset_tm, first.flowstar_normal_state, h=0.1)
    assert rejected.status == "failed"
    assert first.flowstar_normal_state.g2_shared_column_state is before
    assert before.fingerprint == first.flowstar_normal_state.g2_shared_column_state.fingerprint
    assert retained_before == _digest(first.flowstar_normal_state.g2_retained_source_tm)
    assert reset_before == _digest(first.reset_tm)


def test_independent_exact_oracle_uses_no_project_core(tmp_path) -> None:
    blackbox = tmp_path / "blackbox.json"
    result = tmp_path / "oracle.json"
    subprocess.run(
        [sys.executable, str(ROOT / "experiments/export_g2_blackbox_coefficients.py"), "--output", str(blackbox)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    oracle_source = (ROOT / "experiments/independent_g2_exact_oracle.py").read_text(encoding="utf-8")
    for forbidden in (
        "import torch\n",
        "from torch_tm_flowpipe",
        "import torch_tm_flowpipe",
    ):
        assert forbidden not in oracle_source
    subprocess.run(
        [sys.executable, str(ROOT / "experiments/independent_g2_exact_oracle.py"), "--input", str(blackbox), "--output", str(result)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["checks_passed"] >= 15
    assert payload["implementation_independent"] is True
    assert payload["sampling_used"] is False


def test_real_dense_phase_timing_ledger_is_exposed() -> None:
    segment = _step([Interval(1.1, 1.4), Interval(2.35, 2.45)])
    counters = segment.backend_counters
    assert counters["host_to_device_s"] == 0.0
    assert counters["device_to_host_s"] == 0.0
    assert counters["dense_kernel_s"] > 0.0
    assert counters["device_transfer_count"] == 0
