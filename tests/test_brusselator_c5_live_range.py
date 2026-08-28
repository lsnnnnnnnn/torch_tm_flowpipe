from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib

import pytest

from torch_tm_flowpipe import FlowstarNormalFlowpipeState, commit_accepted_boundary_sr
from torch_tm_flowpipe.brusselator_canonical_exchange import (
    CUTOFF,
    ORDER,
    SCHEMA,
    build_exchange_records,
    parse_records,
    take_tmv,
    write_records,
)
from torch_tm_flowpipe.step1_oracle import RationalInterval, RationalPolynomial
from torch_tm_flowpipe.interval import Interval
from torch_tm_flowpipe.symbolic_remainder import accepted_boundary_sr_queue_sha256
from torch_tm_flowpipe.taylor_model import TaylorModel
from torch_tm_flowpipe.tm_vector import TMVector
from experiments.run_brusselator_sr1000_parity import _policy, _step


C4 = "flowstar_raw_remainder_compat_refined"


@pytest.fixture(scope="module")
def step1_exchange():
    pre = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        (("1.48", "1.52"), ("2.98", "3.02")), ORDER
    )
    segment, diagnostics = _step(
        pre.normalized_initial_tm(ORDER),
        pre,
        1,
        _policy(),
        validation_mode=C4,
        lane_label="c5_canonical_unit",
    )
    assert segment.status == "validated"
    assert segment.flowstar_normal_state is not None
    built = build_exchange_records(
        pre_state=pre,
        post_state=segment.flowstar_normal_state,
        accepted_step=1,
        checkpoint_sha256="0" * 64,
        torch_solver_commit="1" * 40,
        flowstar_commit="b85a3211748cb77b736fe4ad42ee02d8d2b81148",
        source_hashes={"unit": "2" * 64},
    )
    return pre, segment, diagnostics, built


def _tm_payload(value):
    return [
        {
            "terms": [
                (tuple(exponent), float(coefficient.detach().cpu()).hex())
                for exponent, coefficient in sorted(model.polynomial.terms.items())
            ],
            "remainder": (
                float(model.remainder.lo.detach().cpu()).hex(),
                float(model.remainder.hi.detach().cpu()).hex(),
            ),
            "domain": [
                (
                    float(interval.lo.detach().cpu()).hex(),
                    float(interval.hi.detach().cpu()).hex(),
                )
                for interval in model.domain
            ],
            "order": model.order,
        }
        for model in value
    ]


def _exact_range(model, *, include_remainder: bool):
    polynomial = RationalPolynomial(
        model.n_vars,
        {
            tuple(exponent): Fraction.from_float(float(coefficient.detach().cpu()))
            for exponent, coefficient in model.polynomial.terms.items()
        },
    )
    domain = [
        RationalInterval(
            Fraction.from_float(float(interval.lo.detach().cpu())),
            Fraction.from_float(float(interval.hi.detach().cpu())),
        )
        for interval in model.domain
    ]
    result = polynomial.bernstein_range(domain)
    if include_remainder:
        result = result + RationalInterval(
            Fraction.from_float(float(model.remainder.lo.detach().cpu())),
            Fraction.from_float(float(model.remainder.hi.detach().cpu())),
        )
    return result


def _contains(interval, exact):
    return (
        Fraction.from_float(float(interval.lo.detach().cpu())) <= exact.lo
        and exact.hi <= Fraction.from_float(float(interval.hi.detach().cpu()))
    )


@pytest.mark.unit
def test_canonical_export_import_round_trip_and_complete_required_payloads(tmp_path, step1_exchange):
    _pre, _segment, _diagnostics, built = step1_exchange
    path = tmp_path / "step1.canonical"
    sha = write_records(path, built.records)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    records = parse_records(path.read_text(encoding="utf-8"))
    assert records["schema"] == SCHEMA
    for key in (
        "boundary.sr_propagated_history.count",
        "boundary.sr_current_owner.count",
        "queue.before.J_count",
        "pre.center.count",
        "post.scale.count",
        "table.step_exp.count",
        "table.step_end_exp.count",
        "cutoff_threshold_hex",
        "source.file_sha256.unit",
    ):
        assert key in records
    for prefix in (
        "tm.segment_tube",
        "tm.segment_endpoint_pre_cutoff",
        "tm.segment_endpoint_raw",
        "tm.boundary_outer_full",
        "tm.boundary_outer_nonlinear",
        "tm.right_map_input",
        "tm.boundary_torch_inserted",
        "tm.right_map_torch_post_cutoff",
    ):
        imported = take_tmv(dict(records), prefix)
        original_records = dict(records)
        original = take_tmv(original_records, prefix)
        assert _tm_payload(imported) == _tm_payload(original)


@pytest.mark.unit
def test_exact_endpoint_and_tube_polynomial_and_full_ranges(step1_exchange):
    _pre, _segment, _diagnostics, built = step1_exchange
    records = dict(built.records)
    for prefix in ("tm.segment_endpoint_raw", "tm.segment_tube"):
        tmv = take_tmv(dict(records), prefix)
        for model in tmv:
            assert _contains(
                model.polynomial.evaluate_interval(model.domain),
                _exact_range(model, include_remainder=False),
            )
            assert _contains(model.range_box(), _exact_range(model, include_remainder=True))


@pytest.mark.unit
def test_cutoff_ownership_is_single_containment_payment(step1_exchange):
    _pre, _segment, _diagnostics, built = step1_exchange
    records = dict(built.records)
    before = take_tmv(dict(records), "tm.segment_endpoint_pre_cutoff")
    after = take_tmv(dict(records), "tm.segment_endpoint_raw")
    for pre_model, post_model in zip(before, after, strict=True):
        kept, payment = pre_model.polynomial.cutoff(CUTOFF, pre_model.domain)
        assert set(kept.terms) == set(post_model.polynomial.terms)
        expected = pre_model.remainder + payment
        assert post_model.remainder.contains(expected.lo)
        assert post_model.remainder.contains(expected.hi)
        assert _contains(post_model.range_box(), _exact_range(pre_model, include_remainder=True))


@pytest.mark.unit
def test_composition_truncation_owner_and_normalization_reconstruct_live_poststate(step1_exchange):
    _pre, segment, diagnostics, built = step1_exchange
    assert built.prepared.composition_branch == "full_reanchor"
    assert all(
        owner.contains(model.remainder.lo) and owner.contains(model.remainder.hi)
        for owner, model in zip(
            built.prepared.current_owner, built.prepared.inserted, strict=True
        )
    )
    assert segment.flowstar_normal_state is not None
    assert _tm_payload(built.reconstructed_post_right) == _tm_payload(
        segment.flowstar_normal_state.tmv_right
    )
    records = dict(built.records)
    for label, expected in (
        ("post.center", segment.flowstar_normal_state.center),
        ("post.scale", segment.flowstar_normal_state.scales),
    ):
        assert int(records[f"{label}.count"]) == len(expected)
        assert [
            float.fromhex(records[f"{label}.{index}"]).hex()
            for index in range(len(expected))
        ] == [float(value).hex() for value in expected]
    assert all(scale > 0.0 for scale in segment.flowstar_normal_state.scales)
    assert diagnostics


@pytest.mark.unit
def test_same_input_live_operator_substitution_is_atomic_and_preserves_polynomial_queue_policy(
    step1_exchange,
):
    _pre, segment, _diagnostics, built = step1_exchange
    assert segment.flowstar_normal_state is not None
    delta = Interval(-2.0**-48, 2.0**-48)
    substituted_inserted = TMVector(
        [
            TaylorModel(
                model.polynomial,
                model.remainder + delta,
                list(model.domain),
                order=model.order,
                truncation_range_split=model.truncation_range_split,
            )
            for model in built.prepared.inserted
        ]
    )
    substituted = replace(
        built.prepared,
        inserted=substituted_inserted,
        current_owner=tuple(owner + delta for owner in built.prepared.current_owner),
    )
    queue_before_hash = accepted_boundary_sr_queue_sha256(built.prepared.queue_before)
    baseline = commit_accepted_boundary_sr(
        built.prepared,
        normalization_scales=segment.flowstar_normal_state.scales,
        cutoff_threshold=CUTOFF,
    )
    shadow = commit_accepted_boundary_sr(
        substituted,
        normalization_scales=segment.flowstar_normal_state.scales,
        cutoff_threshold=CUTOFF,
    )
    assert accepted_boundary_sr_queue_sha256(built.prepared.queue_before) == queue_before_hash
    assert substituted.queue_before is built.prepared.queue_before
    for baseline_model, shadow_model in zip(
        baseline.normalized_right_map, shadow.normalized_right_map, strict=True
    ):
        assert baseline_model.polynomial.terms.keys() == shadow_model.polynomial.terms.keys()
        assert all(
            float(baseline_model.polynomial.terms[exponent]).hex()
            == float(shadow_model.polynomial.terms[exponent]).hex()
            for exponent in baseline_model.polynomial.terms
        )
        assert shadow_model.remainder.contains(baseline_model.remainder.lo)
        assert shadow_model.remainder.contains(baseline_model.remainder.hi)
    assert shadow.queue_after.max_size == baseline.queue_after.max_size == 1000
    assert shadow.queue_after.accepted_boundary_index == baseline.queue_after.accepted_boundary_index


@pytest.mark.unit
def test_reporting_and_live_paths_are_explicitly_separated(step1_exchange):
    _pre, _segment, _diagnostics, built = step1_exchange
    records = dict(built.records)
    assert records["labels.reporting_endpoint"] == "tm.segment_endpoint_raw"
    assert records["labels.reporting_tube"] == "tm.segment_tube"
    assert records["labels.boundary_normalization"] == "tm.boundary_torch_inserted"
    assert records["labels.picard_validation"] == "not_recomputed_by_range_harness"
    assert records["labels.next_step_initialization"] == "post.center,post.scale"


@pytest.mark.unit
def test_c5_off_preserves_frozen_c4_step1_binary64_snapshot(step1_exchange):
    _pre, segment, _diagnostics, _built = step1_exchange
    assert segment.tm is not None
    assert [float(model.remainder.lo.detach().cpu()).hex() for model in segment.tm] == [
        "-0x1.58aa609209c7bp-27",
        "-0x1.d4748bbc7cd95p-28",
    ]
    assert [float(model.remainder.hi.detach().cpu()).hex() for model in segment.tm] == [
        "0x1.bcb368de134ecp-28",
        "0x1.6c320c0104a5fp-27",
    ]


@pytest.mark.unit
def test_canonical_tamper_rejection_duplicate_decimal_and_bad_exponent(step1_exchange):
    _pre, _segment, _diagnostics, built = step1_exchange
    text = "".join(f"{key}={value}\n" for key, value in built.records)
    with pytest.raises(ValueError, match="duplicate"):
        parse_records(text + "schema=duplicate\n")
    records = parse_records(text)
    coefficient_key = next(key for key in records if key.endswith(".coefficient_hex"))
    records[coefficient_key] = "0.1"
    with pytest.raises(ValueError, match="not hexadecimal"):
        take_tmv(records, coefficient_key.split(".component.", 1)[0])
    records = parse_records(text)
    exponent_key = next(key for key in records if key.endswith(".exponents"))
    records[exponent_key] = "999"
    with pytest.raises(ValueError, match="dimension"):
        take_tmv(records, exponent_key.split(".component.", 1)[0])
