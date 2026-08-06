from __future__ import annotations

import csv
from dataclasses import replace
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from torch_tm_flowpipe import Interval, TMVector, tmvector_hashes
from torch_tm_flowpipe.audit_trace import (
    REQUIRED_COMMON,
    REQUIRED_STAGES,
    TransitionTraceWriter,
    decode_float,
    encode_float,
    encode_interval,
    tmv_content_hash,
    verify_recorded_stage_hash,
)
from torch_tm_flowpipe.flowpipe import (
    FlowpipeSegment,
    FlowstarNormalFlowpipeState,
    flowpipe_step_flowstar_style_adaptive,
)
from torch_tm_flowpipe.ode_examples import van_der_pol_ode


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compare = _load("compare_vdp_transition_traces")
flowstar_process = _load("process_flowstar_observation_trace")
lineage = _load("trace_vdp_call44_lineage")


def _writer_fixture(directory: Path) -> tuple[TransitionTraceWriter, TMVector]:
    tmv = TMVector.identity([Interval(1.0, 2.0), Interval(3.0, 4.0)], order=2)
    writer = TransitionTraceWriter(directory, run_id="deterministic-test", source_commit="0" * 40)
    return writer, tmv


def test_trace_number_and_interval_encoding_round_trip_fail_closed():
    for value in (0.0, -0.0, 1.0 / 10.0, -1.25, float.fromhex("0x1.fffffffffffffp+10")):
        encoded = encode_float(value)
        assert decode_float(encoded) == value
        assert set(encoded) == {"decimal", "hex"}
    assert decode_float({"decimal": "0.1", "hex": (0.1).hex()}) == 0.1
    with pytest.raises(ValueError, match="differ"):
        decode_float({"decimal": "0.1", "hex": (0.2).hex()})
    interval = encode_interval(Interval(-2.0, 3.0))
    assert decode_float(interval["lower"]) <= decode_float(interval["upper"])


def test_trace_schema_required_fields_canonical_exponents_and_no_mutation(tmp_path):
    writer, tmv = _writer_fixture(tmp_path / "trace")
    before = tmvector_hashes(tmv)
    writer.emit_tmv(
        tmv,
        step=0,
        attempt=0,
        retry=0,
        t_pre=0.0,
        h_attempt=0.01,
        accepted=True,
        rejection_reason="",
        stage="step_pre_state",
    )
    assert tmvector_hashes(tmv) == before
    writer.close(result_summary={"status": "validated", "runtime_s": 99.0})
    schema = json.loads((tmp_path / "trace" / "trace_schema.json").read_text())
    assert set(REQUIRED_COMMON) == set(schema["required_common_fields"])
    assert set(REQUIRED_STAGES) == set(schema["required_stages"])
    rows = [json.loads(line) for line in (tmp_path / "trace" / "polynomial_terms.jsonl").read_text().splitlines()]
    by_component: dict[int, list[list[int]]] = {}
    for row in rows:
        by_component.setdefault(row["state_component"], []).append(row["exponent_tuple"])
        assert row["degree"] == sum(row["exponent_tuple"])
    assert all(items == sorted(items) for items in by_component.values())


def test_torch_trace_serialization_is_byte_deterministic_and_rejects_duplicates(tmp_path):
    outputs = []
    for name in ("one", "two"):
        writer, tmv = _writer_fixture(tmp_path / name)
        base = dict(
            step=0, attempt=0, retry=0, t_pre=0.0, h_attempt=0.01,
            accepted=True, rejection_reason="", stage="step_pre_state",
        )
        writer.emit_tmv(tmv, **base)
        writer.close(result_summary={"status": "validated", "runtime_s": 1.0 if name == "one" else 2.0})
        outputs.append({path.name: path.read_bytes() for path in sorted((tmp_path / name).iterdir())})
    assert outputs[0] == outputs[1]

    writer, tmv = _writer_fixture(tmp_path / "duplicate")
    base = dict(
        step=0, attempt=0, retry=0, t_pre=0.0, h_attempt=0.01,
        accepted=True, rejection_reason="", stage="step_pre_state",
    )
    writer.emit_tmv(tmv, **base)
    with pytest.raises(ValueError, match="duplicate trace exponent"):
        writer.emit_tmv(tmv, **base)
    writer.close(result_summary={"status": "expected_failure"})


def test_content_hash_is_sensitive_to_basis_center_scale_coefficient_and_remainder():
    tmv = TMVector.identity(
        [Interval(1.0, 2.0), Interval(3.0, 4.0)], order=2
    )
    baseline = tmv_content_hash(tmv, centers=[1.0, 2.0], scales=[3.0, 4.0])
    assert baseline != tmv_content_hash(
        tmv, centers=[1.0, 2.0], scales=[3.0, 5.0]
    )
    assert baseline != tmv_content_hash(
        tmv, centers=[1.0, 3.0], scales=[3.0, 4.0]
    )
    assert baseline != tmv_content_hash(
        tmv,
        centers=[1.0, 2.0],
        scales=[3.0, 4.0],
        basis_variable_order=["physical_x", "physical_y"],
    )
    coefficient_changed = TMVector([tmv[0] + 0.25, tmv[1]])
    remainder_changed = TMVector([tmv[0] + Interval(-1e-6, 1e-6), tmv[1]])
    assert baseline != tmv_content_hash(
        coefficient_changed, centers=[1.0, 2.0], scales=[3.0, 4.0]
    )
    assert baseline != tmv_content_hash(
        remainder_changed, centers=[1.0, 2.0], scales=[3.0, 4.0]
    )


def _lifecycle_segment():
    x0 = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    previous = FlowstarNormalFlowpipeState.from_initial_box(x0, order=4)
    diagnostics: list[dict[str, object]] = []
    segment = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        x0,
        h=0.002,
        h_min=0.002,
        h_max=0.002,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        reset_mode="normalized_insertion",
        flowstar_normal_state=previous,
        diagnostics=diagnostics,
    )
    assert segment.status == "validated"
    assert segment.transition_lifecycle is not None
    assert segment.flowstar_normal_state is not None
    return x0, previous, segment, diagnostics


def test_trace_lifecycle_hashes_match_actual_objects_and_wrong_stage_fails_closed(tmp_path):
    x0, previous, segment, diagnostics = _lifecycle_segment()
    trace_dir = tmp_path / "lifecycle"
    writer = TransitionTraceWriter(
        trace_dir, run_id="lifecycle-identity", source_commit="0" * 40
    )
    current = previous.normalized_initial_tm(order=4)
    before = tmvector_hashes(current), tmvector_hashes(segment.reset_tm)
    writer.record_step(
        step=0,
        t_pre=0.0,
        current=current,
        previous_state=previous,
        segment=segment,
        diagnostics=diagnostics,
        accepted=True,
        attempted_h=0.002,
        order=4,
    )
    writer.close(result_summary={"status": "validated"})
    assert before == (tmvector_hashes(current), tmvector_hashes(segment.reset_tm))

    lifecycle = segment.transition_lifecycle
    next_state = segment.flowstar_normal_state
    transitions = trace_dir / "transitions.jsonl"
    assert lifecycle is not None and next_state is not None
    expected = {
        "step_pre_state": verify_recorded_stage_hash(
            transitions,
            stage="step_pre_state",
            actual=current,
            centers=previous.center,
            scales=previous.scales,
        ),
        "right_map_input": verify_recorded_stage_hash(
            transitions,
            stage="right_map_input",
            actual=previous.tmv_right,
            centers=previous.center,
            scales=previous.scales,
        ),
        "insertion_input": verify_recorded_stage_hash(
            transitions, stage="insertion_input", actual=lifecycle.insertion_input
        ),
        "insertion_output": verify_recorded_stage_hash(
            transitions, stage="insertion_output", actual=lifecycle.insertion_output
        ),
        "normalized_reset_input": verify_recorded_stage_hash(
            transitions,
            stage="normalized_reset_input",
            actual=lifecycle.normalized_reset_input,
        ),
        "normalized_reset_output": verify_recorded_stage_hash(
            transitions,
            stage="normalized_reset_output",
            actual=lifecycle.normalized_reset_output,
            centers=next_state.center,
            scales=next_state.scales,
        ),
        "next_step_pre_state": verify_recorded_stage_hash(
            transitions,
            stage="next_step_pre_state",
            actual=lifecycle.normalized_reset_output,
            centers=next_state.center,
            scales=next_state.scales,
        ),
        "right_map_output": verify_recorded_stage_hash(
            transitions,
            stage="right_map_output",
            actual=next_state.tmv_right,
            centers=next_state.center,
            scales=next_state.scales,
        ),
    }
    assert len(set(expected.values())) >= 3
    with pytest.raises(ValueError, match="object hash mismatch"):
        verify_recorded_stage_hash(
            transitions,
            stage="right_map_input",
            actual=lifecycle.insertion_output,
            centers=previous.center,
            scales=previous.scales,
        )

    second = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        x0,
        h=0.002,
        h_min=0.002,
        h_max=0.002,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        reset_mode="normalized_insertion",
        flowstar_normal_state=previous,
    )
    assert tmvector_hashes(second.reset_tm) == tmvector_hashes(segment.reset_tm)


def test_trace_rejects_lifecycle_with_wrong_reset_object(tmp_path):
    _, previous, segment, diagnostics = _lifecycle_segment()
    lifecycle = segment.transition_lifecycle
    assert lifecycle is not None
    segment.transition_lifecycle = replace(
        lifecycle, normalized_reset_output=lifecycle.insertion_output
    )
    writer = TransitionTraceWriter(
        tmp_path / "wrong-object", run_id="wrong-object", source_commit="0" * 40
    )
    with pytest.raises(ValueError, match="not the segment reset object"):
        writer.record_step(
            step=0,
            t_pre=0.0,
            current=previous.normalized_initial_tm(order=4),
            previous_state=previous,
            segment=segment,
            diagnostics=diagnostics,
            accepted=True,
            attempted_h=0.002,
            order=4,
        )
    writer.close(result_summary={"status": "expected_failure"})


def test_accepted_attempt_predicate_matches_recorded_margins(tmp_path):
    writer, current = _writer_fixture(tmp_path / "predicate")
    segment = FlowpipeSegment(
        tm=current,
        final_tm=current,
        reset_tm=current,
        status="validated",
        h=0.01,
        order=2,
        validation_attempts=1,
        picard_image_remainder=[[-1e-5, -2e-5], [1e-5, 2e-5]],
    )
    diagnostic = {
        "phase": "remainder_validation",
        "h_try": 0.01,
        "validation_status": "validated",
        "rejection_reason": "",
        "picard_image_remainder_lo": [[-1e-5, -2e-5]],
        "picard_image_remainder_hi": [[1e-5, 2e-5]],
    }
    writer.record_step(
        step=0, t_pre=0.0, current=current, previous_state=None, segment=segment,
        diagnostics=[diagnostic], accepted=True, attempted_h=0.01, order=2,
    )
    writer.close(result_summary={"status": "validated"})
    with (tmp_path / "predicate" / "acceptance_attempts.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["accepted"] == "True"
    assert float(row["subset_margin_x"]) == pytest.approx(9e-5)
    assert float(row["subset_margin_y"]) == pytest.approx(8e-5)
    assert min(float(row["subset_margin_x"]), float(row["subset_margin_y"])) >= 0.0


def _base_flowstar_row(*, stage: str, record_type: str, accepted: bool = True) -> dict[str, object]:
    num0 = {"decimal": "0", "hex": "0x0.0p+0"}
    numh = {"decimal": "0.012500000000000001", "hex": (0.0125).hex()}
    row: dict[str, object] = {
        "tool": "flowstar", "source_commit": flowstar_process.SOURCE_COMMIT, "run_id": "synthetic-flowstar",
        "accepted_step_index": 0, "attempt_index": 0, "retry_index": 0, "t_pre": num0,
        "h_attempt": numh, "accepted": accepted, "rejection_reason": "", "state_component": -1,
        "stage": stage, "record_type": record_type,
    }
    if record_type == "transition":
        row.update(
            polynomial_range=None, remainder=None, self_map_candidate_box=None, self_map_image=None,
            basis_variable_order=["tau", "r0", "r1", "r2"],
        )
    return row


def test_flowstar_observation_equivalence_and_missing_stage_fail_closed(tmp_path):
    raw = tmp_path / "raw.jsonl"
    rows = [_base_flowstar_row(stage=stage, record_type="transition") for stage in sorted(flowstar_process.STAGES)]
    rows.append(_base_flowstar_row(stage="scheduler", record_type="acceptance_attempt"))
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    stdout = "time = 0.012500,\tstep = 0.012500,\torder = 4\n"
    for name in ("stock_stdout", "instrumented_stdout"):
        (tmp_path / name).write_text(stdout, encoding="utf-8")
    for name in ("stock_exit", "instrumented_exit"):
        (tmp_path / name).write_text("0\n", encoding="utf-8")
    for name in ("stock_x", "stock_y", "instrumented_x", "instrumented_y"):
        (tmp_path / name).write_bytes(b"same plot\n")
    args = flowstar_process.parse_args(
        [
            "--raw-trace", str(raw), "--stock-stdout", str(tmp_path / "stock_stdout"),
            "--instrumented-stdout", str(tmp_path / "instrumented_stdout"),
            "--stock-exit", str(tmp_path / "stock_exit"), "--instrumented-exit", str(tmp_path / "instrumented_exit"),
            "--stock-plot-x", str(tmp_path / "stock_x"), "--stock-plot-y", str(tmp_path / "stock_y"),
            "--instrumented-plot-x", str(tmp_path / "instrumented_x"),
            "--instrumented-plot-y", str(tmp_path / "instrumented_y"),
            "--output-dir", str(tmp_path / "processed"),
        ]
    )
    flowstar_process.run(args)
    metadata = json.loads((tmp_path / "processed" / "run_metadata.json").read_text())
    assert metadata["observational_equivalence"]["passed"] is True

    missing = tmp_path / "missing.jsonl"
    missing.write_text("".join(json.dumps(row) + "\n" for row in rows if row.get("stage") != "step_pre_state"))
    bad_args = flowstar_process.parse_args([*sum((
        [flag, str(value)] for flag, value in (
            ("--raw-trace", missing), ("--stock-stdout", tmp_path / "stock_stdout"),
            ("--instrumented-stdout", tmp_path / "instrumented_stdout"), ("--stock-exit", tmp_path / "stock_exit"),
            ("--instrumented-exit", tmp_path / "instrumented_exit"), ("--stock-plot-x", tmp_path / "stock_x"),
            ("--stock-plot-y", tmp_path / "stock_y"), ("--instrumented-plot-x", tmp_path / "instrumented_x"),
            ("--instrumented-plot-y", tmp_path / "instrumented_y"), ("--output-dir", tmp_path / "bad"),
        )
    ), [])])
    with pytest.raises(ValueError, match="missing lifecycle stages"):
        flowstar_process.run(bad_args)


def _write_attempts(path: Path, signatures: list[list[tuple[float, bool]]], tool: str) -> None:
    fields = [
        "tool", "source_commit", "run_id", "accepted_step_index", "attempt_index", "retry_index",
        "t_pre_decimal", "t_pre_hex", "h_attempt_decimal", "h_attempt_hex", "accepted",
        "rejection_reason", "state_component", "stage",
    ]
    attempt = 0
    t_pre = 0.0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for step, signature in enumerate(signatures):
            for retry, (h, accepted) in enumerate(signature):
                writer.writerow(
                    {
                        "tool": tool, "source_commit": "x", "run_id": tool, "accepted_step_index": step,
                        "attempt_index": attempt, "retry_index": retry, "t_pre_decimal": format(t_pre, ".17g"),
                        "t_pre_hex": t_pre.hex(), "h_attempt_decimal": format(h, ".17g"), "h_attempt_hex": h.hex(),
                        "accepted": accepted, "rejection_reason": "", "state_component": -1, "stage": "scheduler",
                    }
                )
                attempt += 1
            t_pre += signature[-1][0]


def _write_transition(path: Path, basis: list[str], steps: int, *, center: float | None = None, scale: float | None = None) -> None:
    rows = [
        {
            "tool": "test", "source_commit": "x", "run_id": "x", "accepted_step_index": step,
            "attempt_index": step, "retry_index": 0, "t_pre": encode_float(float(step)),
            "h_attempt": encode_float(0.01), "accepted": True, "rejection_reason": "",
            "state_component": 0, "stage": "step_pre_state", "basis_variable_order": basis,
            "center": encode_float(center) if center is not None else None,
            "normalization_scale": encode_float(scale) if scale is not None else None,
        }
        for step in range(steps)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_streaming_comparator_finds_schedule_divergence_and_basis_guard(tmp_path):
    torch_attempts = tmp_path / "torch.csv"
    flow_attempts = tmp_path / "flow.csv"
    common = [[(0.01, True)], [(0.011, True)]]
    _write_attempts(torch_attempts, [*common, [(0.0121, True)]], "torch")
    _write_attempts(flow_attempts, [*common, [(0.0121, False), (0.00605, True)]], "flowstar")
    torch_transitions = tmp_path / "torch.jsonl"
    flow_transitions = tmp_path / "flow.jsonl"
    _write_transition(torch_transitions, ["u0", "u1", "tau"], 3)
    _write_transition(flow_transitions, ["tau", "r0", "r1"], 3)
    args = compare.parse_args(
        [
            "--torch-attempts", str(torch_attempts), "--torch-transitions", str(torch_transitions),
            "--flowstar-attempts", str(flow_attempts), "--flowstar-transitions", str(flow_transitions),
            "--output-dir", str(tmp_path / "comparison"),
        ]
    )
    result = compare.compare(args)
    assert result["schedule_divergence_step"] == 2
    assert result["last_common_step"] == 1
    assert result["coefficient_comparison_available"] is False
    assert result["all_tolerance_checks_persist"] is True
    assert (tmp_path / "comparison" / "first_schedule_divergence.json").exists()
    with pytest.raises(ValueError, match="no native schedule divergence"):
        compare._first_schedule_divergence(torch_attempts, torch_attempts)

    same_basis_left = tmp_path / "same_basis_left.jsonl"
    same_basis_right = tmp_path / "same_basis_right.jsonl"
    _write_transition(same_basis_left, ["u0", "u1", "tau"], 1, center=1.0, scale=2.0)
    _write_transition(same_basis_right, ["u0", "u1", "tau"], 1, center=1.0, scale=3.0)
    scale_guard = compare.basis_guard(same_basis_left, same_basis_right)
    assert scale_guard["basis_equal"] is True
    assert scale_guard["center_equal"] is True
    assert scale_guard["normalization_scale_equal"] is False
    assert scale_guard["coefficient_comparison_available"] is False


def test_call44_identity_lineage_coverage_and_reconstruction(tmp_path):
    checkpoint = (
        ROOT / "evidence" / "vdp_terminal_range_closure" / "20260805T055556Z" / "05_fresh_horizons"
        / "t6p5_proactive_d1_truncation" / "terminal_checkpoint"
    )
    args = lineage.parse_args(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path / "lineage")])
    result = lineage.run(args)
    assert result["discarded_route_count"] == 1141
    assert result["coverage"] == 1.0
    assert result["parent_chain_complete"] is True
    assert result["reconstruction_passed"] is True
    coverage = json.loads((tmp_path / "lineage" / "lineage_coverage.json").read_text())
    assert coverage["missing_parent_ids"] == []
    assert coverage["reconstruction_max_abs_error"] == 0.0
    assert coverage["historical_max_abs_error"] <= coverage["rounding_tolerance"]
    identity = json.loads((tmp_path / "lineage" / "call44_identity.json").read_text())
    payload = identity["identity_payload"]
    assert payload["component"] == 1
    assert payload["component_name"] == "y"
    assert payload["accepted_count_before_attempt"] == 307
    assert payload["accepted_step_index"] is None
    assert payload["attempt_outcome"] == "rejected"
    assert payload["lineage_scope"] == "terminal_local_expression_lineage"
    assert payload["cross_step_lineage_complete"] is False
    nodes = [
        json.loads(line)
        for line in (tmp_path / "lineage" / "call44_lineage.jsonl").read_text().splitlines()
    ]
    x_roots = [row for row in nodes if row["operation_type"] == "terminal_picard_candidate_x"]
    y_roots = [row for row in nodes if row["operation_type"] == "terminal_picard_candidate_y"]
    assert x_roots and all(row["state_component"] == 0 for row in x_roots)
    assert y_roots and all(row["state_component"] == 1 for row in y_roots)
    assert all(row["accepted_count_before_attempt"] == 307 for row in nodes)
    assert all(row["accepted_step_index"] is None for row in nodes)
    assert all(row["attempt_outcome"] == "rejected" for row in nodes)


def test_call44_rejects_numerically_equal_but_differently_packaged_checkpoint():
    checkpoint = (
        ROOT / "evidence" / "vdp_terminal_range_closure" / "20260805T055556Z" / "05_fresh_horizons"
        / "t10_proactive_d1_truncation" / "terminal_checkpoint" / "terminal_state.json"
    )
    with pytest.raises(ValueError, match="checkpoint full SHA256 changed"):
        lineage._validate_checkpoint_identity(checkpoint)
