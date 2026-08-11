from dataclasses import replace
import importlib.util
import json
from pathlib import Path

from torch_tm_flowpipe import Interval, load_terminal_checkpoint
from torch_tm_flowpipe.batched_dense_tm import DenseRangePolicy
from torch_tm_flowpipe.structured_remainder import StructuredRemainderState


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/run_s1_prefix_complete_o4.py"
SPEC = importlib.util.spec_from_file_location("run_s1_prefix_checkpoint_triad", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _policy():
    spec = runner.CONTRACT["dense_range_policy"]
    return DenseRangePolicy(
        method=spec["method"],
        max_depth=spec["max_depth"],
        max_leaves=spec["max_leaves"],
        split_vars=tuple(spec["split_vars"]),
        trigger=spec["trigger"],
        named_contexts=tuple(spec["named_contexts"]),
        variable_orders=tuple(tuple(row) for row in spec["variable_orders"]),
    )


def _one_step_prestates():
    ode = runner.PolynomialODE.from_system_spec(runner.CONTRACT["canonical_system_spec"])
    policy = _policy()
    h = 0.005
    l0 = runner._run_lane_step(
        ode,
        [Interval(1.1, 1.4), Interval(2.35, 2.45)],
        None,
        lane="L0",
        h=h,
        h_min=h,
        h_max=h,
        policy=policy,
        diagnostics=[],
        diagnostics_context={"lane": "L0"},
    )
    assert l0.status == "validated"
    structured_current, structured_normal = runner._initialize_structured_lane()
    l2 = runner._run_lane_step(
        ode,
        structured_current,
        structured_normal,
        lane="L2",
        h=h,
        h_min=h,
        h_max=h,
        policy=policy,
        diagnostics=[],
        diagnostics_context={"lane": "L2"},
    )
    assert l2.status == "validated"
    l1_structured, _ = runner._materialize_every_boundary(
        l2.flowstar_normal_state.structured_remainder_state
    )
    l1_normal = replace(
        l2.flowstar_normal_state,
        structured_remainder_state=l1_structured,
    )
    return ode, policy, {
        "L0": (l0.reset_tm, l0.flowstar_normal_state),
        "L1": (l2.reset_tm, l1_normal),
        "L2": (l2.reset_tm, l2.flowstar_normal_state),
    }


def test_checkpoint_helper_serializes_none_empty_and_live_structured_states(tmp_path):
    _, _, prestates = _one_step_prestates()
    schedule = json.loads(
        (
            ROOT
            / "outputs/s1_prefix_integrated_complete_o4_20260810/20260810T095423Z"
            / "04_frozen_schedule_prefix/frozen_schedule.json"
        ).read_text(encoding="utf-8")
    )
    frozen = schedule["rows"][1]
    loaded = {}
    for lane, (current, normal) in prestates.items():
        directory = tmp_path / lane
        record = runner._save_boundary_checkpoint(
            directory,
            boundary=1,
            current=current,
            normal_state=normal,
            frozen=frozen,
            rows=[],
            provenance={"lane": lane},
        )
        assert record["byte_stable"] is True
        loaded[lane] = load_terminal_checkpoint(
            directory / "boundary_001_checkpoint",
            expected_order=4,
            expected_dtype="float64",
        )

    assert loaded["L0"].manifest["schema"].endswith("_v1")
    assert loaded["L0"].normal_state.structured_remainder_state is None
    for lane in ("L1", "L2"):
        assert loaded[lane].manifest["schema"].endswith("_v2")
        assert isinstance(
            loaded[lane].normal_state.structured_remainder_state,
            StructuredRemainderState,
        )
    assert loaded["L1"].normal_state.structured_remainder_state.active.sum().item() == 0
    assert loaded["L2"].normal_state.structured_remainder_state.active.sum().item() == 1


def test_full_h_diagnostic_is_first_attempt_only_and_never_commits(tmp_path):
    ode, policy, prestates = _one_step_prestates()
    schedule = json.loads(
        (
            ROOT
            / "outputs/s1_prefix_integrated_complete_o4_20260810/20260810T095423Z"
            / "04_frozen_schedule_prefix/frozen_schedule.json"
        ).read_text(encoding="utf-8")
    )
    frozen = schedule["rows"][1]
    for lane, (current, normal) in prestates.items():
        diagnostic = runner._frozen_full_h_diagnostic(
            tmp_path / lane,
            lane=lane,
            attempt_index=1,
            current=current,
            normal_state=normal,
            frozen=frozen,
            ode=ode,
            policy=policy,
        )
        assert diagnostic["max_validation_attempts"] == 1
        assert diagnostic["adaptive_shrink_allowed"] is False
        assert diagnostic["returned_state_committed"] is False
        assert diagnostic["prestate_sha256"] == diagnostic["poststate_sha256"]
        assert diagnostic["first_validator_diagnostic"]["attempt"] == 1
