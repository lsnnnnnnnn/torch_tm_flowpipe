from torch_tm_flowpipe import Interval, flowpipe_step_flowstar_style_adaptive
from torch_tm_flowpipe.ode_examples import van_der_pol_ode


def _two_steps(backend):
    current = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    normal_state = None
    h_next = 0.01
    segments = []
    for _ in range(2):
        segment = flowpipe_step_flowstar_style_adaptive(
            van_der_pol_ode,
            current,
            h=h_next,
            h_min=0.002,
            h_max=0.1,
            order=4,
            target_remainder_radius=1e-4,
            cutoff_threshold=1e-10,
            max_validation_attempts=2,
            validation_mode="flowstar_raw_remainder_compat",
            reset_mode="normalized_insertion",
            step_policy_mode="flowstar_compat",
            tm_backend=backend,
            flowstar_normal_state=normal_state,
        )
        assert segment.status == "validated"
        segments.append(segment)
        current = segment.reset_tm
        normal_state = segment.flowstar_normal_state
        h_next = segment.next_h
    return segments


def test_hybrid_dense_carry_matches_sparse_schedule_and_has_no_inner_fallback():
    sparse = _two_steps("sparse")
    dense = _two_steps("dense")
    assert [segment.h for segment in dense] == [segment.h for segment in sparse]
    for segment in dense:
        assert segment.backend_lane == "hybrid_dense_core"
        assert segment.backend_counters["inner_loop_conversions"] == 0
        assert segment.backend_counters["inner_loop_scalar_count"] == 0
        assert segment.backend_counters["sparse_fallback_count"] == 0
        assert segment.backend_counters["segment_boundary_conversions"] == 2
        assert segment.endpoint_raw_tm is not None
        assert segment.tm is not segment.endpoint_raw_tm
        assert segment.reset_tm is not None

