# Flow* Adaptive Step Policy Audit

This audit is local-source-only and does not rerun h5 or h10. It records the adaptive step-size policy needed by the opt-in raw-remainder compatibility experiments. It does not change default PyTorch solver behavior and does not claim Flow* parity.

## Sources

- `/srv/local/shengenli/flowstar/flowstar-toolbox/include.h` defines `LAMBDA_DOWN` as `0.5` and `LAMBDA_UP` as `1.1`.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp` shrinks a rejected candidate step by `current_step * LAMBDA_DOWN`, then stops if the new step is below `step_min`.
- `experiments/flowstar_probe/flowstar_vdp_step_trace_probe.cpp` records `h_after_if_rejected_or_next` as `h_try * LAMBDA_DOWN` on rejection and `h_try * LAMBDA_UP` on acceptance. Its main loop computes the next candidate as `current_stepsize * LAMBDA_UP` and clamps through the existing `step_max` logic.

## Audited Policy

- Rejected attempt: multiply the failed candidate step by `0.5`.
- Accepted attempt: multiply the accepted step by `1.1` for the next candidate.
- Lower bound: do not continue after shrink if the candidate is below `h_min`.
- Upper bound: respect `h_max` when choosing the next candidate.
- State carried forward: the policy only changes the next step-size proposal; it does not alter the raw-remainder validation rule, reset mode, ODE, Taylor order, or production sparse TaylorModel API.

## Implementation Boundary

The PyTorch implementation keeps this behavior behind `step_policy_mode="flowstar_compat"`. The default adaptive policy continues to use its caller-provided `grow_factor`, currently `1.5` in these experiments, so existing default behavior is unchanged.
