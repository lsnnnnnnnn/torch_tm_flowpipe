# Flow* Step Trace Divergence Report

This is a diagnostic probe, not a Flow* parity claim.

## Executive conclusion

- Horizon traced: T=0.1
- No causal numeric channel was isolated over this short trace.
- Accepted ordinal comparisons are retained only as noncausal diagnostics when `t` or `h` differ.

## Accepted ordinal comparison

- No accepted ordinal material channel was found.

## Attempt-aligned comparison

- Does Flow* reject h=0.025 at t=0? `yes`
- Does PyTorch no_queue accept h=0.025 at t=0? `no`
- Does PyTorch v2 accept h=0.025 at t=0? `no`
- Flow* h=0.025 evidence: status=`rejected`, residual_width_sum=`0.010439664137713762`, target_width_sum=`0.00040000000000000002`, residual_over_target_sum=`26.099160344284403`
- no_queue h=0.025 evidence: status=`rejected`, residual_width_sum=`0.0045406162714681196`, target_width_sum=`0.00040000000000000002`, residual_over_target_sum=`11.351540678670299`
- v2 h=0.025 evidence: status=`rejected`, residual_width_sum=`0.0045406162714681196`, target_width_sum=`0.00040000000000000002`, residual_over_target_sum=`11.351540678670299`
- First numeric divergence under matched attempts: `picard_residual`.

## Acceptance predicate endpoints

- Flow* h=0.025: subset_x=`no`, residual_x=`[-0.00079615234375000009, 0.00067632812499999985]`, target_x=`[-0.0001, 0.0001]`; subset_y=`no`, residual_y=`[-0.0050102360843866482, 0.003956947584577113]`, target_y=`[-0.0001, 0.0001]`; which_dim_failed=`x;y`.
- PyTorch no_queue h=0.025: subset_x=`no`, residual_x=`[-0.0003793554717500005, 0.00033941406550000046]`, target_x=`[-0.0001, 0.0001]`; subset_y=`no`, residual_y=`[-0.0023864048138212079, 0.0014354419283969116]`, target_y=`[-0.0001, 0.0001]`; which_dim_failed=`x;y`.
- PyTorch v2 h=0.025: subset_x=`no`, residual_x=`[-0.0003793554717500005, 0.00033941406550000046]`, target_x=`[-0.0001, 0.0001]`; subset_y=`no`, residual_y=`[-0.0023864048138212079, 0.0014354419283969116]`, target_y=`[-0.0001, 0.0001]`; which_dim_failed=`x;y`.
- Width comparison is not the acceptance predicate; endpoint-wise interval inclusion is. A residual may have smaller width than the target and still fail if it is shifted outside the target interval.
- Detailed component ledger: `outputs/flowstar_acceptance_predicate_audit/acceptance_predicate_ledger.csv`.

## Forced-h replay

- Under the Flow* accepted h schedule, PyTorch accepts all replayed rows present in the ledger: `no`
- First numeric channel divergence: `unknown` or not reached.

## Interpretation

- The attempt-aligned comparator is the causal guard: channel attribution is valid only when `t_before`, `h_try`, and attempt index align.
- The accepted ordinal diff remains useful for regression monitoring, but its first row compares different step sizes and must not be used as first causal channel attribution.

## Next recommendation

- Investigate cutoff/poly-diff accounting.

## Output files

- `outputs/flowstar_step_trace_compare/flowstar_trace.csv`
- `outputs/flowstar_step_trace_compare/torch_noqueue_trace.csv`
- `outputs/flowstar_step_trace_compare/torch_v2_trace.csv`
- `outputs/flowstar_step_trace_compare/aligned_trace_diff.csv`
- `outputs/flowstar_step_trace_compare/attempt_aligned_trace_diff.csv`
- `outputs/flowstar_step_trace_compare/forced_h_trace_diff.csv`
- `outputs/flowstar_step_trace_compare/attempt_alignment_warnings.csv`
- `outputs/flowstar_step_trace_compare/forced_h_width_channel_ledger.csv`

## Limitations

- The Flow* C++ probe is an oracle/instrumentation probe; this change does not add a new flowpipe mechanism or symbolic queue variant.
- Fields absent in a mode are left blank in the trace and reported as unknown by the comparator.
- This report does not compare PyTorch endpoint boxes to Flow* GNUPLOT segment boxes.
