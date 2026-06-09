# Flow* Step Trace Divergence Report

This is a diagnostic probe, not a Flow* parity claim.

## Executive conclusion

- Horizon traced: T=0.5
- First causal divergence: adaptive acceptance / residual validation.
- Accepted ordinal comparisons are retained only as noncausal diagnostics when `t` or `h` differ.

## Accepted ordinal comparison

- Comparison kind: `accepted_ordinal_trace_diff_noncausal`
- First material channel: `adaptive_step_alignment_mismatch`
- Channel attribution valid: `false`
- Flow* h: `0.012500000000000001`; no_queue h: `0.025`; v2 h: `0.025`
- Warning: noqueue_h differs: Flow*=0.012500000000000001, torch=0.025000000000000001; v2_h differs: Flow*=0.012500000000000001, torch=0.025000000000000001

## Attempt-aligned comparison

- Does Flow* reject h=0.025 at t=0? `yes`
- Does PyTorch no_queue accept h=0.025 at t=0? `yes`
- Does PyTorch v2 accept h=0.025 at t=0? `yes`
- Flow* h=0.025 evidence: status=`rejected`, residual_width_sum=`0.0002072197440771965`, target_width_sum=`0.00040000000000000002`, residual_over_target_sum=`0.51804936019299119`
- no_queue h=0.025 evidence: status=`accepted`, residual_width_sum=`0.00011861120613006697`, target_width_sum=`0.00040000000000000002`, residual_over_target_sum=`0.29652801532516743`
- v2 h=0.025 evidence: status=`accepted`, residual_width_sum=`0.00011861120613006697`, target_width_sum=`0.00040000000000000002`, residual_over_target_sum=`0.29652801532516743`
- First causal divergence: `adaptive_acceptance_policy` at t=`9.9999999999999998e-13`, h=`0.025000000000000001`.
- Flow* rejection reason: Picard_ctrunc_normal remainder not contained in target; shrinking h

## Acceptance predicate endpoints

- Flow* h=0.025: subset_x=`yes`, residual_x=`[-5.7034993171691869e-06, 9.6267419600486818e-06]`, target_x=`[-0.0001, 0.0001]`; subset_y=`no`, residual_y=`[-8.3561112430831106e-05, 0.0001083283903691475]`, target_y=`[-0.0001, 0.0001]`; which_dim_failed=`y`.
- PyTorch no_queue h=0.025: subset_x=`yes`, residual_x=`[-3.5138691795527178e-06, 4.7947416664619525e-06]`, target_x=`[-0.0001, 0.0001]`; subset_y=`yes`, residual_y=`[-5.1533346624557205e-05, 5.8769252659495084e-05]`, target_y=`[-0.0001, 0.0001]`; which_dim_failed=`none`.
- PyTorch v2 h=0.025: subset_x=`yes`, residual_x=`[-3.5138691795527178e-06, 4.7947416664619525e-06]`, target_x=`[-0.0001, 0.0001]`; subset_y=`yes`, residual_y=`[-5.1533346624557205e-05, 5.8769252659495084e-05]`, target_y=`[-0.0001, 0.0001]`; which_dim_failed=`none`.
- Width comparison is not the acceptance predicate; endpoint-wise interval inclusion is. A residual may have smaller width than the target and still fail if it is shifted outside the target interval.
- Detailed component ledger: `outputs/flowstar_acceptance_predicate_audit/acceptance_predicate_ledger.csv`.

## Forced-h replay

- Under the Flow* accepted h schedule, PyTorch accepts all replayed rows present in the ledger: `yes`
- First numeric channel divergence: `center_scaling` at forced step `0`.
- right_map ratios no_queue/v2: `0.10693024082261673` / `0.10693024082261673`
- reset ratios no_queue/v2: `1.0709041391206482` / `1.0709041391206482`
- output_range ratios no_queue/v2: `0.8663881162703099` / `0.8663881162703099`

## Interpretation

- The attempt-aligned comparator is the causal guard: channel attribution is valid only when `t_before`, `h_try`, and attempt index align.
- The accepted ordinal diff remains useful for regression monitoring, but its first row compares different step sizes and must not be used as first causal channel attribution.
- The adaptive acceptance divergence occurs before the forced-h numeric channel divergence.

## Next recommendation

- First align same-source tube/endpoint objects: Flow* full-step tmvTmp tube vs PyTorch full-step validation-candidate tube; and Flow* tau=h endpoint vs PyTorch tau=h endpoint.

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
