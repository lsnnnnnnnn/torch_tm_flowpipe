# Flow* Step Trace Divergence Report

This is a diagnostic probe, not a Flow* parity claim.

## Executive conclusion

- Horizon traced: T=1
- First causal divergence: adaptive acceptance / residual validation.
- Accepted ordinal comparisons are retained only as noncausal diagnostics when `t` or `h` differ.

## Accepted ordinal comparison

- Comparison kind: `accepted_ordinal_trace_diff_noncausal`
- First material channel: `adaptive_step_alignment_mismatch`
- Channel attribution valid: `false`
- Flow* h: `0.0125`; no_queue h: `0.025`; v2 h: `0.025`
- Warning: noqueue_h differs: Flow*=0.012500000000000001, torch=0.025000000000000001; v2_h differs: Flow*=0.012500000000000001, torch=0.025000000000000001

## Attempt-aligned comparison

- Does Flow* reject h=0.025 at t=0? `yes`
- Does PyTorch no_queue accept h=0.025 at t=0? `yes`
- Does PyTorch v2 accept h=0.025 at t=0? `yes`
- Flow* h=0.025 evidence: status=`rejected`, residual_width_sum=``, target_width_sum=``, residual_over_target_sum=``
- no_queue h=0.025 evidence: status=`accepted`, residual_width_sum=``, target_width_sum=``, residual_over_target_sum=``
- v2 h=0.025 evidence: status=`accepted`, residual_width_sum=``, target_width_sum=``, residual_over_target_sum=``
- First causal divergence: `adaptive_acceptance_policy` at t=`0.0`, h=`0.025`.
- Flow* rejection reason: target miss

## Forced-h replay

- Under the Flow* accepted h schedule, PyTorch accepts all replayed rows present in the ledger: `yes`
- First numeric channel divergence: `right_map_range` at forced step `0`.
- right_map ratios no_queue/v2: `2.5` / `2.5`
- reset ratios no_queue/v2: `None` / `None`
- output_range ratios no_queue/v2: `None` / `None`

## Interpretation

- The attempt-aligned comparator is the causal guard: channel attribution is valid only when `t_before`, `h_try`, and attempt index align.
- The accepted ordinal diff remains useful for regression monitoring, but its first row compares different step sizes and must not be used as first causal channel attribution.
- The adaptive acceptance divergence occurs before the forced-h numeric channel divergence.
- The forced-h result is consistent with the prior right_map/preconditioning/output-range width attribution thread.

## Next recommendation

- Fix PyTorch acceptance policy/target residual validation.

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
