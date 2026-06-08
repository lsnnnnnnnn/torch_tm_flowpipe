# Flow* Box Lifecycle Alignment Audit

This audit checks stage-labeled boxes for the first same-t/h Picard residual mismatch. It does not change solver behavior and does not claim Flow* parity.

## Scope

- t_before requested: `0`
- h_try: `0.025000000000000001`
- Input traces: `outputs/flowstar_step_trace_compare/*.csv`
- Output ledger: `outputs/flowstar_box_lifecycle_alignment_audit/box_lifecycle_ledger.csv`

## Answers

- Are Flow* and PyTorch pre_step boxes equal? `true`.
- Are endpoint-before-center boxes comparable? `true`.
- Are reset-after-center boxes comparable? `unknown`.
- Which lifecycle stage first differs? `endpoint_box_before_center`.
- Are residuals computed over the same stage? `false`.
- Picard residual comparison: `noncausal/stage-misaligned`.
- Flow* residual components still missing: `picard_no_remainder_residual`.

## Stage Ledger

| source | status | pre_step box | endpoint-before-center box | reset-after-center box | residual stage valid |
| --- | --- | --- | --- | --- | --- |
| flowstar | rejected | x=[1.1000000000000001, 1.3999999999999999], y=[2.3500000000000005, 2.4500000000000002] | x=[1.0975301322957665, 1.461604968996091], y=[2.2509745318250358, 2.4781433092829666] | unknown | false |
| torch_noqueue | accepted | x=[1.1000000000000001, 1.3999999999999999], y=[2.3500000000000001, 2.4500000000000002] | x=[1.1582107932286989, 1.4600697187422169], y=[2.2532107556132885, 2.4061837607653755] | x=[1.1582107932286996, 1.4600853128697295], y=[2.2532107556132908, 2.4073776576718386] | false |
| torch_v2 | accepted | x=[1.1000000000000001, 1.3999999999999999], y=[2.3500000000000001, 2.4500000000000002] | x=[1.1582107932286989, 1.4600697187422169], y=[2.2532107556132885, 2.4061837607653755] | x=[1.1582107932286996, 1.4600853128697295], y=[2.2532107556132908, 2.4073776576718386] | false |

## Interpretation

The residual endpoint mismatch is not yet a valid same-local-box comparison. Stage-labeled boxes must align before the residual source can be attributed to no-remainder, raw-ctrunc, cutoff, target, or tolerance.
