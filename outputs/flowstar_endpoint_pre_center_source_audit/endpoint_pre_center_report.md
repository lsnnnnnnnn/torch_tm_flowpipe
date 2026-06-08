# Flow* Endpoint-Before-Center Source Audit

This is diagnostic-only and makes no solver change. It does not rerun h10, add symbolic queue variants, or claim Flow* parity.

## Scope

- t_before requested: `0`
- h_try: `0.025000000000000001`
- Input traces: `outputs/flowstar_step_trace_compare/*.csv`
- Output ledger: `outputs/flowstar_endpoint_pre_center_source_audit/endpoint_pre_center_ledger.csv`

## Answers

- Are Flow* and PyTorch endpoint-before-center fields semantically the same object? `false`.
- Causality: The endpoint-before-center comparison is noncausal because the source object labels or domain semantics do not match.
- What should be compared instead? Compare Flow* tmvTmp over the full step with a PyTorch validation candidate evaluated over the full tau domain, or compare both systems after tau=h substitution.
- If same object, what explains Flow* wider y_hi? No term/component attribution is valid yet; the first identified source is semantic stage mismatch, not a polynomial term.
- Does PyTorch currently under-account endpoint-before-center width? This trace does not prove PyTorch under-accounts endpoint-before-center width; it currently labels a different object/stage.
- Is Flow* endpoint-before-center perhaps a different stage? `yes`: Flow* source is `tmvTmp.Picard_ctrunc_normal_post_poly_diff`, while PyTorch source object(s) are `seg.final_tm.range_box;seg.final_tm.range_box`.
- Next minimal diagnostic: emit PyTorch validation-candidate full-step endpoint bounds and Flow* tau=h substituted endpoint bounds under the same source labels.

## Main Endpoint Rows

| source | status | source object | domain semantics | endpoint box | y_hi delta vs Flow* | semantic valid |
| --- | --- | --- | --- | --- | --- | --- |
| flowstar | rejected | tmvTmp.Picard_ctrunc_normal_post_poly_diff | physical_tube_over_step_exp_table_before_next_center_extraction | x=[1.0975301322957665, 1.461604968996091], y=[2.2509745318250358, 2.4781433092829666] | 0 | reference |
| torch_noqueue | accepted | seg.final_tm.range_box | physical_endpoint_after_tau_substitution_tau_dropped | x=[1.1582107932286989, 1.4600697187422169], y=[2.2532107556132885, 2.4061837607653755] | 0.07195954851759101 | false |
| torch_v2 | accepted | seg.final_tm.range_box | physical_endpoint_after_tau_substitution_tau_dropped | x=[1.1582107932286989, 1.4600697187422169], y=[2.2532107556132885, 2.4061837607653755] | 0.07195954851759101 | false |

## Diagnostic Variants

Variant rows in the CSV are diagnostic-only. Blank endpoint columns mean the trace did not expose those endpoints; they are not treated as zero.
