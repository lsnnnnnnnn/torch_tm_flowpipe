# Rescue Variant Comparison Next4

Previous best candidate_order=8/output_order=6: `flowstar_style_o6_candidate8_output6` at t=`2.4007376673997931`.
Did Flow* one-step actually run? yes.
One-step oracle Flow* validates same local box? no.
Best flowstar_ctrunc validation: `flowstar_style_o6_candidate8_output6_flowstar_ctrunc` at t=`2.4007376673997931`.
Best selective validation-path run: `flowstar_style_o6_candidate8_output6_keep8` at t=`2.345909199029081`.
Branch decision: NEEDS_MORE_WORK.
Decision: Flow* one-step also fails from the PyTorch reset box; focus on width reduction before that point.

## Rows

| item | run_id | status | last_validated_t | flowstar_validated | pytorch_validated | notes |
| --- | --- | --- | ---: | --- | --- | --- |
| previous_best_candidate_order | flowstar_style_o6_candidate8_output6 | failed | 2.4007376673997931 |  |  | Picard residual not subset of target remainder |
| one_step_oracle | flowstar_one_step_oracle_candidate8_cutoff | not_completed |  | False | False | local one-step diagnostic only; no full parity claim |
| flowstar_ctrunc_validation | flowstar_style_o6_candidate8_output6_flowstar_ctrunc | failed | 2.4007376673997931 |  |  | Flowstar ctrunc tmp remainder not subset of target remainder |
| selective_validation_path | flowstar_style_o6_candidate8_output6_keep8 | failed | 2.345909199029081 |  |  | Picard residual not subset of target remainder |
