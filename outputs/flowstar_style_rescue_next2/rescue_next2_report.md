# Rescue Variant Comparison Next2

Best variant by decision criteria: `flowstar_style_o6_candidate8_output6` at t=`2.4007376673997931`.
Reached horizon 5 with target_remainder_radius=1e-4? no.
Width ratio vs Flow*: last=`68.368230360584846`, tube=`4.8355287651994372`.
Next recommendation: continue tighter polynomial range bounding because it improved the validated horizon before moving to a symbolic remainder queue.

Residual-shift rows are diagnostic only and are not treated as accepted reachability runs.
Decision criteria: reaches horizon 5, no non-final h below 0.002, width ratio not worse than adaptive fallback, runtime, and no fake parity claims.

## Rows

| group | run_id | status | last_validated_t | candidate_order | output_order | split | last_width_ratio | tube_width_ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adaptive_order_fallback | flowstar_style_o6_target_adaptive_order_8 | failed | 2.2771582567640953 | 6 | 6 |  | 80.930899048992146 | 4.9151377009180992 |
| adaptive_order_fallback | flowstar_style_o6_target_cutoff_adaptive_order_8 | failed | 2.2771582567640953 | 6 | 6 |  | 80.930899048992146 | 4.9151377009180992 |
| candidate_order_output_order | flowstar_style_o6_candidate8_output6 | failed | 2.4007376673997931 | 8 | 6 |  | 68.368230360584846 | 4.8355287651994372 |
| candidate_order_output_order | flowstar_style_o6_candidate8_output6_cutoff | failed | 2.4007376673997931 | 8 | 6 |  | 68.368230360584391 | 4.8355287651994052 |
| candidate_order_truncation_split | flowstar_style_o6_candidate8_output6_truncsplit2 | failed | 2.397165587736743 | 8 | 6 | 2 | 65.414395437961886 | 4.6266107683365485 |
| h5_current_best | flowstar_style_o6_target | failed | 2.1095541733932355 | 6 | 6 |  | 30.050027407228164 | 1.9627712588202828 |
| h5_current_best | flowstar_style_o6_target_cutoff | failed | 2.1095541733932355 | 6 | 6 |  | 30.050027407228164 | 1.9627712588202828 |
| residual_shift_diagnostic | residual_shift_diagnostic_y | diagnostic_only | 2.4007376673997931 |  |  |  |  |  |
| truncation_range_split | flowstar_style_o6_target_truncsplit2 | failed | 2.1236342027212638 | 6 | 6 | 2 | 30.885943327720746 | 2.0029842681327326 |
| truncation_range_split | flowstar_style_o6_target_truncsplit4 | failed | 2.1236342027212638 | 6 | 6 | 4 | 30.885943327720746 | 2.0029842681327326 |
