# Rescue Variant Comparison

Best variant by current decision criteria: `flowstar_style_o6_target_adaptive_order_8` at t=`2.2771582567640953`.
Reached horizon 5? no.
Width ratio vs Flow*: last=`80.930899048992146`, tube=`4.9151377009180992`.
Next recommendation: prioritize tighter polynomial range bounding, then a real Flow*-style symbolic remainder queue.

Decision criteria: highest last_validated_t, target remainder close to Flow* parameter, runtime, width ratio vs Flow*, and no non-final h below 0.002 except diagnostic runs.

## Rows

| group | run_id | status | last_validated_t | radius | last_width_ratio | tube_width_ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| adaptive_order_fallback | flowstar_style_o6_target_adaptive_order_8 | failed | 2.2771582567640953 | 0.0001 | 80.930899048992146 | 4.9151377009180992 |
| adaptive_order_fallback | flowstar_style_o6_target_cutoff_adaptive_order_8 | failed | 2.2771582567640953 | 0.0001 | 80.930899048992146 | 4.9151377009180992 |
| h5_current_best | flowstar_style_o6_target | failed | 2.1095541733932355 | 0.0001 | 30.050027407228164 | 1.9627712588202828 |
| h5_current_best | flowstar_style_o6_target_cutoff | failed | 2.1095541733932355 | 0.0001 | 30.050027407228164 | 1.9627712588202828 |
| refined_target_validation | flowstar_style_o6_target_refined | failed | 2.0436221052452312 | 0.0001 | 37.074927430642745 | 2.2119455800712333 |
| refined_target_validation | flowstar_style_o6_target_refined_cutoff | failed | 2.0436221052452312 | 0.0001 | 37.074927430642745 | 2.2119455800712333 |
| relaxed_target_remainder | flowstar_style_o6_target_r2e-4 | failed | 2.1775565845009179 | 0.00020000000000000001 | 33.751528901285091 | 2.402243110950153 |
| relaxed_target_remainder | flowstar_style_o6_target_r5e-4 | failed | 2.2567245519061094 | 0.00050000000000000001 | 52.322206004362734 | 2.9715313585372285 |
