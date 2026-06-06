# Rescue Vs Original Flow* Comparison

Requested horizon: `10`.
Original Flow* boxes are parsed GNUPLOT segment boxes; this comparison uses overlap hulls, not exact segment-count matching.
This is not a Flow* parity claim unless boxes are numerically identical, which is not expected here.

Best rescue config: `flowstar_style_o6_candidate8_output6_insert_normaleval`.
Reached requested horizon? no.
Width comparable to Flow*? no.

## Metrics

| run_id | py_status | py_last_validated_t | py_segments | last_width_ratio | tube_width_ratio | max_overlap_ratio | median_overlap_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o4_target_insert_normaleval | failed | 6.47300880580919 | 239 | 14.049230879967748 | 1.1807087361819038 | 14.049230879967748 | 1.3359318424177433 |
| flowstar_style_o6_candidate8_output6_insert_normaleval | failed | 7.496039258138734 | 150 | 101.26404571229271 | 2.4989346486924005 | 101.26404571229271 | 3.039289234020476 |
| flowstar_style_o4_target_cutoff_insert_normaleval | failed | 6.47300880580919 | 239 | 14.049241274995854 | 1.1807088850728908 | 14.049241274995854 | 1.3359322478410462 |
| flowstar_style_o6_candidate8_output6_cutoff_insert_normaleval | failed | 7.496039258138734 | 150 | 101.26699060370883 | 2.498998450040144 | 101.26699060370883 | 3.0392999108523204 |
