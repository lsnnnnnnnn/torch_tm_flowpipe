# Rescue Vs Original Flow* Comparison

Requested horizon: `10`.
Original Flow* boxes are parsed GNUPLOT segment boxes; this comparison uses overlap hulls, not exact segment-count matching.
This is not a Flow* parity claim unless boxes are numerically identical, which is not expected here.

Best rescue config: `flowstar_style_o6_candidate8_output6_insert_scalars`.
Reached requested horizon? no.
Width comparable to Flow*? no.

## Metrics

| run_id | py_status | py_last_validated_t | py_segments | last_width_ratio | tube_width_ratio | max_overlap_ratio | median_overlap_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o4_target_insert_scalars | failed | 6.47300880580919 | 239 | 14.015168420704509 | 1.1802424936476639 | 14.015168420704509 | 1.3343125200837533 |
| flowstar_style_o6_candidate8_output6_insert_scalars | failed | 7.496039258138734 | 150 | 101.25756903259277 | 2.498791881550163 | 101.25756903259277 | 3.0392717282476918 |
