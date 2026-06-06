# Rescue Vs Original Flow* Comparison

Requested horizon: `5`.
Original Flow* boxes are parsed GNUPLOT segment boxes; this comparison uses overlap hulls, not exact segment-count matching.
This is not a Flow* parity claim unless boxes are numerically identical, which is not expected here.

Best rescue config: `flowstar_style_o6_candidate8_output6_cutoff_insert`.
Reached requested horizon? yes.
Width comparable to Flow*? no.

## Metrics

| run_id | py_status | py_last_validated_t | py_segments | last_width_ratio | tube_width_ratio | max_overlap_ratio | median_overlap_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o6_candidate8_output6_cutoff | failed | 2.400737667399793 | 95 | 68.36823036058439 | 4.835528765199405 | 68.36823036058439 | 17.777195897097112 |
| flowstar_style_o6_candidate8_output6_cutoff_insert | max_horizon_reached | 5.0 | 56 | 1.042824337979132 | 0.9999174731419563 | 1.042824337979132 | 0.5030069124617351 |
