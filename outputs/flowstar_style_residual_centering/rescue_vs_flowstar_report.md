# Rescue Vs Original Flow* Comparison

Requested horizon: `5`.
Original Flow* boxes are parsed GNUPLOT segment boxes; this comparison uses overlap hulls, not exact segment-count matching.
This is not a Flow* parity claim unless boxes are numerically identical, which is not expected here.

Best rescue config: `flowstar_style_o6_candidate8_output6_centered`.
Reached requested horizon? no.
Width comparable to Flow*? no.

## Metrics

| run_id | py_status | py_last_validated_t | py_segments | last_width_ratio | tube_width_ratio | max_overlap_ratio | median_overlap_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o6_target_centered | failed | 2.1159710308044115 | 127 | 30.850786594767612 | 2.0007043188924944 | 30.850786594767612 | 12.989615772365575 |
| flowstar_style_o6_candidate8_output6_centered | failed | 2.400737667399793 | 95 | 68.36823036058485 | 4.835528765199437 | 68.36823036058485 | 17.777195897097144 |
| flowstar_style_o6_candidate8_output6_cutoff_centered | failed | 2.400737667399793 | 95 | 68.36823036058439 | 4.835528765199405 | 68.36823036058439 | 17.777195897097112 |
