# Rescue Vs Original Flow* Comparison

Requested horizon: `5`.
Original Flow* boxes are parsed GNUPLOT segment boxes; this comparison uses overlap hulls, not exact segment-count matching.
This is not a Flow* parity claim unless boxes are numerically identical, which is not expected here.

Best rescue config: `flowstar_style_o6_target_adaptive_order_8`.
Reached requested horizon? no.
Width comparable to Flow*? no.

## Metrics

| run_id | py_status | py_last_validated_t | py_segments | last_width_ratio | tube_width_ratio | max_overlap_ratio | median_overlap_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o6_target_adaptive_order_8 | failed | 2.2771582567640953 | 185 | 80.93089904899215 | 4.915137700918099 | 80.93089904899215 | 21.68593042270046 |
| flowstar_style_o6_target_cutoff_adaptive_order_8 | failed | 2.2771582567640953 | 185 | 80.93089904899215 | 4.915137700918099 | 80.93089904899215 | 21.68593042270046 |
