# Rescue Vs Original Flow* Comparison

Requested horizon: `5`.
Original Flow* boxes are parsed GNUPLOT segment boxes; this comparison uses overlap hulls, not exact segment-count matching.
This is not a Flow* parity claim unless boxes are numerically identical, which is not expected here.

Best rescue config: `flowstar_style_o6_candidate8_output6_flowstar_ctrunc`.
Reached requested horizon? no.
Width comparable to Flow*? no.

## Metrics

| run_id | py_status | py_last_validated_t | py_segments | last_width_ratio | tube_width_ratio | max_overlap_ratio | median_overlap_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o6_target_flowstar_ctrunc | failed | 2.1095541733932355 | 127 | 30.05005498910863 | 1.962772284043342 | 30.05005498910863 | 12.682114768606521 |
| flowstar_style_o6_candidate8_output6_flowstar_ctrunc | failed | 2.400737667399793 | 95 | 68.36824483254583 | 4.835529788768181 | 68.36824483254583 | 17.777197082202203 |
| flowstar_style_o6_candidate8_output6_cutoff_flowstar_ctrunc | failed | 2.400737667399793 | 95 | 68.36824483254506 | 4.835529788768126 | 68.36824483254506 | 17.777197082202154 |
