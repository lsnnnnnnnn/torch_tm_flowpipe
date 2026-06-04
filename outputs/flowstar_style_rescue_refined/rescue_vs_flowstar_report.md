# Rescue Vs Original Flow* Comparison

Requested horizon: `5`.
Original Flow* boxes are parsed GNUPLOT segment boxes; this comparison uses overlap hulls, not exact segment-count matching.
This is not a Flow* parity claim unless boxes are numerically identical, which is not expected here.

Best rescue config: `flowstar_style_o6_target_refined`.
Reached requested horizon? no.
Width comparable to Flow*? no.

## Metrics

| run_id | py_status | py_last_validated_t | py_segments | last_width_ratio | tube_width_ratio | max_overlap_ratio | median_overlap_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o6_target_refined | failed | 2.0436221052452312 | 180 | 37.074927430642745 | 2.2119455800712333 | 37.074927430642745 | 10.467243332368826 |
| flowstar_style_o6_target_refined_cutoff | failed | 2.0436221052452312 | 180 | 37.074927430642745 | 2.2119455800712333 | 37.074927430642745 | 10.467243332368826 |
