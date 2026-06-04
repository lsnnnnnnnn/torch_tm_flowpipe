# Rescue Vs Original Flow* Comparison

Requested horizon: `5`.
Original Flow* boxes are parsed GNUPLOT segment boxes; this comparison uses overlap hulls, not exact segment-count matching.
This is not a Flow* parity claim unless boxes are numerically identical, which is not expected here.

Best rescue config: `flowstar_style_o6_target`.
Reached requested horizon? no.
Width comparable to Flow*? no.

## Metrics

| run_id | py_status | py_last_validated_t | py_segments | last_width_ratio | tube_width_ratio | max_overlap_ratio | median_overlap_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o6_target | failed | 2.1095541733932355 | 127 | 30.050027407228164 | 1.9627712588202828 | 30.050027407228164 | 12.682109405545757 |
| flowstar_style_o6_target_cutoff | failed | 2.1095541733932355 | 127 | 30.050027407228164 | 1.9627712588202828 | 30.050027407228164 | 12.682109405545757 |
