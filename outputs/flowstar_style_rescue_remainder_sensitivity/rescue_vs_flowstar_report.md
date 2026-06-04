# Rescue Vs Original Flow* Comparison

Requested horizon: `5`.
Original Flow* boxes are parsed GNUPLOT segment boxes; this comparison uses overlap hulls, not exact segment-count matching.
This is not a Flow* parity claim unless boxes are numerically identical, which is not expected here.

Best rescue config: `flowstar_style_o6_target_r5e-4`.
Reached requested horizon? no.
Width comparable to Flow*? no.

## Metrics

| run_id | py_status | py_last_validated_t | py_segments | last_width_ratio | tube_width_ratio | max_overlap_ratio | median_overlap_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o6_target_r2e-4 | failed | 2.177556584500918 | 120 | 33.75152890128509 | 2.402243110950153 | 33.75152890128509 | 16.359409172721353 |
| flowstar_style_o6_target_r5e-4 | failed | 2.2567245519061094 | 105 | 52.322206004362734 | 2.9715313585372285 | 52.322206004362734 | 16.490725235457095 |
