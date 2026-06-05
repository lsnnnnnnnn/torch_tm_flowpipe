# Rescue Vs Original Flow* Comparison

Requested horizon: `5`.
Original Flow* boxes are parsed GNUPLOT segment boxes; this comparison uses overlap hulls, not exact segment-count matching.
This is not a Flow* parity claim unless boxes are numerically identical, which is not expected here.

Best rescue config: `flowstar_style_o6_candidate8_output6_truncsplit2`.
Reached requested horizon? no.
Width comparable to Flow*? no.

## Metrics

| run_id | py_status | py_last_validated_t | py_segments | last_width_ratio | tube_width_ratio | max_overlap_ratio | median_overlap_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o6_target_truncsplit2 | failed | 2.123634202721264 | 127 | 30.885943327720746 | 2.0029842681327326 | 30.885943327720746 | 12.867107531196723 |
| flowstar_style_o6_target_truncsplit4 | failed | 2.123634202721264 | 127 | 30.885943327720746 | 2.0029842681327326 | 30.885943327720746 | 12.867107531196723 |
| flowstar_style_o6_candidate8_output6_truncsplit2 | failed | 2.397165587736743 | 92 | 65.41439543796189 | 4.6266107683365485 | 65.41439543796189 | 18.36693401802532 |
