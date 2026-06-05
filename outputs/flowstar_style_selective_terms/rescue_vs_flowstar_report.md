# Rescue Vs Original Flow* Comparison

Requested horizon: `5`.
Original Flow* boxes are parsed GNUPLOT segment boxes; this comparison uses overlap hulls, not exact segment-count matching.
This is not a Flow* parity claim unless boxes are numerically identical, which is not expected here.

Best rescue config: `flowstar_style_o6_candidate8_output6_keep8`.
Reached requested horizon? no.
Width comparable to Flow*? no.

## Metrics

| run_id | py_status | py_last_validated_t | py_segments | last_width_ratio | tube_width_ratio | max_overlap_ratio | median_overlap_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o6_candidate8_output6_keep4 | failed | 2.3236324377489392 | 112 | 67.07157930147092 | 4.382710422884737 | 67.07157930147092 | 17.166319078243923 |
| flowstar_style_o6_candidate8_output6_keep8 | failed | 2.345909199029081 | 100 | 69.67672951806126 | 4.552940781649833 | 69.67672951806126 | 17.054018952386926 |
| flowstar_style_o6_candidate8_output6_keep4_centered | failed | 2.3398728391401784 | 112 | 71.4738121596852 | 4.6703689517668545 | 71.4738121596852 | 17.58478385729068 |
| flowstar_style_o6_candidate8_output6_keep8_centered | failed | 2.345909199029081 | 100 | 69.67672951806126 | 4.552940781649833 | 69.67672951806126 | 17.054018952386926 |
