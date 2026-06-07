# Rescue Vs Original Flow* Comparison

Requested horizon: `10`.
Original Flow* boxes are parsed GNUPLOT segment boxes; this comparison uses overlap hulls, not exact segment-count matching.
This is not a Flow* parity claim unless boxes are numerically identical, which is not expected here.

Best rescue config: `flowstar_style_o6_candidate8_output6_insert_symqueue_v2`.
Reached requested horizon? no.
Width comparable to Flow*? no.

## Metrics

| run_id | py_status | py_last_validated_t | py_segments | last_width_ratio | tube_width_ratio | max_overlap_ratio | median_overlap_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o4_target_insert_symqueue_v2 | failed | 6.47300880580919 | 239 | 14.373749217186306 | 1.1814463982967092 | 14.373749217186306 | 1.5071424898254695 |
| flowstar_style_o4_target_cutoff_insert_symqueue_v2 | failed | 6.47300880580919 | 239 | 14.373759614306335 | 1.1814465476029106 | 14.373759614306335 | 1.507142918578625 |
| flowstar_style_o6_candidate8_output6_insert_symqueue_v2 | failed | 7.496039258138734 | 150 | 101.53378043476893 | 2.5046094332157787 | 101.53378043476893 | 3.32601524159237 |
| flowstar_style_o6_candidate8_output6_cutoff_insert_symqueue_v2 | failed | 7.496039258138734 | 150 | 101.53672534356728 | 2.5046732353631054 | 101.53672534356728 | 3.3260260122785064 |
