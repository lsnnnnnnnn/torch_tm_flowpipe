# Rescue Vs Original Flow* Comparison

Requested horizon: `10`.
Original Flow* boxes are parsed GNUPLOT segment boxes; this comparison uses overlap hulls, not exact segment-count matching.
This is not a Flow* parity claim unless boxes are numerically identical, which is not expected here.

Best rescue config: `flowstar_style_o6_candidate8_output6_insert_symqueue_split`.
Reached requested horizon? no.
Width comparable to Flow*? no.

## Metrics

| run_id | py_status | py_last_validated_t | py_segments | last_width_ratio | tube_width_ratio | max_overlap_ratio | median_overlap_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o4_target_insert_symqueue_split | failed | 6.47300880580919 | 239 | 14.931558420454357 | 1.1808892942586184 | 14.931558420454357 | 1.3364376132047158 |
| flowstar_style_o4_target_cutoff_insert_symqueue_split | failed | 6.47300880580919 | 239 | 14.931570259420756 | 1.1808894434412616 | 14.931570259420756 | 1.3364380189904703 |
| flowstar_style_o6_candidate8_output6_insert_symqueue_split | failed | 7.496039258138734 | 150 | 124.46569353503585 | 2.9696033909707853 | 124.46569353503585 | 3.168484549235259 |
| flowstar_style_o6_candidate8_output6_cutoff_insert_symqueue_split | failed | 7.496039258138734 | 150 | 124.47028765099832 | 2.969700649019852 | 124.47028765099832 | 3.1684963119871057 |
