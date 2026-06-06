# Rescue Vs Original Flow* Comparison

Requested horizon: `10`.
Original Flow* boxes are parsed GNUPLOT segment boxes; this comparison uses overlap hulls, not exact segment-count matching.
This is not a Flow* parity claim unless boxes are numerically identical, which is not expected here.

Best rescue config: `flowstar_style_o6_candidate8_output6_insert_symqueue`.
Reached requested horizon? no.
Width comparable to Flow*? yes.

## Metrics

| run_id | py_status | py_last_validated_t | py_segments | last_width_ratio | tube_width_ratio | max_overlap_ratio | median_overlap_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o4_target_insert_symqueue | failed | 3.217705703618956 | 77 | 0.8268968507714647 | 0.9968074353019628 | 0.8850050740894254 | 0.7007947724086819 |
| flowstar_style_o4_target_cutoff_insert_symqueue | failed | 3.217705703618956 | 77 | 0.8268969746835797 | 0.9968074382638831 | 0.8850050745582376 | 0.7007947781520536 |
| flowstar_style_o6_candidate8_output6_insert_symqueue | failed | 3.3500000000000014 | 38 | 0.7039951932518582 | 0.9951883436176395 | 0.7083000578757825 | 0.49296531702775104 |
| flowstar_style_o6_candidate8_output6_cutoff_insert_symqueue | failed | 3.3500000000000014 | 38 | 0.7039954503434184 | 0.9951883500104782 | 0.7083000586987275 | 0.49296532765482926 |
