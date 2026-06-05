# Rescue Variant Comparison Next3

Best variant by decision criteria: `flowstar_style_o6_candidate8_output6_cutoff` at t=`2.4007376673997931`.
Reached horizon 5 with target_remainder_radius=1e-4? no.
Width ratio vs Flow*: last=`68.368230360584391`, tube=`4.8355287651994052`.
Width criterion vs candidate_order baseline acceptable? yes.
Target remainder stayed at 1e-4? yes.
Next recommendation: choose between residual-centering refinement, selective sparse over-order terms, or a real Flow*-style symbolic remainder queue.

This comparison is diagnostic-only and does not claim Flow* parity.
Decision criteria: reaches horizon 5, no non-final h below 0.002, target remainder 1e-4, width ratio not worse than candidate_order baseline unless horizon improves substantially, and acceptable runtime.

## Rows

| group | run_id | status | last_validated_t | candidate_order | output_order | K | corrections | last_width_ratio | tube_width_ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adaptive_order_fallback | flowstar_style_o6_target_adaptive_order_8 | failed | 2.2771582567640953 | 6 | 6 |  |  | 80.930899048992146 | 4.9151377009180992 |
| adaptive_order_fallback | flowstar_style_o6_target_cutoff_adaptive_order_8 | failed | 2.2771582567640953 | 6 | 6 |  |  | 80.930899048992146 | 4.9151377009180992 |
| candidate_order_output_order | flowstar_style_o6_candidate8_output6 | failed | 2.4007376673997931 | 8 | 6 |  |  | 68.368230360584846 | 4.8355287651994372 |
| candidate_order_output_order | flowstar_style_o6_candidate8_output6_cutoff | failed | 2.4007376673997931 | 8 | 6 |  |  | 68.368230360584391 | 4.8355287651994052 |
| candidate_order_truncation_split | flowstar_style_o6_candidate8_output6_truncsplit2 | failed | 2.397165587736743 | 8 | 6 |  |  | 65.414395437961886 | 4.6266107683365485 |
| h5_current_best | flowstar_style_o6_target | failed | 2.1095541733932355 | 6 | 6 |  |  | 30.050027407228164 | 1.9627712588202828 |
| h5_current_best | flowstar_style_o6_target_cutoff | failed | 2.1095541733932355 | 6 | 6 |  |  | 30.050027407228164 | 1.9627712588202828 |
| residual_centering | flowstar_style_o6_candidate8_output6_centered | failed | 2.4007376673997931 | 8 | 6 |  | 3 | 68.368230360584846 | 4.8355287651994372 |
| residual_centering | flowstar_style_o6_candidate8_output6_cutoff_centered | failed | 2.4007376673997931 | 8 | 6 |  | 3 | 68.368230360584391 | 4.8355287651994052 |
| residual_centering | flowstar_style_o6_target_centered | failed | 2.1159710308044115 | 6 | 6 |  | 6 | 30.850786594767612 | 2.0007043188924944 |
| selective_high_degree_terms | flowstar_style_o6_candidate8_output6_keep1 | failed | 2.4007376673997931 | 8 | 6 | 1 | 0 | 68.368230360584846 | 4.8355287651994372 |
| selective_high_degree_terms | flowstar_style_o6_candidate8_output6_keep2 | failed | 2.4007376673997931 | 8 | 6 | 2 | 0 | 68.368230360584846 | 4.8355287651994372 |
| selective_high_degree_terms | flowstar_style_o6_candidate8_output6_keep4 | failed | 2.4007376673997931 | 8 | 6 | 4 | 0 | 68.368230360584846 | 4.8355287651994372 |
| selective_high_degree_terms | flowstar_style_o6_candidate8_output6_keep8 | failed | 2.4007376673997931 | 8 | 6 | 8 | 0 | 68.368230360584846 | 4.8355287651994372 |
| selective_terms_centered | flowstar_style_o6_candidate8_output6_keep1_centered | failed | 2.4007376673997931 | 8 | 6 | 1 | 3 | 68.368230360584846 | 4.8355287651994372 |
| selective_terms_centered | flowstar_style_o6_candidate8_output6_keep2_centered | failed | 2.4007376673997931 | 8 | 6 | 2 | 3 | 68.368230360584846 | 4.8355287651994372 |
| selective_terms_centered | flowstar_style_o6_candidate8_output6_keep4_centered | failed | 2.4007376673997931 | 8 | 6 | 4 | 3 | 68.368230360584846 | 4.8355287651994372 |
| selective_terms_centered | flowstar_style_o6_candidate8_output6_keep8_centered | failed | 2.4007376673997931 | 8 | 6 | 8 | 3 | 68.368230360584846 | 4.8355287651994372 |
| truncation_range_split | flowstar_style_o6_target_truncsplit2 | failed | 2.1236342027212638 | 6 | 6 |  |  | 30.885943327720746 | 2.0029842681327326 |
| truncation_range_split | flowstar_style_o6_target_truncsplit4 | failed | 2.1236342027212638 | 6 | 6 |  |  | 30.885943327720746 | 2.0029842681327326 |
