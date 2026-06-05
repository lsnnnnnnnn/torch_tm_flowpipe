# Selective High-Degree Term Diagnostic Report

This is diagnostic-only: sparse over-order terms are retained beyond output_order=6, so this is not fixed-order Flow* parity.
Requested horizon: `5`.
Best selective variant: `flowstar_style_o6_candidate8_output6_keep8` at t=`2.345909199029081`.
Did selective retention beat t~=2.400737? no.
Did any variant reach horizon 5? no.
Which K worked best? all tested K values tied on validated time; K=`8` minimized dropped remainder width.
Did keeping a few terms reduce residual shift? max recorded residual center magnitude=`2.5730789873997168e-05` (compare by row in attempts CSV).
Runtime impact: best runtime_s=`438.3293621139601`.
Width ratio vs Flow*: last=`69.67672951806126`, tube=`4.552940781649833`.
Did this outperform full adaptive order fallback? yes; adaptive tube=`4.9151377009180992`.

## Rows

| run_id | K | status | last_validated_t | retained_terms | dropped_remainder_width | runtime_s | last_width_ratio | tube_width_ratio | failure_reason |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o6_candidate8_output6_keep4 | 4 | failed | 2.3236324377489392 | 8.0 | 8.900439092611666e-05 | 469.88078651670367 | 67.07157930147092 | 4.382710422884737 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_keep8 | 8 | failed | 2.345909199029081 | 16.0 | 6.0162079299997325e-05 | 438.3293621139601 | 69.67672951806126 | 4.552940781649833 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_keep4_centered | 4 | failed | 2.3398728391401784 | 8.0 | 9.412110223284505e-05 | 454.28756806161255 | 71.4738121596852 | 4.6703689517668545 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_keep8_centered | 8 | failed | 2.345909199029081 | 16.0 | 6.0162079299997325e-05 | 419.9502213820815 | 69.67672951806126 | 4.552940781649833 | Picard residual not subset of target remainder |
