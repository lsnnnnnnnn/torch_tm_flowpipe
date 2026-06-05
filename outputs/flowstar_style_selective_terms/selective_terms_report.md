# Selective High-Degree Term Diagnostic Report

This is diagnostic-only: sparse over-order terms are retained beyond output_order=6, so this is not fixed-order Flow* parity.
Requested horizon: `5`.
Best selective variant: `flowstar_style_o6_candidate8_output6_keep8` at t=`2.400737667399793`.
Did selective retention beat t~=2.400737? no.
Did any variant reach horizon 5? no.
Which K worked best? all tested K values tied on validated time; K=`8` minimized dropped remainder width.
Did keeping a few terms reduce residual shift? max recorded residual center magnitude=`3.363648124229425e-05` (compare by row in attempts CSV).
Runtime impact: best runtime_s=`459.55931204557419`.
Width ratio vs Flow*: last=`68.368230360584846`, tube=`4.8355287651994372`.
Did this outperform full adaptive order fallback? yes; adaptive tube=`4.9151377009180992`.

## Rows

| run_id | K | status | last_validated_t | retained_terms | dropped_remainder_width | runtime_s | last_width_ratio | tube_width_ratio | failure_reason |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o6_candidate8_output6_keep1 | 1 | failed | 2.4007376673997931 | 2 | 0.00054686467700602393 | 459.95023195724934 | 68.368230360584846 | 4.8355287651994372 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_keep2 | 2 | failed | 2.4007376673997931 | 4 | 0.00042168507927339713 | 460.31188289355487 | 68.368230360584846 | 4.8355287651994372 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_keep4 | 4 | failed | 2.4007376673997931 | 8 | 0.00027326659566923966 | 460.16786826495081 | 68.368230360584846 | 4.8355287651994372 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_keep8 | 8 | failed | 2.4007376673997931 | 16 | 8.3546359146267375e-05 | 459.55931204557419 | 68.368230360584846 | 4.8355287651994372 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_keep1_centered | 1 | failed | 2.4007376673997931 | 2 | 0.00054686467700602393 | 435.10251819994301 | 68.368230360584846 | 4.8355287651994372 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_keep2_centered | 2 | failed | 2.4007376673997931 | 4 | 0.00042168507927339713 | 433.97707378212363 | 68.368230360584846 | 4.8355287651994372 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_keep4_centered | 4 | failed | 2.4007376673997931 | 8 | 0.00027326659566923966 | 434.86731733568013 | 68.368230360584846 | 4.8355287651994372 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_keep8_centered | 8 | failed | 2.4007376673997931 | 16 | 8.3546359146267375e-05 | 435.09607905521989 | 68.368230360584846 | 4.8355287651994372 | Picard residual not subset of target remainder |
