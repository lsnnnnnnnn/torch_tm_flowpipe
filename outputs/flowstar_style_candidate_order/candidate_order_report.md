# Candidate Order Diagnostic Report

Candidate-order mode validates with a higher Picard polynomial order and truncates the accepted/output Taylor model back to output order with the dropped contribution added to interval uncertainty.
Requested horizon: `5`.
Best candidate-order variant: `flowstar_style_o6_candidate8_output6` at t=`2.400737667399793`.
Did candidate_order=8/output_order=6 beat t~=2.277? yes.
Did it reach horizon 5? no.
Width ratio vs adaptive full-order-8 fallback: improved; candidate last=`68.36823036058485`, tube=`4.835528765199437`, adaptive tube=`4.9151377009180992`.
Runtime impact: best runtime_s=`485.9197028847411` vs adaptive fallback runtime_s=`472.12907700892538`.
Does it reduce truncation containment miss? yes by max residual width sum; candidate max_residual_width_sum=`0.0009281033590735908`, adaptive=`0.0094877522677920233`.

## Rows

| run_id | status | candidate_order | output_order | last_validated_t | runtime_s | last_width_ratio | tube_width_ratio | failure_reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o6_candidate8_output6 | failed | 8 | 6 | 2.400737667399793 | 485.9197028847411 | 68.36823036058485 | 4.835528765199437 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_cutoff | failed | 8 | 6 | 2.400737667399793 | 465.3971140887588 | 68.36823036058439 | 4.835528765199405 | Picard residual not subset of target remainder |
