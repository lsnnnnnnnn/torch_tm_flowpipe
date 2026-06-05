# Residual Centering Diagnostic Report

This opt-in mode keeps the symmetric target remainder and accepts only after recomputing the Picard residual from the corrected candidate.
Requested horizon: `5`.
Best centered variant: `flowstar_style_o6_candidate8_output6_centered` at t=`2.400737667399793`.
Did centered validation beat t~=2.400737? no.
Did it reach horizon 5? no.
Center-correction attempts: `12` attempts, `12` corrected dimensions; subset-after-correction rows=`5`.
Did corrections stay small? max_abs_correction=`5.234819789315124e-06`.
Width ratio vs Flow*: last=`68.368230360584846`, tube=`4.8355287651994372`.
Did target remainder remain at 1e-4? yes.
Any non-final h below 0.002? no.

## Rows

| run_id | status | last_validated_t | corrections | corrected_dims | max_abs_correction | last_width_ratio | tube_width_ratio | failure_reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o6_target_centered | failed | 2.1159710308044115 | 6 | 6 | 4.9170930624816038e-06 | 30.850786594767612 | 2.0007043188924944 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_centered | failed | 2.4007376673997931 | 3 | 3 | 5.2348197893151242e-06 | 68.368230360584846 | 4.8355287651994372 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_cutoff_centered | failed | 2.4007376673997931 | 3 | 3 | 5.2348197893150157e-06 | 68.368230360584391 | 4.8355287651994052 | Picard residual not subset of target remainder |
