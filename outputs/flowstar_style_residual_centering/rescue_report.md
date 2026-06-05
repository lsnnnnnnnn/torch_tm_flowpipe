# Flowstar-Style Rescue Report

Requested max horizon: `5`.
Best old baseline in this run: `` at t=`0`.
Best flowstar_style run: `flowstar_style_o6_candidate8_output6_centered` at t=`2.4007376673997931`.

Did flowstar_style beat the old best t~=0.7661635? yes.
Did flowstar_style_o6_target reach the requested horizon? no.
Did cutoff help? tied.
Did target remainder stay bounded at width sum 0.0004? yes; max target-mode remainder width sum was `0.0004000000000000001`.
Did recenter/rescale help compared to range_only and dependency_preserving? yes; best flowstar_style t=`2.4007376673997931` vs best baseline t=`0`.
Best rescue candidate: `flowstar_style_o6_candidate8_output6_centered`.
Accepted/rejected steps for best rescue: `95` accepted, `60` rejected.
min_regular_h_used for best rescue: `0.0021450425182496136`.
Did any non-final step go below Flow* min step 0.002? no.
How do widths compare to original Flow* over the same horizon? last width ratio=`68.36823036058439`, tube width ratio=`4.835528765199405`.
Is this a reachability success, a tightness success, or both? neither yet.
Failure mode for the best rescue candidate: `Picard residual not subset of target remainder`.
Do not treat this as Flow* parity unless horizon 10 is reached and boxes are compared separately.

## Summary Rows

| run_id | status | last_validated_t | accepted | rejected | min_h_used | min_regular_h_used | non_final_h_below_0.002 | failure_reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o6_target_centered | failed | 2.1159710308044115 | 127 | 80 | 0.0022645215665453703 | 0.0022645215665453703 | 0 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_centered | failed | 2.400737667399793 | 95 | 60 | 0.0021450425182496136 | 0.0021450425182496136 | 0 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_cutoff_centered | failed | 2.400737667399793 | 95 | 60 | 0.0021450425182496136 | 0.0021450425182496136 | 0 | Picard residual not subset of target remainder |
