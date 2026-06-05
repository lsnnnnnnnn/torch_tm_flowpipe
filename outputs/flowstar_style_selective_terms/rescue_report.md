# Flowstar-Style Rescue Report

Requested max horizon: `5`.
Best old baseline in this run: `` at t=`0`.
Best flowstar_style run: `flowstar_style_o6_candidate8_output6_keep8` at t=`2.345909199029081`.

Did flowstar_style beat the old best t~=0.7661635? yes.
Did flowstar_style_o6_target reach the requested horizon? no.
Did cutoff help? inconclusive.
Did target remainder stay bounded at width sum 0.0004? yes; max target-mode remainder width sum was `0.0004000000000000001`.
Did recenter/rescale help compared to range_only and dependency_preserving? yes; best flowstar_style t=`2.345909199029081` vs best baseline t=`0`.
Best rescue candidate: `flowstar_style_o6_candidate8_output6_keep8`.
Accepted/rejected steps for best rescue: `100` accepted, `64` rejected.
min_regular_h_used for best rescue: `0.0022906289001034695`.
Did any non-final step go below Flow* min step 0.002? no.
How do widths compare to original Flow* over the same horizon? last width ratio=`69.67672951806126`, tube width ratio=`4.552940781649833`.
Is this a reachability success, a tightness success, or both? neither yet.
Failure mode for the best rescue candidate: `Picard residual not subset of target remainder`.
Do not treat this as Flow* parity unless horizon 10 is reached and boxes are compared separately.

## Summary Rows

| run_id | status | last_validated_t | accepted | rejected | min_h_used | min_regular_h_used | non_final_h_below_0.002 | failure_reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o6_candidate8_output6_keep4 | failed | 2.3236324377489392 | 112 | 71 | 0.002092052078114505 | 0.002092052078114505 | 0 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_keep8 | failed | 2.345909199029081 | 100 | 64 | 0.0022906289001034695 | 0.0022906289001034695 | 0 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_keep4_centered | failed | 2.3398728391401784 | 112 | 71 | 0.002092052078114505 | 0.002092052078114505 | 0 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_keep8_centered | failed | 2.345909199029081 | 100 | 64 | 0.0022906289001034695 | 0.0022906289001034695 | 0 | Picard residual not subset of target remainder |
