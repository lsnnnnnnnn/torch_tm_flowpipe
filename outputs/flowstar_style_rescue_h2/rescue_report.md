# Flowstar-Style Rescue Report

Requested max horizon: `2` attempted across runs.
Best old baseline in this run: `baseline_range_only_o6_s4` at t=`0.67500000000000027`.
Best flowstar_style run: `flowstar_style_o6_target` at t=`2`.

Did flowstar_style beat the old best t~=0.7661635? yes.
Did target remainder validation prevent huge remainder blowup? yes; target-mode max remainder width sum stayed at `0.0004000000000000001` and failed rows rejected residuals instead of inflating.
Did recenter/rescale help compared to range_only and dependency_preserving? yes; best flowstar_style t=`2` vs best baseline t=`0.67500000000000027`.
Did cutoff help or hurt? tied.
Best rescue candidate: `flowstar_style_o6_target`.
Failure mode for the best rescue candidate: `` with min_h_used=`0.0034958879232613871`.
Do not treat this as Flow* parity unless horizon 10 is reached and boxes are compared separately.

## Summary Rows

| run_id | status | last_validated_t | min_h_used | failure_reason |
| --- | --- | ---: | ---: | --- |
| baseline_range_only_o6_s4 | failed | 0.67500000000000027 | 0.025000000000000001 | Picard remainder validation did not converge |
| baseline_dependency_preserving_o4_s1 | failed | 0.20000000000000001 | 0.10000000000000001 | non-finite residual interval |
| flowstar_style_o4_target | failed | 1.6364220066375745 | 0.0020213320743365801 | Picard residual not subset of target remainder |
| flowstar_style_o6_target | max_horizon_reached | 2 | 0.0034958879232613871 |  |
| flowstar_style_o4_target_cutoff | failed | 1.6364220066375745 | 0.0020213320743365801 | Picard residual not subset of target remainder |
| flowstar_style_o6_target_cutoff | max_horizon_reached | 2 | 0.0034958879232613871 |  |
