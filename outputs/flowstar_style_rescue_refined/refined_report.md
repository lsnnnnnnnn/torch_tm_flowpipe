# Refined Target Validation Report

Best refined variant: `flowstar_style_o6_target_refined` at t=`2.0436221052452312`.
Did refined validation beat t~=2.10955? no.
Did it reach horizon 5? no.
Runtime impact: best runtime_s=`621.6946182763204`.
Residual-over-target ratios are recorded in `rescue_validation_attempts.csv` via the target and residual width fields.

## Rows

| run_id | status | last_validated_t | runtime_s | failure_reason |
| --- | --- | ---: | ---: | --- |
| flowstar_style_o6_target_refined | failed | 2.0436221052452312 | 621.6946182763204 | Picard residual not subset of target remainder |
| flowstar_style_o6_target_refined_cutoff | failed | 2.0436221052452312 | 627.2266834191978 | Picard residual not subset of target remainder |
