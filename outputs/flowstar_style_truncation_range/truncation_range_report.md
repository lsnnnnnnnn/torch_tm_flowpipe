# Truncation Range Diagnostic Report

Dropped/truncated polynomial terms are still bounded conservatively; this diagnostic only changes how their interval range is evaluated.
Requested horizon: `5`.
Best truncation-range variant: `flowstar_style_o6_candidate8_output6_truncsplit2` at t=`2.397165587736743`.
Does tighter dropped-term range bounding beat t~=2.277? yes.
Does it reach horizon 5? no.
Runtime cost for best variant: runtime_s=`1093.3742557130754`.
Width ratio vs Flow*: last=`65.41439543796189`, tube=`4.6266107683365485`.
Did cutoff help when combined with truncsplit? not evaluated by the requested truncation-range config set.

## Rows

| run_id | status | split | candidate_order | output_order | last_validated_t | runtime_s | last_width_ratio | tube_width_ratio | failure_reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o6_target_truncsplit2 | failed | 2 | 6 | 6 | 2.123634202721264 | 807.8879254385829 | 30.885943327720746 | 2.0029842681327326 | Picard residual not subset of target remainder |
| flowstar_style_o6_target_truncsplit4 | failed | 4 | 6 | 6 | 2.123634202721264 | 1779.984035052359 | 30.885943327720746 | 2.0029842681327326 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_truncsplit2 | failed | 2 | 8 | 6 | 2.397165587736743 | 1093.3742557130754 | 65.41439543796189 | 4.6266107683365485 | Picard residual not subset of target remainder |
