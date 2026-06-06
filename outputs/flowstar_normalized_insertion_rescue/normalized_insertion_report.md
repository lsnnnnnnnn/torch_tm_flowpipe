# Flowstar Normalized Insertion Rescue Report

Mechanism: opt-in clean-room normal insertion/composition. The default flowpipe path is unchanged.
Previous best `flowstar_style_o6_candidate8_output6_cutoff` reached t=`2.4007376673997931`.
Normalized insertion `flowstar_style_o6_candidate8_output6_cutoff_insert` reached t=`5`.
Did normalized insertion beat t~=2.400737? yes.
Did it reach horizon 5? yes.
Did reset widths shrink before t~=2.4? yes; previous max reset width sum=`25.543607540190656`, new=`0.7269367782169329`.
Did width ratios vs Flow* improve at 2x/5x/10x crossing times?
- 2x crossing: previous=`0.7125`, normalized_insertion=``; improved; new did not cross.
- 5x crossing: previous=`1.27294921875`, normalized_insertion=``; improved; new did not cross.
- 10x crossing: previous=`1.6675689250230787`, normalized_insertion=``; improved; new did not cross.
Did insertion uncertainty dominate? no; max output remainder width=`0.10194401009325679`, max inserted endpoint width=`0.7238134623624186`.
Runtime cost: previous=`449.62720102537423`, normalized insertion=`374.0728558450937` seconds.
Width ratio vs Flow*: previous last=`68.36823036058439`, tube=`4.835528765199405`; new last=`1.042824337979132`, tube=`0.9999174731419563`.
Failure mode if still failing: ``.
One-step oracle after insertion: not run; normalized insertion reached the requested horizon and produced no PyTorch failure point.
Branch decision: MERGE_CANDIDATE.

## Rows

| run_id | reset_mode | status | last_validated_t | runtime_s | max_inserted_width | max_insertion_truncation | max_insertion_cutoff | failure_reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o6_candidate8_output6_cutoff | normalized_endpoint_box | failed | 2.400737667399793 | 449.62720102537423 |  |  |  | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_cutoff_insert | normalized_insertion | max_horizon_reached | 5.0 | 374.0728558450937 | 0.7238134623624186 | 8.972778387715742e-06 | 4.2233901223208624e-10 |  |
