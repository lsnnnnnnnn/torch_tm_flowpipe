# Flowstar Checkpoint Replay Report

At which checkpoint does Flow* first fail from the PyTorch local box? `o4_insert_t0p75`.
At which checkpoint does PyTorch first become much wider than Flow*? `none observed`.
Does o4 stay Flow*-replayable longer than o6? no; o4 last replayable checkpoint=`1.3`, o6=`5.0`.
Is h10 failure caused by boxes already too wide, or by a local kernel mismatch? boxes already too wide/local step too hard.
Which earlier time should be targeted for width reduction? `o4_insert_t0p75` near t=`0.75`.

## Summary

| checkpoint | path | Flow* mini | PyTorch mini | ratio | reset_width | failure |
| --- | --- | --- | --- | ---: | ---: | --- |
| o4_insert_t0p75 | o4_insert | not_completed | completed |  | 0.35258634125485977 | Flow* reach did not complete |
| o4_insert_t1p3 | o4_insert | completed | completed | 0.7145858388012741 | 0.1483994355879308 |  |
| o4_insert_t1p68 | o4_insert | not_completed | completed |  | 0.12891900426842487 | Flow* reach did not complete |
| o4_insert_t2p12 | o4_insert | not_completed | completed |  | 0.15249897220318512 | Flow* reach did not complete |
| o4_insert_t3 | o4_insert | not_completed | completed | 0.8531044427048027 | 0.38493695702257247 | Flow* reach did not complete |
| o4_insert_t5 | o4_insert | not_completed | completed |  | 0.38509374320270867 | Flow* reach did not complete |
| o4_insert_t6p4 | o4_insert | not_completed | failed | 1.2286340833158063 | 3.7108948602024823 | Picard residual not subset of target remainder |
| o6_insert_t0p75 | o6_insert | not_completed | completed |  | 0.4329813863148694 | Flow* reach did not complete |
| o6_insert_t1p3 | o6_insert | completed | completed | 0.5492508533907776 | 0.14792513892588302 |  |
| o6_insert_t1p68 | o6_insert | completed | completed | 0.5034422291985838 | 0.1260557640353314 |  |
| o6_insert_t2p12 | o6_insert | completed | completed | 0.4677684705426801 | 0.15582351288054544 |  |
| o6_insert_t3 | o6_insert | not_completed | completed |  | 0.40168350128184516 | Flow* reach did not complete |
| o6_insert_t5 | o6_insert | completed | completed | 0.602900427633848 | 0.18997028390508144 |  |
| o6_insert_t6p4 | o6_insert | not_completed | completed |  | 1.1365587320162707 | Flow* reach did not complete |
| o6_insert_t7p4 | o6_insert | not_completed | completed |  | 11.808521197637742 | Flow* reach did not complete |
