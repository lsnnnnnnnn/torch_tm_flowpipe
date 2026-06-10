# Flow* Raw Remainder Compat Attempt Sequence

This is a first-adaptive-attempt diagnostic only. It does not run h10, add NNCS/GPU work, add symbolic queue variants, change defaults, or claim Flow* parity.

## Answers

- Does compat reproduce Flow* attempt status sequence? `yes`.
- Flow* sequence: `['rejected', 'rejected', 'rejected', 'accepted']`.
- Current PyTorch sequence: `['rejected', 'rejected', 'accepted', 'accepted']`.
- Compat sequence: `['rejected', 'rejected', 'rejected', 'accepted']`.
- Does compat reject h=0.1, h=0.05, h=0.025 and accept h=0.0125 if Flow* does? `yes`.
- Does current mode diverge at h=0.025 as before? `yes`.
- Is compat residual close to Flow* at all attempted h? `no` using tolerance `1e-06`.
- Is compat over-conservative at h=0.0125 or does it also accept? `accepted`.
- Should we proceed to T=0.5 short horizon? `yes`, with the same opt-in mode and explicit residual-delta reporting.

## Ledger

| attempt | h | mode | status | residual_y_hi | delta_vs_flowstar | overconservative | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.10000000000000001 | probe | rejected | 0.0066623557610384675 | 0 | false | existing Flow* first-attempt probe trace; no h10 rerun |
| 2 | 0.050000000000000003 | probe | rejected | 0.0006881234349748914 | 0 | false | existing Flow* first-attempt probe trace; no h10 rerun |
| 3 | 0.025000000000000001 | probe | rejected | 0.0001083283903691475 | 0 | false | existing Flow* first-attempt probe trace; no h10 rerun |
| 4 | 0.012500000000000001 | probe | accepted | 1.3422604465478584e-05 | 0 | false | existing Flow* first-attempt probe trace; no h10 rerun |
| 1 | 0.10000000000000001 | current_no_queue | rejected | 0.0028787540143153023 | -0.0037836017467231652 | false | raw ctrunc remainder is recorded before adding poly_diff_range; target remainder is only the containment set |
| 1 | 0.10000000000000001 | flowstar_raw_remainder_compat | rejected | 0.0061508720457428036 | -0.00051148371529566392 | false | compat mode checks replayed raw remainder plus poly_diff_range; target remainder is only the containment set |
| 2 | 0.050000000000000003 | current_no_queue | rejected | 0.00031000515306703005 | -0.00037811828190786135 | false | raw ctrunc remainder is recorded before adding poly_diff_range; target remainder is only the containment set |
| 2 | 0.050000000000000003 | flowstar_raw_remainder_compat | rejected | 0.00064365815043707401 | -4.4465284537817396e-05 | false | compat mode checks replayed raw remainder plus poly_diff_range; target remainder is only the containment set |
| 3 | 0.025000000000000001 | current_no_queue | accepted | 6.3769253659495107e-05 | -4.4559136709652392e-05 | false | raw ctrunc remainder is recorded before adding poly_diff_range; target remainder is only the containment set |
| 3 | 0.025000000000000001 | flowstar_raw_remainder_compat | rejected | 0.00010750515280698436 | -8.2323756216313992e-07 | false | compat mode checks replayed raw remainder plus poly_diff_range; target remainder is only the containment set |
| 4 | 0.012500000000000001 | current_no_queue | accepted | 2.0300501608332201e-05 | 6.8778971428536168e-06 | false | raw ctrunc remainder is recorded before adding poly_diff_range; target remainder is only the containment set |
| 4 | 0.012500000000000001 | flowstar_raw_remainder_compat | accepted | 2.7521187465706136e-05 | 1.4098583000227551e-05 | false | compat mode checks replayed raw remainder plus poly_diff_range; target remainder is only the containment set |

## Outputs

- `outputs/flowstar_raw_remainder_compat_attempt_sequence/attempt_sequence_ledger.csv`
- `outputs/flowstar_raw_remainder_compat_attempt_sequence/attempt_sequence_report.md`
