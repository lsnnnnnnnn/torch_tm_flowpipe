# Flowstar Raw Remainder Compat h5 Divergence Audit

This is an h5-only audit. It does not run h10, work on NNCS/GPU, add symbolic queue variants, change default solver behavior, or claim Flowstar parity.

## Answers

- First schedule divergence accepted step: `5` at t_before `0.076313750000000014`.
- Flowstar h vs compat h there: `0.020131380000000004` vs `0.020131375000000007`.
- First divergence classification: `t-grid drift`.
- Width ratio crosses 1.1: `t=0.92419030000000002, ratio=1.1035025400929366`.
- Width ratio crosses 1.5: `t=3.078875, ratio=1.5513085996398646`.
- Width ratio crosses 2.0: `t=3.8966409999999998, ratio=2.1358225056343265`.
- Which happens first: `schedule divergence`.
- Late spike or gradual accumulation: `gradual accumulation`.
- Tube-close but last-segment-wide: `yes`; last ratio `2.6079115564373536`, tube ratio `1.0110368009992328`.
- Is h5 failure-to-be-close mainly schedule or width/remainder? `width/remainder`: schedule first differs by a tiny h-grid amount, while same-attempt residual_y_hi delta is `2.7436657477369355e-05` and the last-segment ratio grows above 2.
- Did compat+Flowstar policy stay close until some time then diverge? `yes`; it matches the first `5` accepted h values and width ratio stays below 1.1 until the crossing listed above.
- Does the h5 result justify h10? `no`; recommendation `review_width_before_h10`.
- Change before h10: prioritize raw remainder residual magnitude and width source attribution, then revisit adaptive schedule after the divergence point; normalized insertion interaction remains a comparison target, not a parity claim.

## Same t/h Attempt Around First Schedule Divergence

| source | mode | t_before | h_try | status | residual_y_hi | target_y_hi | raw_ctrunc_residual_y_hi | full_step_tube_y_hi | reset_width_sum | right_map_range_width_sum | post_cutoff_residual_y_hi | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flowstar | flowstar_probe | 0.076313750001000005 | 0.020131375000000007 | accepted | 6.4558501223868549e-05 | 0.0001 | 9.1468145281629337e-05 | 2.3149478815308488 | 0.55884864934103495 | 3.9797785293324068 | 6.4558501223868549e-05 | nearest local Flowstar probe row; h5 reference schedule row itself has no residual fields |
| torch | raw_remainder_compat_flowstar_step_policy | 0.076313750000000014 | 0.020131375000000007 | accepted | 9.1995158701237904e-05 | 0.0001 | 9.1995157701237895e-05 | 2.3150670522408547 | 0.59379572892644672 | 0.59100732857041216 | 5.5295043437891466e-05 | compat mode checks replayed raw remainder plus poly_diff_range; target remainder is only the containment set |

## Tube Ratio Note

The tube ratio remains close because the h5 tube is a prefix union over all segment boxes. The final segment is much wider than Flowstar's final segment, but it is still a small part of the accumulated tube envelope, whose earlier extrema dominate the total tube width.

## Outputs

- `outputs/flowstar_raw_remainder_compat_h5_divergence/h5_divergence_ledger.csv`
- `outputs/flowstar_raw_remainder_compat_h5_divergence/h5_width_growth.csv`
- `outputs/flowstar_raw_remainder_compat_h5_divergence/h5_schedule_divergence.csv`
- `outputs/flowstar_raw_remainder_compat_h5_divergence/h5_divergence_report.md`
