# Flow* Raw Remainder Compatibility Experiment

This is an experimental one-step compatibility check. It does not change default solver behavior, rerun h10, add NNCS/GPU work, add symbolic queue variants, or claim Flow* parity.

## Case

- t_before: `0.0`
- h_try: `0.025`
- Initial box: `x=[1.1,1.4]`, `y=[2.35,2.45]`
- Target remainder: `[-0.0001,0.0001]`
- Order: `4`
- PyTorch rows use the algebraic VDP RHS spelling from the Flow* probe: `y - x - x^2*y`.

## Answers

- Does compat mode reject h=0.025 like Flow*? `yes`.
- Does compat mode reproduce Flow* residual_y_hi within tolerance 1e-06? `yes`.
- If it rejects but residual still differs, where? residual_y_hi delta compat-Flow* is `-8.2323756216313992e-07`; remaining difference is in the replayed multiplication/truncation range accumulation, not target remainder or cutoff/polyDiff.
- Did default mode remain unchanged? `yes`; current no_queue status is `accepted` and current v2 status is `accepted`.
- Is compat mode over-conservative? `no evidence from this one-step check`; it is slightly below the Flow* residual_y_hi while still above the target and matching reject/accept behavior.
- Should we try short horizon next? `yes`, after keeping this mode opt-in and carrying the residual delta as an explicit diagnostic.

## Ledger

| source | mode | status | residual_y_hi | target_y_hi | subset_y | matches status | matches residual tol | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flowstar | probe | rejected | 0.0001083283903691475 | 0.0001 | false | true | true | existing Flow* one-step probe trace; no h10 rerun |
| torch | current_no_queue | accepted | 6.3769253659495107e-05 | 0.0001 | true | false | false | raw ctrunc remainder is recorded before adding poly_diff_range; target remainder is only the containment set |
| torch | current_v2 | accepted | 6.3769253659495107e-05 | 0.0001 | true | false | false | raw ctrunc remainder is recorded before adding poly_diff_range; target remainder is only the containment set |
| torch | flowstar_raw_remainder_compat | rejected | 0.00010750515280698436 | 0.0001 | false | true | true | compat mode checks replayed raw remainder plus poly_diff_range; target remainder is only the containment set |

## Outputs

- `outputs/flowstar_raw_remainder_compat/raw_remainder_compat_ledger.csv`
- `outputs/flowstar_raw_remainder_compat/raw_remainder_compat_report.md`
