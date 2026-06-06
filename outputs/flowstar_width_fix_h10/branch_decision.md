# Width Fix Branch Decision

Decision: `NEEDS_MORE_WORK`.

Selected fix: Option 3, right-map scaling/scalar alignment. The attribution report identifies `right_map_scaling` as the dominant width source, with max right-map width sum `21.883875748631645`; Flow* source review shows `Flowpipe::normalize` recenters component remainder midpoints before scale extraction.

## Baseline vs Scalar Fix

| path | baseline_t | baseline_final_width_sum | scalar_t | scalar_final_width_sum | h10_reached |
| --- | ---: | ---: | ---: | ---: | --- |
| o4 target insert | 6.4730088058091901 | 4.8223900700103997 | 6.4730088058091901 | 4.8106981513057709 | no |
| o6 candidate8 output6 insert | 7.4960392581387341 | 21.899038480793845 | 7.4960392581387341 | 21.897637854767588 | no |

The scalar fix made widths slightly tighter but did not move the validated horizon. Both scalar runs still fail with `Picard residual not subset of target remainder`.

## Evidence

- Formatting: requested h10/symqueue/oracle/symbolic CSV and markdown artifacts were regenerated or rewritten with physical newlines, and physical-line tests passed.
- Checkpoint replay: `outputs/flowstar_checkpoint_replay/checkpoint_replay_report.md`. Flow* first fails from the o4 PyTorch local box at `o4_insert_t0p75`; o6 remains replayable through t=`5.0` in this checkpoint set.
- Width attribution: `outputs/flowstar_insertion_width_attribution/insertion_width_report.md`. Dominant component is `right_map_scaling`.
- Width fix output: `outputs/flowstar_width_fix_h10/width_fix_summary.csv` and `outputs/flowstar_width_fix_h10/width_fix_report.md`.
- Sample containment: passed for `flowstar_style_o6_candidate8_output6_insert_scalars` with `500` samples, `75000` checked sample/time pairs, and `0` violations.
- Tests: `conda run -n py11 python -m pytest -q` passed, `128 passed in 137.76s`.

## Merge Gate

This branch is not a merge candidate yet because horizon 10 was not reached and the scalar fix only gives a small width reduction without a clear source-mapped horizon improvement. The reports and tests are ready for the next mechanism pass.
