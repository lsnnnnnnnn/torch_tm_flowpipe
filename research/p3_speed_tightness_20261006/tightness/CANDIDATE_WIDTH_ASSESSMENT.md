# Saved candidate width assessment

This is a new arithmetic summary of saved CSVs and receipts. No experiment or numerical checker was run. Width is the exact difference of saved binary64 bounds; a smaller width is not an interval containment certificate. Each percentage is relative to the corresponding saved reference row.

## QUAD anchored control v2: 40-step diagnostic

All 960 saved state/geometry rows: 878 narrower, 82 equal, 0 wider; 942 are subsets of their reference intervals. The 18 non-subsets are x10 endpoint rows, all narrower. All 24 terminal rows are subsets.

Driver 39.288804109 → 39.576246742 s (+0.731615%); candidate process 46.208694426 s. These are single samples. The geometry reference is the original canonical batch2 saved output; the time reference is the saved private256 batch2 driver. Two additional CROWN calls yield four actual NN calls versus two base calls. Of 6144 control component rows, 3629 have a contracted remainder. Component timers are unsynchronized host timings and do not establish device cost attribution.

Terminal time 0.2 s. Negative changes mean narrower.

| State | Old endpoint width | New endpoint width | Endpoint change % | Tube change % |
|---|---:|---:|---:|---:|
| x1 | 0.960596511406 | 0.960593683846 | -0.000294354519 | -0.000294340697 |
| x2 | 0.960707538707 | 0.960706294618 | -0.000129497214 | -0.000129883417 |
| x3 | 0.820842429786 | 0.820838605413 | -0.000465908194 | -0.000458567037 |
| x4 | 0.806485519357 | 0.806468063527 | -0.00216443192 | -0.00215030758 |
| x5 | 0.807285575152 | 0.807258857683 | -0.00330954371 | -0.003267167 |
| x6 | 1.47775973566 | 1.4777512148 | -0.000576606689 | -0.000576880519 |
| x7 | 0.00519308607333 | 0.00518644455583 | -0.127891535 | -0.128173719 |
| x8 | 0.00441150604312 | 0.00441144677638 | -0.00134345814 | -0.00134867175 |
| x9 | 1.14483429291e-05 | 1.144771751e-05 | -0.00546296624 | -0.0054671242 |
| x10 | 0.0810181739686 | 0.0810038948018 | -0.0176246466 | -0.0183326811 |
| x11 | 0.0782411885141 | 0.0782400020946 | -0.00151636179 | -0.0015146188 |
| x12 | 4.45014771701e-308 | 4.45014771701e-308 | 0 | 0 |

The height state x3 gains only 3.82437214391e-06 absolute width, or 0.000465908194%, at this short prefix. The largest terminal relative gain is x7 (about 0.128%). This supports a full numerical trial to measure accumulated behavior, not a claim of a significant height gain or full-horizon dominance. The same-controller floating CROWN enclosure assumption and missing independent NNCS certificate remain. QUAD is not promoted from this prefix.

## Sigmoid order 4 and order 6

Both use cutoff 1e-8 and compare with the saved order-3/cutoff-1e-6 reference. Reported early increases are retained exactly; they are not removed by a tolerance. Maximum absolute and relative increases may occur on different rows, so both row locations are given.

### tora_sigmoid_order4_cutoff1e8_full500_001

4000 rows: 3800 narrower, 4 equal, 196 wider; 3804 subsets. Driver 6.237087961 s; process 10.496905173 s.

| State/object | Wider steps | Max absolute increase (step) | Relative % at that step | Max relative increase % (step) |
|---|---|---:|---:|---:|
| x3 endpoint | 2–50 (49) | 2.62012633812e-14 (50) | 6.69349605004e-11 | 6.69349605004e-11 (50) |
| x3 tube | 2–50 (49) | 2.62012633812e-14 (50) | 4.26861617865e-11 | 4.26861617865e-11 (50) |
| x4 endpoint | 2–50 (49) | 6.57252030578e-14 (50) | 2.61399515033e-10 | 2.61399515033e-10 (50) |
| x4 tube | 2–50 (49) | 6.57252030578e-14 (50) | 8.65659546099e-11 | 8.65659546099e-11 (50) |

All terminal widths are smaller, but neither order candidate meets the all-state/all-step no-width-increase criterion. The JSON also retains all terminal widths, separate old/new maximum per-step widths, the first wider row, exact rational positive differences, and full wider-step lists.

### tora_sigmoid_order6_cutoff1e8_full500_001

4000 rows: 3800 narrower, 4 equal, 196 wider; 3804 subsets. Driver 10.696474694 s; process 15.511160405 s.

| State/object | Wider steps | Max absolute increase (step) | Relative % at that step | Max relative increase % (step) |
|---|---|---:|---:|---:|
| x3 endpoint | 2–50 (49) | 6.93889390391e-14 (50) | 1.77264196241e-10 | 1.77264196241e-10 (50) |
| x3 tube | 2–50 (49) | 6.93889390391e-14 (50) | 1.13045979307e-10 | 1.13045979307e-10 (50) |
| x4 endpoint | 2–50 (49) | 1.53654866608e-13 (50) | 6.11109677037e-10 | 6.11109677037e-10 (50) |
| x4 tube | 2–50 (49) | 1.53654866608e-13 (50) | 2.02377164156e-10 | 2.02377164156e-10 (50) |

All terminal widths are smaller, but neither order candidate meets the all-state/all-step no-width-increase criterion. The JSON also retains all terminal widths, separate old/new maximum per-step widths, the first wider row, exact rational positive differences, and full wider-step lists.

## Rebuild and provenance

Run `python3 -B research/p3_speed_tightness_20261006/tightness/derive_candidate_width_assessment.py` from this repository. The adjacent JSON includes precise source paths and byte sizes. It reads existing comparison receipts and CSVs only; no checker or source experiment is invoked. All source files remain unchanged.
