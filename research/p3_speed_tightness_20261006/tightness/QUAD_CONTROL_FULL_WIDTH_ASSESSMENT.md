# QUAD control v2 full-horizon saved assessment

This report only reads the full candidate's saved CSV and receipts. It runs no numerical solver or old checker. The candidate is explicit and is not promoted to the report's primary selection here.

All 1000 steps, 1024 initial boxes and 12 physical states are represented by 24000 endpoint/tube rows: 21990 narrower, 2002 equal, 8 wider. 23956 candidate intervals are subsets of their references; narrower is not automatically containment.

Saved internal driver time: 997.675007039 → 1001.43113359 s (+0.376488%). Candidate process wall time: 1008.89494559 s. The time reference is the October 5 private256 full1000 run, while geometry is compared against the original canonical full50 saved observer. Single timings do not establish a stable ranking.

Negative terminal changes mean narrower. Per-state units are kept separate.

| State | Old endpoint width at 5 s | New endpoint width at 5 s | Endpoint change % | Tube at 5 s change % | Max per-step tube change % |
|---|---:|---:|---:|---:|---:|
| x1 | 6.80567156969 | 6.79516928772 | -0.154316615 | -0.154316598 | -0.154316598 |
| x2 | 6.99572425319 | 6.98431282991 | -0.16311997 | -0.163080713 | -0.163080713 |
| x3 | 0.0672257497672 | 0.0655158191585 | -2.54356495 | -2.54035315 | -0.000180328238 |
| x4 | 1.57026897886 | 1.5623983704 | -0.501226769 | -0.501188488 | -0.501188488 |
| x5 | 1.65978924846 | 1.65138112633 | -0.506577696 | -0.506536108 | -0.506536108 |
| x6 | 0.172038148162 | 0.167485360459 | -2.64638265 | -2.6441565 | -0.00130380523 |
| x7 | 0.0112944083423 | 0.0108868815168 | -3.60821756 | -3.60821468 | -0.25584063 |
| x8 | 0.00955945147068 | 0.00917035899997 | -4.07023846 | -4.07020262 | -0.219605061 |
| x9 | 0.00614354041201 | 0.00605512307869 | -1.43919186 | -1.43919186 | -1.43919186 |
| x10 | 0.153198520587 | 0.146304967275 | -4.49975188 | -4.49971759 | -0.227223771 |
| x11 | 0.128642847474 | 0.122523503141 | -4.7568477 | -4.75683315 | -0.147749173 |
| x12 | 4.45014771701e-308 | 4.45014771701e-308 | 0 | 0 | 0 |

Eight positive differences remain in their exact saved values; no tolerance hides them.

| State/object | Step | Time s | Old width | New width | Absolute increase | Relative increase % |
|---|---:|---:|---:|---:|---:|---:|
| x9 endpoint | 43 | 0.215 | 1.44848452073e-05 | 1.44849562199e-05 | 1.11012606436e-10 | 0.000766405196926 |
| x9 tube | 43 | 0.215 | 1.448512993e-05 | 1.44852408095e-05 | 1.10879587624e-10 | 0.000765471819449 |
| x9 endpoint | 44 | 0.22 | 1.55685004492e-05 | 1.55686047267e-05 | 1.04277488409e-10 | 0.000669797895751 |
| x9 tube | 44 | 0.22 | 1.55687801906e-05 | 1.55688843566e-05 | 1.04166025842e-10 | 0.000669069924338 |
| x9 endpoint | 45 | 0.225 | 1.65960596774e-05 | 1.65961026359e-05 | 4.29584856237e-11 | 0.000258847500303 |
| x9 tube | 45 | 0.225 | 1.65963365453e-05 | 1.65963794128e-05 | 4.28674572391e-11 | 0.000258294697279 |
| x9 endpoint | 46 | 0.23 | 1.75674947583e-05 | 1.75675176248e-05 | 2.28664835005e-11 | 0.000130163599393 |
| x9 tube | 46 | 0.23 | 1.75677712042e-05 | 1.75677939986e-05 | 2.27944080728e-11 | 0.000129751280387 |

All source paths, byte sizes, all 24 terminal rows, exact rational differences, old/new maximum per-step widths, wider step lists, and all non-subset rows are retained in the adjacent JSON. The control receipt retains actual additional CROWN call counts and per-refresh contraction statistics. Component wall times remain unsynchronized host measurements.

Rebuild with `python3 -B research/p3_speed_tightness_20261006/tightness/derive_quad_full_width_assessment.py`.
