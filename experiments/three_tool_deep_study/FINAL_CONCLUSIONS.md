# Final conclusions

## Authority

These conclusions are generated from the accepted `20260730T015245Z` run on
`codex/torch-flowstar-diffreach-deep-study`. Acceptance and the recursive
artifact-quality audit both passed. The complete isolated pytest matrix
reported 354
passed, 5
skipped, and 0
failed tests for the frozen numerical producer. After the report-only
tightened-endpoint separation checkpoint, the complete artifact-bound matrix
in the final sandbox reported 350 passed, 10 skipped, and 0 failed. The five
additional skips are host-CUDA/external-interface availability differences;
no numerical CSV changed.

## Revoked conclusions

The earlier “same-order winner” and any Torch-tightened-versus-other-raw
ranking are revoked. Equal order labels do not denote equal polynomial
dictionaries, validators, reset contracts, or arithmetic. A failed prefix
cannot be ranked at another solver's requested final time, common-box carry is
a reset/control protocol rather than a native-solver ranking, and sampling is
not a proof.

## Flow* correctness findings

The Riccati stock miss was caused by a variable-leaf truncation interval that
the full evaluator produced but the cached remainder-only replay omitted. The
record/replay correction and an independent full-Picard revalidation both
restore analytic containment: the primary audit contains
64 rows with
0 analytic violations and
0 endpoint/tube
violations. The stock miss remains in the evidence as a regression target.

The adaptive Van der Pol miss belongs to the collapsed endpoint
restriction/evaluation path, not to the ODE/reference mapping, the
variable-leaf patch, adaptive full-Picard acceptance, or the verified native
flowpipe. The first discrepancy is segment
3, state
0, at absolute time
0.04137500000000001; the collapsed lower endpoint
1.195701727252073 missed the DOP853 sanity sample
1.1957008958185056 by
8.314335673276219e-07. The native composed
flowpipe evaluated on fixed local time enclosed all tested samples. The
exporter therefore uses the hull of the collapsed and fixed-domain native
evaluations and places the hull delta in the independent remainder. The
repaired authoritative path has zero deterministic trajectory misses and is
not excluded. The original upstream/generated schedule parity remains
True with
290 segments to T=10.

## Comparability and trusted numerical results

- The 334 one-step rows use matched ODEs, initial boxes, state
  order, steps, and raw endpoint/tube semantics. They support local enclosure
  observations, but not a cross-tool winner because native bases and
  validators remain different.
- The 73 common-affine and 64 common-box rows control
  the propagated representation. Only rows at the same requested absolute
  horizon are juxtaposed. Box/affine ratios measure the effect of this reset
  control; recentering can make a ratio below one, so it is not “negative
  dependency loss.”
- The 32 matched-basis rows compare B1, B_DR, B2, and B3 inside one
  arithmetic engine with one validator/reset contract. They isolate retained
  monomial families; they do not pretend all three native tools implement
  those dictionaries.
- Native-practical width/runtime dominance is valid only within one tool,
  system, and absolute time. The authoritative nondominated rows are:

| tool | variant | system | h | time | width | successful horizon | steady s | memory KiB | basis | carry/preconditioning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torch_tm_flowpipe | order1_range_only | riccati | 0.01 | 1.0 | 0.11406560545058263 | 1.0 | 1.3080267920158803 | unavailable | complete_total_degree_1 | range_only |
| torch_tm_flowpipe | order4_qr_reset | riccati | 0.01 | 1.0 | 0.1139617329824736 | 1.0 | 12.759399096481502 | unavailable | complete_total_degree_4 | qr_reset |
| torch_tm_flowpipe | order6_affine_reset | riccati | 0.01 | 1.0 | 0.11396089557805959 | 1.0 | 29.561500635463744 | unavailable | complete_total_degree_6 | affine_reset |
| torch_tm_flowpipe | order1_raw_dependency | harmonic | 0.01 | 4.0 | 2.9605024010869757e+38 | 4.0 | 10.067116568330675 | unavailable | complete_total_degree_1 | dependency_raw |
| torch_tm_flowpipe | order1_range_only | harmonic | 0.01 | 4.0 | 28.776816743333047 | 4.0 | 10.175232294481248 | unavailable | complete_total_degree_1 | range_only |
| torch_tm_flowpipe | order4_qr_reset | harmonic | 0.01 | 4.0 | 0.28208927442984455 | 4.0 | 44.16646066773683 | unavailable | complete_total_degree_4 | qr_reset |
| torch_tm_flowpipe | order6_affine_reset | harmonic | 0.01 | 4.0 | 10.494235592238002 | 4.0 | 43.268413042183965 | unavailable | complete_total_degree_6 | affine_reset |
| torch_tm_flowpipe | order1_raw_dependency | coupled_quadratic | 0.01 | 0.25 | 3.794875773050277 | 0.25 | 0.9595310497097671 | unavailable | complete_total_degree_1 | dependency_raw |
| torch_tm_flowpipe | order1_range_only | coupled_quadratic | 0.01 | 0.25 | 0.11521134255829171 | 0.25 | 0.9645263440907001 | unavailable | complete_total_degree_1 | range_only |
| torch_tm_flowpipe | order4_affine_reset | coupled_quadratic | 0.01 | 0.25 | 0.04296626780213819 | 0.25 | 14.348397190216929 | unavailable | complete_total_degree_4 | affine_reset |
| torch_tm_flowpipe | order6_affine_reset | coupled_quadratic | 0.01 | 0.25 | 0.042966255245321994 | 0.25 | 43.10928165912628 | unavailable | complete_total_degree_6 | affine_reset |
| torch_tm_flowpipe | order2_affine_reset | van_der_pol | 0.005 | 1.0 | 1.6187956685895426 | 1.0 | 36.16243710601702 | unavailable | complete_total_degree_2 | affine_reset |
| torch_tm_flowpipe | order2_affine_reset | van_der_pol | 0.01 | 1.0 | 1.9405468677370796 | 1.0 | 18.430426479317248 | unavailable | complete_total_degree_2 | affine_reset |
| torch_tm_flowpipe | order4_affine_reset | van_der_pol | 0.01 | 1.0 | 0.911288076111195 | 1.0 | 73.90071617858484 | unavailable | complete_total_degree_4 | affine_reset |
| torch_tm_flowpipe | order4_qr_reset | van_der_pol | 0.01 | 1.0 | 0.18655965212103298 | 1.0 | 85.00013877591118 | unavailable | complete_total_degree_4 | qr_reset |
| diffreach | quasi_window1_round1 | harmonic | 0.01 | 4.0 | 11.02398764119648 | 4.0 | 0.009970881044864655 | unavailable | {1,tau,xi,tau^2,tau*xi} |  |
| diffreach | quasi_window10_round3 | harmonic | 0.01 | 4.0 | 0.34742983960451834 | 4.0 | 0.013027286157011986 | unavailable | {1,tau,xi,tau^2,tau*xi} |  |
| diffreach | quasi_window10_round3 | coupled_quadratic | 0.005 | 0.25 | 0.04297791041868927 | 0.25 | 0.0033192974515259266 | unavailable | {1,tau,xi,tau^2,tau*xi} |  |
| diffreach | affine_flag | coupled_quadratic | 0.01 | 0.25 | 0.054231722544384464 | 0.25 | 0.0015987358056008816 | unavailable | {1,tau,xi} |  |
| diffreach | quasi_window1_round1 | coupled_quadratic | 0.01 | 0.25 | 0.0959394843105721 | 0.25 | 0.0011850344017148018 | unavailable | {1,tau,xi,tau^2,tau*xi} |  |
| diffreach | restricted_quasiquadratic | van_der_pol | 0.005 | 1.0 | 0.14064663658680115 | 1.0 | 0.011776551138609648 | unavailable | {1,tau,xi,tau^2,tau*xi} |  |
| diffreach | quasi_window10_round3 | van_der_pol | 0.01 | 1.0 | 0.17484872384984157 | 1.0 | 0.004365609027445316 | unavailable | {1,tau,xi,tau^2,tau*xi} |  |
| flowstar | root_cause_fixed_order_2 | riccati | 0.01 | 1.0000000000000007 | 0.11390367060918023 | 1.0 | 0.018349999096244574 | unavailable | complete_total_degree_2 | native_normalized_composition |
| flowstar | root_cause_fixed_order_2 | harmonic | 0.01 | 3.9999999999999587 | 0.4263983814553858 | 4.0 | 0.09094226500019431 | unavailable | complete_total_degree_2 | native_normalized_composition |
| flowstar | root_cause_fixed_order_6 | harmonic | 0.01 | 3.9999999999999587 | 0.2820892232454493 | 4.0 | 0.24367694184184074 | unavailable | complete_total_degree_6 | native_normalized_composition |
| flowstar | root_cause_fixed_order_2 | coupled_quadratic | 0.005 | 0.2500000000000001 | 0.04299183206719949 | 0.25 | 0.022881173994392157 | unavailable | complete_total_degree_2 | native_normalized_composition |
| flowstar | root_cause_fixed_order_3 | coupled_quadratic | 0.005 | 0.2500000000000001 | 0.04296903585388806 | 0.25 | 0.09450633777305484 | unavailable | complete_total_degree_3 | native_normalized_composition |
| flowstar | root_cause_fixed_order_6 | coupled_quadratic | 0.005 | 0.2500000000000001 | 0.0429676782190236 | 0.25 | 1.1399605479091406 | unavailable | complete_total_degree_6 | native_normalized_composition |
| flowstar | root_cause_fixed_order_2 | coupled_quadratic | 0.01 | 0.25000000000000006 | 0.04301766242184424 | 0.25 | 0.013268215116113424 | unavailable | complete_total_degree_2 | native_normalized_composition |
| flowstar | root_cause_fixed_order_3 | coupled_quadratic | 0.01 | 0.25000000000000006 | 0.04297049618274264 | 0.25 | 0.04785529198125005 | unavailable | complete_total_degree_3 | native_normalized_composition |
| flowstar | root_cause_fixed_order_6 | coupled_quadratic | 0.01 | 0.25000000000000006 | 0.042967678236914345 | 0.25 | 0.5799654829315841 | unavailable | complete_total_degree_6 | native_normalized_composition |
| flowstar | root_cause_fixed_order_2 | van_der_pol | 0.005 | 1.0000000000000007 | 0.17937393620021103 | 1.0 | 0.08711940376088023 | unavailable | complete_total_degree_2 | native_normalized_composition |
| flowstar | root_cause_fixed_order_3 | van_der_pol | 0.005 | 1.0000000000000007 | 0.11693427079808316 | 1.0 | 0.3217830960638821 | unavailable | complete_total_degree_3 | native_normalized_composition |
| flowstar | root_cause_fixed_order_6 | van_der_pol | 0.005 | 1.0000000000000007 | 0.11255717016086303 | 1.0 | 8.888198421802372 | unavailable | complete_total_degree_6 | native_normalized_composition |
| flowstar | root_cause_fixed_order_2 | van_der_pol | 0.01 | 1.0000000000000007 | 0.23915014631179776 | 1.0 | 0.045936278998851776 | unavailable | complete_total_degree_2 | native_normalized_composition |
| flowstar | root_cause_fixed_order_3 | van_der_pol | 0.01 | 1.0000000000000007 | 0.12151545904298577 | 1.0 | 0.16295035416260362 | unavailable | complete_total_degree_3 | native_normalized_composition |
| flowstar | root_cause_fixed_order_6 | van_der_pol | 0.01 | 1.0000000000000007 | 0.1125595187263233 | 1.0 | 4.576063263695687 | unavailable | complete_total_degree_6 | native_normalized_composition |
| flowstar | adaptive_order4_symbolic100 | van_der_pol | adaptive_0.002_to_0.1 | 9.999999999999 | 0.5209541309886361 | 10.0 | 1.5103623140603304 | unavailable | complete_total_degree_4 | native_normalized_composition; public QR off/on toggle not exposed in this Flow* checkout |
| torch_tm_flowpipe | order2_affine_reset_selected | harmonic | 0.01 | 4.0 | 10.973092497064886 | 4.0 | 12.701799672329798 | 396496.0 |  | selected_native_practical |
| torch_tm_flowpipe | order4_affine_reset_selected | harmonic | 0.01 | 4.0 | 10.494236021719196 | 4.0 | 23.0829613476526 | 396496.0 |  | selected_native_practical |
| torch_tm_flowpipe | order2_affine_reset_selected | coupled_quadratic | 0.005 | 0.25 | 0.043027088479031264 | 0.25 | 6.148705520201474 | 396496.0 |  | selected_native_practical |
| torch_tm_flowpipe | order2_affine_reset_selected | coupled_quadratic | 0.01 | 0.25 | 0.04303909579551132 | 0.25 | 3.079569525551051 | 396496.0 |  | selected_native_practical |
| diffreach | restricted_quasi_window100_round5_selected | riccati | 0.01 | 1.0 | 0.11389458226420779 | 1.0 | 0.0017184927128255367 | 431688.0 |  | selected_native_practical |
| diffreach | restricted_quasi_window100_round5_selected | harmonic | 0.01 | 4.0 | 0.3031861786118206 | 4.0 | 0.015098338015377522 | 526840.0 |  | selected_native_practical |
| diffreach | restricted_quasi_window100_round5_selected | coupled_quadratic | 0.005 | 0.25 | 0.0429775920432517 | 0.25 | 0.0035914061591029167 | 621616.0 |  | selected_native_practical |
| diffreach | restricted_quasi_window100_round5_selected | coupled_quadratic | 0.01 | 0.25 | 0.04298894317792923 | 0.25 | 0.0021049431525170803 | 657488.0 |  | selected_native_practical |
| diffreach | restricted_quasi_window100_round5_selected | van_der_pol | 0.01 | 1.0 | 0.15133241787615204 | 1.0 | 0.006060307379812002 | 727040.0 |  | selected_native_practical |
| flowstar | root_cause_order4_selected | harmonic | 0.01 | 4.0 | 0.28209137296574344 | 4.0 | 0.1152743804268539 | unavailable |  | selected_native_practical |
| flowstar | root_cause_order4_selected | coupled_quadratic | 0.005 | 0.25 | 0.04296770088971223 | 0.25 | 0.25262487726286054 | unavailable |  | selected_native_practical |
| flowstar | root_cause_order4_selected | coupled_quadratic | 0.01 | 0.25 | 0.042967728064041324 | 0.25 | 0.12507077446207404 | unavailable |  | selected_native_practical |
| flowstar | root_cause_order4_selected | van_der_pol | 0.005 | 1.0 | 0.11282176719263581 | 1.0 | 1.1072829510085285 | unavailable |  | selected_native_practical |
| flowstar | root_cause_order4_selected | van_der_pol | 0.01 | 1.0 | 0.113239997430433 | 1.0 | 0.5571194994263351 | unavailable |  | selected_native_practical |

All exact widths and horizons are in
`artifacts/authoritative/20260730T015245Z/one_step_summary.csv`,
`affine_carry_summary.csv`, `box_carry_summary.csv`,
`native_low_order_summary.csv`, and `native_pareto_summary.csv`. Runtime,
warm-up/build/JIT, ten repetitions, and memory are kept separate in
`runtime_summary.csv` (24 rows); no unavailable memory or
capability value is replaced by zero. Dependency/reset evidence is in the two
carry tables and `component_ablation.csv`. The associated figures are
`plots/01_*.png` through `plots/18_*.png`.

## Evidence strength

Riccati containment and small polynomial identities have analytic checks.
Flow* acceptance uses its native interval/Picard certificates, and CIR
round-trip checks establish export consistency. Torch and DiffReach preserve
their native validation statuses, but their float64 paths are not promoted to
MPFR-style formal roundoff proofs. DOP853 trajectory checks are deterministic
bug-finding sanity checks only; their zero-failure result is an admission gate,
not a certificate.

## BERN and remaining gaps

BERN is feasible only as a polynomial range backend: all
5 analytic cases were contained and
2 cancellation cases were tighter. It is
not a fourth reachability solver and presently lacks integration, Picard
validation, truncation accounting, endpoint substitution, reset, and a
formally outward-rounded sparse backend.

No unexplained native trajectory failure remains in an authoritative row.
Capability gaps remain explicit in `matched_basis_capabilities.csv`, including
unavailable exact-basis mappings and structured-remainder observability.
Further proof-grade claims for float64 Torch/DiffReach or the BERN prototype
require directed-rounding/MPFR evidence. The requested course PDFs were absent
from the server-wide filename audit, so `MATERIALS_MISSING.md` records all 16
exact names and `LITERATURE_MAP.md` does not claim page-level review.
