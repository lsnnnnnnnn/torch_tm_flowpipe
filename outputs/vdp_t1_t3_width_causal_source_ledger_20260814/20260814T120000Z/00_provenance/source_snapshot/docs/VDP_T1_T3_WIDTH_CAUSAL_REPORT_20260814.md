# Van der Pol T=1/T=3 width-gap result — 2026-08-14

Final scientific conclusion:

`T1_T3_WIDTH_CAUSE_CLOSED__EARLY_GAP_IMPROVED__TERMINAL_STILL_OPEN`

## Plain-language answer

Torch is not suddenly becoming wide at its terminal step. The gap is already
present by T=1 because every accepted boundary converts part of the old
parameterization uncertainty into ordinary intervals. When the same uncertain
quantity appears more than once in the next polynomial, especially through
`x*x*y`, ordinary interval arithmetic treats those occurrences as if they
could vary independently. That loses cancellation and shared dependence. The
loss is small at one boundary, but it is repeated at every boundary, so by T=3
it has accumulated into a visible gap.

Later, the wider set is itself fed back through Van der Pol's nonlinear term.
For perturbations `dx,dy`, the exact change in the second right-hand side is

`(1-x^2)dy - (2xy+1)dx - y dx^2 - 2x dx dy - dx^2 dy`.

The last three terms are quadratic/cubic in uncertainty. Consequently the gap
does not merely add at a constant rate: wider `x,y` prestates make the next
`x^2 y` image wider, which makes the following prestate wider again. That is
why an O(1e-2) early excess becomes O(1) near T=6.32.

The new G1 carry proves that source identity is a real consumer variable, not
metadata. It keeps one fresh affine source per state component in the actual
next dense Picard polynomial, then merges and outward-collapses every old-source
term after exactly one generation. This gives small, deterministic reductions
at T=1, T=3, and 6.32. It does not close the terminal: the dominant boundary
mass is still ordinary parameterization/nonlinear-collapse remainder, much
larger than the fresh structured source.

## Frozen contracts and initial-set repair

The main comparison retains the historical
`binary64_literal_matched_contract`: identical Van der Pol expression tree,
complete total-degree O4 basis, cutoff `1e-10`, candidate remainder `1e-4`,
float64 CPU B1, fixed `h=0.01`, and the recorded proactive depth-1 range policy.
Endpoint, last-segment tube, and prefix tube remain separate quantities.

The auxiliary `exact_decimal_contract` now constructs outward affine initial
models containing x `[11/10,7/5]` and y `[47/20,49/20]` in both the Torch and
Flow* audit drivers. Exact rational witnesses and an actual dense consumer test
pass. The compensation is about 1e-16 and therefore cannot explain the T=1/T=3
gap.

## Rebuilt width curve

The raw joined ledger contains 632 common accepted times and 2,528 channel
rows. It stores lower, upper, width, per-step increments, prestates, range and
remainder stages, raw/refined images, margins, source provenance, and observer
parity. The required legacy excesses were reproduced:

| Time | Legacy absolute-excess range |
|---|---:|
| T=1 | 0.002715258977108115 to 0.008898245576982322 |
| T=3 | 0.047012584088458986 to 0.04881416425335772 |
| T=6.32 | 0.7634365472439139 to 1.4682484934615618 |

The apparent Flow* “zero” is closed. Direct subtraction of raw lower/upper
bounds gives minima 0.00861211181140531, 0.026272600935460244,
0.008888711363604695, and 0.030888053869117083. All exceed 0.0086. The evidence
plot shows lower, upper, and width together; no division by a numeric zero is
involved.

## First actual consumer and same-prestate interventions

The first causally active field is
`affine_source_coefficient_in_next_dense_picard_input`, created at the accepted
boundary in `flowpipe.py::_flowstar_bounded_source_ledger_transition` and
consumed by `batched_dense_tm.py::dense_picard_validate_step`.

At step 1→2, before T=1, before T=3, and before T=6.32:

- changing a live coefficient by 1% changed the actual next raw Picard image;
- changing lineage metadata left the complete next result bit-identical;
- exact ordinary materialization of the same affine source set made the y raw
  image about 1.93–2.00 times as wide;
- at the 6.32 boundary, source identity validated with y margin
  `2.7829552162311744e-6`, while ordinary materialization rejected with margin
  `-8.322524825246946e-5`;
- the existing legacy rebox was slightly better than G1 on those isolated
  legacy prestates. This negative result is retained rather than hidden.

A lossless Flow* operator run is marked `UNAVAILABLE`: the Torch prestates do
not serialize Flow*'s exact `Phi_L/J` and ordinary remainder as a
`Symbolic_Remainder` object. No lossy adapter is presented as Flow*.

The candidate's own frozen terminal prestate at 6.382737816137232 was also
audited. Candidate, ordinary-only, and legacy-box controls all reject. The
candidate y margin is `-1.2700015366953245e-6`; ordinary-only is much worse at
`-8.009388109986473e-5`. Payload and metadata tamper behavior remains correct.

## Authoritative G1 source-ledger contract

For `d` state components, the boundary has exactly `2d` uncertainty variables:
`d` base coordinates and `d` one-generation sources. Component `i` represents

`X_i = c_i + Q_i(u) + R_o,i + rho_i z_i`, with `z_i in [-1,1]`.

The complete validated remainder ledger is outward-summed and lifted to one
fresh affine source per state component. The lift proves
`[lo,hi] subseteq midpoint + radius[-1,1]`. Old source-bearing polynomial terms
are merged by complete exponent and outward-evaluated once at the next accepted
boundary. Rejected candidates preserve the immutable state object and hash.
The source count is always `d`; no K tuning, magnitude ranking, state-specific
rule, or horizon growth exists.

All 13 independent micro-oracles pass: two-step affine propagation,
shared-source cancellation, `x^2 y`, mixed/asymmetric products, O4 ownership,
tau substitution and duplicate merge, collapse containment, retry atomicity,
B1/B8/B64 permutation, CPU/CUDA decision parity, exact-decimal consumption,
observer parity, and payload/metadata tampering.

This differs from both rejected predecessors:

- it does not clone the complete endpoint polynomial indefinitely; every old
  source retires after one actual consumer generation;
- it does not use K16 interval linear maps or total-delta padding; source IDs
  are polynomial variables in the dense Picard input.

## Fresh fixed-schedule result

Only carry mode differs. G1 is initially a little wider at T=0.1, then is
slightly narrower from T=0.5 onward.

| Time/channel | Legacy excess | G1 excess | Width reduction |
|---|---:|---:|---:|
| T=1 endpoint x | 0.008438095460857609 | 0.008414338346654171 | 0.000023757114203438 |
| T=1 endpoint y | 0.002715258977108115 | 0.002702636340927378 | 0.000012622636180737 |
| T=1 segment x | 0.008468946029541069 | 0.008445129381821070 | 0.000023816647719999 |
| T=1 segment y | 0.008898245576982322 | 0.008885131245840339 | 0.000013114331141983 |
| T=3 endpoint x | 0.048754263006459575 | 0.048644024032262310 | 0.000110238974197263 |
| T=3 endpoint y | 0.047012584088458986 | 0.046912214622062190 | 0.000100369466396799 |
| T=3 segment x | 0.048814164253357720 | 0.048703574934650050 | 0.000110589318707666 |
| T=3 segment y | 0.047084002379503165 | 0.046984774490028690 | 0.000099227889474474 |
| T=6.32 endpoint x | 0.7634365472439139 | 0.7614462129897419 | 0.001990334254171944 |
| T=6.32 endpoint y | 1.4675631002797296 | 1.4630616231218752 | 0.004501477157854428 |
| T=6.32 segment x | 0.7637141425470748 | 0.7617227868433671 | 0.001991355703707787 |
| T=6.32 segment y | 1.4682484934615618 | 1.4637477948326790 | 0.004500698628882915 |

The improvement is real but small. None of the first ratio crossings above
1.1, 1.5, 2, or 5 moves at the 0.01 schedule resolution.

At fixed T=6.32, G1's prestate scales are `[0.4496154192194197,
0.7653225553623377]`, versus legacy `[0.4505899076597181,
0.7675707013167182]`. The raw y margin improves from
`2.796769042862058e-6` to `3.4430665993788175e-6`. But the post-boundary G1
ordinary mass is 2.1933445893242376 while fresh structured mass is only
0.0001863369380135942. This quantitative imbalance explains why G1 cannot
materially bend the late curve.

## Native horizons and terminal

Native results are not mixed into fixed-schedule ratios.

- Legacy completes T=1, T=3, and T=6, then accepts 307 segments through
  6.397083942944808 and rejects the next attempt.
- G1 completes T=1, T=3, and T=6, then accepts 300 segments through
  6.382737816137232 and rejects the next attempt.
- Fresh T=7.5 and T=10 requests repeat those exact deterministic stopping
  points. Neither reaches 6.5, 7.5, or 10.
- No endpoint repair, fallback, hidden sparse inner loop, source-ledger
  publication failure, or soundness failure occurs. The stop is the unchanged
  target-remainder subset rejection.

Thus G1 improves early dependence accumulation but does not solve the terminal;
under the native controller it stops about 0.01435 earlier than legacy.

## Performance

CPU float64 B1 remains authoritative. A real B1 step separates into median
0.21901137399254367 s for dense Picard/range/validation and
0.04379857698222622 s for accepted-boundary carry; carry is about 16.7% of
their combined time. For the T=0.1 run, non-trace outer-loop overhead is
0.1845683830033522 s.

The tensor affine-lift kernel was measured at B1/B8/B64/B256/B512 on CPU and a
Tesla V100 with synchronization and transfer/memory reporting. CUDA is slower
at every batch (CPU/CUDA kernel speed ratios 0.225–0.331). The complete T=0.1
candidate run takes 5.4713 s on CPU and 16.3824 s on V100, with 20 device
transfers. This does not overturn the prior negative GPU result, and no batched
kernel throughput is described as a multi-step solver speedup.

## Evidence strength and open work

- Formal/discrete: canonical exponent merge, exact-rational fixtures, affine
  source algebra, retry/state hashes, and tamper routing.
- Directed-numerical: binary64 `nextafter` interval lift, exact-decimal outward
  initialization, dense additive ledger containment, and Flow* MPFR audit
  initialization.
- Deterministic empirical: all widths, per-step increments, margins, schedules,
  horizons, runtimes, memory, and threshold times.
- Sampling-only: trajectory sample sanity checks. They are not used as the
  soundness proof.
- CUDA: implementation-consistency evidence only, not a formal directed-
  rounding guarantee.

The remaining scientific problem is the large ordinary parameterization and
retired nonlinear-collapse mass. A future contract would have to retain more of
that owner structure without unbounded source growth and without reverting to
complete endpoint carry or K16 total-delta padding. G1 is frozen; this study
does not retune it after observing the terminal.
