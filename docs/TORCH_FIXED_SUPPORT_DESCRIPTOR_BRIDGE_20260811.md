# Torch fixed-support descriptor bridge

Date: 2026-08-11

## Outcome

`FIXED_SUPPORT_BRIDGE_CLOSED` through G2/T1 for every A0–A4 cell in both B1
and B64.  G3/T10 is a separately gated extension and is not needed to relabel
the completed T1 bridge.

## Eligibility

Empirical ordinary-float64 fixed-workload causal diagnostics only; R7 frozen
regression is separately bit-exact.

## What is comparable

Adjacent rows at the same B, h, time, output object, success status, and
soundness scope.

## What is unavailable

B1/B64 cross-partition deltas, post-failure deltas, and a universal dominant
factor.

## Negative results

Preregistered metrics disagree on a single dominant factor; R35 is not a
promoted complete-O4 production lane.

## Exact evidence paths

New-package directories `09_fixed_support_descriptor/` and
`10_bridge_ladder/`.

The generic descriptor core preserves the frozen DR7 manifest and expression
order while adding complete total-degree O4 support
`R35 = {t^a xi0^b xi1^c : a+b+c <= 4}`.  Its 35-slot order, multiplication
destination/overflow table, differentiation, integration, endpoint
substitution, and support SHA are generated from exponents rather than VDP
cases.  R7 object output remains bit-exact with the existing solver at the
one-step regression gate.

## Causal ladder

Every adjacent cell changes exactly one factor:

| cell | support | Picard | validator | carry |
|---|---|---:|---|---|
| A0 | R7 | 2 | DR-RP | DiffReach J/Phi |
| A1 | R35 | 2 | DR-RP | DiffReach J/Phi |
| A2 | R35 | 4 | DR-RP | DiffReach J/Phi |
| A3 | R35 | 4 | raw-compatible | DiffReach J/Phi |
| A4 | R35 | 4 | raw-compatible | complete normalized insertion |

All cells use fixed `h=0.01`, target remainder radius `0.01`, no cutoff, and
`h_min=0.01`.  G0, G1, and G2 were run in order; each later runner verifies
the immediately preceding closed summary before executing.

## T1 factor evidence

The table reports the maximum component width at the T1 checkpoint.  Values
are empirical ordinary-float64 observations in the same B/h/time/output
scope.

| cell | B | minimum margin | max raw width | max endpoint width | max tube width |
|---|---:|---:|---:|---:|---:|
| A0 | 1 | 0.00915704630678304 | 0.0016656063928320551 | 0.14992401574807945 | 0.15642045239085123 |
| A1 | 1 | 0.009430925940025633 | 0.0011381307591594848 | 0.10190104367667835 | 0.10999112114170861 |
| A2 | 1 | 0.009431050718827121 | 0.0011378813835343663 | 0.10173989555823543 | 0.10982997302326569 |
| A3 | 1 | 0.009409275249298627 | 0.0011814132922979134 | 0.14798095055357652 | 0.15607102801860684 |
| A4 | 1 | 0.009386101905541526 | 0.0012277302209464072 | 0.1928691957680833 | 0.20095927323311363 |
| A0 | 64 | 0.009426568388464066 | 0.0011459354699428915 | 0.017832832285291922 | 0.027009132037398076 |
| A1 | 64 | 0.009432640422020744 | 0.0011347189171907364 | 0.016200143271682454 | 0.02570290881108117 |
| A2 | 64 | 0.009432729547898396 | 0.0011345407511443846 | 0.016101702869913248 | 0.025604468409311965 |
| A3 | 64 | 0.009413133333910839 | 0.0011737274114371984 | 0.058650311670115585 | 0.06619767082699668 |
| A4 | 64 | 0.009406212653736246 | 0.0011875652124402965 | 0.07707558480685472 | 0.08631393094286766 |

Support is the largest favorable first-decision margin change in B1 and
substantially tightens the B1 T1 objects.  Validator and carry changes then
widen the T1 objects, while their relative magnitudes differ between B1 and
B64.  Because the preregistered metrics and partition lanes do not agree on a
single ordering, no dominant-factor label is asserted.

Each checkpoint records retained/dropped counts, retained polynomial range,
truncation and integration observations, polynomial/remainder products,
raw candidate, margin, endpoint, segment tube, carry width, stage runtimes,
peak memory, and decision.  The inherited operation ledger is explicitly
non-additive after scaling/composition; a separate exact boundary coverage
entry is checked to contain the model remainder.

The bridge is an empirical fixed-workload causal diagnostic.  It is not a
universal numerical-soundness proof and does not compare B1 with B64 as the
same object.
