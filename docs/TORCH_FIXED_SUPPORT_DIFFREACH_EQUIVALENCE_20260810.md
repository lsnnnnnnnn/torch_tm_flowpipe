# Torch fixed-support / DiffReach equivalence

Date: 2026-08-10

## Result

The mandatory configurable fixed-support lane is implemented in the canonical
Torch TM package and qualifies against pinned DiffReach source
`dd628eb443b517d6415de93e7035b4baef73963e`.  There is no runtime dependency on
DiffReach, JAX, or CROWN-Reach.

The VDP descriptor has seven slots in the exact upstream order
`[1, t, xi0, xi1, t^2, t*xi0, t*xi1]`.  Its deterministic support SHA256 is
`0ae11ee9d911d45e42294df74ef2896ecb9aeb9f3d7851c09ea90e2bb2631f5e`.
It freezes signed multiplication routes, including the final duplicate-t²
correction, integration routes, grouped discarded-term policies, Horner range,
and source contract.

## Semantic gates

| gate | evidence | result |
|---|---|---|
| constant/affine/quadratic algebra | analytic unit tests | pass |
| multiply/integrate/project routes | exact c/L/Lt expression test and frozen manifest | pass |
| range and discarded terms | endpoint/tube and named ledger tests | pass |
| one polynomial Picard construction | frozen upstream float64 fixture | pass, bit-exact |
| every DR-RP round | 10 masks and accepted lo/hi arrays | pass, bit-exact |
| one VDP segment | canonical solver with normalization/carry | pass |
| complete short horizon | 10-step pinned-operation float64 replay; CPU and V100 B64 | pass |
| full native/matched horizon | B64, h=.01, 1,000 steps, T10 | pass |

The focused suite contains 15 tests including initial-inclusion failure
propagation, batch permutation, CUDA float64, endpoint/tube separation, and the
independent exact-rational replay.  With the related dense regressions, 43
tests passed before the implementation checkpoint.

## Exact semantics reproduced

Coefficients use `[batch, state_output, monomial_slot]`.  Multiplication and
integration routes are precomputed by the descriptor; static route stages
preserve upstream expression order.  Every multiplication exposes:

- pure spatial quadratic overflow;
- time-cubic overflow;
- time-quartic overflow;
- polynomial/remainder cross products;
- remainder/remainder products.

Integration separately exposes time-cubic, time-squared-spatial, and
integrated-input-remainder intervals.

Polynomial construction is exactly two fixed-support Picard iterates.  DR-RP
records the initial inclusion and all 10 later masks.  Failed initial inclusion
is a solver failure.  Later component failure adopts the new polynomial but
retains the previous interval component, matching upstream.  Normalized affine
carry and the J/Phi symbolic queue reproduce the upstream alignment, including
the easily misread rule
`(r_x0 + seed_through).where(count == 0, r_x0)` and clear-on-cap behavior.

The solver always returns two different geometric objects:

- endpoint: local time fixed at h;
- full-step tube: local time ranges over `[0,h]`.

No Python data-dependent scalar decision occurs inside multiply, integrate,
range, DR-RP, or symbolic-carry tensor operations.  The public solver performs
one explicit host synchronization per step to enforce failed-initial-inclusion
rejection; the T10 run records 1,000.  Solver device-transfer count is zero.

## Full-horizon comparison

The clean-source Torch run is `ac970b771a43c24587a70ab97eb08a71036c613b`,
B64, CPU float64, fixed h=.01.  It completed all 1,000 steps with every one of
128,000 initial component inclusions and 1,280,000 later-round component masks
true.  No retain-on-failure event, fallback, or repair occurred.

| output | stock DiffReach native endpoint | Torch explicit-f64 endpoint | absolute endpoint delta |
|---|---:|---:|---:|
| x lower | `-1.3964494166017873` | `-1.3964471717050193` | `2.2448967680e-6` |
| x upper | `-1.2195860990449185` | `-1.2195885421989074` | `2.4431539889e-6` |
| y lower | `-2.5022084859452010` | `-2.5022062536670100` | `2.2322781910e-6` |
| y upper | `-2.2803973198181486` | `-2.2804009831591450` | `3.6633409964e-6` |

This is not claimed as bitwise full-driver equivalence.  The pinned native
driver enables JAX x64 but its model, identity, and symbolic-state builders
default to float32; the Torch lane is explicitly float64 throughout.  The
operator-level fixture forces pinned DiffReach operations to float64 and is
bit-exact.  The small T10 difference is consistent with the native mixed dtype
and does not change completion or any returned initial inclusion flag.

Torch also records geometric data absent from the stock DiffReach artifact:

| object | x | y |
|---|---:|---:|
| last full-step tube | `[-1.3966090523772694, -1.1949924951734890]` | `[-2.5275335398097694, -2.2804009831591450]` |
| full-horizon tube | `[-2.0115852688308564, 2.0574214109557194]` | `[-2.6891205817023787, 2.6857282956487234]` |

The measured T10 CPU solver time was `97.92799484729767 s`, but that run
overlapped other verification work and is not the isolated timing result used
for a performance claim.  Isolated timing and batch scaling are deferred to
the required performance matrix.

## Numerical soundness

An independent exact-rational oracle replays the one-step rational VDP fixture
without calling the Torch algebra.  It checks retained coefficients,
multiplication/integration overflow, initial inclusion, every DR-RP mask, final
accepted remainder, endpoint, and tube on CPU and V100 CUDA.

CPU and CUDA agree.  The ordinary binary64 values are **not directly outward
qualified**: retained points and several interval endpoints miss the exact
rational result by at most 2 ULP.  Therefore the ordinary lane remains
`empirically sampled only`; no universal GPU directed-rounding claim is made.
The independently checked companion envelope obtained by outward expansion of
every observed point/endpoint by 2 ULP is qualified as
`independently outward replayed for exact benchmark workload`.  This envelope
qualification is one-segment/workload-specific and is not silently substituted
into the 1,000-step ordinary run.

## Evidence map

Machine-readable closure is in
`outputs/mainline_realignment_20260810/20260810T025910Z/02_fixed_support/fixed_support_equivalence.json`.
The full CPU run is under `cpu_b64_t10/`, the CUDA short run under
`cuda_b64_t0p1/`, and exact-rational reports are
`fraction_replay_cpu.json` and `fraction_replay_cuda.json`.
