# VDP H2 dense Picard first-loss audit

## Decision

The opt-in H2 operator is sound and passes its preregistered same-input Gate B,
but the overall success target is **not met**.  H1+H2 recovers less than 10% of
the legacy excess in every T=1 channel and in both x channels at T=3.  It does
pass the T=6.32, native-horizon, runtime, checkpoint, and CPU/V100 consistency
gates.  The scientific decision is therefore
`H2_OPERATOR_ACCEPTED__OVERALL_SUCCESS_TARGET_FAILED`, not a claim that the
remaining Torch/Flow* gap is closed.

All measurements below came from the clean detached scientific SHA
`666c51ecc5575f203518d21f34b5c9948741fb17`; every runner summary records an
empty tracked-diff SHA256 and `worktree_dirty=false`.

## Phase 0: clean H1 replay

The exact requested base
`43be6d34461e809c291a2d57e120012755d29d51` was checked out in a fresh detached
clone.  Porcelain was clean before and after the replay.

- `compileall`: pass.
- H1 package verifier: 175 files, 25,209,699 bytes, six Gate-A checkpoints.
- targeted suite: 11 passed.
- full suite: 768 passed, 2 skipped (770 collected).
- T=0.1 legacy and H1 widths reproduced the tracked report bit-for-bit.

This closes the old dirty-worktree provenance gap without rerunning the H1
long-range matrix.

## First strict extra enclosure

The first loss is `raw.B1.x_squared`, reached through
`_DenseRawTraceScalar.__mul__` at `batched_dense_tm.py:2997-3004` and the generic
`BatchedTaylorModel.mul_trunc` remainder product at line 2386.

The binary operation is the same Taylor model multiplied by itself.  Its two
ordinary remainder operands are therefore the same symbol `R_x`, but the
generic multiplication computes them as independent intervals:

```text
generic term:  R_left * R_right = [-r,r] * [-r,r] = [-r^2,r^2]
shared symbol: R_x^2                                 = [0,r^2]
```

For the binary64 target radius `r = 1e-4`, the exact unnecessary lower width is

```text
54445178707350159372354900760041
--------------------------------------------------
5444517870735015415413993718908291383296
= 1.000000000000000020922561e-8
```

No earlier Picard construction, truncation, cutoff, integration, base add, or
raw-expression stage has a strict extra enclosure under the independent exact
oracle.  This is a dependency loss, not a missing truncation term or a changed
initial state.

Stock Flow* source at
`b85a3211748cb77b736fe4ad42ee02d8d2b81148` confirms the reference semantics:
the VDP expression is `(1-x^2)*y-x`, `Picard_ctrunc_normal` evaluates at
`k=order-1`, integrates once in time, and adds `poly_diff` before the subset
test.  Its generic multiply also uses
`P_left R_right + P_right R_left + R_left R_right`; consequently Flow* runtime
is a cross-check only and is not used as the soundness oracle.

## Gate A soundness ledger

The clean-SHA ledger contains all four Picard iterations and 36 operator
stages, including multiplication before/after equal-exponent merge, truncation,
cutoff, time integration, base-remainder add, raw RHS replay, `poly_diff`, and
the final subset-test image.

Each production binary64 coefficient and interval endpoint is lifted exactly
to `Fraction`.  A separate `RationalPolynomial` implementation maps each
polynomial to an exact tensor-product Bernstein basis; deterministic rational
bisection is used only to tighten a Bernstein hull, never to sample.  All 36
operator stages and all four B1/B2 `poly_diff` component checks are contained.
The audit records one explicit downstream `validation_eps` reserve where the
production validation path owns it; it is a non-additive local witness and is
not injected once per audited stage.

The execution ledger separately records all five actual `validation_eps`
inflations in production order: candidate seed, ordinary-residual diagnostic,
time-scaled raw RHS, `poly_diff`, and final raw-compat image.  Each has exact
binary64-rational before/after intervals.  The ordinary-residual replay matches
the production trace; it is a finite diagnostic rather than a decision image
in raw-compat mode.  The audit-harness commit also proves that every production
path is byte-identical to scientific SHA `666c51e`.

Soundness boundary: H2 preserves the shared identity only for an actual
self-square.  It still treats the polynomial range and the one remainder symbol
as independent, so the joint enclosure is conservative.  The extrema of
`2 P R + R^2` are checked at rectangle corners, `R=0`, and the in-range convex
vertex `R=-P`, with outward binary64 operations.  CUDA is tested only for
implementation consistency and is not a directed-rounding oracle.

## Preregistered Gate B

| Cell | Operator | Result |
|---|---|---|
| B1 | current distributed `y-x-x*x*y` | executed baseline |
| B2 | `(1-x*x)*y-x` plus joint shared square | executed, Gate B pass |
| B3 | dependency-preserving tau integration | not executed; stop after first pass |
| B4 | dependency-preserving `poly_diff` | not executed; stop after first pass |

B1 and B2 start from the same serialized target-remainder prestate.  At step 1
B2 removes `4.008111544150814e-6` of the y raw-image width: 18.935036% of the
legacy-vs-Flow* y excess.  The corresponding segment removal is
`4.008111543996051e-6`, or 18.588729%.  Both x widths are byte-identical, hence
no raw or segment channel regresses.  This satisfies the preregistered H2 Gate
B and requires stopping before B3/B4.

## Production change

The new mode
`flowstar_raw_remainder_compat_factorized_joint` is explicit opt-in and is
dense-only.  It combines two changes that were audited together in B2:

1. `PolynomialODE.evaluate_canonical_factorized` uses the deterministic
   `(1-x*x)*y-x` graph for the raw RHS and the matching regular RHS used by
   `poly_diff`.
2. An identity self-square uses `square_trunc_dependency_preserving`, which
   keeps the retained `P^2`, owns truncation/cutoff in the existing ledger, and
   encloses the single-symbol term `2PR+R^2` jointly.

Omitting `--validation-mode` leaves the frozen legacy/H1 validation contract
unchanged.  H2 is combined with H1 by selecting
`normalized_insertion_dependency_preserving` as the reset mode.

## Step-1 stage matrix

| lane | endpoint x | endpoint y | segment x | segment y |
|---|---:|---:|---:|---:|
| Flow* cross-check | 0.301112793636261 | 0.122633766819920 | 0.325245362544320 | 0.149405911444344 |
| legacy | 0.300898496987774 | 0.121334291169393 | 0.325247221932296 | 0.149427473496953 |
| H1 | 0.300898496987774 | 0.121334291169393 | 0.325247221932296 | 0.149427473496953 |
| H1+H2 | 0.300898496987774 | 0.121330283057849 | 0.325247221932296 | 0.149423465385409 |

Flow* runtime widths can be narrower than a Torch width without proving that
Flow* is an enclosure.  Exact-oracle containment, not this table, is the Gate-A
soundness decision.

## Fixed h=0.01 matrix

`recovered` is `(legacy - H1+H2) / (legacy - Flow*)`.  `H2 incremental` is
`(H1 - H1+H2) / (legacy - Flow*)`.

| T | channel | Flow* | legacy | H1 | H1+H2 | recovered | H2 incremental |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | endpoint x | 0.0795178281 | 0.0879559235 | 0.0879439409 | 0.0878008220 | 1.838% | 1.696% |
| 1 | endpoint y | 0.1115769156 | 0.1142921745 | 0.1142770205 | 0.1140809159 | 7.780% | 7.222% |
| 1 | segment x | 0.0837525716 | 0.0922215177 | 0.0922095309 | 0.0920663603 | 1.832% | 1.691% |
| 1 | segment y | 0.1196669930 | 0.1285652386 | 0.1285492647 | 0.1283427687 | 2.500% | 2.321% |
| 3 | endpoint x | 0.1385053267 | 0.1872595897 | 0.1865686202 | 0.1847205549 | 5.208% | 3.791% |
| 3 | endpoint y | 0.1088516715 | 0.1558642556 | 0.1541143565 | 0.1509403569 | 10.474% | 6.751% |
| 3 | segment x | 0.1639208754 | 0.2127350396 | 0.2120438511 | 0.2101951972 | 5.203% | 3.787% |
| 3 | segment y | 0.1256837926 | 0.1727677949 | 0.1710170720 | 0.1678415885 | 10.463% | 6.744% |
| 6.32 | endpoint x | 0.1530755556 | 0.9165121029 | 0.7919255932 | 0.6805171232 | 30.912% | 14.593% |
| 6.32 | endpoint y | 0.1222956280 | 1.5898587283 | 1.2745154469 | 1.0373859831 | 37.646% | 16.158% |
| 6.32 | segment x | 0.1783273000 | 0.9420414425 | 0.8174173573 | 0.7059752141 | 30.910% | 14.592% |
| 6.32 | segment y | 0.1398213090 | 1.6080698025 | 1.2925904925 | 1.0553588399 | 37.644% | 16.157% |

Thus T=1 fails in all four channels.  T=3 passes only endpoint y and segment y;
both x channels fail.  T=6.32 is strictly narrower than H1 in all four
channels.

## Native T=10 requests and CPU performance

| lane | continuous endpoint | accepted | rejected attempts | runtime | peak RSS | terminal cause |
|---|---:|---:|---:|---:|---:|---|
| legacy | 6.397083942944808 | 307 | 48 | 134.621 s | 527,523,840 B | residual subset failure before h_min |
| H1 | 6.441433080631058 | 293 | 46 | 162.135 s | 522,571,776 B | residual subset failure before h_min |
| H1+H2 | 6.482041958201616 | 278 | 44 | 162.909 s | 516,841,472 B | residual subset failure before h_min |

The H1+H2 native runtime is 1.2101x legacy.  On the fixed T=6.32 runs it is
342.551 s versus 255.578 s legacy, or 1.3403x; peak RSS is 640,303,104 B versus
643,198,976 B.  Both measured CPU ratios pass the 2x gate.  The T=10 stretch
goal remains failed.

The terminal H1+H2 rejection is localized to y's upper side: the final image is
`[-1.0064678895982489e-4, 1.0685420052420151e-4]` against target
`[-1e-4,1e-4]`, giving limiting margin `-6.854200524201504e-6`.  The largest
additive validated-ledger category for that component is
`polynomial_truncation`, width `1.9374133210958038e-4`.  For comparison, legacy
is y/lower limited and H1 is y/upper limited; all three have
`polynomial_truncation` as their largest additive category.

## V100 measurement

GPU 0 was measured as `Tesla V100-SXM2-16GB`.  For fixed T=0.1:

| lane | runtime | peak RSS | accepted | max CPU/V100 width delta |
|---|---:|---:|---:|---:|
| legacy | 11.171 s | 1,015,861,248 B | 10 | 0 |
| H1 | 14.260 s | 1,015,185,408 B | 10 | 0 |
| H1+H2 | 14.432 s | 1,034,964,992 B | 10 | 0 |

These are measured values only.  No CUDA directed-rounding soundness or speedup
claim is made.

## Tests and evidence

The production tests cover the exact Bernstein micro-oracle, asymmetric and
zero remainder, a vertex crossing zero, cutoff ownership and exact cutoff
boundary, order overflow, same-step raw-residual reduction, opt-in runner
contract, CPU/CUDA consistency, and bitwise checkpoint/resume on each available
device.  The development full suite passed with 780 passed and 2 skipped.  From
the clean scientific SHA, the focused suite passed 20/20 and a second full
suite passed with 780 passed and 2 skipped in 340.24 s; both XML reports are
stored in the evidence package.

Raw attempts, segments, remainder ledgers, range traces, Gate A/B ledgers,
matrix summaries, test XML, source provenance, and deterministic gzip source
hashes are stored under
`evidence/vdp_h2_dense_picard_first_loss/20260818T091126Z`.  Verify with:

```bash
python experiments/verify_vdp_h2_dense_picard_evidence_20260818.py \
  evidence/vdp_h2_dense_picard_first_loss/20260818T091126Z
```

The verifier deliberately requires the early 10% target and T=10 stretch goal
to be recorded as failures.  It will reject a package that silently promotes
this operator-level success to an overall success.
