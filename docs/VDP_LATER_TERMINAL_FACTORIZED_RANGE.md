# VDP later-terminal factorized polynomial range

## 1. Executive conclusion

The deterministic dense Horner/factorized evaluator passes its CPU and CUDA
correctness gates, but it does **not** close the frozen later terminal step.
Horner alone is wider than the independently validated natural enclosure on
the decisive terms and therefore sound selection keeps natural. Combining
four-leaf subdivision with per-leaf Horner is complementary: it improves the
terminal y subset margin from `-1.99995911680722e-5` to
`-1.5859969428028492e-5`, but the unchanged
`flowstar_raw_remainder_compat` predicate still rejects.

The STOP gate therefore applies. No fresh `T=6.5`, `7.5`, or `10` run was
started. The factorized-range state is
`H1_factorized_range_correctness_complete`; the overall validated-horizon
state remains `R4_historical_range_midpoint_horizon_crossed` at
`6.397083942944808`.

## 2. Exact baseline and numerical contract

The authoritative baseline was branch
`codex/vdp-terminal-range-closure-20260805` at
`cdb54bd3d2ffb49a0b58245055932756ebc3aa47`, verified against the fetched
remote-tracking branch and `git ls-remote`. Work proceeded in an isolated
worktree on `codex/vdp-later-terminal-factorized-range-20260805`; unrelated
user changes in the original worktree were not reset, stashed, copied, or
committed.

The immutable contract is:

- `x' = y`, `y' = y - x - x^2 y`;
- `x(0) in [1.1,1.4]`, `y(0) in [2.35,2.45]`;
- order 4, float64, target remainder `1e-4`, cutoff `1e-10`;
- `h_min=0.002`, `h_max=0.1`;
- normalized insertion, constant right-map centering, Flow*-compatible
  scheduling, and `flowstar_raw_remainder_compat` validation.

No ODE, initial-set, coefficient, exponent-support, degree, scheduler,
acceptance predicate, endpoint, or remainder contract changed.

## 3. Frozen checkpoint and later-terminal attribution

The committed canonical-JSON checkpoint has full SHA256
`dcb8f646d45c9742e0cff23fea596c12e53d8ccd00d1544f70564a44a7463420`.
It loads at exactly `t=6.397083942944808`, attempts exactly
`h=0.003623635847674574`, matches the expected contract/order/dtype/source
hashes, and round-trips payload and manifest byte-for-byte.

Every attribution lane has candidate coefficient hash
`bc1433d0d3c89339fca6091e41c0a6667d70c92d2dd4e35ae8b14236d131863c`
and exponent-support hash
`d0aa354b9057267556d5bb3bc09a36ed4162b36fb44588b0b930dd9e935041e9`.
All use one Picard validation attempt, the same h and predicate, and have zero
nonfinite events, repair, sampling-based tightening, external endpoint
substitution, or sparse fallback.

| Lane | Selected intervention | y image | y margin | Accepted |
|---|---|---:|---:|---:|
| A0 | natural | `[-1.6369929499718532e-4, 1.7584392650575045e-4]` | `-7.584392650575044e-5` | no |
| A1 | subdivision: polynomial truncation | `[-1.199995911680722e-4, 1.1599079695982693e-4]` | `-1.99995911680722e-5` | no |
| A2 | subdivision: integration overflow | same as A0 | `-7.584392650575044e-5` | no |
| A3 | subdivision: polynomial/remainder products | `[-1.6369446406269271e-4, 1.7583902677175938e-4]` | `-7.583902677175937e-5` | no |
| A4 | truncation + integration overflow | same as A1 | `-1.99995911680722e-5` | no |
| A5 | all contexts, depth 1 | `[-1.1999480706960097e-4, 1.1598589846909286e-4]` | `-1.9994807069600968e-5` | no |
| A6 | all contexts, depth 5 / cap 64 | same as A5 | `-1.9994807069600968e-5` | no |

The direct A0 failure is y target-subset containment: both sides exceed
`[-1e-4,1e-4]`, with the upper-side violation larger. After polynomial
truncation subdivision, the dominant failure is the lower side by
`1.99995911680722e-5`. A2 proves that a large `integration_overflow` ledger
width is not by itself a causal blocker: tightening it changes neither final
image nor margin. Polynomial truncation is causal for most of the available
improvement, but neither it nor the A4 joint intervention changes acceptance.
A5 and A6 are numerically identical at the final image, so extra leaves after
depth 1 cannot resolve this state.

## 4. Dense Horner/factorized algorithm

The evaluator accepts generic `[batch, output, term]` coefficient tensors and
`[term, variable]` nonnegative integer exponents; it is not specialized to VDP
or three variables.

1. Equal exponents are ordered lexicographically. Their exact binary64 input
   values are aggregated in original term-index order using an outward
   interval add at every operation. It never treats `scatter_add` as exact,
   and exact or near cancellation retains the aggregation envelope.
2. For a specified variable permutation, exponents are recursively grouped by
   degree. Each coefficient polynomial is recursively evaluated, then missing
   degrees are handled by ordinary Horner multiply-add stages down to degree
   zero. Constants, zero coefficients, unequal maximum degrees, and arbitrary
   dimensions are supported.
3. Every stage records scope, variable, degree, path, coefficient interval,
   product, intermediate interval, operation, and safeguard. Reconstructed
   leaf exponents must exactly equal the canonical support before the result
   is valid.
4. The finite registered family is `[u0,u1,tau]`, `[u1,u0,tau]`, and
   `[tau,u0,u1]`. All are evaluated before selection. Width is primary and the
   lexicographic variable tuple is the stable tie-break.
5. Natural and Horner enclosures are independently checked for finite ordered
   bounds. Horner is selected only when validated and no wider. Invalid
   Horner results cause an explicit natural fallback; a bad natural enclosure
   fails closed. No interval intersection is used.

Named policies are `natural`, `horner_fixed`, `horner_registered_best`,
`subdivision`, and `subdivision_then_horner`/`horner_per_leaf`. The default
natural policy and historical natural schedule remain byte/numerically
unchanged by regression test.

## 5. Floating-point safeguard and limitation

Coefficient aggregation, interval addition, multiplication, and every Horner
multiply-add use `torch.nextafter` outward expansion. Existing power and
natural-reduction safeguards remain in place. Tests cover subnormal and huge
coefficients, alternating signs, exact/near cancellation, shifted and
cross-zero boxes, and nonfinite fail-closed behavior.

The claim is a **safeguarded float64 enclosure**, not a hardware-independent,
machine-checked directed-rounding proof. CPU repetition is byte/numeric/hash
stable in the tested environment. CUDA results for batches 1, 16, and 48 lie
within the same safeguard contract; CUDA byte stability across unrelated GPU
architectures is not claimed.

## 6. Variable-order registration

The three variable orders were fixed before the formal terminal A/B. D2
computes all of them and chooses minimum width with lexicographic tie-break;
D4 independently records each fixed order. No permutation was added after a
terminal outcome was observed. All order intervals, selected masks, reasons,
and stage traces are in `range_context_trace` and `horner_stage_trace`.

## 7. Correctness gates

The baseline was `401 passed, 1 skipped in 45.49s`. The final complete suite
was `434 passed, 1 skipped in 73.56s`; the independent CUDA selection was
`3 passed, 28 deselected in 1.83s`.

Coverage includes constant/linear/affine polynomials, odd/even powers,
cross-zero and positive/negative shifted domains, narrow and zero-width boxes,
mixed monomials, missing degrees, degrees 4--12, duplicate exponents,
exact/near cancellation, alternating signs, huge/subnormal coefficients,
batches 1/16/48, multiple state components, independent natural/Horner
containment, dense/sparse zero-width parity, stage reconstruction, all
registered orders, deterministic tie-break, randomized containment sanity,
nonfinite behavior, CPU repetition, CUDA, candidate/support invariance,
default natural behavior, harmonic one-step regression, the original
`t=6.3172908799330765` checkpoint, and the later checkpoint.

Randomized samples are sanity checks only and never tighten a formal
enclosure.

## 8. Frozen terminal formal A/B

Formal runs used clean implementation/runner commit
`a1fb3527bb7c12ce23aa2fb49d66f6380c463c90`.

| Lane | y image | y margin | Runtime (s) | Accepted |
|---|---:|---:|---:|---:|
| D0 natural | `[-1.6369929499718532e-4, 1.7584392650575045e-4]` | `-7.584392650575044e-5` | `0.1023` | no |
| D1 subdivision | `[-1.199995911680722e-4, 1.1599079695982693e-4]` | `-1.99995911680722e-5` | `0.1306` | no |
| D2 Horner registered-best | `[-1.6369929499717632e-4, 1.7584392650574142e-4]` | `-7.584392650574142e-5` | `1.0371` | no |
| D3 subdivision + Horner | `[-8.583871946077335e-5, 1.158599694280285e-4]` | `-1.5859969428028492e-5` | `2.5736` | no |

Horner alone does not recover the negative margin: sound selection keeps
natural on the decisive whole-domain calls. Subdivision and Horner are
nevertheless complementary. In D3 the lower side moves inside the target and
the remaining failure is an upper-side excess of
`1.5859969428028492e-5`.

The largest range-level improvement is `polynomial_truncation` call 44. Its
selected width drops from `0.06121073179318928` to
`0.05186984389118317`. The decisive per-leaf endpoint is the depth-0,
degree-0, top-level multiply-add (stage index 218); the tau-first order is
selected on three leaves and `[u0,u1,tau]` on the fourth. The terminal margin
gain over subdivision is `4.139621740043708e-6`.

The improvement is dependency preservation, not numerical aggregation. The
D2-versus-D0 margin delta is only about `9e-18`, while the per-leaf factorized
width and D3 terminal improvements are many orders larger. Candidate
coefficients and exponent support remain exactly unchanged.

## 9. Fresh horizons and STOP/GO gate

D2 and D3 both reject; the gate is STOP. Consequently the required fresh
sequence was not entered, no fresh policy was preregistered, and `T=6.5`,
`7.5`, and `10` are recorded as `not_run_stop_go_gate_failed`. The highest
validated horizon remains the prior fresh value `6.397083942944808`. There is
no second `T=10` run because no first `T=10` completion exists.

## 10. Runtime interpretation

Runtime is reported only as an internal comparison. D2 and D3 cost more than
natural/subdivision because all registered orders and full stage evidence are
materialized. No GPU or kernel performance tuning was performed, and no
end-to-end speed claim is made.

## 11. Evidence and provenance

Machine-readable artifacts live in
`outputs/vdp_later_terminal_factorized_range/`. They include `summary.csv`,
`attribution.csv/json`, `terminal_ab.csv/json`, context and Horner stage
traces, `fresh_horizons.csv`, environment and test results, the explicit STOP
record, a raw formal-lane package, `manifest.json`, and `SHA256SUMS`. Large
stage files use deterministic gzip (`gzip -dc <file>`); source line counts,
source/stored hashes, and decompression commands are recorded in
`compression_manifest.json`.

Every formal lane records the clean implementation SHA, command, config,
checkpoint/contract/state/candidate hashes, full Picard and remainder rows,
range selection, timings, and all nonfinite/fallback/repair flags. Stage rows
are independently rehashed against their parent range-call records during
packaging.

## 12. Remaining blocker

The frozen local operator is now correctness-gated and partially tighter, but
the original self-map still fails. Because more subdivision leaves already
saturate and registered factorization cannot close the unchanged step, the
remaining blocker is cross-step dependency/carry representation rather than a
local range heuristic, integration-overflow width, runtime cap, or numerical
aggregation artifact.

## 13. Precise next-step recommendation

Keep R4/H1 and study a new, separately preregistered representation that
preserves dependency through normalized-insertion carry and remainder
composition across step boundaries. Begin from the same frozen checkpoint and
trace how carry/remainder structure creates the call-44 polynomial-truncation
payload. Do not add more subdivision leaves, relax the predicate or remainder,
alter the candidate, or resume fresh horizons until that cross-step method
closes the identical frozen step under the unchanged contract.
