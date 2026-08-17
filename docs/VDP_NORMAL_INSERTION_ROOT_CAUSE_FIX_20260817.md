# VDP normal-insertion root-cause result (2026-08-17)

## Verdict

H1 is accepted.  The first reproducible width loss is caused by repeated
interval materialization of the same carried ordinary remainder in the legacy
monomial-by-monomial normal-insertion graph.  A production Horner graph with a
fixed variable order preserves the shared algebraic dependency longer and is
sound at every audited boundary.

The overall numerical target is only partially met.  The candidate clears the
pre-registered H1 causal threshold at T=6.32, has no fixed-step regression,
runs below 2x legacy, and advances the native terminal from
6.397083942944808 to 6.441433080631058.  It does not remove 10% of every
legacy excess at T=1 and T=3, and it does not reach T=10.

Because H1 passed its frozen causal threshold, H2 (step-1 dense Picard raw
remainder) and H3 (reboxing) were not entered.  They are **not adjudicated**,
not silently accepted or rejected.

## Frozen contract and provenance

- Base tip: `e47ce68c61e73fc38f17fab3037d6cfe1877f3fd`.
- Branch: `codex/vdp-normal-insertion-root-cause-fix-20260817`.
- ODE: `x'=y`, `y'=y-x-x^2*y`.
- Initial box: exact decimals `[1.1,1.4] x [2.35,2.45]`.
- Order 4, cutoff `1e-10`, target ordinary-remainder radius `1e-4`.
- Fixed schedule: `h=0.01`; native schedule and validator are unchanged.
- Dense range policy: adaptive subdivision, proactive depth 1 on
  `polynomial_truncation`, at most 4 leaves, split variables `(0,1)`.
- CPU soundness lane: binary64 outward interval operations plus an independent
  exact-rational Bernstein oracle.  CUDA is consistency-only.

The legacy, G1, G2, initialization, scheduler, validator, and default reset
modes were not changed.  The candidate is opt-in as
`normalized_insertion_dependency_preserving`.

## First loss and causal mechanism

The legacy path is `_insert_ctrunc_normal_like_scalar` through
`_compose_term_with_inner`: each outer monomial is composed separately, so an
ordinary remainder in `previous_state.tmv_right` is interval-expanded again
for sibling monomial paths.  The candidate evaluates the same polynomial as a
canonical Horner graph in variable order `(0,1,...)`, sharing coefficient
branches before multiplying by the inserted right map.  Every multiplication
still performs the same order truncation, cutoff, and outward remainder
products.

Gate A replays D, H, D-P, D-one, and H-P from byte-identical serialized inputs.
D and H are the sound production cells; the three subtraction cells are
explicitly counterfactual diagnostics.  The actual next dense Picard/validator
consumer is run for every cell.

- Step 1 to 2 is a negative control: the initial inner ordinary remainder is
  zero, the repeated nonlinear consumption count is 0, and D-H is zero in all
  four consumer widths.
- Step 2 to 3 is the first nonzero boundary.  The carried inner remainder is
  nonzero, the direct graph consumes it on 13 nonlinear paths, and the
  factorized graph performs 10 inserted-map multiplications.  D-H is already
  nonzero: segment x `2.8966162801680184e-11`, segment y
  `7.745848407125777e-10`.
- The 13-versus-10 pattern persists before T=1, T=3, T=6.32, and at the
  terminal pre-state.

The following values are local changes in the *actual next consumer* width;
positive D-H means the factorized consumer is narrower.

| boundary | D-(D-P) seg x/y | H-(H-P) seg x/y | D-H seg x/y |
|---|---:|---:|---:|
| step 1->2 | `0 / 0` | `0 / 0` | `0 / 0` |
| step 2->3 | `2.671608e-6 / 3.410615e-5` | `2.671506e-6 / 3.410477e-5` | `2.896616e-11 / 7.745848e-10` |
| before T=1 | `1.499531e-3 / 1.358570e-3` | `1.499525e-3 / 1.358143e-3` | `5.554433e-9 / 4.184874e-7` |
| before T=3 | `1.349307e-2 / 2.312507e-2` | `1.349292e-2 / 2.311035e-2` | `1.475989e-7 / 1.462323e-5` |
| before T=6.32 | `7.386443e-1 / 1.457065` | `7.386358e-1 / 1.456219` | `8.494403e-6 / 8.456464e-4` |
| terminal pre | `7.532981e-1 / 1.512675` | `7.532892e-1 / 1.511796` | `8.844304e-6 / 8.789526e-4` |

All six boundaries have identical per-cell input hashes.  For every D and H
component, the production remainder contains the exact Bernstein enclosure of
the exact binary64 polynomial residual.  The oracle introduces one exact
remainder indeterminate per inner Taylor model, composes with `Fraction`
coefficients, subtracts the represented polynomial, and bounds the residual on
the full augmented box.  Thus it checks the shared-remainder semantics without
using either production interval evaluation or Flow* at runtime.

## Fresh fixed-step scientific matrix

`removed` is `(legacy-candidate)/(legacy-Flow*)`.  Endpoint and full segment
tube widths are shown independently.

| T | channel | Flow* | legacy | candidate | removed |
|---:|---|---:|---:|---:|---:|
| 1 | endpoint x | 0.0795178281 | 0.0879559235 | 0.0879439409 | 0.1420% |
| 1 | endpoint y | 0.1115769156 | 0.1142921745 | 0.1142770205 | 0.5581% |
| 1 | segment x | 0.0837525716 | 0.0922215177 | 0.0922095309 | 0.1415% |
| 1 | segment y | 0.1196669930 | 0.1285652386 | 0.1285492647 | 0.1795% |
| 3 | endpoint x | 0.1385053267 | 0.1872595897 | 0.1865686202 | 1.4172% |
| 3 | endpoint y | 0.1088516715 | 0.1558642556 | 0.1541143565 | 3.7222% |
| 3 | segment x | 0.1639208754 | 0.2127350396 | 0.2120438511 | 1.4160% |
| 3 | segment y | 0.1256837926 | 0.1727677949 | 0.1710170720 | 3.7183% |
| 6.32 | endpoint x | 0.1530755556 | 0.9165121029 | 0.7919255932 | 16.3192% |
| 6.32 | endpoint y | 0.1222956280 | 1.5898587283 | 1.2745154469 | 21.4875% |
| 6.32 | segment x | 0.1783273000 | 0.9420414425 | 0.8174173573 | 16.3182% |
| 6.32 | segment y | 0.1398213090 | 1.6080698025 | 1.2925904925 | 21.4868% |

Therefore:

- H1 causal threshold (at least one T=3 or T=6.32 component removes 10%):
  **pass**; all four T=6.32 channels pass.
- T=1 and T=3 all-channel 10% production target: **fail**.
- T=6.32 no-regression target: **pass** in all four channels.

## Native termination and performance

| mode | completed horizon | accepted/rejected attempts | CPU runtime |
|---|---:|---:|---:|
| legacy | 6.397083942944808 | 307 / 48 | 132.784 s |
| candidate | 6.441433080631058 | 293 / 46 | 168.475 s |

Both stop with `minimum_step_reached` before `h_min=0.002`.  The candidate
passes the frozen native non-regression gate but fails the T=10 stretch goal.

Candidate/legacy CPU runtime ratios are 1.3068 (T=1), 1.2753 (T=3), 1.3495
(T=6.32), and 1.2688 (native); the 2x gate passes.  On a Tesla V100-SXM2-16GB,
the T=0.1 legacy and candidate trajectories each match their CPU counterpart
exactly in status, accepted/rejected counts, horizon, and all four widths
(maximum absolute width difference 0).  CUDA runtimes were 11.260 s and
14.210 s respectively, but these are reported only as measured values: CUDA
is neither a directed-rounding soundness lane nor a speedup claim.

## Tests, evidence, and replay

- Candidate-focused suite: `11 passed in 5.47s`.
- Full repository suite: `768 passed, 2 skipped in 338.68s`.
- Gate A v2: 6/6 byte-identical checkpoints and all D/H exact-oracle checks
  passed in 294.57 s.

Reproduction entry points:

```bash
conda run -n py11 python experiments/audit_vdp_normal_insertion_root_cause_20260817.py \
  --output-dir outputs/vdp_normal_insertion_root_cause_20260817/gate_a_v2
conda run -n py11 python experiments/run_vdp_normal_insertion_matrix_20260817.py \
  --output-root outputs/vdp_normal_insertion_root_cause_20260817/scientific_matrix \
  --include-cuda
conda run -n py11 pytest -q tests/test_dependency_preserving_insertion.py
conda run -n py11 pytest -q
```

The tracked evidence package contains the six raw Gate-A boundary payloads,
safe replay checkpoints, fresh-run summaries and traces, CPU/V100 consistency
rows, test XML, environment/provenance, a complete manifest, and SHA-256 sums.

## Frozen decision-tree accounting

| hypothesis | status | evidence |
|---|---|---|
| H1 repeated ordinary remainder across insertion paths | **accepted** | first nonzero step 2->3, exact-oracle sound Gate A, 16.3%-21.5% T=6.32 excess removal |
| H2 step-1 dense Picard raw remainder | **not entered / not adjudicated** | frozen tree stops after H1 acceptance |
| H3 reboxing | **not entered / not adjudicated** | frozen tree stops after H1 acceptance |

The accepted root cause is therefore narrower than a claim of complete Flow*
parity: repeated normal-insertion dependency loss is real and materially
causal late in the prefix, but it is insufficient by itself to close the
T=1/T=3 all-channel target or T=10.
