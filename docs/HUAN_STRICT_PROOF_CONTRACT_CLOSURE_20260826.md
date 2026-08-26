# Huan strict proof-contract closure — 2026-08-26

Primary status: `HUAN_PROOF_CONTRACT_CLOSED__VDP_CONTRACT_NOT_PORTABLE`

Repaired engine: `b0ff55745d69205f3afb4dc8077b9ac1310bfff3` on
`codex/strict-proof-contract-closure-20260826`.

This report covers the polynomial plant engine only. It does not extend to
CROWN, controllers, ONNX, transcendental-library assumptions, coupling, or a
throughput claim. Huan parity remains a reproduction mode with the Flow* point
coefficient trust model; the unconditional polynomial claim is limited to
strict mode under the enforced finite, no-FTZ, binary64, and reduction-length
hypotheses.

## Result

D1–D6 pass at the repaired Huan SHA. The separate scientific gate imports that
source and reruns the operators/tests. The package verifier only validates
hashes, schemas, provenance, invocation records, and cross-file consistency;
it does not rerun or certify the science.

| Gate | Final result | Main evidence |
|---|---:|---|
| D1 elementwise / no-FTZ / non-finite | PASS | CPU and CUDA 7/7 exact cases; production startup assertion; finite-certificate rejection |
| D2 reductions | PASS | CPU 846 host oracle + 564 actual CPU; CUDA 846 host oracle + 423 Torch CUDA + 141 custom CUDA invocations |
| D3 dense/sparse/support | PASS | 84 passed, 2 skipped |
| D4 chunk/lane | PASS | 55 passed, 1 skipped; B=1 embedded in B=2 and chunks 1,2,3,5,7 bitwise in the declared path |
| D5 refinement/cache | PASS | 66 passed; behavioral ledger plus generation/owner tamper rejection |
| D6 strict roundoff | PASS | 50 passed; independent Fraction witness and repaired replay |

The authoritative result is
`outputs/huan_proof_closure/phase_d_gate_v2.json`; the scientific runner and
package verifier are intentionally different programs.

## D2 audit correction

The previous 987/987 wording was too strong because six schedules were Python
host reductions even when the enclosing process selected CUDA. Schema v2 tags
every row with execution backend, actual device, kernel path, and observed
invocation. Host sequential/pairwise/chunk/permuted/FMA schedules remain an
any-order mathematical oracle, but are never counted as CUDA execution.

The actual routes now include `torch.dot`, `torch.sum(a*b)`, the engine point
einsum path, and the production interval-dot path. CUDA evidence includes an
explicit per-call custom-kernel counter; availability alone is rejected by both
the corrected old verifier and the new closure verifier.

## D6 classification and repair

D6 was a concrete under-enclosure bug, not merely an incomplete map. At clean
base `d5f0b68...`, the minimized 2x2, three-Phi-push witness has exact second
component

```text
-4503599627370497 / 38685626227668133590597632
~= -1.1641532182693484e-10
```

while the legacy point-Phi reconstruction encloses approximately
`[-2.3283064365386994e-10, -2.3283064365386942e-10]` and misses the exact
value. The repaired strict interval-Phi result contains it. Operands,
minimality, exact rationals, source SHAs, and replay commands are in the two D6
witness JSON files.

Symbolic `Phi` is repaired by preserving an interval enclosure beside the
parity point matrix. Strict propagation forms the linear-factor product with
outward interval arithmetic, propagates all live `Phi` entries with outward
interval matrix products, and reconstructs `Phi*J` plus the queue sum with
outward interval reductions. Reciprocal scale factors carry `scalars_iv` so a
later queue product cannot treat rounded scaling as exact.

Retained nonlinear monomial coefficients are repaired without replacing the
normal point path. Dense and sparse multiplication also compute an independent
outward interval coefficient convolution, subtract the retained point
coefficient, range that error polynomial on the actual support, and add it once
to that image node's ordinary remainder. Later nonlinear levels propagate it
through normal Taylor-model remainder algebra. Contraction GEMMs, SR linear
images/final coefficient additions, endpoint substitution, and reciprocal
preconditioning have analogous source-specific bounds. The complete dataflow
is in `strict_roundoff_accounting_graph.md`; the machine ledger is
`strict_roundoff_sources.csv`.

## no-FTZ and non-finite closure

Public reach initialization now asserts gradual underflow on CPU and, when
CUDA is selected, on the Torch CUDA path and the active custom interval kernel.
Representative smallest-subnormal identities are checked before a certificate
can be accepted. A tampered probe raises `FloatingPointError`.

NaN/Inf is handled fail closed. Validation and refinement accumulate a generic
per-lane non-finite mask over polynomial, remainder, cache, tail, and proposal
tensors; affected lanes freeze with the historical arithmetic/domain failure
status. `contains` is false on non-finite endpoints and `assert_valid` rejects
infinity as a finite certificate. Extended intervals may exist transiently,
but are never promoted to finite validated flowpipes.

## D5 refinement semantics

Flow* sequential partial commits are retained and are sound under the observed
implementation contract. Components are tested in order; proposals before the
first failing component commit, the first failing and later components remain
unchanged, and the resulting mixed vector is the certified owner for the next
replay. The optional callback records initial self-map input/image, attempted
step, component subset margins, proposal/commit mask, stop-ratio values,
non-finite mask, cache/tail IDs, generations, and final remainder owner.

The fixed cache/tail tensors describe polynomial ranges and truncation tails;
the current remainder is supplied explicitly on every replay. A generation and
full owner-vector token binds proposals to the current mixed vector. Stale
generation or stale owner fails closed. When the callback is disabled no trace
clone/generation path is taken, so production semantics are unchanged.

## Historical dirty sources

The bounded search inspected stashes, reflogs, unreachable objects, worktrees,
sibling repositories, patches, manifests, archives, histories, logs, and build
copies: 154,683 files and 457 candidates. None reconstructs any of the three
recorded dirty source states exactly. Final result:
`HISTORICAL_DIRTY_PATCHES_NOT_FOUND_AFTER_BOUNDED_SEARCH`.

Therefore the historical 450-run scientific claims remain unreproducible as
exact source executions. This does not weaken the clean-source D1–D6 result,
but those historical numbers must not be attributed to the repaired SHA.

## Answers required by the goal

1. D6 was a demonstrated symbolic-Phi under-enclosure bug.
2. Interval Phi/scalars propagate every matrix product and reconstruction;
   retained dense/sparse coefficient error is ranged on its true support and
   inserted once into ordinary remainder.
3. Production enforces gradual underflow and rejects non-finite certificates.
4. Sequential partial commits are observable, owner-bound, and certified; stale
   generation/owner evidence is rejected.
5. Exact frozen Huan parity native T=10: not run; contract not portable.
6. Exact frozen Huan strict native T=10: not run; contract not portable.
7. Flow*/Torch first divergence: not adjudicated after the mandatory stop.
8. Torch C2 terminal `y`-upper cause: not adjudicated by Huan in this round.
9. The dirty-source 450-run claims remain unreproducible.
10. No throughput publication is authorized. A later round first needs a
    reviewed implementation of adaptive step plus symbolic queue 100 under the
    unchanged frozen settings, followed by a new D gate and Phase E.

