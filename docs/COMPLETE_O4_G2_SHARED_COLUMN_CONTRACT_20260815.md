# Complete-O4 G2 two-generation shared-column contract — 2026-08-15

Candidate (the only G2 evaluated in this study):

`normalized_insertion_bounded_shared_source_o4_g2`

The machine-readable contract is
`benchmarks/vdp_g2_shared_column_contract_20260815.json`. ODE, exact-decimal
outward initial set, complete O4 basis, cutoff, target remainder, range method,
validator, endpoint/tube semantics, and adaptive controller are frozen. The
default legacy mode is unchanged; the exact-decimal contract is selected
explicitly by the 2026-08-15 experiment runner.

## Fixed state shape and meaning

For state dimension `d`, every accepted boundary has exactly `3d` variables:

- `u[0:d]`: normalized base variables;
- `z_old[d:2d]`: the retained previous-generation source bank;
- `z_new[2d:3d]`: the fresh complete-ledger source bank.

VDP therefore always uses six variables. No count depends on horizon, term
magnitude, validation outcome, or observed width. A source column has one ID
and may occur in both state components and in every linear/nonlinear monomial;
it is not cloned into independent occurrence intervals.

At the first boundary, the retained bank is inactive and the complete validated
remainder ledger is outward-lifted into `d` fresh affine sources. After one
Picard generation, each fresh source has generally become a complete O4
polynomial shared across x and y. That whole current-generation polynomial is
retained at the next boundary.

## Accepted-boundary transition

1. Remove the physical endpoint constants and compose normalized base
   variables through the prior right map. Substitute both source banks by
   identity. Canonical polynomial construction merges equal full exponents.
2. Partition terms containing any oldest-bank variable. This includes
   oldest×current mixed terms. Evaluate their complete, already-merged
   polynomial outward exactly once per component and add that interval to the
   ordinary ledger.
3. Partition the survivors into source-free and current-source-bearing terms.
   Rebox only the source-free polynomial plus ordinary remainder. Rename the
   complete current-bearing polynomial from the fresh bank into retained slots.
4. Outward-lift the unchanged complete dense validated ledger into `d` new
   fresh affine sources, adding only its midpoint to the physical center.
5. Publish `center + scale*u + retained_polynomial(u,z_old) + rho*z_new` as the
   actual next dense Picard input. Neither retained nor fresh source mass is
   also present in base scale or ordinary remainder.
6. Commit IDs, generation, coefficients, owner ledger, hashes, and counters
   atomically. Any contract violation fails closed; there is no fallback.

The source-free right map, retained polynomial, and fresh affine bank are stored
as distinct payloads. Checkpoint schema v4 serializes all three using exact
binary64 hexadecimal coefficients and canonical hashes.

## Owner accounting and interactions

The transition records every complete dense-ledger category; insertion
truncation/cutoff; retired terms by component, source generation, total-degree
class, and oldest×current status; VDP nonlinear `x^2y` path classification;
rebox symmetry inflation; retained mass; fresh mass; canonical support/payload
hashes; outward intervals; and containment witnesses.

Retired owner subintervals can overlap because dependencies and nonlinear
interactions are real. They are saved as intervention owners but are not
claimed to add exactly to the total. The authoritative total retirement is the
single outward evaluation of all oldest-bearing canonical terms.

## Atomicity and independent oracle

Rejected candidates never call the boundary transition. The accepted prestate
object, source fingerprint, current reset polynomial, and retained payload are
unchanged across a retry. Only an accepted step advances the generation.

`experiments/independent_g2_exact_oracle.py` is standard-library-only. It does
not import project polynomial, Taylor-model, dense Picard, source-ledger, or
interval code. With exact rational arithmetic it independently checks canonical
merge, shared affine substitution, the cubic `x^2y` expansion, two-bank
rotation, oldest×current retirement, degree-4 truncation ownership, retry
immutability, and every exported black-box coefficient. Its containment check
uses an exact rational natural interval enclosure on `[-1,1]^n`; sampling is
not used as proof.

CPU float64 B1 is the scientific lane. CUDA may be compared for implementation
consistency and synchronized end-to-end performance only; it is never labeled
formal directed rounding.
