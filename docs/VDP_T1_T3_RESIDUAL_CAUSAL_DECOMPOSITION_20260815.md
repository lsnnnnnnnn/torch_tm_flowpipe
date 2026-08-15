# VDP T=1/T=3 residual causal decomposition (2026-08-15)

## Decision

`LOSSLESS_CROSS_OPERATOR_CELL_UNAVAILABLE__TOTAL_CAUSE_OPEN`

The current repository status is therefore
`BOUNDED_SOURCE_MATERIALIZATION_CONTRIBUTION_CONFIRMED__TOTAL_T1_T3_CAUSE_OPEN__G1_TERMINAL_REGRESSION`.
This report does not repeat the superseded `CAUSE_CLOSED` claim.  The correction
and the numerical scope of the older G1 result are recorded in
`VDP_G1_CAUSAL_CLAIM_ERRATUM_20260815.md`.

## Frozen comparison contract

All Torch cells use the exact-decimal outward initialization
`x=[1.1,1.4]`, `y=[2.35,2.45]`, the standard VDP right-hand side, complete O4,
CPU float64 B1, fixed `h=0.01`, target remainder radius `1e-4`, cutoff `1e-10`,
the raw-compat validator, and proactive depth-1 subdivision only for the named
polynomial-truncation context.  Endpoint and segment-tube lower/upper bounds
are separate raw channels.  The machine-readable contract is
`benchmarks/vdp_g2_shared_column_contract_20260815.json`.

## Gate A: five-position four-cell matrix

The five prestates are step 1, step 2, immediately before T=1, immediately
before T=3, and immediately before T=6.32.  The raw matrix contains 20 cells:

| operator / prestate | step 1 | step 2 | before T=1 | before T=3 | before T=6.32 |
|---|---:|---:|---:|---:|---:|
| Flow* / Flow* | native lossless | native lossless | native lossless | native lossless | native lossless |
| Torch / Torch | native lossless | native lossless | native lossless | native lossless | native lossless |
| Torch / Flow* | unavailable | unavailable | unavailable | unavailable | unavailable |
| Flow* / Torch | fail-closed refusal | fail-closed refusal | fail-closed refusal | fail-closed refusal | fail-closed refusal |

The Flow* exporter completed 1,000 accepted steps and emitted 28 selected
pre/post fixtures.  All 28 serialize/deserialize byte-exactly and all 28
native next-step continuations match.  The five Torch checkpoints also
save/load/resave byte-exactly.  No component-box adapter was used.

The cross cells remain unavailable for a substantive reason.  Flow*'s native
state at these positions has a three-state/four-variable TM plus non-empty
`Phi_L/J` queues and distinct MPFR ordinary-remainder objects.  The Torch
prestate is a two-state/two-variable complete-O4 object with its own complete
ledger.  Neither side has a lossless consumer for the other's complete state.
The attempted Flow* consumption of the Torch schema exits with an explicit
dimension refusal; Torch consumption of Flow* is marked unavailable.  Zero
fill, queue deletion, per-component interval projection, and sampled fitting
are forbidden and were not substituted.

One further limit matters: the lossless Flow* native state export does not
expose the raw internal Picard image and a complete owner ledger.  Native
continuation parity therefore proves that the serialized native state is
lossless for Flow* continuation, but it does not make Flow*'s internal stage
owners comparable to Torch's ledger.

## What Gate A can and cannot identify

At accepted step 1 there is no prior accepted-boundary `J/Phi_L` source to
retire or materialize.  Nevertheless, the returned Flow* and Torch polynomial
coefficient/range records already differ.  This directly rules out old-source
materialization as the sole cause of the later T=1 excess.

The first *internally named* Flow* layer that creates the step-1 delta is not
identifiable from the available export.  The first observable layer is the
returned one-step coefficient/range object.  Claiming that polynomial
grouping, remainder refinement, range extraction, or validation is the first
internal cause would require the missing cross-operator cells and stage ledger.

For the same reason, boundary materialization, local operator, their ordering
interaction, and an unexplained residual cannot be assigned unique additive
percentages at T=1 or T=3.  The preregistered forward and reverse interventions
are blocked before execution because two of the four required lossless cells
do not exist.  Their interaction interval is consequently unavailable rather
than silently set to zero.  At the total-cause level, the residual is
`NOT_IDENTIFIABLE_WITHOUT_LOSSLESS_CROSS_OPERATOR_CELLS`.

## Gate B: resolving the G1 ordinary owner

Read-only accounting is attached to the real accepted-boundary transition.  It
keeps complete dense-ledger categories, insertion truncation/cutoff, retired
source terms by generation/component/degree and oldest-current mixing, the
cumulative ordinary parameterization remainder, rebox width, and fresh source
mass separate.  Every recoverable owner has canonical support/payload hashes,
an outward interval, width, and containment witness.  Rebox and natural owner
intervals are explicitly non-additive; their widths are not summed as an exact
partition.

The diagnostic T=6.32 boundary has the following raw width masses:

| owner | width mass |
|---|---:|
| cumulative ordinary parameterization plus already-collapsed history | `2.1933445893242403` |
| fresh complete-ledger affine source | `0.00018633693801359448` |
| recoverable retired old-source polynomial | `0.00017614750445605847` |
| insertion truncation/cutoff owner | `0.000011480296511027764` |
| symmetric rebox additional width | `0.05684893479539377` |

The rebox number overlaps other owners and is not added to the first row.  The
recoverable old source is approximately four orders of magnitude smaller than
the cumulative ordinary mass.  At the earlier selected boundaries the same
pattern evolves from an absent/below-cutoff old owner at step 1 to a dominant
cumulative ordinary owner before T=1, T=3, and T=6.32.  The actual-next-Picard
intervention artifact records both preregistered recoverable transformations:
retaining the retired old-source polynomial and lifting cumulative ordinary
parameterization into a shared source.  Payload controls alter the real
consumer output for live above-cutoff owners; metadata-only controls preserve
it.  Below-cutoff/absent owners are marked not applicable and no artificial
source is inserted.

This evidence motivates exactly two generations for G2: retain the complete
current source polynomial for one additional Picard generation, then retire
only the oldest bank.  It was selected from owner identity and the fixed shape
constraint, not by sweeping generation counts or owner subsets.  The owner
mass also warns that this mechanism is unlikely by itself to remove 10% of the
large Flow*–Torch excess.

## Direct answers to the causal questions

1. **First step:** a real coefficient/range delta is visible in the returned
   one-step objects, before any old accepted-boundary source exists.  The first
   internal local-operator sublayer remains unknown.
2. **T=1 and T=3 split:** no sound numeric boundary/local/interaction split is
   available.  The missing lossless cross cells prevent the preregistered
   forward/reverse counterfactuals.
3. **Residual:** it is not numerically identifiable from the current common
   state contract.  It is not zero and is not contained in a declared rounding
   envelope.
4. **Post-T=6.32 margin:** G1 shows that cumulative ordinary dependency loss is
   the dominant recoverable boundary mass, but no same-prestate Flow* cell
   establishes it as the unique controller.  Local Picard/range/validator and
   scheduler feedback remain coupled candidates in the native lane.

## Evidence classification

- Flow* byte round trips and independent exact rational G2 algebra are
  formal/discrete evidence within their stated schemas.
- CPU float64 containment witnesses are directed numerical evidence for the
  implemented operations, not a universal proof about Flow* equivalence.
- Fixed widths, native horizons, timings, and actual-consumer changes are
  deterministic empirical evidence.
- Sampling is sanity-only and is never used to establish containment.

In plain language: this round proves that G1's source really enters the next
Picard solve and identifies the largest recoverable G1 owner.  It does not
prove a total Flow*–Torch root cause, because the two missing cross-operator
cells make the key counterfactual decomposition impossible.
