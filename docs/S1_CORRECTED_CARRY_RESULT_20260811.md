# S1 corrected carry result

> This remains the authoritative S1 numerical result. Claims about old
> packager verification and the historical b5ba fresh clone are qualified by
> [the evidence-integrity corrections](EVIDENCE_INTEGRITY_CORRECTIONS_20260811.md).

Date: 2026-08-11

## Implemented candidate

Exactly one production candidate was implemented:

```text
normalized_insertion_structured_total_delta_k16
```

It uses `Delta = R_o + sum(Phi_k J_k)` and the complete retained-polynomial
contract
`P(Q+Delta)-P(Q) subseteq A Delta + N_total`. The ordinary update owns
`A R_o`, `N_total`, ineligible sources, eligible centers, and proved padding;
each live column owns `A Phi_k`; eligible symmetric sources enter new unique
columns. No dominance guard, K change, target change, cutoff change, smaller
`h_min`, or validator change is present.

## Soundness gates

The candidate tests cover exact/Fraction affine through quartic fixtures,
asymmetric intervals, zero structured input, zero-scale fail-closed,
cancellation, duplicate exponents, endpoint and tube publication,
ordinary-only, structured-only, ordinary×structured interactions, multiple
live columns, eviction, atomic commit/rejection, and checkpoint roundtrip.

On accepted VDP boundaries, machine gates check that the raw Picard target is
unchanged, the typed ledger contains the raw-compatible image, the candidate
total contains the canonical target, source ownership is unique, endpoint and
tube publications are complete, the endpoint is in the tube, and the total
normalized right map stays in `[-1,1]`. The candidate has no fallback branch;
it either commits the complete transition or fails closed without state
mutation.

## Corrected frozen accepted prefix

The candidate directly replays every historical accepted `h` with
`h_min=h_max=h`. Decisions on the historical larger attempted steps are
recorded separately and discarded. The result is:

```text
accepted fixed steps: 307 / 307
candidate gate failures: 0
attempted-diagnostic state mutations: 0
scheduler divergences: 0
boundary-307 checkpoint byte-stable: true
checkpoint SHA256: f4a75682f00e38fa9916b3c9dd6e727e5cb9e1257b598587772e1094b0518cd1
```

Thus the corrected accepted prefix reaches the historical terminal prestate.
This is a safeguarded/conditional prefix result, not an end-to-end formal
claim. See
[`frozen_prefix.csv`](../outputs/s1_boundary164_causal_guarded_carry_20260811/20260811T033447Z/frozen_prefix.csv).

## Historical terminal gate

The frozen terminal contract remains:

```text
t = 6.397083942944808
h = 0.003623635847674574
order = 4
target = 1e-4
cutoff = 1e-10
h_min = 0.002
validator = flowstar_raw_remainder_compat
```

T0 uses the native candidate prestate. T1 materializes that exact represented
set into an ordinary-only carrier without reboxing. T2 uses the canonical
historical L0 prestate. T0/T1 have equal center, scale, right polynomial, and
materialized-total hex. All three controls reject:

| control | x margin | y margin | rejections |
|---|---:|---:|---:|
| T0 native total-delta | `+9.963763341523255e-5` | `-1.9999591170254726e-5` | 1 |
| T1 ordinary-only same set | `+9.963763341523255e-5` | `-1.9999591170254726e-5` | 1 |
| T2 historical L0 | `+9.963763341523255e-5` | `-1.99995911680722e-5` | 1 |

The proposed half-step is below the unchanged `h_min`, so no returned state is
committed. The terminal result is
`CORRECTED_S1_REACHES_TERMINAL_BUT_DOES_NOT_CLOSE_IT`.

## Authorization and registry

| gate | result |
|---|---|
| candidate implemented | yes |
| unique candidate name | `normalized_insertion_structured_total_delta_k16` |
| frozen accepted prefix | 307/307 |
| terminal gate | rejected |
| fresh horizon authorized | no; `not_run_after_stop` |
| +0.5 promotion | not run, therefore false |
| second system authorized | no; `not_run_after_stop` |

The single primary outcome is:

```text
S1_REACHES_TERMINAL_BUT_DOES_NOT_CLOSE_IT
```

Independent claim fields are retained in
[`claim_registry.csv`](../outputs/s1_boundary164_causal_guarded_carry_20260811/20260811T033447Z/claim_registry.csv);
terminal failure is not used to infer primitive eligibility, finiteness, or
mathematical-contract knowledge.

## Next step

End S1 promotion under this frozen contract. Any return to fixed-support
representation research must be a separately authorized goal; do not extend
this run with K32, smaller `h_min`, a wider target, fresh-horizon probing, or a
second-system experiment.
