# Handoff: S1 boundary-164 causal guarded carry

> Evidence-integrity qualification (2026-08-11): the fresh-clone and test
> statements below were written by a packager that did not retain the raw
> command bundle needed to derive them independently. The b5ba clone is
> historical, not verification of the later 7b880d0 branch tip. See
> [the correction register](docs/EVIDENCE_INTEGRITY_CORRECTIONS_20260811.md).

Date: 2026-08-11

## Outcome

Primary outcome: `S1_REACHES_TERMINAL_BUT_DOES_NOT_CLOSE_IT`.

The causal ladder isolates the current post-hoc polynomial image decomposition
as the first inflation source. C1/C2 are bit-exact to L0; consequential
coefficient/scale drift begins at boundary 5, physical-hull and margin drift at
boundary/attempt 8, and outward renormalization at boundary 12. K16 eviction
is not the primary cause. At boundary 164, the L0→L1 y-margin loss is
`-1.1930523868969271e-5` and is entirely a scale contribution under the fixed
same-input validator projection.

Outcome B authorized the only new carry,
`normalized_insertion_structured_total_delta_k16`. It passes the corrected
307/307 fixed accepted-step prefix and saves a byte-stable boundary-307
checkpoint with SHA
`f4a75682f00e38fa9916b3c9dd6e727e5cb9e1257b598587772e1094b0518cd1`.
The unchanged historical terminal step still rejects with T0 y margin
`-1.9999591170254726e-5`. Fresh horizons and the second system are therefore
`not_run_after_stop`.

## Delivery

- Branch: `codex/s1-boundary164-causal-guarded-carry-20260811`
- Start SHA: `8683183e48b7795d13edbdc9a5910fba9d21d16c`
- Evidence run:
  `outputs/s1_boundary164_causal_guarded_carry_20260811/20260811T033447Z`
- Phase-5 choice: Outcome B, `S1_POSTHOC_IMAGE_INTRINSIC_INFLATION`
- Full-h boundary-164 y margins: L0 `+8.058292550874906e-6`,
  L1 `-3.872231318094365e-6`, L2 `-3.773875528686747e-6`
- Corrected frozen accepted prefix: 307/307
- Terminal gate: rejected
- Fresh/+0.5/second system: not authorized
- Final regression: 572 passed, 2 skipped in 270.30 s
- Evidence checksums: 234 repository-root-relative entries
- Historical selected-test clone at
  `b5ba3200901e331f01343c7d05608a1d542dbb8c`; this is not a final-HEAD
  verification claim, and the old package does not retain the raw command
  bundle required for independent derivation
- Primitive formal eligibility: true for the bounded CPU outward image
- Prefix formal eligibility: false
- Prefix class: `safeguarded_binary64_interval_shell`,
  `conditional_on_retained_coefficient_arithmetic`
- Performance and cross-tool ranking eligibility: false

## Unique next action

End S1 promotion under this frozen contract. Return to fixed-support
representation research only under a separately authorized goal; do not tune
K, target, cutoff, validator, or `h_min`, and do not append fresh-horizon or
second-system runs to this evidence package.
