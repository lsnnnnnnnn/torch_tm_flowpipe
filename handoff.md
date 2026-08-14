# Flow*–Torch T=1/T=3 bounded-source carry handoff — 2026-08-14

Current scientific decision:

`T1_T3_WIDTH_CAUSE_CLOSED__EARLY_GAP_IMPROVED__TERMINAL_STILL_OPEN`

This experiment refines the carry diagnosis without superseding the broader
canonical compatibility outcomes:
`FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_ONLY`,
`DIFFREACH_TORCH_DR7_FULL_HORIZON_DIVERGED`,
`CARRY_MISSING_SYMBOLIC_SEMANTICS`, and `NO_FIX_AUTHORIZED`.

## Result

The T=1 gap is the accumulated cost of repeatedly materializing shared
parameterization uncertainty at accepted boundaries. By T=3, the same loss has
been applied hundreds of times. Near T=6.32, the wider prestate is fed through
the quadratic/cubic uncertainty terms of `x^2 y`, producing nonlinear positive
feedback rather than constant linear growth.

The authoritative raw ledger reproduces legacy excess ranges
`0.002715–0.008898` at T=1, `0.047013–0.048814` at T=3, and
`0.763437–1.468248` at T=6.32. Direct raw Flow* minima are all greater than
0.0086; the apparent graph zero remains a projection artifact.

The preregistered production candidate is
`normalized_insertion_bounded_source_ledger_o4_g1`. It has exactly one live
source per state component for one Picard generation, a fixed `2d` boundary
shape, immutable retry state, and no fallback. All 13 independent micro-oracles
and actual payload/metadata consumer tests pass. The first causally active field
is the affine source coefficient in the next dense Picard input.

Fresh fixed-schedule reductions versus legacy are:

- T=1: `1.26e-5–2.38e-5` across the four endpoint/segment channels;
- T=3: `9.92e-5–1.11e-4`;
- T=6.32: `0.00199–0.00450`.

No ratio crossing above 1.1/1.5/2/5 moves at 0.01 resolution. The dominant
post-boundary mass at 6.32 remains ordinary (`2.1933445893`) rather than fresh
structured (`0.00018633694`), explaining the limited effect.

Native G1 completes T=1/T=3/T=6 but accepts only 300 segments through
`6.382737816137232`; legacy accepts 307 through `6.397083942944808`. Fresh
T=7.5/T=10 requests repeat those stops. The candidate therefore improves the
early curve but does not close the terminal and is slightly worse in native
horizon.

The V100 provides no speedup: the synchronized lift kernel is slower for
B1/B8/B64/B256/B512, and the full T=0.1 candidate is 16.38 s on V100 versus
5.47 s on CPU. CPU float64 B1 remains authoritative.

Gate A's exact-decimal outward initialization is also complete in both audit
drivers and consumed by the real step path. Its 1e-16 correction is not the
long-horizon cause.

Detailed contract and report:

- `docs/COMPLETE_O4_BOUNDED_SOURCE_LEDGER_CONTRACT_20260814.md`
- `docs/VDP_T1_T3_WIDTH_CAUSAL_REPORT_20260814.md`

Evidence package:

`outputs/vdp_t1_t3_width_causal_source_ledger_20260814/20260814T120000Z`

Scientific commit:
`8ac2962bf691dd81ae5d06a9ea146bb011b7ec42`.

That exact SHA was pushed, cloned again from GitHub, checked out detached, and
accepted independently: package verifier PASS, `compileall` PASS, focused
41/41, full pytest 745 passed plus 2 skipped, and clean clone status. The
attestation is the immediate child commit containing
`07_acceptance/scientific_acceptance.json`; no scientific source is changed by
that commit. The scientific result above is frozen.

---

# Prior Flow*–Torch step-1 oracle handoff — 2026-08-14

Current decision: fail closed at Gate D; no candidate is authorized.

The canonical broader outcomes remain
`FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_ONLY`,
`DIFFREACH_TORCH_DR7_FULL_HORIZON_DIVERGED`,
`CARRY_MISSING_SYMBOLIC_SEMANTICS`, and `NO_FIX_AUTHORIZED`. This step-1
negative result refines their mechanism evidence without superseding them.

## Plain-language result

1. The first decision-relevant failure is the normalized initial-TM encoding,
   before Picard, remainder refinement, or range extraction. Both point-
   coefficient runtimes omit endpoints of the exact-rational input required by
   the launch contract.
2. Torch's narrower step-1 endpoint is nevertheless formally sound. An
   independent four-corner monotonicity proof with exact degree-100 Taylor
   series and a Cauchy tail encloses the true endpoint inside both tools.
3. The prior Horner `+4` steps remain diagnostic-only.
4. L1 is `NOT_AUTHORIZED`; L2/L3 are `NOT_RUN`. Gate D's under-enclosure stop
   rule prevents the P/R/X swap payloads from being propagated.
5. No sound candidate was run, so no candidate horizon exists and T10 was not
   reached by a sound candidate.
6. This run advances the original objective by closing the common mathematical
   contract, producing complete actual-path ledgers, proving the exact fourth
   Picard images equal, and formally explaining the narrower Torch endpoint.
7. The minimum blocker is an outward, exact-set-preserving normalized affine
   input encoding on both engines. Gates C/D must then be rerun before any
   operator candidate.

## Current status table

```text
COMMON_STEP1_MATHEMATICAL_INPUT_CLOSED
INDEPENDENT_STEP1_SOUNDNESS_ORACLE_INCOMPLETE
LOCAL_OPERATOR_SOURCE_DELTA_OPEN
NOT_AUTHORIZED
NOT_RUN
NOT_RUN
LEGACY_DEFAULT_UNCHANGED
NOT_REACHED
```

The first under-enclosure gaps are exact, not sampled:

```text
Flow* x: lower/upper each 1/11258999068426240
Flow* y: lower 3/5629499534213120
Torch x: lower/upper each 11/180143985094819840
Torch y: lower 219/720575940379279360
```

## Publication metadata

- branch: `codex/step1-stage-oracle-sound-carry-candidate-20260813`
- scientific SHA: `57a4763287b3e9a38866cb55b09fce27dd6798b7`
- detached fresh clone: PASS from the GitHub origin; compileall, 23 focused
  tests, 724 full tests with 2 skipped, package verification, and clean
  porcelain status all passed
- scientific tree hashes: `src`
  `9f9b6d6832d27849cd5ba3ac96487d2a5a673a03`, `experiments`
  `443815b2fd8c0e1fffe067b40d237ad2365362e1`, `tests`
  `ed4d275f0303eced414302661d0364151c8c9c6d`
- attestation commit: `19867c9108b23c00e179532d2108de0c9b0428a3`;
  its diff from the scientific SHA changes no `src`, `experiments`, or `tests`
- publication tip: remote branch ref
  `refs/heads/codex/step1-stage-oracle-sound-carry-candidate-20260813`;
  the exact tip is checked after every push and reported in the terminal
  summary because a commit cannot contain its own SHA
- publication semantics:
  `scientific_sha_fresh_clone_verified=true`,
  `attestation_tip_contains_no_scientific_tree_changes=true`,
  `final_tip_fresh_clone_verified=false`
- evidence package:
  `outputs/flowstar_torch_step1_stage_oracle_sound_carry_20260813/20260814T014356Z`

The prior publication semantics are corrected in
`docs/EVIDENCE_LABEL_AND_PUBLICATION_SEMANTICS_20260813.md`.

Detailed reports:

- `docs/STEP1_COMMON_OPERATOR_CONTRACT_20260813.md`
- `docs/FLOWSTAR_TORCH_STAGE_LEDGER_20260813.md`
- `docs/INDEPENDENT_STEP1_SOUNDNESS_ORACLE_20260813.md`
- `docs/LOCAL_OPERATOR_CAUSAL_CLOSURE_20260813.md`
- `docs/SOUND_CANDIDATE_DECISION_20260813.md`

---

# Prior Flow*–Torch causal mechanism closure handoff — 2026-08-13

Final scientific decision: `NO_FIX_AUTHORIZED`.

Repository-level compatibility context (historical suites remain superseded
where their own banners say so): `FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_ONLY`,
`DIFFREACH_TORCH_DR7_FULL_HORIZON_DIVERGED`, and
`CARRY_MISSING_SYMBOLIC_SEMANTICS` remain the canonical broader outcomes; this
closure adds evidence but does not rewrite those orthogonal contracts.

## Plain-language result

1. The copied probe is not the stock runtime entry, but it is now proven
   equivalent to the actual `ode.reach` path for this pinned VDP contract:
   1000 clean/instrumented/copied steps agree, including exact retained state
   and queue transitions.
2. The graph's apparent “zero” is a positive coordinate-projection minimum.
   Every Flow* minimum is greater than `0.0086`, not numerically near zero.
3. Step 1 already differs in 23 of 31 returned Picard coefficients.  No old
   `J/Phi_L` source has crossed a boundary then, so queue-only attribution is
   false.
4. Flow* Q1/Q2/Q10/Q100 reach 620/640/685/1000 steps.  In Torch, Horner changes
   widths from step 3 and moves 632→636; the diagnostic queue changes segment
   widths from step 2 but not endpoints, scales, or horizon.
5. The earliest source candidate is local Picard construction/grouping, then
   remainder refinement and endpoint/tube extraction.  No unique
   decision-relevant source line is causally closed because two full-state
   cross-operator replay cells are unavailable.
6. The lossless bridge is real: all 24 Flow* import/export and one-step
   continuations are exact, and Torch→Flow*→schema is byte exact.  Flow* and
   Torch still have incompatible full operator state dimensions/queues.
7. The independent outward source-ledger oracle was not authorized and remains
   `SOURCE_LEDGER_ORACLE_INCOMPLETE`.
8. No production candidate was implemented.  L1/L2/L3 are all not run; the new
   Horner/queue combinations are feature-gated diagnostics and the default is
   unchanged.
9. Canonical byte equality and exact-rational fixtures are formal/discrete;
   MPFR direction and dyadic conversion are directed-numerical; widths and
   horizons are deterministic empirical evidence; factorial modes are
   diagnostic; unique source attribution is unresolved.
10. The only next action is to define a shared full operator sub-contract that
    preserves `t`, `Phi_L/J`, terms, and separate remainder ledgers, then run
    the missing Torch-on-Flow* and Flow*-on-Torch same-prestate cells.

## Selected outcomes

- `BASELINE_CONCLUSIONS_REPRODUCED`
- `FLOWSTAR_WIDTH_MINIMUM_POSITIVE_NOT_NUMERICALLY_NEAR_ZERO`
- `STOCK_COPIED_PROBE_EQUIVALENCE_CLOSED`
- `CAUSAL_FACTOR_SPLIT_PARTIAL`
- `SAME_PRESTATE_LOSSLESS_BRIDGE_AVAILABLE`
- `SOURCE_MECHANISM_CANDIDATES_LOCALIZED_CAUSAL_SPLIT_OPEN`
- `SOURCE_LEDGER_ORACLE_INCOMPLETE`
- `NO_FIX_AUTHORIZED`

## Publication metadata

- branch: `codex/flowstar-torch-causal-mechanism-closure-20260813`
- start SHA: `cdda27bf2c0e7f72e135edbfd2b2ba10a8c5f96d`
- scientific tested SHA: `a8653a7d9ea6f54b1450da6bee9af0e2a5a19695`
- publication tip: remote ref
  `refs/heads/codex/flowstar-torch-causal-mechanism-closure-20260813`;
  its exact self-referential commit SHA is resolved by the required post-push
  `git ls-remote` check and reported in the terminal summary (a commit cannot
  contain its own hash)
- Flow* SHA: `b85a3211748cb77b736fe4ad42ee02d8d2b81148`
- Flow* binary SHA256:
  - clean `libflowstar.a`: `a23109e2b40bbcbe80a242c33f0a23f3473c5fbc8ccef2de5d0e28874535fc36`
  - instrumented `libflowstar.a`: `accfdf9bcffaa73d83dbef76a04d7d1583fb18afba816d752e1d358f4af1675d`
  - stock driver: `da512eda8035fd60ce2983e15835dc3cda08f170a829f44d24bb045453657820`
  - instrumented driver: `ceaf5e9df402507ac4ffce88001c6b32f9759219cab4d5f75359b4508f2d0394`
  - copied probe: `2d64456d6186f97519eb432fd168e2267e00a5e0ab787452ab2ddf4434b77ec2`
  - lossless bridge: `4bd2fec16dbf752ee1b6b63993b4d9c72e0ba353c82339e5b43a6a618945d320`
- evidence package:
  `outputs/flowstar_torch_causal_mechanism_closure_20260813/20260813T060020Z`
- test counts: local and fresh-clone full suites both `710 passed, 2 skipped`;
  focused suites `18 passed` and `23 passed`; compileall passed
- fresh clone: `PASS` from HTTPS origin, detached exact scientific SHA;
  package verifier passed and final porcelain status was empty
- two-stage proof: scientific `src`, `tests`, and `experiments` tree hashes are
  respectively `7be43ed900a99308af24d7dbb13a46d51e1e7280`,
  `dca1cc7e3277098c88803c4da417ae72cc005741`, and
  `79855f638ca9fbfefd568087458c1701166c1062`; the attestation commit changes
  only this handoff and the evidence package, so these relevant trees remain exact
- final worktree status: clean after the attestation commit; verified again
  before publication

Detailed results are in:

- `docs/FLOWSTAR_NATIVE_VS_COPIED_PROBE_EQUIVALENCE_20260813.md`
- `docs/FLOWSTAR_TORCH_CAUSAL_FACTOR_SPLIT_20260813.md`
- `docs/FLOWSTAR_TORCH_LOSSLESS_STATE_QUEUE_BRIDGE_20260813.md`
- `docs/COMPLETE_O4_SOURCE_LEDGER_ORACLE_20260813.md`
- `docs/COMPLETE_O4_CARRY_FINAL_DECISION_20260813.md`
