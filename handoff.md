# Flow*–Torch causal mechanism closure handoff — 2026-08-13

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
- scientific tested SHA: `PENDING_SCIENTIFIC_COMMIT`
- publication tip: `PENDING_ATTESTATION_COMMIT`
- Flow* SHA: `b85a3211748cb77b736fe4ad42ee02d8d2b81148`
- evidence package:
  `outputs/flowstar_torch_causal_mechanism_closure_20260813/20260813T060020Z`
- test counts: `PENDING_FINAL_TEST_RUN`
- fresh clone: `PENDING_REMOTE_SCIENTIFIC_SHA`
- final worktree status: `PENDING_PUBLICATION`

Detailed results are in:

- `docs/FLOWSTAR_NATIVE_VS_COPIED_PROBE_EQUIVALENCE_20260813.md`
- `docs/FLOWSTAR_TORCH_CAUSAL_FACTOR_SPLIT_20260813.md`
- `docs/FLOWSTAR_TORCH_LOSSLESS_STATE_QUEUE_BRIDGE_20260813.md`
- `docs/COMPLETE_O4_SOURCE_LEDGER_ORACLE_20260813.md`
- `docs/COMPLETE_O4_CARRY_FINAL_DECISION_20260813.md`
