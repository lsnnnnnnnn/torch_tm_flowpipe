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
