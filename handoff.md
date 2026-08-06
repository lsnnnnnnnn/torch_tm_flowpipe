# Handoff: Xiangru Q3 matched-baseline audit

## State

- Branch: `codex/xiangru-q3-matched-baseline-audit-20260806`.
- Starting Torch lineage: `4707abed5e8d28ec56c2b5e76b800bd284f0008b`.
- Xiangru reproduction: clean detached `27d29050a5f214b56f211ca9cb411e734ed80230`.
- Xiangru canonical checkout observed: `84184de6c2b3f1ff2da6755f732d91925037025d`; fetch/ls-remote unavailable because private HTTPS credentials were absent.
- DiffReach: `dd628eb443b517d6415de93e7035b4baef73963e`.
- Stock Flowstar: `b85a3211748cb77b736fe4ad42ee02d8d2b81148`.
- Verdict: **Case B — FORMAL MATCHED COMPARISON NOT AUTHORIZED**.

## Reproduction and contract

- Fresh Xiangru B48 complete-Q3 verifies all 200 fixed 0.1 s segments to T=20 with no failure or retry.
- Author/fresh semantic comparison: 2,850 numeric fields, max absolute error `1.1546319456101628e-13`, no difference outside declared `1e-6` tolerance.
- Q3 is complete total degree <=3 over six variables (84 slots), K2 polynomial Picard plus ten DR-RP rounds.
- Torch's degree-3 retention predicate agrees, but the full algorithm and model contract do not.
- First gate blocker: Xiangru closed-loop homogeneous TORA + frozen NN + B48 + T20 versus available Torch plant-only Van der Pol + no controller + T10.
- Stock Flowstar is an independent VDP reference, not a third Q3 winner.

## Trace repair

- Transition lifecycle stages now record actual objects and stable full-content hashes.
- Rejected attempts use `accepted_step_index=null` and retain accepted-count-before-attempt.
- Terminal call 44 is correctly y-component `-x²y`, after 307 accepted states in a rejected attempt; its DAG is explicitly terminal-local.
- One-step, T=0.2 instrumented/uninstrumented and terminal replay evidence pass. Instrumentation numerical equivalence is exact for all selected fields.

## N/A results

- Formal endpoint/tube tightness: N/A.
- Formal runtime comparison: N/A.
- Native fresh Xiangru CUDA timing is recorded by stage, but one reproduction is not a repeated performance benchmark and is not compared against CPU plant-only timing.

## One next action

Implement a native Torch lane for the exact homogeneous-TORA plant, frozen controller/bounder, periodic control update, B48 initial partition and endpoint/tube contract. Validate it independently, then restart at Gate 1. Do not create a winner table before Gates 1–5 pass.

## Evidence

All raw commands, streams, exit codes, configs, contracts, traces, tests, environments and checksums are under `outputs/xiangru_q3_matched_audit_20260806/`. See the four top-level reports for the reproduction, order/contract, trace repair and gated comparison conclusions.
