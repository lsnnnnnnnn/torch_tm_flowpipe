# Huan `flowstar-gpu` reproduction audit

Date: 2026-08-26

Primary status: `HUAN_SOURCE_BUILDS__PROOF_MAPPING_INCOMPLETE`

## Decision

The current clean Huan plant-engine source is available, builds with a pinned
audit dependency overlay, and its local D1--D4 numerical batteries reproduce.
The full paper-level proof contract does not close: production startup has no
no-FTZ assertion, strict mode does not completely charge symbolic-Phi and
retained monomial-image point-coefficient roundoff, and refinement exposes no
proposal/commit or cache-freshness ledger.  These are source-level contract
gaps, not failed VDP comparisons.

Phase D therefore fails closed at D5/D6.  In accordance with the frozen goal,
the exact VDP cross-tool run and high-batch throughput campaign were not
started.  No empty scientific CSV/JSON was fabricated.

## Scope, isolation, and provenance

- Torch base branch:
  `origin/codex/vdp-post-accept-refinement-c2-20260820`.
- Required and observed Torch tip:
  `0fea2657b30aea5f8cfe326dbcd06d659b8dd26c`.
- Frozen Torch scientific code:
  `29c9ee8f1fe96b860052b86a2b37d79a37bbb2ca`.
- Audit branch:
  `codex/huan-flowstar-gpu-reproduction-audit-20260826`.
- Clean upstream Huan source:
  `/srv/local/shengenli/flowstar-gpu`, commit
  `d5f0b68fcd36ba5f582733624f074728fe9720d8`, branch `main`, no diff,
  no tags, and no submodules.
- Upstream remote: `https://github.com/huanzhang12/flowstar-gpu`.
- Authoritative paper input: `docs/paper/main.tex` at that source commit.  A
  PDF is absent, but the engine repository identifies the TeX as authoritative.
- Controller/CROWN/auto_LiRPA/ONNX/TMNP/NDB execution:
  `PROHIBITED_NOT_RUN`.

The user's dirty `/srv/local/shengenli/torch_tm_flowpipe` worktree was left
untouched.  The audit changed no Huan source file and no Torch C2 solver file;
all new code is audit harness, tests, documentation, or evidence in the
isolated Torch worktree.

Current clean-source closure must not be conflated with historical result
closure.  `/srv/local/shengenli/CROWN-Reach-GPU` is a clean integration/result
repository at `c28f0db949b87c475bf58404767b771f42d26dbe`.  Its 450 records all
name one of three `-dirty` Huan revisions (374 + 25 + 51 records).  The base
commits exist, but their binary diffs and untracked files do not; those results
are not exact reproductions of either the historical dirty state or current
clean source.

## Build and upstream regression result

The captured host has 80 logical Xeon Gold 6138 CPUs and four compute-capability
7.0 GPUs.  Device 0 is a Tesla V100-SXM2 with 16 GiB, driver 580.159.03.
The audit uses CUDA 12.6.68, GCC 13.4.0, Python 3.13.13, and
Torch 2.13.0+cu126.  This is not the paper's stated 32 GiB V100 environment,
so it would not support a hardware-identical throughput claim even if Phase F
were open.

`uv sync --active --frozen` succeeds but removes `ninja` and `PyYAML`; the
shipped CUDA JIT then reports `kernel_available=false`.  A pinned audit-only
overlay of `ninja==1.13.0` and `pyyaml==6.0.3` restores
`kernel_available=true`.  The source and `uv.lock` remain unchanged.  This is
classified as an incomplete project lock, not silently repaired packaging.

The complete plant-only upstream suite, including upstream slow tests and
excluding explicitly neural/controller files, produced:

```text
5 failed, 997 passed, 7 skipped, 5 xfailed in 1844.43s
```

All five failures occur before numerical execution when
`test_tape_kernels_valid.py` accesses the hard-coded author path
`/home/huan/projects/CROWN-Reach/src/configs`.  An audit-side pytest plugin
mapped only that exact path to the accessible same-name configurations under
`/srv/local/shengenli/CROWN-Reach-GPU/src/configs`; the five tests then passed
in 7.17s.  No upstream source or test file was edited.  Thus the raw suite has
one portability defect and no observed plant numerical regression.

## Proof-to-code result

The required 14-row map is
`outputs/huan_repro_audit/proof_to_code_map.csv`; the detailed analysis is
`docs/HUAN_ENGINE_PROOF_CONTRACT_20260826.md`.

| Contract | Status | Finding |
|---|---|---|
| elementwise outward arithmetic | `MAPPED_AND_TESTED` | D1 passes CPU/CUDA; generic overflow is an extended interval, never accepted here as a finite certificate |
| no-FTZ startup condition | `CONTRADICTED` | behavior was observed in this environment, but `enable_determinism` has no production assertion |
| overflow/div-zero fail closed | `PARTIALLY_MAPPED` | division returns a bad mask; generic overflow has no primitive status and `assert_valid` permits infinity |
| any-order reduction and computable inflation | `MAPPED_AND_TESTED` | shipped `dot_error_bound` matches the stated formula and encloses every finite-hypothesis exact test |
| strict versus parity | `CONTRADICTED` | final composition GEMM is inflated; symbolic Phi einsums and retained image-product coefficients are not completely charged |
| sparse support and pinned chunking | `MAPPED_AND_TESTED` | differential/support and member-separable bitwise batteries pass in their stated regions |
| symbolic queue | `PARTIALLY_MAPPED` | reconstruction/reset behavior passes, but strict Phi-product roundoff remains uncharged |
| transcendental ulp budgets | `ASSUMPTION_ONLY` | mpmath tests are out of band; production startup does not calibrate the deployed library |
| polynomial-only unconditional scope | `CONTRADICTED` | missing no-FTZ assertion and incomplete strict accounting prevent the unconditional claim |

The local any-order kernel result is strong but conditional: the exact oracle
tested 987 finite schedules per device across sequential, pairwise, chunks 3
and 17, permutations, FMA where available, `torch.dot`, boundary lengths,
severe cancellation, subnormal products, near-overflow, mixed magnitudes, and
seeded searches.  It does not erase the theorem hypotheses (finite
intermediates, `m*u <= 1/4`, and no FTZ).

## Phase D microreproduction

| Gate | Executable result | Contract decision |
|---|---|---|
| D1 elementwise | CPU 7/7; CUDA 7/7 | local pass; non-finite overflow separately classified |
| D2 any-order reduction | CPU 987/987; CUDA 987/987; fused CUDA kernel loaded | local conditional pass |
| D3 dense/sparse/support | 84 passed, 2 skipped | pass; skipped CUDA cases are covered by D4's CUDA battery |
| D4 chunk/lane divergence | 55 passed, 1 skipped; custom chunks 1,2,3,5,7 pass bitwise on CPU/CUDA; B=1 embedded in B=2 passes | pass only for fixed/fused member-separable regions; fallback/library scope remains enclosure, not universal bit equality |
| D5 refinement | 41 passed; CPU/CUDA probes confirm zero replay after first-map failure, exact 490/491 caps, subset commits, and Flow* first-failing-dimension partial-vector semantics | **fail**: no public proposal/commit ledger or remainder-cache freshness metadata establishes last-ledger ownership or stale-cache rejection |
| D6 strict/parity | 25 operator tests pass and strict final-GEMM remainder contains parity | **fail**: the tested widening does not cover the identified Phi and retained point-product accounting holes |

The machine-readable decision is
`outputs/huan_repro_audit/phase_d_gate.json`.

## Phase E/F stop decision

The exact frozen VDP contract was not ported or run because Phase D did not
pass.  Therefore no claim is made for step 1, `T=1`, `T=3`, `T=6.32`, native
`T=10`, first divergence, or any endpoint/segment width channel.  Likewise no
batch result at B=1,16,64,256,1024,4096 and no speedup claim was produced.

This is not a negative VDP result.  It is `NOT_RUN_D_GATE_FAILED`.  In
particular, existing Huan default VDP runs or historical dirty CROWN-Reach
records cannot substitute for the exact frozen contract.

## Required decision questions

1. **Is the paper's any-order reduction proof correctly implemented?**
   The shipped inflation formula and runtime `m*u` guard are mapped correctly,
   and all 1,974 CPU/CUDA exact checks pass under finite-intermediate
   hypotheses.  The unqualified engine claim is not closed because production
   does not assert no-FTZ and non-finite absolute reductions have no explicit
   status.  The correct answer is “locally yes under its hypotheses,” not an
   unconditional yes.

2. **Which claims are unconditional, and which depend on assumptions or a
   trust model?** Elementwise and reduction claims depend on IEEE binary64,
   finite intermediates, gradual underflow/no FTZ, and the stated length bound.
   Transcendentals depend on empirical library ulp assumptions.  Parity adopts
   Flow*'s point-coefficient trust model.  Strict is intended to charge those
   errors but is incomplete in the mapped paths.  No full polynomial-flowpipe
   claim is established as unconditional by this source state.

3. **Is sparse execution functionally consistent with dense execution?**
   Yes within the tested plant scope: support supersets, strict/oversized
   supports, truncation/cutoff carries, random coefficients, full operator
   matrices, five-step dense/sparse trajectories, and reset boundaries pass.
   CUDA point coefficients use the documented ulp contract where bit equality
   is not promised.

4. **Is chunking sound, and where is it bitwise neutral?** The tested
   member-axis chunks are bitwise neutral on CPU and the fused CUDA path for
   chunks 1,2,3,5,7 and B=1-in-B=2.  CUDA fixed reductions and frozen-lane
   batteries also pass.  Outside those member-separable fixed schedules,
   any-order inflation supports enclosure, but batch-shaped library fallbacks
   are not promoted to universal bitwise identity.

5. **Does the exact frozen VDP reach T=10?** Unknown: deliberately not run
   because D5/D6 failed.  No other VDP configuration answers this question.

6. **Does Huan's engine expose the cause of Torch's terminal y-upper
   failure?** No cross-tool trace was authorized.  The frozen Torch evidence
   still identifies first-self-map y-upper containment failure at attempted
   `h=0.003950348390361663`, followed by a retry below native
   `h_min=0.002`.  Huan's source agrees that initial failure is never rescued,
   but without the exact run it does not expose an earlier causal divergence.

7. **Is any part ready to be ported?** Do not port or merge the engine.  The
   exact-oracle tests, proof-map schema, and conditional reduction-test method
   are reusable audit methodology.  Huan's engine should remain an external
   reference until strict roundoff and runtime hypothesis closure are fixed and
   re-audited.

8. **What is the smallest next experiment with maximum information gain?**
   Build a one-step, 2-by-2 exact-rational adversarial witness for
   `symbolic_remainder.propagate` and the retained monomial-image multiply,
   choosing cancellation/half-ulp coefficients and comparing strict output
   against the exact Phi/J image.  It directly distinguishes “missing proof
   instrumentation” from a concrete strict under-enclosure.  Add a production
   no-FTZ startup check in the same isolated upstream change, then rerun only
   D1/D2/D5/D6 before any VDP.

## Deliverable ledger

| Deliverable | Status | Evidence/reason |
|---|---|---|
| `docs/HUAN_ENGINE_REPRODUCTION_AUDIT_20260826.md` | `READY` | this report |
| `docs/HUAN_ENGINE_PROOF_CONTRACT_20260826.md` | `READY` | 14-claim proof contract and D1/D2 interpretation |
| `outputs/huan_repro_audit/artifact_inventory.json` | `READY` | current-source and historical-dirty closure kept separate |
| `outputs/huan_repro_audit/source_manifest.json` | `READY` | clean engine commit and source hashes |
| `outputs/huan_repro_audit/environment.txt` | `READY` | hardware/software, Git, lock hashes, and package capture |
| `outputs/huan_repro_audit/build.log` | `READY_WITH_LOCK_GAP` | frozen-lock missing-dependency path plus pinned overlay success |
| `outputs/huan_repro_audit/upstream_tests.log` | `READY_WITH_CLASSIFIED_FAILURES` | 997 passes; 5 absolute-path portability failures |
| `outputs/huan_repro_audit/proof_to_code_map.csv` | `READY` | required schema, 14 unique claims |
| `outputs/huan_repro_audit/phase_d_gate.json` | `READY` | D1--D6 evidence and fail-closed decision |
| `outputs/huan_repro_audit/step1_common_input.csv` | `NOT_RUN_D_GATE_FAILED` | absent; Phase E prohibited |
| `outputs/huan_repro_audit/fixed_horizon_matrix.csv` | `NOT_RUN_D_GATE_FAILED` | absent; Phase E prohibited |
| `outputs/huan_repro_audit/native_terminal.json` | `NOT_RUN_D_GATE_FAILED` | absent; Phase E prohibited |
| `outputs/huan_repro_audit/batch_throughput.csv` | `NOT_RUN_D_GATE_FAILED` | absent; Phase F prohibited |
| `outputs/huan_repro_audit/raw_logs/*` | `READY` | provenance, microkernels, D3--D6, boundary probes, and path replay |
| `outputs/huan_repro_audit/SHA256SUMS` | `READY` | exact recursive evidence coverage except itself |

## Test and integrity evidence

Focused audit/verifier tests:

```text
18 passed
```

Complete Torch suite after all report and verifier changes:

```text
845 passed, 2 skipped in 368.68s (0:06:08)
```

`git diff 0fea2657b30aea5f8cfe326dbcd06d659b8dd26c -- src` is empty.

Regenerate the mutable provenance/checksum envelope, or verify the finalized
package without running scientific phases:

```bash
/srv/local/shengenli/miniforge3/envs/py11/bin/python \
  scripts/huan_repro_artifact_audit.py \
  --search-root /srv/local/shengenli \
  --candidate-repo /srv/local/shengenli/CROWN-Reach-GPU \
  --engine-repo /srv/local/shengenli/flowstar-gpu \
  --torch-repo /srv/local/shengenli/torch_tm_flowpipe_huan_repro_audit_20260826 \
  --output-root outputs/huan_repro_audit

/srv/local/shengenli/miniforge3/envs/py11/bin/python \
  scripts/verify_huan_repro_package.py \
  --repo-root . \
  --output-root outputs/huan_repro_audit
```
