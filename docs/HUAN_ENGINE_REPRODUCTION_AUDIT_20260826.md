# Huan engine reproduction audit

Date: 2026-08-26

Primary status: `HUAN_REPRO_BLOCKED_MISSING_CORE_SOURCE`

## Decision

The accessible artifacts fail the mandatory Phase A source-closure gate. The
only discovered candidate is the `CROWN-Reach-GPU` integration/result
repository, not the `flowstar_gpu` engine repository. Its result records refer
to inaccessible external paths and only dirty engine revisions whose patches
are absent.

The authoritative Phase A findings and unblock request are in
`docs/HUAN_REPRO_BLOCKED_MISSING_ENGINE_SOURCE_20260826.md`. This report does
not promote report text, logs, sampling, or Flow* parity into reproduction or
soundness evidence.

## Evidence summary

| Requirement | Evidence | Finding |
|---|---|---|
| Torch audit base | `artifact_inventory.json`, `torch_provenance.log` | target tip exactly `0fea2657…`; scientific SHA `29c9ee8…` |
| isolated clean start | audit branch/worktree provenance | created from the target remote tip; user worktree preserved |
| actual engine package | bounded `/srv/local/shengenli` discovery and candidate Git tree | no `src/flowstar_gpu` or engine build closure |
| proof note and engine docs | bounded discovery plus all candidate Git history paths | PDF, `REPRODUCE.md`, and `OPTIMIZATION.md` unavailable |
| wrapper/integration source | clean `CROWN-Reach-GPU` at `c28f0db…` | available, but not the engine |
| external dependencies/inputs | `symlink_inventory.tsv` | 94/94 links inaccessible; 93 absolute external links |
| run provenance | `record_provenance.tsv` | 450/450 records use one of three `-dirty` engine revisions; zero clean |
| dirty patches | candidate tree/history | absent |
| Phase B–F gates | Phase A stop rule | not run; no scientific table fabricated |

## Required decision questions

1. **Is the paper's any-order reduction proof correctly implemented?**
   Unknown and untested. The proof note and shipped kernel source are
   unavailable, so no theorem-to-operation mapping is possible.

2. **Which claims are unconditional, and which depend on assumptions or a
   trust model?** None can be independently classified from the available
   artifacts. Report labels such as `fp-rigorous` are retained as upstream
   claims, not accepted as proof. Polynomial-only and transcendental scopes,
   no-FTZ assumptions, and strict/parity trust boundaries remain unaudited.

3. **Is sparse execution functionally consistent with dense execution?**
   Unknown; the shipped implementations and differential tests are absent.

4. **Is chunking sound, and where is it bitwise neutral?** Unknown; member
   separation, masking/freezing, and pinned bitwise-neutral regions cannot be
   inspected or executed.

5. **Does the exact frozen VDP reach T=10?** Not run. A differently configured
   benchmark or existing report cannot answer this question.

6. **Does Huan's engine expose the cause of Torch's terminal y-upper failure?**
   No accessible evidence does. The Torch C2 terminal classification remains
   the first self-map y-upper subset failure at
   `h=0.003950348390361663`, followed by a retry below `h_min=0.002`.

7. **Is any part ready to be ported?** No. The hard prohibition on merging or
   reconstructing the engine applies. It should remain an external reference
   until source, proof, and microkernel gates close.

8. **What is the smallest next experiment with maximum information gain?**
   After obtaining an immutable clean source plus all dirty patches and the
   proof note, run only D1/D2 against MPFR: shipped interval primitives and
   shipped any-order inflation for small boundary lengths, severe
   cancellation, subnormal/underflow, and near-overflow cases, with runtime
   checks for `m*u <= 1/4` and no-FTZ. Do not run VDP or throughput first.

## Deliverable ledger

| Deliverable | Status | Reason/evidence |
|---|---|---|
| `docs/HUAN_ENGINE_REPRODUCTION_AUDIT_20260826.md` | `READY` | this report |
| `docs/HUAN_REPRO_BLOCKED_MISSING_ENGINE_SOURCE_20260826.md` | `READY` | detailed Phase A stop report |
| `docs/HUAN_ENGINE_PROOF_CONTRACT_20260826.md` | `NOT_RUN_SOURCE_MISSING` | proof PDF and engine source unavailable; no contract fabricated |
| `outputs/huan_repro_audit/artifact_inventory.json` | `READY` | machine-readable source-closure decision |
| `outputs/huan_repro_audit/missing_artifacts.tsv` | `READY` | exact request list (additional Phase A deliverable) |
| `outputs/huan_repro_audit/source_manifest.json` | `NOT_RUN_SOURCE_MISSING` | Phase C inapplicable |
| `outputs/huan_repro_audit/environment.txt` | `NOT_RUN_SOURCE_MISSING` | engine build environment not created |
| `outputs/huan_repro_audit/build.log` | `NOT_RUN_SOURCE_MISSING` | no engine source to build |
| `outputs/huan_repro_audit/upstream_tests.log` | `NOT_RUN_SOURCE_MISSING` | no engine tests available |
| `outputs/huan_repro_audit/proof_to_code_map.csv` | `NOT_RUN_SOURCE_MISSING` | no source or paper; empty mapping would be fabricated |
| `outputs/huan_repro_audit/step1_common_input.csv` | `NOT_RUN_SOURCE_MISSING` | Phase E prohibited by stop rule |
| `outputs/huan_repro_audit/fixed_horizon_matrix.csv` | `NOT_RUN_SOURCE_MISSING` | Phase E prohibited by stop rule |
| `outputs/huan_repro_audit/native_terminal.json` | `NOT_RUN_SOURCE_MISSING` | Phase E prohibited by stop rule |
| `outputs/huan_repro_audit/batch_throughput.csv` | `NOT_RUN_SOURCE_MISSING` | Phase F prohibited by stop rule |
| `outputs/huan_repro_audit/raw_logs/*` | `READY` | discovery, Git, record, and symlink inventories |
| `outputs/huan_repro_audit/SHA256SUMS` | `READY` | recursive output-package integrity |

The `NOT_RUN_SOURCE_MISSING` files intentionally do not exist. Their status is
recorded here rather than represented by empty scientific tables.

## Test and no-regression evidence

The new artifact auditor has focused tests for bounded discovery, source/build
closure, clean-versus-dirty record classification, exact missing-item requests,
and checksum tamper/uncovered-file rejection.

Test environment: Python 3.11.15 and Torch 2.5.1+cu121 from the same `py11`
environment recorded by the C2 package.

| Gate | Command | Result |
|---|---|---|
| focused auditor tests | `PATH=/srv/local/shengenli/miniforge3/envs/py11/bin:$PATH PYTHONPATH=.:src pytest -q tests/test_huan_repro_artifact_audit.py` | `7 passed in 0.04s` |
| complete Torch suite | `PATH=/srv/local/shengenli/miniforge3/envs/py11/bin:$PATH PYTHONPATH=.:src pytest -q` | `834 passed, 2 skipped in 432.67s` on the finalized report commit |

The environment path and `PYTHONPATH` are explicit because two historical
tests spawn `python` or a repository script. An initial invocation through an
absolute interpreter path, without activating its `bin` directory or exporting
the repository source path, produced two subprocess import/lookup failures.
Both failing cases passed when rerun in the correctly activated environment,
and the complete activated run then passed.
