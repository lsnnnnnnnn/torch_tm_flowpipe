# Huan reproduction blocked: engine source unavailable

Date: 2026-08-26

Primary status: `HUAN_REPRO_BLOCKED_MISSING_CORE_SOURCE`

## Scope and stop decision

The Phase A source-closure gate fails closed. The accessible artifact set does
not contain the `flowstar_gpu` engine package, its build files, its exact clean
source state, or its dirty patches. The repository at
`/srv/local/shengenli/CROWN-Reach-GPU` is a clean wrapper/integration and result
repository; it is not the engine repository.

Phases B through F were not run. In particular, this audit did not reconstruct
the paper, run a long flowpipe, treat sampling or Flow* equality as a soundness
oracle, or use the available reports as proof that the engine was reproduced.

## Torch provenance and isolation

- Required remote branch: `origin/codex/vdp-post-accept-refinement-c2-20260820`.
- Expected and observed tip: `0fea2657b30aea5f8cfe326dbcd06d659b8dd26c`.
- Frozen scientific code SHA: `29c9ee8f1fe96b860052b86a2b37d79a37bbb2ca`.
- Isolated audit branch:
  `codex/huan-flowstar-gpu-reproduction-audit-20260826`.
- Isolated worktree:
  `/srv/local/shengenli/torch_tm_flowpipe_huan_repro_audit_20260826`.

The user's existing `/srv/local/shengenli/torch_tm_flowpipe` worktree had
uncommitted work and was preserved. The audit worktree was created directly
from the observed remote tip. No C2 solver file has been modified.

Machine-readable provenance is in
`outputs/huan_repro_audit/artifact_inventory.json` and
`outputs/huan_repro_audit/raw_logs/torch_provenance.log`.

## What is available

The bounded search required by the goal found one candidate:
`/srv/local/shengenli/CROWN-Reach-GPU`.

That candidate is a clean Git repository at
`c28f0db949b87c475bf58404767b771f42d26dbe`, with 33 locally available
commits. It contains:

- CROWN-Reach wrapper/integration source, including `src/CrownReach.cpp`,
  `src/CrownSettings.cpp`, `src/crown.py`, and `src/run.py`;
- reports and campaign tables such as `comparison/REPORT.md` and
  `comparison/SUITE_RESULTS.md`;
- 1,063 tracked `.log`, `.out`, or `.err` paths;
- 171 regular ONNX model files;
- 407 tracked YAML paths, of which 317 are regular files and 90 are symlinks;
- 450 individual JSON run records carrying a `flowstar_gpu` revision.

The 450 record revisions are all dirty:

| recorded revision | individual records | source state |
|---|---:|---|
| `3e12487d10604f5d3fa33e0a22c87acd368d3bc6-dirty` | 374 | dirty; patch absent |
| `876db965465c5af10e8c258f8d73833d653fc22e-dirty` | 25 | dirty; patch absent |
| `d1b67e0bbfe7ee6cbaa774cd5d8f7dbcc27e480f-dirty` | 51 | dirty; patch absent |

No available record is tied to a clean engine state. The full one-row-per-file
mapping is `outputs/huan_repro_audit/raw_logs/record_provenance.tsv`.

## External and inaccessible links

The integration repository has 94 symlinks. All 94 resolve through locations
that the audit account cannot traverse; 93 have absolute external targets and
the remaining relative link transitively reaches the same inaccessible tree.
These include:

- `/home/huan/projects/flowstar`;
- `/home/huan/projects/Verifier_Development/auto_LiRPA`;
- many `/home/huan/projects/flowstar_gpu/benchmarks/...` configs.

The account can stat `/home/huan`, but that directory is mode `0750` and owned
by `huan:huan`; it cannot inspect `projects`. Therefore the audit does not claim
that every target is physically absent. It classifies them as `INACCESSIBLE`,
which is equally insufficient for source closure and reproduction. See
`outputs/huan_repro_audit/raw_logs/symlink_inventory.tsv`.

## What is unavailable

The accessible closure lacks:

1. the actual engine repository containing `src/flowstar_gpu` and build files;
2. engine Git history or an immutable archive with a complete checksum manifest;
3. the exact clean target commits;
4. complete binary diffs and untracked files for all three recorded dirty states;
5. `flowstar_gpu/docs/REPRODUCE.md`;
6. `flowstar_gpu/docs/OPTIMIZATION.md`;
7. the `new_crown_reach*.pdf` proof note;
8. exact engine benchmark configs, controllers/models, specifications, and
   generation scripts, including a frozen VDP port;
9. a complete dependency/source lock and exact machine capture.

The exact request list and evidence for each item are in
`outputs/huan_repro_audit/missing_artifacts.tsv`.

## Consequences for scientific claims

- The engine has not been reproduced from source.
- The reported floating-point proof cannot be mapped to or checked against the
  kernels.
- Strict and Flow*-parity modes cannot be distinguished in executable evidence.
- Dense/sparse consistency, support exactness, chunking, lane freezing,
  Picard/refinement behavior, and any-order reduction inflation are untested.
- The exact frozen VDP contract was not ported or run in Huan's engine.
- No throughput claim is reproduced; existing B=1 or high-batch logs are
  retained only as unverified records.
- Nothing is ready to port into the production Torch solver. The unavailable
  engine remains an external reference only.

## Exact unblock request

Huan should either grant read-only access to the engine tree or copy an
immutable archive into an accessible sibling location. The delivery must
include the clean reproduction commit, `git diff --binary` plus untracked files
for every `-dirty` state, both engine docs, the exact proof PDF, all referenced
inputs and generation scripts, submodule/dependency SHAs, environment locks,
and machine/command captures.

Once closure is restored, the smallest high-information experiment is the
Phase D1/D2 gate: run the shipped elementwise enclosure and any-order dot
inflation kernels against an independent MPFR oracle, beginning with
`m=1,2,3`, cancellation, subnormal/underflow, and near-overflow cases while
checking `m*u <= 1/4` and no-FTZ. No VDP or throughput run should precede that
gate.

## Evidence integrity

Regenerate and verify the Phase A package with:

```bash
/srv/local/shengenli/miniforge3/envs/py11/bin/python \
  scripts/huan_repro_artifact_audit.py \
  --search-root /srv/local/shengenli \
  --candidate-repo /srv/local/shengenli/CROWN-Reach-GPU \
  --torch-repo /srv/local/shengenli/torch_tm_flowpipe_huan_repro_audit_20260826 \
  --output-root outputs/huan_repro_audit

/srv/local/shengenli/miniforge3/envs/py11/bin/python \
  scripts/huan_repro_artifact_audit.py \
  --search-root /srv/local/shengenli \
  --candidate-repo /srv/local/shengenli/CROWN-Reach-GPU \
  --torch-repo /srv/local/shengenli/torch_tm_flowpipe_huan_repro_audit_20260826 \
  --output-root outputs/huan_repro_audit \
  --verify-only
```

`outputs/huan_repro_audit/SHA256SUMS` covers every stored file under that
output directory except itself.
