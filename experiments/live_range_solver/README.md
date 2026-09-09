# Live range solver experiments

Use the existing py11 environment. All commands below run on CPU2 and GPU0.

```bash
export PATH=/srv/local/shengenli/miniforge3/envs/py11/bin:$PATH
export PYTHONPATH=src:.:tests
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
taskset -c 2 python -m pytest -q -p no:cacheprovider \
  tests/test_live_range_service.py tests/test_live_range_solver_integration.py \
  tests/test_live_range_evidence.py
taskset -c 2 python -m experiments.live_range_solver.runner \
  --plant van_der_pol --route G --batch 2 --steps 2 --diagnostic \
  --output /tmp/live-range-fresh-check
taskset -c 2 python -m experiments.live_range_solver.verify /tmp/live-range-fresh-check
```

The runner creates each actual initial complete state, lets each task execute the
unchanged solver, submits its current sparse range requests, and consumes the
returned values before generating later requests. Q/G use the same threaded
service; S/S_gpu run independently with the same respective range arithmetic.
L retains the legacy fastest packed path and its documented power limitation.

Only `--checkpoint` loads saved mathematical state, for explicitly labelled
`RESUMED_LOCAL_WINDOW` runs. No mode loads previously computed requests or answers
to advance a task. The fixed partition is referenced from the parent evidence.

`campaign.py --phase diagnostic` runs the finite correctness matrix. It records a
live child PID and immutable command per case; an existing partial output is never
silently restarted. `--resume` accepts only complete hashed existing records with
the same scientific files. Formal timing requires an explicit correctness gate.
The two wait candidates have already been selected using `pilot.py`; do not rerun
selection using formal outcomes.

`--warm` records one separate CUDA compile/load/self-test startup before creating
the tasks measured by the full-run timer. Normal checks, transfers, synchronization,
worker startup, task construction and cleanup remain inside that timer. Diagnostic
serialization and Fraction checks are not performance samples.

The verifier checks source-state/request/attempt identities, causal predecessors,
semantic groups, device-written receipts, full state continuity, the common
observer's actual bounds, clock accounting and accepted-step numerators. It
recomputes all captured diagnostic requests and their independent exact rational
powers/terms/sums. `--no-recompute` omits backend recomputation and must not be
reported as complete acceptance. Long runs only save full request operands at the
preregistered check steps; subsequent calls reuse the unchanged operator contract.

Historical source-locked verifiers belong to the old worktree. The original range
evidence verifier and its 131 local tests were rerun there and are REUSED, not
counted as fresh source tests. Do not weaken or skip their assertions to admit new
source. Fresh root testing excludes the three historical identity-locked evidence
modules and lists that scope explicitly.

## Complete-package acceptance

After installing the post-measurement packaging files, use the same environment
and CPU2/GPU0 settings above:

```bash
taskset -c 2 python -m experiments.live_range_solver.verify_package \
  artifacts/runs/live_range_solver_20260909T053007Z
```

The default command checks every saved diagnostic run and recomputes every
captured request, validates source files against the frozen scientific git
commit, reconstructs each diagnostic task's first state from the fixed partition
or checkpoint, checks raw process/job identities, accepted generations and clock spans, and regenerates the
derived tables in a temporary directory for comparison. It also verifies the
fault, cancellation, heterogeneous-refinement and fallback evidence, and deduplicates
test identities. Finally, it executes fresh B2 two-step S/Q/S_gpu/G solves for
both plants from the fixed initial partition and compares their full state and
current-call request sequences to the recorded diagnostic references. This is
32 new accepted lane-steps. It does not repeat the full performance campaign or
historical 1000-step trajectories.

`--no-live-replay` and `--no-recompute` are partial diagnostic options. Their
output has `verified: false`; neither is complete acceptance. The historical
source-locked verifiers remain in the parent worktree.

`enrich.py` and `package_reports.py` only read completed measured evidence and
write tables/reports. They do not advance a solver or load answers into a worker.
Do not replace missing formal runs with these tools. `package_reports.py` leaves
independent-copy and final push/SHA acceptance pending until actual receipts exist.

The fixed scientific commit is `ef4e2f0c17518a4c4989071a596cd8631235c5ea`.
Later commits contain documentation, packaging and verification code; the
scientific runtime and runner bytes remain checked against that commit.

## Using the service in a caller

The concrete continuous-solve example is `runner.run_case`, with `route="Q"`
or `route="G"`. It creates all tasks, registers independent copies of their
complete accepted states, starts a `ThreadPoolExecutor`, and calls the existing
step function inside each task's `execution()` context. A successful step commits
the complete `(reset_tm, flowstar_normal_state)` tuple; rejected or cancelled
attempts do not become the next accepted state. Callers must use the solver's
existing immutable-input discipline and never mutate a task's accepted state in
place. Set Torch thread configuration before creating workers.

For saved tasks, use `save_live_range_checkpoint` and
`load_live_range_checkpoint` from `torch_tm_flowpipe.live_range_checkpoint`.
The safe JSON sidecar preserves ordinary diagnostic metadata and ordering that
the historical terminal writer intentionally omitted. Loading registers a new
epoch, including when a fresh process reuses the old run ID. The real
cancel/save/resume exercises and their reference comparisons are in `faults.py`.

## Scope of the measurements

The main workload uses the unchanged fixed 8×4 partition. `original-*` records
are separate unpartitioned B1 diagnostics. `history-*` records are explicitly
`RESUMED_LOCAL_WINDOW`. `heterogeneous/` uses test-only different h values to
exercise different actual refinement and request counts. It is not a performance
sample. Main parameters were never selected from formal outcomes.

Formal timing excludes independent Fraction auditing, model/operand diagnostic
capture and evidence serialization. Necessary CPU exact-power certification and
correction remains inside the measured S/Q computation. CUDA compilation,
module loading and primitive self-tests are recorded separately before a new
timed service reuses the module. Group times are host-observed evaluator spans,
including synchronization; worker step spans include waits. Overlapping task
waiting seconds are never summed as wall time. GPU memory is the Torch allocation
peak, excluding the driver context; RSS is the process lifetime high-water mark.

The independent Fraction operand capture covers all requests in the first two
steps of B32, B8, original B1 and each three-step history window; all two-step
small/split/order tests; and steps 1/2/60/100/119/120 in the long runs. Fault
boundaries, explicit numerical probes and heterogeneous-refinement examples
have separate complete checks. Subsequent calls retain the unchanged local
operator contract, without a claim that every later request was reaudited.
