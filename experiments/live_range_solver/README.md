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
