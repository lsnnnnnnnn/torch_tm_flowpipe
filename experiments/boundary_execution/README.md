# Ordered packed boundary range experiment

Scientific source and formal runner: `5f37cbe0427c480ef0ebbbcaba292143bc8f4ede`.
Parent numerical/evidence anchor: `f6af6f67565a954d0c68f88a50cc9c181a2d0b08`;
its prepared production archive was run at `1551ab57aef7324f91882beeba9d368f36b3cdd5`.
The package's SOURCE_MAP identifies the separate packaging code commit. The final
delivery SHA is the checkout SHA printed by the verifier, avoiding a self hash.

Run from the delivered checkout, with the existing py11 environment:

```bash
PY_BOUNDARY=/srv/local/shengenli/miniforge3/envs/py11/bin/python
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 2 "$PY_BOUNDARY" -m experiments.boundary_execution.verify \
  artifacts/runs/boundary_execution_20260908T172756Z
PYTHONPATH=src:.:tests OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 2 "$PY_BOUNDARY" -m pytest -q -p no:cacheprovider \
  tests/test_boundary_range_plan.py tests/test_boundary_execution_evidence.py \
  tests/test_prepared_remainder_replay.py tests/test_our_reference_endpoint_containment.py \
  tests/test_endpoint_roundoff_repair.py tests/test_endpoint_roundoff_carry.py
```

The verifier recomputes clocks and timing sums from raw events, full fixed-run
published/common ranges from coefficients, all paired numerical payloads, saved
checkpoints, eight captured real-step local workloads using exact rational term
oracles, six complete boundary state sequences including resume/failure/export,
allocation counts, source/switch identity, repeated timing decisions and status.
It does not repeat the long ODE runs. Six narrowly scoped tamper cases change a
coefficient, endpoint error, actual switch, denominator seconds, source or status;
they refresh the byte manifest and still require semantic rejection.

The old verifier remains unchanged and is run at the parent snapshot with its
own README command. Its log and the 75 parent tests are under raw_minimal/parent_checks.
The new root suite command and fixture hashes are under tests/. It excludes only
`tests/test_our_solver_performance_evidence.py`, whose historical source-identity
checks target an older frozen audit. That old evidence was not rewritten to fit
this numerical source. The candidate-on 99 checks, parent tests and independent
checkout repetitions are not added twice to the unique current test total.

Reproduction of timing must use a clean worktree at the scientific SHA. Keep
CPU64, one intra/inter-op thread, CPU2 affinity and the existing dependencies.
Both modes explicitly enable prepared replay; only the new boundary switch
differs. Example fresh initial window from that checkout:

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 2 "$PY_BOUNDARY" -m experiments.boundary_execution.run \
  --scientific-sha 5f37cbe0427c480ef0ebbbcaba292143bc8f4ede \
  --plant brusselator --mode baseline --steps 20 --output /tmp/boundary-baseline-new
```

Use `--mode candidate` with a different new output directory. For the middle and
late windows, pass the complete `raw_minimal/input_checkpoints/brusselator100` or
`brusselator980`; VDP uses `vdp90`. Never initialize from a published box. Every
actual command, environment, PID/start ticks, exit and timing is saved in
raw_minimal/formal/commands.json. `launch.py` runs the finite alternating schedule,
requires completed tests/state gates, and rejects unfinished existing jobs;
inspect a recorded live process rather than launching a duplicate.

`profile.py` is the frozen pre-implementation attribution harness. It binds all
actual module aliases, uses exclusive time, and performs dispatch counting in a
separate pass. `prototype.py` and `prototype_range.py` preserve the isolated
same-input experiment before production changes. Private .pt captures are retained
only as hashed provenance; evidence verification loads safe hex JSON checkpoints.
`remaining_profile.py` observes three actual candidate steps only after the formal
schedule is complete. None of these observed seconds is a formal denominator.

`build.py` packages only this run, derives its CSV/JSON and Chinese report, and
references parent artifacts in place. It preserves both the dated original
implementation decision and the more conservative repetition policy frozen before
formal runs. `--refresh` incorporates later test receipts and regenerates the
report and byte manifest; numerical raw records are unchanged.

See [the Chinese report](../../docs/boundary_execution/REPORT_PLAIN_CHINESE.md) and
[the numerical/ownership contract](../../docs/boundary_execution/NUMERICAL_AND_LIFETIME_CONTRACT.md).
