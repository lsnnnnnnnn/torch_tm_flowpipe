# Real range requests and one CUDA operator

Use the existing py11 environment; no dependency installation is needed. The
CUDA module loads the installed PyTorch NVRTC 12.1 library and NVIDIA driver,
compiles one source with directed double arithmetic, and targets compute_70.
CPU request inputs are float64; the public solver defaults remain unchanged.

```bash
export PATH=/srv/local/shengenli/miniforge3/envs/py11/bin:$PATH
export PYTHONPATH=src:.:tests
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
taskset -c 2 python -m experiments.range_batch_device.verify \
  artifacts/runs/range_batch_device_20260909T030609Z
taskset -c 2 python -m pytest -q -p no:cacheprovider \
  tests/test_range_requests.py tests/test_range_cuda.py tests/test_range_device_evidence.py
```

The default verifier checks all raw powers/terms/sums, request IDs and source
states, grouping barriers, all 150 unprofiled timing events and derived decisions.
It also replays the 140 bounded capture steps from complete states and the local
CPU/GPU integration cases. It does not repeat the historical 1000-step runs.
`--no-replay` checks static/raw arithmetic and timing evidence only; it is useful
while packaging and is never described as the full integration acceptance.

`RangeRequest` and `evaluate_range_requests` are in `range_requests.py`.
`range_request_execution("cpu" | "cuda")` connects real B1 solver callers when
both prepared replay and packed boundary execution are explicitly enabled.
The source caller packing and per-ID output wrapping are included in full timing.
External power tables use the original normal evaluator and are not GPU work.

The fixed partition precedes capture and timing. `corpus.py` creates 32 distinct
tasks per plant and records their first two real steps, plus 12 saved-state
windows. The capture wrapper observes the actual callable bindings, including
module aliases. Each dependency barrier contains at most one request per task;
future requests are not aggregated to inflate a batch size. Recorded values are
lossless hex JSON/gzip, with complete-state hashes and source checkpoint hashes.

`audit_corpus.py` produces independent rational checks of every power, term and
sum and records device-written invocation receipts. `benchmark.py` runs five
alternating blocks at B1/B8/B32 after root checks have closed. Its full modes
include sparse packing, grouping, arithmetic validation, scatter and wrapping;
GPU full additionally includes both transfers and synchronization. The resident
mode and prepacked CPU compute mode are separate tables. It uses explicit input
ownership and never adds hidden numeric caching. Repeated blocks are measurement
replicates of the same distinct tasks, not duplicated tasks presented as B32.

`integration.py` checks complete returned state, observer purity, checkpoint
resume and failure rollback, including VDP99/100/101 and Brusselator999/1000/1001.
CUDA integration is only optional B1 range offload, 20 steps per plant. It does
not claim integrated batch solver throughput or a new complete horizon.

`build.py` derives the report, CSV/JSON and byte manifest. The evidence code and
test SHA must be committed before building; numerical source is separately
locked. Source-locked old verifiers remain untouched and run at their own parent
snapshot. Their receipts are REUSED, never added to the current unique count.
The two excluded old evidence modules are `test_our_solver_performance_evidence.py`
and `test_boundary_execution_evidence.py`; the latter was run on the parent in
this goal. A PATH-only historical subprocess failure in the current root suite
is preserved and resolved by a recorded targeted rerun, with identities deduped.

For a fresh reproduction use a separate output copy of the harness constants;
do not overwrite the frozen evidence. Corpus and timing runners reject existing
output directories. No new full solver timing or Flow* timing is part of this
experiment. See the [Chinese report](../../docs/range_batch_device/REPORT_PLAIN_CHINESE.md)
and [operator contract](../../docs/range_batch_device/OPERATOR_CONTRACT.md).
