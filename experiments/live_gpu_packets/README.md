# Live heterogeneous GPU range packets

This directory is the finite runner and verifier for the opt-in `Gp` route.
It never loads saved requests or answers to advance a solver. The only reused
numerical inputs are explicitly labelled parent lifecycle records for static
opportunity analysis and complete CPU/Flow* model objects for a fresh common
observer pass.

Use the existing py11 environment and one fixed CPU/GPU allocation:

```bash
export PYTHONPATH=src:.:tests
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
PY=/srv/local/shengenli/miniforge3/envs/py11/bin/python
ART=artifacts/runs/live_gpu_packets_20260910T023603Z

taskset -c 2 "$PY" -m experiments.live_gpu_packets.freeze_plan --output "$ART"
taskset -c 2 "$PY" -m experiments.live_gpu_packets.capture_tests --output "$ART/tests"
taskset -c 2 "$PY" -m experiments.live_gpu_packets.parent_opportunity --output "$ART/opportunity"
taskset -c 2 "$PY" -m experiments.live_gpu_packets.campaign \
  --phase diagnostic --output "$ART/diagnostic"
taskset -c 2 "$PY" -m experiments.live_gpu_packets.analyze diagnostic "$ART/diagnostic"
taskset -c 2 "$PY" -m experiments.live_gpu_packets.campaign \
  --phase formal --output "$ART/formal"
taskset -c 2 "$PY" -m experiments.live_gpu_packets.analyze formal "$ART/formal"
taskset -c 2 "$PY" -m experiments.live_gpu_packets.long_campaign --artifact "$ART"
taskset -c 2 "$PY" -m experiments.live_gpu_packets.compare_horizons \
  --root "$ART/full_horizon" --output "$ART/horizon_comparison"
taskset -c 2 "$PY" -m experiments.live_gpu_packets.package "$ART"
taskset -c 2 "$PY" -m experiments.live_gpu_packets.tamper "$ART"
taskset -c 2 "$PY" -m experiments.live_gpu_packets.verify_package "$ART"
```

The diagnostic matrix is fixed at 26 runs: B1/B2 three-step G0/Gp/independent
GPU, B8/B32 twenty-step G0/Gp, one online B2×120 Gp run per plant, and G0/Gp
checkpoint windows around each history reset. Captured current requests are
independently rechecked with exact rational powers, terms and ordered sums.

The formal matrix is exactly five blocks. Every block contains S, Q, G0 and Gp
for both plants at B32×20 (640 successful lane-steps per run). Its order is in
`PLAN_FROZEN.json`; no timing-driven samples may be appended. Compilation and
the one-time CUDA startup probe are recorded outside the main denominator, but
the first packet scratch allocation remains inside it.

The long campaign is separate from the timing claim. It runs one original,
unpartitioned B1 task for 1000 fixed steps in each plant. It streams every
accepted step, saves safe checkpoints, and records complete remainder/queue and
timing evidence without retaining every service group in memory. This is a
complete solver chain with GPU range participation, not a full-GPU solver and
not a formal proof of the whole ODE engine.

`package` writes deterministic root aliases, `RESULT.json`, `SOURCE_MAP.json`,
the Chinese report and an immutable `SHA256SUMS`. The manifest deliberately
excludes only later `tamper/` and `acceptance/` receipts; each such receipt
records hashes of its own logs. `tamper` changes and rehashes packet offset,
request identity, generation, epoch, endpoint, CUDA receipt, timing denominator,
successful lane-step count and complete-horizon flag, and requires the semantic
verifier to reject all nine. The default package verifier also performs four
fresh B2×2 live solves (both plants, G0 and Gp; 16 successful lane-steps). It
reloads and re-observes the committed 1000-step evidence but does not rerun those
long solves.
