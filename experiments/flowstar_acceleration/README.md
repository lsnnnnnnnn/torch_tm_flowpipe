# Strict PyTorch/CUDA acceleration

The original-box VDP and Brusselator routes use strict Horner composition,
structural Picard supports, polynomial deferred-leaf validation, directed SR
kernels, certified injective CUDA writes and atomic batch transactions. The numerical contracts (initial box,
h, order, cutoff, remainder estimate, refinement and SR reset) remain frozen.
VDP explicitly uses the equivalent RHS `(1-x*x)*y-x`.

The supported review entry checks a clean pinned engine revision, passes every
algorithm option explicitly and rejects incomplete solves or a mismatched
summary. The original `whole-engine` backend still selects its older baseline.
The `whole-engine-accelerated` backend selects the revision in
`experiments/review_suite/cli.py`; a subsequent qualified revision must update
that pin before the entry accepts it.

## Server environment

```sh
run_root=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z
python_bin=/srv/local/shengenli/miniforge3/envs/py11/bin/python
cd "$run_root/review_release"
export PYTHONPATH="$run_root/review_release/src:$run_root/review_release"
export PATH="/srv/local/shengenli/miniforge3/envs/py11/bin:$PATH"
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CXX=/srv/local/shengenli/.huan-audit-gxx13/bin/x86_64-conda-linux-gnu-g++
export CC=/srv/local/shengenli/.huan-audit-gxx13/bin/x86_64-conda-linux-gnu-gcc
export CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR="$run_root/cache_qualified_six/py311_torch251_cu126_gcc13"
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The review_release worktree is a clean, committed copy of the supported adapter.
The development repo retains separate untracked benchmark experiments; the
supported run command intentionally requires a clean source checkout.

## Run and observe a complete original-box trajectory

Every output directory must be new. `brusselator-fixed-full` selects the other
1000-step task. `smoke` uses two steps on both plants.

```sh
"$python_bin" -m experiments.review_suite.cli smoke \
  --backend whole-engine-accelerated --engine-root "$run_root/engine_accelerated" --device cuda
"$python_bin" -m experiments.review_suite.cli run --experiment vdp-fixed-full \
  --backend whole-engine-accelerated --engine-root "$run_root/engine_accelerated" \
  --device cuda --out "$run_root/runs/vdp_accelerated_user_run"
"$python_bin" -m experiments.review_suite.cli observe --arithmetic outward \
  --input "$run_root/runs/vdp_accelerated_user_run/factored.jsonl.gz" \
  --out "$run_root/runs/vdp_accelerated_user_observed"
```

`--arithmetic outward` uses full polynomial substitution with interval
coefficients and directed binary64 operations. No coefficient cutoff or model
truncation occurs in observation. The default `fraction` route remains the
exact rational reference. Both outputs retain source hashes and separate
observer time from solver time; incomplete requested horizons are rejected by
the outward route.

## Real 32-way partition, 20 steps

The adapter reads the frozen 8x4 partition; lanes are distinct subboxes. This
explicit adapter also supports a diagnostic CPU run. Source identity and every
algorithm choice are recorded in `summary.json`.

```sh
"$python_bin" -m experiments.whole_engine_feasibility.candidate \
  --plant van_der_pol --batch 32 --steps 20 --device cuda \
  --engine-root "$run_root/engine_accelerated" --composition horner --glue graph \
  --support-policy structural --validation-policy defer_polynomial --rhs-form regrouped \
  --injective-maps on --injective-glue on \
  --record --output "$run_root/runs/vdp_b32_user_run"
```

For Brusselator, use `--plant brusselator --rhs-form original`. The same
`observe --arithmetic outward` command accepts either complete partitioned task.
Omitting `--record` produces timing-only solves and cannot support subsequent
full observation.

## Evidence and remaining work

The pinned engine is 1a27bf58fdadcc8d429044304f19d725c8386cab. It includes the
endpoint FULL-to-spatial support correction and both injective-write stages.
Four complete qualification tasks (3280 lane-steps) match the corrected 29db74a
checkpoint byte for byte after decompressing their factored records. All 16
width channels meet p95<=1.10 and max<=1.25 against fixed-count native Flow*
over identical full local domains, including step 1000. There are 72 independent
Fraction bound checks; 75 focused CPU/CUDA regressions pass on the final engine.
Worst original-box p95/max are 1.0995047211/1.2127639702 for VDP and
1.0122278389/1.0383255462 for Brusselator. Evidence lives under
runs/qualified_six_endpoint_002 and runs/qualified_final_glue. These finite
checks are numerical evidence, not a formal proof of the entire implementation.

The overall acceleration goal remains active: VDP timing is still above the
target. Single recorded-run timings are diagnostics, not five-pair performance
claims. `cold_startup_s` covers CUDA/determinism/extension preflight; full process
time also includes Python imports and teardown. `solve_wall_s` includes plans,
state construction, graph capture, all advances, SR copies, status and guards.
Actual loaded extension paths and binary hashes are recorded.

Expanded NNCS runs, original configuration failures and resource-limited runs
are reported separately. Their CROWN controller bounds are still marked
`controller_unqualified`; successful plant propagation alone does not establish
strict end-to-end NNCS verification. Airplane has completed one original plant step with sparse metadata; its full
controller loop and adaptive+SR remain separate engineering work.
