# Complete PyTorch engine experiment

The supported user entrypoint is now `python -m experiments.review_suite.cli`:
see [complete-engine commands](../../docs/REPRODUCING.md#complete-pytorch-engine).
The direct candidate and campaign commands below also remain available for
frozen research replay.

This adapter runs Huan's sparse engine through every accepted step. It does not
call the old CPU range-service solver. The supported tasks are the frozen Van
der Pol and Brusselator plants, with B1 original boxes, B2 partition lanes 0/31,
or the existing B32 partition. CPU and CUDA are explicit choices.

Use the external engine revision **280abb400610f56210a7a5be61d5f98be3e27251**:
it includes the fresh-error symbolic-history repair and makes strict SR
normalization use `settings.cutoff`. The earlier cff8758 experiment retains an
internal 1e-4 cutoff and did not complete either original-box 1000-step horizon.
Do not substitute its performance label for the final engine version.

On the current server, from this repository:

```bash
export PYTHONPATH=src:.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_HOME=/usr/local/cuda-12.6
export CXX=/srv/local/shengenli/.huan-audit-gxx13/bin/x86_64-conda-linux-gnu-g++
export CC=/srv/local/shengenli/.huan-audit-gxx13/bin/x86_64-conda-linux-gnu-gcc
export TORCH_EXTENSIONS_DIR=../cache/py311_torch251_cu126_gcc13
export TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=2
```

Select an available GPU explicitly with `CUDA_VISIBLE_DEVICES`. The GPU index
used in old receipts is a historical observation, not a reservation. Then use
a new, nonexistent output directory:

```bash
/srv/local/shengenli/miniforge3/envs/py11/bin/python \
  -m experiments.whole_engine_feasibility.candidate \
  --engine-root ../engine_cutoff_aligned \
  --plant van_der_pol --batch 1 --steps 1000 --device cuda \
  --record --output ../runs/my_new_vdp_run
```

Use `--plant brusselator` for the second frozen contract. For CPU, hide CUDA
with `CUDA_VISIBLE_DEVICES=''` and use `--device cpu`. For a timing-only solve,
omit `--record`; this intentionally does not produce all-step physical bounds.

Observe a recorded run separately:

```bash
/srv/local/shengenli/miniforge3/envs/py11/bin/python \
  -m experiments.whole_engine_feasibility.observe_saved \
  --input ../runs/my_new_vdp_run/factored.jsonl.gz \
  --output ../runs/my_new_vdp_run_observed
```

The observer uses exact rational interval-coefficient substitution, preserves
all degrees, and checks every accepted endpoint/tube x/y record. It can be much
slower than solving: the recorded final B1 runs took approximately 57 s (VDP)
and 819 s (Brusselator) to observe offline. This cost is explicitly excluded
from solver timing, as physical observation is also excluded from the old Gr
timing boundary. The factored models remain available without full expansion.

`summary.json` reports numerical completion separately from process success.
Always check `completed`, `accepted_steps`, and `failure`; an ordinary numerical
rejection is recorded without changing the fixed step or returning a fabricated
model. A batch is committed atomically, and speculative SR copies provide real
rollback. The copy cost is included in solve time.

`campaign.py` reproduces the finite experiment phases. Timing campaigns first save a separate B32×2 warmup for each plant. All three CUDA extensions (including the lazily loaded validation extension) are checked before the solve timer; cold builds remain in startup/process time. The extension-cache path is resolved to avoid rebuilding identical sources merely because a new run directory uses a symlink. Its historical default
engine directory is `../engine`; pass `--engine-root ../engine_cutoff_aligned`
for the final variant. The 2026-09-21 old-Gr paired measurements used cff8758,
whereas the separate same-engine CPU/CUDA measurements used 280abb4.

The two Python backends were tested with Python 3.11.15 and PyTorch 2.5.1+cu121.
The external package declares a newer deployment environment; the tested older
environment is an explicit source-path configuration. CUDA extensions were
built with the existing GCC13/CUDA12.6 toolchain. No engine source is vendored
into this repository, and the external repository keeps its existing license.

Tests cover scoped arithmetic repairs, failure rollback, lane isolation, history
reset, and export arithmetic. They are not a whole-engine formal proof. The
final full-range comparison also retains the old Flow* last-step domain caveat.
