# Old-author QUAD P3 weighted256: new full-horizon candidate

Status as of 2026-10-03 Asia/Shanghai: **completed and directly compared**.
The unique server directory is
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/p3_quad_old_weighted256_full1000_nohash_20261003_001/`.
This is a new 256-row implementation candidate; the archived 128-row
1,000-step runs were not restarted or modified.

[Automatic preflight](PREFLIGHT.json) passed before launch. It directly checked
the archived 1,000-step/1,024-box/50-control completion, presence of all 1,000
archived observer PT files, exact old-author full YAML, 40/40 pooled steps from
the already completed 256-row short candidate against the old full run,
precompiled extensions, source, unused run ID, related processes, and idle
GPU 3. The [launcher](launch_once.sh) repeated the run-ID and GPU checks.
The [supervisor START](run_001/START.json), [solver START](run_001/data/START.json),
and [actual generated config](run_001/data/config.yaml) are preserved locally.

The [runner](runner.py) retains the old author x2/x4/x5 equations, all 1,024
initial boxes, 50 controls and 1,000 steps, working P3/point P2/validation
P4, both early-weighted rounds, strict endpoint/injection, full SR/K20,
trig reuse, and per-step observer. The active weighted graph block changes
from 128 to 256 rows in [weighted_chunk256.py](weighted_chunk256.py); ordinary
validation stays at 128 rows. The run and observer wrapper also differs from
the archived full run, so wall-time attribution to the block change is unavailable.
The runner stops at first rejected numerical
step and saves original per-step 1,024×12×4 bounds, accepted masks and statuses
as NPY files for direct comparison with the archived PT data. Per-step bulk
arrays remain on the server; compact original receipts and final per-box arrays
are copied here. [compare_full_saved.py](compare_full_saved.py) checked all 1,000
steps by direct numerical bytes after the run. The source blocks content digest,
TorchScript and CUDA compilation entry points; saved modules are loaded.

Outer no-digest supervision had a 2,520-second cap. The nested original-style
[resource watchdog](nncs_watchdog_gpu14.py) had a 2,400-second cap, 14 GiB
measured owned-GPU and 11.5 GiB process-tree RSS guards; the runner retains
the 13.5 GiB Torch allocator cap. Initial live PIDs were 567436 (supervisor),
567438 (resource watchdog), and 567439 (solver); all have exited.

After launch, a read-only check of the first 40 new per-box observer NPY files
against archived `observer_1.pt` through `observer_40.pt` found all bounds,
accepted masks and statuses directly equal by bytes. That evidence covers only
the first 40 steps. The older full run's observer PT is a pre-endpoint stage;
the short candidate's final arrays were computed after driver completion and
are therefore not used as per-step reference data.

The [outer RESULT](run_001/RESULT.json) is `completed/exit0`, wall
**1093.267186 s**, without timeout. The [inner RESULT](run_001/data/RESULT.json)
reports 1,000/1,000 completed substeps, **1,024,000/1,024,000 accepted
lane-steps**, 50 controller calls, no broken lane, and driver time
**1086.231490 s**. The original [watchdog receipt](run_001/watch/process.json)
reports process wall **1093.192982 s**, exit 0, peak owned GPU
14,925,430,784 bytes below its 14 GiB guard, and peak process-tree RSS
6,008,995,840 bytes. Weighted graph map calls were **8,000** for 2,048,000
target rows with no padding; the ordinary validation block remained 128 rows.
The [metrics](run_001/data/metrics.json), [1,000 observations](run_001/data/observations.jsonl),
[progress](run_001/data/progress.json), final tube/endpoint/status arrays,
stdout and stderr are preserved locally. The 1,000 per-step NPY files remain
in the server directory above because they occupy hundreds of megabytes.

The [full saved-value comparison](FULL_SAVED_COMPARISON.json) returned
`all_1000_saved_steps_direct_equal`: every step's 1,024×12×4 bounds,
accepted mask and status matched the archived old-author P3/trig observer PT
by direct bytes; all 1,000 pooled observations and count/status rows also
matched. The reference has no separately saved post-driver final NPY arrays,
so those candidate arrays are preserved but not claimed directly identical
to absent reference arrays. This establishes equality of the checked saved
numerical outputs, not every hidden intermediate TM/SR value or an independent
floating-point NNCS certificate.

The archived old-author P3/trig watchdog time was one process at
1533.752052 s, versus this new watchdog's 1093.192982 s. Both finished the
same requested horizon and checked numerical outputs, but the runs were on
different dates with no interleaved same-day 128-row control. This pair is a
descriptive single-run comparison, **not** a stable or causal speedup claim.
The shortened 40-step candidate's `FALSIFIED` was only an early application
of the T=5 target checker; the completed full candidate did not run an
independent full-time reach-and-remain checker.
