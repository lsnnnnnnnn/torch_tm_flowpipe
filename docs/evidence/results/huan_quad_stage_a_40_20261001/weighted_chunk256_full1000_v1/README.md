# Old-author QUAD P3 weighted256: new full-horizon candidate

Status as of 2026-10-03 Asia/Shanghai: **running**, with no final result yet.
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
trig reuse, and per-step observer. Only the active weighted graph block changes
from 128 to 256 rows in [weighted_chunk256.py](weighted_chunk256.py); ordinary
validation stays at 128 rows. The runner stops at first rejected numerical
step and saves original per-step 1,024×12×4 bounds, accepted masks and statuses
as NPY files for direct comparison with the archived PT data. Per-step bulk
arrays remain on the server; compact receipts will be copied here after
completion. [compare_full_saved.py](compare_full_saved.py) will check all 1,000
steps by direct numerical bytes after the run. The source blocks content digest,
TorchScript and CUDA compilation entry points; saved modules are loaded.

Outer no-digest supervision has a 2,520-second cap. The nested original-style
[resource watchdog](nncs_watchdog_gpu14.py) has a 2,400-second cap, 14 GiB
measured owned-GPU and 11.5 GiB process-tree RSS guards; the runner retains
the 13.5 GiB Torch allocator cap. Initial live PIDs were 567436 (supervisor),
567438 (resource watchdog), and 567439 (solver). This file records launch
state, not a completed time or speed result.

After launch, a read-only check of the first 40 new per-box observer NPY files
against archived `observer_1.pt` through `observer_40.pt` found all bounds,
accepted masks and statuses directly equal by bytes. That evidence covers only
the first 40 steps. The older full run's observer PT is a pre-endpoint stage;
the short candidate's final arrays were computed after driver completion and
are therefore not used as per-step reference data.
