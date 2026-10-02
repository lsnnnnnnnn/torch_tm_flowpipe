# Old-author QUAD P3: isolated same-wrapper 128-row full control

Status: **running; no final result yet**. The unique new server
ID is `p3_quad_old_weighted128_full1000_samewrapper_nohash_20261003_001` under
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/`.
This does not restart or edit the historical 128-row full run or the completed
256-row candidate.

The [preflight receipt](PREFLIGHT.json) passed on the server before launch,
including CPU padding/ownership checks, all 1,000 archived observer files,
3,000 saved 256-arm arrays, idle GPU 3 and no related process. The original
[outer START](run_001/START.json) and [driver START](run_001/data/START.json)
were mirrored after the single launch at `2026-10-02T22:28:02Z`; initial
supervisor PID was 605627. The job remains under the stated resource and time
guards. These START files do not establish a completion time or numerical result.

The [runner](runner.py), [launcher](launch_once.sh), [resource watchdog](nncs_watchdog_gpu14.py),
old-author [YAML](quad_author_resolved.yaml), and [step comparator](compare_full_saved.py)
come from the completed [256-row full candidate](../weighted_chunk256_full1000_v1/README.md).
Only the active early-weighted CUDA graph scratch/padding block changes from
256 to 128 rows in [the adapter](weighted_chunk128.py); the runner changes its
adapter path and identifying labels. Ordinary validation remains 128 rows.
The old-author x2/x4/x5 equations, 1,024 initial boxes, 50 controls, 1,000
small steps, working P3/point P2/validation P4, two early-weighted rounds,
strict endpoint/injection, full host K20 SR, trig reuse, observer mode, and
saved numerical arrays stay on the same contract. The historical short
`weighted_chunk128.py` contains digest-related constants and is not used.

[Preflight](preflight.py) must pass before any launch. It checks direct source
text and YAML equality against the completed 256 arm except for the expected
128-row substitutions, the completed historical reference and 256 receipts,
all 1,000 old observer files and 3,000 saved 256 arrays, model and precompiled
extensions, unused run ID, related processes, and idle physical GPU 3. Its
adapter check is CPU padding/ownership only. The launcher repeats the run-ID
and idle-GPU checks. There is no content digest check. A failed gate means no
launch; a rejected numerical step means stop at that step without retrying the
same ID.

The full run uses the same `taskset -c 14-17`, 2,400-second inner watchdog,
2,520-second outer supervisor, 14 GiB measured owned-GPU guard, 11.5 GiB
process-tree RSS guard, and runner Torch allocator limit as the 256 arm. If it
completes, compare each saved 1,024×12×4 bounds array, acceptance mask,
status, pooled observation, and count/status row against all 1,000 archived
`observer_*.pt` files with `compare_full_saved.py`; the completed 256 arm already
passed this same direct comparison. Also inspect 1,000/1,000 steps,
1,024,000/1,024,000 accepted box-steps, 50 controller calls, resource receipts,
and weighted map counters. With 2,048,000 target rows divisible by 128, the
expected weighted graph count is 16,000 with zero padding, but the receipt must
establish the observed count.

This same-wrapper pair can reduce implementation differences in a wall-time
comparison. One 128 run after one 256 run still leaves order, warm state, and
system load uncontrolled; a single difference cannot establish a stable or
causal speedup. No independent full-time NNCS certificate follows from direct
saved-value equality.
