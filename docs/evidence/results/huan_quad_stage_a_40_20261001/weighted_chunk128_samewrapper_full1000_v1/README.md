# Old-author QUAD P3: isolated same-wrapper 128-row full control

Status: **completed, one full-horizon run**. The unique new server
ID is `p3_quad_old_weighted128_full1000_samewrapper_nohash_20261003_001` under
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/`.
This does not restart or edit the historical 128-row full run or the completed
256-row candidate.

The [preflight receipt](PREFLIGHT.json) passed on the server before launch,
including CPU padding/ownership checks, all 1,000 archived observer files,
3,000 saved 256-arm arrays, idle GPU 3 and no related process. The original
[outer START](run_001/START.json) and [driver START](run_001/data/START.json)
were mirrored after the single launch at `2026-10-02T22:28:02Z`; initial
supervisor PID was 605627. The [outer RESULT](run_001/RESULT.json) records
natural completion with exit 0 at `2026-10-02T22:50:33Z`, without a timeout or
failure. The [driver RESULT](run_001/data/RESULT.json) records 1,000/1,000
substeps, 1,024,000/1,024,000 accepted box-substeps, and 50 controller calls.
The completed trajectory covers the old-author five-second horizon.

A separate [read-only first-40-step comparator](compare_live_prefix40.py)
returned [40/40 directly equal saved steps](LIVE_PREFIX40_COMPARISON.json)
during the run. After completion, the [full saved-step comparison](FULL_SAVED_COMPARISON.json)
directly compared all 1,000 saved 1,024×12×4 bounds arrays, acceptance masks,
status arrays, pooled bounds, and count/status rows against the archived old P3/trig
reference. It found 1,000/1,000 equal steps with no first issue. A separate
[direct file-byte comparison](FINAL_ARRAYS_DIRECT_COMPARISON.json) found the
three final array files directly equal to the completed 256-row arm. These
comparisons use no digest. The 3,000 large per-step array files remain at the
server run path above; the original JSON, logs, observations, watchdog receipt,
and three final arrays are mirrored here.

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

The full run used the same `taskset -c 14-17`, 2,400-second inner watchdog,
2,520-second outer supervisor, 14 GiB measured owned-GPU guard, 11.5 GiB
process-tree RSS guard, and runner Torch allocator limit as the 256 arm. Its
[watchdog receipt](run_001/watch/process.json) records process wall 1,350.820362 s,
peak owned GPU 14,912,847,872 bytes below the 15,032,385,536-byte guard,
and peak process-tree RSS 6,011,478,016 bytes below the 12,348,030,976-byte
guard. The [driver RESULT](run_001/data/RESULT.json) records driver elapsed
1,343.874426 s, 16,000 weighted graph map calls for 2,048,000 target rows,
and zero padding. The [outer RESULT](run_001/RESULT.json) records wall
1,350.901441 s. Original [observations](run_001/data/observations.jsonl),
[metrics](run_001/data/metrics.json), [progress](run_001/data/progress.json),
[watchdog log](run_001/watch/stdout.log), and [driver log](run_001/stdout.log)
are preserved. The author checker printed `VERIFIED`; the runner explicitly
records `end_to_end_strict_certificate=false`.

For this same-wrapper single pair, 128-row outer wall was 1,350.901441 s and
the completed [256-row arm](../weighted_chunk256_full1000_v1/README.md) was
1,093.267186 s: 128 took 257.634255 s longer. The 128 run followed the 256
run; order, warm state, and system load were not randomized or controlled. This
descriptive pair cannot establish a stable or causal speedup. Direct equality
of saved values also does not establish an independent full-time NNCS
certificate or equality of internal TM/SR state.
