# Old-author QUAD P3 early-weighted two-round gate

Date: 2026-10-02 UTC / 2026-10-03 Asia/Shanghai. This is **one new isolated
40-step diagnostic**, not a restart or continuation of the archived 1,000-step
run. Before launch, the saved old-author 40-step process was absent, the
`observer_on_002` reference and its configuration were present, physical GPU 3
was idle, and the new `run_001` directory did not exist. The
[supervisor START](run_001/START.json) records the exact command, preexisting
runner, comparison helper, 120-second limit, GPU and CPU affinity. The saved
runner checks the old-author `x2/x4/x5` equations and configuration by direct
bytes, blocks content-digest calls and the TorchScript/extension-build entry
points used by this route, then loads the
existing precompiled CUDA modules. No content hash was calculated or checked.

The [new probe](round_probe.py) wraps the existing `refine_accepted` and
`validate_failed` calls without changing their arguments, return values, or
two-round algorithm. It captures each round's accepted mask and proposed
remainder, reconstructs each intersection, and compares the final reconstructed
remainder to the engine result. It also stops at the first rejected numerical
step. [Independent full-array audit](audit_full_arrays.py) replays all 80
saved rounds and compares the final physical arrays and 40 observation rows to
the earlier `observer_on_002` reference by direct values/bytes.

| Gate | Saved result |
| --- | --- |
| Outer process | `completed`, exit 0, 60.178204 s; no timeout |
| Numerical coverage | 40/40 substeps, 40,960/40,960 box-steps accepted, 2 controller calls, `broken=0` |
| First weighted round | 40,960/40,960 optional self-maps accepted; all 40,960 box-rounds changed the remainder; 518,869 interval endpoints changed |
| Second weighted round | 40,615/40,960 optional self-maps accepted; every accepted box-round changed the remainder; 564,916 endpoints changed |
| Second-round non-acceptances | Steps 7/16/17/19/38 accepted 1,022/825/896/1,009/1,023 of 1,024 boxes, respectively; the other candidate bounds were retained from round one |
| Numerical equivalence gate | Each of 40 reconstructed final per-box remainders is bitwise equal to the engine result; all 40 observation rows and final 1,024×12 tube/endpoint/status arrays directly equal `observer_on_002` |

The original shortened driver prints `FALSIFIED` because it applies the
unchanged T=5 endpoint target at t=0.2. This is not a T=5 property verdict.

The second round's sum of width reductions across the saved 40×1,024×16
box-state rows is `1.3863845427716655e-05`; small but nonzero. For example,
at step 1, lane 0, state 0, its remainder lower endpoint changes from
`-1.8871081076672917e-09` after round one to
`-1.8862904384674295e-09` after round two. The complete 1,024-bit masks per
round are in [ROUND_ACCEPTED_MASK.npy](run_001/data/ROUND_ACCEPTED_MASK.npy),
and [ROUND_ROWS.jsonl](run_001/data/ROUND_ROWS.jsonl) has every round's mask,
counts, timings, and first three changed lanes with all 16 interval remainders.
The [full-array audit receipt](run_001/FULL_ARRAY_AUDIT.json) records all-round
reconstruction, counts, and direct comparison; the [probe receipt](run_001/ROUND_PROBE.json)
and [solver result](run_001/data/RESULT.json) retain the original status.

The full `STEP_CURRENT_BEFORE.npy`, `STEP_CURRENT_AFTER.npy`, and
`ROUND_NEW_REMAINDER.npy` source arrays remain on the server under
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/p3_quad_old_weighted_round_gate_20261003_001/run_001/data/`.
The repository contains the compact original receipts, all masks, change
witnesses, observations and final arrays, avoiding a second copy of roughly
42 MB of intermediate arrays. The server also retains the exact probe and
audit scripts and complete stdout/stderr. The local [stdout](run_001/stdout.log)
and [stderr](run_001/stderr.log) are copied unchanged.

The two measured CUDA stream spans are 15.633157 s and 14.834444 s. The probe
adds events, synchronization after each round, and diagnostic GPU-to-CPU
copies; these spans are **not** an estimate of removable runtime. The direct
numerical result rules out omitting round two as an implementation-equivalent
shortcut even on this 40-step prefix. It does not measure or establish any
1,000-step speedup, full-horizon equivalence, or end-to-end certificate.
