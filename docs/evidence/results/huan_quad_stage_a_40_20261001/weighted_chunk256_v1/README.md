# Old-author QUAD P3: isolated 256-row weighted graph candidate

Date: 2026-10-03 Asia/Shanghai. This was **one new 40-step candidate**, not a
restart of an archived 1,000-step run. Server source and original output are
under `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/p3_quad_old_weighted256_40_nohash_20261003_001/`.
Before launch, the old 40-step receipts and precompiled CUDA modules were
present, no related process was found, physical GPU 3 was idle, and the new
run ID was unused. The copied [old-author YAML](quad_40.yaml) directly matched
the saved 40-step YAML. The 256-row adapter's CPU padding/ownership self-check
passed; that check did not run the numerical CUDA map.

The candidate [runner](runner.py) uses the existing no-digest/no-JIT 40-step
route and its saved supporting adapters. Only the active early-weighted CUDA
graph block size changes in [weighted_chunk256.py](weighted_chunk256.py): 128
to 256 rows. Both weighted rounds, their candidates and acceptance tests, the
ordinary validation 128-row route, old `x2/x4/x5` equations, all 1,024 boxes,
P3/P2/P4 orders, strict boundary, full host K20 SR, trig reuse, controller,
and observer stay in place. The copied runner additionally stops at the first
numerical refusal. The source blocks digest constructors, CUDA compilation,
and TorchScript compilation; no digest was computed or checked. The saved
`START.json` adapter inventory lists the existing 128-row source directory,
but this run loads the new 256-row weighted adapter, as the runner and recorded
`weighted_policy` show.

| Saved result | Candidate |
| --- | ---: |
| Coverage | 40/40 substeps; 40,960/40,960 box-steps accepted |
| Controller calls / broken boxes | 2 / 0 |
| Weighted graph maps / target rows / padded rows | 320 / 81,920 / 0 |
| Ordinary validation block calls | 320, still 128 rows each |
| Outer process / driver time | 49.620295 / 42.802811 s |
| GPU peak reserved | 10,183,770,112 bytes |

[Supervisor START](run_001/START.json), [supervisor RESULT](run_001/RESULT.json),
[solver START](run_001/data/START.json), [solver RESULT](run_001/data/RESULT.json),
[metrics](run_001/data/metrics.json), [40 saved observations](run_001/data/observations.jsonl),
[stdout](run_001/stdout.log), [stderr](run_001/stderr.log), and the final per-box
arrays are copied unchanged from the server. The generated config directly
matches the saved `observer_on_002` generated config.

[Direct saved-value comparison](SAVED_COMPARISON.json), reproduced by
[compare_saved.py](compare_saved.py), passed against **both** existing
`observer_on_002` and `weighted_round_gate_v1/run_001`: all 40 observation
JSONL rows matched directly by bytes; the selected full-contract,
controller-step, and terminal numerical metrics matched; all final 1,024×12
tube, endpoint, and status arrays matched directly by bytes. This comparison
does not include every intermediate plant/SR tensor. The unchanged shortened
driver prints `FALSIFIED` because it applies the original `T=5` target at
`t=0.2`; it is not the full-horizon verdict.

The candidate's single time is **not** a causal speedup measurement against
older single runs on other dates. No matched timing pair, 1,000-step completion,
or independent end-to-end floating-point NNCS certificate has been established.
