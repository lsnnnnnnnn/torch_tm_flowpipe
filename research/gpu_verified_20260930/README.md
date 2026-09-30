# Latest GPU reachability checkpoint — 2026-09-30

This is the latest organized source, evidence, and reporting checkpoint for the QUAD / ARCH-COMP GPU work. The numerical source bytes are preserved from the completed experiment. Large checkpoints, compiled CUDA binaries and environment caches remain on the research server. No solver or GPU experiment was run to produce this publication.

Start with the [next-stage goal](GOAL_HUAN_QUAD_PLOTTING_ARCHCOMP_20260930.md), [current status](STATUS.md), [the source map](source/SOURCE_MAP.json), [the source snapshot guide](source/SOURCE_SNAPSHOT_README.md), and [the English report/slides bundle](report/README.md). The branch preserves the previous CPU/whole-engine review and Git history; this directory is the current entry point for the newer QUAD work.

| Path | Purpose |
|---|---|
| `source/engine/src/flowstar_gpu/` | All 36 pinned Python files from base engine `a3fb2e94ba976aaf498c4a9cb3f98165cddcc272` |
| `source/adapters/` | Exact P3, strict-boundary, reciprocal, SR, memory and trig-reuse sources, plus required checkers/prerequisites |
| `source/integration/` | Frozen shared Xiangru NNCS driver |
| `source/benchmark/` | Exact QUAD configuration, 1024 initial boxes and ONNX controller |
| `source/generated/reciprocal/` | Generated Python/CUDA source used in reciprocal qualification |
| `source/upstream_metadata/` | Unmodified upstream metadata, retained for provenance; not the tested installation recipe |
| `report/` | English 42-slide PDF/TeX, Chinese Word script, numerical tables, configurations and evidence copies |
| `evidence/native_terminal_20260930/` | Newly collected terminal records: original native QUAD timed out after six hours |
| `reference/` | Latest ARCH-COMP26 source index and the complete non-VCAS benchmark checklist; third-party PDF remains external |
| `handoff/` | Prior comprehensive handoff; its old native-400 snapshot is superseded by `STATUS.md` |
| `EXTERNAL_ASSETS.json` | Existing server assets required by frozen execution; files not silently replaced or bundled |
| `PUBLICATION_MANIFEST.json` | Checksums for the publication payload |
| `verify_snapshot.py` | Standard-library integrity check; does not load Torch, CUDA or ONNX |

## What the latest result establishes

Our P3 strict candidate completed all 1024 boxes × 1000 ODE steps and 50 controller updates. Reusing repeated direct-variable sin/cos evaluations reduced one full watchdog run from **3153.449456 s to 1533.752052 s**. The actual audit found all 1000 observer files and 101 controller/transfer/terminal PT files byte-identical to the previous P3 baseline, with matching final plant/SR state.

Huan's historical parity/box configuration was separately reproduced in five complete runs: **75.250099 s** median full-process time, with unchanged numerical outputs after temporary SR storage was partitioned along B. This and our strict P3 use different numerical contracts and are not an equal-guarantee speed comparison. Full end-to-end floating-point NNCS certification remains incomplete.

The original native QUAD six-hour run is now known to have timed out. Its records contain 30 complete controller periods, all 1024 lanes accepted for each period: a recorded complete-period prefix of **600 steps**. There is no full-T5 native time or terminal width. No replacement run was launched during publication.

The historical ARCH matrix contains seven configurations with five fresh-process timings per method. Those use an earlier frozen engine; they are not regressions of the latest P3/trig implementation.

## Reproduction boundary

The actual latest entry is `source/adapters/quad_fullbatch_p3_20260928/run_fullbatch_p3_trig_continue.py`. Read `report/evidence/quad_trig_INPUT.json` and `quad_trig_COMMAND.json` before attempting execution. The source retains the original absolute server paths, Git/binary identity checks, and prerequisite receipts/PT gates. This publication does not bypass those checks or claim clean-machine execution.

The measured environment was Python 3.11, PyTorch 2.5.1+cu121, CUDA 12.6 extension tools, and a V100 16GB. The preserved upstream `pyproject.toml` declares other versions; do not use it as a tested environment lock. Use the existing research environment until a separate portable runtime is validated.

Run this local integrity check from this directory:

```bash
python3 verify_snapshot.py
```

It verifies copied source/model identities, payload hashes, JSON structure and Python syntax. It does not constitute numerical validation or a new timing run.

Read [source attribution](source/ATTRIBUTION.md) before reusing external code. The name “verified handoff” refers to provenance and packaging checks; it does not claim a complete NNCS safety theorem.
