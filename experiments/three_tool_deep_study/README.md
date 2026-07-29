# Torch TM, DiffReach, and Flow* deep study

## Authoritative delivery

The authoritative branch is `codex/torch-flowstar-diffreach-deep-study`; the
default branch may be stale. A run becomes authoritative only after the
ten-repetition final acceptance, artifact-quality audit, and `RUN_COMPLETE`
gate all pass. `curate_artifacts.py` then copies the commit-safe evidence to
`artifacts/authoritative/<run-id>/`, and `generate_final_delivery.py` generates
the detailed English/Chinese reports, artifact index, reproduction guide, and
table/plot manifest.

No result is interpreted as a universal cross-tool same-order winner. Pareto
dominance is within one tool, system, and absolute time only; deterministic
trajectory samples are non-proof sanity checks.

This experiment replaces the ill-defined question “which implementation is
best at order 1?” with controlled representation contracts and native-tool
comparisons. The canonical systems and all protocol settings live in
`benchmark_spec.yaml`.

The study has five layers:

1. one validated local segment from the same box, ODE, and step;
2. a common affine endpoint carry;
3. a common componentwise-box carry;
4. native low-order configurations with their true basis semantics;
5. native practical configurations summarized as width/runtime/horizon Pareto
   frontiers.

The common JSON segment format is analysis-only. Native arithmetic and native
validation remain inside each solver. Torch exports its sparse exponent map,
DiffReach exports `c`, `L`, `Lt`, its remainder, and the composed physical
parameterization, and Flow* exports the official `Flowpipe::compose` result.

Quick validation:

```bash
./experiments/three_tool_deep_study/run_smoke.sh
```

Full foreground run:

```bash
./experiments/three_tool_deep_study/run_all.sh
```

Complete isolated tests against a finished artifact:

```bash
DEEP_STUDY_RESULTS_DIR=/path/to/results/<run-id> \
  ./scripts/run_complete_pytest.sh
```

Background run:

```bash
./experiments/three_tool_deep_study/launch_background.sh
```

Every timestamped result directory includes provenance, frozen-input checksum
manifests, raw rows, protocol summaries, correctness gates, the primary CPU
runtime/Pareto tables, a separately labelled native CPU/CUDA capability and
throughput table, plots, logs, and a generated report. Existing result
directories are never regenerated in place.
