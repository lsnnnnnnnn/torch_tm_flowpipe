# Torch TM, DiffReach, and Flow* deep study

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

Background run:

```bash
./experiments/three_tool_deep_study/launch_background.sh
```

Every timestamped result directory includes provenance, frozen-input checksum
manifests, raw rows, protocol summaries, correctness gates, plots, logs, and a
generated report. Existing result directories are never regenerated in place.
