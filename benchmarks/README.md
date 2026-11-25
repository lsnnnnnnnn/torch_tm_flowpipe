# Benchmark inputs

Look up a case by its `case_id` in the [asset index](assets.json) or the [comparison results](../results/benchmarks.json). The index covers 24 cases. The eight vertical collision avoidance cases use the same set of nine advisory controllers. Upstream file paths are relative to the pinned [ARCH-COMP 2026 benchmark repository](https://github.com/Kiguli/ARCH-COMP2026/tree/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26). The [category report](https://doi.org/10.29007/637n) describes the benchmark tasks.

The [configuration files](configs/) contain the equations, initial sets, time steps and reference integration settings used by the shared driver. The two Quadrotor configurations differ in the derivatives of states `x2`, `x4` and `x5`. Select `quad_paper.yaml` for the report equations or `quad_author.yaml` for the CROWN-Reach implementation. Both use the same controller and [1,024 initial boxes](quad/initial_boxes.json). Each box lists 12 physical state intervals, time and three held controls, in configuration order. Integration orders in these files describe the supplied configurations. Experiment-specific settings belong to the individual runs.

The [bundled models](models/) include the official Double Pendulum and Quadrotor controllers and two official TORA weight files. The project’s two derived TORA networks return the plant control directly as `11 × network(state)`. Use an output scale of one and an offset of zero with these exports. The asset index records the upstream paths and SHA-256 digests of the bundled files. The Quadrotor implementation configuration and controller came from the CROWN-Reach benchmark collection.

The adaptive cruise control and single pendulum configurations load controllers from the upstream collection. To place them at the paths in the configurations, run these commands from the project root.

```sh
git clone https://github.com/Kiguli/ARCH-COMP2026.git benchmarks/upstream
git -C benchmarks/upstream checkout --detach d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26
```
