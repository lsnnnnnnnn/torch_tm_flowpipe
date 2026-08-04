# External PyTorch Taylor-model audit preconditions

Status: satisfied for identity and code-level inspection; interval-soundness audit
still required.

The external implementation is identified as Xiangru's complete-Q3 PyTorch
Taylor-model implementation at clean commit
`27d29050a5f214b56f211ca9cb411e734ed80230` in the author repository
`https://github.com/xiangruzh/CROWN-Reach_Development.git`.  The native B48 result
and field-level code comparison are documented in `XIANGRU_NATIVE_REPRODUCTION.md`
and `XIANGRU_VS_OUR_TORCH_TM_CODE_AUDIT.md`.

Identity does not imply scope equivalence.  The successful route includes a
TORA-specific NNCS/controller orchestration layer; that controller and closed-loop
code remains outside this repository's plant-only numerical core.  The Q3 dynamics
path uses ordinary float64 interval operations and is classified empirical, so a
separate interval-soundness audit remains the next relevant external-code task
after the Flow* scalar-affine diagnosis.

The completed identity record includes:

- paper title;
- authors and thesis context;
- paper URL or DOI;
- repository URL;
- license;
- exact commit or tag;
- supported environment;
- claimed algorithm or guarantee;
- benchmark source.

The audit used a sibling clean checkout and captured repository, environment,
dependency, dirty-state, and artifact provenance.  No external source is vendored
or copied into this repository.

A code-level comparison must cover at least:

- state/set representation;
- interval arithmetic and outward-rounding/soundness contract;
- polynomial representation, order, and basis;
- time-variable treatment;
- truncation and remainder generation;
- Picard/self-map validation and step acceptance;
- multi-step dependency and reset/preconditioning;
- endpoint versus segment/tube outputs;
- CPU/GPU batching, autograd, and dtype;
- supported dynamics and failure semantics;
- tests and runtime boundary.

The completed code audit reports unknowns as unknown and does not infer
algorithmic equivalence from the project name, nominal order, or benchmark label.
It expressly does not establish end-to-end directed rounding.
