# External PyTorch Taylor-model audit preconditions

This is a template for a later audit. The external implementation is not yet
identified; this repository does not guess its identity.

Before cloning or comparing code, record:

- paper title;
- authors and thesis context;
- paper URL or DOI;
- repository URL;
- license;
- exact commit or tag;
- supported environment;
- claimed algorithm or guarantee;
- benchmark source.

Use a sibling checkout, keep it read-only, and capture its own repository,
environment, dependency, dirty-state, and patch provenance. Do not vendor or
copy its source into this repository.

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

The later audit must report unknowns as unknown and must not infer algorithmic
equivalence from a project name, nominal order, or benchmark label.
