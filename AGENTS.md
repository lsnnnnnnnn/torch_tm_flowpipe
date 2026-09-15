# Research review branch

The supported project scope is plant-only polynomial ODE flowpipes for Van der Pol and Brusselator. Start with `README.md`, `docs/PROJECT_REVIEW.md`, and `experiments/review_suite/registry.yaml`. The sole numerical package is `src/torch_tm_flowpipe`.

Keep frozen run packages immutable. A changed implementation or benchmark contract needs a new run identity; derived tables and figures must name their exact source. Report endpoint and whole-segment tube bounds separately, including all steps and both coordinates. Distinguish current CPU reference, optional GPU range and resident prototypes, historical mechanism tests, and withdrawn claims. A green local test never implies an end-to-end numerical proof.

Use `python -m experiments.review_suite.cli` for supported review operations. Historical runners remain available for fixed-revision replay, but their output and claims are not the current review index. Do not silently alter numerical thresholds, floating-point order, CUDA defaults, or old evidence while maintaining the review branch.
