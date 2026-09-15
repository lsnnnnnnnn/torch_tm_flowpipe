# Experiment entrypoints

The supported research-review entrypoint is `python -m experiments.review_suite.cli`. It lists the selected studies, re-observes immutable saved objects, redraws the main figures, runs two-step smoke checks, and delegates current full-horizon reruns to the established CPU and GPU-range runners. [The registry](review_suite/registry.yaml) is the sole current experiment index; [the guide](../docs/REPRODUCING.md) gives exact commands.

Sibling directories still contain real dependencies and frozen historical runners. In particular, the endpoint repair imports `run_vdp_dense_backend`, `run_brusselator_sr1000_parity`, and the tracked matched-contract artifact; the live range and packet routes import their original helpers. Keep these as support until their import and subprocess paths are migrated deliberately. A historical runner does not become a supported current backend merely because its source remains here. No C5 algorithm or whole GPU engine exists.
