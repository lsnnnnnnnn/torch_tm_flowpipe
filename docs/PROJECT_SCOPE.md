# Project scope

The active scope is plant-only polynomial ODE reachability with Taylor models:
the PyTorch TM kernel, one- and multi-step propagation, Flowstar and DiffReach
adapters, common/native comparison contracts, matched-basis diagnostics, and
versioned correctness/runtime reporting.

In scope are interval and polynomial arithmetic, truncation and remainder
handling, validation, raw versus tightened bounds, reset/preconditioning,
failure/completion semantics, repeated runtime measurement, and independent
acceptance.

Out of scope are neural-network controllers, NNCS closed loops, CROWN,
auto_LiRPA, BERN integration, Jacobian/sensitivity bounds, new adaptive-basis
algorithms, transcendental dynamics, hybrid automata, guards/jumps, and
rewriting or binding Flowstar. Course attachments and literature-map work are
unrelated and are absent from the active tree.
