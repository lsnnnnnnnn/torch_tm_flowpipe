# Project scope

The active project is a PyTorch-native, plant-only prototype for polynomial
ODE reachability with Taylor models. It contains one numerical core, one
multi-step API, explicit comparison contracts, read-only adapters for
Flowstar and DiffReach, and bounded diagnostics.

In scope are interval and polynomial arithmetic, total-degree truncation,
remainder construction and validation, endpoint/segment/tube semantics,
raw/tightened separation, reset and preconditioning, failure/completion
semantics, repeated runtime measurement, provenance, and independent
acceptance.

Out of scope are NNCS closed loops, controllers, CROWN/auto_LiRPA, BERN,
Jacobian or sensitivity bounds, new adaptive-basis algorithms,
transcendental dynamics, hybrid guards/jumps, and rewriting or binding
Flowstar. Historical NNCS/BERN branches are inventoried but not migrated.

An external “PhD PyTorch Taylor-model implementation” has not been
identified. This cleanup neither guesses its identity nor clones, vendors,
copies, or compares external code. The next audit must first satisfy
`docs/EXTERNAL_PYTORCH_TM_AUDIT_PRECONDITIONS.md`.
