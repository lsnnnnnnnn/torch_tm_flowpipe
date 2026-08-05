# Project scope

The active project is a PyTorch-native, plant-only prototype for polynomial
ODE reachability with Taylor models. It contains one numerical core, one
multi-step API, explicit comparison contracts, read-only adapters for
Flowstar and DiffReach, and bounded diagnostics.

In scope are interval and polynomial arithmetic, total-degree truncation,
remainder construction and validation, endpoint/segment/tube semantics,
raw/tightened separation, reset and preconditioning, failure/completion
semantics, deterministic finite domain subdivision for polynomial range
enclosure, safe terminal-state replay, repeated runtime measurement,
provenance, and independent acceptance.

Out of scope are NNCS closed loops, controllers, CROWN/auto_LiRPA, BERN,
Jacobian or sensitivity bounds, new adaptive-basis algorithms,
transcendental dynamics, hybrid guards/jumps, and rewriting or binding
Flowstar. Historical NNCS/BERN branches are inventoried but not migrated.

The previously unidentified external PyTorch Taylor-model implementation is now
identified as Xiangru's clean `27d29050...` complete-Q3 implementation and has a
field-level audit in `docs/XIANGRU_VS_OUR_TORCH_TM_CODE_AUDIT.md`.  That evidence
does not expand this repository's scope: Xiangru's NNCS/controller orchestration
and TORA-specific closed-loop machinery remain outside this plant-only numerical
core, and no external source is vendored here.
