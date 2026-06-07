# Flowstar Source Queue Semantics Audit

## Scope

This is a clean-room source-guided note for the local checkout at `/srv/local/shengenli/flowstar`. It records file and function names consulted and summarizes behavior in project-local words. No Flow* source code is copied into this repository.

## Files Consulted

| Topic | Local Flow* file/function names consulted | PyTorch files checked |
| --- | --- | --- |
| Queue state shape | `flowstar-toolbox/Continuous.h`: `Symbolic_Remainder`; `flowstar-toolbox/Continuous.cpp`: constructors, `reset` | `src/torch_tm_flowpipe/symbolic_remainder.py`: `FlowstarSymbolicRemainderQueue` |
| Queue update | `flowstar-toolbox/Continuous.cpp`: `Flowpipe::advance` overloads and adaptive variants around the symbolic-remainder path | `flowstar_normalized_insertion_linear_queue_v2_reset`, `_propagate_queue_v2` |
| Queue limit | `flowstar-toolbox/Continuous.h`: `ODE::reach_symbolic_remainder*`; `flowstar-toolbox/Discrete.h`: symbolic-remainder reach loops | v2 max-size tests in `tests/test_symbolic_remainder.py` |
| Scalars | `flowstar-toolbox/Continuous.cpp`: symbolic scalar update; `flowstar-toolbox/Matrix.h`: `right_scale_assign` | `_right_scale_matrix`, `_inverse_scales` |
| Normal insertion | `flowstar-toolbox/TaylorModel.h`: `TaylorModel::insert_ctrunc_normal`, `TaylorModelVec::insert_ctrunc_normal`, `HornerForm::insert_ctrunc_normal`; `flowstar-toolbox/Continuous.cpp`: `Flowpipe::compose_normal` | `insert_ctrunc_normal_like`, `_flowstar_normalized_insertion_transition` |

## Semantics Summary

`Symbolic_Remainder` stores three queue channels: interval columns `J`, real matrices `Phi_L`, and per-coordinate `scalars`, plus `max_size`. Constructors initialize scalar entries to one. `reset(dim)` clears `J` and `Phi_L` and restores unit scalars while preserving the configured limit.

Flow* extracts the linear part of the accepted endpoint map, right-scales that linear map by the previous scalar vector, left-multiplies existing `Phi_L` entries by the current scaled map, pushes the current scaled map, then accumulates older queued `J` columns through those updated matrices. The current step's local remainder becomes a new `J` column.

The scalar direction is inverse magnitude. After a local range is evaluated, Flow* forms coordinate magnitudes `S`, stores inverse entries in `symbolic_remainder.scalars`, and scales the current flowpipe by those inverse entries. `Matrix::right_scale_assign` applies the scalar vector column-wise to the next linear map.

The queue limit is checked after an accepted step in the reach loop. When `J.size() >= max_size`, Flow* resets the queue before the next step. The clean-room v2 implementation mirrors the externally visible post-step state: when appending would reach the configured size, the returned queue is reset, so a max size of 100 yields observed stored sizes up to 99 in the next-step state.

## V2 Mapping

`normalized_insertion_symqueue_v2` stores `J`, `Phi_L`, inverse scalars, linear-map entries, queue counts, and output-only symbolic widths. It uses the degree-1 part of the clean-room normalized insertion endpoint as the current linear map. It leaves the ordinary target-check remainder clean and reports propagated older queue width as output/range-only contribution.

The new tests pin three source-guided invariants: reset after reaching `max_size`, inverse scalars as right column scales, and current-left multiplication order for existing `Phi_L` entries.

## Known Gaps

The v2 path does not claim exact Flow* parity. It does not implement every Flow* preconditioning, Horner insertion, invariant contraction, or source-order detail. Nonlinear terms remain in the ordinary Taylor model and range accounting rather than being fully separated into Flow*'s symbolic queue machinery. Exact ordering across all Flow* overloads remains source-guided, not proven equivalent.

## Target And Output Channels

Flow* symbolic-remainder paths can combine current and propagated symbolic remainders into the local flowpipe remainder while also using contraction and normalization. This branch's v2 diagnostic deliberately separates channels: target validation sees the ordinary clean reset channel, while output/range accounting includes propagated symbolic contributions. That separation is diagnostic evidence, not a backend parity claim.
