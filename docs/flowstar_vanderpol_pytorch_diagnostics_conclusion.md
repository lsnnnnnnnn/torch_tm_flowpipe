# Flow* Van der Pol PyTorch Diagnostics Conclusion

This document is the final decision record for the Flow* Van der Pol benchmark parity and PyTorch Taylor-model diagnostics artifacts. It is diagnostic-only; it does not claim a new reachability algorithm.

## A. Flow* Parity

Flow* original/generated parity succeeded. The original Flow* benchmark and the generated Flow* C++ harness both completed to horizon 10 on the same Van der Pol parameters, with matching segment counts and parsed segment boxes.

## B. PyTorch TM Parity Attempts

The PyTorch TM `range_only` and `dependency_preserving` runs did not reproduce horizon 10 on the original Flow* segment grid. In the parity run, `range_only` last validated t ~= 0.63429 and `dependency_preserving` last validated t ~= 0.494293 before non-finite residual interval failure.

## C. Stage-1 Diagnostics

Stage-1 varied Taylor order and substep factor for the PyTorch TM prototype. Higher order and smaller substeps only delayed failure marginally. The best fixed diagnostic run was `range_only_o6_s4`, which last validated t ~= 0.7661635 and then failed before horizon 10.

## D. Stage-2 Diagnostics

Stage-2 localized the dominant Van der Pol RHS blowup to polynomial_range * remainder and remainder * remainder interactions in the `x*x*y` term. Dependency reset windows, adaptive bisection, and validation parameter tuning did not beat the Stage-1 best fixed run.

## E. Stage-3 Diagnostics

Stage-3 tested an experimental symbolic remainder prototype as a diagnostic. The symbolic remainder prototype did not beat baseline: the best symbolic run reached t ~= 0.1441472, while the local `range_only_o6_s4_baseline` row reached t ~= 0.7661635. The prototype reduced local ordinary interval-remainder interaction, but most symbolic runs hit wall-time caps and did not improve the benchmark objective.

## F. Final Decision

Stop trying to match original Flow* horizon 10 using the current PyTorch TM prototype.

The current prototype is useful for short-horizon plant Taylor-model experiments and diagnostics. It is not suitable, in its current form, for long-horizon Flow* benchmark parity on this Van der Pol case.

## G. Future Work

Only revisit this direction if the project scope expands to include one or more substantial algorithmic changes:

- real Flow*-style symbolic remainder queue
- tighter polynomial range bounding
- domain splitting
- adaptive validated integrator
- preconditioning

These are not implemented now.
