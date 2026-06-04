# Flow* Source Rescue Notes

These notes are based on the local Flow* checkout at `/srv/local/shengenli/flowstar/flowstar-toolbox`. They summarize behavior only; the PyTorch implementation remains clean-room Python/PyTorch code.

## Defaults

- `Continuous.cpp:59-80` initializes `Computational_Setting` with adaptive stepsize `0.002..0.1`, fixed order `4`, cutoff threshold `[-1e-10, 1e-10]`, and remainder estimation `[-1e-4, 1e-4]` for every variable.
- `Continuous.cpp:154-184` stores adaptive min/max steps and starts from the max step for the selected order.
- `Continuous.cpp:186-201` exposes setters for cutoff threshold and remainder estimation.

## Symbolic Remainder

- `Continuous.h:19-35` declares `Symbolic_Remainder`.
- `Continuous.cpp:12-58` shows the stored fields and copy/reset behavior: `J`, `Phi_L`, `scalars`, and `max_size`. Construction from an initial flowpipe initializes `scalars` to one per state dimension and stores the queue limit; reset clears `J` and `Phi_L` and recreates unit scalars.

## Flowpipe Advance Mechanics

- `Continuous.cpp:857-879` evaluates the endpoint of the current flowpipe at the step end and extracts the center constants.
- `Continuous.cpp:881-890` removes the center and composes the endpoint through the previous normalized flowpipe using insert/truncate/normal operations and the cutoff threshold.
- `Continuous.cpp:918-951` computes the endpoint range, builds scaling factors from interval magnitudes, normalizes ranges to `[-1,1]`, scales the flowpipe, and constructs `new_x0 = center + S*r0`.
- `Continuous.cpp:953-964` runs `Picard_no_remainder` up to the order and then sets each candidate remainder to the configured target remainder estimation.
- `Continuous.cpp:969-989` runs `Picard_ctrunc_normal`, computes polynomial differences caused by roundoff/truncation, adds those uncertainties, and accepts only when the computed remainder is a subset of the target remainder.
- `Continuous.cpp:1014-1034` refines remainders with the remainder-only Picard path while preserving the subset requirement.
- `TaylorModel.h:620-648`, `683-711`, and `829-866` show Taylor-model multiplication variants that account for polynomial-range times remainder, remainder times remainder, truncation, and cutoff uncertainty.
- `Polynomial.h:1505-1550` evaluates truncated-away polynomial terms over the current domain; `Polynomial.h:1791-1832` removes coefficients inside the cutoff threshold and returns their interval range as extra uncertainty.

## Reach And Adaptive Loops

- `Continuous.h:590-644` shows the fixed-step reach loop: call `advance`, push the accepted flowpipe, update `currentFlowpipe`, and stop on failure.
- `Continuous.h:832-914` shows the symbolic-remainder reach loop with the same accept/push/update pattern and a reset when `J.size() >= max_size`.
- `Continuous.h:914-972` starts symbolic-remainder adaptive stepsize from `step_max`, calls `advance_adaptive_stepsize`, pushes accepted flowpipes, updates the current flowpipe, resets the symbolic remainder queue when full, and advances time by the accepted step.
- `Continuous.cpp:1233-1447` implements adaptive advance. On failed target-containment it shrinks the step (`LAMBDA_DOWN`) and retries; on success the caller can continue with the accepted step.

## Differences From The Current PyTorch Prototype

- `range_only` collapses each endpoint to an interval box and restarts from identity variables over that box. It resets dependency but uses unnormalized variables.
- `dependency_preserving` carries the raw endpoint Taylor model forward. That keeps dependency but also carries old variables and accumulated interval remainders into nonlinear products for too long.
- The current validation loop inflates candidate remainders until containment. That can validate very wide boxes and permits the huge remainder blowup seen in `x*x*y`.
- The rescue mode should instead recenter and normalize after each step, set a target remainder, reject on residual-not-subset rather than inflating without bound, shrink the step on failure, and optionally move small coefficients into conservative interval uncertainty.

## Rescue Mode Claim Boundary

The `flowstar_style` path in this repository is experimental clean-room
Flow*-inspired behavior. It uses the local Flow* checkout only as a behavioral
reference while keeping the implementation in Python/PyTorch. The rescue path
recenters and rescales endpoint boxes, validates against a fixed target
remainder, shrinks adaptive steps on failed containment, and records when any
non-final accepted step drops below the original Flow* minimum step of `0.002`.

These artifacts are not a Flow* parity claim. Horizon 10 and overlap-based box
comparisons against the original Flow* segment boxes are required before making
any stronger statement about reachability or tightness.

