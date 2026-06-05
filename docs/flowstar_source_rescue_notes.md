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

## Kernel Equivalence Pass 2

The local Flow* source reference is `/srv/local/shengenli/flowstar/flowstar-toolbox`. Line numbers below are from that checkout and are used only as behavioral anchors for clean-room PyTorch logic.

| Flow* kernel | Flow* behavior | Current PyTorch behavior | Equivalence status | Clean-room target |
| --- | --- | --- | --- | --- |
| `Flowpipe::advance` (`Continuous.cpp:857`, `Continuous.cpp:953`, `Continuous.cpp:969`) | Evaluates the endpoint, removes constants, inserts the endpoint through the prior normalized flowpipe, rescales to a unit normal domain, builds Picard polynomials with no remainders, sets target remainders, then validates a `Picard_ctrunc_normal` image plus polynomial-difference uncertainty. | `flowpipe_step_flowstar_style_adaptive` recenters and normalizes endpoint boxes, builds Picard polynomials, and validates ordinary Picard residual containment against the target remainder. | Partly equivalent. Recenter/reset and target remainder exist; the ctrunc-normal image-minus-candidate decision was missing before this pass. | Keep the default path, add an opt-in `target_remainder_flowstar_ctrunc` validator that decides acceptance from the tmp Picard image remainder plus normal-evaluated polynomial difference. |
| `Flowpipe::advance_adaptive_stepsize` (`Continuous.cpp:1233`, `Continuous.cpp:1301`, `Continuous.cpp:1323`) | Reuses the fixed advance kernel, but updates step tables, shrinks the step on target-containment failure, and retries until the minimum step. | PyTorch halves `h_try` after target containment failure and tracks non-final steps below the Flow* minimum. | Mostly equivalent in policy, not in kernel details. | Preserve the halving policy and route the new ctrunc-normal validation mode through the same adaptive wrapper. |
| `TaylorModelVec::Picard_no_remainder_assign` (`TaylorModel.h:3698`) | Evaluates ODE expressions with no interval remainder feedback, integrates in time, and assigns the polynomial Picard iterate. | `_picard_polynomial` feeds only polynomial parts through iterations, truncates to candidate order, and records cutoff uncertainty outside the polynomial. | Equivalent in intent for polynomial-only Picard construction. | Continue using `_picard_polynomial` as the clean-room Picard-no-remainder analogue. |
| `TaylorModelVec::Picard_ctrunc_normal` (`TaylorModel.h:3707`) | Evaluates ODE expressions on the candidate with normal-domain truncation/cutoff, integrates by the current time-step table, and returns a tmp Taylor-model vector. | Ordinary target validation recomputed the Picard residual using standard Taylor-model arithmetic and standard interval evaluation. | Missing mechanism before this pass. | `_picard_ctrunc_normal_image` computes a clean-room tmp image, truncates and cuts off with normal-domain interval ranges, and feeds that tmp remainder into the acceptance decision. |
| `TaylorModel::mul_ctrunc` and `mul_ctrunc_normal` (`TaylorModel.h:620`, `TaylorModel.h:683`) | Multiplies polynomial parts, adds polynomial-times-remainder and remainder-times-remainder uncertainty, then moves truncated and cutoff terms into the remainder; the normal variant evaluates polynomial ranges on the normal step domain. | `TaylorModel.__mul__` does polynomial multiplication, truncation, and interval remainder accounting, but uses ordinary interval range evaluation. | Conservative but not equivalent to the normal variant. | Keep arithmetic clean-room and add normal-domain range helpers for validation diagnostics and tmp Picard truncation; a deeper future target is normal-domain multiplication throughout expression evaluation. |
| `Polynomial::intEvalNormal` (`Polynomial.h:589`) | Sums term interval contributions using the precomputed step-exp table and normal-domain state variables. | `Polynomial.evaluate_interval` evaluates over the model domain directly; in flowstar-style resets the state domain is already normalized, but no explicit normal evaluator existed. | Equivalent only when domains are already normal; not explicit. | Add `_poly_interval_normal`, preserving the local time interval and evaluating non-time variables on unit normal intervals. |
| `Polynomial::ctrunc` / `ctrunc_normal` (`Polynomial.h:1505`, `Polynomial.h:1530`) | Removes terms above order and returns their interval contribution, with the normal variant using `intEvalNormal`. | `Polynomial.truncate` returns kept/dropped polynomials; callers evaluate dropped terms with ordinary interval or split-domain bounds. | Partly equivalent; normal evaluation was missing. | For ctrunc validation, truncate explicitly and add `_poly_interval_normal(dropped)` to the tmp remainder. |
| `Polynomial::cutoff` / `cutoff_normal` (`Polynomial.h:1791`, `Polynomial.h:1814`) | Removes coefficients inside the cutoff threshold and adds the removed polynomial range to the interval remainder. | `TaylorModel.apply_cutoff` removes small coefficients and uses ordinary interval evaluation for removed terms. | Conservative but not normal-equivalent. | Add `_cutoff_polynomial_normal` for the ctrunc validation path while leaving existing default cutoff behavior unchanged. |
| `TaylorModelVec::insert_ctrunc_normal` (`TaylorModel.h:3378`) | Composes a Taylor-model vector through normalized variables, truncating/cutoff-normalizing the inserted result. | PyTorch reset boxes are rebuilt as fresh normalized identity Taylor models rather than composing the prior flowpipe through insertion. | More conservative and structurally different. | Keep reset-box normalization for now; if one-step Flow* validates where PyTorch fails, inspect clean-room insertion/composition before parameter sweeps. |
| `rmConstant`, `scale_assign`, normal domain construction (`Continuous.cpp:875`, `Continuous.cpp:918`, `TaylorModel.h:3453`, `TaylorModel.h:3462`) | Removes endpoint constants, computes magnitude scalars, scales the current flowpipe by inverse scalars, and builds `center + S*r0` over `[-1,1]` variables. | `_normalized_tm_from_box` builds `center + radius*r` directly from interval endpoint boxes. | Equivalent for box reset shape, but PyTorch does not compose the old polynomial dependency before reset. | Continue saving reset boxes and use the one-step oracle to decide whether missing composition/preconditioning detail matters. |
| `Symbolic_Remainder` advance/update path (`Continuous.h:19`, `Continuous.cpp:12`, `Continuous.cpp:2123`, `Continuous.cpp:2715`, `Continuous.h:832`, `Continuous.h:914`) | Maintains a queue of symbolic remainder objects, advances with symbolic-remainder overloads, resets the queue when the configured size is reached, and uses adaptive reach loops that update this state. | PyTorch has an experimental symbolic remainder module, but the flowstar-style rescue path does not implement the Flow* queue/update mechanism. | Missing mechanism. | Do not implement in this pass; if ctrunc validation and the one-step oracle do not clear the bottleneck, the next clean-room target is a real Flow*-style symbolic remainder queue. |

### Pass 2 Implementation Target

The immediate implementation target is `validation_mode="target_remainder_flowstar_ctrunc"`. It builds the polynomial candidate with the existing Picard-no-remainder analogue, sets the candidate remainder to the target `[-1e-4, 1e-4]`, computes one ctrunc-normal Picard image, evaluates `tmp.polynomial - candidate.polynomial` on the normal domain, adds that range to the tmp remainder, and accepts only if the tmp remainder is contained in the target for every state dimension. Diagnostics compare this decision against the existing ordinary residual containment decision.

