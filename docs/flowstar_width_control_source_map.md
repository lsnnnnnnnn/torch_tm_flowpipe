# Flow* Width-Control Source Map

Scope: local source only, rooted at `/srv/local/shengenli/flowstar/flowstar-toolbox`, plus the local Van der Pol benchmark at `/srv/local/shengenli/flowstar/benchmarks/continuous/vanderpol/vanderpol.cpp`. This is a clean-room behavioral map; it intentionally records paths and summaries rather than copying Flow* implementation blocks.

## Executive Summary

Flow* does not advance the Van der Pol benchmark by repeatedly materializing endpoint boxes. A `Flowpipe` is a composition of a left preconditioning Taylor model (`tmvPre`) and a right normalized Taylor model (`tmv`), with a normal-domain insertion step at each transition. The original benchmark also passes `Symbolic_Remainder sr(initialSet, 100)` into `ode.reach`, so the continuous Van der Pol path uses the symbolic remainder queue overloads rather than only interval remainders.

Current PyTorch `flowstar_style` accepts a step, computes an endpoint Taylor model, turns the endpoint range into an interval box, and restarts from fresh normalized variables. That is much easier to implement, but it discards Flow*'s composed normalized map and the linearized remainder queue.

## 1. Normalized Flowpipe Transition And Insertion

Relevant source ranges:

- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.h:75-100`: `Flowpipe` stores `tmvPre`, `tmv`, and `domain`; comments identify `tmvPre` as the preconditioning side of the composition.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:407-445`: `Flowpipe::compose_normal` computes `tmv.polyRangeNormal(...)` and calls `tmvPre.insert_ctrunc_normal(...)` for all outputs or selected axes.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:450-468`: `Flowpipe::intEvalNormal` composes in normal form and interval-evaluates; `Flowpipe::normalize` normalizes the right Taylor model and moves constants into `tmvPre`.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:857-948`, `1233-1324`, `2123-2319`, `2715-2922`: accepted-step transition code evaluates `tmvPre` at step end, removes constants, inserts through current `tmv` with `insert_ctrunc_normal`, evaluates range, builds scaling factors, scales `result.tmv`, and constructs `new_x0`.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:3378-3427`: `TaylorModelVec::insert_ctrunc_normal` overloads dispatch each component through scalar insertion with one order or per-output orders.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:4213-4338`: Horner insertion recursively substitutes normalized variables, normal-evaluates truncation/cutoff uncertainty, pushes intermediate ranges for later remainder evaluation, and accumulates uncertainty into Taylor model remainders.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:3748-3795`: normalization recenters non-time domains, constructs scaled fresh variables, sets state domains to `[-1,1]`, and reinserts old Taylor models through the normalized variables.

Flow* mechanism: transition is not a plain endpoint box reset. It preserves a composed normalized map, moves the endpoint center into `new_x0`, scales the local normalized variables by coordinate magnitudes, and stores a right-side map for subsequent insertion.

Current PyTorch mechanism: `_normalized_tm_from_box` in `src/torch_tm_flowpipe/flowpipe.py` builds `center + radius * r` from `seg.final_tm.range_box()`. The accepted endpoint dependency is materialized to a box before the next step.

Gap: PyTorch loses dependencies in the accepted endpoint polynomial and all insertion/truncation uncertainty is collapsed into independent interval widths.

Implementation target: add opt-in reset modes that keep the accepted endpoint Taylor model in normal form and then layer Flow*-style queue propagation. The first implemented target in this branch is the symbolic queue skeleton; normalized insertion remains the next gap if the queue does not improve widths.

## 2. `Flowpipe::advance` Fixed And Adaptive Variants

Relevant source ranges:

- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.h:118-148`: declares fixed/adaptive interval-remainder overloads and fixed/adaptive symbolic-remainder overloads.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:857-1039`: fixed-step real ODE interval-remainder advance.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:1045-1225`: fixed-step interval ODE interval-remainder advance.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:1233-1445`: adaptive-stepsize real ODE interval-remainder advance.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:1452-1665`: adaptive-stepsize interval ODE interval-remainder advance.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:2123-2415`: fixed-step real ODE symbolic-remainder advance.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:2715-3040`: adaptive-stepsize real ODE symbolic-remainder advance.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:3577-3728`: Picard no-remainder seed, Picard normal ctrunc, and Picard normal remainder routines used by advance.

Preconditioning sequence:

1. Evaluate `tmvPre` at the current step end.
2. Extract polynomial constants as endpoint center `c0`.
3. Remove constants so the local set is origin-centered.
4. Insert through the previous right normalized flowpipe via `insert_ctrunc_normal`.
5. Evaluate range, optionally contract remainders with invariants.
6. Compute per-dimension magnitudes `S` and inverse scalars.
7. Scale `result.tmv` by inverse scalars and construct `new_x0 = c0 + S*r`.
8. Run Picard no-remainder expansion, assign target remainders, validate via normal ctrunc, shrink step if adaptive validation fails, then refine remainder.
9. Store `result.tmvPre = x`, copy the old domain, and set the time domain to the accepted step.

Failure and step shrink: adaptive variants shrink `tm_setting.step_exp_table[1]` by `LAMBDA_DOWN` while Picard remainder containment fails, stopping when the candidate step is below `step_min`.

Flow* mechanism: accepted flowpipe result consists of the validated Picard Taylor model in `tmvPre` plus a normalized right map `tmv`, not only a final interval.

Current PyTorch mechanism: `flowpipe_step_flowstar_style_adaptive` shrinks `h_try` by halves and validates residual boxes against target remainders. On success it resets from endpoint range unless the new opt-in queue reset is selected.

Gap: PyTorch has no stored `tmvPre/tmv` split and no exact `insert_ctrunc_normal` transition.

Implementation target: keep the new queue path isolated and use width diagnostics to decide whether the next useful step is normalized insertion rather than more target-remainder tuning.

## 3. Symbolic Remainder Queue

Relevant source ranges:

- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.h:19-35`: `Symbolic_Remainder` fields are `J`, `Phi_L`, `scalars`, and `max_size` with constructor/reset declarations.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:12-53`: constructors initialize scalars to one per state dimension, copy queue fields, and reset by clearing `J`/`Phi_L` and restoring scalars to one.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:2123-2323`: fixed-step symbolic advance decomposes the endpoint map into linear and other parts, updates `Phi_L`, accumulates older `J` columns, inserts nonlinear/other terms, forms `J_ip1`, scales by new magnitudes, pushes the new `J`, and updates `scalars`.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:2715-2922`: adaptive symbolic advance performs the same queue update before adaptive Picard validation and step shrink logic.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.h:831-968`: symbolic reach loops call the symbolic advance overloads and reset the queue when its size reaches `max_size`.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.h:1148-1176`: public `ODE::reach(..., Symbolic_Remainder&)` dispatches to symbolic fixed/adaptive order/stepsize routines.

Exact fields: `J` is a vector of interval matrices, `Phi_L` is a vector of real matrices, `scalars` stores per-coordinate inverse magnitudes from normalization, and `max_size` bounds the queue.

Queue update behavior: each accepted step extracts the linear part of the endpoint reset, scales it by the previous scalar vector, left-multiplies older `Phi_L` entries, pushes the new linear map, computes accumulated older remainders `J_i`, computes current-step remainder `J_ip1`, and stores current plus propagated remainders in the local initial set.

Conversion back to intervals: older symbolic remainders are converted by multiplying queued `J` columns through queued `Phi_L` matrices and summing into interval remainders. Queue reset clears symbolic history when the configured size is reached.

Scope: this handles linearized propagation of remainder sources through accepted reset maps. Nonlinear terms stay in the ordinary Taylor model insertion path; the queue is not a general nonlinear symbolic-noise system.

Difference from the previous PyTorch prototype: the older `SymbolicRemainderState` represented residuals as explicit polynomial noise variables with domain `[-1,1]`. Flow* instead stores matrix/list state and propagates interval columns through linear maps. This branch adds a separate `FlowstarSymbolicRemainderQueue` skeleton rather than reusing the noise-variable prototype.

Flow* mechanism: symbolic remainder queue is on the original Van der Pol benchmark path.

Current PyTorch mechanism: default flowstar-style reset has no queue; historical symbolic diagnostics are not the same mechanism.

Gap: missing queue state likely contributes to width growth before t around 2.4.

Implementation target: implemented opt-in `reset_mode="flowstar_symbolic_remainder_queue"`, with `J`, `Phi_L`, scalars, queue reset, and per-step diagnostics.

## 4. Preconditioning And Linear Transformation

Relevant source ranges:

- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.h:75`: Flowpipes are represented as a composition, with the left Taylor model the preconditioning part.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:8428`: later linear/nonlinear flow code mentions preconditioned matrices, but this is not reached by the simple continuous Van der Pol benchmark path inspected here.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.h:2919-2924` and `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:10206-10891`: aggregation utilities can construct interval or parallelotope aggregation, separate from the benchmark reach call.
- `/srv/local/shengenli/flowstar/benchmarks/continuous/vanderpol/vanderpol.cpp:31-37`, `52-72`, `89`: benchmark uses default computational setting, constructs a `Flowpipe` from a box, creates a symbolic remainder, and calls `ode.reach` with that symbolic remainder.

Search result: `rg QR` under the toolbox finds parser tokens for `SQRT`, not a QR preconditioning routine on this path. `rg precond|precondition|linear_trans|QR` finds only the general Flowpipe preconditioning comment and unrelated later code, not a Van der Pol-specific QR call.

Flow* mechanism: Van der Pol uses the normal Flowpipe composition/preconditioning and symbolic remainder queue; no benchmark-local QR or template transform is selected.

Current PyTorch mechanism: scaling-only normalized endpoint boxes.

Gap: PyTorch lacks the composition split and queue, but QR is not evidenced as the missing Van der Pol mechanism.

Implementation target: do not implement QR for this task; focus on symbolic queue and then normalized insertion.

## 5. Polynomial And Range Evaluation

Relevant source ranges:

- `/srv/local/shengenli/flowstar/flowstar-toolbox/Term.h:38-40`, `167-184`: `Term::intEvalNormal` assumes a normal domain with time `[0,s]` and non-time variables `[-1,1]`.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:3013-3032`: vector normal interval evaluation dispatches each component to scalar `intEvalNormal`.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:3058-3091`: vector `ctrunc_normal` drops terms above order using the step exponent table.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:4213-4338`: insertion normal ctrunc adds truncation intervals to remainders and records intermediate ranges.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:969-1014`, `1351-1421`, `2343-2388`, `2940-3010`: Picard normal ctrunc evaluates polynomial differences, adds cutoff/truncation uncertainty, and checks target containment.

Ordinary interval eval vs normal eval: ordinary eval uses the full domain vector; normal eval uses the step exponent table for time powers and normalized state domains for non-time variables.

Step variable powers: `step_exp_table` carries time-step powers; terms involving time are bounded through this table rather than an arbitrary interval box.

Cutoff and dropped terms: insertion/ctrunc paths evaluate dropped high-order terms and small cutoff terms as intervals and add them to the Taylor model remainder, rather than silently deleting them.

Current PyTorch mechanism: polynomial evaluation uses interval domains and custom `target_remainder_flowstar_ctrunc` diagnostics, but it does not fully reproduce Flow*'s Horner insertion intermediate-range queue.

Gap: PyTorch's current validation approximates normal evaluation but is missing the exact insertion/remainder accounting path.

Implementation target: keep `target_remainder_flowstar_ctrunc` as diagnostics; implement queue now; evaluate normalized insertion next.

## 6. Original Van Der Pol Benchmark Path

Relevant source ranges:

- `/srv/local/shengenli/flowstar/benchmarks/continuous/vanderpol/vanderpol.cpp:16-25`: declares variables `x`, `y`, `t` and ODE `x'=y`, `y'=(1-x^2)y-x`, `t'=1`.
- `/srv/local/shengenli/flowstar/benchmarks/continuous/vanderpol/vanderpol.cpp:31-37`: default computational setting is used; comments state adaptive stepsize `0.002` to `0.1` and fixed TM order `4`.
- `/srv/local/shengenli/flowstar/benchmarks/continuous/vanderpol/vanderpol.cpp:42-53`: initial box is `x in [1.1,1.4]`, `y in [2.35,2.45]`; `Flowpipe initialSet(box)` constructs the preconditioned Taylor-model flowpipe.
- `/srv/local/shengenli/flowstar/benchmarks/continuous/vanderpol/vanderpol.cpp:58-72`: safety set is `y <= 2.75`; symbolic remainder queue is created with size `100`.
- `/srv/local/shengenli/flowstar/benchmarks/continuous/vanderpol/vanderpol.cpp:74-90`: horizon is `10` and the call is `ode.reach(result, initialSet, T, setting, safeSet, sr)`.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.h:1148-1176`: this overload dispatches to symbolic remainder reach routines according to adaptive/fixed settings.

Conclusion: original continuous Van der Pol uses adaptive stepsize, default order 4, default cutoff/remainder settings from `Computational_Setting`, and symbolic remainder queue size 100. It calls `reach()` with a symbolic remainder argument, which dispatches to symbolic remainder reach routines; it does not call a non-symbolic `reach()` path for this benchmark.

Flow* mechanism: adaptive normal insertion plus symbolic remainder queue.

Current PyTorch mechanism: candidate/output order 8/6 target-remainder validation with endpoint box reset, optionally cutoff.

Gap: missing symbolic queue and full normalized insertion explain why the PyTorch reset box can become too wide before the local t around 2.400737 failure.

Implementation target: first implement `flowstar_symbolic_remainder_queue`; if it fails to improve, prioritize a true `insert_ctrunc_normal`-like reset over more parameter sweeps.
