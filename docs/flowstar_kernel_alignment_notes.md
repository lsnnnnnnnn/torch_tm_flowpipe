# Flow* Kernel Alignment Notes

Local Flow* source root: `/srv/local/shengenli/flowstar/flowstar-toolbox`.

These notes use the local Flow* checkout as a behavioral reference only. They do not copy Flow* source code into the PyTorch implementation. Clean-room targets refer to Python/PyTorch mechanisms in this repository.

## 1. Computational Settings

Flow* behavior:
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:59-78` constructs default reachability settings with adaptive step `0.002..0.1`, fixed order `4`, cutoff threshold `[-1e-10, 1e-10]`, and per-variable remainder estimation `[-1e-4, 1e-4]`.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:98-121` sets fixed stepsize and order tables.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:154-184` sets adaptive min/max steps, starts the active step at the max step, and prepares reachability tables for the selected order.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:186-201` stores cutoff threshold and target remainder estimation.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/settings.h:126` and `/srv/local/shengenli/flowstar/flowstar-toolbox/settings.h:153` carry the remainder-estimation field and setter through the Taylor-model settings object.

Current PyTorch behavior:
- `src/torch_tm_flowpipe/flowpipe.py:2180-2284` runs flowstar-style adaptive stepping with `h_min=0.002`, `h_max=0.1`, and `target_remainder_radius=1e-4` by default.
- `experiments/flowstar_style_rescue_vanderpol.py:584-672` defines the requested rescue configs, including cutoff `1e-10`, candidate order 8/output order 6, and the opt-in `target_remainder_flowstar_ctrunc` mode.

Status:
- Equivalent for the main scalar settings used in these diagnostics.
- More conservative where PyTorch halves failed steps instead of matching Flow*'s exact `LAMBDA_DOWN` policy and step table updates.

Clean-room target:
- Keep target radius `1e-4` for main rescue runs.
- Continue reporting accepted non-final steps below `0.002` rather than silently allowing a different adaptive policy.
- Do not loosen target remainder as the main rescue path.

## 2. Flowpipe Advance

Flow* behavior:
- Endpoint evaluation and center extraction happen in `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:857-879` for the fixed-step real-expression advance path.
- The endpoint constant is removed, then the endpoint is inserted through the previous normalized flowpipe with `insert_ctrunc_normal` at `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:881-890`.
- Endpoint range computation, scaling-factor construction, normalized-domain reset, and `new_x0 = center + S*r0` are in `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:918-951`.
- The same source pattern appears in adaptive stepsize advance at `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:1233-1320`, with failed target containment shrinking the active step in `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:1321-1363`.
- Symbolic-remainder adaptive advance repeats the same preconditioning shape with queue state at `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:3041-3367`.

Current PyTorch behavior:
- `src/torch_tm_flowpipe/flowpipe.py:345-371` builds a fresh normalized Taylor model from an interval reset box.
- `src/torch_tm_flowpipe/flowpipe.py:1843-2142` builds a one-step segment, validates it, truncates output order, records reset boxes, and returns a fresh normalized reset model.
- `src/torch_tm_flowpipe/flowpipe.py:2180-2284` wraps that step in adaptive retry logic.

Status:
- Equivalent for box recentering/rescaling shape.
- Missing Flow*'s `insert_ctrunc_normal` composition through the previous normalized flowpipe. The PyTorch path resets from endpoint boxes, which is simpler and can lose dependency/preconditioning detail.
- More conservative because endpoint dependency is reduced to boxes before the next local step.

Clean-room target:
- Keep saving reset boxes for oracle comparisons.
- If Flow* validates a PyTorch-failed same-box step, prioritize clean-room insertion/composition and normal-domain preconditioning before heuristic sweeps.
- Do not implement Flow* source directly; reproduce behavior through independent Taylor-model composition primitives.

## 3. Picard Candidate

Flow* behavior:
- Horner-form Picard-no-remainder construction is in `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:3556-3577`.
- Expression-form Picard-no-remainder construction and assignment are in `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:3677-3701`.
- Flow* builds polynomial iterates order-by-order in `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:953-959` and `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:1288-1294`.

Current PyTorch behavior:
- `src/torch_tm_flowpipe/flowpipe.py:584-619` constructs `_picard_polynomial` by feeding only polynomial parts through Picard iterations, truncating to candidate order, and leaving remainder feedback to validation.
- Candidate order can exceed output order in `src/torch_tm_flowpipe/flowpipe.py:1881-1951`, allowing candidate order 8 with output order 6.

Status:
- Equivalent in intent for polynomial-only Picard construction.
- PyTorch has no Flow* Horner-form evaluator, but the Van der Pol ODE is represented directly as Taylor-model arithmetic.

Clean-room target:
- Keep `_picard_polynomial` as the Picard-no-remainder analogue.
- Preserve candidate-order diagnostics and separate output-order truncation so validation-path effects remain observable.

## 4. Target Remainder Validation

Flow* behavior:
- Candidate remainders are set to the configured target remainder at `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:963` and `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:1345`.
- `Picard_ctrunc_normal` computes the Picard image at `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:969` and `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:1301`.
- Polynomial difference between tmp and candidate is evaluated in the normal domain at `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:976-989` and `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:1304-1319`.
- The tmp remainder plus polynomial-difference uncertainty must be a subset of the candidate target remainder; otherwise fixed advance fails or adaptive advance shrinks the step.
- Remainder-only refinement uses `Picard_ctrunc_normal_remainder` at `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:1014-1034` and `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:1365-1392`.

Current PyTorch behavior:
- Default target validation is `src/torch_tm_flowpipe/flowpipe.py:820-984`, which evaluates an ordinary Picard residual and checks it against the target remainder.
- The opt-in Flow*-style ctrunc validation is `src/torch_tm_flowpipe/flowpipe.py:1059-1285`: it builds a tmp image, evaluates `tmp.polynomial - candidate.polynomial` over the normal domain, adds that range to tmp remainders, and checks subset containment.
- Diagnostics include tmp remainder, polynomial difference range, ordinary residual range, normal-eval range, subset flags, and decision differences.

Status:
- The opt-in mode is a clean-room analogue of the Flow* target-containment decision.
- It remains incomplete because Taylor-model expression evaluation still uses PyTorch's arithmetic and reset-box composition rather than Flow*'s full `insert_ctrunc_normal` path.

Clean-room target:
- Keep `validation_mode="target_remainder_flowstar_ctrunc"` opt-in.
- Compare ordinary residual vs normal/diff decisions in reports.
- If this mode does not beat t around `2.400737667399793`, avoid more residual-centering or keepK sweeps until insertion/composition and symbolic remainder are addressed.

## 5. Multiplication And Truncation

Flow* behavior:
- Ordinary `mul_ctrunc` accounts for polynomial-product, polynomial-times-remainder, remainder-times-polynomial, remainder-times-remainder, truncation, and cutoff uncertainty in `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:620-648`.
- `mul_ctrunc_normal` uses normal-domain polynomial evaluation and then normal truncation/cutoff in `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:683-711`.
- In-place normal multiplication is in `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:716-741`.
- Insert-and-truncate normal multiplication variants are in `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:797-866` and drive composition through normalized variables.

Current PyTorch behavior:
- `src/torch_tm_flowpipe/taylor_model.py` implements polynomial multiplication, truncation, cutoff, and interval remainder accounting.
- `src/torch_tm_flowpipe/flowpipe.py:1059-1086` adds normal-domain truncation/cutoff accounting for the ctrunc validation tmp image.
- Selective high-degree retention is in `src/torch_tm_flowpipe/flowpipe.py:228-299` and is audited through validation-path hashes.

Status:
- Conservative for ordinary arithmetic.
- Partly equivalent for ctrunc validation diagnostics.
- Missing Flow*'s normal-domain multiplication and insert-normal variants throughout expression evaluation.

Clean-room target:
- Keep default arithmetic unchanged.
- Add normal-domain arithmetic only where diagnostics prove it changes the validation decision.
- Treat full normal insert/multiplication as a future kernel-alignment target, not a parameter sweep.

## 6. Polynomial Range Evaluation

Flow* behavior:
- `Polynomial::intEvalNormal` is in `/srv/local/shengenli/flowstar/flowstar-toolbox/Polynomial.h:589-608`; each term is evaluated with the step exponent table and accumulated as interval uncertainty.
- Ordinary truncation is in `/srv/local/shengenli/flowstar/flowstar-toolbox/Polynomial.h:1505-1528`; normal truncation and normal range evaluation are in `/srv/local/shengenli/flowstar/flowstar-toolbox/Polynomial.h:1530-1550`.
- Normal cutoff removes coefficients contained in the cutoff threshold and evaluates the removed polynomial over the normal domain in `/srv/local/shengenli/flowstar/flowstar-toolbox/Polynomial.h:1791-1809`.
- Ordinary cutoff uses the ordinary domain at `/srv/local/shengenli/flowstar/flowstar-toolbox/Polynomial.h:1814-1835`.
- Taylor-model vector range helpers call ordinary or normal evaluation around `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:3905` and `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:3923`.

Current PyTorch behavior:
- `src/torch_tm_flowpipe/polynomial.py` evaluates interval ranges over explicit domains.
- `src/torch_tm_flowpipe/flowpipe.py:132-162` adds `_normal_domain`, `_poly_interval_normal`, and `_cutoff_polynomial_normal` for Flow*-style ctrunc diagnostics.
- In normalized reset boxes, non-time variables already live on unit intervals, but the explicit normal evaluator makes the intended comparison visible.

Status:
- Equivalent when the PyTorch domain is already normalized.
- Not fully equivalent when Flow*'s step exponent table or composed normal variables carry tighter structure than PyTorch's reset-box domain.

Clean-room target:
- Keep explicit normal-eval diagnostics in the ctrunc mode.
- Add tests that the oracle and reports expose normal-vs-ordinary decision differences.
- Do not claim Flow* parity from normal evaluation alone.

## 7. Symbolic Remainder

Flow* behavior:
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.h:19-35` defines `Symbolic_Remainder` fields `J`, `Phi_L`, `scalars`, and `max_size`.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:12-58` initializes unit scalars, copies queue fields, and resets by clearing `J` and `Phi_L` while rebuilding unit scalars.
- Fixed symbolic reach resets the queue when `J.size() >= max_size` in `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.h:832-914`.
- Adaptive symbolic reach starts from `step_max`, calls symbolic-remainder adaptive advance, pushes accepted flowpipes, updates the current flowpipe, and resets the queue in `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.h:914-972`.
- Invariant/symbolic paths have additional resets around `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.h:2036`, `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.h:2141-2143`, and related invariant loops.

Current PyTorch behavior:
- `src/torch_tm_flowpipe/symbolic_remainder.py` has an experimental symbolic remainder representation.
- The flowstar-style rescue path in `src/torch_tm_flowpipe/flowpipe.py:1843-2284` does not implement a Flow*-style symbolic remainder queue, reset threshold, or `Phi_L`/`J` update loop.

Status:
- Missing mechanism for the rescue branch.
- This is a likely next source-guided kernel target if real Flow* one-step behavior or ctrunc validation shows PyTorch is still missing a containment mechanism.

Clean-room target:
- Do not implement the full queue in this pass.
- Document queue behavior and keep output reports pointing to symbolic remainder as the next target when ctrunc and selective validation do not pass horizon 5.
- Build any future queue from independent Python/PyTorch data structures, not copied Flow* code.

## Current Branch Interpretation

Flow* original/generated parity previously succeeded, but this branch is not a parity claim. The current clean-room PyTorch rescue has the Flow*-style settings, reset boxes, candidate-order diagnostics, and an opt-in ctrunc-normal validation analogue. The main missing kernel detail is Flow*'s insertion/composition through normalized variables, followed by the symbolic remainder queue.

The one-step oracle must be treated as authoritative only if Flow* compiles and runs. If Flow* is skipped, the oracle conclusion is inconclusive. If Flow* runs but does not complete the same local box/h, the local reset box is likely too wide or the step is too hard. If Flow* validates while PyTorch rejects, continue kernel alignment before any more heuristic sweeps.
