# Flow* Right-Map Scaling Source Map

Local Flow* source root: `/srv/local/shengenli/flowstar/flowstar-toolbox`.
Current PyTorch source root: `/srv/local/shengenli/torch_tm_flowpipe`.

This map focuses on the normal-domain right-map range and scaling path. It does not claim Flow* parity; it identifies the behavior to match and the current implementation target.

## 1. Flowpipe::normalize

Flow* line ranges:
- `Continuous.cpp:457-513`: `Flowpipe::normalize`.
- `TaylorModel.h:3748-3788`: `TaylorModelVec::normalize`.

Flow* behavior:
- `Flowpipe::normalize` first calls `tmv.normalize(domain, cutoff_threshold)` (`Continuous.cpp:459-460`).
- `TaylorModelVec::normalize` removes the midpoint from every non-time domain interval, stores those centers, builds a diagonal affine map from interval magnitudes, resets every non-time domain entry to `[-1,1]`, and inserts that map back into every Taylor model (`TaylorModel.h:3748-3788`). Time is intentionally omitted from this domain-centering loop.
- Back in `Flowpipe::normalize`, constants are extracted from `tmv`, removed from `tmv`, and each component remainder midpoint is removed and added to the component constant (`Continuous.cpp:464-475`).
- The remaining `tmv` range is interval-evaluated over the current domain. Each component scale is `mag(range)`, i.e. max absolute bound. Zero magnitude maps to scale 0; otherwise `tmv` is divided by that magnitude (`Continuous.cpp:477-493`).
- `tmvPre` is updated by inserting the affine map `center + scale * new_normal_var` (`Continuous.cpp:496-512`).

Current PyTorch behavior:
- `_flowstar_normalized_insertion_transition` removes endpoint constants, composes the endpoint with the previous right map, optionally recenters the inserted remainder midpoint, evaluates the inserted box, chooses scale as max absolute bound, divides the right map by scale, and creates `center + scale * z` reset TMs (`src/torch_tm_flowpipe/flowpipe.py:720-812`).
- `_normalized_tm_from_center_scale` resets state domains to `[-1,1]` (`src/torch_tm_flowpipe/flowpipe.py:472-506`).

Gap:
- PyTorch has the same high-level scalar extraction, but the range source used for the inserted right map was generic range evaluation. Flow*'s normal path separates normal-domain assumptions and time powers.

Implementation target:
- Keep generic evaluation unchanged. For opt-in `right_map_range_mode="normal_eval"`, compute both old and normal-domain inserted ranges, use the normal-domain range for scale construction, and record both ranges in reset diagnostics.

## 2. Flowpipe::compose_normal

Flow* line ranges:
- `Continuous.cpp:415-427`: full-output `Flowpipe::compose_normal` overloads.
- `Continuous.cpp:428-441`: selected-output overload.

Flow* behavior:
- Full-output `compose_normal` calls `tmv.polyRangeNormal(tmvPolyRange, step_exp_table)` and passes those ranges into `tmvPre.insert_ctrunc_normal(...)` (`Continuous.cpp:415-427`).
- The selected-output overload uses `tmv.polyRange(tmvPolyRange, domain)` rather than `polyRangeNormal` (`Continuous.cpp:428-441`), so it is a distinct path.
- The ranges passed to insertion are the polynomial ranges of the right map, not final endpoint boxes.

Current PyTorch behavior:
- `insert_ctrunc_normal_like` composes endpoint polynomials with the previous right map, truncates, moves truncation/cutoff into remainders, and records component widths (`src/torch_tm_flowpipe/flowpipe.py:576-654`).
- The transition then evaluates the composed/inserted result for scaling (`src/torch_tm_flowpipe/flowpipe.py:782-793`).

Gap:
- PyTorch did not previously make the Flow* normal range mode explicit at the right-map scaling point, and its insertion helper is direct sparse substitution rather than Flow*'s Horner form with intermediate ranges.

Implementation target:
- Use `evaluate_interval_normal` for the opt-in scaling range and expose old-vs-normal range diagnostics. Do not replace the global substitution or global interval evaluation path.

## 3. TaylorModelVec::polyRangeNormal

Flow* line ranges:
- `TaylorModel.h:3812-3823`: vector loop.
- `TaylorModel.h:1248-1252`: scalar Taylor model `polyRangeNormal`.
- `Polynomial.h:589-604`: polynomial normal interval sum.
- `Term.h:167-194`: term normal interval rule.

Flow* behavior:
- `TaylorModelVec::polyRangeNormal` loops over Taylor models and calls scalar `polyRangeNormal` (`TaylorModel.h:3812-3823`).
- Scalar `TaylorModel::polyRangeNormal` evaluates only the expansion; it does not add the Taylor-model remainder (`TaylorModel.h:1248-1252`).
- The polynomial sums term normal ranges (`Polynomial.h:589-604`).
- Each term multiplies by `step_exp_table[degree_of_time]`; for non-time variables, any odd state exponent yields `[-1,1]`, all-even positive state exponents yield `[0,1]`, and no state exponent yields `[1,1]` (`Term.h:167-194`).

Current PyTorch behavior:
- Generic `Polynomial.evaluate_interval` multiplies each variable interval power independently (`src/torch_tm_flowpipe/polynomial.py:315-329`).
- New `evaluate_interval_normal` mirrors Flow*'s time-table plus normal-state term rule (`src/torch_tm_flowpipe/polynomial.py:57-112`).

Gap:
- Before this change, PyTorch normal validation used a helper that still delegated to generic interval evaluation on a constructed normal box. That is conservative but did not encode Flow*'s explicit time-power table and state-term rule.

Implementation target:
- Use `evaluate_interval_normal(poly, domain, step_exp_table, state_var_indices, time_var_index)` where normal-domain behavior is being studied. Keep it clean-room and tested by sampled containment.

## 4. TaylorModelVec::intEvalNormal, Polynomial::intEvalNormal, Term::intEvalNormal

Flow* line ranges:
- `TaylorModel.h:3013-3023`: vector `intEvalNormal`.
- `TaylorModel.h:448-452`: scalar Taylor model `intEvalNormal`.
- `Polynomial.h:589-604`: polynomial normal evaluation.
- `Term.h:167-194`: term normal evaluation.

Flow* behavior:
- Vector `intEvalNormal` loops over scalar models and calls scalar `intEvalNormal` (`TaylorModel.h:3013-3023`).
- Scalar `TaylorModel::intEvalNormal` evaluates the expansion normally and then adds the Taylor-model remainder (`TaylorModel.h:448-452`).
- Time is not treated as `[-1,1]`; it is looked up in `step_exp_table` by time degree (`Term.h:175`).
- Non-time variables are not evaluated through their stored domain intervals in this path; the term rule assumes normalized variables.

Current PyTorch behavior:
- `_taylor_model_range_box_normal` now calls `_poly_interval_normal`, which uses `evaluate_interval_normal` in the unsplit case and then adds the Taylor-model remainder (`src/torch_tm_flowpipe/flowpipe.py:1473-1494`).
- Existing split diagnostics still preserve the previous split-box behavior for `normal_eval_range_split`.

Gap:
- Flow* distinguishes polynomial-only `polyRangeNormal` from Taylor-model `intEvalNormal` with remainder. PyTorch diagnostics must label which one is being measured.

Implementation target:
- In reset diagnostics, `old_right_map_range_*` and `normal_right_map_range_*` are Taylor-model ranges including inserted remainders. Diagnostics that need polynomial-only ranges should explicitly call the polynomial helper.

## 5. insert_ctrunc_normal Horner Path

Flow* line ranges:
- `TaylorModel.h:3378-3430`: vector `insert_ctrunc_normal` dispatch.
- `TaylorModel.h:4213-4244`: Horner `insert_ctrunc_normal` without intermediate range output.
- `TaylorModel.h:4247-4287` and `TaylorModel.h:4294-4343`: Horner path with intermediate range logging.
- `TaylorModel.h:797-856`: `mul_insert_ctrunc_normal` and its diagnostic variant.
- `Polynomial.h:1530-1552`: `ctrunc_normal` moves high-order terms into a normal-domain interval.
- `Polynomial.h:1791-1810`: `cutoff_normal` moves cutoff terms into a normal-domain interval.

Flow* behavior:
- Horner insertion treats the first nonconstant part as time, recursively inserts it, multiplies by `t`, scales its remainder by `step_exp_table[1]`, then normal-truncates (`TaylorModel.h:4213-4232`).
- For state variables, the recursive result is multiplied by the corresponding right-map Taylor model using `varsPolyRange[i-1]`, which came from `polyRangeNormal` (`TaylorModel.h:4236-4239`).
- The diagnostic overload pushes intermediate coefficient ranges, right-map polynomial ranges, and truncation intervals into `intermediate_ranges` (`TaylorModel.h:4247-4287`, `4294-4343`).
- Multiplication uses the provided `tmPolyRange` for `P2 * I1`; if the left operand has a remainder, it evaluates the left expansion normally and multiplies by the right remainder (`TaylorModel.h:797-856`).

Current PyTorch behavior:
- `insert_ctrunc_normal_like` performs direct sparse substitution at a sufficient work order, then truncates/cutoffs and moves uncertainty to remainders (`src/torch_tm_flowpipe/flowpipe.py:576-654`).
- It does not preserve a Horner intermediate range stack, so it cannot yet reproduce Flow*'s intermediate-range accounting.

Gap:
- The missing mechanism may be Horner-stage range reuse, not just final full-box range evaluation.

Implementation target:
- Keep the current direct helper for production. Add diagnostics for top monomial interval contributions and old-vs-normal right-map range. If needed later, implement a separate experimental Horner insertion path that records intermediate ranges before replacing any production path.

## 6. Scaling Factor Construction in Flowpipe::advance

Flow* line ranges:
- `Continuous.cpp:866-965`: one advance path ending in scaling matrix construction.
- `Continuous.cpp:927-946`: scale and inverse-scale extraction from `range_of_x0`.
- `Continuous.cpp:3763-3869`: another normal insertion branch before scale construction.
- `Continuous.cpp:3868-3890`: symbolic-remainder branch scale update.

Flow* behavior:
- Before scaling, Flow* evaluates/recenters the candidate endpoint/right-map range, including remainder midpoint removal in the surrounding advance path.
- Scale `S[i]` is `mag(range_of_x0[i])`. Intervals crossing zero are handled by magnitude as max absolute endpoint; small nonzero intervals are still scaled by their magnitude; exact zero gets scale 0 and inverse scale 1 (`Continuous.cpp:927-946`).
- In symbolic remainder mode, `symbolic_remainder.scalars[i]` is updated with the inverse scale, and zero magnitude stores scalar 0 (`Continuous.cpp:3868-3890`).

Current PyTorch behavior:
- `_interval_magnitude` returns max absolute endpoint and `_flowstar_normalized_insertion_transition` maps zero/None to scale 0 and inverse scale 1 (`src/torch_tm_flowpipe/flowpipe.py:686-793`).
- Split symbolic queue reset receives `scales` from the same transition (`src/torch_tm_flowpipe/flowpipe.py:797-807`).

Gap:
- The scalar construction is aligned at a high level, but the range used as input to magnitude needed explicit normal-eval instrumentation.

Implementation target:
- Opt-in `right_map_range_mode="normal_eval"` uses normal-domain range for magnitude and records both old and normal range. If no horizon/width improvement appears, the next target is Horner intermediate-range parity rather than another scalar tweak.
