# Flow* Normalized Insertion Source Map

Scope: local source only. Flow* references point to `/srv/local/shengenli/flowstar/flowstar-toolbox` and the local Van der Pol benchmark. This is a clean-room behavioral map; it records paths, line ranges, and implementation intent rather than copying Flow* implementation blocks.

## Executive Summary

This branch adds an opt-in PyTorch reset mode, `reset_mode="normalized_insertion"`, that keeps a Flow*-style normal-coordinate state across accepted Van der Pol steps. The default endpoint-box reset is unchanged.

The new path substitutes an accepted endpoint Taylor model through the previous normal map, truncates and cuts off the inserted result, moves those uncertainties into remainders, recenters the next initial set, and scales each coordinate back to a unit normal domain. The implementation is deliberately conservative and direct; it does not copy Flow*'s Horner insertion code.

## Flow* Behavioral Anchors

- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.h:79-108`: `Flowpipe` stores `tmvPre`, `tmv`, and `domain`, giving Flow* a split flowpipe representation instead of only a materialized endpoint box.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:415-420`: `Flowpipe::compose_normal` computes a normal-domain range for the right map and inserts it into `tmvPre` with `insert_ctrunc_normal`.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:450-455`: `Flowpipe::intEvalNormal` composes in normal form and then interval-evaluates the result.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:457-514`: `Flowpipe::normalize` removes constants, computes coordinate scales from magnitudes, builds fresh normal variables, and inserts the normalized variables into the stored map.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:857-1039`: fixed-step interval-remainder advance performs endpoint evaluation, constant removal, insertion, range evaluation, scaling, and `new_x0` construction before Picard validation.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:1233-1445`: adaptive interval-remainder advance uses the same normal insertion/preconditioning sequence inside the step-shrink loop.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:2123-2415`: fixed-step symbolic-remainder advance combines the normal insertion path with symbolic queue updates.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:2715-3040`: adaptive symbolic-remainder advance combines normal insertion, symbolic queue updates, and adaptive Picard validation for the benchmark-relevant path.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:1026-1053`: scalar `insert_ctrunc_normal` delegates to the Horner form and adds the old remainder to the inserted result.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:3378-3388` and `3419-3428`: `TaylorModelVec::insert_ctrunc_normal` applies scalar insertion component by component.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:3453-3459`: `scale_assign` scales each Taylor-model component by the inverse coordinate magnitude.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:3462-3468`: `rmConstant` removes the polynomial constant before range/scaling work.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:3748-3794`: vector normalization constructs scaled fresh variables and inserts them into the current map.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:3812-3821`: `polyRangeNormal` evaluates polynomial ranges on normal domains.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Polynomial.h:1530-1550`: `ctrunc_normal` moves high-degree polynomial terms to interval uncertainty by normal evaluation.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Term.h:167-203`: normal interval evaluation treats time with the step exponent table and state variables with normal-domain factors.
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:4247-4290` and `4294-4346`: the Horner insertion path recursively substitutes variables, multiplies through time factors, normal-truncates, and records intermediate ranges.
- `/srv/local/shengenli/flowstar/benchmarks/continuous/vanderpol/vanderpol.cpp:31-37`, `52-72`, and `74-90`: the local Van der Pol benchmark uses default adaptive settings and calls `reach` with a symbolic remainder queue.

## Function-By-Function Behavioral Map

| Flow* function or structure | Input/output role | Normalized variables | Substitution path | Constant handling | Range and scale | Uncertainty accounting | Current PyTorch gap | Python target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Flowpipe` fields, `Continuous.h:79-108` | Carries a left preconditioning map, a right normal map, and the domain for one flowpipe. | State variables live in normal coordinates; time has its own step domain. | The stored maps are composed when ranges or endpoint sets are needed. | Constants belong in the preconditioning side after normalization. | Domain and right map determine range calls. | Remainders stay attached to Taylor-model components. | Endpoint-box reset discarded the split representation. | `FlowstarNormalFlowpipeState` stores `tmv_pre`, `tmv_right`, domain, center, and scales. |
| `Flowpipe::compose_normal`, `Continuous.cpp:415-420` | Produces a composed Taylor-model vector from the split flowpipe. | The right map is evaluated over normal domains. | Inserts the right map into `tmvPre`. | Uses the already normalized split. | Requests normal polynomial ranges before insertion. | Insertion remainders include old and inserted uncertainty. | No reset path substituted through a carried normal map. | `insert_ctrunc_normal_like` composes `outer` through `inner` and records inserted widths. |
| `Flowpipe::intEvalNormal`, `Continuous.cpp:450-455` | Evaluates the composed flowpipe as intervals. | Non-time variables use normal-domain factors. | Calls normal composition, then interval evaluation. | Constants are included in the composed result. | Returns interval boxes from the composed map. | Uses component remainders from insertion. | PyTorch reports mostly used endpoint boxes from the accepted map. | Diagnostics compare endpoint boxes, inserted endpoint boxes, and normalized reset boxes. |
| `Flowpipe::normalize`, `Continuous.cpp:457-514` | Rewrites a flowpipe into fresh normal variables. | New state variables are centered and scaled to `[-1,1]`. | Inserts fresh scaled variables into the existing map. | Extracts constants as the new center. | Magnitudes of polynomial ranges determine coordinate scales. | Inserted truncation/cutoff uncertainty is kept in remainders. | `_normalized_tm_from_box` normalized boxes without preserving composed dependencies. | `_flowstar_normalized_insertion_transition` extracts centers, computes scales, and builds `center + scale*r`. |
| Fixed real ODE symbolic advance, `Continuous.cpp:2123-2415` | Advances one step while carrying symbolic remainder state. | Uses the same normal flowpipe coordinates as interval advance. | Inserts endpoint maps before symbolic queue updates. | Removes endpoint constants before scaling. | Builds `new_x0` from center and scale. | Adds insertion uncertainty plus symbolic queue propagated intervals. | Queue and insertion were separate experiments. | This branch isolates insertion while leaving queue state untouched unless its reset mode is selected. |
| Adaptive real ODE symbolic advance, `Continuous.cpp:2715-3040` | Shrinks and validates candidate steps on the benchmark-relevant path. | Candidate state variables remain normal through validation. | Accepted maps are inserted into the previous right map before the next step. | Constants become the next local center. | Accepted step range determines scaling; failed steps shrink h. | Picard and insertion uncertainties feed target-containment checks. | PyTorch had target containment but no carried normal insertion. | `flowpipe_step_flowstar_style_adaptive(..., reset_mode="normalized_insertion")` carries state across accepted segments. |
| `TaylorModelVec::insert_ctrunc_normal`, `TaylorModel.h:3378-3388` and `3419-3428` | Applies scalar insertion to every output component. | Inner vector components are normal-coordinate Taylor models. | Each output polynomial substitutes the inner vector. | Scalar insertion adds the old remainder after substitution. | Output range is computed after insertion. | Dropped/cutoff terms move to interval remainders. | No public helper existed for insertion diagnostics. | `insert_ctrunc_normal_like` supports both scalar `TaylorModel` and `TMVector`. |
| Scalar insertion Horner path, `TaylorModel.h:1026-1053`, `4247-4290`, `4294-4346` | Substitutes a vector into one Taylor model efficiently. | Uses normal evaluation and time factors during recursive insertion. | Horner recursion substitutes old variables into new variables. | Old remainder is added to the inserted model. | Intermediate ranges are stored for conservative remainder work. | Truncation and cutoff intervals are accumulated. | Exact Horner/intermediate stack was too Flow*-specific to copy. | Direct clean-room composition uses expanded work order, truncates, cuts off, and adds range intervals. |
| `TaylorModelVec::normalize`, `TaylorModel.h:3748-3794` | Builds a normalized Taylor-model vector and domain. | Non-time variables are mapped to normal intervals. | Inserts scaled fresh variables into the existing vector. | Centers are removed and preserved separately. | Coordinate magnitudes become scaling factors. | Normal insertion accounts for terms dropped during normalization. | PyTorch normalized from boxes only. | `FlowstarNormalFlowpipeState.normalized_initial_tm()` rebuilds fresh normal initial sets from center/scale. |
| `rmConstant`, `TaylorModel.h:3462-3468` | Removes polynomial constants from every component. | Leaves remaining normal variables origin-centered. | Prepares a zero-centered map for insertion/scaling. | Constants become the next center. | Zero-centered range feeds magnitude computation. | Remainders are unchanged by constant removal. | Existing code did not expose component constant removal for resets. | `_tmvector_constant_part` and `_tmvector_rm_constants` split constants cleanly. |
| `scale_assign`, `TaylorModel.h:3453-3459` | Scales each vector component by an inverse magnitude. | Makes the right map fit the next normal box. | Applied after insertion and range computation. | Centers are already removed. | Uses inverse scales from inserted ranges. | Scaling also scales remainder intervals through Taylor arithmetic. | Endpoint box reset did not keep a right map to scale. | `_scale_tmvector_components` builds `tmv_right` and applies cutoff. |
| `polyRangeNormal` and `intEvalNormal`, `TaylorModel.h:3812-3821`, `Term.h:167-203` | Bounds polynomial or full Taylor-model values on normal domains. | State variables use normal intervals; time uses step powers. | Supplies conservative range data for insertion and validation. | Constants contribute normally unless already removed. | Range widths drive diagnostics and scale choices. | Remainder intervals are added after polynomial bounds. | Ordinary domain evaluation only approximates Flow* normal evaluation. | Current implementation uses normal state domains and reports conservative width comparisons. |
| `ctrunc_normal`, `Polynomial.h:1530-1550` | Truncates high-order terms under normal-domain range evaluation. | Normal domains bound discarded monomials. | Applied after multiplication/insertion. | Constants below order are kept unless cut off. | Dropped polynomial range is added to uncertainty. | High-degree terms are not lost. | Earlier cutoff/truncation diagnostics were not tied to insertion. | `_insert_ctrunc_normal_like_scalar` accumulates `insertion_truncation_width` and `insertion_cutoff_width`. |
| `Picard_ctrunc_normal`, `TaylorModel.h:2641-2644`, `Continuous.cpp:3918`, `3964`, `3998` | Builds and validates the Picard image using normal truncation. | Candidate flowpipe variables are normal. | Interacts with insertion by validating the map produced after normalized reset. | Center/scaling from insertion defines the next candidate set. | Remainder target checks decide acceptance or h shrink. | Picard truncation uncertainty is separate from insertion uncertainty. | PyTorch validation existed, but the reset feeding it was an endpoint box. | Keep target radius `1e-4`; only change the opt-in reset feeding the next Picard step. |

## PyTorch Implementation Map

- `src/torch_tm_flowpipe/flowpipe.py`: `FlowpipeSegment` gained optional `flowstar_normal_state` and `flowstar_normal_stats` fields so the opt-in path can return state without changing default callers.
- `src/torch_tm_flowpipe/flowpipe.py`: `FlowstarNormalFlowpipeState` stores the current normal reset box, right normal map, normal domain, per-coordinate centers, scales, and diagnostics.
- `src/torch_tm_flowpipe/flowpipe.py`: `insert_ctrunc_normal_like` is the public clean-room insertion helper. It composes `outer` through `inner`, truncates to the requested output order, applies cutoff, and accumulates truncation/cutoff uncertainty into the Taylor-model remainder.
- `src/torch_tm_flowpipe/flowpipe.py`: `_flowstar_normalized_insertion_transition` removes endpoint constants, inserts the nonconstant endpoint through the stored normal map, computes coordinate scales from inserted ranges, builds the next normalized reset Taylor model, and records reset/insertion diagnostics.
- `src/torch_tm_flowpipe/flowpipe.py`: `flowpipe_step_flowstar_style_adaptive` accepts `reset_mode="normalized_insertion"` plus a carried `flowstar_normal_state`. Other reset modes keep their previous behavior.
- `src/torch_tm_flowpipe/__init__.py`: exports `FlowstarNormalFlowpipeState` and `insert_ctrunc_normal_like` for tests and external diagnostics.
- `experiments/flowstar_style_rescue_vanderpol.py`: adds `flowstar_style_o6_candidate8_output6_cutoff_insert`, carries the normal state across adaptive segments, emits insertion diagnostics, writes specialized normalized-insertion CSV/report files, and plots reset width and insertion uncertainty.
- `experiments/flowstar_one_step_oracle.py`: writes stable `oracle_after_insertion_*` aliases when the oracle is run on an `after_insertion` output directory.
- `tests/test_normalized_insertion.py`: checks sampled containment for direct insertion, truncation/cutoff accounting, normalized-insertion endpoint containment, and unchanged default reset behavior.
- `tests/test_flowstar_style_rescue_experiment.py`: includes a tiny specialized-output smoke test that verifies multiline CSV/report formatting.

## Attribution-Targeted Scale Update

The h10 attribution run identifies `right_map_scaling` as the dominant width source: max right-map width sum `21.883875748631645`, max output-range width sum `21.800746276090599`, max Picard residual width sum `0.00021478626710226962`, and max insertion truncation width sum `7.5777280987120131e-05`. The source-map target is therefore the Flow* scale update, not symbolic queue tuning or Horner truncation.

Observed PyTorch component: `normalized_right_map_range_width_sum` grows with the inserted right map before reset scaling. Relevant Flow* mechanism: `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:457-514` computes constants, removes them, calls `remainder.remove_midpoint(m)`, adds that midpoint into the next center, then measures `tmvRange[i].mag(sup)` before dividing the right-map component by `sup`. This means Flow* does not let an asymmetric interval remainder inflate the right-map magnitude while leaving its midpoint outside the center.

Observed PyTorch component: `scale_x` and `scale_y` track the inserted right-map magnitude, and the attribution report recommends `right-map scalar alignment`. Relevant Flow* mechanism: `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:3453-3459` applies scalar assignment component by component after the magnitude has been selected. The clean-room target is the same component-local scalar rule: recenter each inserted component remainder first, add that midpoint to the reset center, then compute the scale from the recentered range.

Implementation target: `src/torch_tm_flowpipe/flowpipe.py` exposes this as the opt-in normalized-insertion scalar path used by `flowstar_style_o4_target_insert_scalars` and `flowstar_style_o6_candidate8_output6_insert_scalars`. The default normalized insertion configs stay as the h10 baseline, so the scalar run can be compared directly against the reported o4/o6 baseline widths.

## Equivalence And Differences

Equivalent intent:

- Constants are removed before scaling the next normal coordinate frame.
- High-order and cutoff terms are not silently dropped; their ranges are added to interval remainders.
- The inserted endpoint range determines per-coordinate scale factors for the next normalized reset.
- State is carried across accepted segments only when the opt-in reset mode is selected.

Conservative differences:

- Flow* uses a Horner-form insertion algorithm with intermediate range stacks. The PyTorch helper uses direct term-by-term composition at an expanded work order, then truncates.
- Flow* normal evaluation uses its step exponent table. The PyTorch helper evaluates over the Taylor-model domain, which is already normal for the state variables in this experiment.
- Flow* stores a richer `tmvPre`/`tmv` split. The PyTorch state records the accepted step map and the right normal map needed for diagnostics and reset propagation, but it remains a research prototype rather than a parity claim.
- The symbolic remainder queue remains a separate opt-in mechanism; this branch isolates normalized insertion so the width impact is measurable.

## Output Contract

The targeted run writes:

- `outputs/flowstar_normalized_insertion_rescue/normalized_insertion_summary.csv`
- `outputs/flowstar_normalized_insertion_rescue/normalized_insertion_segments.csv`
- `outputs/flowstar_normalized_insertion_rescue/normalized_insertion_reset_diagnostics.csv`
- `outputs/flowstar_normalized_insertion_rescue/normalized_insertion_validation_attempts.csv`
- `outputs/flowstar_normalized_insertion_rescue/normalized_insertion_vs_flowstar_comparison.csv`
- `outputs/flowstar_normalized_insertion_rescue/normalized_insertion_report.md`
- `outputs/flowstar_normalized_insertion_rescue/reset_width_compare.png`
- `outputs/flowstar_normalized_insertion_rescue/insertion_uncertainty_vs_t.png`

If the run reaches a horizon with Flow* reference data, the generic comparison writer can also produce `rescue_vs_flowstar_ratio_trace.csv`, `rescue_vs_flowstar_report.md`, overlay plots, and `width_ratio_vs_t.png`.

## Verification Questions

The report answers the requested branch-decision questions:

- Did normalized insertion beat the prior failure time near `2.400737667399793`?
- Did it reach the requested horizon?
- Did reset widths shrink before `t ~= 2.4`?
- Did 2x, 5x, or 10x Flow* width-ratio crossings move later?
- Did insertion truncation/cutoff/remainder uncertainty dominate the width?
- What was the runtime cost?
- What failure mode remains if the run still fails?

The branch is a merge candidate only if the normalized-insertion run reaches horizon 5 or improves the prior validated time. Otherwise the report marks it `NEEDS_MORE_WORK` and preserves the diagnostics for the next mechanism pass.
