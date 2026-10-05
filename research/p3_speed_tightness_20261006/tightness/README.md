# Unchanged-polynomial control remainder candidate

The executable candidate is `run_quad_control_candidate.py`. It uses the October 5 private-output / weighted256 wrapper, retaining its resource cap, P3 order, initial partition, ODE, strict injection and endpoint, full SR history, observer, first-refusal stop and saved-width comparison. Default `batch2` covers 1024 boxes and 40 steps; `--mode full50` explicitly requests 1000 steps. This folder contains no new solver result.

```sh
python -B run_quad_control_candidate.py \
  --base-wrapper /path/to/october5/run_quad_candidate.py \
  --gate /path/to/passed/private-output-gpu-gate \
  --output /new/independent/attempt
```

The inherited wrapper requires physical GPU3 and CPUs 14–17. An optional `--source-runner` selects the explicit saved canonical runner; otherwise the October 5 wrapper's original source applies. Do not choose an output inside the frozen wrapper directory. The same frozen model/config are read; old experiments and checkers are not run. No extension is built and no digest is computed.

`CONTROL_BINDING.json` declares the change and conditional evidence boundary. `CONTROL_RESULT.json` contains every refresh's old/new mean and maximum control remainder widths, number of contracted control rows, actual base/extra NN attempts, and restoration status. The original payload's `nn_calls` still counts its original same-slope calls; a completed candidate has twice as many actual NN calls. No coefficient slope or center is silently replaced. `SAVED_COMPARISON.json` / `ABSOLUTE_WIDTHS.csv` retain the inherited exact-rational saved-bound width comparison across all twelve physical states and both endpoint and tube objects. A narrower width is not a containment claim.

## Conditional mathematical operation

The original strict injection produces the point polynomial `P_u` and remainder `R_old`. Keep `P_u` exactly. Given input `x = P_x + R_x` and another valid affine envelope of the **same controller** on the entire represented input hull,

```
lA*x + lb <= NN(x) <= uA*x + ub,
Dlo = range(lA*P_x + lb - P_u) + lA*R_x,
Dhi = range(uA*P_x + ub - P_u) + uA*R_x,
R_new = R_old intersect [lower(Dlo), upper(Dhi)].
```

`anchored_control.contract` evaluates coefficient products, sums, subtraction of the actual RN `P_u`, polynomial ranges and input-remainder dot products with the existing strict interval operations. It checks that the supplied certificate hull contains the independently recomputed represented input hull. It returns owned data only after finite/order/domain checks; disjoint enclosures raise. The hook writes only control remainder rows 13:16 after this succeeds. It never writes physical-state coefficients, support, normalization, endpoint logic or the SR ledger.

This preserves input-state correlation in `lA*P_x - P_u`, unlike first boxing that difference. It also avoids separately bounding `(lA-T0)*R_x` and then adding `T0*R_x`, which would lose cancellation again. The same-polynomial remainder is locally a subset of its old value. No claim is made that every state width at every later closed-loop time must decrease: changed interval endpoints can affect subsequent relaxations, support, validation and rounding. Promotion requires new complete-state endpoint and tube measurements.

Both old and extra floating CROWN envelopes remain conditional inputs: this does **not** establish an independent floating-point NN/CROWN or end-to-end NNCS certificate. An independently built two-slope module supplies the additional envelope; the original same-slope module, point slope and center remain in use. No author/native range is intersected into P3. The hook intentionally rejects RPC-float32 transport and applies initially only to native-f64 paper QUAD.

## Exact integration points

- `tools/archcomp26_quad_paper_p3_nohash.py`: `execute` installs `crown_bounds` counting, `strict_inject`, strict endpoint, holder, observer and first-refusal machinery before calling `driver.main`.
- `research/gpu_verified_20260930/source/integration/crown_reach.py`: `build_crown` constructs same-slope or independent two-slope modules; `crown_bounds_two_slope` provides both affine matrices and biases. The `box`/same-slope branch calls `inject_controls_s` after native-f64 transport.
- `research/gpu_verified_20260930/source/adapters/quad_control_transfer_20260927/strict_injection.py`: `inject_controls_s` preserves RN point coefficients and charges coefficient error, input remainder and exact displacement from the RN center.
- `quad_controller_hook.install` must run at the wrapped `driver.main` entry, **after** those canonical wrappers are installed. It wraps model construction, NN bounds and strict injection, and captures the sole engine through `initial_sparse_state`. Installing during `prepare` itself would let the canonical runner overwrite these hooks.
- `run_quad_control_candidate.py` supplies that ordering without changing either frozen runner. Its wrapper requires every expected base/extra call and injection to complete and restoration to succeed before invoking the inherited width comparison.

## Saved evidence and next candidate choices

The paired data below are saved `metrics.json#/ctrl_steps` entries, not fresh runs or a causal decomposition. The source's `u_box_width_*` key in same-slope mode is the **affine bias gap**, not the concretized controller output width.

- QUAD sources: `docs/evidence/results/archcomp26_20261001/quad_paper_p3_nohash_v1/full50_001/data/metrics.json` and `quad_paper_huan_full50_001/parity/metrics.json`. At refresh 0, both have exactly equal input hull widths and mean bias gaps `[0.06671007069264898, 0.017206176240886127, 0.01688054894026283]`. P3's first mean control remainder is `0.06671007069265167`, versus Huan's `0.06671007069264899`; the strict initial fee is tiny. By refresh 49, P3/Huan first mean bias gaps are `0.013886383309983995` / `0.006394006243665908`, and x3 input-hull means `0.06207022876274018` / `0.042889536880994776`. This motivates reducing coupled controller enclosure loss, while not proving that it caused all later widening. The saved P3 order is 3, Huan order 2. Existing direct two-slope 40-step results mix narrower and wider states; they are not rerun or treated as a default replacement.
- Official sigmoid sources: `docs/evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/{later05_ours_p3,later05_huan}/payload/metrics.json`. Initial mean bias gaps agree at `4.1961669921875e-05`; at refresh 9 both are zero. Yet P3/Huan x1 hull means are `0.025161972421761103` / `0.025002330087108948`. The saved P3 order is 3 and Huan order 6. This supports testing plant representation/cutoff at fixed P3 order before paying for extra CROWN. Root separately owns the new lower-cutoff experiment.
- NAV robust P3 source: `docs/evidence/results/archcomp26_20261001/nav_author_robust_working_p3_full30_001/metrics.json`. At refresh 29 its mean bias gaps are `[0.00011408366961404681, 0.0005824661254882812]` versus remainder widths `[0.00011414809398873245, 0.0005825751095700843]`. A paired full Huan controller-metrics source was not established in this local audit; author endpoint/tube comparisons alone do not identify the internal cause. RPC-float32 transport also prevents directly applying this native-f64 hook without a separate transport-enclosure argument.

Do not select shared-driver `--crown-domain hybrid` as a shortcut: its direct candidate assignment bypasses the qualified strict injection hook. Likewise, do not discard normalization, endpoint, cutoff or SR error charges. `sparse_exec.py` retains centered-normalization displacement, reciprocal scaling and composition errors; `support.cutoff_spatial_s` moves discarded coefficients into interval remainder. Lowering cutoff retains more polynomial terms but can cost memory/time. More accepted weighted-refinement rounds are another conditional local contraction, with extra map cost and no global monotonicity guarantee.

## Completed local check

`/opt/anaconda3/bin/python -B check_anchored_control.py` passed eight checks with Torch 2.2.2 CPU, the frozen interval functions and the real frozen strict-injection function. Exact-rational cases cover correlated linear cancellation, ReLU breakpoints and a large-constant rounding residual. The hook check uses mock NN modules and covers certificate pairing, duplicate consumption, original polynomial, counters, transport rejection and owned restoration. All digest constructors and accidental CUDA extension discovery are disabled in the check. `CPU_CHECK.json` is the actual receipt. No CUDA kernel, real NN, solver, old checker or old experiment was executed by this check; GPU qualification remains pending.

## Independent TORA candidates

`run_tanh_cutoff_candidate.py` reuses the frozen `expansion/run_b1_candidate.py` with private output, preloaded Horner and fused1. It changes only the new staged tanh configuration's cutoff from 1e-6 to 1e-8. Required arguments are `--base-wrapper`, `--adapters`, `--gate`, and `--output`; optional `--source-snapshot` / `--reference` retain the original meanings. The inherited resource profile is GPU3 / CPUs18–19. Its own comparator verifies the ReLU³/tanh model and internal u=11f, external scale1/offset0, unchanged source bytes and order3, original driver route and 500 accepted observations. It writes 4,000 per-step, per-state endpoint/tube width comparisons and a new saved-endpoint target observation. `TANH_CUTOFF_CPU_CHECK.json` records a passed synthetic 500-step parser test, not a solver result.

`run_sigmoid_order4_candidate.py` reuses the frozen October 5 `run_small1_candidate.py` under GPU2 / CPUs10–13. Required arguments are `--adapters`, `--gate`, and `--output`, with the same optional source/reference overrides. It must stage from the original order3/cutoff1e-6 source. The new configuration uses working order4, point order3, validation order5 under the unchanged `solution_plus_one` policy, and cutoff1e-8. The new staged `run_full.py` updates its declared method, preflight, order argument and order receipt; the staged support module changes its active engine guard to `(6,4)`. Every replacement has an exact occurrence guard and is recorded in `STAGED_INPUTS.json`; the original sources are untouched. The generic strict injection, endpoint, SR, observer, cap and fused1 logic remain unchanged. Its comparator verifies the exact declared source edits and all other source/model bytes, benchmark fields, sigmoid⁴/u11 identity, runtime order and full horizon before writing the same 4,000 width rows. It imports only the tanh file's geometry reader: both official reach contracts have the same target, as fixed in `docs/ARCHCOMP26_TORA_REACH_EXECUTION_GATE_20261002.md`.

`SIGMOID_ORDER4_CPU_CHECK.json` records four new CPU checks: exact source edits and repeat-edit refusal; a real degree4 support through the unchanged strict injection; exact rational residual sample containment with both bias extrema; and the order4 extended basis being a prefix of order5. It does not test the GPU kernel, real NN, complete flowpipe, speed or final width. Both TORA wrappers preserve `property_evaluated=False` in the payload, replace equivalence wording with numerical-width-candidate wording, and never inherit an old property verdict for changed numerical bounds.
