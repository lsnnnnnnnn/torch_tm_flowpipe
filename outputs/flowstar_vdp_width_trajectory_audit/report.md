# Flowstar Van der Pol Width/Trajectory Audit

This is an audit over existing artifacts. It does not add a new flowpipe mechanism, does not add a symbolic queue variant, and does not claim Flow* parity beyond exact generated-vs-original Flow* segment-field equality.

## 1. What is already exact?

Original Flow* vs generated Flow* parity: `completed/completed`.
Segment count: original=`290`, generated=`290`; max_abs_segment_field_diff=`0`.
Last segment width sum: `0.7047129999999997`; tube width sum: `9.5663119999999999`.

## 2. What is visually/short-horizon close?

Order 4 loose fixed-step trajectory overlay is short-horizon close in a visual/segment sense: last ratio `1.2462973043534347`, tube ratio `1.1085948747450092`.
Order 8 strict fixed-step overlay is closer: last ratio `1.0435607171356178`, tube ratio `1.0325273555652099`.
Phase/t-x/t-y/width-over-time plots are linked below. Sampling trajectories are visual diagnostics only, not proof.

## 3. What is normalized insertion h5 status?

The h5 normalized insertion artifact reached t=`5` with `last=1.042824337979132, tube=0.99991747314195634, overlap max/median=1.042824337979132/0.50300691246173512`. It is width-close for that requested horizon.

## 4. What is normalized insertion h10 status?

o4 target insert: last_validated_t=`6.4730088058091901`; `last=14.049230879967741, tube=1.1807087361819038, overlap max/median=14.049230879967741/1.3359318424177438`; reached h10: no. Widths are not last-segment-close, though the tube ratio is much nearer than o6.
o6 candidate8/output6: last_validated_t=`7.4960392581387341`; `last=101.26404571229142, tube=2.4989346486923725, overlap max/median=101.26404571229142/3.0392892340204725`; reached h10: no. It reaches farther but is far wider late.

## 5. What did width attribution find?

Which component causes width growth? `right_map_scaling`; max component widths: {'insertion_truncation': 7.577728098712013e-05, 'right_map_scaling': 21.883875748631645, 'picard_residual': 0.00021478626710226962, 'output_range_evaluation': 21.8007462760906}.
Is the next fix source-map insertion Horner/intermediate range, symbolic queue, or polynomial range bounding? `right-map scalar alignment`.

## 6. What did scalar alignment/width fix do?

Scalar alignment/width fix did not materially improve h10: o6 stayed at t=`7.4960392581387341` with `last=101.25756903259277, tube=2.4987918815501629, overlap max/median=101.25756903259277/3.0392717282476918`.

## 7. What did split/v2 symbolic queue do?

Did v2 reach h10? no; best v2 t=`7.4960392581387341`.
Did v2 beat no_queue on last_validated_t? no; no_queue best t=`7.4960392581387341`.
Did v2 reduce max reset width before the common horizon? no; no_queue=`21.902999702465412`, split=`21.902999702465412`, v2=`21.902999702465415`.
Did v2 keep symbolic width out of target check? yes; max target contribution=`9.8813129168249309e-324`.
split best: t=`7.4960392581387341`, `last=124.46569353503585, tube=2.9696033909707853, overlap max/median=124.46569353503585/3.1684845492352589`.
v2 best: t=`7.4960392581387341`, `last=101.53378043476893, tube=2.5046094332157787, overlap max/median=101.53378043476893/3.3260152415923701`, sample containment `passed`, Flow* one-step oracle `not_completed`.

## 8. What did accepted-step comparator do?

The accepted-step comparator is diagnostic infrastructure, but the current accepted ordinal comparison is `accepted_ordinal_trace_diff_noncausal`: step `0` has Flow* h=`0.012500000000000001` versus PyTorch h=`0.025000000000000001`.
Channel attribution is therefore invalid/noncausal and must be treated as `adaptive_step_alignment_mismatch`.
Next comparator fix: produce `attempt_aligned_trace_diff.csv` and `forced_h_trace_diff.csv`.

## 9. What did GPU benchmark prove?

- Are current data structures tensorizable, or are Python dict/sparse loops blocking GPU? The current production `Polynomial`/`TaylorModel` path uses Python dictionaries keyed by exponent tuples and scalar tensors, so Python object and sparse-loop overhead blocks real GPU use. Existing sparse Python rows ran at 9.7e-05x to 0.091x of torch dense CPU throughput for the measured scalar batches.
- Is the project still justified as PyTorch-native, or should plant remain Flow* C++? Dense batched kernels show clear CUDA speedups at realistic batch sizes.
This is a representation-redesign signal, not h10 rescue evidence.

## 10. Overall Conclusion

We are not merely timing out. The PyTorch rescue has real width/trajectory differences versus Flow* in late horizon. Short-horizon and h5 evidence can be close, but h10 normalized insertion is not width-close: o4 stays tighter but stops earlier, o6 reaches farther but becomes far wider. Symbolic queue v2 improves diagnostics, not horizon/tightness. The next correctness task is aligned Flow* step comparison and right-map/preconditioning/source-order width mechanism; the next performance task is dense batched TM representation.

## Claim Boundary Checks

- `endpoint_ratio_disabled_without_flowstar_endpoint`: `pass` - Flow* GNUPLOT rows have endpoint_ratio_allowed=false; no endpoint ratio is emitted.
- `flowstar_parity_exact_only`: `pass` - segment_count_match=true; max_abs_segment_field_diff=0
- `missing_artifacts_recorded`: `pass` - Missing artifact count recorded explicitly: 0
- `accepted_step_h_or_t_mismatch_noncausal`: `noncausal_guarded` - first_material_channel must be adaptive_step_alignment_mismatch; ordinal channel attribution invalid/noncausal
- `h10_not_timeout_only`: `pass` - h10 failures include width/trajectory divergence, not only runtime timeout.
- `samples_visual_only`: `pass` - Sampling trajectories and overlays are recorded as visual diagnostics only, not proof.

## Plot Links

- `outputs/trajectory_audit/figures/contact_sheet_torch_orders.png`
- `outputs/trajectory_audit/figures/contact_sheet_flowstar_overlays.png`
- `outputs/trajectory_audit/figures/contact_sheet_width_trends.png`
- `outputs/trajectory_audit/figures/flowstar_rem1e-4_cut1e-10_h0p01_s10_o4_overlay_phase_xy.png`
- `outputs/trajectory_audit/figures/flowstar_rem1e-4_cut1e-10_h0p01_s10_o4_overlay_t_x.png`
- `outputs/trajectory_audit/figures/flowstar_rem1e-4_cut1e-10_h0p01_s10_o4_overlay_t_y.png`
- `outputs/trajectory_audit/figures/flowstar_rem1e-4_cut1e-10_h0p01_s10_o4_overlay_width_over_time.png`
- `outputs/trajectory_audit/figures/flowstar_rem1e-10_cut1e-15_h0p0025_s10_o8_overlay_phase_xy.png`
- `outputs/trajectory_audit/figures/flowstar_rem1e-10_cut1e-15_h0p0025_s10_o8_overlay_t_x.png`
- `outputs/trajectory_audit/figures/flowstar_rem1e-10_cut1e-15_h0p0025_s10_o8_overlay_t_y.png`
- `outputs/trajectory_audit/figures/flowstar_rem1e-10_cut1e-15_h0p0025_s10_o8_overlay_width_over_time.png`
- `outputs/flowstar_width_fix_h10/overlay_rescue_vs_original_flowstar_phase_xy.png`
- `outputs/flowstar_width_fix_h10/overlay_rescue_vs_original_flowstar_t_x.png`
- `outputs/flowstar_width_fix_h10/overlay_rescue_vs_original_flowstar_t_y.png`
- `outputs/flowstar_width_fix_h10/width_ratio_vs_t.png`
- `outputs/flowstar_width_fix_h10/reset_box_width_trace.png`
- `outputs/flowstar_width_fix_h10/residual_vs_t.png`
- `outputs/flowstar_width_fix_h10/step_size_trace.png`
- `outputs/flowstar_insertion_width_attribution/insertion_component_stack.png`
- `outputs/flowstar_insertion_width_attribution/o4_vs_o6_width_sources.png`

## Missing Requested Plot Paths

- None

## Generated Audit Files

- `summary.csv`
- `width_comparison_ledger.csv`
- `trajectory_overlay_ledger.csv`
- `claim_boundary_checks.csv`
- `evidence_inventory.csv`
- `evidence_inventory.md`
- `report.md`

