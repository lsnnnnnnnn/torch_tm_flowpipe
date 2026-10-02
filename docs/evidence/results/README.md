# 2026-10-01 experiment evidence carried by this branch

This folder mirrors the local run records used by the new report: `archcomp26_20261001/` and `huan_quad_stage_a_40_20261001/`. It includes copied START/RESULT/log files, compact range records, figures, CSVs, and the interim DOCX/PDF. New independent-process campaigns cover ACC, two-physical-state Single Pendulum, and corrected-unsafe Attitude, each with four methods × six processes and independent saved-range audits. It also contains full P3 TORA remain, official-file TORA reach-sigmoid `u=11f` four-method 500-step runs, a TORA reach-tanh first-period historical reuse audit, paper-equation constant-speed-disturbance Unicycle four-method 500-step runs, Docking four-method full numeric runs, Airplane full-box entry and discrete Euler diagnostics, Balancing fixed-repository raw-four-state four-method early stops, and P3 Double Pendulum less-robust split-2/split-4 attempts. TORA reach-sigmoid and Unicycle each have a four-method terminal-width CSV linked from the report. The saved-data DP four-method comparison provides absolute endpoint widths, all-time tube unions, PNG/PDF, and a generated MATLAB script. The paper-equation QUAD four-method comparison adds a full native 1024×1000 range scan, full twelve-state terminal-width CSV, saved-range remain diagnostic, and a figure with native/P3 full-time saved tubes and Huan/Xiangru endpoint intervals. Double Pendulum more-robust has a P3 full-box first-period attempt, a full-contract numerical early stop, a nominal-boundary numerical replay, and a strict-interior numerical replay; none is a full-horizon four-method result. Historical NAV Xiangru, old `ours`, and robust native full runs are audited separately without re-entering them as new attempts; the new NAV standard native/current-P3 full runs and NAV robust current-P3 full run have separate receipts. NAV standard/robust historical-vs-new tube and width figures come from full saved ranges with their generations labeled. The attempt index one directory above records original remote directories and distinguishes full runs, smoke runs, and incomplete attempts. Method-independent diagnostics are described in their own summaries and the progress report.

Most official third-party `.onnx` and `.mat` model files are not duplicated here; the Double Pendulum more numerical replay directories include convenience copies of their fixed ONNX input. Other fixed source locations and original server copies are identified in the contract notes and run records. The completed full native paper-QUAD 417,792,000-byte `ranges.bin` stays in its original server directory; its complete scan and original small receipts are mirrored [here](archcomp26_20261001/native_quad_paper_full50_001/SUMMARY.md). The new NAV standard native 640-box×600-step 58,368,000-byte `ranges.bin` likewise remains on the server; its [original small receipts and full saved-interval scans](archcomp26_20261001/nav_author_standard_native_full30_001/SUMMARY.md) are mirrored here. The historical NAV robust native 25-box range file is included only for historical contract audit. File copies, path/link checks, and record parsing in this package used no content digest validation. The raw binary range files do not by themselves encode solver acceptance or certify the neural-network bounder.

This is a dated evidence snapshot. A later distinct run needs its own attempt entry and original directory. The native paper-QUAD row was updated in place when its original long-running job naturally finished; the earlier raw receipts were not rewritten.

The [official `u=11f` TORA reach-sigmoid four-method saved-tube figure](archcomp26_20261001/tora_reach_sigmoid_u11_fourway_saved_figure_20261002/SUMMARY.md) uses all 500 stored `x1/x2` whole-step boxes per method and shows T=5 endpoint-bound differences separately. All four saved endpoints lie in the target; Huan/Xiangru curves coincide. The three author-engine property checkers were not run; native's `VERIFIED` is its author endpoint-checker label. The figure is a box projection, not correlated octagon geometry or an independent end-to-end NNCS certificate. Its MATLAB script was generated but not executed.

The [historical official `u=11f` TORA reach-tanh four-method saved-tube figure](archcomp26_20261001/tora_reach_tanh_historical_fourway_saved_20261002/SUMMARY.md) uses copied 500-step raw range records for Huan, Xiangru, historical P3, and Flow* native. It includes a four-state absolute T=5 endpoint CSV and a same-axis `x1/x2` plot with small endpoint-bound differences expanded alongside. All four saved endpoints lie in the target; Huan/Xiangru saved bounds coincide. Historical P3 used `engine_linear_leaf_v2`, rather than the current working P3. The archived GPU results have no property verdict; native's `VERIFIED` is its author endpoint-checker label. These copies and adjacent run receipts are numerically cross-checked but not content-bound, and the figure is neither an independent end-to-end certificate nor a speed ranking. Its MATLAB script was generated but not executed.

The [new current working-P3 official `u=11f` TORA reach-tanh run](archcomp26_20261001/tora_reach_tanh_official2026_mat_u11_workingp3_full500_20261002_001/SUMMARY.md) has a separate 50-step gate and a 500-step full-initial-box run, with CPU source/model preflight, original supervisor receipts and logs, raw ranges, 500 accepted observations, and an independent saved-interval scan. Its `engine_quad_normalization_center` identity is distinct from the archived `engine_linear_leaf_v2` P3 result. The [mixed-generation four-state endpoint table](archcomp26_20261001/tora_reach_tanh_official2026_mat_u11_workingp3_full500_20261002_001/terminal_current_p3_historical_author_fourway_T5.csv) labels the new P3 and three historical author/native sources separately. All four saved T=5 target-coordinate endpoint boxes lie in target; the new P3 property checker was not run. The dated 187/189-attempt DOCX/PDF exports remain snapshots; the current index is 198 attempts with 38 new full cells and eight historical coverage cells.

The [paper-equation constant-speed-disturbance Unicycle four-method saved-tube figure](archcomp26_20261001/unicycle_paper_speed_w_constant_v1/plots/fourway_saved_20261002/SUMMARY.md) overlays all 500 stored `x1/x3` whole-step tubes and separately compares T=10 `x3/x4` endpoints. Huan/Xiangru bounds coincide; their stored endpoints do not satisfy the full four-state target inclusion criterion, while P3 and native do. Target markers refer only to T=10, and the adjacent saved-endpoint window audit gives the four-state timing detail. PNG, PDF, geometry JSON, and an unexecuted MATLAB script are included; no solver run was repeated for this plot.

The [TORA remain `h=0.05` four-method supplemental profile](archcomp26_20261001/tora_remain_h005_fourway_saved_20261002/SUMMARY.md) links fresh P3, Huan, Xiangru, and native 12-box×400-step receipts, independent scans, a 16-row absolute-width CSV, and a saved `t,x4` figure. It is outside the fixed `h=0.1` four-method matrix. The [CPU accepted-step octagon stream smoke](archcomp26_20261001/tm_octagon_stream_harmonic_smoke_20261002_001/SUMMARY.md) has its own raw JSONL, source snapshots, geometry, plots, and an unobserved baseline comparison; it is not an ARCH-COMP26 instance attempt. The separate [native Flow* eight-direction gate](../../ARCHCOMP26_NATIVE_OCTAGON_PRODUCTION_GATE_20261002.md) retains failing analytic-containment probes and prevents promotion to native QUAD plots. These diagnostic records are outside the fixed four-method matrix and the dated 187-attempt DOCX/PDF snapshots.

The native gate now includes a [frozen-library paper-QUAD first-box,
one-controller, one-small-step failure receipt](archcomp26_20261001/native_quad_sr_octagon_gate_20261002_005/README.md)
and a [separate copied-library symbolic-remainder refinement mechanism check](archcomp26_20261001/native_quad_sr_refinement_fix_20261002_006/README.md).
Observer off/on had equal reach status, controller response and terminal axes.
The unpatched symbolic-remainder path's terminal `x7/x8` bounds missed finite
numerical trajectories admitted by the C++ CROWN affine-plus-residual control
relaxation by up to `1.98259e-6`; the sampled control has not been shown to
be the neural network output. The copied-library skip widened bounds greatly
and is not a production fix. These method diagnostics do not add a benchmark
attempt or change the current 198-attempt, 38-new-full-cell numerical coverage;
they prevent promoting native accepted/`VERIFIED` to independent soundness.

The [stage report directory](archcomp26_20261001/stage_report/README.md) now contains a 189-attempt, 38-new-full-cell editable DOCX and Word-exported 21-page PDF with 13 saved-data figures and the native QUAD short-gate caveat. The older 187-attempt DOCX/PDF remains a separate dated snapshot. The Markdown report and linked raw records are the continuing source of truth.

Three further actual Airplane continuous one-step native runs are preserved as
[the unused-overload alignment control](archcomp26_20261001/native_airplane_first_reject_trace_20261002_004/README.md)
and [the active Real-path Picard trace](archcomp26_20261001/native_airplane_first_reject_trace_20261002_005/README.md),
plus [the widened `[-1,1]` trace](archcomp26_20261001/native_airplane_rem1_first_reject_trace_20261002_006/README.md).
All used the full initial box, one controller RPC, and one requested `0.01 s`
step; all refused with no accepted range. The two active-path traces identify
five proposed remainders outside their respective preset intervals, with
different failed-coordinate sets. The current attempt index is **198**, while the 21-page DOCX/PDF remains the
unchanged **189-attempt** snapshot; numerical full-horizon coverage remains
38 new plus eight audited historical same-contract cells. These traces do not
establish an Airplane full-time or property result.

The [isolated native QUAD VAR-tail candidate gate](archcomp26_20261001/native_quad_var_tail_repair_gate_20261002_007/README.md)
rebuilds a copied Flow* library with a two-site variable truncation-tail
repair, preserving the remainder refinement loops. Ordinary harmonic short
checks and the paper-equation QUAD first-call, first-step finite-sample checks
passed, and the repaired QUAD intervals were much narrower than the copied
skip-refinement diagnostic. The original failing library and full50 receipts
are unchanged. The [independent short-step interval gate](archcomp26_20261001/native_quad_independent_interval_gate_20261002_001/README.md)
checked the saved first-call RPC domain and relaxed-control hull throughout
`0.005 s`: 27/32 repaired saved columns contained the independent intervals,
versus 23/32 from the original library. A [later algebraic check](archcomp26_20261001/native_quad_algebraic_gate_20261003_001/README.md)
established 20/20 composed physical-state comparisons on that RPC domain.
The [initial recenter audit](archcomp26_20261001/native_quad_initial_recenter_gate_20261003_001/README.md)
found the RPC's first three lower bounds one binary64 ULP inside the C++
initial box. Thus these checks do not prove inclusion for the whole declared
first box. Other boxes, control periods, and NN/CROWN remain open; the native
production octagon gate stays closed. These diagnostics are not new benchmark
attempts.

The [two-state Single Pendulum four-method saved figure and width table](archcomp26_20261001/sp_two_state_fourway_saved_20261002/README.md)
read the four existing complete 100-step range files without another solve.
The figure overlays the four tube boundaries on one set of axes per physical
state, with endpoint differences below; it does not supply the missing
official three-state MATLAB identity. The [old-author QUAD P3 post-trig 40-step
profile](huan_quad_stage_a_40_20261001/post_trig_phase_profile_v1/SUMMARY.md)
is a separate short diagnostic with direct numerical equality to the saved
observer-on run, not a new 1,000-step result.

The [Airplane native binary-six first-step sample](archcomp26_20261001/native_airplane_binary6_firststep_20261002_007/README.md) and [opposite-corner gate](archcomp26_20261001/native_airplane_binary6_coverage_gate_20261002_008/README.md) each accepted one numerical `0.01 s` step from one of 64 planned subboxes. The upper-corner saved tube crosses the safe boundary, so its property status is Unknown and the other 62 subboxes remain unattempted. A separate [actual-ONNX high-corner point replay](archcomp26_20261001/native_airplane_highcorner_point_replay_20261002_009/README.md) found no crossing in 11 numerical samples; it is not a four-method attempt or an all-box certificate.

The [TORA remain `h=0.1` first-refusal trace](archcomp26_20261001/author_tora_remain_h01_refusal_trace_20261002_001/README.md) identifies Huan lane 2 `x2` Picard self-containment failure at step 190. The first tracing attempt failed after a 189-step prefix because of an instrumentation API mismatch; both raw results remain saved. The [x2-only wider remainder profile](archcomp26_20261001/author_tora_remain_h01_x2rem002_20261002_001/README.md) completed one period but again stopped at step 192 before `T=20`. These four new solver runs are indexed as supplemental and do not replace the fixed uniform-remainder `h=0.1` main cell. The [Double Pendulum more bounded validated-witness attempt](archcomp26_20261001/dp_more_validated_witness_20261002_001/AUDIT.md) stopped at `t=0.1954`, before the saved numerical violation candidate; it gives no rigorous counterexample and is outside the four-method attempt index.
