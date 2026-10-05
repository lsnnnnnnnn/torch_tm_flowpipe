# P3 speed and tightness recovery — 2026-10-05

This is a new implementation study on `codex/p3-speed-tightness-20261005`.
Frozen ARCH-COMP26 attempts and source snapshots remain unchanged. Existing
results are read as references; their experiments and checkers are not rerun.
No content digests, hash verification, extension compilation or MATLAB plots
are used. New numerical attempts stop at the first rejected lane.

## Recovered fast implementation

The September 23 timing campaign used `engine_linear_leaf_v2`, including two
explicit CUDA hooks: private output allocation and the Horner edge kernel.
Current P3 launchers preload those libraries, but some omit their bindings.
The paper QUAD launcher already binds the Horner edge kernel; its private
output library was loaded without selecting it for interval operations.

The recovered report branch is `codex/progress-report-20260923`; its server
engine branch is `codex/linear-leaf-validation-v2`. The historical five-process
medians in `HISTORICAL_FAST_BRANCH_TIMINGS.json` explain the recollection:
NAV robust 9.730 s, TORA tanh 7.685 s, old sigmoid 8.208 s, old Unicycle
6.909 s and ACC 7.793 s were below all three corresponding historical arms.
It was not a universal win: Attitude was slower, and Single Pendulum remained
slower than native. No historical benchmark or checker was rerun to recover
these values.

The old campaign is not an interchangeable current benchmark table. Its timing
did not include the current per-step range export. NAV robust and TORA tanh
have matching physical contracts but different numerical implementations.
The old TORA sigmoid control scaling and Unicycle disturbance equations differ
from the current selected contracts. Historical times remain historical.

For the current paper QUAD, Huan/Xiangru use work/point/validation orders
2/1/1 and parity mode; P3 uses 3/2/4 with strict rounding, injection, endpoint
and interval SR bookkeeping, plus two accepted-path weighted rounds. All
three use 1,024 boxes, h=0.005, 1,000 steps and SR capacity 1,000. P3's K=20
rebuild schedule processes its full retained history; it does not truncate
history to 20, and Huan/Xiangru have no equivalent K20 ledger. P3 computes
per-box geometry at each step and saves pooled projections and acceptance/
status data. The complete paper runs have no matched phase profile, so these
known execution differences cannot be assigned individual causal percentages.
See [the existing mode audit](../../docs/HUAN_QUAD_SPEED_AND_MODES.md).

## Private output candidate

`private_outputs.py` selects only the already loaded, PyTorch 2.5.1-specific
`flowstar_seg_private_output_v1` module. It replaces only
`flowstar_gpu.cuda_kernels._ext`, before any graph warm-up/capture. Arithmetic,
global determinism and public allocation filling remain unchanged. The
private extension bypasses redundant fill for eight audited fully overwritten
output allocations; its ninth exported operation, `iv_neg`, is unchanged.

`test_private_outputs.py` checks binding, rejection and restoration using CPU
module doubles. `gpu_gate.py` compares new finite inputs, strided wrapper
inputs, supported empty reductions and changing-input graph replays by direct
tensor bytes. This is a new bounded diagnostic, not the old 237-case checker.

`run_quad_candidate.py` wraps the saved paper QUAD runner and retains its
configuration checks, strict endpoint/injection, validation, complete SR
history and every saved tube/endpoint observation. It writes a separate
optimization receipt and compares the new saved observations directly with
the existing reference. Pooled observation equality does not establish
per-lane Taylor-model identity or an independent NNCS proof.

## Horner and tighter controller candidates

`horner_edge.py` binds the already loaded directed Horner remainder kernel to
a fresh engine. The existing dispatcher retains its shape guards and fallback.
`run_nav_candidate.py` enables it together with private outputs for a new NAV
robust run, retaining all 25 boxes, 600 steps and per-box range records. It
also records the eager import-time helper policy used to prevent compilation.

The current paper QUAD tool now accepts `--crown-relax two-slope`; its default
remains `same-slope`. The two-slope midpoint residual is converted to affine
bias bounds with outward interval addition, then passed through the existing
strict injection. Its actual method, NN call count and certificate boundary
are recorded. This is a new numerical candidate, so its widths must be
measured; it is not an equivalent-output optimization.

The existing controller `hybrid` option is not used here: its direct control
row writes bypass this strict injection. Metadata adapters named `hybrid`
are unrelated to controller coupling.

## Acceptance and comparison

An implementation-only change must preserve the saved ranges, status and
acceptance data. Timing boundaries and sample counts are reported explicitly;
one new process against a historical process is not a stable speed ranking.
Tightness changes need their own numerical candidate and full matching
contract. Endpoint, tube, pooled and per-box widths remain distinct. A few
narrower endpoint coordinates are not an overall tightness advantage.

The existing Python PNG/PDF and JSON/CSV plotting interfaces remain the output
contract. New results must preserve the data needed by those interfaces.
The native octagon production gate remains closed.

## 当前统一比较报告

16 个 benchmark 的时间、全部状态宽度和不能完成的原因已统一重写至[当前中文报告](../../docs/ARCHCOMP26_FINAL_REPORT_DRAFT.md)，并提供[Word、PDF 与完整 CSV](../../docs/evidence/results/archcomp26_report_20261005/README.md)。这份研究说明保留优化机制和所有候选收据，宽度/时间主表以新报告为入口。

旧 NAV P3 的 28.999885 / 18.703898 是 payload wall，不是完整进程时间；历史 NAV / TORA tanh 的 driver call 也不与内部 elapsed 混列。新报告把这些边界拆开。ACC、Attitude、Unicycle 的优势仅限具体保存终点；QUAD、SP、sigmoid 与 NAV robust 不能据此宣称整体更紧。TORA remain 的全初集数值共同前缀是 18.9 秒，保存安全共同前缀是 18.4 秒。

## Verified results so far

Each new timing below is one process, compared with its saved historical
reference. These are not repeated-process rankings.

| New candidate | Covered work | Saved-output result | Matching internal driver loop, old → new | New whole-process wall |
|---|---|---|---|---|
| Private-output GPU gate | 192 new eager/graph byte comparisons | All equal; deterministic algorithms/fill retained | Not a benchmark timer | 20.028 s |
| Paper QUAD, private output, weighted 128 | 1,024 boxes × 40 steps | All 40 pooled tube/endpoint rows and numerical final fields equal | 53.460 → 49.709 s | 60.936 s; historical process 60.384 s |
| Paper QUAD, private output, weighted 256 | 1,024 boxes × 40 steps | All 40 pooled tube/endpoint rows and numerical final fields equal | 53.460 → 39.289 s | 45.708 s; historical process 60.384 s |
| Paper QUAD, private output, weighted 256, full horizon | 1,024 boxes × 1,000 steps | All 1,000 pooled tube/endpoint observations and scientific final fields equal; all 1,024,000 lane-steps accepted | 1350.526 → 997.675 s | 1004.197 s; saved full process 1357.555 s |
| NAV robust, private output + Horner | 25 boxes × 600 steps | All 2,040,000 range bytes, 600 observation rows, initial boxes and final fields equal; author `VERIFIED` retained | 15.532 → 13.689 s | 18.064 s |
| NAV robust, private output + Horner + weighted 32 | 25 boxes × 600 steps | All range bytes, 600 observation rows, initial boxes and final fields equal | 15.532 → 12.667 s | 16.616 s |
| Unicycle, private output + Horner + weighted 1 | 1 box × 500 steps | All 416,584 range bytes, config, 50 control-period records and final fields equal; terminal target label retained | 13.617 → 10.016 s | 15.864 s |
| Unicycle, fused two-round single-row graph | 1 box × 500 steps | Same complete saved output; first live input also matches original refinement bytes and statistics | 13.617 → 7.652 s | 11.698 s; saved process 17.820 s |
| ACC, private output + Horner + weighted 1 | 1 box × 50 steps | All 50 range and safety rows, config, driver arguments and scientific fields equal | 4.720 → 3.982 s | 8.037 s; saved sample 8.790 s |
| ACC, fused two-round single-row graph | 1 box × 50 steps | Same complete saved output; first live input also matches original refinement bytes and statistics | 3.961 s | 8.089 s; no extra whole-process gain over small1 |
| TORA sigmoid, official u=11f, private output + Horner + weighted 1 | 1 box × 500 steps | All 76,000 range bytes and 500 observation rows, config/model and scientific fields equal | 9.794 → 7.759 s | 11.952 s; saved sample 13.959 s |
| TORA sigmoid, official u=11f, fused two-round single-row graph | 1 box × 500 steps | Same complete saved output; first live input also matches original refinement bytes and statistics | 9.794 → 5.255 s | 9.392 s; saved sample 13.959 s |
| Paper QUAD, two-slope + private output + weighted 256 | 1,024 boxes × 40 steps | Complete numerical prefix; widths are mixed, so not an equivalent-output optimization | 39.435 s, different controller relaxation | 46.007 s |

The private128 QUAD short run has no demonstrated whole-process speed improvement.
The private256 short run completed with fewer graph calls and lower measured
process time; this does not establish full-horizon timing or repeated-run ranking.
Its separate full-horizon qualification subsequently passed all 1,000 steps,
with 50 controller calls at steps 0,20,…,980. The final observation retained
SR length 1,000 at epoch 0, followed by the permitted capacity reset to
length 0 / epoch 1,000, matching the saved reference. Full process time fell
from 1357.555 s to 1004.197 s (26.03% in this one sample); the matched driver
loop fell from 1350.526 s to 997.675 s. Short separate GPU2 candidates overlapped
part of this run. This remains far slower than the historical Huan/Xiangru
paper-contract processes and is not a repeated-process ranking.
NAV's new wrapper imports Torch before the inherited payload timer, while the
old runner imported it inside that timer. Therefore its payload times
18.704 → 15.624 s are **not** a matched 16.5% optimization measurement.
The new wrapper-inclusive wall is 17.217 s, and its process wall is 18.064 s.
Use the matched internal loop separately from these startup boundaries.
The same early-import distinction applies to Unicycle: its payload wall is
12.941 s, wrapper wall 14.486 s and process wall 15.864 s. The matched driver
loop decreased by 26.45% in this single new process. NAV32's matched loop
decreased by 18.45% against the old reference, and by 7.47% against the new
512-row candidate. Neither measurement establishes that the current full
process is faster than every competitor.

Raw receipts and source snapshots are under `results/`. The original 289-run
ARCH-COMP26 index and report are not replaced by these new candidate records.

## Small-batch waste and next candidates

The frozen weighted map pads every CUDA graph call with fewer than 512 rows
to 512 rows. NAV robust has 25 input boxes; Unicycle, ACC, Attitude, Docking,
TORA sigmoid/tanh and Single Pendulum have one. The selected P3 entries enable
early weighted refinement with up to two rounds; old receipts do not record
that every possible round was executed.

`weighted_small_batch.py` retains the original map and refinement but supports
fixed 1/16/32-row graph scratch, with fresh-owner checks and owned output
clones. NAV's new option is `--weighted-chunk 32`.
`run_unicycle_candidate.py` selects one row and retains the paper model's
constant disturbance acting only on the speed derivative, all 500 steps,
strict injection, endpoint and saved geometry. These smaller shapes remain
numerical candidates until their own complete-output comparisons pass.
NAV32 and Unicycle1 have now passed those complete saved-output comparisons.
NAV32 used 600 refinements, 1,200 graph maps, 30,000 real rows and 8,400 padded
rows; Unicycle1 used 500 refinements, 1,000 maps, 1,000 real rows and no padding.
Both restored all three bindings. Hidden Taylor-model/SR arrays were not
compared, and the independent NNCS certificate remains false.

`run_small1_candidate.py` preserves the complete ACC and official `u=11f`
TORA sigmoid contracts. ACC's 50 range rows and 50 safety rows agree with the
saved sample, including the positive independent tube halfspace margin.
TORA reads the existing campaign's property receipt only after all its checker
inputs compare equal; it does not run the old checker or change the frozen
payload's `property_evaluated=false`. These new sources disable bytecode writes
before importing frozen code.

The two-slope QUAD prefix is not promoted to the default. At step 40, widths
of states x1–x6 are smaller in both geometries, x7–x11 are larger and x12 is
equal. Across all 40 steps x5 is always narrower, but x9 is wider at 37 tube
and 39 endpoint observations. A smaller width does not imply containment:
only 392 of 960 candidate projections are subsets of their reference. The
absolute CSV retains all states and both geometries, without adding widths
with different physical units. No two-slope full-horizon result is claimed.

The isolated `weighted_chunk256.py` is a direct byte-for-byte copy of the
previously saved 256-row adapter. Its source receipt is
`weighted_chunk256_SOURCE.json`; its old checker is not run. The new paper
QUAD 256-row candidate passed its 40-step saved-output comparison: 40 refinement
calls, 320 graph maps, 81,920 real rows and no padded rows. Ordinary validation
still uses 128 rows. No validation rounds are removed.

## Single-row fused refinement candidate

`weighted_fused_single.py` calls the unchanged interval map twice in one
CUDA graph. It commits only when the preconditions and both inclusions pass
and neither intersection is disjoint. All other cases use the original
refinement, including its diagnostics and exceptions; tracing, failed-lane
validation and recentered validation retain their original paths. Outputs
own their storage. `run_small1_candidate.py --weighted-mode fused` checks the
first successful live input against the original refinement's bytes and all
statistics; that one-time diagnostic is included in its reported times.

Six new CPU tests with real Torch 2.2.2 exercised successful rounds, first and
second round failures, invalid inputs, disjoint-bound exceptions, output
ownership, trace routing and binding restoration. Production is restricted
to the preloaded Torch 2.5.1 environment. ACC's new GPU candidate used this
path on all 50 steps, with one successful reference check, no fallbacks and
no candidate exceptions. It has no measured additional whole-process gain
over the simpler one-row candidate, so the default remains `small`.
The official TORA sigmoid fused candidate used the fast path on all 500
steps, with zero fallbacks/exceptions and one successful reference check.
Its full process took 9.392 s versus the saved old sample's 13.959 s; its
driver loop took 5.255 s versus 9.794 s. The process is below the historical
Huan/Xiangru campaign medians but above native's 8.943 s median. It overlaps
the separate GPU3 QUAD run and is a single sample, not a stable speed ranking.
Unicycle's fused candidate ran after QUAD finished and also passed all 500
fast calls, with no fallbacks/exceptions and one successful live comparison.
Its driver loop was 7.652 s and process wall 11.698 s; all 500 saved eight-state
tube/endpoint rows, control metrics and final fields remain equal. The
terminal target label remains true and the NNCS certificate remains false.

## Running the qualified choices

These wrappers require the saved Huan-server environment and preloaded CUDA
libraries. They do not compile extensions. Always use a fresh output directory
and `python -B`, and supply `--gate` pointing to the new private-output GPU
gate's `data` directory. Do not relaunch the completed run IDs in `results/`.

| Contract | Entry and choices | Physical GPU / CPUs |
|---|---|---|
| NAV robust | `run_nav_candidate.py --mode fast --weighted-chunk 32` | 1 / 6–9 |
| ACC | `run_small1_candidate.py --instance acc --weighted-mode small` | 2 / 10–13 |
| TORA sigmoid, official u=11f | `run_small1_candidate.py --instance tora-sigmoid --weighted-mode fused` | 2 / 10–13 |
| Unicycle, paper disturbance | `run_unicycle_candidate.py --weighted-mode fused` | 3 / 32–35 |
| Paper QUAD | `run_quad_candidate.py --mode full50 --weighted-chunk 256 --compare equivalent` | 3 / 14–17 |

Each also requires `--output`; QUAD's `--source-runner` selects an explicit
saved paper runner. Numerical refusal stops the candidate. The wrappers retain
the range/observation formats consumed by the existing Python plotting CLIs.
`python -B build_summary.py` only indexes saved new receipts into
`RUN_INDEX.json` and `RUN_INDEX.csv`; it executes no solver or old checker.

## Execution handoff

The first paper QUAD private256 candidate was launched once under
`quad_paper_p3_private256_20261005_001/run_001` (remote supervisor PID 701727,
240-second limit). Its SSH channel stopped responding before its result was
retrieved. The user restored authentication; the existing run was read back,
without relaunch, and had completed successfully at 13:39:36 UTC.
`LOCAL_OBSERVATION_STATUS.json` preserves the intervening transport observation;
it is not a numerical failure or timeout finding.

NAV32, Unicycle1, paper QUAD two-slope256, ACC1 and official TORA sigmoid1
completed under separate IDs. The unchanged-relaxation QUAD private256 also
completed its new full 1,000-step qualification. ACC and official TORA sigmoid
and Unicycle have separate fused-single candidates. Each used an idle-GPU preflight and
preserved the first-refusal stop; all timings and comparisons come from the
individual receipts rather than a restarted frozen experiment.

CPU verification: four private-binding tests, four Horner-binding tests,
four small-batch wrapper tests, two exact-rational bias/CLI tests and six
fused-single semantic/ownership tests passed.
New saved-data comparison fixtures also checked equality and deliberate
differences. No old numerical checker was run.
