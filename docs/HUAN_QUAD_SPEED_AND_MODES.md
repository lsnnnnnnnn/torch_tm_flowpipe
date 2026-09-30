# Huan QUAD speed and mode audit

Status: **evidence-backed draft; profiling/ablation experiments remain paused**
Date: 2026-09-30

## Plain-language answer

Huan's fast QUAD result is real for the configuration that was actually run:
the archived parity/box/native-f64 route completed all 1,024 boxes and 1,000
ODE steps in five fresh processes, with a median process wall time of
75.250099 s. The nested medians are 70.727803 s for the author's internal
control/plant loop, 72.173857 s around the wrapper's `driver.main()` call, and
75.250099 s for the watchdog child-process wall clock. The historical author
log reports 83.048290 s, but the exact
dirty engine commit for that July run is not present locally, and its hardware
and software environment differs from the reproduction. The two numbers are
therefore supporting reproductions, not a measured optimization ratio.

This is not a like-for-like speed comparison with the current PyTorch P3
result. Huan used working order 2 and validation level 1 in parity mode. The
current completed PyTorch route uses working order 3, point order 2,
validation order 4, and the repaired strict arithmetic/dataflow. Its single
completion run took 1,533.752052 s; 1,234.852041 s was in advance, while range
observation took 75.132491 s, the working-prune timed region took 192.952505 s,
and NN bounds took 3.533022 s. That working-prune region mixes diagnostic
hashing with required graph eviction, so 192.952505 s is only an upper bound
on removable diagnostic overhead. Dividing
1,533.752052 by 75.250099 gives about 20.38, but that ratio mixes algorithmic
order, numerical guarantee, implementation, instrumentation, and timing
boundaries. It is not an eligible same-contract speedup claim.

No single trick is currently proven to explain the gap. The evidence supports
several contributors: lower order/validation work in Huan parity, different
roundoff charges, different SR propagation, and substantially different
kernel/scheduling organization. Huan's batch-only SR chunk has
implementation-equivalence evidence over the covered qualification matrix and
saved strict metrics; there is no completed unpatched parity run for a full
run byte comparison. Preloading and cache release
were present in the successful runs, but no controlled ablation proves their
individual time benefit.

The existing result and timing evidence is frozen in
[`huan_quad_reproduction.md`](../research/gpu_verified_20260930/report/evidence/huan_quad_reproduction.md),
[`huan_parity_campaign.json`](../research/gpu_verified_20260930/report/evidence/huan_parity_campaign.json),
and [`quad_trig_full1000.md`](../research/gpu_verified_20260930/report/evidence/quad_trig_full1000.md).

## Identity and evidence boundary

| Item | Verified identity / boundary |
|---|---|
| Frozen clean Huan engine | `d5f0b68fcd36ba5f582733624f074728fe9720d8` |
| Actual successful QUAD variant | clean Huan engine plus the single batch-axis SR patch `f83f6d84f0624608a5bdf85e5af7ac4d394e679e` |
| Frozen Huan source location in the evidence workspace | `results/flowstar_acceleration_20260921/four_way_matched_v1/original_sources/huan/src/flowstar_gpu` |
| QUAD integration driver SHA-256 | `e4856f16f67d277ccf70e9fdcf9f62361ae7a410dbff1f1c56315c5f8e6bfe6b` |
| Controller | `quad_controller_3_64_torch.onnx`, SHA-256 `fabd84e411f4b0ebe0d6b996be6e4bd2adc48cf1118ab99c82180563b95532dd` |
| Historical author run | 83.048290 s log exists; exact dirty source commit/patch is unavailable locally |
| Current handoff source map | describes our P3 strict+trig engine, not Huan; do not use it as a Huan source manifest |

The repaired Huan polynomial-engine proof closure at `743f6205...` is
documented in
[`HUAN_STRICT_PROOF_CONTRACT_CLOSURE_20260826.md`](HUAN_STRICT_PROOF_CONTRACT_CLOSURE_20260826.md),
but that repaired source tree and its complete diff are not present in the
current workspace. The report does not extend its claim to CROWN, the
controller, ONNX, coupling, transcendental assumptions, or throughput.

## Mode and configuration map

The word `strict` controls only specific plant-arithmetic paths in the clean
Huan engine. It does not mean that every NNCS component is automatically
strict. Conversely, `parity` is a deliberate Flow*-compatibility/reproduction
mode, not evidence that every author result is invalid.

| Switch / path | Source location in frozen Huan evidence | Mathematical effect | Extra work / guarantee boundary | Observed evidence |
|---|---|---|---|---|
| Wrapper `parity` versus `strict` | `results/archcomp_failure_20260923/evidence/run_huan_quad_v3.py:4,33` | `parity` omits the low-level `--strict`; `strict` adds it. | Wrapper itself changes no arithmetic. | Archived parity argv has no `--strict`; strict argv has it. |
| Driver `--strict` | `results/archcomp_review_20260923/sources/huan/integrations/crown_reach/gpu_driver.py:721-727,868-875` | Sets `Settings.mode`; CLI default is parity even though the Settings class default is strict. | Actual argv, not the class default, determines the run. | Five completed runs used parity. |
| Sparse engine | `flowstar_gpu/sparse_exec.py:1369-1389` | QUAD uses the union-support sparse plant path, not dense `flowpipe.advance`. | Representation/performance choice; not a proof switch. | Successful and failed QUAD runs both used sparse. |
| Strict final composition contraction | `sparse_exec.py:944-960`; dense analogue `composition.py:231-251` | Adds a Rump-style dot-product error bound for the final point-coefficient contraction and ranges it on the actual support. | Extra absolute contraction, dot bound, support range, and interval addition. Clean `d5f0` still leaves other point products uncharged. | Strict is wider, but no one-factor ablation assigns the widening to this charge alone. |
| Strict VAR truncation tail in refinement replay | `ode_compiler.py:27-33,286-290,491-506,533-539,642-658,684-689`; preloaded path `tape_kernels.py:428-478,794-841` | Valid-pass VAR truncation tails are computed in both modes; strict adds them back at each replay while parity omits them for Flow* compatibility. | Strict adds a tail read and interval addition for each VAR leaf/refinement replay. | ACC has an independent missing-tail counterexample; that evidence is not automatically a QUAD-specific counterexample. |
| Clean strict gaps in symbolic Phi and retained coefficients | `sparse_exec.py:924-938,1406-1425`; `symbolic_remainder.py:219-235` | Clean strict still treats several rounded point coefficients/Phi products as exact. | No corresponding charge means no complete clean-source strict guarantee. | The 2x2 three-Phi exact-rational witness in the proof-closure report demonstrates under-enclosure. Our current P3 engine contains analogous interval-Phi and coefficient-error repairs. |
| `box` versus `hybrid` | integration driver `gpu_driver.py:744-774` | `box` interval-hulls the state before same-slope CROWN. `hybrid` also evaluates the TM-affine route and selects a certified narrower remainder per output. | Hybrid makes two CROWN calls. Independent of strict/parity. | Strict box and strict hybrid both first reject at step 597. |
| `same-slope` | integration driver and archived argv | Fixes the CROWN relaxation choice. | Independent of plant strict/parity. | Historical and reproduced fast runs use same-slope. |
| `native-f64` versus older RPC32 transfer | archived argv/result | Controls controller-bound transport precision. | Independent of box/hybrid and strict/parity. | Switching to native-f64 did not change the recorded first strict rejection step. |
| Three distinct order contracts | benchmark/run settings | Huan uses working P2/RHS validation order 1; our corrected P2 uses work2/point1/validation3; our completed P3 uses work3/point2/validation4. | P3/validation4 has more coefficients and validation work and is the currently completed repaired variant; this does not prove it is the only possible closing contract. | Our corrected P2 and P3 widths diverge strongly by their common step 799. Huan strict already rejects at 597, so step 799 is not a Huan/P3 common-prefix claim. |
| SR queue 1000 + batch-only chunk | `symbolic_remainder.py` patch archived as `huan_sr_chunk.patch` | Keeps the full history and original Q-axis summation, but evaluates independent B lanes in chunks of 128. | Reduces broadcast temporary memory without dropping history or terms. | Qualification covers B=1/128/129/257/1024 and queue lengths through 999 with bitwise-equal outputs/SR state; saved original/chunk strict metrics differ only in elapsed time. Five completed chunked parity runs agree with each other, but no complete unpatched parity comparator exists. |
| Kernel preload and unused-cache release | `run_huan_quad_v3.py:4,15-16,25-29` | Startup/allocator behavior only. | No reachability arithmetic change. | Present in the five runs; individual timing benefit is not isolated. |
| Shared outward interval primitives | clean Huan `interval.py`; used throughout both modes | Directed/outward interval add, subtract, multiply, divide/reciprocal, reductions, and point-by-interval operations are shared infrastructure. | `parity` does not turn these primitives off; the switch controls selected additional accounting around point-coefficient work and VAR replay. | Source classification only; it is not an ablation result. |
| Point-coefficient trust boundary | clean Huan `polynomial.py`, `sparse_exec.py`, `symbolic_remainder.py` | Both modes retain point-coefficient/Phi paths. Clean strict adds selected final-contraction and replay charges but still leaves named point/Phi roundoff gaps. | Therefore clean `--strict` is not an end-to-end certificate, while parity deliberately omits still more charges. | Exact-rational witness and proof-closure report identify the gap; no claim that every point operation explains step 597. |
| Reciprocal and remainder routes | clean Huan `interval.py:188-230`, compiled replay/VAR paths | Base interval reciprocal/division is shared; the strict-dependent difference in the qualified replay is the addition of saved VAR truncation tails, not a wholly different reciprocal algorithm. | Reciprocal/domain handling and strict remainder replay must be classified separately. | Five-route reciprocal qualification belongs to the repaired current engine; it must not be retroactively attributed to the clean Huan run. |
| Endpoint and controller injection | Huan driver `_inject_core[_s]`, `inject_controls[_s]`; endpoint publication paths | Same-slope CROWN injection and endpoint publication run in both modes; their point-weighted coefficients/remainders inherit the surrounding arithmetic limitations. | `box`, `hybrid`, transfer dtype, endpoint handling, and strict/parity are distinct axes. | The saved parity and strict runs share same-slope/native-f64 but diverge in plant-mode accounting. |
| Current P3 K20 interval-SR route (not clean Huan) | frozen current `symbolic_remainder.py:300-350`, `sr_kernels.py`, `sr_sum_kernels.py`, and `quad_sr_reassociate_20260928/host_ledger.py:39-128` | Current P3 propagates interval-valued SR/Phi accounting. Its CUDA matrix update avoids the `Q*B*n^3` broadcast, its history image uses a `Q*B*n*2` workspace, and K20 reconstruction chunks only independent lanes while retaining the full original Phi/J queues and Q reduction. | This already applies the portable principle at the strict representation's actual cost centers. Huan's point-SR patch is not a drop-in strict-P3 optimization. | Frozen RESULT records both CUDA extensions, K20, and lane chunk 16; no extra Huan-style patch is justified without a profile showing a remaining SR bottleneck. |

`box`, `hybrid`, `same-slope`, transfer dtype, plant mode, and P2/P3 are
separate axes. In particular, box does not mean strict, and hybrid does not
mean parity.

### B-axis SR portability decision

The archived Huan patch is present in the evidence workspace as
`results/archcomp_failure_20260923/evidence/source_evidence/huan_sr_chunk.patch`
(SHA-256 `065f577a26e137f86f4986086e2aec1093ee71ea165a6e8d1c0ac28a8b719ef0`).
It changes only the non-strict point-Phi history image: `dot_point_iv` is run
on independent blocks of 128 lanes, the complete `[B,Q,n,2]` terms tensor is
still retained, and the original Q-axis reduction is unchanged. At
`B=1024,Q=999,n=16,float64`, that terms tensor is about 250 MiB; the patch's
purpose is to avoid the much larger broadcast temporary inside the dot.

That exact operation is not the completed P3 strict path. The frozen P3 source
uses a directed-rounding interval-matrix kernel for Phi history updates and a
two-stage directed history-sum kernel whose only large scratch has shape
`[Q,B,n,2]`. The K20 host-factor adapter separately reconstructs eligible
lanes in blocks of 16 while preserving the full-B Phi/J storage and original
Q reduction. The completed RESULT binds `sr_kernels.py` to
`69bb07257ad8cba029a611f9099e015c5175bbba72a0f86bf8f54549dc29dfc2`,
`sr_sum_kernels.py` to
`de76dc539c109065779292751a5650f330b8dddbade8248d1bbd23050737bd3b`,
the host adapter to
`1a7d9bbf0f16a4f3726b75a27ff938ff87b918141a5f0fe1504f26198dc4035b`,
and records both loaded extension paths, K20, and lane chunk 16.

Therefore the Huan patch is classified as an already-consumed design lesson,
not as an unimplemented P3 code change. A new SR implementation is warranted
only if a matched profile of the completed strict route shows SR is still a
material bottleneck. The present full-run profile instead places the dominant
cost in advance/validation, so the next bounded optimization study should
start there.

## Completed and failed results

| Route | Full target | First rejection | Accepted lane-steps | Timing boundary | Result |
|---|---:|---:|---:|---|---|
| Huan parity, box, native-f64, SR chunk | yes | none | 1,024,000 | five fresh processes, median | internal 70.727803 s; driver-call 72.173857 s; process 75.250099 s |
| Huan strict, box, native-f64, SR chunk | no | 597 | 611,954 | failed diagnostic only | 33.765 s process; not a full-runtime sample |
| Huan strict, hybrid, native-f64, SR chunk | no | 597 | 611,930 | failed diagnostic only | 36.596 s process; not a full-runtime sample |
| Current PyTorch P3 strict+trig | yes | none | 1,024,000 | one completion run | 1,533.752052 s watchdog |
| Native matched initial-affine-cover variant (otherwise original-library P2/f64) | no | no rejection in complete prefix | 614,400 (=600×1,024) | natural six-hour timeout | 21,609.612654 s watchdog; no T=5 width; fullbatch/end-to-end qualification flags remain false |

At Huan strict step 597, 79 lanes reject with no bad-domain count. For the
first failing lane, the validation ranges for x5/x6 exceed the ±0.1 cap, and
the maximum saved truncation-tail magnitude is about 0.371585. This proves
widening and rejection, not which single strict charge caused it. The saved
control-period metrics likewise show large strict/parity width separation by
period 29, but are observational rather than a causal ablation.

## Width results and comparability

Width must remain attached to the mode and saved-object semantics. The
completed Huan parity/box/native-f64/SR-chunk run saved the T=5 endpoint union
hull over all 1,024 partitions. `quad_baselines.json` records these x1--x12
widths (the `huan_parity_1000_union_endpoint` field):

| State | Huan parity T=5 endpoint union | Current P3 T=5 endpoint union |
|---|---:|---:|
| x1 | 6.499709442860031 | 6.623621351875684 |
| x2 | 7.053018715211994 | 7.25355368193984 |
| x3 | 0.047687232284709835 | 0.06897238207275103 |
| x4 | 1.422631744073749 | 1.5039562607279868 |
| x5 | 1.6005968003519206 | 1.722072457491394 |
| x6 | 0.12335095690703003 | 0.17992709753985986 |
| x7 | 0.007016562530468149 | 0.01085313974198335 |
| x8 | 0.006119904294179676 | 0.009710827817861502 |
| x9 | 0.004449406794516559 | 0.006143487644367451 |
| x10 | 0.08617957686540537 | 0.15565646011563147 |
| x11 | 0.0710206705569302 | 0.13084117073828946 |
| x12 | 0.0 | 4.450147717014403e-308 |

The Huan values are exact differences of the saved binary64 `final_hull`
endpoints, not per-partition means and not full-step tubes. The P3 column is
the same named T=5/requested-box endpoint-union statistic. Working, point, and
validation orders, remainder accounting, and boundary contracts still differ;
a narrower number does not mean a stricter or more correct method, so no
cross-contract width multiplier is reported. Huan strict has no T=5 width.

Huan strict actually accepts every lane through step 596 and first rejects at
597. Step 580 (t=2.9) is not that maximum accepted prefix: it is the latest
20-step control boundary inside the common accepted prefix for which the
three archived sources save comparable width summaries. Huan has only
per-partition endpoint-width mean/max at that boundary, with no endpoint
union, tube, or 580-step timing. Representative mean/max rows are:

| State | Huan strict | Corrected P2 | Current P3 |
|---|---:|---:|---:|
| x3 | 0.8007371043250104 / 0.8520737812769105 | 0.19008228006870231 / 0.19235476390073836 | 0.09226700861577442 / 0.09313853408665218 |
| x6 | 2.6279395310657967 / 2.860352139116201 | 0.4687640913995893 / 0.47589134497654717 | 0.21567174738190245 / 0.2179400085874669 |
| x7 | 0.40238270310312396 / 0.4324206410567387 | 0.06399811831903117 / 0.06454045606942328 | 0.018724504500417977 / 0.0189341926509305 |

The complete twelve-state table is
`research/gpu_verified_20260930/report/data/p3_common580_comparison_20260929_WIDTHS.csv`.
Missing statistics are not inferred, and the failed strict process time is not
reported as a step-580 runtime.

The three parity timing boundaries also must not be collapsed. The internal
timer starts after state/metrics construction and ends after the control/plant
loop, before final hull/verdict serialization. The wrapper timer surrounds
`driver.main()` but excludes module import, engine/config checks, box creation,
and post-return hashes/result writing. The watchdog timer starts before
`Popen` and ends after child exit, so it includes Python startup/import,
wrapper work, and polling tail but excludes writing its own process receipt.
None is a GPU-kernel-only timer.

## What the available timing does and does not explain

The current PyTorch full-run breakdown is concrete:

| Component | Seconds | Interpretation |
|---|---:|---|
| Advance | 1,234.852041 | Dominant solver work; about 80.6% of the 1,531.395193 s internal process time |
| Observer | 75.132491 | Saved tube/endpoint bounds; already comparable in magnitude to an entire Huan parity process |
| Working-prune timed region | 192.952505 | About 12.6% of internal process time and larger than observer cost; includes diagnostic hashing plus required graph eviction, so it is an upper bound rather than a measured removable overhead |
| NN bounds | 3.533022 | Not the dominant gap |
| Boundary | 8.830551 | Strict endpoint/control boundary work |
| Checkpoint | 0.312904 | Not the dominant gap |
| Other internal work | remaining recorded/internal work only | Timing components are not assumed to be exhaustive or additive; the outer watchdog/process difference is separate wrapper overhead |

The paired 40-step profile further located about 97.38% of later-step advance
time in weighted plus ordinary validation for the pre-trig baseline. Direct
sin/cos reuse then reduced the full-run advance from 2,868.586754 s to
1,234.852041 s. All 1,000 observer and 101 controller/transfer/terminal PT
files were byte-identical, and the two final plant/SR snapshots audited equal.
That explains a major
within-P3 improvement, not the remaining P3-versus-Huan gap.

Huan's archived metrics do not expose a matching full-run phase breakdown.
Until a bounded matched profile exists, the gap cannot be apportioned among
order, validation, SR, Python/kernel scheduling, observer cost, preload, or
allocator behavior.

## Portable optimization ledger

| Candidate | Correctness condition | Evidence now | Current disposition |
|---|---|---|---|
| Independent B-axis SR chunking | Preserve full terms/history, Q summation order, interval-Phi enclosure, and per-lane state | Huan point-Phi qualification is bitwise equal over the covered matrix; current strict P3 already uses no-cubic-broadcast interval kernels plus lane-chunked K20 reconstruction, with exact source/extension identities in the completed RESULT | Design lesson already consumed; do not transplant the point-Phi patch. Reopen only if a matched strict-P3 profile identifies a remaining SR bottleneck |
| Direct trig reuse within one immutable validation plan | Reused entry must have identical operation, source slot, order, tables, and owner generation | 1,000 observer and 101 controller/transfer/terminal PT files are byte-identical; final plant/SR snapshots audit equal; about 2.056x wall improvement over prior P3 | Implemented and retained |
| Kernel preload | Same kernels/build identity; no fallback hidden by preload | Availability present, benefit not isolated | Profile/ablate later; no speed claim now |
| Cache release policy | Never release live tensors; numerical state signatures unchanged | Huan runner asserts allocated bytes unchanged | Low-priority allocator experiment after matched profile |
| Reduce observer/export overhead | Exported geometry and acceptance/status must remain identical; solver state immutable | Observer is 75.13 s in current P3; new plotting path works entirely from saved output | Future accepted-step streaming can be profiled independently; plotting must remain outside solver timing |
| Lower order/validation | Must be declared as a different algorithm contract and still close the full benchmark | Huan parity P2/RHS1 completes; Huan strict first rejects at 597; our corrected strict P2/validation3 first rejects at 800; current P3/validation4 completes | Not an implementation-equivalent optimization and cannot be presented as one |
| Omit strict tails/roundoff charges | None: would change the guarantee contract | Parity is a faithful compatibility mode, while strict evidence shows real missing-roundoff counterexamples | Never use as a hidden acceleration |

## Minimum future ablation plan

Experiments remain paused. When the user explicitly resumes them, the first
action is a read-only recheck of the native matched initial-cover QUAD job's
terminal state; no old job is restarted. After that gate, the smallest interpretable Huan/P3
study is a pair of fresh-process 40-step diagnostic arms, not another full
1,000-step launch and not evidence that may be extrapolated to a full run:

1. Run a qualification/profile arm on the same saved-input stage with the
   existing detailed instrumentation, matching timing boundaries, and the
   existing full output/plant/SR/host-state equivalence checks.
2. Run a clean performance arm with reference hashing, comparison, and log I/O
   outside the timed solver window, while retaining real pruning and graph
   eviction. Compare it only with the matched qualification arm.
3. Separate algorithm variants (P2/P3 and validation level) from
   implementation-equivalent changes.
4. For Huan strict diagnostics, toggle one existing charge at a time only in a
   labelled non-certifying ablation and compare the common accepted prefix;
   never report an incomplete failure time as a full-run speed.
5. Do not port the old point-Phi B-axis patch into strict P3. First measure the
   frozen interval-SR kernels on the same saved-input stage; only a material SR
   bottleneck may justify a new representation-correct candidate, followed by
   real-input output/full-state equivalence gates.
6. Measure preload/cache behavior in fresh processes with frozen source,
   extension, model, box, and hardware identity.

## Unresolved items

- The exact dirty source/patch for the historical 83.048290 s run is absent.
- The repaired Huan `743f6205...` source tree/diff and its D6 machine ledger
  are absent; only the report/verifier evidence is local.
- No matched phase breakdown yet exists for Huan's complete parity run.
- No A/B evidence assigns the strict-specific step-597 widening between the
  final-composition roundoff charge and VAR-tail replay. Shared SR/controller
  paths may subsequently amplify the divergence, but are not independent
  strict-mode switches in clean `d5f0`.
- Huan strict has no T=5 runtime or final width.
- The current PyTorch P3 timing has one full completion sample, not five
  isolated formal timing repeats.
- The native matched initial-affine-cover variant has no complete original-1024
  T=5 time or width; its fullbatch and end-to-end strict qualification flags
  are false.

These gaps are retained as unknowns; no missing source or result is inferred.
