# Native Flow* eight-direction production gate — 2026-10-02

**Decision: fail closed for native Flow* octagon production and for treating the
frozen QUAD run's accepted status as an independent enclosure certificate.**
The original library has a finite-sample under-enclosure in an ordinary
plant-only oscillator and in a new, isolated, actual paper-equation QUAD
symbolic-remainder **one-box/one-small-step** test. The latter witness uses a
constant control inside the CROWN affine-plus-residual relaxation constructed
by the frozen C++ program; it has **not** been shown to be the neural network's
actual output. It therefore diagnoses containment of the relaxed input set,
not a real closed-loop counterexample or an error claim about every saved
range of the original long run. No old long experiment was restarted; frozen
QUAD source and library were left untouched. One-line changes in separate
copied libraries removed these finite-sample counterexamples by widening
enclosures substantially. Neither change is a production repair or proof.

## Read-only source trace and minimal interface

The frozen native QUAD source is
`docs/evidence/results/archcomp26_20261001/native_quad_paper_full50_001/build/archcomp/Quadrotor/quad_paper_full50.cpp`.
Each lane calls `author_matched::reach` in a four-thread pool; only after the
pool joins does `arch_ranges::record(results)` write axis intervals. The server
build's `matched_reach.h` fixed-step path and Flow* `Continuous.h` template
`ODE::reach_symbolic_remainder` add a `Flowpipe` to `result.flowpipes` only after
`advance` returns 1. These appended items form the accepted numerical prefix;
the final `result.status` still determines whether the requested period
completed. The existing `ranges.bin` has no accepted/status field.

`Flowpipe` represents a composition `tmvPre ∘ tmv`, not merely `tmvPre`.
Flow* already provides `Flowpipe::compose(projected, {x,y}, order, cutoff)`
and `TaylorModelVec<Real>::rho(Interval&, direction, domain)`. The smallest
correctly located observer would process each unseen accepted item before
`arch_ranges::record` discards correlation, use the full local domain for a
tube, and save lower/upper intervals for `x`, `y`, `x+y`, and `x-y` with lane,
step, local duration, projection and run status. `Interval::inf()` and
`sup()` convert MPFR values to binary64 with downward/upward rounding.
Four forms give eight half-plane inequalities; normalized diagonals in the
existing plotter describe the same geometry up to scaling.

An endpoint needs separate semantics. Directly fixing the accepted
`Flowpipe`'s local time to `h` and then composing gave a **different** range
from the object used for propagation. The probe reproduced Flow*'s own
`ODE::reach` construction: copy the accepted flowpipe, evaluate `tmvPre` at
the local endpoint with its `Real` power table, then compose and range that
object. For the last step of every short reach, the resulting four direction
intervals were exactly equal to `result.fp_end_of_time`'s intervals. This is
the proposed *propagated endpoint* meaning; it failed the analytic gate below.

The existing `Plot_Setting::plot_2D_octagon_MATLAB` is useful as a reference
but requires `TaylorModelFlowpipes`. The readily available
`Result_of_Reachability::transformToTaylorModels` composes **all** saved
flowpipes, then clears `result.flowpipes`; calling it on a live QUAD result
would interfere with continuation and may materialize 1,024,000 projected
segments. The saved QUAD `ranges.bin` contains only per-coordinate bounds;
it cannot be post-processed into genuine diagonal supports. Its
`arch_ranges::record` calls `tmvPre.intEval` directly, so agreement with a
fully composed projection must be checked separately before treating those
old axis intervals as equivalent to this new observer. A three-step
oscillator check is insufficient to establish that for QUAD.

## Isolated short tests and the failing numerical gate

[Probe source](../tools/native_flowstar_octagon_probe_nohash.cpp) built
against the frozen native Flow* static library without changing it. The
plant-only model is `x1'=x2`, `x2'=-x1`, initial box
`x1∈[1,1.2], x2=0`, fixed `h=0.05`, order 4, cutoff `1e-8`, remainder
estimate `[-0.1,0.1]`, total `T=0.15`. The server used `/usr/bin/g++-15`
with the original library/includes; no controller, RPC service or NNCS
property was involved. The exact sample is
`(x1,x2)=(a cos(t), -a sin(t))`, `a∈[1,1.2]`.

| New isolated run | Reach calls | Accepted steps | Direction rows | Exact-sample checks | Violations |
|---|---:|---:|---:|---:|---:|
| [periodic](evidence/results/archcomp26_20261001/native_octagon_harmonic_probe_20261002_001/AUDIT.json) | 3 × 0.05 s | 3/3 | 24 | 1,512 | 28: tube 6, propagated endpoint 22 |
| [single](evidence/results/archcomp26_20261001/native_octagon_harmonic_probe_20261002_002/AUDIT.json) | 1 × 0.15 s | 3/3 | 24 | 1,512 | 28: tube 6, propagated endpoint 22 |

The first failure is already at the first endpoint: the native propagated
`x1` lower bound is `0.99875028645838326` in the periodic run, while the
allowed exact sample `a=1,t=0.05` gives `0.99875026039496628`. At that
step, `fp_end_of_time.tmvPre.intEval`, composed `intEval`, and directional
`rho(e_x)` share the same lower bound. Therefore this specific discrepancy
is present in the propagated Flow* endpoint object before the CSV writer;
it is not introduced by the observer's support calculation. The step-2 tube
also excludes the allowed `a=1,t=0.05` sample on `x1-x2`:
saved lower bound `1.048729453124051` versus exact
`1.0487294296656446`. A single 0.15 s reach has the same failing counts,
so the counterexample is not caused solely by restarting at a control-period
boundary. The analyzer uses ordinary binary64 `math.cos`/`math.sin`; the
first discrepancy is around `2.6e-8`, far above a last-bit rounding issue.

Both probes separately confirmed that the observer left an otherwise
identical no-observer run's step count and all 24 directional intervals
unchanged. Their **tube** directional intervals exactly matched those from
the library's `transformToTaylorModels` representation used by the original
`plot_2D_octagon_MATLAB`; each generated a three-segment
`native_reference.m`. These are implementation cross-checks, not independent
proof of trajectory containment. The observer took about 0.0008 s for this
three-step/two-state example, with no production throughput claim.

Raw [periodic CSV](evidence/results/archcomp26_20261001/native_octagon_harmonic_probe_20261002_001/directional.csv),
[single CSV](evidence/results/archcomp26_20261001/native_octagon_harmonic_probe_20261002_002/directional.csv),
`stdout.log`, `stderr.log`, original MATLAB scripts and the
[finite-sample checker](../tools/check_native_flowstar_octagon_probe_nohash.py)
are saved. Both `AUDIT.json` files retain their failed status and first
counterexamples. The checker intentionally exits nonzero on this evidence.

## Source-level cause and isolated repair probe

The harmonic probe called the ordinary `ODE::reach` overload, which uses the
**non-symbolic** `Flowpipe::advance` at the frozen server
`native_quad_paper_build_001/flowstar/flowstar-toolbox/Continuous.cpp:857–1044`.
Its first accepted `x1` Taylor polynomial contains the center `1.1`, initial
symbol `0.1 r1`, terms `-0.55 t²`, `-0.05 r1 t²`, and center-only
`0.045833333333333337 t⁴`. The `r1 t⁴` term has total degree five and is
truncated at order four; its omitted magnitude at `r1=±1,t=0.05` is about
`2.60417×10⁻⁸`. The frozen first-step [raw TM dump](evidence/results/archcomp26_20261001/native_octagon_harmonic_rootcause_20261002_003/stdout.log)
shows an eventual `x1` remainder of only about `±2.17×10⁻²³`.

The first validated Picard pass does see this truncation: at
`expression.h:1629`, the `x2` variable leaf is truncated to order three,
producing a `±2.08333×10⁻⁶` RHS remainder from its `r1 t³` term. After
integration over `h=0.05`, this fixed contribution alone can cover about
`±1.04167×10⁻⁷`. The source then replaces the initially accepted Picard
remainder in the refinement loop at `Continuous.cpp:1008–1038`.
`Picard_ctrunc_normal_remainder` calls the variable-leaf-only remainder path
at `expression.h:1976`, which returns the TM's existing remainder and does
not carry forward the freshly truncated polynomial term. Successive
refinement shrinks the accepted `x1` remainder to near zero. The missing
term and the violated exact sample match in sign and scale. This identifies
the concrete mechanism for this ordinary-path short example; other modes
still need their own checks.

The isolated [patch script](../tools/patch_native_flowstar_refinement_probe_nohash.py)
copied `Continuous.cpp` and `libflowstar.a` to a **new** server directory,
changed only the ordinary `Expression<Real>` advance loop's
`bool bfinished = false` to `true`, rebuilt `Continuous.o`, and replaced
that object only in the copied archive. The exact [unified diff](evidence/results/archcomp26_20261001/native_octagon_harmonic_rootcause_fix_20261002_004/Continuous.patch)
is saved. This diagnostic leaves the first validated Picard remainder in
place and skips subsequent refinement; it is not a production-quality fix
or a speed improvement. The [patched first-step dump](evidence/results/archcomp26_20261001/native_octagon_harmonic_rootcause_fix_20261002_004/first_step.stdout.log)
shows `x1` remainder `±0.0050001041666666683`, versus the frozen
`±2.17×10⁻²³`. The patched first endpoint `x1` range becomes roughly
`[0.9937501823,1.2035003906]`, wider than the original
`[0.9987502865,1.1985002865]`.

Using the **same** three-step periodic probe and independent sample checker
against this copied archive gave [3/3 accepted, 24 directional rows, and
0/1,512 sample violations](evidence/results/archcomp26_20261001/native_octagon_harmonic_rootcause_fix_20261002_004/AUDIT.json).
Its no-observer control and original Flow* plot-TM cross-check also passed.
A finite sampled result cannot certify all oscillator trajectories, and this
one-line workaround does not test the QUAD symbolic-remainder branch at
`Continuous.cpp:2123–2418`. That branch has a similarly shaped refinement
loop around line 2382. The following separate QUAD gate tests that branch.

## Actual paper QUAD symbolic-remainder short gate

The [isolated source diff](evidence/results/archcomp26_20261001/native_quad_sr_octagon_gate_20261002_005/quad_gate.patch)
starts from the frozen `quad_paper_full50.cpp`, retains the 2026 paper ODE,
`h=0.005`, order 2, cutoff and controller injection, and limits execution to
the **first of 1,024 boxes, one CROWN call, and one 0.005 s ODE step**. It uses
the same `author_matched::reach(..., Symbolic_Remainder&)` path with
`AUTHOR_FIXED_STEPS=1`. The source, observer header, copied original archive,
build and outputs live in new server directory
`.../archcomp26_20261001/native_quad_sr_octagon_gate_20261002_005`;
the small [receipt bundle](evidence/results/archcomp26_20261001/native_quad_sr_octagon_gate_20261002_005/README.md)
is mirrored here. The observer was enabled and disabled in separate native
processes under one fresh CROWN server, with an explicit 120 s per-process
limit. Both returned status `2 = COMPLETED_SAFE` and one accepted step. Their
12 terminal axes, `ranges.bin`, and captured RPC input/response were exactly
equal, and the server recorded two one-box calls. The printed target
`UNKNOWN` is expected at `T=0.005` for the full `T=5` property and does not
change the numerical acceptance statement.

The frozen C++ builds each control state as
`u_j = Σ_i float32(T_ji) x_i + (float32(u_max,j)+float32(u_min,j))/2 + r_j`,
where `r_j` is in the interval with half-width
`(float32(u_max,j)-float32(u_min,j))/2`. Thus RPC `u_min` and `u_max` are
**affine intercept bounds**, not direct bounds on the full `u_j` after the
`T·x` term. The [raw RPC receipt](evidence/results/archcomp26_20261001/native_quad_sr_octagon_gate_20261002_005/on/rpc.json)
and [checker](../tools/check_native_quad_sr_octagon_gate_nohash.py) use that
exact float32 transport. One concrete relaxed-input witness has
`x1…x3=-0.39999999999999997`, `x4…x6=-0.4`, `x7…x12=0` and the constant
control `u=(12.730082304606912, -0.001486676491913386,
-0.006470973906107247)`, obtained by choosing all three affine remainder
lower ends. It belongs to the set the C++ code injected. Whether this exact
control equals the NN output at that point was **not** established.

An independent Python transcription of the **paper ODE**, integrated at
`t=(0,0.0025,0.005)` by SciPy DOP853 (`rtol=1e-13`, `atol=1e-15`,
`max_step=0.00125`), gives at `t=0.005`:

| State | Relaxed-input sample | Frozen Flow* propagated endpoint | Exceedance |
|---|---:|---:|---:|
| `x7` | `-3.441380768320594e-7` | `[1.6384537157764776e-6, 1.6545006846119772e-6]` | `1.982591792608537e-6` |
| `x8` | `-1.4979106264136853e-6` | `[3.9749375661109013e-7, 4.0249593908225256e-7]` | `1.8954043830247753e-6` |

Radau, with `rtol=1e-12`, `atol=1e-14` and the same maximum step, differed
from DOP853 by at most `2.054e-15` over all twelve states of this witness.
Across 64 initial-state corners × 8 residual corners plus a midpoint, the
[audit](evidence/results/archcomp26_20261001/native_quad_sr_octagon_gate_20261002_005/AUDIT.json)
made 20,520 tube, endpoint and terminal-axis sample checks. Exactly 2,048
terminal `x7`/`x8` checks exceeded `1e-11`; the largest exceedance was
`1.9825917926144243e-6`. **No sampled `x1`/`x2` eight-direction check
failed.** Both `tmvPre.intEval` and composed endpoint axes missed the witness;
the observer did not introduce this discrepancy. The audit exits nonzero and
preserves the failure. Finite integration plus a second solver strongly
supports this local numerical counterexample, but is not an all-trajectory
proof or a validated floating-point NN certificate.

The symbolic-remainder branch's validated first Picard pass is followed by
the refinement loop at frozen `Continuous.cpp:2382–2410`; its remainder-only
evaluation traverses the same `expression.h:1976` variable-leaf path that
omits newly truncated polynomial contributions in the ordinary example.
For a mechanism check, a **different copied archive** changed only that SR
loop's `bool bfinished = false` to `true`; the exact
[patch](evidence/results/archcomp26_20261001/native_quad_sr_refinement_fix_20261002_006/Continuous.patch)
is saved. With the identical one-box/one-step QUAD source and checker, both
observer modes still accepted, had equal saved outputs, and gave
[0/20,520 finite-sample violations](evidence/results/archcomp26_20261001/native_quad_sr_refinement_fix_20261002_006/AUDIT.json).
The first endpoint `x7` interval widened from roughly
`[1.638e-6,1.655e-6]` to `[-5.576e-4,5.609e-4]`; this deliberately broad
diagnostic skip is **not** a production fix, performance result, or soundness
proof. The original frozen library's failed receipt remains the relevant
production gate.

## Isolated two-site VAR-tail repair diagnostic

The previously saved historical VAR-tail patch at `/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/expression.patch` identified a more targeted candidate than skipping refinement. In a **new copied library** it keeps both ordinary and symbolic-remainder refinement loops, saves each variable leaf's fixed polynomial truncation tail during the full Picard pass, and adds that tail to the changing input remainder on each replay. Its earlier local witness and four one-step ACC checks were not QUAD qualification. The current frozen source was checked for the exact two active VAR sites and unchanged refinement loops before applying the patch to the isolated copy; the original library and full50 run were not modified. The [new receipt bundle](evidence/results/archcomp26_20261001/native_quad_var_tail_repair_gate_20261002_007/README.md) contains the two-site patch, build record, raw short runs, independent checks, and width comparison.

With the rebuilt copy, both three-step harmonic variants accepted all three steps, observer on/off saved equal numerical outputs, and each original finite analytic regression had **0/1,512** violations. On the paper-equation QUAD first box, one original CROWN call per arm and one `h=0.005` symbolic-remainder step, observer off/on again had equal status, RPC, twelve terminal axes and saved range records. The same 513 relaxed-control numerical samples gave **0/20,520** direction/axis exceedances, including the previously excluded `x7/x8` witness. The repaired `x7` pre-composition terminal interval was `[-2.382516064701138e-6,5.675470465089593e-6]`, versus the original `[1.6384537157764776e-6,1.6545006846119772e-6]` and skip-refinement `[-5.575585333117811e-4,5.608514877121697e-4]`. The [saved comparison](evidence/results/archcomp26_20261001/native_quad_var_tail_repair_gate_20261002_007/quad/WIDTH_COMPARISON.json) shows all 32 compared intervals inside the skip-refinement intervals and 30 strictly narrower; repaired `x7/x8` terminal widths are about 139 times narrower than the skip variant.

This repairs the observed finite-sample omission and retains narrowing in the isolated first step. These finite checks alone do **not** prove enclosure for every initial state, every control admitted by the affine relaxation, or every time inside the step; they also do not validate the controller bounder or a 50-period closed loop. The original-library failure remains the production gate evidence, and native octagon promotion stays closed. No new QUAD benchmark cell or long-task timing is claimed from this method diagnostic.

## Independent whole-set short-step interval check

A separate [read-only interval gate](evidence/results/archcomp26_20261001/native_quad_independent_interval_gate_20261002_001/README.md) covers the **saved first-call RPC/recenter TM input domain**, an interval hull of all controls admitted by its saved float32 CROWN affine-plus-residual injection, and the entire first `0.005 s` step. The saved RPC lower bounds for `x1`–`x3` are `-0.39999999999999997`, one binary64 ULP above the C++ source's first physical box lower bound `-0.4`, as tracked in the separate [initial-box audit](evidence/results/archcomp26_20261001/native_quad_initial_recenter_gate_20261003_001/README.md). Thus this gate does **not** cover the complete source-defined first physical box. Its 1,000 outward-rounded Decimal substeps each pass a strict Picard self-inclusion check; sine/cosine use Taylor enclosures with explicit remainder bounds. This is a mathematical plant ODE check over the recorded RPC domain and relaxed control hull, rather than a finite trajectory sample, but it does not independently validate the neural network or CROWN calculation. It does not start Flow* or modify the old run.

The checker compares four `x1/x2` direction forms for the whole-step tube and propagated endpoint, plus 12 terminal axes in both pre and composed representations. Of 32 comparisons over the saved RPC domain, the original frozen archive contains **23** independent intervals and the isolated VAR-tail repair contains **27**. For the repair, all eight `x1/x2` direction comparisons and both representations of `x7/x8` terminal axes contain the independent RPC-domain bounds. Five remaining repair comparisons are **inconclusive**: `x6` composed loses initial-state/control correlation in the independent box, while `x10/x11` pre and composed differ at approximately `2–3×10⁻¹⁵`. Noncontainment of a broad independent interval is not a reachable-state counterexample. The raw [original](evidence/results/archcomp26_20261001/native_quad_independent_interval_gate_20261002_001/original_AUDIT.json) and [repaired](evidence/results/archcomp26_20261001/native_quad_independent_interval_gate_20261002_001/repaired_AUDIT.json) comparison records retain each result. The excluded first-box strip, 1,023 other initial boxes, later control periods, CROWN's own correctness, and independent NNCS proof remain outside this gate; native octagon production stays closed.

## Read-only algebraic resolution of the five interval inconclusives

A subsequent [algebraic receipt](evidence/results/archcomp26_20261001/native_quad_algebraic_gate_20261003_001/README.md) stays within the same saved first-call RPC/recenter domain and uses only its RPC and observer CSV, the frozen paper ODE, and the prior directed-rounding Picard audit. It does not run Flow*, CROWN, a numerical ODE solver, or the full50 job. The five new algebraic comparisons use exact rational arithmetic; the other 27 retain the earlier directed-Decimal Picard derivation. The script rechecks the 27 prior numerical inclusions, then exploits `x12(0)=x12'=0` and constant first-step control: `x10(h)=h u2/0.054`, `x11(h)=h u3/0.054`. These physical-state bounds lie inside the two saved **composed** x10/x11 intervals and numerically inside the two **pre** columns, even without enlarging their binary64 bounds. The pre-column comparisons do not prove inclusion of the pre symbolic set. The independent 1,000-substep hull had accumulated approximately `2–3×10⁻¹⁵` of dependency width there. The narrowest new numerical margin, about `3.30×10⁻¹⁹`, uses the paper ODE's exact decimal `h` and `0.054`; substituting their nearest binary64 values still yields column inclusion as a sensitivity check, but does not validate Flow*'s parser or internal arithmetic.

For the fifth comparison, the script keeps the shared initial-state symbols in `x6(0)−h u1/1.4` and proves strict whole-step bounds `|x4|,|x5|,|x6|<1`, `|x7|,|x8|<0.01`, `|x10|<0.0015`, `|x11|<0.001` by a first-exit argument. The remaining x6 integral has magnitude at most `1.7405×10⁻⁵`. The resulting correlated endpoint interval `[-0.445722661279,0.349584029959]` is inside the copied repair's **composed** `x6` interval `[-0.446199541413,0.349816005255]`, also without a serialization ULP allowance.

The repaired candidate thus has **20/20 numerical inclusions for the composed physical-state view** on the saved first-call RPC/recenter domain and one step under the exact-decimal paper ODE: eight x1/x2 tube/endpoint directions plus twelve terminal axes. It does **not** establish enclosure for the omitted one-ULP strip of the first physical box. Twelve physical-state bounds also lie numerically inside twelve pre-composition axis columns, but this is **not** an independent pre-set inclusion result. A Flowpipe state is `tmvPre ∘ tmv`; `tmvPre.intEval` alone is not the general physical-state observer. The earlier `27/32` audit remains unchanged as the result of its particular plain-box propagation method; the later algebraic argument resolves its three undecided composed physical comparisons and two pre-column numerical comparisons for this RPC-domain gate only. The original frozen-library counterexample remains, and production promotion stays closed pending full initial-box coverage, the other boxes, later controls, controller/CROWN validation, and an independent long-horizon argument.

## Isolated first-box recentering candidate

A separate [copied-library first-step gate](evidence/results/archcomp26_20261001/native_quad_initial_recenter_repair_gate_20261003_002/README.md) retains the two VAR-tail repair sites and changes `Interval::toCenterForm` so its radius encloses **both** sides of the center under upward MPFR rounding. The original library, original long run and earlier isolated results were not changed. On the source-defined first physical QUAD box, the new saved RPC x1–x3 lower bounds equal the original split lower bounds exactly; the new upper bounds are one binary64 ULP wider, and all twelve RPC coordinates enclose the source box. Observer on/off each accepted one `h=0.005` step, saved one finite ordered range record and equal control/terminal data. The 513 relaxed-control numerical samples gave 0/20,520 exceedances. This is a new short diagnostic, not a new 50-period run.

The new RPC was checked independently rather than reusing the old 20/20 result. Its 1,000 directed-Decimal Picard substeps pass strict self-inclusion; plain-box propagation contains 27/32 saved columns. Exact-rational closed forms and a correlated x6 bound resolve the three remaining **composed physical-state** comparisons, giving 20/20 first-box, first-step plant comparisons under the exact-decimal paper ODE. The two additional pre columns are only numerical diagnostics; the narrowest unexpanded saved-bound margin among new composed comparisons is approximately `5.12×10⁻¹⁹`. This verifies neither Flow* decimal parsing nor the CROWN/NN calculation, other boxes, later controls or full-time reach-and-remain. The native octagon **production gate remains closed**.

## Gate to continue

1. Preserve both original-library counterexamples, both refinement-skip
   mechanism checks, and the separate two-site VAR-tail repair diagnostic.
   The short repaired candidate passes finite samples in the ordinary and
   symbolic-remainder paths and all 20 composed physical-state comparisons
   in the saved-RPC-domain, first-step independent plant gate. `COMPLETED_SAFE` is
   numerical status, not a complete validity check.
2. The isolated recentering candidate closes the identified one-ULP gap and
   passes the first physical box's first-step plant comparison. Qualify the
   other 1,023 boxes and later control calls before considering production.
   Compare observer on/off on an isolated short subset after any candidate repair. Do not alter
   the frozen full50 source or its original receipt; do not infer from this
   one-box failure that every saved long-run range is wrong.
3. Only then add a streaming support observer with per-lane accepted step
   and status metadata, bounded temporary TM storage, separately measured
   observation/export time, and a reader that treats any missing or rejected
   prefix honestly. A full QUAD rerun solely to improve its image is not
   justified by the current evidence.

The CPU Python TM octagon stream and its separate plant-only success are a
different implementation and do not clear this native Flow* gate.
