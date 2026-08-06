# Handoff: native Torch TORA-Q3 matched audit

> Publication note: the Git-state statements below describe the frozen source
> audit point.  The owner later requested a separate parentless review branch;
> see `CLEAN_REVIEW_PUBLICATION.md`.  That clean lineage does not make any
> authorization-unknown historical object reachable.

## Final outcome

The required common-control plant lane is **FORMALLY ALIGNED and VERIFIED to
T20**.  The native Torch full closed loop passes T1, certifies through T=4.3,
and fails closed at segment 44 (T=4.4) when leaf 0's `x3` tube crosses the
frozen ±2 property.  The result is therefore **Case B + Case D**: the formal
plant comparison is available, full-loop T20 tightness/runtime is `N/A`, and
public delivery remains blocked by unknown historical asset authorization.

The independent VDP `t=6.397...` issue remains unresolved and out of scope.

## Exact repository state

- Torch implementation worktree: detached HEAD
  `c49d74bbf48d1004f7f3818174e7f40b6200b142`, no upstream, intentionally
  dirty with the reviewed implementation and evidence.
- Frozen Xiangru reproduction: detached HEAD
  `27d29050a5f214b56f211ca9cb411e734ed80230`, clean, no upstream.
- DiffReach: branch `main`, HEAD
  `dd628eb443b517d6415de93e7035b4baef73963e`, upstream `origin/main`, clean.
- The user's pre-existing dirty Torch worktree was not modified by this task.
- Source-lineage commits: none.  The owner later requested the separate
  parentless `codex/tora-q3-native-clean-review-20260806` review branch, which
  was pushed without making the source lineage reachable.  History rewrite or
  force push: not performed.

## Formal gates

- Public/private artifact boundary: new deliverables contain no controller or
  checkpoint bytes, raw per-leaf traces, private source patch, or unsanitized
  logs.  Existing history contains three identical transformed ONNX objects
  with no established redistribution grant, so governance is
  `BLOCKED_UNKNOWN_AUTHORIZATION`.
- Sine TM: PASS for scalar points, symmetric/asymmetric intervals, degrees
  0–3, nonzero input remainder, extrema crossings, overflow, high-precision
  grid containment, and TORA domains.  Wide unsupported domains fail closed.
- Q3 basis/backend: PASS; six variables, complete total degree ≤3, 84 slots,
  identity Xiangru/Torch slot permutation, fingerprint
  `fa135259d41a68a73a6fc609880c4fd466bf2d53b2dddeba30298a484fa5e44d`,
  CPU/CUDA float64.
- Common-control plant gates: one leaf/one step PASS; B48 one step PASS; B48
  T1/T5/T10/T20 all PASS; 200/200 segments and 48/48 leaves per segment.
  This lane restarts from the same observed box and control interval each
  controller period and is not an independent closed loop.
- Native controller: nominal ONNX comparison PASS at maximum error
  `5.1034085e-7` under tolerance `1e-6`; initial B48 interval bounds PASS at
  maximum difference `3.5527137e-15`.
- Native full loop: T1 PASS; T5 FAIL at segment 44; T10/T20 `N/A`; completed
  43 segments and certified horizon 4.3.  The failed leaf-0 tube margins were
  `[1.462986, 1.114402, -0.0420120, 0.0626519]`.

## Tightness and first divergence

At T20, median Torch/Xiangru endpoint-width ratios for
`[x1,x2,x3,x4,u1]` are
`[0.963604, 0.916275, 1.00000000009, 1.00000000007, 1.00000000007]`;
tube ratios are
`[1.025418, 0.969176, 1.012227, 1.036472, 1.00000000007]`.
Neither method is uniformly tighter across state, time, leaf, and enclosure
kind.

The corrected common-control lane has no acceptance divergence.  Its first
numeric difference is segment 1 `x1`: endpoint maximum upper difference
`3.8234e-6`, tube maximum upper difference `1.0082e-3`, and leaf-0 remainder
width difference `4.0129e-6`.  Segment-1 K1/K2/final/physical coefficient
maximum difference is `5.1096e-10`; normalization center is exact and the map
linear difference is `4.44e-16`.  Remaining plant differences are classified
as expected analytic-sine, fixed-support overflow/roundoff, remainder-Picard,
and range-factorization differences.

The first behavior-relevant full-loop difference occurs at the T1 controller
refresh: Torch and Xiangru controller input boxes differ by up to `0.0142110`,
and their resulting control intervals by up to `0.171190`.  This accumulates
through method-native state projection and feedback; it is not a controller
hash or nominal-network mismatch.  No acceptance criterion was relaxed.

## Runtime

On the same V100, float64, B48, with one excluded full warm-up and five T20
repeats:

- Torch solver excluding serialization: median 525.862164 s, IQR 1.085531 s,
  range 522.575163–526.762578 s; peak CPU/GPU
  1,074,946,048 / 927,533,568 B.
- Xiangru solver excluding serialization: median 1.033485 s, IQR 0.001846 s,
  range 1.027768–1.034936 s; peak CPU/GPU
  1,455,931,392 / 6,471,680 B.
- Torch/Xiangru solver-median quotient: 508.824144, reported only as a
  descriptive end-to-end ratio.
- The formal GPU speedup claim is disallowed: one profiled Torch full step
  contained 128,472 paired host scalar synchronization events.
- Controller 10-repeat synchronized median: 0.026925 s.  Native full-loop T20
  runtime: `N/A` because the sound run stops at segment 44.

## Final verification

- Editable test install: PASS.
- Portable full suite: `506 passed, 6 skipped`.
- py11 external integration: `2 passed, 2 skipped, 508 deselected`; optional
  ONNX/controller-bound cases are intentionally executed in the frozen CROWN
  environment.
- CROWN environment preflight: Python 3.11.15, ONNX 1.22.0, auto_LiRPA 0.7.2,
  Torch 2.8.0+cu128, CUDA 12.8, Tesla V100-SXM2-16GB.
- CROWN controller external integration: `2 passed, 1 deselected`.
- `git diff --check`: PASS.  Final status is intentionally dirty and detached
  because Case D forbids creating a public descendant before authorization.

## Evidence and sole next action

Reviewed aggregate evidence is under
`outputs/tora_q3_native_matched_20260806/`; raw traces, controller bytes,
private observation code, decimal/hex details, and raw logs remain in the
separate private evidence root.  `manifest.sha256` covers only reviewed public
files.

**Sole next action:** review the parentless clean branch.  Do not merge it into
the authorization-unknown historical lineage; any later integration or asset
publication still requires an explicit license/authorization decision.
