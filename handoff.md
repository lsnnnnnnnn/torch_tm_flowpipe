# Handoff: TORA-Q3 stage parity, fused runtime, and native closure

## Bottom line

The final result is **Case C**. P0--P5 pass, including both internal 10x
runtime gates. The new sound aligned implementation and one evidence-selected
h=.05 fallback were run through the strict native hierarchy, but neither
passes T5. The best Torch-native certified horizon is `4.4 s`; T10 and T20 are
`NOT_RUN`, and no Torch T5/T10/T20 width is fabricated. Xiangru independently
verifies its own native lane through `20.0 s`.

## The 22 required answers

1. **What are the start and final HEAD?** The exact start is
   `63efe66cfe7bdda907f8255ba23cebaa9b878233`. Checkpoint 6 is
   `34030070b24b40a37e014b9ef4571017ed9ca9af`. The authoritative final HEAD is
   the 40-character value returned by `git rev-parse HEAD` after checkpoint 7
   and must equal the remote branch value; the completion response records it.
   A commit cannot embed its own not-yet-computed hash without making the hash
   self-referential.

2. **Are clean lineage and blocked history still isolated?** Yes. The branch
   descends from parentless clean root
   `9fc45344c4379422244b75af705dffd17304f824`; merge-base with blocked tip
   `c49d74bbf48d1004f7f3818174e7f40b6200b142` is absent. The user's original
   dirty worktree was not modified.

3. **What are the portable, external, and GPU test results?** Final full pytest
   is `135 passed, 15 skipped`. External integration with explicit controller
   and stage assets is `3 passed, 2 skipped, 145 deselected`. The independent
   GPU-focused rerun is `4 passed, 20 deselected` on a Tesla V100-SXM2-16GB,
   PyTorch 2.8.0+cu128, and CUDA 12.8. The GPU lane is recorded separately and
   cannot pass through a no-CUDA skip.

4. **Did Xiangru observation prove that outputs were unchanged?** It proves
   non-invasive aggregate behavior under the declared contract: 200 segments,
   accepted leaves, status, failure state, and `20.0 s` horizon are equal; all
   2,850 correctness fields are within `1e-6`, with maximum aggregate
   difference `2.4868995751603507e-14`. The prior exporter versus stage
   exporter maximum is `5.861977570020827e-14`. Per-leaf bitwise equivalence to
   an uninstrumented raw array is not claimed because that array was never
   exported.

5. **Where does the T1 `0.014211` first appear mathematically?** A2 point-sine
   outward rounding is the first numerical difference, only `4.22e-15`. A3
   sine-composition remainder routing and analytic-tail semantics is the first
   material difference (`0.0145972551` width at segment 1, leaf 0). After ten
   plant steps, `99.924210%` of the direct
   `0.014211021942602` endpoint difference is present before projection.

6. **Is the important difference a center shift or radius growth?** Radius
   growth dominates. A3 changes the remainder width, while retained
   coefficients remain at roundoff scale. At baseline failure, aligned versus
   baseline center moves only about `6.43e-5`; the x3 radii are `1.045678` and
   `1.046220`, already larger than one. The width gap is one to two orders of
   magnitude larger than the center gap at the common native horizon.

7. **What dominates the segment-40 remainder `1.218619`?** The carried
   `composition_overflow` ledger category reaches `1.2186185882008727`, versus
   current-step `picard_residual` `0.001269783182269888` (about `959.7x`
   smaller). A3 is the earliest material generator; A7/A8 integration
   degree-overflow routing is secondary. Projection inflation is only
   `3.92e-12`.

8. **What mathematical semantics changed in the new aligned lane?** It keeps
   complete Q3, K2 polynomial Picard, ten remainder rounds, natural range,
   h=.1, and the property. It independently replaces generic sine remainder
   routing with a centered quadratic retained polynomial, signed
   input-remainder propagation including `2pr`, a full line-segment cosine
   bound for the third derivative, and separate outward routing of analytic,
   composition-overflow, and retained-route errors.

9. **How far is it from Xiangru at one step, R1, and R2?** G0 one-leaf and G1
   B48 have maximum coefficient difference `5.110e-10`, local-remainder center
   differences `1.952e-5`/`1.966e-5`, and endpoint-center differences
   `1.738e-5`/`1.752e-5`. R1/G3 is `4.664e-10`, `1.780e-5`, and `1.713e-5`;
   R2/G4 is `1.322e-9`, `2.572e-5`, and `2.512e-5`. All accepted-leaf gates
   match and pass; endpoint, tube, radius, remainder, and containment counts
   remain separate in the CSV/JSON evidence.

10. **How tight is common-control T20?** All 200 segments pass with minimum
    property margin `0.2870697251`. Endpoint width-ratio median/max are
    `1.81738/12.1162`, maximum center difference `0.0236115`, and maximum width
    difference `1.22604`. Tube values are `1.63951/5.50192`, `0.0236115`, and
    `1.22535`. Zero reference widths are N/A/excluded from ratios. This is a
    period-local plant comparison, not native closed-loop T20.

11. **Which native T1/T5/T10/T20 gates pass?** Baseline K2, K3, aligned Q3,
    and aligned h=.05 all pass one-leaf, B48 one-step, and B48 T1. All fail the
    attempted T5 gate. Under strict previous-pass-only gating, T10 and T20 are
    `NOT_RUN` for every Torch lane.

12. **What is the first property or numerical failure?** Baseline K2 and
    aligned Q3 first fail the unchanged property at segment 44, leaf 0,
    certifying through `4.3 s`. K3 and h=.05 first fail property at segment 45,
    leaves 0/1/6, certifying through `4.4 s`. At each first failure,
    finiteness, initial inclusion, and all remainder rounds still pass: the
    numerical certificate is true, so this is not a numerical failure reported
    as a safety result.

13. **Which stages does the fused kernel cover?** Natural polynomial range,
    polynomial RHS and K2 Picard, initial remainder inclusion, ten remainder
    Picard rounds, endpoint/tube bounds, and all local acceptance predicates.
    Taylor-model objects and diagnostic ledgers stay outside the pure-Tensor
    boundary.

14. **What are graph breaks, launches, syncs, and `aten::to`?** The deployed
    four full graphs have zero internal graph breaks and 13 fixed invocations.
    Per B48 logical step, frozen/aligned/fused CUDA launch APIs are
    `75,440/139,417/7,941`; Kineto item/local counts are `80/80`, `80/80`, and
    `7/7`; program-issued syncs are `4/4/1`; `aten::to` counts are `81/81/25`.

15. **Where did B48 one-step and T20 land from 0.508/105.480?** The formal
    B48 logical-step median is `0.1296528154052794 s`, down from
    `0.508396873716265 s`. Common-control T20 is
    `26.18573095370084 s`, down from `105.48005206231028 s`, over five measured
    repeats after one excluded complete warm-up.

16. **What is the speedup over the 512.024 baseline?** The allowed internal
    PyTorch speedup is `19.553566325453474x` over the frozen
    `512.0244269836694 s` common-control baseline.

17. **How much slower is it than Xiangru 1.20676?** The descriptive matched
    common-control ratio is `21.6992x` Torch/Xiangru. It is not an internal
    optimization speedup and is not a GPU-parity claim.

18. **How expensive is cold compilation and when does it amortize?** B48 cold
    compilation, eager reference, and signature verification take
    `209.21813482325524 s`. Against the prior T20 path this amortizes after
    about `2.639` T20 runs, or about `552` individual logical steps; it is not
    attractive for a one-off verification.

19. **What are peak CPU and CUDA memory?** A full-protocol resource-only rerun
    measured maximum process RSS `6,925,746,176` bytes and peak CUDA allocation
    `1,031,874,048` bytes. Its slower runtime is excluded; the original formal
    `26.185731 s` timing remains authoritative.

20. **Is the old VDP issue still unresolved?** Yes. The historical Van der Pol
    failure near `t=6.397083942944808` remains explicitly unresolved and is an
    independent workstream. This branch makes no VDP correctness claim.

21. **Which raw/private evidence stayed out of Git?** Xiangru raw source copy,
    observation patch, raw stage tensors, raw per-leaf traces, controller and
    ONNX bytes, Chrome traces, complete commands/environment dumps, Inductor
    caches, failed debug snapshots, server paths, and credentials. Public
    records contain only independent Torch code, schemas, hashes, aggregates,
    sanitized telemetry, tests, reports, and figures.

22. **What is the smallest next technical problem?** For tightness, implement
    and validate a representation that preserves useful correlations so
    carried interval remainder does not dominate the period-5 controller
    input; another sine micro-tweak or unbounded range sweep is unsupported.
    For performance, reduce the 13-invocation dense 84-slot route and memory
    traffic while retaining outward containment.

## Publication and replay contract

Use branch `codex/tora-q3-stage-parity-fused-kernel-native-t20-20260809`.
Acceptance requires a clean worktree and exact equality between `git rev-parse
HEAD` and `git ls-remote origin
refs/heads/codex/tora-q3-stage-parity-fused-kernel-native-t20-20260809`.
The three manifest locations must be byte-identical, pass `sha256sum -c`, and
cover every tracked non-manifest file.

Checkpoint commits are `60f89ed` (baseline freeze), `4b1f547` (observation),
`efb9e02` (first divergence), `7e4c53f` (aligned lane), `8b46643` (fused
kernel), and `3403007` (native hierarchy). Checkpoint 7 publishes this handoff,
final tests, comparison summaries, scan, audit, and manifest without force,
merge, or history rewrite.
