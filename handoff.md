# Handoff: TORA-Q3 performance and closed-loop closure

## Bottom line

The result is **Case C**. The sound K3 full-loop candidate improves the formal
horizon from `T=4.3` to `T=4.4`, but the required 10x B48 one-step and
common-control T20 GPU gates remain unmet. P0/P1/P2 pass; P3/P4 fail. No T5,
T10, or T20 native-width value is invented after the candidate fails closed at
segment 45.

## The twelve handoff answers

1. **Is the clean branch self-contained and runnable?** Yes, for the stated
   native TORA-Q3 review surface. README links/commands are machine-checked,
   the portable example runs without private assets, editable installation and
   the full portable suite pass, and optional controller/Xiangru tests skip
   explicitly when their external inputs are absent. The branch descends from
   clean tip `7dcbe7cd901a941bd7508a107ecb0cc6f877ca1f` and parentless root
   `9fc45344c4379422244b75af705dffd17304f824`; it has no merge base with blocked
   historical tip `c49d74bbf48d1004f7f3818174e7f40b6200b142`.

2. **What is current common-control T20?** It is 20 matched one-period plant
   replays. Every period restarts from the frozen Xiangru pre-controller state
   box and held-control interval. It is useful for algorithm-aligned plant
   runtime/tightness comparison, but it is not a Torch state propagated
   independently and continuously to T20, and it never substitutes for the
   native full-loop gates.

3. **How much of the historical ~509x came from software environment versus
   host sync/kernel structure?** The newly matched baselines are Torch
   `512.024427 s` and Xiangru `1.206760 s`, or `424.296975x`. Torch py11 versus
   matched CROWN is only `1.002752x` (about 0.275%), so the software-version
   difference explains very little. Tensorization/compiled fixed-shape work
   reduces matched Torch T20 to `105.480052 s`, a `4.854230x` runtime ratio
   versus its own matched baseline. It remains `87.407680x` slower than Xiangru
   and therefore does not pass the required 10x GPU gate.

4. **Where did the baseline host synchronizations come from?** The reproduced
   baseline has 140,012 Kineto item/local events. The largest source-line
   family is identity-cover validation in
   `validate_dense_subdivision_cover`: 56,550 events (40.389%). Its two cell
   enumeration comprehensions add 22,464 each; the three together account for
   72.478%. The largest conversion site is `_power_interval_bounds` in
   `sin_tm`, with 6,552 `aten::to` events.

5. **What are the optimized sync and runtimes?** The dispatcher audit sees 3
   program-issued host scalar extractions per logical step. Kineto reports 77
   item/local observations, of which 74 do not pass the Torch dispatcher.
   `aten::to` falls from 19,226 to 80 (99.583897%). B48 one-step medians are
   eager `0.680230 s` and compiled `0.508397 s`; compiled common-control T20 is
   `105.480052 s` median over five measured repeats after one excluded complete
   warm-up. Status and checksum are stable. These are matched-stack runtime
   ratios; P3/P4 still fail 10x, so they are not presented as a claim that the
   required GPU acceleration gates passed.

6. **What exactly happens at T4.4?** Baseline segment 44 is a property failure,
   not a numerical certificate failure. All 48 leaves pass finiteness, the
   initial subset check, and all ten remainder rounds; leaf 0 fails the
   unchanged `|x1..x4| <= 2` property. Diagnostic-only continuation remains
   numerically certified through segment 47 and first loses the numerical
   certificate at segment 48. Formal safety still stops at segment 44.

7. **Where does the T=1 `0.014211` difference arise?** The direct exact-time
   endpoint already differs from Xiangru by `0.014211021942602`. Projection and
   carry materialization change that result by only
   `1.07787637813328e-05`; 99.9242% is present before projection. The dominant
   source is the preceding ten native plant segments, not affine projection,
   carry materialization, or controller-input construction.

8. **Which sound candidate was implemented, and why?** The selected candidate
   is complete-Q3 K3 polynomial Picard with ten remainder rounds, unchanged
   support, step size, outward rounding, and property. L1 tight endpoint-box
   control worsens the horizon, L2 physical-endpoint projection is unchanged,
   and L3 Horner is unchanged and slower. K3 is the only tested sound lane that
   improves the formal horizon. It is explicitly a method ablation and is not
   called algorithm-identical to frozen K2.

9. **How far does native full loop go?** Baseline K2 certifies through `T=4.3`
   and fails property at segment 44. K3 certifies through `T=4.4` and fails
   property at segment 45 for leaves 0, 1, and 6. The one-leaf, B48 one-step,
   and B48 T1 gates pass before the attempted T5 gate fails. T10/T20 are
   `NOT_RUN` by the required hierarchy, and target widths are `N/A`.

10. **What is the smallest remaining technical problem?** For tightness, reduce
    the remainder-dominated width entering the period-5 controller. At segment
    40 the maximum pre-projection interval-remainder width is `1.218619`, versus
    `0.156407` for the polynomial range, while projection inflation is only
    about `2e-12` to `4e-12`. For performance, the next bounded change is a pure
    tensor boundary around the whole natural-range/K2-plus-ten-remainder phase;
    compiling the Python dataclass/ledger entry already failed and is not used.

11. **What is the VDP status?** The historical VDP issue near
    `t=6.397083942944808` remains unresolved and is outside this TORA branch.
    Nothing here claims it was fixed.

12. **What exact branch, validation, manifest, and remote contract should a
    reviewer use?** Use branch
    `codex/tora-q3-performance-closed-loop-closure-20260806`; the authoritative
    commit is the branch's `HEAD`, and publication is accepted only when
    `git ls-remote origin refs/heads/codex/tora-q3-performance-closed-loop-closure-20260806`
    equals `git rev-parse HEAD`. The formal server external suite passes its two
    supplied-asset tests and intentionally skips two excluded legacy-raw tests.
    The portable/full counts and clean-worktree result are recorded in
    `outputs/tora_q3_perf_closure_20260806/tests/final_validation.json`. The two
    manifest locations must be byte-identical, pass `sha256sum -c`, and cover
    every tracked non-manifest file. The final user-facing completion response
    records the exact pushed 40-character HEAD after this handoff and its
    manifest are committed, avoiding an impossible self-referential commit
    hash inside the commit itself.

## Evidence map

- profiler and source attribution: `TORA_Q3_GPU_BOTTLENECK_REPORT.md`
- optimized runtime and resource protocol: `TORA_Q3_OPTIMIZED_RUNTIME_REPORT.md`
- lifecycle/root-cause attribution: `TORA_Q3_T4_4_WIDTH_ATTRIBUTION_REPORT.md`
- full-loop conclusion: `TORA_Q3_CLOSED_LOOP_CLOSURE_REPORT.md`
- deterministic replay: `outputs/tora_q3_perf_closure_20260806/full_loop_attribution/r1_r2_replay.json`
- hierarchical gates: `outputs/tora_q3_perf_closure_20260806/full_closed_loop/hierarchical_gates.json`
- requirement audit: `outputs/tora_q3_perf_closure_20260806/provenance/final_requirement_audit.json`
- governance: `PUBLIC_ARTIFACT_GOVERNANCE_AUDIT.md`

## Published checkpoints

The remote branch contains the following non-rewritten clean-lineage
checkpoints; the final supplement commit adds only the completion-audit gaps
identified after `059c7ef`:

- `1c12e76` — `Make clean TORA-Q3 review branch self-contained and portable`
- `54f0827` — complete-tree public manifest checkpoint
- `c691348` — `Reproduce TORA-Q3 baseline in matched software environments`
- `00d6571` — source-stage profiler evidence plus validation tensorization
- `5c08ecb` — `Add compiled fixed-shape TORA-Q3 kernel with sound fallback`
- `4b89cb1` — `Attribute TORA closed-loop width growth at controller refreshes`
- `66f163d` — `Add sound TORA-Q3 reconditioning candidate and T5 gate`
- `059c7ef` — initial closure reports and manifest

All were pushed without force, merge, or history rewrite. The source-stage
profiler and tensorization landed together in `00d6571`; the public files retain
the separate baseline and five optimization-iteration reports even though
those two closely coupled changes share one checkpoint commit.
