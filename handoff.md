# Flow*–Torch source/carry audit handoff

Current full-horizon canonical outcomes:

- `FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_ONLY`
- `DIFFREACH_TORCH_DR7_FULL_HORIZON_DIVERGED`
- `CARRY_MISSING_SYMBOLIC_SEMANTICS`
- `NO_FIX_AUTHORIZED`

The preceding bridge and S1 claims remain historical/superseded; this audit
refines the complete-O4 carry cause without replacing those four repository
headline outcomes.

Final scientific status:

- `BASELINE_CONCLUSIONS_REPRODUCED`
- `FLOWSTAR_WIDTH_IS_POSITIVE_NEAR_ZERO`
- `SOURCE_LEVEL_DEPENDENCY_LOSS_LOCALIZED`
- `NO_FIX_AUTHORIZED`

## Publication identity

- remote ref: `origin/codex/flowstar-torch-source-carry-root-cause-20260813`
- final tested source-and-evidence SHA:
  `adb985e703b61a384703bfa724021472caa3f870`
- Flow* source SHA: `b85a3211748cb77b736fe4ad42ee02d8d2b81148`
- baseline Torch source SHA: `8e7dbfbd305042adbd1bede47381c33ba73d7d7b`
- portable evidence:
  `outputs/flowstar_torch_source_carry_root_cause_20260813/20260813T030338Z/`

The publication branch has one child attestation commit containing this handoff
and the recorded fresh-clone result. A Git commit cannot literally contain its
own object ID without changing that ID; resolve the exact publication tip with
`git rev-parse origin/codex/flowstar-torch-source-carry-root-cause-20260813`.
The final user handoff records that resolved value explicitly.

## Verification

The tested SHA was fetched into a new clone and checked out detached from
origin. The following passed there:

- editable test install: exit 0;
- focused audit suite: 18 passed;
- full repository suite: 687 passed, 2 skipped (689 collected);
- `python -m compileall -q src experiments tests`: exit 0;
- package checksum/load/rederivation: 55 files hashed, 27 JSON files loaded,
  Flow* 1000 accepted rows, Torch 632 accepted plus one rejected candidate,
  four minima and 16 checkpoint ratios rederived;
- final fresh-clone worktree: clean.

The committed `12_final_clone/` attestation includes the exact tested SHA,
JUnit files, and rederivation summary. After the attestation commit is pushed,
the publication tip is checked once more from a second fresh clone; that final
check is reported with the resolved publication SHA and does not mutate tracked
scientific content.

## Result and remaining question

The Flow* minima are positive (`0.00861`, `0.02627`, `0.00889`, `0.03089`),
not zeros or serialization artifacts. Flow* carries linear old sources once
through `Continuous.cpp:2151-2177`'s `Phi_L/J` queue and uses Horner insertion;
legacy Torch independently composes monomials in `flowpipe.py:698-739`, after
the whole constant-removed state is sent through `flowpipe.py:1470-1511`.
The first published difference is step 1 and it changes step-2 scales.

No candidate is authorized because the post-step Flow* state/queue lacks a
lossless bridge, the pinned Flow* correctness gate is independently open, and
no complete outward source-ledger primitive proves nonlinear O4 carry.

The single remaining question is: can a lossless binary state-and-queue fixture
and independently outward-rounded source-ledger carry primitive prove the full
one-step O4 containment contract, including nonlinear multiplication,
truncation, cutoff, and renormalization?
