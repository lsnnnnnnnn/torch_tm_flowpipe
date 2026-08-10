# Xiangru `2026_experiment` direction audit

Date: 2026-08-10

## Provenance and audit boundary

The server checkout at `/srv/local/shengenli/CROWN-Reach_Development` is clean
at local `84184de6c2b3f1ff2da6755f732d91925037025d`.  Its existing
`origin/2026_experiment` reference is
`5a3f94b28a7303c42a34fb6d57ebdaba63f25e42`, 22 commits newer than the local
checkout and exactly equal to the uploaded archive header SHA cited in this
round's goal.  A fresh fetch was attempted on 2026-08-10 and failed because the
private GitHub remote was not authenticated.  Therefore `5a3f94b` is the newest
server-resident remote observation, not a claim about a later unseen upstream
tip.

The audit read the mandatory policy, roadmap, backend decision, Flow*/DiffReach
analysis, NAV plan, experiment READMEs, the NAV representation decision, their
machine-readable result paths, the fixed-basis Torch source entry point, and
its semantic/compiled tests.  Xiangru's repository was not modified.

## Result classification

| experiment | system | representation | validator | partitions | horizon | certificate status | property status | runtime boundary | reusable lesson |
|---|---|---|---|---:|---:|---|---|---|---|
| original CROWN-Reach | homogeneous TORA | Flow* native O3 complete TM | native Flow* Picard | B12 | 20 | complete | `VERIFIED` | S3C median core 2.072s; process 5.865s | native complete/high-order reference needs fewer leaves |
| upstream DiffReach | homogeneous TORA | DR13-K2 fixed support | DR-RP | up to B768 | 20 requested | `PARTIAL_HORIZON`, through 13.0s in corrected accounting | no T20 property certificate | a fast failed prefix is ineligible | returned arrays are not completion when required inclusion fails |
| Xiangru Q3 incumbent | homogeneous TORA | complete-Q3-K2 | DR-RP | B48 | 20 | complete | `VERIFIED` | populated-cache core 2.185s; process 39.475s | representation and validator must be named separately |
| Xiangru fixed31 candidate | homogeneous TORA | enriched fixed31-K2 | DR-RP | B48 | 20 | complete | `VERIFIED` | core 1.0374x Q3; peak allocation 0.9852x | fewer declared routes did not imply faster execution; do not promote |
| P0 one-shot partition | homogeneous TORA | complete-Q3-K2 | DR-RP | B26 | 20 requested | through 18.5s, failure at segment 186 | failed | incomplete work cannot be a speedup | stop after preregistered partition failure |
| native Flow* N0/N3 | NAV standard | Flowstar-O4 | native Flow* Picard | B640 | 6 | complete | `VERIFIED` | N3 median solver/process 79.173/93.546s | preserve native contract and avoid inferred phase splits |
| upstream DiffReach N0 observer | NAV standard | DR15-K2 fixed support | DR-RP | B640 | 6 | all 384,000 initial inclusions pass | full property qualified only after separate tube observer | after-JIT 0.713s endpoint, 0.708s tube | upstream output was endpoint-only because `BOUND_TIME_STEP=True` |
| Torch N1 equivalence | NAV standard | Torch DR15-K2 | DR-RP | B1 plus selected B640 leaves | 0.2 | complete semantic gate | diagnostic | no timing claim | 1,537 fields/comparison, max error 4.441e-16, zero masks/count differences |
| Torch N2 | NAV standard | Torch DR15-K2 | DR-RP | B640 | 6 | complete | obstacle and target property verified | 6.129s controller + 47.708s eager dynamics are prototype-only | fixed support already succeeds; Q3 was correctly stopped |
| Torch N3/N4 | NAV standard | compiled Torch DR15-K2 | DR-RP | B640 | 6 | complete | verified | warm core/process 7.708/18.923s; decisive 70-digit replay passes | cold/warm and deployment/algorithmic work remain distinct |

## Native baselines versus additions

### 1. Native baseline results

The unmodified Flow* and DiffReach rows above remain native only when their own
entry points, configs, partition policies, output semantics, and certificate
flags are retained.  TORA Flow* B12 T20 and NAV Flow* B640 T6 are complete
native certificates.  Native homogeneous-TORA DiffReach is only a certified
prefix; its after-JIT time is not a finite time-to-certificate.  Native NAV
DiffReach constructs the fixed-support flowpipe, but its stock saved bounds are
endpoint enclosures and its launcher does not implement the ARCH-COMP property.

### 2. Xiangru-added adapters and observers

ReachBench launchers for TORA/NAV/ACC/Double Pendulum, the NAV full-step tube
observer, controller wrappers, property checkers, trace exporters, and fair
timing harnesses are additions.  They are valuable observations but must not
be relabelled as upstream native features.  The NAV tube observer is
behavior-preserving because it changes output evaluation from `tau=h` to
`tau in [0,h]`; it is not the stock output mode.

### 3. Algorithmic changes

Complete-Q3-K2, enriched fixed31, the rejected DEF-CERT path, and the rejected
B26 partition are algorithmic alternatives.  The promoted Q3 TORA result is a
matched Torch lane, not a native Flow* or DiffReach result.  DR-RP remains the
fixed-support incumbent because paired evidence found DEF-CERT tighter in zero
of 52 state dimensions and showed no backend-independent work dominance.

### 4. Implementation-only changes

Static routing, tensor fusion, `torch.compile`, cache reuse, batching, the
native auto_LiRPA controller backend after qualification, and the compiled
DR15 NAV engine are implementation changes when paired coefficient, mask,
carry, and verdict equivalence holds.  These changes may support deployment
claims, not an algorithmic enclosure-quality claim.

### 5. Formal certificates

Formal/certificate-bearing rows require every initial DR-RP inclusion, finite
enclosures, and the applicable continuous property.  TORA Flow* B12 and Q3 B48
meet their declared contracts.  NAV N2/N3 meet the complete reach-avoid
contract.  N4 independently outward-replays decisive CUDA workloads, but its
controller check validates affine composition around the neural bounds rather
than independently proving the neural verifier itself.

### 6. Reachability-only artifacts

Upstream DiffReach saves flowpipes without the ARCH-COMP property verdict.
Endpoint-only NAV arrays, controller fixtures, local one-step traces, and
sampled trajectories remain reachability/diagnostic evidence until an
appropriate tube/property checker is applied.

### 7. Warning and non-contracting results

Any initial Picard inclusion failure invalidates the segment.  A later
componentwise refinement failure can soundly retain the previous interval but
must be reported rather than described as an accepted shrink.  Homogeneous
TORA's corrected DiffReach row is a partial horizon, not a complete T20 result.
P0 B26 is likewise a sound but incomplete negative result.

### 8. Timing semantics

Cold import/startup, graph construction, compile/JIT, first call, warm
dynamics, controller, validation, certification core, and process wall are
separate.  The TORA S3C Q3 core is near Flow*, but fresh-process Q3 remains
6.730x slower due to 32.918s graph reconstruction.  NAV's compiled deployment
gain is valid under its rotated environment contract; it is not proof of less
backend-independent algorithmic work.

### 9. Reusable architectural lessons

- Use a deterministic support descriptor and support hash.
- Reproduce the exact fixed-support polynomial, two Picard constructions, all
  DR-RP masks, endpoint, tube, normalization, and symbolic carry before timing.
- Return fail-closed completion separately from flowpipe arrays.
- Keep endpoint and full-step tube as distinct products.
- Keep representation, validator, carry/reset, partition, step policy, and
  backend as separate factorial axes.
- Use independent outward replay to qualify decisive CUDA workloads without
  claiming universal hardware-directed rounding.
- Compare algorithmic work separately from compiled wall time.

### 10. Choices frozen outside the general Torch engine

The homogeneous-TORA ODE/controller, complete-Q3 specialization, B48 grid,
fixed31 support, B26 split, controller lineage, 9,600 decisions, and T20
property are benchmark-specific.  None is copied into the generic Torch TM
kernel.  NAV's six-coordinate DR15 instance and B640 property policy are also
fixtures, not generic state indices or solver defaults.

## Direction decision

TORA is now a frozen stress-test reference.  This round independently derives
the VDP fixed support from DiffReach `dd628eb` and implements it in the general
Torch architecture; it does not transplant Xiangru's TORA/NAV solvers.  The
mainline research question is cross-step dependency preservation relative to
stock Flow*, after fixed-support behavior equivalence is established.
