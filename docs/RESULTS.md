# Results

## Current three-tool pairwise/causal result

The current outcomes are `RAW_REMAINDER_ROOT_CAUSE_CLOSED`,
`SCHEDULE_VALIDATOR_INTERACTION`, and `FIXED_SUPPORT_BRIDGE_BLOCKED`. Every
A0--A4 B1/B64 cell completes G2/T1, but A0/B1 stops at T=5.36 and A4/B1 and
A4/B64 stop at T=3.19 and T=3.33 in G3. The first decision-changing node is
Picard iteration 4 `x*x`: Flow* direct interval-coefficient multiplication
adds uncertainty `[-0.00011204861774257546,0.00008935810062010431]` absent
from Torch's point-binary64 retained-coefficient path. Replacing only that
frozen contribution changes the Flow* y margin from
`-3.662398821521699e-6` to `+2.4888083156873676e-7`.

Both receiving subset predicates accept the Torch candidate and reject the
Flow* candidate, so candidate construction causes the first split and the
different accepted schedule then changes later producer states. Flow*/Torch
O4 is `PAIRWISE_COMPARISON_PARTIAL`; DiffReach/Torch DR7 explicit-f64 is
`VALID_PAIRWISE_COMPARISON_CLOSED`. No transitive ranking follows.

The improvement outcome is `IMPROVEMENT_NOT_AUTHORIZED_BY_EVIDENCE`: the
proven extra uncertainty is on the Flow* side, Torch already contains the
independent MPFR replay, and the narrower Flow* counterfactual is not proved
sound for arbitrary MPFR coefficients.

## Current authoritative S1 result

The current S1 outcome is
`S1_REACHES_TERMINAL_BUT_DOES_NOT_CLOSE_IT`. The corrected
`normalized_insertion_structured_total_delta_k16` carry replays all 307 frozen
historical accepted step sizes and produces a byte-stable boundary-307
checkpoint. The unchanged historical terminal proposal still rejects, with T0
y margin `-1.9999591170254726e-5`; fresh horizons and a second system were not
authorized. Primitive formal eligibility remains limited to the CPU outward
image for given binary64 coefficients; the full prefix is not formally
eligible.

Verification provenance is qualified by
[the evidence-integrity correction register](EVIDENCE_INTEGRITY_CORRECTIONS_20260811.md).
The following prefix-rejection section is preserved as a superseded historical
result, not the current headline.

## Superseded: S1 prefix-integrated complete-O4 result

The primary result of the current round is
`S1_PREFIX_REJECTS_BEFORE_TERMINAL`. The three observation-controlled lanes
use the checksum-verified 307-step historical schedule:

| lane | longest matching accepted prefix | time | first divergence |
|---|---:|---:|---|
| L0 historical baseline | 307 | 6.397083942944808 | none; terminal rejection also matches |
| L1 materialize every boundary | 164 | 4.738198114669049 | frozen proposed step rejected, half-step accepted off schedule |
| L2 structured K16 | 164 | 4.738198114669049 | same causal divergence as L1 |

L2 first fills K16 at boundary 16 and first evicts at boundary 17. The largest
single observed eviction contribution has maximum component width
`0.001549673642858923` at boundary 70. Across all 164 committed boundaries,
source decomposition, materialization conservation, unique ownership,
finiteness, normalized-domain containment, and endpoint/tube publication pass.
The boundary-164 checkpoint round-trips byte-for-byte with full checkpoint SHA
`9162f267fcdcf44ca7bb9acfa73975eb8f4f4b80c03ca217aac2f07450cd585b`.

From that exact prestate, the historical proposed
`h=0.03661680691961388` produces an S1 raw-compatible y subset margin of
`-3.773875528686747e-6`; it is rejected once and the adaptive helper returns
the half step. The half-step state is explicitly not part of the frozen prefix.
Because the historical terminal prestate was not reached, terminal A/B, the
fresh horizon ladder, +0.5 promotion, T10, and integrated second-system gates
are all `not_run_after_stop`, not negative numerical runs.

Current machine evidence:
`outputs/s1_prefix_integrated_complete_o4_20260810/20260810T095423Z`.

The remainder of this file records the preceding closure results.

Date: 2026-08-10
Canonical run: `outputs/mainline_realignment_20260810/20260810T025910Z`

Current closure run:
`outputs/structured_remainder_compiled_fixed_support_20260810/20260810T070908Z`

## Current closure outcomes

| lane | completed result | qualification / decision |
|---|---|---|
| object ↔ functional fixed | all 15 preregistered CPU/CUDA signatures bit-exact | ordinary, empirically sampled |
| compiled fixed B64 T10 | CPU 5.038 s, V100 6.927 s stable warm; zero graph breaks | arithmetic changed; performance-only; no same-semantics speedup |
| fixed outward | exact oracle and 1/10-step rows pass; B1 failure 33, B64 first failure 90 | `FIXED_SUPPORT_FORMAL_SOUNDNESS_NOT_CLOSED` |
| complete-O4 S1 | generic K16 primitive; local empty-history terminal attribution closes | prefix A/B unavailable; `STRUCTURED_REMAINDER_LOCAL_GATE_FAILED`; no horizons |
| second system | harmonic + Riccati, CPU/V100, B1/B64, 100 steps | `GENERALITY_GATE_PASSED`, plant-only fallback scope |

The compiled B64 raw timing ratios against isolated frozen object medians are
about 15.0x CPU and 18.4x V100. They are retained as raw ratios only because
Inductor changes arithmetic. The compiled CPU is faster than V100. B1 object
and compiled lanes agree on first failure step 536 despite non-bit-exact
arithmetic.

At the frozen complete-O4 terminal, baseline y margin is
`-1.99995911680722e-5`. Empty-history local S1 ordinary y margin is
`+9.090310982602511e-5` and full materialization contains the original image,
but the missing 307-step S1 state prevents promotion. Every fresh ladder row
is machine-recorded as not run after STOP.

## Native design points

| lane | native result | exact output semantics | soundness / eligibility |
|---|---|---|---|
| stock Flow* `b85a321...` | T10, 290 segments, core `0.441634 s` | full-segment tubes | `unsound/ineligible on a demonstrated counterexample` for this pinned native build after the scalar-affine MPFR defect |
| stock DiffReach `dd628eb...` | B64 h=.01 T10; all 128,000 returned initial masks pass | endpoint at local time h; no stock tube | `empirically sampled only`; mixed builder dtype |
| Torch complete O4 | partial; highest validated `6.397083942944808` | raw endpoint, last tube, prefix tube separate | `formally outward by construction`; ineligible as T10 completion |

These rows reproduce native behavior; they do not form a ranked comparison.

## Torch factorial

| representation + validator + carry | result | decision |
|---|---|---|
| fixed DR7 + two Picard + DR-RP + normalized affine/symbolic carry | B64 T10 completes; explicit-f64 fixture bit-exact | qualified DiffReach-like lane |
| complete O4 + raw-remainder-compatible validator + normalized insertion | partial through `6.397083942944808` | authoritative complete baseline |
| complete O4 + same validator + exact complete endpoint carry | partial through `0.04345468750000001` | candidate rejected |

The fixed DR7 B64 lane independently completes every requested
T=.1/.5/1/4/6/6.5/7.5 run and the fresh T10 run. The adaptive complete-O4
baseline independently requests T=7.5 and T=10 and stops at the same
`6.397083942944808` boundary. The complete-carry candidate independently
requests all eight horizons and stops at the same `0.04345468750000001`
boundary.

The validators cannot be blindly crossed: DR-RP is defined on the restricted
slot contract with component-retain semantics, while the complete raw-remainder
validator consumes a complete-basis truncation ledger. A forced cross would
change construction and acceptance semantics, not isolate one factor.

## Causal answer

At the first split, common-basis coefficient error is at most `1.421e-14` in y.
Flow* raw y remainder is
`[-1.0366239882151062e-4, 1.0359846643018429e-4]`, already outside the target;
Torch raw y is `[-9.14291532216261e-5, 9.358938647674799e-5]`, inside. Swapping
polynomials, endpoints, and right maps does not swap the receiving decision.

## Performance answer

Fixed-support CPU warm time for 10 steps rises from `0.4772511 s` at B1 to
`0.6959472 s` at B512; synchronized V100 remains about `1.25 s`. Complete O4
one-step-plus-carry rises from `0.1168205 s` to `9.3192035 s` on CPU and from
`0.3070624 s` to `10.4725577 s` on V100. Inputs are actual independent grid
partitions. The complete batch rows are a one-step kernel scope because the
outer adaptive scheduler remains batch-one.

No eligible cross-tool deployment speedup, GPU speedup, or precision-throughput
Pareto frontier is claimed.

Completion, certificate semantics, finiteness, numerical class/scope, formal
claim eligibility, performance eligibility, and cross-tool ranking eligibility
are independent columns in the machine tables. Track N contains native
reproductions without ranking, Track M contains matched-contract facts only
where natively expressible, and Track F contains the Torch representation /
validator / carry / backend factorial.

## Claim status

- Valid: native entrypoints reproduced; fixed-support explicit-f64 semantics;
  exact causal stage; exact carry soundness; measured partial horizons.
- Invalid: global Torch-vs-Flow* tightness/speed; native/matched equivalence;
  ordinary CUDA as universally outward rounded; complete carry improvement.
- Blocked: formal stock Flow* primary comparison; formal ordinary
  fixed-support CPU/CUDA certificate; multi-step batched complete lane.
