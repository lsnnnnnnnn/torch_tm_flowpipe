# torch-tm-flowpipe

`torch-tm-flowpipe` is a PyTorch-native Taylor-model flowpipe research backend
for polynomial ODEs. The main research line compares three explicit design
points rather than treating one ambiguous `order` as a common algorithm:

- stock Flow*: mature complete high-order polynomials, normalization,
  preconditioning, and symbolic remainder;
- upstream DiffReach: restricted fixed support, fixed Picard/DR-RP, JAX/JIT,
  and large batch throughput;
- Torch TM: one architecture with a configurable DiffReach-like fixed-support
  lane and a Flow*-like complete-total-degree lane.

The latest research round closes the first Flow*/Torch raw-remainder split at
Picard-4 `x*x`, identifies a schedule/validator interaction, and closes the
R7-to-R35 A0--A4 bridge through T1 in B1 and B64. Flow*/Torch comparison is
partial, DiffReach/Torch explicit-f64 comparison is closed, and no Torch
production improvement is authorized by the evidence. The preceding S1
result remains `S1_REACHES_TERMINAL_BUT_DOES_NOT_CLOSE_IT`. Evidence-integrity
corrections are documented in
[`docs/EVIDENCE_INTEGRITY_CORRECTIONS_20260811.md`](docs/EVIDENCE_INTEGRITY_CORRECTIONS_20260811.md).
The frozen
TORA complete-Q3 work is a stress-test reference, not the project objective.
Adaptive DEF-CERT, obsolete winner/Pareto tables, and prior TORA-specific
comparisons remain historical or rejected.

## Current result

The fixed seven-slot Torch lane now has a cached immutable kernel plan and a
26-tensor functional state. Object and functional eager are bit-exact on the
full preregistered CPU/CUDA matrix. B64 T10 completes in the compiled lane with
zero graph breaks and no solver-core synchronization, but Inductor changes
floating-point arithmetic. Its 5.038 s CPU and 6.927 s V100 stable warm times
are performance-only empirical observations, not same-semantics speedups.

The separate CPU outward reference passes an independent 11-family exact
oracle and one-step containment, but fails closed before T1 (B1 step 33; B64
first failure step 90). Its result is
`FIXED_SUPPORT_FORMAL_SOUNDNESS_NOT_CLOSED`.

The complete-O4 Torch baseline is formally outward by its declared interval
path but stops at `t=6.397083942944808`, while stock Flow* completes its native
T10 request and upstream DiffReach completes its different native B64 T10
request. These native rows are not ranked: representations, validators, carry,
partitions, output objects, timing, and numerical qualification differ. The
stock Flow* build is itself ineligible for a primary formal claim after a
scalar-affine MPFR counterexample.

Machine reports separate mathematical-contract knowledge, requested-horizon
completion, certificate semantics, finite output, numerical class/scope,
formal-claim eligibility, performance eligibility, and cross-tool-ranking
eligibility. Native (N), matched-contract (M), and in-framework factorial (F)
rows remain separate.

The first Flow*/Torch schedule split is now causally observed at
`t=0.18187433604506256`: their transformed polynomial coefficients agree at
roundoff scale, but the raw candidate Picard remainder already makes Flow*'s y
subset fail while Torch passes. Polynomial, endpoint, and right-map swaps
preserve the receiving validator's decision.

The boundary-164 audit now explains that negative result. Diagnostic-only
typed ledgers and exact carrier split/remerge are bit-exact to the baseline;
the first causal inflation is the post-hoc `base=range(Q+R_o), perturbation=Z`
image decomposition. Coefficient/scale drift first appears at boundary 5,
physical-hull and margin drift at boundary/attempt 8, and outward
renormalization at boundary 12. K16 fill/eviction is not the primary cause.

Exactly one corrected carry was implemented:
`normalized_insertion_structured_total_delta_k16`. It evaluates
`P(Q + (R_o+Z)) - P(Q)` so all ordinary, structured, and mixed nonlinear
routes enter `N_total` once. It accepts all 307 historical accepted steps in
the corrected fixed-step replay and its boundary-307 checkpoint round-trips
exactly. It still rejects the frozen historical terminal step at
`t=6.397083942944808`, with y margin `-1.9999591170254726e-5`. The primary
outcome is `S1_REACHES_TERMINAL_BUT_DOES_NOT_CLOSE_IT`; fresh horizons and a
second system are `not_run_after_stop`. The primitive image is outward for
given binary64 coefficients, while the full prefix remains a
`safeguarded_binary64_interval_shell` conditional on retained coefficient
arithmetic and is not end-to-end formal.

Start with:

- [research direction](docs/RESEARCH_DIRECTION_20260810.md);
- [three-lane contract](docs/THREE_LANE_ALGORITHM_CONTRACT_20260810.md);
- [native baselines](docs/NATIVE_FLOWSTAR_DIFFREACH_TORCH_BASELINE_20260810.md);
- [fixed-support equivalence](docs/TORCH_DIFFREACH_FIXED_BASIS_EQUIVALENCE_20260810.md);
- [causal divergence](docs/VDP_FLOWSTAR_TORCH_CAUSAL_DIVERGENCE_20260810.md);
- [candidate result](docs/GENERIC_TORCH_TM_IMPROVEMENT_RESULT_20260810.md);
- [compiled fixed core](docs/FIXED_SUPPORT_COMPILED_CORE_20260810.md);
- [fixed soundness](docs/FIXED_SUPPORT_SOUNDNESS_20260810.md);
- [structured S1 result](docs/STRUCTURED_REMAINDER_RESULT_20260810.md);
- [S1 complete-O4 prefix result](docs/S1_PREFIX_INTEGRATION_RESULT_20260810.md);
- [S1 terminal causal gate](docs/S1_TERMINAL_CAUSAL_GATE_20260810.md);
- [S1 boundary-164 causal attribution](docs/S1_BOUNDARY164_CAUSAL_ATTRIBUTION_20260811.md);
- [S1 corrected carry result](docs/S1_CORRECTED_CARRY_RESULT_20260811.md);
- [three-tool pairwise result](docs/THREE_TOOL_PAIRWISE_COMPARISON_20260811.md);
- [Flow*/Torch O4 matched result](docs/FLOWSTAR_TORCH_O4_MATCHED_COMPARISON_20260811.md);
- [DiffReach/Torch DR7 matched result](docs/DIFFREACH_TORCH_DR7_MATCHED_COMPARISON_20260811.md);
- [raw-remainder root cause](docs/VDP_RAW_REMAINDER_ROOT_CAUSE_20260811.md);
- [schedule/validator causality](docs/VDP_SCHEDULE_VALIDATOR_CAUSALITY_20260811.md);
- [fixed-support bridge](docs/TORCH_FIXED_SUPPORT_DESCRIPTOR_BRIDGE_20260811.md);
- [single-improvement decision](docs/TORCH_SINGLE_IMPROVEMENT_RESULT_20260811.md);
- [second-system generality](docs/SECOND_SYSTEM_GENERALITY_20260810.md);
- [handoff](handoff.md).

## Quick start

```bash
conda run -n py11 python -m pip install -e ".[test]"
conda run -n py11 pytest -q
conda run -n py11 python examples/scalar_quadratic.py
```

The canonical package is `src/torch_tm_flowpipe`. It implements interval
arithmetic, sparse and dense complete bases, configurable fixed support,
Taylor models `p + R`, validated Picard propagation, explicit endpoint/tube
semantics, and fail-closed multi-step runners.

```python
from torch_tm_flowpipe import Interval, flowpipe_multi_step
from torch_tm_flowpipe.ode_examples import scalar_quadratic_ode

result = flowpipe_multi_step(
    scalar_quadratic_ode,
    [Interval(0.0, 0.1)],
    h=0.01,
    steps=5,
    order=4,
    mode="dependency_preserving",
)
print(result.status, result.final_tm.range_box())
```

The previous-round evidence is committed under the
[canonical run](outputs/mainline_realignment_20260810/20260810T025910Z/).
Raw output is kept separate from derived summaries, large text traces use
deterministic gzip storage, and the repository-root-prefixed `SHA256SUMS`
covers the complete stored tree.

The current closure package is the
[structured/compiled run](outputs/structured_remainder_compiled_fixed_support_20260810/20260810T070908Z/).
It keeps sanitized public raw evidence separate from derived tables and
figures; ignored compiler caches and source-mixed exploratory timings are not
part of the manifest.

The current prefix-integration package is
[S1 complete-O4 run](outputs/s1_prefix_integrated_complete_o4_20260810/20260810T095423Z/).
Its tables distinguish the sound 164-boundary common prefix from the discarded
off-schedule half-step and mark every prohibited later experiment explicitly
as `not_run_after_stop`.

The current boundary-attribution and corrected-carry package is
[S1 boundary-164 run](outputs/s1_boundary164_causal_guarded_carry_20260811/20260811T033447Z/).
Its 234-entry `SHA256SUMS` is repository-root-relative. Machine tables retain
the causal ladder, A0--B16 ledger, component substitutions, corrected
307-step gate, terminal rejection, independent claim fields, and explicit
stop rows for every unauthorized later stage.
