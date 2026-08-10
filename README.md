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

The active branch is
`codex/s1-prefix-integrated-complete-o4-closure-20260810`. The frozen
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

The bounded structured-remainder candidate S1 is now a real opt-in
complete-O4 lane from `t=0`: K=16 state is owned by the normalized flowpipe,
typed dense sources feed accepted boundaries directly, complete degree-four
endpoint and tube sensitivities use safeguarded outward arithmetic, every
insertion/eviction has a unique source ledger, and schema-v2 checkpoints store
the full state exactly. On the frozen historical schedule, all conservation,
ownership, finiteness, and publication gates pass through boundary 164 at
`t=4.738198114669049`. The next historical step
`h=0.03661680691961388` is rejected by S1 (raw-compatible y margin
`-3.773875528686747e-6`) although the historical baseline accepted it. The
primary outcome is therefore `S1_PREFIX_REJECTS_BEFORE_TERMINAL`. The
historical terminal A/B, fresh horizon ladder, and integrated second-system
gate were not authorized.

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
