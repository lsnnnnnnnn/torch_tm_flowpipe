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
`codex/torch-tm-flowstar-diffreach-mainline-realignment-20260810`. The frozen
TORA complete-Q3 work is a stress-test reference, not the project objective.
Adaptive DEF-CERT, obsolete winner/Pareto tables, and prior TORA-specific
comparisons remain historical or rejected.

## Current result

The fixed seven-slot Torch lane reproduces pinned DiffReach float64 operator,
Picard, every DR-RP round, endpoint/tube, and symbolic-carry semantics with no
external runtime dependency. Its B64 fixed-step T10 run completes. Ordinary
CPU/CUDA float64 remains `empirically sampled only`; the exact-workload 2-ULP
companion envelope is independently replay-qualified.

The complete-O4 Torch baseline is formally outward by its declared interval
path but stops at `t=6.397083942944808`, while stock Flow* completes its native
T10 request and upstream DiffReach completes its different native B64 T10
request. These native rows are not ranked: representations, validators, carry,
partitions, output objects, timing, and numerical qualification differ. The
stock Flow* build is itself ineligible for a primary formal claim after a
scalar-affine MPFR counterexample.

The first Flow*/Torch schedule split is now causally observed at
`t=0.18187433604506256`: their transformed polynomial coefficients agree at
roundoff scale, but the raw candidate Picard remainder already makes Flow*'s y
subset fail while Torch passes. Polynomial, endpoint, and right-map swaps
preserve the receiving validator's decision.

One generic experimental improvement was implemented: exact complete
polynomial endpoint carry. It is sound and batch-generic as a carry primitive,
but is **rejected**: every independent T=.1/.5/1/4/6/6.5/7.5/10 request stops at
`t=0.04345468750000001`. It is not the default.

Start with:

- [research direction](docs/RESEARCH_DIRECTION_20260810.md);
- [three-lane contract](docs/THREE_LANE_ALGORITHM_CONTRACT_20260810.md);
- [native baselines](docs/NATIVE_FLOWSTAR_DIFFREACH_TORCH_BASELINE_20260810.md);
- [fixed-support equivalence](docs/TORCH_DIFFREACH_FIXED_BASIS_EQUIVALENCE_20260810.md);
- [causal divergence](docs/VDP_FLOWSTAR_TORCH_CAUSAL_DIVERGENCE_20260810.md);
- [candidate result](docs/GENERIC_TORCH_TM_IMPROVEMENT_RESULT_20260810.md);
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
