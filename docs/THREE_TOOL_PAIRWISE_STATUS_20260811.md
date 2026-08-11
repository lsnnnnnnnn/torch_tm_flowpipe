# Three-tool pairwise status

Date: 2026-08-11

## Outcome

There is no three-tool winner. The closed statuses are:

- Flow*/Torch: `FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_ONLY`;
- historical one-step DR7 operator:
  `DIFFREACH_TORCH_DR7_OPERATOR_EQUIVALENCE_CLOSED`;
- DiffReach/Torch: `DIFFREACH_TORCH_DR7_FULL_HORIZON_DIVERGED`;
- complete-O4 carry: C4 `CARRY_MISSING_SYMBOLIC_SEMANTICS`;
- fix decision: `NO_FIX_AUTHORIZED`.

## Eligibility

Native capability, Flow*/Torch common-prefix, and DiffReach/Torch explicit-f64
full-horizon results remain three separate tables. Formal and empirical rows,
different partitions, different representations, and endpoint/tube objects
are never merged into a ranking.

## Contract

Table N records native capability. Table M-F uses B1 complete-O4 fixed
`h=0.01` only through the 632-step common prefix. Table M-D uses B64 DR7
explicit-f64 through 1,000 steps. Every output object and soundness scope is
named.

## What was actually run

Flow* and Torch complete-O4 fixed schedules, upstream DiffReach and Torch DR7
explicit-f64 full traces, stock DiffReach capability, and the A3/A4 carry
causal study were executed. Five causal figures and three independent tables
were derived from those artifacts.

## Exact results

Flow* reaches T10 while matched Torch stops after 632 fixed steps. Both
explicit-f64 DR7 lanes reach T10 with equal masks but diverge numerically from
step 1 and fail J/Phi and endpoint/tube closure. A4 reproduces failures at
3.19/3.33; C4 is selected and no fix is authorized.

## What is comparable

Only fields within Table N, M-F, or M-D under that table's declared contract.
Source CSVs accompany every figure.

## What remains unavailable

Universal fastest/tightest results, matched performance ratios, a three-tool
common output contract, formal cross-tool closure, GPU timing after the DR7
CPU divergence, and an authoritative dense CNI parity result.

## Negative results

The results do not imply `Flow* > Torch > DiffReach`, or any permutation of
that ranking. Earlier one-step DR7 closure does not override full-horizon
divergence. Earlier A3 T10 completion does not authorize a carry replacement.

## Limitations

Flow* build qualification is open; Torch and JAX lanes use ordinary float64;
stock DiffReach is mixed-builder-dtype and endpoint-only; A3/A4 are empirical
carry diagnostics.

## Evidence paths

Final package target:
`outputs/three_tool_full_horizon_pairwise_carry_closure_20260811/20260811T191549Z/`.
Primary tables are in `12_pairwise_tables/`; causal figures and source CSVs are
in `13_figures/`.

## Reproduction commands

```bash
python experiments/build_full_horizon_pairwise_tables.py --help
python experiments/build_full_horizon_pairwise_figures.py --help
python experiments/verify_full_horizon_pairwise_package.py --help
```

## Next authorized action

Finish H1/H2/H3 package delivery. Scientifically, the unique next question is
the independently specified authoritative complete-O4 symbolic carry
contract; parameter tuning and ad-hoc narrower carries remain prohibited.
