# Three-tool pairwise comparison

> Superseded historical bridge report. The current full-horizon outcomes are
> `FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_ONLY` and
> `DIFFREACH_TORCH_DR7_FULL_HORIZON_DIVERGED`; see
> `THREE_TOOL_PAIRWISE_STATUS_20260811.md`.

Date: 2026-08-11

## Outcome

Overall outcome: `PAIRWISE_COMPARISON_PARTIAL`.

## Eligibility

Native rows are capability-only; matched rows are pairwise-only; diagnostic
rows are not ranking eligible.

## What is comparable

Flow*/Torch same-prestate O4 candidates and DiffReach/Torch explicit-f64 DR7
operators under their frozen contracts.

## What is unavailable

Universal ranking, Flow*/Torch native T10 same-object tightness, and a matched
cross-tool timing ratio.

## Negative results

Flow*/Torch remains partial and no Torch improvement is evidence-authorized.

## Exact evidence paths

`outputs/three_tool_matched_divergence_fixed_support_20260811/20260811T100304Z/`
directories `07_flowstar_torch_raw_remainder/`,
`08_schedule_validator_matrix/`, `10_bridge_ladder/`, and
`11_pairwise_tables/`.

This study freezes separate native-capability, matched-pair, and diagnostic
tracks.  It does not construct a transitive three-tool ranking.

## Native capability table

| lane | native result | available object | numerical scope |
|---|---|---|---|
| Flow* O4 B1 adaptive | completed T10, 290 accepted segments | segment and prefix tubes | pinned build is ineligible as a formal oracle after reproduced scalar-affine under-enclosure |
| DiffReach DR7 B64 fixed | completed T10 | endpoints; stock full-step tube unavailable | empirical; stock full driver has mixed builder dtypes |
| Torch complete O4 B1 adaptive | partial, highest validated `6.397083942944808` | endpoint, last segment tube, prefix tube stored separately | fail-closed ordinary Torch lane |
| Torch fixed DR7 B64 fixed | completed T10 | endpoint, segment tube, prefix tube, J/Phi carry | empirical ordinary float64; one-step 2-ULP companion replay separately qualified |

These rows answer capability and availability only.  Their partitions,
representations, validators, schedules, output objects, and soundness classes
differ.

## Matched pair outcomes

- Flow* / Torch complete O4: `PAIRWISE_COMPARISON_PARTIAL`.  Common-basis,
  first split, expression-tree root cause, same-prestate validator matrix, and
  T1 schedule diagnostics close.  Full-horizon same-object tightness and a
  matched timing ratio remain unavailable.
- DiffReach / Torch DR7:
  `DIFFREACH_TORCH_DR7_OPERATOR_EQUIVALENCE_CLOSED` and
  `DIFFREACH_TORCH_DR7_FULL_HORIZON_PAIRWISE_PENDING`. Explicit-f64 one-step
  operator semantics are bit-exact; the stock mixed-dtype full driver remains
  a separate native row, and Torch self-parity does not close cross-tool J/Phi
  carry.

Diagnostic factorial and bridge rows are marked
`diagnostic_only=true` and `formal_ranking_eligible=false`.  Pairwise closure
does not imply `Flow* > Torch > DiffReach`, a universal fastest tool, or a
universal tightest tool.  The required G3 bridge outcome is
`FIXED_SUPPORT_BRIDGE_BLOCKED`; its failed cells do not invalidate the matched
pair facts above, but they prevent a complete descriptor-bridge claim.
