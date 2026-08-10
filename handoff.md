# Handoff: Flow*/DiffReach/Torch mainline realignment

Date: 2026-08-10
Branch: `codex/torch-tm-flowstar-diffreach-mainline-realignment-20260810`
Run ID: `20260810T025910Z`

## Delivered

- audited Xiangru `2026_experiment` and froze TORA complete-Q3;
- reproduced stock Flow* `b85a3211748cb77b736fe4ad42ee02d8d2b81148`,
  upstream DiffReach `dd628eb443b517d6415de93e7035b4baef73963e`, and
  the Torch complete-O4 lineage;
- added configurable fixed support, precomputed signed routes, exact two-Picard
  construction, every DR-RP mask, endpoint/tube separation, and symbolic carry;
- added a read-only dense Picard observer, an optional stock Flow* causal hook,
  common affine coordinate transforms, and six stage counterfactuals;
- implemented exactly one generic candidate: complete retained endpoint
  polynomial carry, including dense batch-generic exact cloning;
- ran independent one-step, short/medium/full horizon, CPU/V100 B1…512, and
  cold/warm evidence.

## Decisions

The fixed-support lane is qualified against explicit-f64 DiffReach operations.
The first Flow*/Torch decision split is the raw candidate Picard remainder,
before roundoff. The complete carry is `CANDIDATE_REJECTED`: all requested
horizons stop at `0.04345468750000001`; it remains opt-in and non-default.

The native baseline table is not ranked. Stock Flow* is formal-comparison
ineligible after the scalar-affine gap; stock DiffReach and ordinary Torch CUDA
remain empirically sampled; the Torch complete baseline is outward by its
declared interval construction but does not complete T10.

## Evidence map

- provenance: `outputs/mainline_realignment_20260810/20260810T025910Z/00_provenance/`
- native: `.../01_native_baselines/`
- fixed support: `.../02_fixed_support/`
- causal: `.../03_flowstar_causal_divergence/`
- candidate: `.../04_generic_carry_candidate/`
- scaling: `.../05_batch_scaling/`
- final baseline: `.../06_final_baseline_ladder/`
- root machine tables, figures, manifest, and `SHA256SUMS`: run root

## One next step

Implement a bounded, fixed-shape structured-symbol overflow carry for the
terminal `integration_overflow` and `polynomial_truncation` terms. Use
deterministic capacity/eviction with sound interval collapse, then first replay
the immutable `t=6.397083942944808`, `h=0.003623635847674574` pre-state. Promote
it to a horizon sweep only if that unchanged checkpoint closes or the critical
y remainder improves by at least 20% without x regression. Do not retry full
polynomial carry, range ordering, smaller h, or a benchmark-specific formula.

## Final verification

- pre-change baseline: `441 passed, 2 skipped in 60.12 s`;
- final full suite: `469 passed, 2 skipped in 60.87 s` (an immediately prior
  identical pass took `72.85 s`);
- focused package/carry/common-basis gate: `21 passed`, plus package prefix
  regression `3 passed`;
- `python -m compileall -q src experiments tests`: pass;
- `git diff --check`: pass;
- all required machine files and figures: present;
- recursive checksum and remote/clean identities: verified by the final
  delivery commands and reported with exact SHAs in the final response.
