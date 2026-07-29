# Flow* Riccati root cause and repair

## Reproduction

The minimal case is

```text
x' = x^2
x(0) in [0, 0.1]
h = 0.01
order = 2
cutoff = 1e-15
candidate remainder = [-1e-4, 1e-4]
```

Its exact endpoint is
`[0, 0.1 / (1 - 0.1 * 0.01)] =
[0, 0.10010010010010011]`.

The standalone Flow* regression reports:

| path | raw endpoint | analytic containment | endpoint in tube |
|---|---|---:|---:|
| stock | `[-2.5012551320951528e-05, 0.10010008767642772]` | no | yes |
| full-Picard fallback | `[-2.5312800018753841e-05, 0.10010033783129377]` | yes | yes |
| leaf-cache repair | `[-2.5112814178874151e-05, 0.10010018797692960]` | yes | yes |

Thus the stock upper endpoint misses the analytic endpoint by about
`1.242367239e-08`.  This is an under-enclosure even though `advance` returns
success and the exported endpoint is inside the exported tube.

Run the complete machine-readable reproduction with:

```bash
conda run -n py11 python \
  experiments/three_tool_deep_study/flowstar_root_cause.py \
  --output-dir /tmp/flowstar-root-cause
```

It emits all four full CIR records, a stage trace CSV, a summary CSV, and a
JSON gate.  The fourth record repeats stock arithmetic with
`intervalNumPrecision=256`.

## First lost inclusion

The relevant call chain on audit commit `fa39f7a` is:

1. fixed-step/fixed-order `Flowpipe::advance` in
   `flowstar-toolbox/Continuous.cpp:904`;
2. complete first image by `Picard_ctrunc_normal` at line 1058;
3. successful first candidate inclusion at line 1083;
4. cached replay by `Picard_ctrunc_normal_remainder` at line 1154;
5. stock componentwise acceptance of `newRemainders` at lines 1292–1296;
6. return of that refined native remainder at line 1307.

The first complete Picard remainder is
`[-3.2530001875001326e-07, 2.5325331293751586e-05]`, and it is contained in
the configured candidate.  Stock cached refinement narrows it to
`[-2.5051320947700027e-08, 2.5075176427711236e-05]`.  Regenerating the full
Picard image for that refined candidate yields
`[-1.2511387786432351e-07, 2.5175276528407708e-05]`, which is not a subset of
the refined candidate.  Inclusion is therefore first lost when stock accepts
the cached remainder-only proposal, before composition or endpoint export.

## Exact cache defect

The complete expression evaluator handles a `NODE_VAR` at
`flowstar-toolbox/expression.h:1651`.  It copies the variable Taylor model and
degree-truncates it.  For this case the variable-leaf truncation interval is
exactly recorded as
`[-5.0000000000000016e-05, 5.0012500000000016e-05]`.

The stock remainder-only evaluator reaches the matching `NODE_VAR` at
`expression.h:2013` but returns only the variable's existing remainder.  It
does not consume the truncation interval because the stock intermediate-range
stream never stored it.  Subsequent nonlinear multiplication scales this
missing leaf contribution; the final discrepancy is the approximately
`1e-7` amount that invalidates the refined self-map.

This identifies an algorithmic cache-replay defect, not a wrapper extraction
error, not post-`advance` remainder mutation, and not the native remainder
returned by `advance`.  The polynomial-difference interval is only
`[-1.3010426069826054e-20, 8.6763278853152506e-21]`, so it is not the missing
term.  A 256-bit MPFR repetition returns the same exported stock upper endpoint
to the reported precision and still misses the analytic bound, which is
evidence against default precision as the first cause.

## Repairs

The audit branch provides two independent safe paths:

- `FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION=1` stores the variable-leaf
  truncation interval in the intermediate-range stream at
  `expression.h:1657–1664` and consumes it during remainder replay at
  `expression.h:2019–2024`.  Trace phases
  `cache_leaf_truncation_recorded` and
  `cache_leaf_truncation_replayed` prove the paired stream operations.
- `FLOWSTAR_AUDIT_REVALIDATE_REFINEMENT=1` treats every cached result only as a
  proposal, regenerates a full Picard image at
  `Continuous.cpp:1210–1248`, and commits the vector proposal atomically only
  after inclusion at line 1275.  Otherwise it retains the last completely
  validated remainder.

Neither path overwrites the native remainder after `advance`.  The leaf-cache
repair is minimal; the full-Picard path is the conservative fallback and
cross-check.

## Repository and patch

- upstream/base: `b85a3211748cb77b736fe4ad42ee02d8d2b81148`;
- independent branch: `codex/full-picard-revalidated`;
- final audit commit: `fa39f7ac29d5dc6ca09c2ee3f2d11454f6e6a353`;
- regression: `./tests/run_refinement_revalidation.sh`;
- portable patch series:
  `flowstar_patches/fa39f7a_series/`.

Publishing to the configured upstream remote was attempted and rejected
because this environment has no write credentials.  Apply the portable series
from a clean checkout of the base with:

```bash
git am /path/to/fa39f7a_series/*.patch
```

Original Van der Pol parity to `T=10`, stock/generated schedule parity, and the
full three-tool regression matrix are separate gates; a fixed order-2
configuration rejection is reported as such and is not described as a Flow*
crash.
