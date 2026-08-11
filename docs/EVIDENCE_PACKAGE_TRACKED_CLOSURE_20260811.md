# Evidence package tracked closure

Date: 2026-08-11

## Outcome

The missing `20260811T100304Z` package has been recovered as a compact tracked
historical package. The new full-horizon package is fail-closed and
hash-complete; its final H1/H2/H3 identities remain pending until the three
true-remote-clone gates finish.

## Eligibility

Integrity claims are eligible only after H2 tracks every package file and an
independent H2 clone passes `SHA256SUMS`, manifest-path, checkpoint-load,
full-test, compile, ancestry, remote-equality, and clean-tree checks. Numerical
claims retain the narrower qualifications in their pairwise reports.

## Contract

The package manifest uses package-root-relative paths, names `tested_source_sha`
without embedding a circular package-commit hash, enumerates every payload
file with bytes and SHA256, and requires exact checksum coverage of every file
except `SHA256SUMS` itself. Large traces, compiled binaries, environments, and
caches are excluded; their hashes remain in copied scientific summaries.

## What was actually run

The historical server-local package was checksum-verified and compactly
recovered. The new builder was smoke-run against the complete raw round and its
verifier loaded 300 NPZ members. The definitive H1 numerical rebuild and H2/H3
clone executions are intentionally not claimed before they occur.

## Exact results

- Historical recovery: all 534 original checksum rows passed; the compact
  recovery is tracked.
- New package schema: `three_tool_full_horizon_pairwise_carry_package_v3`.
- Package outcome registry: Flow*/Torch common-prefix only, DiffReach/Torch
  full-horizon diverged, carry C4, dense parity not expressible, and
  `NO_FIX_AUTHORIZED`.
- Final tested/package/delivery SHAs: pending the H1/H2/H3 gates.

## What is comparable

Package integrity, source ancestry, outcome derivation, file presence, hashes,
JSON finiteness, and checkpoint loadability are comparable across clones.

## What remains unavailable

The final H1/H2/H3 clone logs and remote commit identities remain unavailable
until those gates execute. Excluded large traces are not portable replay
inputs.

## Negative results

The historical `14_fresh_clone` marker was produced from the source worktree
and is not a true-clone result. A missing file, wrong outcome, wrong H1, unsafe
path, incomplete checksum coverage, unloadable NPZ, or untracked package file
causes verification to fail.

## Limitations

The compact package preserves causal and comparison evidence, not every
intermediate trace event. Package integrity does not upgrade empirical
ordinary-float64 results to formal claims.

## Evidence paths

- Historical package: `outputs/three_tool_matched_divergence_fixed_support_20260811/20260811T100304Z/`.
- Final package target: `outputs/three_tool_full_horizon_pairwise_carry_closure_20260811/20260811T191549Z/`.
- Builders: `experiments/build_full_horizon_pairwise_package.py` and
  `experiments/verify_full_horizon_pairwise_package.py`.

## Reproduction commands

```bash
python experiments/build_full_horizon_pairwise_package.py --help
python experiments/verify_full_horizon_pairwise_package.py --help
sha256sum -c outputs/three_tool_full_horizon_pairwise_carry_closure_20260811/20260811T191549Z/SHA256SUMS
```

## Next authorized action

Complete H1, track the compact package as H2, then record H2 and final H3
true-remote-clone audits without changing executable semantics.
