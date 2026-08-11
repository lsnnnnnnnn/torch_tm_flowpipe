# Evidence integrity corrections

Date: 2026-08-11

## Outcome

Source-derived verification replaces hardcoded package passes; historical
fresh-clone claims are qualified rather than repeated.

## Eligibility

Eligible for integrity claims when every named source and SHA validates;
numerical solver claims retain their own scope.

## What is comparable

Claim status, source paths/hashes, command exit, scope, and limitations.

## What is unavailable

The historical package is absent from the Git tree at `2cb647cd...`. Its
server-local `14_fresh_clone/` directory was produced with every command
running at the packager's `ROOT`; the builder itself did not clone `origin`
into a new `mktemp -d` directory. Those rows are source-worktree checks, not a
true-clone claim. A new package is eligible only after separate H1/H2/H3
remote-clone gates.

## Negative results

Missing sources produce `not_run`/`unknown`; SHA mismatch produces `fail`.

## Exact evidence paths

`src/torch_tm_flowpipe/evidence_verification.py`, focused integrity tests, and
the tracked package produced by the full-horizon closure round. The ignored
`20260811T100304Z` path is not currently a Git-tree evidence path.

This document corrects claim provenance without rewriting the immutable S1
evidence run at
`outputs/s1_boundary164_causal_guarded_carry_20260811/20260811T033447Z`.
The numerical S1 result remains
`S1_REACHES_TERMINAL_BUT_DOES_NOT_CLOSE_IT`; the corrections below concern how
verification and portability claims are justified.

## Audit register

| claim | old source | actual evidence | status | correction |
|---|---|---|---|---|
| final fresh clone | old `verification.json` and `handoff.md` | one selected-test clone was recorded at `b5ba3200901e331f01343c7d05608a1d542dbb8c`; it is not the later `7b880d0bf6ea2f6182faaaff1c267f1e2ab2c06a` branch tip and the old package contains no raw command bundle for an independent derivation | overstated | treat the b5ba run as historical only; rerun install, full tests, package verification and clean-tree checks together at the final HEAD |
| private path gate | old `verification.json` | four JSON-family files retain `/srv/local/shengenli/...` in frozen-schedule/checkpoint provenance | qualified | report `private_path_present=true`, classify those occurrences as provenance, and independently prove that packaged replay inputs are root-relative and self-contained |
| baseline/final tests | packager constants | the old package has a derived test summary but no command, stdout, stderr, exit code and timestamps from which the counts can be independently derived | not independently derived | command runners must retain all six source files; missing sources produce `not_run` or `unknown` |
| Fraction oracle support | packager constant | the old packager wrote `true`; support must instead come from an exact-oracle runner result plus source SHA | not independently derived by packager | consume a source-derived claim or report `unknown/not_run` |
| ordinary/structured nonlinear interactions | packager constant | implementation/tests exist, but the old decision field was not derived from a named hashed result | not independently derived by packager | consume a source-derived interaction-coverage claim or fail the candidate authorization gate |
| prefix formal eligibility | canonical documents | `false` | correct | retain the narrow classification |
| primitive formal eligibility | canonical documents | CPU outward image for given binary64 coefficients | correct if qualified | retain this exact scope; do not promote it to a complete-solver proof |
| S1 result | old canonical `RESULTS/STATUS/LIMITATIONS` | the corrected carry replays 307/307 frozen accepted steps, then the unchanged historical terminal step rejects | stale | supersede `S1_PREFIX_REJECTS_BEFORE_TERMINAL` as the current headline while preserving it as historical process evidence |

## Source-derived verification contract

Every verification claim uses
`torch_tm_flowpipe_source_derived_verification_v1` and contains:

```text
claim_id
status = pass | fail | not_run | unknown | qualified
source_paths[]
source_sha256[]
command
exit_code
started_at
finished_at
derived_by
derivation_version
scope
limitations[]
```

`pass` and `qualified` are invalid without at least one source path and matching
SHA256. Missing command evidence is never converted to success. An expected
source SHA mismatch is a failure. Source paths in a portable package are
package-root-relative.

The accepted-state replay gate is separately frozen as
`torch_tm_flowpipe_adaptive_fixed_state_equality_v1`. It records shape, dtype,
device, hexadecimal float values, raw bytes and hashes for the natural accepted
state and the fixed-h rerun. The default and currently registered relation is
bit equality for every field; there are no implicit outward-containment
exceptions.

## Immutable old-package anchors

- `manifest.json`: `485b24d0b63badf0833b264514a45bb58a7447c429c7ec9f3cbc83f4223af9e6`
- `SHA256SUMS`: `ee2ff2c0feafa16e7603257e14c3d307c01fca50501ec9db78981c7768a71148`
- `verification.json`: `9cb7cbeeb01629a358a2fcd14fa3b54356c2e6f40f99260afa4ff75b8b248b9f`

These files remain historical and are not regenerated in place.
