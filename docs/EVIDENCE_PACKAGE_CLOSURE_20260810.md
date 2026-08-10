# Previous evidence package closure

Date: 2026-08-10

## Outcome

The fresh-clone gap at `05ae30b4e41c7b77e39778629ada084623b7270b`
was confirmed and closed from real server-resident evidence. The Git tree at
that SHA contained native files and selected fixed-support files only, while
the source worktree's canonical run contained 851 checksum-verified files,
including groups 03–08, root machine tables, figures, `manifest.json`, and
`SHA256SUMS`.

No report value was reconstructed. The recovery utility hashes each original,
requires existing destination bytes to agree, records original size and
SHA256, and stores text traces of at least 5 MiB as gzip streams with zero
mtime and no source filename. The immutable source tree contains 329,693,657
bytes; deterministic storage reduces the recovered tree to 149,293,272 bytes
before the recovery inventory itself.

## Authoritative paths

- [canonical run](../outputs/mainline_realignment_20260810/20260810T025910Z/)
- [recovery inventory](../outputs/mainline_realignment_20260810/20260810T025910Z/00_provenance/evidence_recovery.json)
- [package manifest](../outputs/mainline_realignment_20260810/20260810T025910Z/manifest.json)
- [root checksums](../outputs/mainline_realignment_20260810/20260810T025910Z/SHA256SUMS)

The recovery source is identified publicly by worktree label and source commit,
without embedding an absolute server path. Original digests remain in the
inventory even when a large raw text artifact is stored compressed. Generated
tables and figures are labeled `derived-regenerated` because their authority is
the committed raw evidence and deterministic builder, not their pre-recovery
bytes.

## Closed inconsistencies

- The missing 03–08 directories and root package files are now part of the
  repository package rather than merely present in an ignored worktree.
- Validated horizons in figures come from generated horizon tables, whose rows
  in turn read raw summaries.
- Native rows, terminal decisions, causal decisions, soundness rows, claim
  statuses, and external SHAs are derived from raw machine files.
- Nested artifact manifests and local checksum manifests are verified before a
  package is built.
- JSON output rejects NaN and infinity; required paths and required row fields
  fail closed.
- Public root machine files are rejected if they contain an absolute
  `/srv/local/` path.
- The authoritative `tmv_right` hash ends in `...ed3bb3`.

## Fresh-copy regression

`tests/test_artifact_package.py` copies the complete canonical run into a clean
temporary directory, verifies recovery provenance, every nested manifest, and
the root-prefixed checksums, rebuilds all required machine files and figures,
and requires byte identity with the committed derivatives. It also checks that
all canonical artifact references in the active reports resolve and are known
to Git, and explicitly fails if required groups 03–08 disappear.

The portable test does not execute any numerical reachability job.

## Current-round package

The follow-on closure package is at
[20260810T070908Z](../outputs/structured_remainder_compiled_fixed_support_20260810/20260810T070908Z/).
Its builder reads frozen object summaries plus current functional, compiled,
outward, structured-terminal, and generality raw artifacts; it excludes
compiler caches and the source-mixed exploratory matrix. `raw_public/` is a
deterministically sanitized copy with regenerated nested manifests, while the
root tables and ten figures are derived. The root-prefixed `SHA256SUMS` covers
the committed package surface.
