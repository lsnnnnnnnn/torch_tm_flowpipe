# Public artifact governance audit

Status: **historical lineage BLOCKED; parentless clean review lineage authorized.**

The public `torch_tm_flowpipe` history already contains controller bytes and raw
outputs copied from the Xiangru/CROWN-Reach workflow.  The audit found no
repository-root license in either the Torch repository or the frozen Xiangru
checkout that grants redistribution of those controller bytes.  A publicly
reachable upstream repository is not, by itself, a redistribution license.

After this source audit, the repository owner explicitly selected the clean
publication option.  The review branch is independently initialized and does
not make any object from the authorization-unknown source history reachable.
It contains only the reviewed native code, tests, sanitized aggregate
artifacts, and reports; it is not a license grant for excluded assets.

## Findings

- Commit `438ee68fd71fa6182eb66cac17229e20dd3cb7d3` first added one copy of the
  transformed TORA controller.
- Commit `c692173e399272a3602a6abea4f24c0728e4306a` added two more copies, the raw
  Q3 result, source manifest, commands, environment records, and logs.
- All three tracked ONNX paths contain the identical 168,486-byte object.  Its
  SHA-256 is `bb80479ce51b6f2558ac4a47cae2831ff3f49275ffaf7b1b874adf3c3b14703e`.
- The source/original controller used to generate that transformed model has
  SHA-256 `52a50c6bc6b1b45b89319edc809cd4d3baca06c32f9fd2ddceb2e95007414418`.
- Logs, manifests, and command records disclose machine-specific paths and
  preserve substantially more private raw evidence than a public aggregate
  result needs.
- No new TORA controller, checkpoint, raw per-leaf trace, private source, or
  unsanitized log is placed under `outputs/tora_q3_native_matched_20260806/`.
  New runners accept external assets through `TORA_CONTROLLER_PATH` or an
  explicit trace path and verify a declared SHA-256.

The historical inventory is in `public_artifact_inventory.csv`; origin and
authorization decisions are in `license_and_origin_map.csv`. The Phase 0
scanner now reads every tracked working-tree file, inventories every untracked
file, and scans every blob/path reachable from clean-lineage `HEAD`. It also
inventories sensitive suffixes, large files, and high-entropy candidates.
Exact matches and paths remain under the private evidence root; the public tree
receives sanitized counts, policy metadata, and a private-log hash only.

## Frozen boundary

Public Git may contain implementation code, schemas, SHA-256 identifiers,
aggregated comparison tables, and sanitized reports.  The following remain
outside Git:

- controller/checkpoint bytes;
- raw Xiangru and Torch per-leaf segment traces;
- the observation-only private worktree patch;
- server paths, full commands, environment dumps, and raw logs.

The detached source worktree intentionally has no descendant branch or commit.
The point-in-time no-push statement from that source audit was superseded only
by the owner's explicit request for a separate parentless clean review branch.
Work still must not continue as a public descendant of `c49d74bb...` until the
existing-history authorization question is resolved.

## Remediation plan requiring owner approval

1. Obtain written redistribution authorization covering the exact controller
   hashes and raw evidence, then record its scope and source.
2. If authorization is denied or unavailable, use the selected clean orphan
   review branch containing only reviewed code and sanitized aggregate
   artifacts; do not merge it into the blocked lineage.
3. Separately decide whether already-published history must be rewritten with
   a reviewed `git filter-repo` path/object list.  Such a rewrite would require
   coordinated force updates and clone invalidation; it will not be performed
   without explicit user approval.
4. Run the complete clean-lineage history and whole-tree artifact scan before
   every public push. Intentional hermetic sandbox mounts are enumerated by
   exact literal and reason; the scanner does not construct strings to evade
   its own patterns. Any unallowlisted match stops publication.

No existing remote branch was deleted, rewritten, or force-pushed.  The only
new remote ref is the explicitly requested parentless clean review branch.

## Performance-closure publication result

The performance/closed-loop closure continues exclusively on the parentless
lineage. Its whole-tree and every-reachable-blob scanner reports
`PASS_CLEAN_LINEAGE`, with zero unallowlisted path/credential matches, zero
current-tree sensitive-suffix candidates, and zero high-entropy candidates.
The aggregate scan record is
`outputs/tora_q3_perf_closure_20260806/provenance/public_artifact_scan_summary.json`;
raw matches and command logs remain private.

Only sanitized aggregates were added for profiler iterations, runtime repeats,
R1/R2 lifecycle replay, shadow lanes, and hierarchical full-loop gates. The
R1/R2 public file is regenerated from a hash-verified private snapshot and
contains no per-leaf endpoint/tube arrays. The original controller and observed
trace remain external and are referenced only by expected SHA-256.

Both public manifest locations cover the same complete tracked tree and exclude
all `manifest.sha256` files to avoid circular dependencies:

- `outputs/tora_q3_perf_closure_20260806/manifest.sha256`;
- `outputs/tora_q3_native_matched_20260806/manifest.sha256` (compatibility path).

Publication remains fail closed: any mismatch, newly tracked sensitive suffix,
unallowlisted scanner match, or private/raw path in a public aggregate blocks a
subsequent push.
