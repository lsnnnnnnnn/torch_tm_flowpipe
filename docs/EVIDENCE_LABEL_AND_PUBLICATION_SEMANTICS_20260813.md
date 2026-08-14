# Evidence-label and publication semantics — 2026-08-13

The prior package is valid only under these narrow labels:

```text
LOSSLESS_STATE_SERIALIZATION_CLOSED
SAME_ENGINE_REPLAY_CLOSED
CROSS_OPERATOR_SAME_PRESTATE_NOT_AVAILABLE
OPERATOR_ATTRIBUTION_OPEN
```

`SAME_PRESTATE_LOSSLESS_BRIDGE_AVAILABLE` did not prove a Flow*-on-Torch or
Torch-on-Flow* operator replay. It meant that the stored schema could preserve
and replay its own engine's state without loss.

The prior scientific commit is
`a8653a7d9ea6f54b1450da6bee9af0e2a5a19695`. Its detached fresh-clone test is
valid evidence. The later branch tip
`3940386a61bdd6edbf3dc1722be031a1da572171` is an attestation commit. The diff
between them does not change `src`, `experiments`, or `tests`; their tree hashes
at both commits are respectively:

```text
src          7be43ed900a99308af24d7dbb13a46d51e1e7280
experiments  79855f638ca9fbfefd568087458c1701166c1062
tests        dca1cc7e3277098c88803c4da417ae72cc005741
```

Accordingly, the publication statement is:

```text
scientific_sha_verified = true
attestation_tip_contains_no_scientific_tree_changes = true
final_tip_fresh_clone_verified = false/unknown
```

The old `publication_tip: null` must not be interpreted as a fresh-clone test
of the final attestation tip. The new package records the scientific SHA and
attestation tip separately and derives publication status from raw Git output.
