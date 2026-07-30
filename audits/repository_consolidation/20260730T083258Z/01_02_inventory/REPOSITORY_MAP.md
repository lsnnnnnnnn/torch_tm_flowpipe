# Repository map before consolidation

Target tree: `origin/codex/torch-flowstar-diffreach-deep-study` at
`9a684d9106633e067bfac0747244b769fa49aa0b`.

The tree has 1,576 tracked files and 316,962,163 tracked bytes. The artifact
inventory identifies 1,270 generated or historical files using 313,185,282
bytes. In other words, almost all tracked bytes are outputs rather than active
source.

| top-level path | purpose / owner | entrypoint and references | tests | current status | decision |
| --- | --- | --- | --- | --- | --- |
| `.DS_Store` | Finder metadata | none | none | generated noise | remove from active tree and ignore |
| `.gitignore` | repository ignore policy | Git | n/a | canonical support | keep and strengthen artifact/cache rules |
| `README.md` | current entrypoint | researchers and developers | command replay in final checks | contradictory and server-specific | replace with one local canonical entrypoint |
| `src/` | `torch_tm_flowpipe` core | package API | root `tests/` | one flat canonical implementation plus optional batched prototype | keep the core implementation; document logical boundaries and isolate optional prototype status |
| `comparisons/` | old Flow* adapter/harness | old comparison CLIs | root tests | historical supported harness | migrate essential adapter behavior behind the canonical experiment path; mark legacy |
| `experiments/` | many generations of runners, adapters, reports, and committed results | 100+ Python/shell entrypoints | root and nested tests | heavily duplicated/mixed | retain one supported consolidation runner/analysis path; archive historical generations |
| `outputs/` | historical generated evidence | old reports only | several tests reference formats | 919 files / 105.7 MB | remove from active tree after archive tag; retain only compact registry/manifests |
| `docs/` | algorithm, limitations, and many intermediate audit notes | README links | documentation gates | 27 documents with overlapping historical states | consolidate active docs; move necessary lineage evidence to `docs/history/` |
| `examples/` | three package examples | direct Python execution | final smoke | small and active | keep |
| `scripts/` | setup, checks, and historical study orchestration | shell | syntax/final gates | overlapping entrypoints | keep one check path and one formal path; mark or archive historical scripts |
| `tests/` | core, regression, and experiment tests | pytest | self | 40 files on candidate tree | keep core/protocol tests; remove artifact-dependent tests from the default unit path |
| `pyproject.toml` | package and pytest configuration | build tools | installation gate | underspecified markers | keep and add explicit test markers |

## Duplication and schema evidence

- `duplicate_files.csv`: 97 exact blob groups with independent SHA256 values.
- `duplicate_symbols.csv`: 221 repeated class/function signatures for manual
  review; many are legitimate local helpers, while collector/plot/report
  repetitions are consolidation targets.
- `entrypoint_inventory.csv`: 125 Python or shell entrypoints.
- `schema_field_inventory.csv`: 561 CSV/JSON shapes, largely because result
  bundles are tracked as if they were source.
- `technical_debt.csv`: 2,063 automated hits. These are audit signals, not
  automatic defects. The dominant categories are runtime assertions,
  hard-coded private/server paths, non-finite handling, broad exceptions,
  silent fallbacks, and string-status truthiness.

## Artifact decision

The old `20260730T015245Z` deep-study bundle is classified
`provisional_due_to_known_protocol_defects` for this consolidation. It remains
recoverable from the original branch and the pre-consolidation archive tag.
It must not be copied forward as a new authoritative run.

The active tree will retain small canonical configs, schemas, source,
independent audit code, compact provenance/registry records, and new accepted
deliverables. Bulk prior outputs will be removed only on the new branch; no
history rewrite is permitted.
