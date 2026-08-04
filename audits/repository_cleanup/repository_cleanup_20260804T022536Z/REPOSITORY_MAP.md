# Repository map

The exhaustive file-level inventory is `repository_map_inventory.csv`; entrypoints, exact duplicates, symbol-name collisions, benchmark sources, schemas, absolute paths, and technical debt have separate CSVs.

## Canonical active boundaries

- Numerical core: `src/torch_tm_flowpipe/{interval,polynomial,taylor_model,tm_vector,flowpipe,symbolic_remainder}.py`. There is one canonical interval, polynomial, Taylor model, and propagation implementation.
- Protocol core: `src/torch_tm_flowpipe/protocol/` for schema, configuration identity, eligibility, runtime, Pareto, and provenance.
- Benchmark source: `benchmarks/canonical.yaml` plus the versioned `smoke.yaml`/`formal.yaml` selection profiles.
- Active orchestration: `experiments/consolidated_study/cli.py`. It must remain disabled for primary cross-tool claims until backend identity/parity gates are met.
- Supported order-2 diagnostic: `experiments/flowstar_step_trace_compare.py` plus its single C++ probe source.
- Independent analysis: `analysis/independent_audit.py`.
- Active documentation: README and `docs/{PROJECT_SCOPE,ARCHITECTURE,ALGORITHM,EXPERIMENT_PROTOCOL,RESULTS_STATUS,REPRODUCIBILITY,STATUS,ARTIFACT_POLICY,EXTERNAL_PYTORCH_TM_AUDIT_PRECONDITIONS}.md`.

Historical experiment directories retain code/provenance but are not alternate supported runners. Output trees and frozen artifacts are never import sources.

## Mechanical inventory summary

- 548 files were inventoried in the selected base snapshot.
- 242 Python/shell entrypoint-like files were found; only the explicitly documented CLI and order-2 diagnostic are supported entrypoints. The rest require historical/diagnostic context and are not README quick starts.
- 27 exact-content duplicate groups were found. They include frozen copies/checksums and small source fixtures; none is deleted mechanically.
- 262 repeated Python symbol-name groups were found. Most are ordinary local helpers/tests or historical adapters, not proof of duplicate implementations.
- 11 YAML benchmark/config definitions were found. Only the three files under `benchmarks/` are canonical study inputs.
- 132 CSV/JSON/Python schema sources were inventoried.
- 280 absolute-root/environment/implicit-latest matches and 86 debt-pattern matches were recorded for review. Supported runners must use explicit output directories and environment-resolved backends; no supported report may discover a `latest` run implicitly.

The largest tracked file is the 72,039,158-byte historical Flowstar log already preserved as evidence. It is not imported by active code and is not removed because provenance/recovery requirements take precedence over cosmetic size reduction.
