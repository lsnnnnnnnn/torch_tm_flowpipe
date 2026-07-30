# Artifact policy

Source-controlled by default:

- canonical benchmark/profile files;
- curated formal summaries, eligibility, primary Pareto, figures, reports;
- environment/provenance/config/run manifests;
- essential command/test logs and SHA-256 manifest;
- repository audit inventories.

Not source-controlled by default:

- caches and compiled files;
- temporary Flowstar C++ builds;
- duplicated raw expansions;
- debug dumps and intermediate plots;
- profiler traces;
- `__pycache__` and test caches.

The full historical provisional bundles are removed from the active tree after
their source SHA, classification, replacement, and verified archive tag are
recorded. No history rewrite is used.
