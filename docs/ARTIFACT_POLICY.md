# Artifact policy

Tracked:

- canonical benchmark/profile files;
- versioned schemas;
- curated diagnostic fixtures;
- concise reports and registries;
- selected figures whose source table/hash is explicit;
- provenance manifests and checksums;
- repository audit inventories and essential acceptance logs.

The VDP terminal-range bundle is a deliberate exception to omitting duplicated
raw expansions: it tracks complete segments, attempts, remainder ledgers, and
range traces needed to audit the terminal claim. Large raw CSV/JSONL files are
stored as deterministic gzip (`compresslevel=9`, `mtime=0`). Its manifest
records both uncompressed source hashes/sizes and stored hashes/sizes, while
`SHA256SUMS` covers every committed stored file. Decompression does not change
the evidentiary identity declared in the manifest.

Untracked:

- caches and compiled files;
- temporary binaries and builds;
- duplicated raw expansions;
- scratch logs and intermediate plots;
- profiler dumps;
- implicit `latest` links;
- `__pycache__` and test caches.

Large or duplicated tracked artifacts may be removed only after inventorying
the exact path, recording a recovery commit and replacement, proving active
code/tests do not depend on it, and updating the migration map. No history is
rewritten. Unknown files are investigated rather than hidden with a new
ignore rule. Frozen withdrawn artifacts remain byte-for-byte provenance and
must not be edited to repair a current claim.
