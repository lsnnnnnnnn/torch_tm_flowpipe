# VDP generic-refactor zero regression

This package contains deterministic gzip copies of both the frozen reference and
post-refactor candidate CSVs. The verifier recomputes all hashes, tolerance checks,
source provenance, horizons, and accepted/rejected counts from package-local raw data.

```bash
python scripts/verify_vdp_generic_refactor_regression.py
```

Maximum observed C3 numeric delta: `0.0`.
