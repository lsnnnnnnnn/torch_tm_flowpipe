# Pre-registered Brusselator generic-core validation

This package contains the only three numerical lanes allowed by
`SECOND_SYSTEM_CONTRACT.md`, the exact 2D Fraction JUnit record, and the
raw inputs needed to recompute every soundness, horizon, divergence,
late-prefix, owner-accounting, and terminal-status field.

```bash
python scripts/verify_brusselator_second_system_evidence.py
```

Observed terminal status: `C3_GENERICITY_SOUNDNESS_GATE_FAILED_STOP`.
