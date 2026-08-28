# Brusselator SR1000 and C4 closure

This package preserves the frozen SR1000 baseline, output-equivalent Flow*
operator traces, the mechanically selected step-one checkpoint, and the one
same-input C4 gate. It derives the non-capacity verdict and the first material
operator divergence from raw records. No C4 native-prefix rerun was performed.

```bash
python scripts/verify_brusselator_sr1000_c4_evidence.py
```

Recomputed status: `BRUSSELATOR_SR1000_OPERATOR_C4_CLOSED`.
