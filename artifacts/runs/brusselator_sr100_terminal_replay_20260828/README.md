# Brusselator SR100 terminal replay closure

This supplemental package preserves the one terminal attempt and both
canonical v5 checkpoints. It recomputes the previously missing rollback
gate without rewriting the original three-lane evidence.

```bash
python scripts/verify_brusselator_terminal_replay_evidence.py
```

Recomputed status: `C3_GENERIC_CORE_VALIDATED__SECOND_SYSTEM_NO_MATERIAL_GAIN`.
