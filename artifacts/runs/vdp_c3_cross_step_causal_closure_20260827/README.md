# VDP C3 remote evidence closure

Highest recomputed status: `CROSS_STEP_CAUSE_IDENTIFIED__C3_PRODUCTION_GATE_PASSED__NATIVE_T10_REACHED`.

All values below are recomputed by `scripts/verify_vdp_c3_remote_evidence.py` from
the package-local raw CSV/JSON/JUnit XML. No external raw root is required.

## Gates

- `fixed_runs_complete`: `True`
- `T1_T3_no_regression`: `True`
- `T6p32_recovery`: `True`
- `runtime`: `True`
- `native_horizon_and_counts`: `True`
- `source_shas`: `True`
- `tests`: `True`

## T6.32 recovery

- endpoint_x: 0.720181899168491
- endpoint_y: 0.847178663466812
- tube_x: 0.71991365408499
- tube_y: 0.846920482533309

## Native outcome

- C2: T=6.71491466960718, 233 accepted / 37 rejected
- C3 SR100: T=10, 246 accepted / 35 rejected

## Independent verification

```bash
python scripts/verify_vdp_c3_remote_evidence.py
python scripts/verify_vdp_c3_remote_evidence.py --run-tests
```

`--run-tests` additionally reruns pytest and compares testcase identities and counts
with the committed JUnit XML. The ordinary verifier never trusts `RESULT.json`; it
recomputes that file and the highest status from raw evidence.
