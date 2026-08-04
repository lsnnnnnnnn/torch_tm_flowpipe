# Torch VDP T=10 long-horizon audit

## Outcome

Torch did not reach T=10 in the validated sparse Flowstar-compatible protocols. No partial prefix is labeled complete and no failed width/runtime row enters a ranking.

| backend | lane | completed_horizon | validation_status | soundness_level | primary_eligible | accepted | rejected | final segment width sum | runtime s |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| official-stock | native_reproduction | 10 | completed | formal_outward_rounding | false | 290 | unavailable | 0.704713 | 1.07 correctness wall |
| torch-sparse constant center | native_reproduction | 6.31729088 | validation_rejected | safeguarded_float64_not_fully_proved | false | 308 | 48 | 4.18646949 | 245.259 |
| torch-sparse range-midpoint center | native_reproduction | 6.39093111 | validation_rejected | safeguarded_float64_not_fully_proved | false | 293 | 46 | 4.23646226 | 291.151 |
| torch-sparse range midpoint on constant schedule | matched_plant_backend | 6.07366175 | timeout | safeguarded_float64_not_fully_proved | false | 255 | 0 | 1.89087962 | 300.422 |
| torch-sparse constant on midpoint schedule | matched_plant_backend | 0.43677646 | validation_rejected | safeguarded_float64_not_fully_proved | false | 29 | 1 | 0.82403352 | 34.904 |

The constant-center terminal attempt starts at `t=6.3172908799`, tries `h=0.0039859994`, has finite raw residual width sum `2.0160638e-4`, polynomial range width sum `4.2424016`, negative target margin `-5.11167e-6`, and fails `subset_result`. The range-midpoint attempt starts at `t=6.3909311097`, tries `h=0.0038168663`, has raw residual width sum `2.2844834e-4`, polynomial range width sum `4.2947488`, margin `-1.79427e-5`, and fails the same self-map predicate.

## Single-factor diagnosis

All accepted segments had finite enclosures, zero accepted raw-target violations, and no silent fallback. Range-midpoint centering extends the validated horizon by only `0.07364023`, below the preregistered material threshold `0.5`; the decision is `h10_not_reached_no_material_improvement`. It produces a common-prefix width saving, but 26 common-time checkpoints worsen and the terminal failure remains a raw remainder target rejection.

Cross-schedule replay further rules out centering as a sufficient cause: range midpoint on the constant schedule times out before the schedule ends, while constant centering on the midpoint schedule fails at `t=0.4368`. This indicates a coupled schedule/reset state, not a one-line centering fix.

The earliest recorded ratio checkpoint where Torch's segment width exceeds twice the time-aligned Flowstar segment is about `t=4.205867`: constant-center ratio `2.6269`, midpoint ratio `2.3353`. Around `t=6.225303`, the ratios are `6.2495` and `5.0880`. The dominant current blocker is accumulated polynomial-range and raw-remainder growth leading to self-map rejection; cutoff-only, nonfinite, accepted raw-target violation, and simple right-map centering explanations were excluded by the trace.

The separate order-2 attempt-aligned trace finds its first numeric divergence in the Picard residual at `h=0.025` (Flowstar width sum `0.01043966`, Torch `0.00454062`, both reject). It is not used to explain later order-4 fields independently.

## Protocol coverage

- Flowstar schedule replay: the official accepted time intervals are available from plot segments, but the field-level official schedule/carry trace gate is false. Older replay rows that became nonfinite before T=1 are diagnostic only.
- Matched fixed step: preregistered `h=0.005` now passes the aligned one-step status/support test, but the required internal Picard field gate and exporter sanity gate do not pass; a T=10 fixed run was therefore not promoted.
- Native end to end: the current constant and range-midpoint adaptive runs are authoritative failures with replayable checkpoints.

Raw evidence and minimal replay command:

```bash
OMP_NUM_THREADS=1 conda run -n py11 python \
  experiments/flowstar_raw_remainder_compat_h10_right_map_centering.py \
  --horizon 10 --wall-cap-s 300 \
  --flowstar-segments outputs/three_tool_reaudit/20260804T060058Z/raw/flowstar_official_generated_parity/original_flowstar/original_flowstar_segments.csv \
  --out-dir outputs/three_tool_reaudit/20260804T060058Z/vdp_t10/h10_right_map_centering
```

The summary, 980 attempts, 1,178 segment rows, 40 checkpoints, cross-schedule rows, and decision report are committed under that directory.
