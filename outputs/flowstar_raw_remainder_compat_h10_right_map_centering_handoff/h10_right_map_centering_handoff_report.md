# h10 Range-Midpoint Handoff Continuation Audit

The frozen replay uses the constant-adaptive accepted h sequence exactly, then continues from the replayed state with the standard Flowstar-compatible adaptive policy.

## Decision

- Decision: `centering_effect_does_not_survive_continuation`.
- Replay h modified count: `0`.
- Handoff first h_try: `0.0039859994324420315`.
- Continuation reached t: `6.4065548591418384`.
- Extension vs pure range adaptive: `0.015623749460241143`.
- Minimum accepted continuation margin: `7.1466828532850411e-07`.
- Accepted raw-target violations: `0`.
- Terminal raw-target rejection: `true`.
- Width improvement at handoff vs pure range: `0.037533661633505563`.
- Final common width improvement vs pure range: `0.024816808012121078`.
- Recommendation: Next task should stop tuning centering and inspect the full-step polynomial-range operation ledger around t=6.2..failure.

## Formatting

| path | physical lines | csv.reader rows | status |
| --- | --- | --- | --- |
| outputs/flowstar_raw_remainder_compat_h10_right_map_centering_handoff/h10_right_map_centering_handoff_attempts.csv | 342 | 342 | ok |
| outputs/flowstar_raw_remainder_compat_h10_right_map_centering_handoff/h10_right_map_centering_handoff_segments.csv | 338 | 338 | ok |
| outputs/flowstar_raw_remainder_compat_h10_right_map_centering_handoff/h10_right_map_centering_handoff_summary.csv | 2 | 2 | ok |
| outputs/flowstar_raw_remainder_compat_h10_right_map_centering_handoff/h10_right_map_centering_handoff_report.md | 27 |  | ok |
| outputs/flowstar_raw_remainder_compat_h10_right_map_centering_handoff/h10_right_map_centering_handoff_decision.txt | 1 |  | ok |
