# Flow* Raw Remainder Compat Short Horizon

This is a short-horizon diagnostic only. It does not run h10, add NNCS/GPU work, add symbolic queue variants, change defaults, or claim Flow* parity.

## Scope

- Horizon: `0.5`
- PyTorch ODE spelling: `y - x - x^2*y`
- Sample containment: endpoint checks for four initial corners and the center using RK4 samples.

## Answers

- Does compat follow Flow* accepted h schedule more closely than current? `yes`.
- Does compat remain sample-contained? `yes`.
- Does compat become too conservative and stop too early? `no`.
- Does compat improve or worsen width relative to current? `worsens`; ratio `1.0108208813155113`.
- Should next step be h5, or should compat mechanism be revised? `h5`.

## Summary

| mode | status | reached_t | accepted_steps | rejected_attempts | schedule_distance | sample_contained | final_width_sum | width_ratio | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| probe_schedule | available | 0.5 | 34 | 7 | 0 |  |  |  | existing Flow* probe accepted schedule; no h10 rerun |
| current_no_queue | completed | 0.5 | 24 | 4 | 0.32093427702025074 | true | 0.58576095702694309 | 1 | endpoint samples from corners and center contained in PyTorch final segment boxes |
| flowstar_raw_remainder_compat | completed | 0.49999999999999978 | 27 | 3 | 0.21596859108339828 | true | 0.59209940682219198 | 1.0108208813155113 | endpoint samples from corners and center contained in PyTorch final segment boxes |
| compat_vs_current | closer_to_flowstar |  |  |  |  | true | 0.59209940682219198 | 1.0108208813155113 | current_distance=0.32093427702025074; compat_distance=0.21596859108339828 |
| step_policy_audit_gate | required_before_h5 |  |  |  |  | true |  |  | audit Flow* accept-step growth policy before any h5 run |

## Outputs

- `outputs/flowstar_raw_remainder_compat_short_horizon/short_horizon_summary.csv`
- `outputs/flowstar_raw_remainder_compat_short_horizon/short_horizon_report.md`
