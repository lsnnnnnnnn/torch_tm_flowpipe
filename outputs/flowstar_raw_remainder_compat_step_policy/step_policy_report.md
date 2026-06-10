# Flow* Raw Remainder Compat Step Policy

This is an opt-in schedule-policy audit only. It does not run h5 or h10, add NNCS/GPU work, add symbolic queue variants, change defaults, or claim Flow* parity.

## Audited Flow* Policy

- Rejected attempt shrink: `0.5`.
- Accepted step grow: `1.1`.
- Bounds used here: `h_min=0.002`, `h_max=0.1`.

## Answers

- What is Flow* post-accept grow policy? `h_next = min(h * 1.1, h_max)`.
- Does flowstar_step_policy make the accepted h prefix match Flow* for T=0.5? `true`; prefix count `34`.
- Does schedule distance improve vs compat default? `yes`; compat default `0.13932396542335354`, flowstar step policy `9.999778782798785e-13`.
- Sample-contained? `yes`; max violation `0`.
- Stop too early? `no`.
- Width increase material? `no`; width ratio vs compat default `0.969796817471265`.
- Is h5 now justified? `true`; recommendation `flowstar_step_policy_h5_candidate`.

## Summary

| mode | status | reached_t | accepted_steps | rejected_attempts | schedule_distance | prefix_count | sample_contained | final_width_sum | width_ratio_vs_compat | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| probe_schedule | available | 0.5 | 34 | 7 | 0 | 34 |  |  |  |  |
| current_no_queue_default_policy | completed | 0.5 | 26 | 16 | 0.27400334208250954 | 0 | true | 0.58077213542477135 | 0.97300728636476796 |  |
| raw_remainder_compat_default_policy | completed | 0.5 | 32 | 20 | 0.13932396542335354 | 1 | true | 0.5968836447202589 | 1 |  |
| raw_remainder_compat_flowstar_step_policy | completed | 0.5 | 34 | 7 | 9.999778782798785e-13 | 34 | true | 0.5788558590503563 | 0.969796817471265 |  |
| flowstar_step_policy_vs_compat_default | improved |  |  |  |  | 34 | true | 0.5788558590503563 | 0.969796817471265 | flowstar_step_policy_h5_candidate |
| h5_gate | justified |  |  |  | 9.999778782798785e-13 | 34 | true | 0.5788558590503563 | 0.969796817471265 | flowstar_step_policy_h5_candidate |

## Schedule Distances

- Current default policy: `0.27400334208250954`.
- Raw remainder compat default policy: `0.13932396542335354`.
- Raw remainder compat Flow* step policy: `9.999778782798785e-13`.

## Outputs

- `outputs/flowstar_raw_remainder_compat_step_policy/step_policy_summary.csv`
- `outputs/flowstar_raw_remainder_compat_step_policy/step_policy_report.md`
