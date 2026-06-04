# Target Remainder Sensitivity Report

This is diagnostic only; larger target remainders are relaxed parameters, not Flow* parity.
Does loosening target remainder reach horizon 5? no.
Is 2e-4 enough? no.
Is 5e-4 enough? no.

## Rows

| radius | run_id | status | last_validated_t | final_width_sum | width_vs_1e-4 | rejected_steps |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 0.0001 | flowstar_style_o6_target | failed | 2.1095541733932355 | 9.0341200395994434 | 1.0 | 80 |
| 0.0002 | flowstar_style_o6_target_r2e-4 | failed | 2.177556584500918 | 11.585019911651365 | 1.2823628489405177 | 76 |
| 0.0005 | flowstar_style_o6_target_r5e-4 | failed | 2.2567245519061094 | 14.658672949711871 | 1.6225900126916855 | 67 |

Relaxed target remainders can reduce rejections only if the validated horizon improves without unacceptable width growth.
Do not recommend relaxed remainders as parity unless the report explicitly labels the parameter change.
