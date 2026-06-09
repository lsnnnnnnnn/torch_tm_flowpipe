# Flow* Validation Candidate Decomposition Audit

This is diagnostic-only. It does not change solver behavior, rerun h10, add symbolic queue variants, or claim Flow* parity.

## Scope

- t_before requested: `0`
- h_try: `0.025000000000000001`
- Input traces: `outputs/flowstar_step_trace_compare/*.csv`
- Output ledger: `outputs/flowstar_validation_candidate_decomposition_audit/validation_candidate_decomposition_ledger.csv`

## Answers

- Full-step tube total boxes are width-close: `true`; width ratio torch/Flow* is `0.99985013194593675`.
- Acceptance-critical residual y_hi gap equals the full-step tube y_hi gap: `true`.
- Same-source full-step y_hi delta torch-Flow*: `-4.9559136703347662e-05`.
- Post-cutoff residual y_hi delta torch-Flow*: `-4.9559136709652406e-05`.
- Verdict: `residual_decomposition_mismatch`.
- Exposed component carrying the gap: `raw_ctrunc_residual`.
- Polynomial range component: `same`.
- Ordinary remainder component: `unknown`.
- Raw ctrunc residual component: `differs`.
- Post-cutoff residual component: `differs`.
- Cutoff/polyDiff explains the y_hi gap: `false`.
- Is PyTorch putting width into polynomial range that Flow* puts into remainder: `unknown_needs_remainder_partition`.
- Is Flow* raw no-remainder still missing: `true`.
- Exact field to expose next if attribution remains unknown: any still-blank polynomial_range, ordinary_remainder/picard_no_remainder, raw_ctrunc_residual, or cutoff_poly_diff endpoints for the first differing component.

## Candidate Rows

| source | status | full-step box | target-check residual y | target y | subset y | y_hi margin | domain | center/scale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flowstar | rejected | x=[1.0975301322957665, 1.461604968996091], y=[2.2509745318250358, 2.4781433092829666] | [-8.3561112430831106e-05, 0.0001083283903691475] | [-0.0001, 0.0001] | false | 8.3283903691474946e-06 | physical_tube_over_full_step_tau_domain_before_tau_h_substitution | center=(1.25, 2.4000000000000004), scale=(0.14999999999999991, 0.049999999999999822) |
| torch_noqueue | accepted | x=[1.097532321924902, 1.4616001369967995], y=[2.2510065595898365, 2.4780937501462632] | [-5.1533347624557215e-05, 5.8769253659495094e-05] | [-0.0001, 0.0001] | true | -4.1230746340504911e-05 | physical_tube_over_full_step_tau_domain_before_tau_h_substitution | center=(1.3091480530492146, 2.3302942066425647), scale=(0.15093725982151429, 0.077083451030272407) |
| torch_v2 | accepted | x=[1.097532321924902, 1.4616001369967995], y=[2.2510065595898365, 2.4780937501462632] | [-5.1533347624557215e-05, 5.8769253659495094e-05] | [-0.0001, 0.0001] | true | -4.1230746340504911e-05 | physical_tube_over_full_step_tau_domain_before_tau_h_substitution | center=(1.3091480530492146, 2.3302942066425647), scale=(0.15093725982151429, 0.077083451030272407) |

## Decomposition Notes

- Flow* target-check residual y_hi is `0.0001083283903691475`; PyTorch target-check residual y_hi is `5.8769253659495094e-05`.
- Flow* y margin to target is `8.3283903691474946e-06`; PyTorch y margin to target is `-4.1230746340504911e-05`.
- The exposed Flow* raw ctrunc and post-cutoff residuals differ only by the recorded cutoff width on this row.
- The exposed PyTorch ordinary no-remainder and post-cutoff residuals differ only by the recorded cutoff width on this row.
- Polynomial range endpoints are exposed in the current traces; component status is `same`.
- Raw ctrunc residual construction audit: `outputs/flowstar_raw_ctrunc_residual_audit/raw_ctrunc_residual_report.md`; do not treat the root cause as closed unless that audit completes component attribution.
- Blank component endpoint columns mean unknown, not zero.
- Missing component fields: flowstar.ordinary_remainder_x_hi; flowstar.ordinary_remainder_x_lo; flowstar.ordinary_remainder_y_hi; flowstar.ordinary_remainder_y_lo.
