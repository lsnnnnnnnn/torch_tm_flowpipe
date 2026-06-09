# Flow* Validation Candidate Decomposition Audit

This is diagnostic-only. It does not change solver behavior, rerun h10, add symbolic queue variants, or claim Flow* parity.

## Scope

- t_before requested: `0`
- h_try: `0.025000000000000001`
- Input traces: `outputs/flowstar_step_trace_compare/*.csv`
- Output ledger: `outputs/flowstar_validation_candidate_decomposition_audit/validation_candidate_decomposition_ledger.csv`

## Answers

- Full-step tube total boxes are width-close: `true`; width ratio torch/Flow* is `0.99985013193917116`.
- Acceptance-critical residual y_hi gap equals the full-step tube y_hi gap: `true`.
- Same-source full-step y_hi delta torch-Flow*: `-4.9559137703436562e-05`.
- Post-cutoff residual y_hi delta torch-Flow*: `-4.9559137709652415e-05`.
- Verdict: `residual_decomposition_mismatch`.
- Exposed component carrying the gap: `post_cutoff_residual`.
- Polynomial range component: `unknown`.
- Ordinary remainder component: `unknown`.
- Raw ctrunc residual component: `unknown`.
- Post-cutoff residual component: `differs`.
- Cutoff/polyDiff explains the y_hi gap: `false`.
- Is PyTorch putting width into polynomial range that Flow* puts into remainder: `unknown_missing_polynomial_range`.
- Is Flow* raw no-remainder still missing: `true`.
- Exact field to expose next if attribution remains unknown: flowstar ordinary_remainder/picard_no_remainder_residual endpoints; torch raw_ctrunc_residual/Picard_ctrunc_raw endpoints; full-step polynomial_range endpoints for both sources.

## Candidate Rows

| source | status | full-step box | target-check residual y | target y | subset y | y_hi margin | domain | center/scale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flowstar | rejected | x=[1.0975301322957665, 1.461604968996091], y=[2.2509745318250358, 2.4781433092829666] | [-8.3561112430831106e-05, 0.0001083283903691475] | [-0.0001, 0.0001] | false | 8.3283903691474946e-06 | physical_tube_over_full_step_tau_domain_before_tau_h_substitution | center=(1.25, 2.4000000000000004), scale=(0.14999999999999991, 0.049999999999999822) |
| torch_noqueue | accepted | x=[1.0975323219259019, 1.4616001369957994], y=[2.2510065595908366, 2.4780937501452631] | [-5.1533346624557205e-05, 5.8769252659495084e-05] | [-0.0001, 0.0001] | true | -4.1230747340504921e-05 | physical_tube_over_full_step_tau_domain_before_tau_h_substitution | center=(1.3091480530492146, 2.3302942066425647), scale=(0.15093725982051429, 0.077083451029272401) |
| torch_v2 | accepted | x=[1.0975323219259019, 1.4616001369957994], y=[2.2510065595908366, 2.4780937501452631] | [-5.1533346624557205e-05, 5.8769252659495084e-05] | [-0.0001, 0.0001] | true | -4.1230747340504921e-05 | physical_tube_over_full_step_tau_domain_before_tau_h_substitution | center=(1.3091480530492146, 2.3302942066425647), scale=(0.15093725982051429, 0.077083451029272401) |

## Decomposition Notes

- Flow* target-check residual y_hi is `0.0001083283903691475`; PyTorch target-check residual y_hi is `5.8769252659495084e-05`.
- Flow* y margin to target is `8.3283903691474946e-06`; PyTorch y margin to target is `-4.1230747340504921e-05`.
- The exposed Flow* raw ctrunc and post-cutoff residuals differ only by the recorded cutoff width on this row.
- The exposed PyTorch ordinary no-remainder and post-cutoff residuals differ only by the recorded cutoff width on this row.
- Polynomial range endpoints are blank in the current traces, so polynomial-vs-remainder width placement is not inferred.
- Blank component endpoint columns mean unknown, not zero.
- Missing component fields: flowstar.cutoff_poly_diff_x_hi; flowstar.cutoff_poly_diff_x_lo; flowstar.cutoff_poly_diff_y_hi; flowstar.cutoff_poly_diff_y_lo; flowstar.ordinary_remainder_x_hi; flowstar.ordinary_remainder_x_lo; flowstar.ordinary_remainder_y_hi; flowstar.ordinary_remainder_y_lo; flowstar.polynomial_range_x_hi; flowstar.polynomial_range_x_lo; flowstar.polynomial_range_y_hi; flowstar.polynomial_range_y_lo; torch_noqueue.cutoff_poly_diff_x_hi; torch_noqueue.cutoff_poly_diff_x_lo; torch_noqueue.cutoff_poly_diff_y_hi; torch_noqueue.cutoff_poly_diff_y_lo; torch_noqueue.polynomial_range_x_hi; torch_noqueue.polynomial_range_x_lo; torch_noqueue.polynomial_range_y_hi; torch_noqueue.polynomial_range_y_lo; torch_noqueue.raw_ctrunc_residual_x_hi; torch_noqueue.raw_ctrunc_residual_x_lo; torch_noqueue.raw_ctrunc_residual_y_hi; torch_noqueue.raw_ctrunc_residual_y_lo; torch_v2.cutoff_poly_diff_x_hi; torch_v2.cutoff_poly_diff_x_lo; torch_v2.cutoff_poly_diff_y_hi; torch_v2.cutoff_poly_diff_y_lo; torch_v2.polynomial_range_x_hi; torch_v2.polynomial_range_x_lo; torch_v2.polynomial_range_y_hi; torch_v2.polynomial_range_y_lo; torch_v2.raw_ctrunc_residual_x_hi; torch_v2.raw_ctrunc_residual_x_lo; torch_v2.raw_ctrunc_residual_y_hi; torch_v2.raw_ctrunc_residual_y_lo.
