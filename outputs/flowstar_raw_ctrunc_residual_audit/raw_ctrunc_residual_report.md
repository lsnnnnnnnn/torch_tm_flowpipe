# Flow* Raw Ctrunc Residual Audit

This is diagnostic-only. It does not change solver behavior, rerun h10, add queue variants, or claim Flow* parity.

## Scope

- t_before requested: `0`
- h_try: `0.025000000000000001`
- Input traces: `outputs/flowstar_step_trace_compare/*.csv`
- Output ledger: `outputs/flowstar_raw_ctrunc_residual_audit/raw_ctrunc_residual_ledger.csv`

## Answers

- Are Flow* and PyTorch raw_ctrunc_residual semantically the same object: `true`; semantic mismatch: `none`.
- Raw y_hi delta torch-Flow*: `-4.955913870964615e-05`.
- First exposed component explaining raw y_hi gap: `raw_remainder_gap`.
- Raw polynomial range component: `same`.
- Raw returned remainder component: `differs`.
- No-remainder Picard range component: `unknown`.
- No-remainder Picard remainder component: `unknown`.
- Target remainder component: `same`.
- Does Flow* put a component into raw_ctrunc_residual that PyTorch does not: `false_at_exposed_flags`.
- Does PyTorch put a component elsewhere that Flow* puts into residual: `unknown_needs_picard_ctrunc_internal_partition`.
- Component attribution complete: `false`.
- Exact Flow* fields still missing: flowstar.ordinary_remainder_x_hi; flowstar.ordinary_remainder_x_lo; flowstar.ordinary_remainder_y_hi; flowstar.ordinary_remainder_y_lo.

## Candidate Rows

| source | status | raw residual y | raw polynomial y | raw remainder y | no-remainder range y | target y | component | domain | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flowstar | rejected | [-8.3561112430812905e-05, 0.00010832839036914122] | [2.2510580929374671, 2.478034980892597] | [-8.3561112430812905e-05, 0.00010832839036914122] | [1.8885861120605458, 2.5788354915608749] | [-0.0001, 0.0001] | reference | physical_remainder_interval_over_full_step_tau_domain_before_cutoff_polyDiff | target=false, ordinary=false, cutoff=false |
| torch_noqueue | accepted | [-5.1533345624557196e-05, 5.8769251659495075e-05] | [2.2510580929374617, 2.4780349808926032] | [-5.1533345624557196e-05, 5.8769251659495075e-05] | [, ] | [-0.0001, 0.0001] | raw_remainder_gap | physical_remainder_interval_over_full_step_tau_domain_before_cutoff_polyDiff | target=false, ordinary=false, cutoff=false |
| torch_v2 | accepted | [-5.1533345624557196e-05, 5.8769251659495075e-05] | [2.2510580929374617, 2.4780349808926032] | [-5.1533345624557196e-05, 5.8769251659495075e-05] | [, ] | [-0.0001, 0.0001] | raw_remainder_gap | physical_remainder_interval_over_full_step_tau_domain_before_cutoff_polyDiff | target=false, ordinary=false, cutoff=false |

## Notes

- Flow* raw_ctrunc_residual y_hi is `0.00010832839036914122`; PyTorch raw_ctrunc_residual y_hi is `5.8769251659495075e-05`.
- `raw_remainder_gap` means the exposed returned remainder carries the raw y_hi gap; it does not by itself close the internal Flow* Picard_ctrunc_normal attribution.
- Blank component endpoint columns mean unknown, not zero.
