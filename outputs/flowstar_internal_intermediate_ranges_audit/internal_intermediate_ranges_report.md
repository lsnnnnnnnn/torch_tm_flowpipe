# Flow* Internal Intermediate Ranges Audit

This is diagnostic-only. It does not change solver behavior, rerun h10, add queue variants, commit Flow* source, or claim Flow* parity.

## Scope

- t_before requested: `0`
- h_try: `0.025000000000000001`
- Input traces: `outputs/flowstar_step_trace_compare/*.csv`
- Output ledger: `outputs/flowstar_internal_intermediate_ranges_audit/internal_intermediate_ranges_ledger.csv`

## Flow* Source Inspection

- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h::TaylorModelVec<DATA_TYPE>::Picard_ctrunc_normal(... Expression ..., intermediate_ranges, Global_Setting)`
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h::TaylorModelVec<DATA_TYPE>::Picard_ctrunc_normal_remainder(... Expression ..., intermediate_ranges, Global_Setting)`
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h::TaylorModel<DATA_TYPE>::mul_insert_ctrunc_normal(... Interval &tm1, Interval &intTrunc ...)`
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h::HornerForm<DATA_TYPE>::insert_ctrunc_normal(... intermediate_ranges ...)`
- `/srv/local/shengenli/flowstar/flowstar-toolbox/expression.h::AST_Node<DATA_TYPE>::evaluate(... intermediate_ranges ...)`
- `/srv/local/shengenli/flowstar/flowstar-toolbox/expression.h::AST_Node<DATA_TYPE>::evaluate_remainder(... iterator over intermediate_ranges ...)`
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp::advance/result constructions around Picard_ctrunc_normal`
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.cpp::not present in this checkout; relevant template implementations are in TaylorModel.h`
- `experiments/flowstar_probe/flowstar_vdp_step_trace_probe.cpp::traced_advance_adaptive_symbolic`

## Answers

- Which Flow* internal object first explains raw y_hi gap: `accumulated_remainder_before_x0_add_gap`.
- Raw y_hi delta torch-Flow*: `-4.955913870964615e-05`.
- Cause classification: `accumulation before x0 add`.
- Dropped terms evidence: intTrunc `differs`, intTrunc2 `differs`.
- Multiplication remainder evidence: `differs`.
- Expression evaluate_remainder evidence: `differs`.
- Accumulation before x0 add evidence: `differs`.
- Accumulation after x0 add evidence: `differs`.
- If still unknown, inaccessible Flow* object: none_for_this_classification.
- Missing Flow* fields: flowstar.horner_insert_ctrunc_remainder_x_hi; flowstar.horner_insert_ctrunc_remainder_x_lo; flowstar.horner_insert_ctrunc_remainder_y_hi; flowstar.horner_insert_ctrunc_remainder_y_lo.

## Candidate Rows

| source | status | raw y | expression y | intTrunc y | intTrunc2 y | mul y | before x0 y | after x0 y | component |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flowstar | rejected | [-8.3561112430812905e-05, 0.00010832839036914122] | [-8.3561112430812905e-05, 0.00010832839036914122] | [-4.0286032915115376e-06, 1.5072147467136393e-06] | [-4.6558726699055393e-05, 6.6151732498770011e-05] | [-7.1237867352530167e-05, 9.707824339525425e-05] | [-8.3561112430812905e-05, 0.00010832839036914122] | [-8.3561112430812905e-05, 0.00010832839036914122] | reference |
| torch_noqueue | accepted | [-5.1533345624557196e-05, 5.8769251659495075e-05] | [-3.0506936197884941e-05, 2.8947833100805121e-05] | [-2.1026408426672235e-05, 2.9821417558689934e-05] | [-2.1026408426672235e-05, 2.9821417558689934e-05] | [-3.0506936197884938e-05, 2.8947833100805118e-05] | [-5.1533344624557186e-05, 5.8769250659495065e-05] | [-5.1533345624557196e-05, 5.8769251659495075e-05] | accumulated_remainder_before_x0_add_gap |
| torch_v2 | accepted | [-5.1533345624557196e-05, 5.8769251659495075e-05] | [-3.0506936197884941e-05, 2.8947833100805121e-05] | [-2.1026408426672235e-05, 2.9821417558689934e-05] | [-2.1026408426672235e-05, 2.9821417558689934e-05] | [-3.0506936197884938e-05, 2.8947833100805118e-05] | [-5.1533344624557186e-05, 5.8769250659495065e-05] | [-5.1533345624557196e-05, 5.8769251659495075e-05] | accumulated_remainder_before_x0_add_gap |

## Notes

- Blank internal fields mean unknown or not-applicable, not zero.
- The Flow* probe uses the `Expression<Real>` ODE overload; `horner_insert_ctrunc_remainder` is blank unless a Horner call path is exposed.
- PyTorch comparison columns use same-named fields when present and fall back to the existing raw remainder partition diagnostics where noted in the ledger.
