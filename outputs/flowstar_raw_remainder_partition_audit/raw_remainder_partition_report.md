# Flow* Raw Remainder Partition Audit

This is diagnostic-only. It does not change solver behavior, rerun h10, add queue variants, or claim Flow* parity.

## Scope

- t_before requested: `0`
- h_try: `0.025000000000000001`
- Input traces: `outputs/flowstar_step_trace_compare/*.csv`
- Output ledger: `outputs/flowstar_raw_remainder_partition_audit/raw_remainder_partition_ledger.csv`

## Flow* Source Inspection

- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h::TaylorModelVec<DATA_TYPE>::Picard_ctrunc_normal(... Expression ..., intermediate_ranges, Global_Setting)`
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h::TaylorModelVec<DATA_TYPE>::Picard_ctrunc_normal_remainder(... Expression ..., intermediate_ranges, Global_Setting)`
- `/srv/local/shengenli/flowstar/flowstar-toolbox/expression.h::AST_Node<DATA_TYPE>::evaluate(... step_exp_table ..., intermediate_ranges, Global_Setting)`
- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h::HornerForm<DATA_TYPE>::insert_ctrunc_normal(... intermediate_ranges ...)`
- `experiments/flowstar_probe/flowstar_vdp_step_trace_probe.cpp::traced_advance_adaptive_symbolic`

## Answers

- Is Flow* raw returned remainder decomposable from exposed fields: `false`.
- Raw y_hi delta torch-Flow*: `-4.955913870964615e-05`.
- First subcomponent explaining y_hi delta: `unknown_missing_internal_partition`.
- Cause classification: `unknown`.
- Dropped-term range component: `unknown`.
- Multiplication remainder component: `unknown`.
- Integration/Picard remainder component: `differs`.
- Range-enclosure evidence: `unknown_or_mismatch`.
- Domain-scaling evidence: `same_no_scaling`.
- Soundness implication: `no_pyTorch_unsoundness_evidence_representation_split_or_hidden_flowstar_partition`.
- Exact Flow* object to expose next if still unknown: TaylorModel.h Expression/Horner intermediate_ranges entries mapped per state dimension: ctrunc_normal intTrunc/intTrunc2, mul_ctrunc_normal remainder contributions, and evaluate_remainder accumulation before result = tmvTmp2 + x0.
- Missing Flow* fields: flowstar.after_dropped_terms_x_hi; flowstar.after_dropped_terms_x_lo; flowstar.after_dropped_terms_y_hi; flowstar.after_dropped_terms_y_lo; flowstar.before_accumulation_x_hi; flowstar.before_accumulation_x_lo; flowstar.before_accumulation_y_hi; flowstar.before_accumulation_y_lo; flowstar.dropped_terms_x_hi; flowstar.dropped_terms_x_lo; flowstar.dropped_terms_y_hi; flowstar.dropped_terms_y_lo; flowstar.multiplication_remainder_x_hi; flowstar.multiplication_remainder_x_lo; flowstar.multiplication_remainder_y_hi; flowstar.multiplication_remainder_y_lo.

## Candidate Rows

| source | status | raw residual y | dropped y | multiplication y | integration y | after cutoff y | component |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flowstar | rejected | [-8.3561112430812905e-05, 0.00010832839036914122] | [, ] | [, ] | [-7.3737867352530173e-05, 9.9578243395254256e-05] | [-8.3561112430812905e-05, 0.00010832839036914122] | reference |
| torch_noqueue | accepted | [-5.1533345624557196e-05, 5.8769251659495075e-05] | [-2.1026408426672235e-05, 2.9821417558689934e-05] | [-3.0506936197884938e-05, 2.8947833100805118e-05] | [-3.0506936197884941e-05, 2.8947833100805121e-05] | [-5.1533344624557186e-05, 5.8769250659495065e-05] | unknown_missing_internal_partition |
| torch_v2 | accepted | [-5.1533345624557196e-05, 5.8769251659495075e-05] | [-2.1026408426672235e-05, 2.9821417558689934e-05] | [-3.0506936197884938e-05, 2.8947833100805118e-05] | [-3.0506936197884941e-05, 2.8947833100805121e-05] | [-5.1533344624557186e-05, 5.8769250659495065e-05] | unknown_missing_internal_partition |

## Notes

- Blank internal partition columns mean unknown, not zero.
- `hidden_raw_remainder_gap` means all exposed partitions matched but raw_ctrunc_residual still differed, so attribution requires deeper Flow* internal instrumentation.
- No evidence here by itself suggests PyTorch is missing a soundness component; the remaining issue is a Flow*/PyTorch representation split or hidden Flow* partition until those internal objects are exposed.
