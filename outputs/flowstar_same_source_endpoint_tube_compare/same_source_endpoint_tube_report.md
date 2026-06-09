# Flow* Same-Source Endpoint/Tube Comparison

This is diagnostic-only and makes no solver change. It does not rerun h10, add queue variants, or claim Flow* parity.

## Scope

- t_before requested: `0`
- h_try: `0.025000000000000001`
- Input traces: `outputs/flowstar_step_trace_compare/*.csv`
- Output ledger: `outputs/flowstar_same_source_endpoint_tube_compare/same_source_endpoint_tube_ledger.csv`

## Answers

- Full-step tube comparison semantically valid: `true`.
- Tau=h endpoint comparison semantically valid: `true`.
- Which same-source object differs first: `full_step_tube:same_source_x_lo_divergence`.
- Does the tau=h endpoint match endpoint-wise at tolerance: `false`.
- Does the full-step tube match endpoint-wise at tolerance: `false`.
- previous_endpoint_before_center_y_hi_delta: `-0.071959548516590921`.
- same_source_full_step_y_hi_delta: `-4.9559136703347662e-05`.
- same_source_tau_h_y_hi_delta: `-0.0021665797665910702`.
- previous_gap_reduced_factor: `1451.9935838940903`.
- The old large y_hi gap is mostly explained by source/stage mismatch; a smaller same-source y_hi gap remains and is acceptance-critical.
- Likely component if mismatch remains: full-step validation candidate tube construction or range evaluation.
- Missing fields: none.

## Boxes

| comparison | semantic valid | verdict | Flow* box | torch no_queue box | torch v2 box | y_hi delta no_queue-flowstar |
| --- | --- | --- | --- | --- | --- | --- |
| full_step_tube | true | same_source_x_lo_divergence | x=[1.0975301322957665, 1.461604968996091], y=[2.2509745318250358, 2.4781433092829666] | x=[1.097532321924902, 1.4616001369967995], y=[2.2510065595898365, 2.4780937501462632] | x=[1.097532321924902, 1.4616001369967995], y=[2.2510065595898365, 2.4780937501462632] | -4.9559136703347662e-05 |
| tau_h_endpoint | true | same_source_x_lo_divergence | x=[1.1575306267512309, 1.4607525275898412], y=[2.2510617072175996, 2.4083503405329667] | x=[1.158210793227699, 1.460069718743217], y=[2.2532107556122885, 2.4061837607663756] | x=[1.158210793227699, 1.460069718743217], y=[2.2532107556122885, 2.4061837607663756] | -0.0021665797665910702 |

## Notes

Blank endpoint fields are reported as unknown and are not treated as zero.
The older endpoint-before-center fields remain a separate source-stage diagnostic; this report compares the explicitly labeled full-step tube and tau=h endpoint objects.
