# Flow* Acceptance Predicate Audit

This audit uses the first same-t/h divergence only. It does not add a solver mechanism and does not rerun h10.

## First Divergence

- t_before: `0`
- h_try: `0.025000000000000001`
- Flow*: `rejected`
- PyTorch no_queue: `accepted`
- PyTorch v2: `accepted`

## Predicate Ledger

| source | residual x | target x | subset x | residual y | target y | subset y | failed dim | width sum / target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flowstar | [-5.7034993171691869e-06, 9.6267419600486818e-06] | [-0.0001, 0.0001] | yes | [-8.3561112430831106e-05, 0.0001083283903691475] | [-0.0001, 0.0001] | no | y | 0.00020721974407719647 / 0.00040000000000000002 |
| torch_noqueue | [-3.5138691795527178e-06, 4.7947416664619525e-06] | [-0.0001, 0.0001] | yes | [-5.1533346624557205e-05, 5.8769252659495084e-05] | [-0.0001, 0.0001] | yes |  | 0.00011861121013006696 / 0.00040000000000000002 |
| torch_v2 | [-3.5138691795527178e-06, 4.7947416664619525e-06] | [-0.0001, 0.0001] | yes | [-5.1533346624557205e-05, 5.8769252659495084e-05] | [-0.0001, 0.0001] | yes |  | 0.00011861121013006696 / 0.00040000000000000002 |

## Flow* vs PyTorch Residual

- Flow* Picard_ctrunc_normal subset: `no`.
- PyTorch no_queue Picard_ctrunc-style subset: `yes`.
- PyTorch v2 Picard_ctrunc-style subset: `yes`.
- Flow* failed dimension: `y`.

## Why Width Is Not Enough

Flow* rejects because interval inclusion is endpoint-wise. In this row, the Flow* residual width sum relation is `0.00020721974407719647 < 0.00040000000000000002`, but the residual interval is shifted outside the symmetric target in dimension `y`. A smaller interval can still fail containment when its lower bound is below the target lower bound or its upper bound is above the target upper bound.

## Output

- `outputs/flowstar_acceptance_predicate_audit/acceptance_predicate_ledger.csv`
- `outputs/flowstar_acceptance_predicate_audit/acceptance_predicate_report.md`

## Limitation

The audit compares the diagnostic residual intervals exposed by the Flow* probe and the PyTorch flowstar-ctrunc validator for the same local box and h. It is not a new acceptance policy and not an end-to-end reachability comparison.
