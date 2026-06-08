# Flow* Picard Residual Source Audit

This audit uses the first same-t/h divergence only. It does not add a solver mechanism, does not rerun h10, and does not claim Flow* parity.

## Scope

- t_before requested: `0`
- h_try: `0.025000000000000001`
- Input traces: `outputs/flowstar_step_trace_compare/*.csv`
- Output ledger: `outputs/flowstar_picard_residual_source_audit/picard_residual_source_ledger.csv`
- Lifecycle ledger: `outputs/flowstar_box_lifecycle_alignment_audit/box_lifecycle_ledger.csv`

## Lifecycle Gate

- Pre-step boxes equal: `true`.
- Endpoint-before-center comparable: `true`.
- Reset-after-center boxes equal: `false`.
- First lifecycle stage divergence: `endpoint_box_before_center`.
- Residual comparison same-stage valid: `false`.
- Picard residual comparison: `noncausal/stage-misaligned`.
- Flow* missing residual components: `picard_no_remainder_residual`.

The residual endpoint mismatch is not yet a valid same-local-box comparison.

## Target-Check Residuals

| source | status | residual x | residual y | target x | target y | subset x | subset y | failed dim | local box |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flowstar | rejected | [-5.7034993171691869e-06, 9.6267419600486818e-06] | [-8.3561112430831106e-05, 0.0001083283903691475] | [-0.0001, 0.0001] | [-0.0001, 0.0001] | true | false | y | x=[1.1000000000000001, 1.3999999999999999], y=[2.3500000000000005, 2.4500000000000002] |
| torch_noqueue | accepted | [-3.5138691795527178e-06, 4.7947416664619525e-06] | [-5.1533346624557205e-05, 5.8769252659495084e-05] | [-0.0001, 0.0001] | [-0.0001, 0.0001] | true | true |  | x=[1.1582107932287002, 1.4600853128697289], y=[2.2532107556132921, 2.4073776576718373] |
| torch_v2 | accepted | [-3.5138691795527178e-06, 4.7947416664619525e-06] | [-5.1533346624557205e-05, 5.8769252659495084e-05] | [-0.0001, 0.0001] | [-0.0001, 0.0001] | true | true |  | x=[1.1582107932287002, 1.4600853128697289], y=[2.2532107556132921, 2.4073776576718373] |

## Attribution Answers

- Picard no-remainder: `not attributed`; lifecycle stage alignment is invalid or unknown.
- Picard ctrunc: `not attributed`; Flow* post-cutoff/Picard_ctrunc_normal y upper is `0.0001083283903691475`, PyTorch no_queue y upper is `5.8769252659495084e-05`, but their local-box stages are not yet proven comparable.
- Polynomial difference/cutoff: `not attributed`; PyTorch ordinary-to-post y upper shift is `1.0000000027804608e-12`, but cutoff attribution would be noncausal before same-stage boxes align.
- Domain/center/scale mismatch: `not evaluated from generic center/scale fields`; use stage-labeled boxes. Lifecycle first divergence is `endpoint_box_before_center`.
- Target remainder interval mismatch: `no`.
- Interval subset tolerance: `not the observed predicate issue`. Flow* fails endpoint inclusion in y by `8.3283903691474946e-06`, but this does not identify the residual source while stage alignment is invalid or unknown.
- Missing term in PyTorch residual accounting: `unknown`; the Flow* vs PyTorch y-upper gap `4.9559137709652415e-05` is not a causal residual-accounting comparison until lifecycle boxes align.

## Missing Fields

- Blank component endpoint columns mean the source trace did not expose that component endpoint.
- Width-only component evidence is kept in the row notes instead of being converted into fabricated intervals.
- Generic center/scale local_box columns are preserved for continuity but are deprecated for same-stage residual attribution.
