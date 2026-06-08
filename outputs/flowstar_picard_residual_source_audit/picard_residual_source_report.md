# Flow* Picard Residual Source Audit

This audit uses the first same-t/h divergence only. It does not add a solver mechanism, does not rerun h10, and does not claim Flow* parity.

## Scope

- t_before requested: `0`
- h_try: `0.025000000000000001`
- Input traces: `outputs/flowstar_step_trace_compare/*.csv`
- Output ledger: `outputs/flowstar_picard_residual_source_audit/picard_residual_source_ledger.csv`

## Target-Check Residuals

| source | status | residual x | residual y | target x | target y | subset x | subset y | failed dim | local box |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flowstar | rejected | [-5.7034993171691869e-06, 9.6267419600486818e-06] | [-8.3561112430831106e-05, 0.0001083283903691475] | [-0.0001, 0.0001] | [-0.0001, 0.0001] | true | false | y | x=[1.1000000000000001, 1.3999999999999999], y=[2.3500000000000005, 2.4500000000000002] |
| torch_noqueue | accepted | [-3.5138691795527178e-06, 4.7947416664619525e-06] | [-5.1533346624557205e-05, 5.8769252659495084e-05] | [-0.0001, 0.0001] | [-0.0001, 0.0001] | true | true |  | x=[1.1582107932287002, 1.4600853128697289], y=[2.2532107556132921, 2.4073776576718373] |
| torch_v2 | accepted | [-3.5138691795527178e-06, 4.7947416664619525e-06] | [-5.1533346624557205e-05, 5.8769252659495084e-05] | [-0.0001, 0.0001] | [-0.0001, 0.0001] | true | true |  | x=[1.1582107932287002, 1.4600853128697289], y=[2.2532107556132921, 2.4073776576718373] |

## Attribution Answers

- Picard no-remainder: `unknown` for Flow* because the probe does not expose no-remainder residual endpoints. The PyTorch ordinary residual endpoints are inside the target at h=0.025, so this row does not support a PyTorch no-remainder rejection.
- Picard ctrunc: `yes, at the exposed target-check residual`. Flow* post-cutoff/Picard_ctrunc_normal y upper is `0.0001083283903691475`, above target `0.0001` by `8.3283903691474946e-06`; PyTorch no_queue y upper is `5.8769252659495084e-05`, lower than Flow* by `4.9559137709652415e-05`.
- Polynomial difference/cutoff: `not supported as the primary source by exposed widths`. The endpoint fields are missing, but width-only trace fields are tiny here; PyTorch ordinary-to-post y upper shift is `1.0000000027804608e-12`.
- Domain/center/scale mismatch: `yes`. The inferred local boxes differ; max center delta Flow* vs no_queue is `0.069705793357435653` and max scale delta is `0.027083451029272579`.
- Target remainder interval mismatch: `no`. All exposed target intervals are `[-0.0001, 0.0001]`.
- Interval subset tolerance: `no`. Flow* fails endpoint inclusion in y; the upper endpoint exceeds the target by `8.3283903691474946e-06`, so width-only comparison is not the predicate.
- Missing term in PyTorch residual accounting: `not indicated by this trace`. PyTorch records both ordinary residual endpoints and post-cutoff/Picard_ctrunc_normal endpoints; the recorded post-cutoff change is far too small to explain the Flow* y-upper gap. Flow* raw ctrunc and no-remainder endpoints remain missing, so the precise pre/post-ctrunc split is still unknown.

## Missing Fields

- Blank component endpoint columns mean the source trace did not expose that component endpoint.
- Width-only component evidence is kept in the row notes instead of being converted into fabricated intervals.
