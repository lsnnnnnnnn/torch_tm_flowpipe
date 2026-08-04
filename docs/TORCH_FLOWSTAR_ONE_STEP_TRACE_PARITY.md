# Torch and Flowstar one-step trace parity

## Configuration repair

The original Torch one-step exporter used identity variables over physical state boxes. Total-degree truncation is coordinate-dependent, so this representation discarded large terms that Flowstar retains after center/radius normalization. For scalar quadratic order 4, `h=0.01`, target remainder `1e-4`, the physical-coordinate exporter rejected while the Flowstar-normalized exporter accepted. For VDP order 2, `h=0.001`, the physical representation produced y residual about `[-1.0075e-3,1.0075e-3]`; normalization reduced it to `[-8.16088e-5,2.65024e-5]` and accepted.

The current exporter therefore has an explicit `--source-coordinates flowstar_normalized` option. It uses `FlowstarNormalFlowpipeState.from_initial_box(...).normalized_initial_tm(...)`, the same normalization constructor used by adaptive normalized-insertion runs (`src/torch_tm_flowpipe/flowpipe.py:107-145`, `:3910-3917`). The physical path remains available as a diagnostic and cannot silently substitute for the matched path.

## Current observed parity

After permuting Flowstar's `[tau, xi...]` order to Torch's `[xi..., tau]`, observed last-segment support hashes match exactly in all five cases. Retained coefficient differences are floating-point noise:

| backend | lane | completed_horizon | validation_status | soundness_level | primary_eligible | case | support SHA256 | max coefficient difference | raw endpoint max difference |
| --- | --- | ---: | --- | --- | --- | --- | --- | ---: | ---: |
| generated-stock + torch-sparse | matched_plant_backend | 0.01 | completed | mixed | false | scalar affine o4 | `8b5fa9f0...` | 1.39e-17 | 2.00141e-6 |
| generated-stock + torch-sparse | matched_plant_backend | 0.01 | completed | mixed | false | scalar quadratic o4 | `b6295cb4...` | 6.94e-18 | 2.48172e-5 |
| generated-stock + torch-sparse | matched_plant_backend | 0.01 | completed | mixed | false | harmonic o4 | `f10191ca...` | 4.16e-17 | 3.31637e-8 |
| generated-stock + torch-sparse | matched_plant_backend | 0.005 | completed | mixed | false | VDP o4 | `61a0e19f...` | 3.55e-15 | 5.68910e-4 |
| generated-stock + torch-sparse | matched_plant_backend | 0.001 | completed | mixed | false | VDP o2 sensitivity | `8354507a...` | 2.29e-16 | 5.17263e-5 |

`mixed` means Flowstar uses its stock interval/MPFR path while Torch is `safeguarded_float64_not_fully_proved`; these rows are not claims of equivalent formal certificates.

The first observable output difference after source-domain, support, and coefficient alignment is the independent interval remainder. For VDP order 4, Flowstar remainders are approximately x `[-7.15e-9,8.33e-9]`, y `[-1.430e-6,1.665e-6]`; Torch has x `[-5.350e-7,5.648e-7]`, y `[-7.047e-6,7.279e-6]`.

## First contract blocker

The stock generated exporter exposes the accepted polynomial, remainder, endpoints, and boxes, but not source/Picard iteration, discarded monomials, nonlinear multiplication remainder, or candidate inclusion defect. Torch exposes a detailed validation trace. Therefore the first required field that cannot be compared is `flowstar.validation_trace.picard_iteration[0]`.

Output support and coefficient agreement cannot be used to infer those missing internal fields. `official_parser_generated_stock_field_parity` remains false even though official/generated T=10 plot segments match exactly. The endpoint exporter gate also remains false because the scalar-affine independent trajectory sanity check found a small Flowstar miss.

## Basis contract and DiffReach

The `order_basis_contract` gate passes because every compared support is explicit and hashed, not because all tools use the same basis. Flowstar/Torch observed supports match in the aligned cases. DiffReach's allowed support `{1,z_i,tau*z_i}` hashes to `7e0864b0...` for one state and `d07f966b...` for two states and is explicitly `matched_to_complete_order4=false`. Nominal `order=2/3/4` is never used as the sole grouping key.

Primary evidence: `outputs/three_tool_reaudit/20260804T060058Z/gate_evidence/one_step_parity.json`. Regression tests cover the old coordinate-dependent rejection, analytic scalar-affine containment, exponent permutation, and current-run fail-closed trajectory result.
