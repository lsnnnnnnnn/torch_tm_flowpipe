# Flowstar endpoint exporter audit

## Field contract

The generated-stock exporter now writes six independent objects: native fixed-time `endpoint_raw`, `endpoint_collapsed` from `evaluate_time`, unavailable-or-distinct `endpoint_tightened`, diagnostic `repaired_hull`, `last_segment`, and `full_tube`. It obtains a native box by fixing the composed Taylor model's local-time domain to the accepted right endpoint. It never mutates the Taylor-model remainder to make the two endpoint paths agree.

Term access uses a compile-time `#define protected public` shim in the generated harness. The stock headers and `libflowstar.a` are not edited by the exporter, and numerical routines are unchanged. Backend identity is checked before compilation.

For the matched VDP order-4 step (`h=0.005`, candidate `1e-4`, cutoff `1e-10`), the exported paths are:

| backend | lane | completed_horizon | validation_status | soundness_level | primary_eligible | field | x interval | y interval |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| generated-stock | matched_plant_backend | 0.005 | completed | formal_outward_rounding | false | raw endpoint | [1.1117027716, 1.4122315921] | [2.3310301354, 2.4425879540] |
| generated-stock | matched_plant_backend | 0.005 | completed | formal_outward_rounding | false | collapsed diagnostic | [1.1117297579, 1.4122046058] | [2.3313406142, 2.4422774751] |
| generated-stock | matched_plant_backend | 0.005 | completed | formal_outward_rounding | false | repaired hull | [1.1117027716, 1.4122315921] | [2.3310301354, 2.4425879540] |

The collapsed interval misses the native path by `2.69863e-5` per x side and `3.10479e-4` per y side. It is therefore marked an under-enclosure diagnostic and is ineligible for headline widths. The raw/tightened/collapsed/repaired separation gate passes.

## Containment and independent sanity

Every current raw endpoint box is contained in its corresponding one-step segment box for both generated-stock and Torch. An independent deterministic RK4 check integrates every initial-box corner with 1,000 substeps and checks the endpoint and eleven path samples. Zero violations would only be empirical sanity, never a formal proof.

That sanity check exposed a blocker in the custom scalar-affine generated-stock export for `x'=1+2x`, `x(0) in [0,0.1]`, `h=0.01`:

- exact/RK4 lower endpoint: `0.010100670013377895`; exported lower: `0.010100670333333329`;
- exact/RK4 upper endpoint: `0.11212080401605326`; exported upper: `0.11212080366666670`;
- maximum miss: about `3.50e-10`; both endpoint corners violate, and the upper final path sample also violates the segment box.

The evidence file records the exact points. Because the requirement says any sampled violation blocks the exporter, `endpoint_segment_tube_exporter_semantics` remains false. No tolerance was retrofitted after observing the discrepancy.

## Commands and evidence

```bash
conda run -n py11 python experiments/three_tool_reaudit/analyze_one_step.py \
  --run-dir outputs/three_tool_reaudit/20260804T060058Z \
  --output outputs/three_tool_reaudit/20260804T060058Z/gate_evidence/one_step_parity.json
```

Primary evidence is `gate_evidence/endpoint_segment_tube_exporter_semantics.json`; the explicit field-separation evidence is `gate_evidence/raw_tightened_separation.json`. Plots from the older T=10 parity script read GNUPLOT segment boxes and label them as such; no endpoint plot is inferred from those files.
