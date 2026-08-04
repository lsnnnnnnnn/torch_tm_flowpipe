# Flowstar stock Van der Pol reproduction

## Identity and model

The official program route is `/srv/local/shengenli/flowstar/benchmarks/continuous/vanderpol/vanderpol.cpp`, SHA256 `aab94a62...`, repository SHA `b85a3211748cb77b736fe4ad42ee02d8d2b81148`. This checkout has one tracked GCC15 compatibility edit in `flowstar-toolbox/TaylorModel.h`: `remainder = 0` became `result.remainder = 0` inside `TaylorModel::derivative` (`TaylorModel.h:897-900`). Backend classification is `stock-plus-gcc15-compat`, not `patched-audit`; the patch SHA and dirty state are recorded in the manifest.

The official source defines `x'=y`, `y'=(1-x^2)y-x`, `t'=1` (`vanderpol.cpp:17-26`), initial `x in [1.1,1.4]`, `y in [2.35,2.45]` (`:42-53`), adaptive `h in [0.002,0.1]`, order 4 (`:32-37`), symbolic remainder queue 100 (`:67-72`), and T=10 (`:74-89`). The stock defaults used by the program are cutoff `[-1e-10,1e-10]`, candidate remainder `[-1e-4,1e-4]`, and precision 53, recorded in `raw/flowstar_official_vdp/evidence.json`.

## Command and outcome

```bash
conda run -n py11 python experiments/three_tool_reaudit/flowstar_vdp_reproduction.py \
  --flowstar-root /srv/local/shengenli/flowstar \
  --output outputs/three_tool_reaudit/20260804T060058Z/raw/flowstar_official_vdp \
  --repetitions 4
```

| backend | lane | completed_horizon | validation_status | soundness_level | primary_eligible | segments | wall seconds |
| --- | --- | ---: | --- | --- | --- | ---: | ---: |
| official-stock cold | native_reproduction | 10 | completed | formal_outward_rounding | false | 290 | 1.074450 |
| official-stock steady 1 | native_reproduction | 10 | completed | formal_outward_rounding | false | 290 | 1.065326 |
| official-stock steady 2 | native_reproduction | 10 | completed | formal_outward_rounding | false | 290 | 1.085778 |
| official-stock steady 3 | native_reproduction | 10 | completed | formal_outward_rounding | false | 290 | 1.067662 |

All four processes exited 0 and both plot streams reached T=10. The official program does not print rejected adaptive attempts, so that count is `unavailable`, not zero. Its plotting code transforms flowpipes and writes GNUPLOT interval rectangles (`vanderpol.cpp:115-130`); the evidence labels them as segment/tube boxes, never fixed-time endpoints. Consequently these correctness reproductions are real but not primary-comparison eligible under the raw-endpoint contract.

The executable SHA256 is `c4eca6f5...`; `libflowstar.a` is `3a658f95...`. The generated-stock one-step exporter compiled against the same sole `-lflowstar` archive and repository SHA. `gate_evidence/stock_backend_identity.json` includes both binary hashes, stable `ldd` dependency lists, the archive hash, the exact tracked patch, and the compile link contract. The identity gate passes.

## Official versus generated route

The current generated T=10 harness also produced 290 segments and T=10. Every parsed time/state segment field matched the official plot output exactly (`max_abs_segment_field_diff=0`). This is useful plot parity only. The official route does not export the source polynomial, Picard iterations, discarded terms, raw candidate remainder, rejection reasons, or native fixed-time endpoint, so field-level official/generated parity remains blocked.

The generated and official process walls near 1.06 s are not a performance comparison: generated internal reach time, compile wall, process wall, and official wall are different boundaries, and only three official steady correctness repetitions were collected. The runtime gate remains false.
