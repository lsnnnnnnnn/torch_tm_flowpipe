# Three-tool final correctness report

## Executive conclusion

Official stock Flowstar order-4 Van der Pol reproduced T=10 four times with 290 segments. Torch sparse did not reach T=10; its best current validated native horizon is `6.39093111` before a finite self-map rejection. DiffReach's checked-out official VDP model cannot enter its native RHS interpreter because `jnp.stack` is unsupported; a separately labeled canonical-polynomial adapter completed T=10 for 64 partitions with every initial contraction flag true. The private Xiangru 2026 source and raw timing artifacts are absent.

Five of eight gates pass. Three remain blocked, so the formal comparison runner still refuses to start and there is no headline ranking, Pareto table, time-to-certificate comparison, or B48 speedup.

## Why compare these backends

Flowstar supplies the stock validated-flowpipe reference, DiffReach supplies a restricted tensor/JAX basis and batched execution model, and Torch tests whether a sparse complete-degree semantics can preserve validation while moving toward GPU batching. The scientific question is meaningful only after identical plants, source coordinates, effective support, remainder/self-map rules, endpoint fields, completion, and timing boundaries are established. The implementation-level audit is in `WHY_COMPARE_FLOWSTAR_DIFFREACH_TORCH.md`.

## Source identities

- Torch: branch `codex/three-tool-correctness-parity-20260804`; exact freeze SHA is recorded at finalization.
- Flowstar: official remote `https://github.com/chenxin415/flowstar`, SHA `b85a321...`, one tracked GCC15 derivative assignment fix, archive `libflowstar.a` SHA256 `3a658f95...`.
- DiffReach: `https://github.com/trustworthyrobotics/DiffReach`, clean `main`, SHA `dd628eb...`.
- Xiangru private: missing. Public `Verified-Intelligence/CROWN-Reach` SHA `7b90f308...` is explicitly distinct and not substituted.

Full paths, remotes, branches, SHA/dirty states, submodules, binary/library hashes, host/GPU/software, benchmark hashes, and command records are in `outputs/three_tool_reaudit/20260804T060058Z/manifest.json`.

## Reproduction status

| backend | lane | completed_horizon | validation_status | soundness_level | primary_eligible | result | runtime boundary/value |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| official-stock Flowstar o4 | native_reproduction | 10 | completed | formal_outward_rounding | false | 290 segments, 4/4 correctness runs; raw fixed-time endpoint unavailable | cold 1.07445 s; 3 steady 1.06533–1.08578 s |
| generated-stock Flowstar o4 | native_reproduction | 10 | completed | formal_outward_rounding | false | plot segments exactly match official | unmatched internal/compile/process boundary |
| torch-sparse best o4 | native_reproduction | 6.39093111 | validation_rejected | safeguarded_float64_not_fully_proved | false | finite target self-map failure | 291.151 s diagnostic total |
| diffreach-native official model | native_reproduction | 0 | unsupported_configuration | unknown | false | unsupported `stack` RHS primitive | setup failed |
| diffreach canonical adapter | matched_plant_backend | 10 | completed | unknown | false | 1,000/1,000 steps, 128,000/128,000 contraction flags | CPU cold 3.868 s, one steady 2.902 s |
| Xiangru private 2026 | native_end_to_end_certificate | 0 | blocked_missing_source | unknown | false | source/raw result absent | unavailable |

DiffReach adapter output is not a native-route claim. Its JAX environment is CPU-only despite host GPUs (`jax 0.10.2` reported no CUDA-enabled jaxlib), uses an optional stub only for an unused neural-bound import, and lacks a directed-rounding proof. It returns an endpoint aggregate `x in [-1.39645,-1.21959]`, `y in [-2.50221,-2.28040]` at T=10, but no lossless raw endpoint TM or last-segment object.

## Eight gates

| backend | lane | completed_horizon | validation_status | soundness_level | primary_eligible | gate | decision |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| Flowstar | identity | 10 | completed | formal_outward_rounding | false | `stock_backend_identity` | PASS: same SHA/archive; exact GCC15 patch classified |
| Flowstar | route parity | 10 | schema_invalid | formal_outward_rounding | false | `official_parser_generated_stock_field_parity` | BLOCK: official internal fields unavailable |
| Flowstar/Torch | exporter | 0.01 | schema_invalid | mixed | false | `endpoint_segment_tube_exporter_semantics` | BLOCK: scalar-affine Flowstar sanity violation |
| all | field semantics | n/a | completed | mixed | false | `raw_tightened_separation` | PASS: no fallback; collapsed/repaired diagnostic-only |
| three tools | basis | one step | completed | mixed | false | `order_basis_contract` | PASS: explicit hashes; DiffReach mismatch disclosed |
| all | timing | mixed | incomplete_unknown | mixed | false | `runtime_boundary_parity` | BLOCK: no matched 1+10 timing set |
| all | completion | mixed | completed | mixed | false | `completion_validation_fail_closed` | PASS: failed prefixes excluded |
| patched/dense | eligibility | 0 | schema_invalid | empirical_enclosure_only | false | `patched_rows_excluded_from_primary` | PASS: env, backend, and row negatives recorded |

Every decision has machine evidence, checksum, automated test, human report, and applicability in `benchmarks/cross_tool_gates.yaml`. Gate PASS describes that individual contract only; every displayed result remains primary-ineligible because the full gate set is blocked. Because at least one gate is false, `headline_comparison_generated=false`.

## Flowstar order 4 and order 2

Order 4 official T=10 is completed. Official/generated plot segment count is 290 and parsed maximum difference is zero, but deeper field parity is unavailable.

Order 2 is `validation_rejected`, not crashed or unsupported, in the adaptive probe. At the last `h=0.003125` attempt, x passes while y candidate `[-2.779354e-4,1.062582e-4]` exceeds the target. The multiplication remainder `[-2.634298e-4,9.120954e-5]` dominates cutoff near `1e-18`; the y self-map defect is `1.779354e-4`. A separate fixed `h=0.001` one-step succeeds in both generated-stock and normalized Torch and is not T=10 completion.

## One-step first divergence

Aligning Torch's source coordinates to Flowstar normalization fixes the first representation mismatch. Five cases then match accepted status, observed support, and retained coefficients (maximum coefficient error up to `3.55e-15`). The first observable output difference is the independent remainder; for VDP order 4 Torch's y remainder is about four to five times wider. The first required internal field cannot be compared because stock Flowstar does not export Picard iteration zero. The one-step gate remains false rather than guessing the missing root cause.

## Torch long horizon

The constant-center and range-midpoint runs stop at `6.31729088` and `6.39093111`. The latter terminal raw residual width sum is `2.28448e-4`, polynomial range width sum `4.29475`, and minimum target margin `-1.79427e-5`. Centering extends the horizon by only `0.07364`, not the preregistered `0.5`, and does not remove self-map failure. The current trace points to accumulated polynomial-range/remainder growth; it excludes cutoff-only, nonfinite, accepted raw-target violations, and a simple right-map-centering explanation.

## DiffReach correctness position

The native model route fails before step one because the analytic RHS uses `jnp.stack`, which the quadratic interpreter rejects. The canonical adapter uses the identical polynomial ODE expressed with supported concatenate/reshape primitives and completes T=10. This adapter has explicit restricted support SHA `d07f966b...`, 64 initial partitions, h=0.01, candidate remainder 0.01, ten refinement rounds, CPU x64 output, and contraction rate 1.0. Its ordinary round-to-nearest JAX arithmetic and missing raw endpoint/last-segment objects keep it out of an equivalent formal-certificate claim.

The upstream runner would scan all requested steps and only print an informational warning if contraction rate were below one (`run_dyn.py:193-239`). The audit wrapper independently rejects the first false flag; unit tests cover false contraction and nonfinite interval paths.

## Xiangru and timing

No private raw artifact exists from which to recompute the historical B12/B48 figures. The public C++ route's `asFloat()` calls are recorded, but it is not the private source. No B48 performance ratio is shown. Runtime gate failure also independently prohibits any speedup: Flowstar has three steady correctness runs, DiffReach one steady CPU run, Torch only long diagnostic totals, and Xiangru none under a verified workload.

## Withdrawn conclusions

- Any old “Torch tightest/fastest” three-tool ranking remains withdrawn.
- Plot segment parity is not field-level parity.
- Collapsed or repaired endpoints are not raw endpoints.
- Partial horizons and warning-bearing scans are not certificates.
- The dense GPU Euler kernel is not a Taylor-model flowpipe.
- The public release is not the private Xiangru 2026 experiment.

## Minimum next research step

First fix or independently explain the scalar-affine Flowstar exporter miss, then expose stock Picard/candidate fields through a read-only, numerically inert route and rerun the preregistered one-step contract. Only after those two gates pass should the exact official step schedule and fixed `h=0.005` protocols be promoted to T=10. Timing and Xiangru B48 work remain downstream of correctness and source availability.
