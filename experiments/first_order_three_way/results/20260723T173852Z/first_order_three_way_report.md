# First-order three-way reachability benchmark

> **Superseded for cross-tool ranking:** internal bases are not matched, so any
> width ordering below is descriptive historical output, not a solver ranking.
> Use `experiments/three_tool_deep_study/`.

## 1. Executive summary

The canonical sweep produced 279 tool/protocol/configuration runs: 141 certified and 138 unsupported or failed. The central result is semantic rather than a winner: the three projects do not expose the same object under their apparent first-order controls. Flow* fixed order 1 is explicitly rejected by the installed toolbox, and DiffReach's affine flag has transient degree-two time terms plus a nonstandard final projection. Torch TM is therefore the only primary path here that directly executes complete-total-degree-one Taylor models.

This report does not claim that one tool is globally better based on three plant systems.

## 2. Repository SHAs and environments

| repository | audited HEAD |
| --- | --- |
| torch_tm_flowpipe benchmark worktree | 5582ae97347d7a0a1c0126d4a50c5df2cd706387 |
| torch_tm_flowpipe original checkout | 26a254ef585a9dee394b7e41922c06bf8799f501 |
| DiffReach | dd628eb443b517d6415de93e7035b4baef73963e |
| Flow* toolbox | b85a3211748cb77b736fe4ad42ee02d8d2b81148 |

Commits recorded in the actual adapter rows:

| adapter | execution commit |
| --- | --- |
| DiffReach | dd628eb443b517d6415de93e7035b4baef73963e |
| Flow* | b85a3211748cb77b736fe4ad42ee02d8d2b81148 |
| Torch TM | 62edadcd1913e175f99dc95cb39ebaea500ba08f |

- Torch/plotting: py11 (Python 3.11, torch 2.5.1+cu121); the benchmark specification selects float64 CPU batch-1 even when CUDA devices are visible.
- DiffReach: diffreach312 (Python 3.12, CPU JAX 0.10.2, x64 enabled).
- Flow*: system GCC/G++ 15.2 and existing static toolbox library.
- Host: Intel Xeon Gold 6138 CPU. CUDA devices were visible, but the frozen benchmark specification selected float64 CPU batch 1 for Torch and CPU JAX for DiffReach.
- Full command output, branch listings, remotes, 100-commit graphs, untracked files, compiler, OS, CPU, memory, and device probes are preserved in `environment.json` and `environment.txt`.

The declared DiffReach editable install was attempted but pip found an internal dependency conflict: `jax2onnx` required Equinox ≥0.13.1 while available `immrax[cuda]` releases required Equinox ~=0.12.2. The plant-only analytic path used a minimal read-only source import in `diffreach312`.

## 3. Benchmark definitions

| system | dynamics | initial box | step sizes | horizons |
| --- | --- | --- | --- | --- |
| riccati | $\dot{x}=x^2$ | [[0.0, 0.1]] | 0.005, 0.01, 0.02, 0.05 | 0.1, 0.5, 1.0 |
| harmonic | $\dot{x}_1=x_2,\ \dot{x}_2=-x_1$ | [[-0.1, 0.1], [-0.1, 0.1]] | 0.01, 0.02, 0.05 | 1.0, 5.0, 10.0 |
| van_der_pol | $\dot{x}_1=x_2,\ \dot{x}_2=(1-x_1^2)x_2-x_1$ | [[1.1, 1.4], [2.35, 2.45]] | 0.005, 0.01 | 0.1, 0.25, 0.5, 1.0, 2.0 |

All configurations use a fixed grid, batch size 1, one partition, seed 20260723, 24 deterministic initial samples, and 5 steady timing repetitions. Main protocols do not rescue failures by raising order, adapting the step, or partitioning.

## 4. What order 1 means in each implementation

| implementation | setting | retained/effective basis | degree | discarded-term handling |
| --- | --- | --- | --- | --- |
| Torch TM | order=1 | all monomials of total degree ≤1 in local time and initial generators | 1 | higher-degree products are interval-bounded and added to the TM remainder |
| Flow* | fixed order=1 | unsupported by this toolbox API; setFixedStepsize requires order ≥2 | — | no order-2 result is relabeled first order |
| DiffReach | TRUNCATE_TO_AFFINE=True | affine final polynomial, but t² and t·z are created during integration | 2 transient | Lt interval radius is embedded into reused L generators, not an independent remainder |
| DiffReach | TRUNCATE_TO_AFFINE=False | {1, z, t², t·z} as implemented by c/L/Lt (restricted quasi-quadratic) | 2 restricted | supplemental, not a complete total-degree-2 basis |

This distinction is measured by automated support diagnostics, not inferred from option names. In particular, DiffReach's final `Lt==0` under the affine flag does not establish common affine semantics because nonzero `Lt` is observed immediately after time integration.

## 5. Primary native-first-order results

Aggregate statuses: {'certified_ok': 52, 'contraction_failed': 4, 'unsupported_order': 31, 'validation_failed': 6}.

| tool | system | configurations | status counts |
| --- | --- | --- | --- |
| DiffReach | harmonic | 9 | certified_ok: 9 |
| DiffReach | riccati | 12 | certified_ok: 12 |
| DiffReach | van_der_pol | 10 | certified_ok: 6, contraction_failed: 4 |
| Flow* | harmonic | 9 | unsupported_order: 9 |
| Flow* | riccati | 12 | unsupported_order: 12 |
| Flow* | van_der_pol | 10 | unsupported_order: 10 |
| Torch TM | harmonic | 9 | certified_ok: 9 |
| Torch TM | riccati | 12 | certified_ok: 12 |
| Torch TM | van_der_pol | 10 | certified_ok: 4, validation_failed: 6 |

Exact-width inflation for certified Riccati and harmonic endpoint states:

| tool | system | states | minimum | median | maximum |
| --- | --- | --- | --- | --- | --- |
| DiffReach | harmonic | 18 | 1.97717 | 125.6477 | 20627 |
| DiffReach | riccati | 12 | 1.00251 | 1.0126 | 1.02572 |
| Torch TM | harmonic | 18 | 2.17843 | 350.7508 | 1.942e+05 |
| Torch TM | riccati | 12 | 1.00003 | 1.00076 | 1.00352 |

Flow* entries are absent from quantitative order-one tables because its public fixed-step configuration API returned false for order 1. Unsupported is a result, not a missing run.

## 6. Strict-common-affine results

Aggregate statuses: {'certified_ok': 25, 'unsupported_order': 62, 'validation_failed': 6}.

| tool | system | configurations | status counts |
| --- | --- | --- | --- |
| DiffReach | harmonic | 9 | unsupported_order: 9 |
| DiffReach | riccati | 12 | unsupported_order: 12 |
| DiffReach | van_der_pol | 10 | unsupported_order: 10 |
| Flow* | harmonic | 9 | unsupported_order: 9 |
| Flow* | riccati | 12 | unsupported_order: 12 |
| Flow* | van_der_pol | 10 | unsupported_order: 10 |
| Torch TM | harmonic | 9 | certified_ok: 9 |
| Torch TM | riccati | 12 | certified_ok: 12 |
| Torch TM | van_der_pol | 10 | certified_ok: 4, validation_failed: 6 |

Torch dependency-preserving order 1 meets the declared common affine polynomial basis. Flow* cannot instantiate fixed order 1. DiffReach is marked unsupported because its tested affine projection converts dropped Lt radius into existing generator coefficients; the benchmark found no demonstrated independent-remainder projection with matching semantics.

## 7. Supplemental native representations

Aggregate statuses: {'certified_ok': 64, 'sample_violation': 9, 'validation_failed': 20}.

| tool | system | configurations | status counts |
| --- | --- | --- | --- |
| DiffReach | harmonic | 9 | certified_ok: 9 |
| DiffReach | riccati | 12 | certified_ok: 12 |
| DiffReach | van_der_pol | 10 | certified_ok: 10 |
| Flow* | harmonic | 9 | certified_ok: 6, validation_failed: 3 |
| Flow* | riccati | 12 | sample_violation: 9, validation_failed: 3 |
| Flow* | van_der_pol | 10 | validation_failed: 10 |
| Torch TM | harmonic | 9 | certified_ok: 9 |
| Torch TM | riccati | 12 | certified_ok: 12 |
| Torch TM | van_der_pol | 10 | certified_ok: 6, validation_failed: 4 |

The supplements are Torch range-only restarts, Flow* fixed total-degree 2 diagnostics, and DiffReach with `TRUNCATE_TO_AFFINE=False`. They illuminate wrapping and basis effects but are not relabeled as primary first-order results.

## 8. Tightness analysis

DiffReach controlled wrapping best among supported native first-order settings (median exact inflation: DiffReach 125.6477, Torch TM 350.7508). Supplemental medians were DiffReach 1.16977, Flow* 1.03805, Torch TM 128.50673; Flow* there is fixed order 2 and is not a first-order comparison.

DiffReach quasi-quadratic change relative to its affine-dynamics path (positive means the quasi-quadratic result is narrower):

| system | paired runs | minimum | median | maximum |
| --- | --- | --- | --- | --- |
| harmonic | 9 | 44.552% | 99.069% | 99.9921% |
| riccati | 12 | 0.0001871% | 0.0019% | 0.0183% |
| van_der_pol | 6 | 11.0467% | 44.0689% | 86.6047% |

Across 27 paired certified configurations, disabling affine projection changed DiffReach's final summed width by a median reduction of 11.4766% (range 0.0001871% to 99.9921%).

For Riccati, nonlinear products are the direct precision-loss point: Torch moves all degree-two-and-higher products to an interval remainder at order 1, while DiffReach temporarily keeps its restricted Lt terms and then either projects or retains them. Range-only Torch additionally discards cross-step generator dependence. Flow* supplies no scalar order-one trajectory because the installed API refuses that order.

## 9. Validation and failure analysis

Van der Pol validated horizons and first reported failure times:

| tool | protocol | h | max certified T | earliest failure time | status counts |
| --- | --- | --- | --- | --- | --- |
| DiffReach | native first-order setting | 0.005 | 0.5 | 0.62 | certified_ok:3, contraction_failed:2 |
| DiffReach | native first-order setting | 0.01 | 0.5 | 0.54 | certified_ok:3, contraction_failed:2 |
| DiffReach | strict common affine | 0.005 | — | 0 | unsupported_order:5 |
| DiffReach | strict common affine | 0.01 | — | 0 | unsupported_order:5 |
| DiffReach | supplemental native representation | 0.005 | 2 | — | certified_ok:5 |
| DiffReach | supplemental native representation | 0.01 | 2 | — | certified_ok:5 |
| Flow* | native first-order setting | 0.005 | — | 0 | unsupported_order:5 |
| Flow* | native first-order setting | 0.01 | — | 0 | unsupported_order:5 |
| Flow* | strict common affine | 0.005 | — | 0 | unsupported_order:5 |
| Flow* | strict common affine | 0.01 | — | 0 | unsupported_order:5 |
| Flow* | supplemental native representation | 0.005 | — | 0.005 | validation_failed:5 |
| Flow* | supplemental native representation | 0.01 | — | 0.01 | validation_failed:5 |
| Torch TM | native first-order setting | 0.005 | 0.25 | 0.485 | certified_ok:2, validation_failed:3 |
| Torch TM | native first-order setting | 0.01 | 0.25 | 0.47 | certified_ok:2, validation_failed:3 |
| Torch TM | strict common affine | 0.005 | 0.25 | 0.485 | certified_ok:2, validation_failed:3 |
| Torch TM | strict common affine | 0.01 | 0.25 | 0.47 | certified_ok:2, validation_failed:3 |
| Torch TM | supplemental native representation | 0.005 | 0.5 | 0.63 | certified_ok:3, validation_failed:2 |
| Torch TM | supplemental native representation | 0.01 | 0.5 | 0.6 | certified_ok:3, validation_failed:2 |

Each adapter stops exporting enclosures after its first failed contraction, failed validation, non-finite interval, or timeout. The collector can further downgrade a nominally certified run to `sample_violation`; it never heals a failure.

## 10. Runtime analysis

Primary certified configurations:

| tool | system | median build | median warmup | median steady | device |
| --- | --- | --- | --- | --- | --- |
| DiffReach | harmonic | 0 | 0.57655 | 0.00567 | jax_cpu |
| DiffReach | riccati | 0 | 0.50986 | 0.00041489 | jax_cpu |
| DiffReach | van_der_pol | 0 | 0.93122 | 0.00151 | jax_cpu |
| Torch TM | harmonic | 0 | 2.72291 | 2.72161 | cpu |
| Torch TM | riccati | 0 | 0.19605 | 0.19178 | cpu |
| Torch TM | van_der_pol | 0 | 0.57382 | 0.57601 | cpu |

Supplemental certified configurations:

| tool | system | median build | median warmup | median steady | device |
| --- | --- | --- | --- | --- | --- |
| DiffReach | harmonic | 0 | 0.84557 | 0.00554 | jax_cpu |
| DiffReach | riccati | 0 | 0.6883 | 0.00030296 | jax_cpu |
| DiffReach | van_der_pol | 0 | 1.82728 | 0.00358 | jax_cpu |
| Flow* | harmonic | 1.83473 | 0.25292 | 0.24986 | cpu |
| Torch TM | harmonic | 0 | 2.83826 | 2.82045 | cpu |
| Torch TM | riccati | 0 | 0.19778 | 0.19776 | cpu |
| Torch TM | van_der_pol | 0 | 0.98569 | 0.97699 | cpu |

Across 25 paired certified primary configurations, DiffReach was faster in 25; the median faster/slower speed ratio was 502.53321× and the faster method's median final summed-width ratio was 0.90761×. The faster result was >10% wider in 0 pairs and >10% narrower in 13 pairs. First Van der Pol failures were DiffReach h=0.005: t=0.62; DiffReach h=0.01: t=0.54; Torch TM h=0.005: t=0.485; Torch TM h=0.01: t=0.47.

Build/source generation, first-call or JIT time, and steady runtime are deliberately separate. These are CPU measurements on one shared host, not a hardware-fair CPU/GPU claim. A fast unsupported run is not treated as useful speed, and timing is interpreted alongside width and validation horizon. The all-CPU sweep includes the requested batch-1 subsets Riccati h=0.01/T=1, harmonic h=0.01/T=5, and Van der Pol h=0.01/T=0.5.

## 11. Exact references and sampled-trajectory sanity checks

| check | runs | values checked | violations | interpretation |
| --- | --- | --- | --- | --- |
| exact_endpoint_containment | 150 | 35691 | 551 | analytic exact-hull containment |
| sampled_trajectory_tube_containment | 150 | 2827440 | 551 | sanity check only |

Riccati and harmonic endpoint hulls are analytic. Van der Pol and whole-segment tubes use high-accuracy SciPy DOP853 trajectories only as bug-catching samples. Samples do not prove soundness; passing them is strictly weaker than a formal enclosure proof.

## 12. Limitations

- The retained polynomial bases differ materially, including DiffReach's restricted Lt basis.
- Numerical soundness backends differ: Torch floating-point intervals, Flow* MPFR intervals, and JAX floating-point interval arithmetic are not interchangeable guarantees.
- All reported primary timings are CPU-only by specification; visible GPUs were deliberately excluded, and GPU results could alter performance but not representation semantics.
- Endpoint enclosures and whole-segment tubes are distinct and were extracted/evaluated separately.
- The benchmark exercises plant dynamics only; it imports no controller or CROWN component.
- Sampled trajectories are sanity checks and never establish formal soundness.
- Flow* total-degree-two results and range-only/quasi-quadratic ablations are supplemental, not substitutes for missing common-affine runs.

## 13. Recommended next experiment

Implement and unit-test in DiffReach an explicit projection that sends every Lt term to a fresh independent interval remainder, then expose or safely enable Flow* fixed order 1 in a separate toolbox branch. Re-run the same frozen plant spec with those two semantics changes before expanding to partitions, controllers, or GPU batching.

## Conclusion

**Scalar nonlinear precision:** Torch order 1 loses nonlinear dependence when degree-two products enter its interval remainder; Torch range-only loses additional cross-step dependence. DiffReach loses or reshapes precision at its Lt projection, while its quasi-quadratic mode retains restricted time interactions. Flow* order 1 is unsupported.

**Linear oscillator wrapping:** DiffReach controlled wrapping best among supported native first-order settings (median exact inflation: DiffReach 125.6477, Torch TM 350.7508). Supplemental medians were DiffReach 1.16977, Flow* 1.03805, Torch TM 128.50673; Flow* there is fixed order 2 and is not a first-order comparison.

**Van der Pol failure horizon:** The per-step-size values are reported in the validation table above; unsupported configurations have no validated horizon and are not interpreted as numerical failures.

**Same DiffReach basis?** No. Its native affine path creates nonzero degree-two Lt terms during integration and its default non-affine path retains a restricted quasi-quadratic basis, unlike complete-total-degree-one Torch semantics. Flow* order 1 did not run.

**Quasi-quadratic improvement:** Across 27 paired certified configurations, disabling affine projection changed DiffReach's final summed width by a median reduction of 11.4766% (range 0.0001871% to 99.9921%).

**Runtime versus quality:** Across 25 paired certified primary configurations, DiffReach was faster in 25; the median faster/slower speed ratio was 502.53321× and the faster method's median final summed-width ratio was 0.90761×. The faster result was >10% wider in 0 pairs and >10% narrower in 13 pairs. First Van der Pol failures were DiffReach h=0.005: t=0.62; DiffReach h=0.01: t=0.54; Torch TM h=0.005: t=0.485; Torch TM h=0.01: t=0.47. Unsupported order-one Flow* timings are not gains, and validation failures are not treated as speedups.

## Reproduction

```bash
cd /srv/local/shengenli/torch_tm_flowpipe_first_order_bench
experiments/first_order_three_way/run_smoke.sh
experiments/first_order_three_way/run_all.sh
# or: experiments/first_order_three_way/launch_background.sh
```
