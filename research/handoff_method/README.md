# Numerical components used in the handoff

These are the project-side controller and validation components retained from the October 23 engine transfer. The selected October 26–27 runs built on that transfer and chose their degrees, mode, scheduling and reporting at run time. A useful first check on a new machine is to run the controller and history arithmetic on a small input whose exact answer is known.

## Run the CPU check

Use a Python environment with PyTorch 2.5.1 (binary64 on CPU is sufficient). From this directory inside the repository:

```sh
repo=$(git rev-parse --show-toplevel)
PYTHONPATH="$repo/research/gpu_verified_20260930/source/engine/src" python check_components.py
```

The first check constructs a four-input network with two paths whose outputs cancel. It runs the original delayed residual evaluator, checks that the exact output zero belongs to the intersection and checks that the directed partition produces 256 cells. The second check propagates a remainder of 3 through a factor in [0.33, 0.34]. It checks the actual history update and prints bounds close to [0.99, 1.02], which contain 1. This check exercised PyTorch 2.5.1 on CPU. The benchmark timings in the slides came from the recorded graphics-processor runs.

## From a controller sample to a validated step

| Operation | Source and entry point | Integration requirement |
| --- | --- | --- |
| Construct the stored control polynomial and charge coefficient rounding | `controllers/strict_injection.py`, `strict_injection`, and `controllers/quad_injection.py`, `inject_controls_s` | Pass the live state, certificate slope and intercept bounds, control indices and sparse engine. Both versions check layouts and binary64 arithmetic. |
| Refine the fixed-slope controller residual | `controllers/residual.py`, `DelayedReLUResidual.evaluate` | Accepts the live weights and biases for a rectified network with widths 4, 25, 25, 2. The slope and the subbox are shared between both residual evaluations. |
| Cover the controller input and combine bounds | `controllers/partition_batch.py`, `partition_chunks` and `evaluate`. `controllers/partition_core.py`, `install`, binds the evaluator to the original driver. `controllers/gate.py` checks selected exact rational points in qualification mode. | The driver adds the controller module directory to the import path and supplies its existing affine evaluator. The production batch covers 256 cells in four batches. |
| Intersect a second bound around an unchanged control polynomial | `controllers/anchored_control.py`, `contract` | The driver supplies the second certificate, its input domain and the existing polynomial and remainder. The function checks domain coverage and returns the contracted remainder. |
| Validate an error tube that grows through the step | `engine/weighted_validation.py`, `validate_failed`, `validate_recentered` and `refine_accepted` | The transfer loader imports this module in the `flowstar_gpu` package namespace. Relative imports resolve to the sparse engine, support range evaluator and directed interval primitives. |
| Transport accumulated errors | `engine/history_factor.py`, `_sr_linear_prepare`, forms point and interval factors. The repository engine's `flowstar_gpu/symbolic_remainder.py`, `propagate(..., strict=True, phi_i_iv=...)`, updates the products and sums the transported columns. | The archived strict-history file matched the staged project engine byte for byte. The engine dispatches optional fused kernels on supported devices and uses the interval implementation on CPU. |

The non-history controllers and weighted module are verbatim copies of their retained component files. The history factor is the exact function extracted from the transferred numeric executor, with its two module imports made explicit. The strict-history code is in the repository engine named above.

## Locate the two selected executions

The double-pendulum width run was `dp_less_ours_order323_no_early_recorded_full100_20261026_001`. Its driver `run_huan_guarantee_transfer_001.py` came from the staged directory `dp_order323_no_early_sources_001`. The receipt `DP323_NO_EARLY_RECORDED_RESULT_001.json` records degrees 3, 2, 3, all 225 boxes completing 100 steps of 0.01, 20 network-bound calls, 300 point sweeps and zero weighted maps. The recorded trajectory supplied widths. Separate production repetitions supplied the comparison times.

The quadrotor used `quad_final_dualmode_order322_sources_001` with degrees 3, 2, 2. The first production identifier was `quad_ours_final322_correctedqa_noreport_production1_full1000_20261027_003`. Its companion receipt `QUAD_FINAL322_PRODUCTION3_CORRECTED_QA_RESULT_003.json` records 1,024,000 accepted lane steps, 50 original network certificates, 50 additional anchor certificates and 2,000 weighted maps over 1,000 steps. The three production runs used the same qualified source with reporting disabled.

The two named receipts and staged directory names identify the original experiment archive. Replaying a complete timing requires the frozen launch directory, controller weights and certificate provider, registered benchmark model and initial boxes, matching PyTorch/CUDA toolchain, patched neural-bound library and compiled sparse execution dependencies. The component check above exercises the bundled method on synthetic inputs. It does not require those benchmark-specific assets.
