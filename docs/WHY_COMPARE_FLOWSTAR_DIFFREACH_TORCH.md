# Why compare Flowstar, DiffReach, and Torch TM

## Executive rationale

The three implementations occupy deliberately different points in the design space. Flowstar is the stock, interval/MPFR-based reference for validated Taylor-model flowpipes. DiffReach fixes a small tensor-friendly polynomial basis and uses JAX batching/JIT. This repository implements a sparse, complete-total-degree Torch Taylor model and explores GPU-oriented batching. A useful comparison therefore asks whether tensorization preserves the plant, basis, remainder, validation, completion, and output contracts before asking which implementation is faster.

This audit uses three lanes and never mixes them in one ranking:

- `native_reproduction`: the tool's own model, runner, scheduler, partitions, and output semantics;
- `matched_plant_backend`: an exactly stated plant and initial set with every unmatched basis or validation field disclosed;
- `native_end_to_end_certificate`: controller, plant, property, completion, and timing all included.

The current formal comparison remains disabled because three cross-tool gates are still false. No winner, Pareto frontier, or speedup is produced.

## Representation and implementation audit

- Flowstar's official Van der Pol program creates a preconditioned `Flowpipe` from the box and a symbolic-remainder queue of size 100 (`benchmarks/continuous/vanderpol/vanderpol.cpp:42-53`, `:67-89`, Flowstar SHA `b85a3211748cb77b736fe4ad42ee02d8d2b81148`). Its advance path constructs Picard polynomials, seeds a candidate remainder, checks subset inclusion, and refines accepted remainders (`flowstar-toolbox/Continuous.cpp:956-1028`).
- Torch stores sparse coefficients by exponent tuple. Multiplication forms exponent sums and `mul_truncate` moves terms above total degree to a separate dropped polynomial (`src/torch_tm_flowpipe/polynomial.py:223-250`). Cutoff evaluates removed terms over the domain before adding them to the remainder (`:252-274`).
- DiffReach stores dense `c`, `L`, and `Lt` arrays for the restricted basis `1`, `z_i`, and `tau*z_i` (`src/polynomial.py:15-31`, DiffReach SHA `dd628eb443b517d6415de93e7035b4baef73963e`). It is not a complete total-degree-two basis.
- Torch and Flowstar both use an explicit local-time variable. Torch appends `tau in [0,h]` in `flowpipe_step_from_tm` (`src/torch_tm_flowpipe/flowpipe.py:3494-3508`); DiffReach makes time coordinate zero and state generators `[-1,1]` (`src/reachability.py:58-71`).
- Torch interval operations compute with ordinary Torch operations and nudge the final bound using `torch.nextafter` (`src/torch_tm_flowpipe/interval.py:36-41`, `:135-165`). This is a safeguard, not a proof that every intermediate GPU/CPU operation was directed-rounded.
- DiffReach uses ordinary JAX arrays. Its native constructors default to `float32` (`src/reachability.py:44-80`; `src/taylor_model.py:45-55`), while the current audit adapter enabled x64 and observed float64 outputs. Neither route exposes an MPFR/directed-rounding contract.
- DiffReach performs two polynomial Picard iterations, seeds an absolute remainder, and then calls remainder Picard (`src/reachability.py:155-183`). The initial contraction flag is computed once; failed later updates keep the old interval rather than terminating (`src/picard.py:13-58`).
- Torch's Flowstar-compatible target-remainder path explicitly constructs raw multiplication/truncation/cutoff remainders and performs the target subset test (`src/torch_tm_flowpipe/flowpipe.py:2608-2912`). Adaptive steps return only after validation and finite range checks; otherwise they halve to the minimum and return a failed segment (`:3972-4065`).
- Torch's normalized insertion composes the endpoint into a right map, computes center/scales, resets on `[-1,1]`, and optionally carries a symbolic queue (`src/torch_tm_flowpipe/flowpipe.py:1304-1538`). The current long-horizon ablation compares constant versus range-midpoint right-map centering without changing validation.
- DiffReach composes the local model with a symbolic linear parameterization and bounds either the right endpoint or the step (`src/reachability.py:138-195`). Its public `verify` uses `jax.lax.scan` for all requested steps and returns the contraction flags without stopping (`:198-232`).
- DiffReach's `eval_interval` always takes the Horner return; the following quad-box return is unreachable (`src/taylor_model.py:167-171`). Its optional affine truncation returns the truncated polynomial with the old remainder, while the intended discarded-term addition is commented out (`:173-190`). The option is false by default (`src/settings.py:1-6`) and is excluded from primary claims.
- Flowstar's official output path transforms flowpipes for plotting and writes GNUPLOT segment rectangles (`benchmarks/continuous/vanderpol/vanderpol.cpp:115-130`). These are segment/tube boxes, not fixed-time endpoints.
- Torch now exports `endpoint_raw`, optional `endpoint_tightened`, `last_segment`, and `full_tube` as separate fields; `collapsed_endpoint` and `repaired_hull` cannot fall back to either (`experiments/three_tool_deep_study/common.py`, schema `cir-1.2.0`).
- DiffReach's native `verify` returns interval boxes for every scan step, not a lossless raw endpoint Taylor model, last-segment object, and whole-tube metadata. The adapter records those unavailable fields rather than inventing them.
- The historical Torch dense path was an Euler/kernel feasibility prototype. It
  has now been replaced in the canonical module by true Picard/self-map
  validation; the production identity is `hybrid_dense_core`, not the frozen
  `torch-dense-prototype` artifact. It remains ineligible for a T=10 claim
  because the unmodified lane stops at 6.3172908799330765.
- Controller bounds are outside the matched plant suite. The public CROWN-Reach C++ route reads controller coefficients via `asFloat()` before constructing Flowstar Taylor models (`CROWN-Reach/src/CrownReach.cpp:88-115`). No private Xiangru source was available, so that public path is not treated as the private 2026 experiment.

## Failure and timing semantics

Completion means the actual validated horizon equals the request, every step's validation passed, outputs are finite, and no fallback occurred. A JAX scan returning 1,000 boxes is not completion when a contraction flag is false. A Flowstar process exiting normally is not completion when `Result_of_Reachability::isCompleted()` is false. A Torch prefix is not renamed as `T=10`.

The runtime boundary is `total_configuration_v2`: configuration/setup, core propagation, range projection, endpoint construction, validation, reset/carry, and completion checking must be included; compile/JIT, cold, steady, and fresh-process time remain separately visible. The current run does not have 1 cold plus 10 matched steady runs for all eligible backends, so no runtime ratio is admissible.

## Reproduction entry points

```bash
conda run -n py11 python experiments/three_tool_reaudit/flowstar_vdp_reproduction.py \
  --flowstar-root /srv/local/shengenli/flowstar \
  --output outputs/three_tool_reaudit/20260804T060058Z/raw/flowstar_official_vdp \
  --repetitions 4

OMP_NUM_THREADS=1 DIFFREACH_ROOT=/srv/local/shengenli/DiffReach \
conda run -n diffreach312 python \
  experiments/three_tool_reaudit/diffreach_native_reproduction.py \
  --horizon 10 --rhs-route canonical-polynomial-adapter --steady-runs 1 \
  --output outputs/three_tool_reaudit/20260804T060058Z/raw/diffreach_adapter_vdp_t10.json
```

The exact provenance is in `outputs/three_tool_reaudit/20260804T060058Z/manifest.json`; the eight evidence decisions are in `gate_evidence/index.json`.
