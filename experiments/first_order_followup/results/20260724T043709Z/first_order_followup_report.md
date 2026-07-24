# First-order follow-up: correctness and matched bases

## Executive result

All 24559 analytic endpoint checks and all
130104 deterministic trajectory checks passed with zero
violations.  The sampled checks are bug-catching evidence, not a formal proof.
The frozen baseline artifact remained byte-for-byte unchanged during the run.

## Confirmed facts

- The smallest frozen Flow* failure was Riccati at `h=0.02`, `T=0.1`; the first
  failing endpoint was `t=0.02`.  The exact upper bound was
  `0.10020040080160321`, while the exported upper bound was
  `0.10020035141645481`.
- Raw `Flowpipe::compose`, transformed `TaylorModelFlowpipe`, direct endpoint
  evaluation, and local-time substitution agree to the recorded audit
  tolerance.  The extraction gate passes:
  `True`.
- The Flow* fault is its fixed-order remainder-refinement lifecycle: the first
  candidate is proved self-mapping, then a refined Picard image is accepted
  without proving that the new image self-maps.  The experiment retains the
  already-proved candidate.  It does not remove Flow*'s order-two guard.
- DiffReach projection ranges each `t²` and `t·z` term over `[0,h]×[-1,1]^n`,
  shifts the interval midpoint into `c`, adds the residual to the pre-existing
  independent remainder, leaves `L` unchanged, and zeros `Lt`.  Unit tests cover
  both coefficient signs, asymmetric time, multiple generators, preservation,
  and the zero-`Lt` identity.  A stock-kernel transcription parity test covers
  one step for both affine-flag settings.

## Experiment-supported conclusions

The current Torch dependency-preserving order-one path repeatedly keeps the
same generator coefficients while the discarded `τ·ξ` rotation terms
accumulate in the interval remainder.  Range-only and fresh affine reset absorb
the endpoint box into new affine generators every step, so their smaller
remainder is a reparameterization advantage, not evidence that throwing away
dependencies is fundamentally superior.

Matched affine-carry results:

| tool | system | completed_steps | requested_steps | successful_horizon | final_endpoint_width_max | exact_reference_violations | sample_violations |
|---|---|---|---|---|---|---|---|
| diffreach_experimental_strict_affine | harmonic | 990 | 1000 | 9.9 | 3987.3947586926747 | 0 | 0 |
| flowstar | harmonic | 450 | 1000 | 4.4999999999999485 | 1.9836458385553004 | 0 | 0 |
| torch_tm_flowpipe | harmonic | 1000 | 1000 | 10.0 | 56261.49596108419 | 0 | 0 |
| diffreach_experimental_strict_affine | riccati | 100 | 100 | 1.0 | 0.1140429756033784 | 0 | 0 |
| flowstar | riccati | 100 | 100 | 1.0000000000000007 | 0.13185402992023038 | 0 | 0 |
| torch_tm_flowpipe | riccati | 100 | 100 | 1.0 | 0.11783746006995535 | 0 | 0 |
| diffreach_experimental_strict_affine | van_der_pol | 125 | 400 | 0.625 | 8.153538232960464 | 0 | 0 |
| flowstar | van_der_pol | 16 | 400 | 0.08 | 0.33492992467844473 | 0 | 0 |
| torch_tm_flowpipe | van_der_pol | 134 | 400 | 0.67 | 388.35633446128924 | 0 | 0 |

Finite-basis Torch ablation:

| system | basis | completed_steps | requested_steps | successful_horizon | final_endpoint_width_max | steady_step_time_s |
|---|---|---|---|---|---|---|
| harmonic | B1 | 1000 | 1000 | 10.0 | 56261.49596108419 | 0.01571940677240491 |
| harmonic | B2 | 1000 | 1000 | 10.0 | 4462.172298785031 | 0.020264299120754004 |
| harmonic | B_DR | 1000 | 1000 | 10.0 | 4462.172298785031 | 0.020306662656366825 |
| riccati | B1 | 100 | 100 | 1.0 | 0.11783746006995535 | 0.021875188685953617 |
| riccati | B2 | 100 | 100 | 1.0 | 0.1147466600142902 | 0.028418265748769045 |
| riccati | B_DR | 100 | 100 | 1.0 | 0.1147466600142902 | 0.028400284238159657 |
| van_der_pol | B1 | 134 | 400 | 0.67 | 388.35633446128924 | 0.0567113240249455 |
| van_der_pol | B2 | 355 | 400 | 1.7750000000000001 | 908.8722979213908 | 0.08371231378987432 |
| van_der_pol | B_DR | 355 | 400 | 1.7750000000000001 | 908.8722979213908 | 0.0834366362541914 |

Van der Pol basis horizons:

| basis | completed_steps | requested_steps | successful_horizon | final_endpoint_width_max |
|---|---|---|---|---|
| B1 | 134 | 400 | 0.67 | 388.35633446128924 |
| B2 | 355 | 400 | 1.7750000000000001 | 908.8722979213908 |
| B_DR | 355 | 400 | 1.7750000000000001 | 908.8722979213908 |

Under the literal complete-total-degree definition used here, B_DR and B2 often
coincide after endpoint substitution and affine box reset.  The observed runs
therefore do not confirm the proposed large harmonic B_DR-vs-B1 mechanism as a
pure basis effect, nor the Riccati `τ·ξ²` hypothesis (that monomial has total
degree three and is absent from both B_DR and B2).  Van der Pol does show a
material B_DR/B2 horizon improvement over B1; see the table above.

## Implementation-specific effects

- Torch is sparse eager float64 CPU, DiffReach is shape-static JAX CPU, and
  Flow* is compiled C++ with MPFR intervals.  Timing is implementation
  throughput, not an algorithmic speed ratio.
- Flow* Protocol B uses order-two local construction and stepwise affine
  lowering.  Protocol C carries Flow*'s complete degree-two representation.
- DiffReach's Protocol-C row is explicitly labeled restricted
  quasi-quadratic—not complete total degree two.
- The strict DiffReach adapter is experiment-local; the external checkout is
  unchanged.

## Remaining hypotheses and limitations

- `torch.float64` interval operations use explicit outward `nextafter` in the
  relevant kernels but are not MPFR proofs.
- Van der Pol trajectory checks do not prove enclosure soundness, and a common
  cross-tool defect/Jacobian certificate remains future work.
- Interval-valued DiffReach polynomial coefficients are not supported by the
  upstream representation, so projection tests use point coefficients.
- Adaptive top-K and a dense fixed-shape Torch kernel were optional and were not
  added; no TORA experiment was run.

## Provenance and reproduction

Repository SHAs: `{"diffreach": "dd628eb443b5", "flowstar": "b85a3211748c", "followup": "206b7d316770", "frozen_baseline_worktree": "13e5eece2002"}`.

```bash
./experiments/first_order_followup/run_smoke.sh
./experiments/first_order_followup/launch_background.sh
```

The full command is `run_all.sh <result-directory>`.  Raw rows,
validation JSON, generated C++ sources/logs, plots, and timing components are
stored beside this report.
