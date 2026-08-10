# Causal Flow*/Torch divergence on Van der Pol

Date: 2026-08-10  
Run ID: `20260810T025910Z`

## Result

The first native schedule split occurs after the last common accepted state at
`t=0.18187433604506256`, on the proposed step
`h=0.019615177354506262`. Torch accepts the candidate; Flow* rejects it in y
and accepts `h=0.0098075886772531311` instead. The earliest decision-changing
stage is the **raw candidate Picard remainder before polynomial-roundoff is
added**. It is not a coefficient-coordinate mismatch, endpoint substitution,
right-map normalization, or floating-point polynomial roundoff.

| lane | raw x candidate | raw y candidate | minimum x/y target margin | decision |
|---|---|---|---|---|
| stock Flow* | `[-5.3757103669508146e-6, 7.073210412034566e-6]` | `[-1.0366239882151062e-4, 1.0359846643018429e-4]` | `9.292678958796544e-5`, `-3.662398821521699e-6` | reject |
| Torch complete O4 | `[-1.961520735450628e-6, 1.961520735450628e-6]` | `[-9.14291532216261e-5, 9.358938647674799e-5]` | `9.803847926454938e-5`, `6.4106135232520195e-6` | accept |

Flow*'s polynomial-roundoff interval is only
`[-8.378898847772202e-22, 0]` in x and
`[-1.1074878945164581e-17, 4.480863180864205e-17]` in y. Its y raw remainder
has therefore already crossed the `[-1e-4, 1e-4]` target before that roundoff
is included.

## Observation-only stock hook

The exact observer patch is preserved as
`experiments/flowstar_causal_observer_20260810.patch.b64`; the decoded patch
SHA256 is
`9d7a9edeaba07dfb819b4e054cd2e473fa598ada72331e5327b8a3567e5ef3f8`.
It applies cleanly to stock Flow* commit
`b85a3211748cb77b736fe4ad42ee02d8d2b81148` and records the adaptive Picard,
candidate, subset, endpoint, insertion, right-map, and symbolic-queue stages.
Replay-only term injection is guarded by separate environment variables and is
not used on the logged/unlogged equivalence path.

The final observer build used image SHA
`6549fefc...` and binary SHA
`6e4d4af60154239d7f281c367337f6ff52958ed146fb3ac1956b104b31e7f2ba`.
Logged and unlogged official output is identical:

| artifact | logged SHA256 | unlogged SHA256 | result |
|---|---|---|---|
| `vanderpol_t_x.plt` | `63facf7f12f58c0e034942e7d568bba2bea62cf37f2027332b9a1fd61f6c4bd4` | same | byte-identical |
| `vanderpol_t_y.plt` | `c734e3427ccea50d4c373ce69df7e887046d01732a839519ae0f6550522d6533` | same | byte-identical |
| normalized stdout | same | same | only elapsed-time text removed |
| stderr / exit / accepted schedule | empty / 0 / 290 | empty / 0 / 290 | identical |

The logged observer JSONL SHA256 is
`fd545841691aeccb04fba6dca5f7c9e2d93829a2dc7f5616e44e5c9dd0b19109`.
This establishes that the observation path changes no stock decision or
official numerical artifact.

## Frozen Torch state and common basis

The clean Torch checkpoint is
`03_flowstar_causal_divergence/torch_causal_checkpoint_final/torch_causal_checkpoint.json`
with SHA256
`dda8c07f6e9999542ae8bbc2fb6a38ab858a18e5f214be6a5e4b3781c22d093d`.
The observer saw all four Picard iterates and the watched final coefficient
tensor is bit-identical to the production result.

The generic affine common-basis transform explicitly accounts for state
center, state scale, local time, and exponent order. Analytic affine/quadratic
and round-trip tests pass. At the causal candidate the largest transformed
coefficient midpoint error is `8.88e-16` in x and `1.421e-14` in y; the largest
interval enclosure error is `1.421e-14` and `7.105e-14`. Natural polynomial
ranges and constant centers agree at roundoff scale. The different
normalization scales—Flow* `[0.1425544460888901, 0.2039824671893622]`, Torch
`[0.142644343197362, 0.2042119841652547]`—are representational, not causal.

## Controlled substitutions

| counterfactual | receiving decision | attribution |
|---|---|---|
| Flow* polynomial in Torch validator | accept, essentially native Torch margin | polynomial coefficients do not cause split |
| Torch polynomial in Flow* validator | reject, essentially native Flow* y margin | receiving remainder semantics cause split |
| Flow* candidate remainder in Torch subset test | reject | candidate remainder is decision-changing object |
| Torch endpoint in Flow* normalization | center unchanged; scales become `[0.1425552522996264, 0.20399979248372554]`; still reject | endpoint normalization not causal here |
| Flow* right map in Torch next step | current decision unchanged; native and replayed next step accept | right-map carry not causal at first split |
| Torch right map in 80-digit replay | conversion error below `1.94e-17`; physical polynomial difference reaches `1.227e-4` x and `2.732e-4` y | material later dependency difference, not first split |

The common-basis comparison SHA256 is
`9279e7c5ef8b9f78a0cedcbc8ff402d5b5fea2f250abb5ea9921c9d7d7e0ddd2`;
the counterfactual registry SHA256 is
`fb31c1d48de1a216779100812c6977aafdb0d57a4499a0e29c3f04a6c8c812bd`.

## Consequence for the improvement choice

The first split is a native-validator raw-remainder difference, but the
receiving validators are intentionally not made identical. The selected F1
experiment instead addresses the separately quantified later dominant source:
cross-step intervalization during normalized insertion. Near the Torch terminal
boundary, y integration-overflow width is `1.0839579370510149e-4`, polynomial
truncation width is `8.131373677071774e-5`, and the terminal y subset margin is
`-1.99995911680722e-5`. This satisfies the goal's rule permitting a clearly
quantified later dominant source; it does not mislabel F1 as a fix for the
first native-validator split.

## Evidence

Raw observers, exact replay inputs, common-basis coefficients, and all
counterfactual outputs are under
`outputs/mainline_realignment_20260810/20260810T025910Z/03_flowstar_causal_divergence/`.

