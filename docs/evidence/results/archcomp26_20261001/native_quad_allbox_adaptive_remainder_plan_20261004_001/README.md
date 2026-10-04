# Saved all-box adaptive remainder plan — 2026-10-04

**Result: all 1,024 first-control boxes and 3,072 output rows have an
offline, conditionally verified endpoint plan. The native production gate
remains CLOSED.** No endpoint from this plan has yet been applied in a new
native process, and no network, CROWN, ONNX or ODE evaluation was performed.

The authoritative [RESULT.json](RESULT.json) records 196,608 exact-rational
affine vertex checks, 3,072 endpoint minimality checks, and a 29.430 s local
analysis wall time. This is an analysis cost, not a benchmark or solver time.
[INPUT.json](INPUT.json) names all five saved inputs and the scope before
execution. The new program does not import or execute any archived checker.
It preserves the verified CSV prefix and stops at the first new failed
precondition, proposed containment, or minimality check.

## Reference and exact meaning of minimum

Let the saved input models be `x_i = p_i(z) + R_i`, with their exact binary64
coefficients lifted to rational numbers, and let `q_j(z)` be the saved output
polynomial before the old fixed pad. Let `a_i` denote the saved transported
binary32 slopes, and `[b_L,b_U]` the already proposed conditionally corrected
binary32 bias interval. The new ledger covers this reference set:

```
sum_i a_i (p_i(z) + R_i) + [b_L,b_U],   z in the saved normalized domain.
```

The saved output polynomial is held fixed. Exact extrema of
`d(z) = sum_i a_i p_i(z) - q_j(z)` give the required remainder interval

```
L = b_L + min_z d(z) + sum_i min(a_i R_i)
U = b_U + max_z d(z) + sum_i max(a_i R_i).
```

For saved **original_tm_remainder** `[r_L,r_U]`, choose the greatest binary64
`new_L <= min(r_L,L)` and the least binary64 `new_U >= max(r_U,U)`.
Thus “minimum” means the smallest outward binary64 endpoint changes that
preserve that original interval and contain the above reference while holding
the polynomial fixed. It is not the smallest bound for the true network, nor
the smallest correction using unused slack in the original CROWN inequalities.
The archived `[-2^-50,0]` pad is not retained or used as the baseline.

The verifier independently evaluates the input and output affine polynomials
at all 64 active vertices for every output, with exact sign-dependent input
remainder extrema. Affine functions attain their extrema at those vertices,
so this checks the whole normalized box, not a numerical sample of a nonlinear
map. It checks agreement with the separate coefficient-based extrema and
checks that moving either proposed endpoint one representable value inward
would violate reference coverage or preservation of the original remainder.

Before each proposal the program checks the original source box, saved/native
RPC equality, input TM source coverage within the RPC box, independent input
symbols, same-slope replay equality, transported slope equality, corrected
bias equality, conditional coefficient transfer and native center/radius
trace. All input values are exact rationals lifted from binary64 or binary32;
all ledger bounds and margins are stored as exact rational strings or exact
binary64 hexadecimal strings.

## Observed endpoint changes

All 3,072 output rows require at least one change relative to their original
un-padded remainder. Changes can be required on **either side**.

| Output | Lower changed | Upper changed | Max lower expansion | Max upper expansion | Max lower/upper ULP moves |
|---|---:|---:|---:|---:|---:|
| 1 | 562 | 558 | 3.1988300897012323e-15 | 3.1086244689504383e-15 | 461 / 448 |
| 2 | 575 | 557 | 1.734723475976807e-18 | 1.734723475976807e-18 | 1 / 1 |
| 3 | 532 | 575 | 1.734723475976807e-18 | 1.734723475976807e-18 | 1 / 1 |

Counts for the two sides overlap. Lane 0 requires lower moves of 48, 1, and
1 ULPs and no upper moves, matching the archived three-output diagnostic.
Lane 1/output 1 requires 300 lower ULPs from the original remainder. The
maximum output-1 lower change occurs at lane 944, and the maximum upper
change at lane 644. [PLAN.csv](PLAN.csv) contains every exact proposed pair,
required rational pair, expansion, ULP count and nonnegative final margin.
These are newly computed plans from saved data, not new native observations.

## Minimal next native construction-only gate

Use a **new isolated run identity** and a copy of the saved all-box construction
driver, linked to the same existing recentered candidate library. Keep its
source-box reconstruction, `.asFloat()` slopes and corrected biases, double
center/radius order and TM addition order unchanged. Remove the old fixed-pad
operation. For each output:

1. Compare the newly constructed input models, polynomial, original remainder,
   slopes, biases and center/radius with the saved exact trace; refuse the first
   mismatch and retain its prefix.
2. Parse `proposed_lower_hex` and `proposed_upper_hex` as exact binary64 values
   and assign `Interval(new_L,new_U)` directly to the final remainder. Direct
   endpoint replacement avoids another rounded interval addition. Require that
   the new interval contains the original interval and is finite and ordered.
3. Save full-precision original and resulting models, and independently verify
   both exact endpoint equality to the plan and the same conditional inclusion
   on the actual native result. Stop at the first numerical refusal.

No CROWN server, network call, controller inference or plant integration is
needed for that gate. A successful native gate would establish only application
of this conditional, saved-first-batch construction plan. Any plant audit with
these changed controls needs its own new evidence; old plant receipts do not
transfer to wider controls automatically.

## Evidence boundary and runnable check

Every passing row assumes the original real-arithmetic CROWN affine
inequalities are valid on the saved RPC box. The recorded `uA=lA` equality and
conditional transport audit support the chosen reference but do not prove
CROWN numerical soundness, actual NN output containment, runtime model identity,
later controls, any ODE step or the full-time property. This is neither an
independent NNCS certificate nor a production repair.

The small synthetic check verifies directed rounding, a successful affine
inclusion and rejection of a deliberately insufficient remainder:

```sh
python3 -B plan_remainders.py --self-check
```

The recorded full command was `python3 -B plan_remainders.py > stdout.log
2> stderr.log`, and exited 0. The script refuses to overwrite INPUT, RESULT or
PLAN. A fresh reproduction requires a new sibling diagnostic directory; there
is no reason to rerun this already recorded analysis without changing its
inputs or method. No hashes or digest verification were used.
