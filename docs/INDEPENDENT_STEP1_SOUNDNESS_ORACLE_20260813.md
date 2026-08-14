# Independent step-1 soundness oracle — 2026-08-13

Overall status: `INDEPENDENT_STEP1_SOUNDNESS_ORACLE_INCOMPLETE`.

The arithmetic oracle itself closes. `step1_oracle.py` uses only exact
`Fraction` polynomial and interval operations and imports neither tested core.
It proves that Flow*'s staged and Torch's complete construction have the same
fourth complete-O4 image. Affine, quadratic, cubic, and quartic fixtures are
hand checked.

`step1_mpfr_oracle.cpp` is compiled separately against MPFR/GMP. It applies
`MPFR_RNDD` to every lower operation and `MPFR_RNDU` to every upper operation.
The 128/256/512-bit ladder encloses the exact-rational natural interval,
truncation, Picard remainder images, final remainder, segment range, and
endpoint range; all subset signs are stable and non-margin intervals nest as
precision rises. The algorithm is explicitly named
`natural_interval_termwise`; it is not presented as Horner or Bernstein.

For a direct truth enclosure, four point-corner VDP Taylor series are computed
exactly through degree 100. On the complex disk `|t| <= 1/50`, a radius-one
state ball has self-map bounds `0.069` and `0.51444`; its Lipschitz contraction
bound is `0.4864`. Cauchy's tail is at most `Mx/2^100` and `My/2^100` at
`h=1/100`. A sensitivity proof gives `S11,S12,S22>0` and `S21<0`; moreover
`x'>0` and `y'<0` on the certified real region. Therefore the four corners are
formal endpoint extrema and the time-zero/endpoint pair gives the full tube.
This is not sampling.

Approximate displays of that exact enclosure are:

```text
endpoint x  [1.1234179659059609, 1.4243098190029190]
endpoint y  [2.3127356267793000, 2.4330896173407517]
segment  x  [1.1,                1.4243098190029190]
segment  y  [2.3127356267793000, 2.45]
```

Both tools' final step-1 segment and endpoint boxes contain these exact
intervals. Thus Torch's narrower endpoint is formally sound and comes from
legal endpoint dependency preservation.

Gate D nevertheless fails earlier. Against the declared exact input, the
normalized point-coefficient TMs have these missing gaps:

| tool | component | missing lower | missing upper |
|---|---|---:|---:|
| Flow* | x | `1/11258999068426240` | `1/11258999068426240` |
| Flow* | y | `3/5629499534213120` | `0` |
| Torch | x | `11/180143985094819840` | `11/180143985094819840` |
| Torch | y | `219/720575940379279360` | `0` |

This is an explicit `UNDER_ENCLOSURE_WITNESS`. Later wide boxes happening to
contain the true solution cannot repair a missing initial semantic set. The
mandatory stop rule therefore prevents stage-swap propagation and candidate
implementation.
