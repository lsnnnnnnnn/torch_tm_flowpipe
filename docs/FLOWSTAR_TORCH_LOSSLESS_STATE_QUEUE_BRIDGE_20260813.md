# Flow*–Torch lossless state/queue bridge — 2026-08-13

Serialization status: `SAME_PRESTATE_LOSSLESS_BRIDGE_AVAILABLE`.

Operator-matrix status: schema/operator mismatch; the complete cross-tool 2×2
is not available.  These are deliberately separate conclusions.

## Canonical schema

`flowstar_lossless_state_queue_v1` is an ordered, newline-terminated
`key=value` format.  Every `Real` or interval endpoint uses the exact tuple
`precision:sign:hex_mantissa:binary_exponent`.  It contains:

- every `tmvPre` and `tmv` term, exponent vector, explicit total degree,
  coefficient, and remainder endpoint;
- the complete domain and its dimensional metadata;
- `Symbolic_Remainder.J`, `Phi_L`, scalars, and `max_size`;
- local time, step tables, order, cutoff, target remainder, safety and
  constrained flags, producer, phase, and term-order declaration.

No 15/17-digit decimal is canonical, no physical-range-only export is used,
and no common-box reboxing or sampling participates in roundtrip validation.

## Executed roundtrips

The actual-path exporter captured pre/post-reset fixtures at steps 1, 2, 10,
50, 99, 100, 101, 200, 300, 397, 474, and 632.  All 24 C++ export→import→export
files are byte exact, and all 24 imported states continue one step to exactly
the same decision and canonical next state as the original in-memory object.
This includes fixtures around the Q100 reset.

The Python/Torch exporter maps binary64 values to the same exact dyadic form.
Flow* imports and re-exports the Torch initial state byte-for-byte.  Across the
24 Flow* fixtures plus the Torch fixture, 8658 precision-53 dyadics map exactly
to Python float.  Unit fixtures cover positive and negative zero, normal and
subnormal values, signs, small and large exponents, interval endpoints, and
binary64 maximum.  Missing and duplicate fields, NaN, unknown fields, wrong
dimension, and wrong order are all rejected.

## Same-prestate operator matrix

The lossless container can represent both producers, but their full operator
contracts differ:

| operator / prestate | Flow* prestate | Torch prestate |
|---|---|---|
| Flow* | executed; exact at 1→2, 99→100, 100→101 | executed refusal: schema/operator mismatch |
| Torch | not run: cannot consume complete `Phi_L/J` | native initial step executed |

Flow* VDP state has three components `(x,y,t)`, four TM variables, and a
nonempty three-component `J/Phi_L` queue.  Torch's normal state has two state
components, two variables, and no full consumer for that queue.  Conversely,
the frozen Flow* ODE operator refuses the 2×2 Torch state.  Dropping `t`,
discarding the queue, or reboxing both to a common interval would change the
prestate and is prohibited.

Thus Gate D is genuinely closed as bidirectional serialization/import and
same-producer exact continuation, while Gate E cannot close the complete
cross-operator attribution.  The bridge remains useful and soundly fail-closed;
it does not authorize the source-ledger candidate.
