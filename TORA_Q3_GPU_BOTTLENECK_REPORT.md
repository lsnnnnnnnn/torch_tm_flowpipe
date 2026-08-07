# TORA-Q3 GPU bottleneck report

This report separates raw Kineto events from program-issued scalar extraction. Raw Chrome traces remain private; public CSV files contain sanitized stage and source attribution.

## Baseline source attribution

The baseline one-step trace contains 140,012 paired `aten::item` / `aten::_local_scalar_dense` events and 19,226 `aten::to` events. Streaming attribution covers 100% of all targeted events.

The three largest synchronization sites are:

| source | events | share |
|---|---:|---:|
| `validate_dense_subdivision_cover` identity-cover validation | 56,550 | 40.389% |
| first cell-enumeration list comprehension | 22,464 | 16.044% |
| second cell-enumeration list comprehension | 22,464 | 16.044% |

Together these identity-cover checks account for 72.478% of baseline synchronization events. The largest conversion site is `_power_interval_bounds` in `sin_tm` with 6,552 `aten::to` events.

## Optimization iterations

| trace | Kineto item/local events | `aten::to` | interpretation |
|---|---:|---:|---|
| baseline | 140,012 | 19,226 | generic identity-cover validation dominates |
| iteration 1 | 6,585 | 12,137 | identity cover proved by construction; same-layout `.to()` removed |
| iteration 2 | 269 | 9,500 | device predicates deferred to a phase boundary |
| iteration 3 | 253 | 80 | cached device integer/scalar tensors remove 99.584% of `.to()` |
| final compiled | 77 | 80 | transient diagnostic ledger moved out of inner math |

Kineto's final 77 item/local events are not all program-issued synchronization. A separate dispatcher audit observes exactly three `_local_scalar_dense` calls in the full logical step: local acceptance, composed acceptance, and the fail-closed validation-batch exit. The other 74 Kineto observation events do not pass the Torch dispatcher. Gate P1 therefore uses the auditable program-issued count of 3; both counts remain public.

## Remaining runtime bottleneck

Compiling the fixed-shape 32-term point sine/cosine enclosure is bitwise equal to eager output and has zero graph breaks, but it compiles only that pure tensor boundary. Caching immutable domain/basis monomial bounds and suppressing transient ledgers reduce B48 one-step time to 0.508397 s, a 4.621x improvement over the frozen 2.349308 s baseline.

The remaining cost is thousands of small outward interval kernels in natural range evaluation and K2 plus ten-round remainder arithmetic (`nextafter`, `mul`, `all`, `min/max`, slicing, and stacking). The next concrete engineering change is a pure tensor boundary for the whole natural-range/remainder phase, with its existing operation order and outward-error contract preserved. Directly compiling the Python dataclass/ledger entry is a negative result: it hits Dynamo cache limits and invalidates deferred ledger construction, so it is not used.
