# Order-2 open gates

The cheap single-step diagnostic supports only the recorded
`validation_rejected / remainder_self_map_failed` outcome for the stated
configuration and `stock-plus-gcc15-compat` backend.

Open before any broader claim:

- reproduce against a clean, exact unmodified-stock build where supported by
  the compiler environment;
- independently verify official-program versus generated-stock fields;
- confirm whether the first-step multiplication remainder decomposition
  matches the upstream implementation's intended order-2 semantics;
- validate endpoint, accepted-segment, and full-tube exporters separately;
- repeat full-horizon order-2 work only after the single-step self-map issue is
  understood;
- compare effective basis and retained degree rather than nominal order alone;
- keep patched audit backends excluded from primary comparison.

The current smoke does not authorize a sweep, a claim that Flowstar lacks
order-2 support, or any cross-tool ranking.
