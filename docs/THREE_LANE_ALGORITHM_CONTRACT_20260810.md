# Three-lane algorithm contract

Date: 2026-08-10

## Track N: native reproduction

Each tool runs its own unmodified official entry point and configuration.

- Flow*: stock `benchmarks/continuous/vanderpol/vanderpol.cpp`, adaptive
  O4, native symbolic remainder, T10.
- DiffReach: `run_dyn.py config/ct_dyn/van_der_pol.yaml --sim --ver`, fixed
  support, B64, h=0.01, 1,000 steps.
- Torch: authoritative complete-O4 VDP adaptive request at the frozen VDP
  lineage.

Native rows establish environment and semantics.  They are not ranked when
initial partition, output, validator, step, or soundness contracts differ.

## Track M: matched mathematical contract

The common plant is `x'=y`, `y'=y-x-x^2*y`, initial box
`[1.1,1.4] x [2.35,2.45]`, and requested physical horizon T10.  Only settings
exposed by a native backend may change.  A field the backend cannot express is
`UNAVAILABLE`; no ODE adapter is added to rescue a row.

Every output separately records raw endpoint, last full-segment tube,
full-horizon tube, validation/certificate status, first failed time, and first
failed reason.  Endpoint bounds are never compared to tube bounds as though
they were the same object.

## Track F: in-framework factorial

The required Torch rows are:

1. complete-O4 + current Flow*-raw-compatible validator + current normalized
   insertion carry;
2. exact DiffReach VDP fixed support + two polynomial Picard iterates + DR-RP
   + DiffReach-equivalent normalization/symbolic carry;
3. complete-O4 + the evidence-selected generic carry candidate.

Crossing a validator with a representation is allowed only when candidate
remainder, retained/overflow semantics, and acceptance predicate have a typed
mathematical meaning.  Otherwise the cell is `INCOMPATIBLE`, with a reason.

## Required row schema

Each row contains tool/source SHA/execution kind, RHS hash, initial set,
coordinate order, support name/hash, Picard depth/order semantics, validator,
seed and every acceptance round, step/partition/carry/symbolic/range policies,
dtype/device, requested and validated horizons, completion/certificate/property
status, first failure, endpoint/tube/polynomial/remainder widths, accepted and
rejected work, algorithmic work, cold/warm/core/process time, peak memory,
soundness class, and artifact hashes.

## Eligibility

A full-horizon row is eligible only when its mathematical contract is known,
validated horizon equals requested horizon, every required inclusion passes,
outputs are finite, no undeclared fallback/repair occurred, endpoint and tube
semantics are not mixed, applicable properties were checked, and numerical
soundness is explicit.  One-step ratios are never headline results.

The allowed soundness labels are exactly:

```text
formally outward by construction
independently outward replayed for exact benchmark workload
empirically sampled only
unsound/ineligible
unknown
```
