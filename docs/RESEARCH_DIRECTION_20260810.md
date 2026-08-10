# Research direction: Flow*, DiffReach, and generic Torch Taylor models

Date: 2026-08-10

## Mainline

The project builds and explains a sound, generic, PyTorch-native Taylor-model
flowpipe backend.  Flow* is the mature complete-polynomial and long-horizon
precision reference.  DiffReach is the restricted fixed-shape, tensor-friendly
reference.  Torch must express both design points and support controlled,
sound representation/carry changes between them.

Every result names the full algorithm contract:

```text
representation + polynomial construction + validator + carry/reset
+ step policy + partition policy + numerical backend
```

An ambiguous shared `order` label is prohibited.

## Ordered research questions

1. **Semantic reproduction:** reproduce pinned DiffReach fixed-support
   polynomial/Picard/DR-RP semantics and the existing Torch complete-support
   semantics without changing completion or validation.
2. **Precision versus throughput:** compare support points under one VDP plant
   contract using enclosure widths, certificate survival, failure horizon,
   work, memory, cold time, and warm time.
3. **Long-horizon dependency:** identify the earliest causal cross-step
   operation where Torch loses future-useful dependency relative to stock
   Flow*.
4. **Generic improvement:** test one sound, tensor-compatible carry or
   representation change, without a VDP/TORA formula.
5. **Real GPU value:** measure batch size crossover only after mathematical
   eligibility and explicit soundness classification.

## Current hypotheses and stop rules

The frozen VDP line excludes natural/Horner/subdivision-only range evaluation,
more remainder rounds, higher order, smaller step alone, and range-midpoint
centering as the primary contribution.  Evidence selection must first compare
rejected candidates and right-map parents in a common coordinate basis.

The preferred candidate is complete polynomial normalized carry if endpoint
substitution/normalization is the earliest loss.  Structured overflow carry is
eligible only if retained polynomial carry is already intact and immediate
intervalization is causal.  QR/affine preconditioning is eligible only if the
common-basis evidence identifies conditioning as causal.

Promotion requires all containment/equivalence gates and at least one declared
benefit threshold from the round specification.  A sound negative result is
kept as an experimental lane and is not made default.

## Evidence hierarchy

One-step rows diagnose semantics only.  Short and medium horizons establish
survival and failure behavior.  Every full-horizon claim comes from an
independent fresh request.  Failed prefixes retain their exact validated
horizon and first failure.  Only eligible completed rows can support runtime,
speedup, or Pareto language.

## Frozen and historical lines

- Main research line: generic Torch TM versus native Flow* and DiffReach.
- Frozen stress-test reference: homogeneous TORA complete-Q3.
- Historical/rejected: DEF-CERT default replacement, adaptive basis without a
  new isolated gate, obsolete one-step and generated-harness comparisons.
