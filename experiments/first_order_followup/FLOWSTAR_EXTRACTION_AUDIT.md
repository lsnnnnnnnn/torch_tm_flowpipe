# Flow* extraction and remainder audit

## Reproduction

The frozen `correctness_checks.json` identifies the smallest completed failure
as Riccati with `h=0.02`, `T=0.1`, fixed order 2.  At the first endpoint,
`t=0.02`, the analytic interval is:

```text
[0, 0.10020040080160321]
```

The transformed Flow* export was:

```text
[-5.005041474153472e-05, 0.10020035141645481]
```

The focused audit in `run_flowstar_audit.py` reproduces all five bad upper
bounds.

## Representation-layer result

The generated C++ audit records the raw domain, `tmvPre`, `tmv`, official
`Flowpipe::intEval`, `compose`, `compose_normal`, transformed
`TaylorModelFlowpipe`, tube ranges, direct `t=h` ranges, substitution ranges,
step indices, and absolute times.

The following agree to floating-point print precision:

- raw `Flowpipe::compose` and transformed `TaylorModelFlowpipe`;
- raw and transformed whole-segment ranges;
- direct local-time evaluation and `TaylorModelVec::evaluate_time`;
- state ordering, segment indexing, and accumulated absolute time.

The maximum raw/transformed endpoint delta in the audit is zero; the largest
direct/substitution delta is roundoff scale.  Therefore the failure is not an
omitted or doubled `tmvPre`, transformed-object corruption, a wrong time power,
an endpoint reset to zero, or an off-by-one export.

Flow*'s plot/dump path is downstream of the same transformed Taylor-model
flowpipes, so it cannot repair an enclosure already made too small by
advancement.

## Root cause and experiment-local correction

In `flowstar-toolbox/Continuous.cpp`, fixed-order `Flowpipe::advance`:

1. seeds the configured remainder;
2. proves the first Picard image is a subset of that candidate;
3. replaces the candidate with that image;
4. repeatedly computes another Picard image; and
5. can stop after assigning or encountering an image without proving that the
   final returned image is a self-map.

The relevant lifecycle is visible around the first inclusion check and
refinement loop (the repeated fixed-order implementations have the same
pattern).  For the failing Riccati step, the returned refined remainder makes
the endpoint upper bound roughly `4.94e-8` too small.

The follow-up does not modify the external Flow* checkout.  It calls the public
step API, requires a successful initial inclusion, restores the configured
candidate remainder that was actually proved self-mapping, then composes and
extracts.  This is conservative.  Riccati and harmonic use `1e-4`; Van der Pol
needs `1e-3` for the initial inclusion.

The order-2 guard remains intact.  Flow* order 1 is still reported unsupported.

## Gate

`run_flowstar_audit.py` fails unless:

- every safe Riccati endpoint contains the analytic result;
- deterministic Riccati samples are contained;
- raw and transformed extraction agree;
- endpoint and tube rows remain distinct.

The full result stores the generated source, stdout, representation rows,
domains, and `flowstar_correctness.json` under `logs/flowstar_audit/`.
