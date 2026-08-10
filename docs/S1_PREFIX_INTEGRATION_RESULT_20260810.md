# S1 complete-O4 prefix integration result

Date: 2026-08-10  
Primary outcome: `S1_PREFIX_REJECTS_BEFORE_TERMINAL`

## Result

S1 is no longer an empty-history terminal accounting exercise. The canonical
accepted-boundary state owns the ordinary normalized remainder, K16 structured
columns, normalization metadata, accepted boundary index, unique source
identities, and insertion/eviction count from `t=0`.

The checksum-verified historical schedule contains 307 accepted steps and the
terminal failed attempt. L0 reproduces it exactly. L1 and L2 share the first
164 accepted boundaries, through `t=4.738198114669049`. Every committed L2 row
passes source decomposition, materialization conservation, no-double-count,
finite, normalized-domain, endpoint publication, and tube publication gates.
At the next proposed step, the historical lane accepts
`h=0.03661680691961388`, but S1 rejects it. The adaptive helper can accept
`h=0.01830840345980694`; that poststate is off schedule and discarded.

## Represented set and coordinates

At a boundary, the normalized remainder is

```text
R = R_ordinary + sum_k Phi_k J_k.
```

`J_k` is stored in the source's physical coordinates. `Phi_k` maps that
physical source into the current normalized coordinates. The full current
normal-to-physical forward scale is owned by `FlowstarNormalFlowpipeState`, and
the exact inverse scale is stored with S1. A boundary scale enlargement caused
by outward rounding transforms the right polynomial, ordinary interval, every
Phi output row, and the reset scale together; it never weakens the `[-1,1]`
gate with a tolerance.

Old columns are propagated by the outward affine degree-one map. The complete
degree-four polynomial difference is independently enclosed over the endpoint
domain and over the full tube domain `tau in [0,h]`. Every degree-two through
degree-four route involving a structured perturbation is included once in the
nonlinear residual. The endpoint specializes `tau=h`; the tube retains the
whole local-time interval and is separately required to contain the endpoint
structured image.

Only `polynomial_truncation` and `integration_overflow` may create live
columns. Their interval centers remain ordinary and their symmetric parts are
structured. Every other declared source is ordinary. K16 eviction selects the
oldest live column, materializes its propagated contribution exactly once,
then installs the new uniquely identified source. The event file records the
ownership path for all 328 insertion/eviction events attempted on the common
prefix and first divergent call; committed-state counts stop before the
discarded call.

## Capacity and checkpoint facts

- First full K16 state: boundary 16.
- First eviction: boundary 17.
- Largest observed single eviction maximum-component width: boundary 70,
  `0.001549673642858923`.
- Final committed L2 event count: 312 at boundary 164.
- Boundary-164 v2 checkpoint full SHA:
  `9162f267fcdcf44ca7bb9acfa73975eb8f4f4b80c03ca217aac2f07450cd585b`.
- Save-load-save payload and manifest: byte-identical.

K32 is not authorized because the terminal GO gates were never reached,
regardless of eviction attribution.

## Answers to the promotion questions

The previous local result proved that an empty-history source split could make
the ordinary terminal y interval fit while the materialized total still
contained the raw image. It did not prove prefix ownership, nonlinear
propagation, publication, checkpoint continuity, or causal horizon benefit.

The integrated prefix did not reach the historical terminal prestate.
Consequently there is no same-pre-state terminal A/B, no authorized fresh
horizon ladder, no fresh validated horizon, no +0.5 promotion, no T10 result,
and no integrated second-system result. What remains blocked is the
representation decision exposed at boundary 164, not a missing accounting
field or a license to tune step size.

Evidence: `outputs/s1_prefix_integrated_complete_o4_20260810/20260810T095423Z`.
