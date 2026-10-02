# Taylor-model octagon plotting smoke (2026-10-02)

This is a **new three-step, plant-only plotting smoke** on the saved research
server, not a QUAD or ARCH-COMP26 four-method experiment. Before launch, the
new run path and octagon process were checked absent. No previous solver run
was restarted or overwritten. The original server directory is
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/tm_octagon_harmonic_smoke_20261002_001`.

The existing CPU Taylor-model solver ran `x1'=x2`, `x2'=-x1` from
`x1∈[1,1.2]`, `x2=0`, with `h=.05`, order 4, dependency-preserving mode.
All three segments validated to `T=.15`; the solver wall was `0.148691 s`.
At the accepted segment boundary, the exporter evaluated `x1`, `x2`,
`x1+x2`, and `x1-x2` as Taylor models, producing eight half-plane bounds for
each tube and raw endpoint. Observation/geometry took `0.030253 s`.

- [Directional geometry](harmonic_3step.geometry.json) contains the numerical
  supports, display-only polygon vertices, and both tube and endpoint views.
- [Tube PNG](harmonic_3step.png), [tube PDF](harmonic_3step.pdf),
  [tube MATLAB script](harmonic_3step.m), and [tube render receipt](harmonic_3step.render.json).
- [Endpoint PNG](harmonic_3step_endpoint.png),
  [endpoint PDF](harmonic_3step_endpoint.pdf),
  [endpoint MATLAB script](harmonic_3step_endpoint.m), and
  [endpoint render receipt](harmonic_3step_endpoint.render.json) were made
  from saved geometry, without another numerical solve.
- [Run receipt](harmonic_3step.run.json),
  [source used for the numerical smoke](source_used_for_smoke.py), and
  [independent sampled check](AUDIT.json).

The third endpoint polygon display area is `0.0050171863`, versus
`0.0059104555` for its coordinate box (ratio `0.8488663`). The independent
check tested 108 exact harmonic-oscillator sample cases against all four
directional intervals and checked every displayed polygon vertex against the
saved half-planes. Finite samples and floating-point plotting vertices are
not a universal certificate; the direction intervals, not the vertices, are
the source data. Neither MATLAB nor Octave was available on the server, so
the `.m` files were generated but not executed there.

The updated repo exporter additionally passed a synthetic one-segment API
export/render check; that check did not run a solver. The current QUAD/P3/native
archived observers hold axis-aligned intervals only, so this result does not
turn those existing QUAD plots into octagons. No property was specified or
checked for this harmonic smoke. No content digest or hash was computed.

This saved smoke used the initial exporter snapshot in `source_used_for_smoke.py`
and its raw `tau=h` endpoint. A later code review found that some normalized
insertion modes add symbolic output remainders after that substitution; the
repo exporter now uses the accepted `final_tm` propagation endpoint. This
historical smoke remains raw-endpoint evidence and has not been regenerated or
presented as a test of those other modes.
