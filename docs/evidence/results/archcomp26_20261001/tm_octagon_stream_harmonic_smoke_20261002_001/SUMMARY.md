# Accepted-step eight-direction streaming smoke

This isolated CPU plant-only smoke used `x1'=x2`, `x2'=-x1`, initial box
`x1∈[1,1.2], x2=0`, `h=0.05`, order 4, dependency-preserving mode. The
new callback recorded [JSONL](harmonic_stream.jsonl) immediately after each
accepted segment; 3/3 segments reached `T=0.15`. The [run receipt](RUN.json)
records 108 finite exact-solution samples inside all four directional
intervals. The [unobserved control](BASELINE_COMPARE.json) independently
validated 3/3 and matched every saved tube and endpoint directional support.

[Tube PNG](harmonic_stream_tube.png) · [Endpoint PNG](harmonic_stream_endpoint.png)
· [geometry JSON](harmonic_stream.geometry.json) · [MATLAB tube](harmonic_stream_tube.m)
· [MATLAB endpoint](harmonic_stream_endpoint.m) · [PDF tube](harmonic_stream_tube.pdf)
· [PDF endpoint](harmonic_stream_endpoint.pdf)

The raw server output remains at
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/tm_octagon_stream_harmonic_smoke_20261002_001`.
The independent source snapshot was staged at the sibling
`tm_octagon_stream_harmonic_source_20261002_001`; the four executed source
files were copied back under [source_used_for_smoke](source_used_for_smoke).
The generated MATLAB scripts have not been run in MATLAB or Octave.

The saved supports come from correlated CPU Taylor models. Polygon vertices
are floating-point display intersections. This demonstrates the stream
boundary and plot path for one short plant, not an NNCS result, a 2026 QUAD
octagon, or an independent end-to-end certificate. A JSONL stream establishes
only its accepted prefix; a separate run receipt must establish the planned
full horizon.
