# Airplane continuous: fixed-ONNX high-corner point replay

The 64-box first-step gate stopped at property `UNKNOWN` for cell `111111`:
its accepted numerical tube crossed the safety upper bound in
`phi,theta,psi` ([saved gate](../native_airplane_binary6_coverage_gate_20261002_008/README.md)).
This separate, bounded **point replay** checks one actual initial state at
that cell's upper corner, `u=v=w=phi=theta=psi=1`, with all other physical
states zero. It does not replace any original experiment.

The [script](replay.py) reads the fixed official 2026
`controller_airplane.onnx` from the server, checks its graph, and evaluates
its 4 MatMul, 4 Add and 3 ReLU nodes directly in float32 at the 12-state
point. The six raw outputs are applied in official
`(Fx,Fy,Fz,Mx,My,Mz)` order and held over `t∈[0,0.01]`. The plant equations
follow the saved official `dynamics.m`, including the source state order
`(x,y,z,u,v,w,phi,theta,psi,r,p,q)`. This is a numerical point calculation,
with no relaxed controller output substituted for the ONNX output.

The raw [RESULT](RESULT.json) gives the ONNX output
`[-3.3967220783,4.7935113907,13.7066135406,-0.3058854043,-1.1542012691,0.1472433954]`.
It lies within the earlier CROWN affine-plus-bias bounds at this point.
Independent DOP853 and Radau integrations used `rtol=1e-12`, `atol=1e-14`,
and maximum step `0.0001 s`; their largest difference over 11 saved sample
times was `1.33e-15`. The [DOP853 trajectory](trajectory_DOP853.csv) and
[Radau trajectory](trajectory_Radau.csv) both end at `t=0.01` with
`phi≈0.9999153`, `theta≈0.9999626`, `psi≈0.9999175`; none of their 11
sampled angles exceeds `1`.

This point replay **does not find an unsafe sample at the chosen corner**.
It cannot show that all trajectories in cell `111111` are safe, that no
crossing occurs between saved samples, or that the interval tube's overlap
with the unsafe side is merely wrapping. The all-box `T=2` property remains
Unknown. The saved model path and numerical versions appear in
[START](START.json) and [RESULT](RESULT.json); no content digest was used.
