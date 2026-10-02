# Double Pendulum more: strict-interior numerical point candidate

This is one new numerical replay from `(θ₁, θ₂, θ̇₁, θ̇₂) = (1.299, 1.299, 1.299, 1.299)`. Each stored binary64 coordinate is strictly inside the official `[1,1.3]` initial interval, avoiding the exact-decimal upper-edge ambiguity of the earlier `(1.3)^4` candidate. The earlier candidate and all 225-box experiments were left unchanged. This replay is **not** a validated trajectory, a strict counterexample, an end-to-end floating-point neural-network certificate, or a four-method timing sample.

The server run used the fixed 2026 `controller_double_pendulum_more_robust.onnx` at `native_dp_more_prep_001/`. Its stored weights and biases are float32, and the copied [script](replay.py) evaluates the graph's `MatMul→Add→ReLU→MatMul→Add→ReLU→MatMul→Add` operators in NumPy float32. It converts each two-component output to float64 and holds it over the following `0.02 s` control period. The four-state continuous plant is the official 2026 `dynamics_dp.m` equation with `m=L=0.5`, `c=0`, `g=1`; the script integrates 20 periods through `T=0.4 s`. This specifies the **numerical replay semantics**; it does not bound all possible ONNX backend rounding behavior.

| Check | Numerical result |
| --- | ---: |
| First sampled 0.005 s violation | `t=0.325`, `θ̇₁=-1.5004299419210523` |
| DOP853 estimated downward crossing of `θ̇₁=-1.5` | `t=0.3248652629941985` |
| `θ̇₁` at `t=0.36` | `-1.6071729076084833` |
| `θ̇₁` at `t=0.4` | `-1.696465435601376` |
| Largest period-end state difference, DOP853 versus RK4 at `h=0.0001 s` | `6.661338147750939e-15` |
| Largest period-end state difference, RK4 `h=0.0001` versus `h=0.00005 s` | `1.0436096431476471e-14` |
| Largest difference in saved float32 controls across these integrations | `0` |

DOP853 used `rtol=1e-12`, `atol=1e-14`, `max_step=0.001 s`. RK4 was run separately at both fixed step sizes, with the controller reevaluated at every period boundary. The agreement is a convergence check, **not** a rigorous numerical error bound. A formal falsification would still need sound control-output and ODE enclosures through a violating time, under one explicitly frozen neural-network arithmetic contract.

Raw files: [81 sampled states](trajectory_0p005.csv), [period controls and results](RESULT.json), the copied [replay script](replay.py), and a convenience copy of the [ONNX model](controller_double_pendulum_more_robust.onnx). The replay actually read the server's fixed model path; no content digest was calculated. Its isolated server directory is `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/dp_more_interior_point_candidate_v1/`, with output in `point_001/`.

Server command:

```sh
PYTHONDONTWRITEBYTECODE=1 /srv/local/shengenli/flowstar_acceleration_20260921T153643Z/nncs_env/bin/python -B /srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/dp_more_interior_point_candidate_v1/replay.py --model /srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_dp_more_prep_001/controller_double_pendulum_more_robust.onnx --out /srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/dp_more_interior_point_candidate_v1/point_001
```
