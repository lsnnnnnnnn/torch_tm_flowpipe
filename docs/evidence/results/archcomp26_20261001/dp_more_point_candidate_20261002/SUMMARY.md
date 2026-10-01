# Double Pendulum more: independent numerical point candidate

This is a **single point trajectory**, not an interval proof, an end-to-end floating-point NN certificate, or a four-method runtime sample. It uses the fixed 2026 `controller_double_pendulum_more_robust.onnx` copied here from the server's `native_dp_more_prep_001` directory. Its graph is float32 `[N,4]→[N,2]`, with MatMul/Add/ReLU/MatMul/Add/ReLU/MatMul/Add and no preprocessing nodes. No content digest was calculated.

The initial point `(θ₁,θ₂,θ̇₁,θ̇₂)=(1.3,1.3,1.3,1.3)` belongs to the full `[1,1.3]^4` set and to the saved 225-box grid's last cell. The [script](replay.py) evaluates the ONNX operators in NumPy float32 at each `0.02 s` period start, holds `(T₁,T₂)` fixed, and integrates the official continuous four-state ODE for 20 periods. DOP853 uses `rtol=1e-12`, `atol=1e-14`, `max_step=0.001`; separate fixed-step RK4 integrations use `h=0.0001` and `0.00005 s`.

| Check | Numerical result |
| --- | ---: |
| First saved 0.005 s grid violation | `t=0.325`, `θ̇₁=-1.501479829624199` |
| DOP853 estimated downward crossing of `θ̇₁=-1.5` | `t=0.32453650913459947` |
| `θ̇₁` at `t=0.36` | `-1.6082814694345662` |
| `θ̇₁` at `t=0.4` | `-1.697533320680845` |
| Max four-state endpoint difference, DOP853 vs RK4 `h=0.0001` | `8.43769498715119e-15` |
| Max four-state endpoint difference, RK4 `h=0.0001` vs `0.00005` | `1.0436096431476471e-14` |
| Max controller difference across those integrations | `0` in saved float32 outputs |

The numerical point at `t=0.36` lies inside the saved Huan lane-224 endpoint interval for `θ̇₁`, `[-1.6351055822633862,-1.5172577882223077]`; that lane's whole-step tube is `[-1.6351055832064725,-1.5029748409727741]`. This cross-check is consistent with the prior [Huan/Xiangru early-stop evidence](../author_dp_more_v1/SUMMARY.md): both ran the full 225-box initial grid, saved 72/80 ODE substeps, and stopped after the checker printed `Unsafe.`. The existing one-period Huan smoke already covered all 225 boxes, so no interval experiment was restarted here. A rigorous falsification claim still needs a validated concrete trajectory/controller evaluation with numerical error bounds or another accepted certificate.

Raw files: [period/control details](RESULT.json), [81 sampled states](trajectory_0p005.csv), [script](replay.py), and the copied [ONNX controller](controller_double_pendulum_more_robust.onnx). The server originals are under `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/dp_more_point_candidate_v1/`.

Server command (after copying `replay.py` into that directory):

```sh
PYTHONDONTWRITEBYTECODE=1 /srv/local/shengenli/flowstar_acceleration_20260921T153643Z/nncs_env/bin/python -B /srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/dp_more_point_candidate_v1/replay.py --model /srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_dp_more_prep_001/controller_double_pendulum_more_robust.onnx --out /srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/dp_more_point_candidate_v1/point_corner_001
```
