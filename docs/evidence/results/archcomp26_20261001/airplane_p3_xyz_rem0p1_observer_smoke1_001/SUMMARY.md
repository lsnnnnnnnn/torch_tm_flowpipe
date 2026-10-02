# Airplane continuous P3：首步 observer 区间拒绝的精确分量

上一独立 run 使用 `x,y,z` 各 `[-0.1,0.1]`、其余 16 个变量各 `[-0.01,0.01]` 的 Picard 余项估计，首步引擎返回 `accepted=true`，随后入口的合并区间检查报 `invalid accepted interval at 1`，但未留下四界。本次仅在独立[入口副本](../airplane_p3_xyz_rem0p1_observer_prep_20261002/run.py)中、**原合并检查之前**保存 12 物理态的 `tube_lo,tube_hi,endpoint_lo,endpoint_hi` 原始 binary64 和逐维检查结果。与上次放宽余项的运行配置直接逐字节比较一致；官方完整初盒、ODE、ONNX、控制顺序、时间步、P3 数值设置均未改变。回调仍使用 eager refinement 路径。本次唯一新运行是远端 `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/airplane_p3_xyz_rem0p1_observer_smoke1_001`，限首个 0.1 秒控制周期，不是完整 `T=2 s`。

[原始 bounds](payload/precheck_bounds_step_0001.npy)是形状 `(1,12,4)` 的 float64 `.npy`；[原始逐谓词记录](payload/precheck_predicates_step_0001.json)与独立重算的[审计](AUDIT.json)一致。**全部 48 个四界值有限，12 个 endpoint 区间自身均有序。** 合并检查失败在以下五个分量，差额均是二进制浮点相邻值的 1–2 步：

| 物理态 | 失败条件 | tube 边界 | endpoint 边界 | 差额 | binary64 步数 |
|---|---|---:|---:|---:|---:|
| `y` | `endpoint_hi ≤ tube_hi` | `0.0867929493774228` | `0.08679294937742281` | `1.3877787807814457e-17` | 1 |
| `z` | `endpoint_hi ≤ tube_hi` | `0.05695675141820998` | `0.056956751418209994` | `1.3877787807814457e-17` | 2 |
| `phi` | `tube_lo ≤ endpoint_lo` | `-0.005828386693158816` | `-0.0058283866931588164` | `8.673617379884035e-19` | 1 |
| `theta` | `endpoint_hi ≤ tube_hi` | `1.004525454048824` | `1.0045254540488242` | `2.220446049250313e-16` | 1 |
| `psi` | `endpoint_hi ≤ tube_hi` | `1.003989249232481` | `1.0039892492324811` | `2.220446049250313e-16` | 1 |

这是稀疏整步 hull 与密集重嵌后的 endpoint 投影在边界处给出不相容浮点界的**直接观测**；具体算术步骤为何产生这些差别，现有记录不能证明。引擎 trace 仍在首步返回 `accepted=true,status=0`；入口随即于原检查抛 `FloatingPointError`。外层[RESULT](RESULT.json)为 `failed/exit1`、未超时、11.651068 秒。由于异常早于保存，[ranges.bin](payload/ranges.bin)和[observations.jsonl](payload/observations.jsonl)均为空：**0/10 个可用已验证子步**。原始候选 tube 的 `phi,theta,psi` 上界也超过安全阈值 1，但它未通过 observer 有效性检查；这既不是全时安全证明，也不是实际轨迹违反安全性质的证据。没有重新启动旧实验或执行完整时域。
