# Airplane continuous P3：仅放宽 x/y/z 余项估计的一次首周期诊断

此前完整初盒同阶 P3 在首个 `h=0.01 s` 子步因 `x,y,z` 的 Picard 自映射提议不包含于 `[-0.01,0.01]` 估计而拒绝。实际服务器驱动的 `parse_rem_est` 和引擎的 `build_rem_est` 支持按 19 个声明变量分别给估计。本次在独立[数值 profile](../airplane_p3_xyz_rem0p1_prep_20261002/PREFLIGHT.json)中仅将 `x,y,z` 三行改为 `[-0.1,0.1]`；其余 16 行仍为 `[-0.01,0.01]`。逐字段比较确认其余配置和此前同阶 smoke 一致，包括**一个官方未分割完整 12 态初盒**、ODE、固定 ONNX、六控制注入、`h=0.01 s`、工作/严格验证 P3 和安全表达式。`run.py` 仅把 20 个控制周期限定为首个 `0.1 s` 周期。完整 `T=2 s` 没有运行。

唯一新 run ID 是 `airplane_p3_xyz_rem0p1_trace_smoke1_001`；远端原始目录为 `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/airplane_p3_xyz_rem0p1_trace_smoke1_001`。已镜像原始 [START](START.json)、[RESULT](RESULT.json)、[stdout](stdout.log)、[stderr](stderr.log)、[内层 RESULT](payload/RESULT.json)、[配置](payload/config.yaml)、[43 条引擎 trace](payload/refinement_trace.jsonl)和[扫描审计](AUDIT.json)。回调选择 eager refinement 路径，故 **11.700555 s 外层 wall 不是可比较的正式方法时间**。

首个数值步的内部 `initial_self_map_ok=true`，19 个分量全部包含、`bad_nonfinite_mask=false`；`x,y,z` 提议分别为 `[-0.05077170298099375,0.05244133050503847]`、`[-0.049382686976157525,0.0503423319237774]`、`[-0.0227573486719841,0.02341137885883231]`，均落在新的 `[-0.1,0.1]` 中。已记录的 refinement 和 early-weighted 路径通过，引擎 `advance_return.accepted=true`、状态码 0 (`ACTIVE`)。43 条 trace 中记录的数值均有限。

**这不是一个已保存的接受流管。** 紧接着入口在计算 12 物理态的整步 tube/endpoint 后，于独立检查抛出 `FloatingPointError: invalid accepted interval at 1`，外层 `failed/exit1`、未超时。该检查合并了区间有限性、有序性和 endpoint 包含于 tube 的条件；原入口没有把触发条件及 `bounds` 写盘，故不能根据目前证据确定究竟哪项失败。异常发生在写入观察和范围记录前：`observations.jsonl`、`ranges.bin` 均为 0 字节，**0 个可用已验证子步**。原始失败保留，未重复启动。官方全时安全性质未评估；此结果既不证明安全，也不构成真实轨迹反例或独立端到端浮点 NNCS 证明。
