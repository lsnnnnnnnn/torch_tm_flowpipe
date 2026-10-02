# Airplane continuous P3：完整初盒首次数值拒绝的内部记录

此前同阶 P3 smoke 已记录第一个 `0.01 s` 子步 `accepted=false`，但没有内部失败分量。本次使用独立 run ID `airplane_p3_first_reject_trace_smoke1_001`，仅为该拒绝补引擎记录；没有重启旧 run 或启动完整时域。远端原始目录是 `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/airplane_p3_first_reject_trace_smoke1_001`。原始 [START](START.json)、[RESULT](RESULT.json)、[stdout](stdout.log)、[stderr](stderr.log)、[payload](payload/RESULT.json)、[逐步观察](payload/observations.jsonl)和[内部 trace](payload/refinement_trace.jsonl)均已保存；[AUDIT](AUDIT.json)列出直接扫描所得数值。

合同仍是官方连续模型的**一个未分割完整 12 态初盒**：`x,y,z,r,p,q=0`，`u,v,w,phi,theta,psi∈[0,1]`，固定 2026 ONNX 六控制 `(Fx,Fy,Fz,Mx,My,Mz)`，控制周期 `0.1 s`，物理目标时域 `T=2 s`。本诊断只允许运行首周期的十个 `h=0.01 s` 子步；工作 P3 / 严格同阶验证 P3、初始余项估计 `[-0.01,0.01]`、物理 GPU 3、CPU 14–17、外层 300 s 时限。引擎 `Settings.refinement_callback` 记录自映射和重心化尝试，会选择 eager refinement 路径，因此 **12.250824 s 外层 wall 仅是诊断耗时**，不能与无回调运行做速度比较。入口和已加载基础源码保存在 [source](source/)；[TRACE_METHOD](payload/TRACE_METHOD.json)标明回调影响。

首个子步的 `initial_self_map` 直接记录 `initial_self_map_ok=false`、`bad_nonfinite_mask=false`、尝试步长 `0.01`。19 个分量中仅物理位置 `x,y,z` 的 Picard 提议未包含在各自 `[-0.01,0.01]` 余项估计中：`x` 为 `[-0.04854096687445621,0.050098924759290696]`，`y` 为 `[-0.04713869620976931,0.04807390038902599]`，`z` 为 `[-0.021909162333042152,0.0224412930789049]`。随后保存的四次 `recentered_self_map` 尝试（编号 0–3）均再次只在 `x,y,z` 失败，`bad=false`；最终 `advance_return.accepted=false`、状态码 1。服务器实际加载的 `sparse_exec.py` 把状态码 1 定义为 `FAILED_CONTRACTION`。七条 trace 的数值全有限，记录的余项和提议区间全有序。

外层 [RESULT](RESULT.json)是 `failed/exit1`、未超时；内层记录首步拒绝、**0/10 已接受子步**。原始 [ranges.bin](payload/ranges.bin)为空，无可用流管、终点区间或性质检查。因此不能据此判定官方全时 `y,phi,theta,psi∈[-1,1]` 成立或不成立，也不能把数值自映射失败解释为真实轨迹违反性质。原始 `T=2 s` 全程没有运行；没有独立端到端浮点 NNCS 证明。
