# Airplane discrete：精确零恒等式后的有界前缀

新运行目录：`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/airplane_discrete_paper_euler_exactzero_prefix_001`。原始 [`RESULT.json`](RESULT.json) 和 [`EVENTS.jsonl`](EVENTS.jsonl) 已保存；旧的一步 [`RESULT.json`](../airplane_discrete_paper_euler_interval_smoke1_001/RESULT.json) 保持原样。新入口为 [`tools/archcomp26_airplane_discrete_euler_exact_zero_nohash.py`](../../../../tools/archcomp26_airplane_discrete_euler_exact_zero_nohash.py)。

固定官方 `dynamics.m` 先定义 `a3=[p;q;r]`，再用 `mat_2*a3` 赋值 `dphi,dtheta,dpsi`。完整初盒有 `p=q=r=[0,0]`，且 `cos(theta)∈[0.5403023058681393,1]` 排除零，故首步三个角度导数**代数上恰为零**。新入口只在旧态这三个分量均为精确 `[0,0]` 时，对它们应用 `x+0=x` 恒等式；其余状态与控制器的转移没有改变。此处理消除了旧一步运行在闭带边界产生的一格外舍入溢出。

完整 12 维初盒单盒出发，旧态调用固定 12→6 ONNX，按 `xₖ₊₁=xₖ+(1/10)f(xₖ,NN(xₖ))` 同步更新，目标最多 20 次转移。`k=1` 端点四个性质坐标均落在 `[-1,1]`；`sy₁∈[-0.08414709848078979,0.31453676396724295]`，`phi₁=theta₁=psi₁=[0,1]`。于是区间诊断接受了 `k=0,1`，完整安全端点前缀为 **1/20 次转移**。

首个实质性 `Unknown` 出现在 `k=2`，随后立即停机，未尝试 `k=3`。此时 `sy₂∈[-6.656617804770197,7.3050397433887015]`、`phi₂∈[-0.709066234709197,1.8099688057992496]`、`theta₂∈[-0.3654609903923575,1.4125903523280192]`、`psi₂∈[-0.6086419525553797,1.7737862537382103]`。四个包络均远越过安全带，而本次 `exact_zero_angular_identity=false`。这次 `Unknown` 来自实质性的区间外包络宽度，不是 `k=1` 那个零项浮点舍入伪差；它仍**不是**真实不安全反例。

控制器第一次盒输出已宽，例如 `Fx∈[-197.1800425294152,146.23170353268074]`，导致 `k=1` 速度盒约达 ±20；第二次 `Fx∈[-5039.7940174270125,3551.2518876200397]`，说明独立区间传播严重失去相关性。结果只说明这条 CPU directed interval 诊断入口的前缀，不属于 P3、Huan、Xiangru 或 Flow* native 四方法主表。其三角函数外舍入余量未在本次 `torch 2.5.1+cu121` 环境独立复核，因此也不是端到端浮点证书。若要复现官方参赛者离散运行，仍缺其控制取样与离散转移源码或同等权威记录。

原始 `RESULT.json` 的 `property_check_indices=0..20` 记录预定检查范围；实际只执行并检查了 `k=0,1,2`，其中 `k=2` 为首个 Unknown，随后停止。
