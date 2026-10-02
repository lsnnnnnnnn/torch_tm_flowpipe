# Unicycle paper-speed native：独立首周期门检

日期：2026-10-02。新 run ID：`native_unicycle_paper_speed_smoke1_001`；构建为 `native_unicycle_paper_speed_build_002`，不替代历史 `w` 同时进入 yaw/speed 的原生实验。2026 论文合同在此入口为 `x3'=u2-20, x4'=u1+w-20, w'=0`；状态内部次序 `x1,x2,x3,x4,t,u1,u2,w`，控制器只读前四态，raw 两输出各减 20 一次，控制每 0.2 秒刷新。

## 初盒与构建

- 旧 native `Flowpipe(box)` 在 x2 下界有 1 ULP 欠包。新入口保留旧构造器的中心，半径取上下端点两侧有向距离的最大值，不改原库。`build_001`/`preflight_001` 已通过，但其 ledger 在赋值后只写了预定值；`build_002`/`preflight_002` 增加了**实际赋值后系数**读取与一致性检查，作为本次 smoke 使用的版本。两次源码、构建日志与预检收据分别保存于同名目录。
- `preflight_002` 原生进程退出 0，输出 `INITIAL_AFFINE_REPAIRED 1/8`。独立 [有理数审计](../native_unicycle_paper_speed_preflight_002/INITIAL_FRACTION_AUDIT.json)从实际赋值后的 53 位十六进制 center/radius 重构 8 个仿射像，逐维与保存的请求初盒比较：8/8 覆盖，只扩了 x2。这个门检未调用 NN 或 ODE。

## 一次 0.2 秒 smoke

- GPU2、CPU36–39、RPC `127.0.0.1:5107`；[START](START.json) 保存完整入口和环境，runner [RESULT](RESULT.json) 为退出 0、未超时、3.926593 秒。服务器与 native PID 在运行后均已退出。
- [native.log](native.log) 记录 `PERIOD 0 STATUS 2 CUMULATIVE_SEGMENTS 10`、`COMPLETED_PERIODS 1/1` 和 `COMPLETED_SHORT_PREFIX PROPERTY_NOT_APPLICABLE`。`2` 为 Flow* 的 `COMPLETED_SAFE` 内部代码；此处安全集为空，不把代码 2 解释为 reach 证明。[RPC](controller_rpc.jsonl) 恰好 1 次，HTTP 200；实际 NN 输入四态边界含请求初盒。
- [原始范围](ranges.bin) 为 10 条记录。独立 [范围扫描](RANGE_SCAN.json) 检查完整 1×10 网格、数值有限有序、每段终点落在该段 tube 内；全部通过。t=0.2 的物理终点盒为 x1 `[9.288735768120628,9.341513277572146]`、x2 `[-4.263990156726647,-4.210130005466063]`、x3 `[2.469576332690379,2.481368010914232]`、x4 `[1.6790355057749142,1.6917260309052464]`。
- [smoke 初盒审计](INITIAL_FRACTION_AUDIT.json) 再次通过 8/8；只修 x2。没有进行完整 50 周期或 T=10 终点目标检查，性质记 `UNKNOWN`，不能当作完整四方成绩或端到端浮点 NNCS 证书。
