# TORA remain：h=0.05 隔离数值 profile

2026-10-02，在读完[原 h=0.1 数值拒绝审计](../../../../ARCHCOMP26_TORA_REMAIN_NUMERIC_STOP_AUDIT_20261002.md)并确认服务器 GPU2 空闲、旧作业已结束后，分别新开一次 Huan 与 Xiangru `h=0.05` 作业。两份[事前意图](INTENT.json)、[Xiangru 追加意图](INTENT_XIANGRU.json)固定了新 ID、资源和首次盒拒绝即停规则；均未重启旧 `full20_001`。外层监督各限时 120 秒。

对照原运行的 `config.yaml`，新 Huan 配置**只有 `ode_step_size: 0.1 → 0.05` 一行改变**；Xiangru 新配置相同。初集及 12 盒覆盖、四态 ODE、官方 2026 ONNX 原始输出与 `x4'=u1-10`、1 秒控制保持、`T=20`、全时四态 `[-2,2]` 性质、Taylor order 3、余项估计、strict plant、CROWN 设置均保持。h=0.05 是**另一数值方法设置**，不补写到固定 h=0.1 四方主表，也不用其时间作同设置排名。

| 方法和设置 | 小步 | 接受盒步 | 保存 tube 全时安全带 | 作者 checker | 外层进程 wall |
| --- | ---: | ---: | --- | --- | ---: |
| 原 Huan h=0.1 | 200/200 已观察 | 2357/2400 | 第 185 步起无法证实 | `Unknown.`；第 190 步首拒 | 8.372 s，失败尝试 |
| 原 Xiangru h=0.1 | 200/200 已观察 | 2357/2400 | 第 185 步起无法证实 | `Unknown.`；第 190 步首拒 | 8.270 s，失败尝试 |
| [Huan h=0.05](huan_full20_firstreject_001/payload/RESULT.json) | **400/400** | **4800/4800** | 全部保存的接受 tube 在带内 | 静默，无盒拒绝 | 10.898 s，单次诊断 |
| [Xiangru h=0.05](xiangru_full20_firstreject_001/payload/RESULT.json) | **400/400** | **4800/4800** | 全部保存的接受 tube 在带内 | 静默，无盒拒绝 | 10.899 s，单次诊断 |

两份新作业的原始 [Huan 外层收据](huan_full20_firstreject_001/RESULT.json)、[Xiangru 外层收据](xiangru_full20_firstreject_001/RESULT.json)、stdout/stderr、实际 YAML、每步观察记录和 `ranges.bin` 均保留。独立标准库扫描 [Huan](huan_full20_firstreject_001/INDEPENDENT_INTERVAL_SCAN.json)、[Xiangru](xiangru_full20_firstreject_001/INDEPENDENT_INTERVAL_SCAN.json)分别读取 4800 条区间，确认每条接受记录有限有序、步末端点落在同段 tube、无盒拒绝、无保存 tube 越出安全带。扫描器的旧 h=0.1 默认行为也对原 Huan 结果只读复核，仍为 200 步、2357 接受盒步。

新两方保存的全时四态 tube union 均为 `x1∈[-0.978065,0.848117]`、`x2∈[-1.004544,0.919148]`、`x3∈[-1.172530,1.031457]`、`x4∈[-1.410581,1.565374]`；终点 union 分别列在原始收据中。数值上，减半 ODE 小步后这两方在同一数学合同上完整走到 `T=20`，且作者 checker 静默、保存 tube 在安全带内。首拒即停分支已装备，但两次新作业都没有触发；旧 h=0.1 收据没有失败状态子类，不能据此断言其根因。两方仍共享控制驱动与大部分数值实现，控制器浮点包络注入没有独立端到端 NNCS 证书。
