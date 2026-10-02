# TORA reach-sigmoid：P3 官方合同完整数值时域

远端原始 run_dir：`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_p3_full500_001`；[独立外层监督收据](../tora_reach_sigmoid_official2026_mat_u11_p3_full500_outer_001/RESULT.json)另存于同级目录。此新 run ID 在已完成的[一期门检](../tora_reach_sigmoid_official2026_mat_u11_p3_firstperiod_001/SUMMARY.md)之后启动，没有覆盖或重启旧作业。

使用用户选定的 2026 官方四层 sigmoid `.mat`、ONNX 图内 `u=11f`、外部缩放 `1/0`、论文四态 TORA ODE、完整初盒，10 个 0.5 s 控制期到 `T=5`。working P3 / validation P4 的 plant engine 以严格 endpoint/injection 接共享作者 CROWN 驱动；GPU3、CPU18–19。源文件、模型、[配置](config.yaml)、[启动](START.json)和[CPU 预检](PREFLIGHT.json)与本次原始输出一并保存。

[内层 RESULT](RESULT.json)记录 **500/500** 个 0.01 s 小步接受、10/10 控制期完成、内层 wall **11.278381096 s**；外层监督进程 exit 0、wall **13.741790566 s**，含进程启动与回收。[独立保存区间扫描](INDEPENDENT_INTERVAL_SCAN.json)重读[全部 500 条范围](ranges.bin)和[观察](observations.jsonl)，确认四态 tube/endpoint 均有限、有序，lane 与步号连续，endpoint 没有超出同段 tube。保存 `T=5` 的 endpoint 为 `x1∈[0.13452581591208781,0.16060475161859264]`、`x2∈[-0.8764374242552789,-0.8505126711494655]`，完整落入目标 `[-0.1,0.2] × [-0.9,-0.6]`；扫描还记录第 496–500 步 endpoint 包含于目标。

这个结果是完整数值时域及终点包含目标的候选充分观察。运行时没有启用性质 checker；保存区间与共享作者驱动也不构成独立端到端浮点 NNCS 证明。它是一次新进程，不能据单次壁钟判定四方稳定速度排名。没有做内容摘要校验。
