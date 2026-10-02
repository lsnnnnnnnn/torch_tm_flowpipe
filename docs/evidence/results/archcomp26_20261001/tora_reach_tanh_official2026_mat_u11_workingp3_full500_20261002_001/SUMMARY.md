# TORA reach-tanh：新 working P3 完整数值时域

原始服务器目录：`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/tora_reach_tanh_official2026_mat_u11_workingp3_full500_20261002_001`；[独立外层监督收据和日志](../tora_reach_tanh_official2026_mat_u11_workingp3_full500_outer_20261002_001/RESULT.json)在相邻目录。此新 run ID 在[独立首周期门检](../tora_reach_tanh_official2026_mat_u11_workingp3_firstperiod_20261002_001/SUMMARY.md)50/50 接受后启动，旧四方历史作业均未重启。

用户选定的主合同为 2026 官方 ReLU³/tanh `.mat`、ONNX 图内 `u=11f`、外部缩放 `1/0`、四态 TORA ODE、完整初盒、10 个 0.5 秒控制期至 `T=5`，目标 `x1∈[-0.1,0.2]` 且 `x2∈[-0.9,-0.6]`。[来源与 CPU 审计](SOURCE_AUDIT.json)直接核对固定官方 `.mat`，检查 ONNX 算子及缩放、22 点前向误差小于 `10⁻¹²`，且 CPU 预检未初始化 GPU。保留[配置](config.yaml)、[模型](controller_plant_u.onnx)、[导出收据](controller_plant_u.onnx.json)、[CPU 预检](PREFLIGHT_CPU_ONLY.json)、[启动](START.json)、源码、指标和原始日志。

方法是 `engine_quad_normalization_center` 的**当前 working P3 / validation P4**、严格 endpoint/injection 与共享作者 CROWN 驱动；GPU3、CPU18–19。它与[旧历史 P3](../tora_reach_tanh_historical_fourway_saved_20261002/SUMMARY.md)的 `engine_linear_leaf_v2` 明确是不同代际，旧运行不计作本次 attempt。[内层 RESULT](RESULT.json)记录 10 期/500 小步全部接受、wall **10.998220621 s**；外层监督进程 exit 0，wall **13.456628790 s**。两者都是单次、具名环境时间，不形成四方稳定速度排名。

[独立扫描](INDEPENDENT_INTERVAL_SCAN.json)重读[500 条四态保存 tube/endpoint](ranges.bin)与[500 条逐步观察](observations.jsonl)：步号和 lane 连续、范围有限有序、每步 endpoint 包含于同段 tube，首次拒绝不存在。保存的 `T=5` endpoint 是 `x1=[0.06791520996448479,0.09301949154372924]`、`x2=[-0.8033787322621515,-0.7759834292813406]`，两个完整初盒数值终点区间都在目标内；`x3=[0.0616983865226157,0.08371331701618905]`、`x4=[0.3600040232048114,0.38165749984538183]`。扫描还记录第 487–500 步保存 endpoint 均入目标，但并未对连续时间窗单独运行性质 checker。

[当前 P3 与三方同合同历史结果的四态 T=5 绝对终点 CSV](terminal_current_p3_historical_author_fourway_T5.csv)逐行标记证据代际、来源和目标坐标包含情况。先前的[历史四方图](../tora_reach_tanh_historical_fourway_saved_20261002/SUMMARY.md)画的是**旧 P3**、Huan、Xiangru、原生原始保存范围，不暗换为本次新 P3 图；本次新结果的原始 500 步范围已单独保存。

此结果是完整数值时域加终点目标包含的**数值充分观察**，作者性质 checker 未执行；保存轴盒及有限点控制器预检都不构成独立端到端浮点 NNCS 证明。与旧四方结果的比较须注明 P3 代际和 ODE 工作阶数差异；当前主表只按新运行身份引用。
