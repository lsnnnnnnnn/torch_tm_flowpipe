# TORA reach-tanh：新 working P3 首控制期门检

原始服务器目录：`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/tora_reach_tanh_official2026_mat_u11_workingp3_firstperiod_20261002_001`；[独立外层监督收据](../tora_reach_tanh_official2026_mat_u11_workingp3_firstperiod_outer_20261002_001/RESULT.json)在相邻目录。这是新 run ID，没有重启或覆盖旧 `engine_linear_leaf_v2` P3 的 TORA tanh 全程。

从固定官方 2026 `nn_tora_relu_tanh.mat` 在本目录构造图内 `u=11f` 的 ONNX，外部缩放 `1/0`；四态 ODE、完整初盒 `[-0.77,-0.75]×[-0.45,-0.43]×[0.51,0.54]×[-0.3,-0.28]` 和 0.5 秒控制周期按[合同门](../../../../ARCHCOMP26_TORA_REACH_EXECUTION_GATE_20261002.md)。[直接来源/CPU 审计](SOURCE_AUDIT.json)确认此 `.mat` 与固定官方文件及旧 Huan 同合同副本逐字节一致，ONNX 为三层 ReLU 加末层 tanh，22 点 MAT/ONNX 最大差 `9.99e-16`，CPU 预检未初始化 GPU。[配置](config.yaml)、[控制器导出收据](controller_plant_u.onnx.json)、[独立 CPU 预检](PREFLIGHT_CPU_ONLY.json)、[启动](START.json)和源入口均保存。

新作业使用 `engine_quad_normalization_center` 的 **working P3 / validation P4**、严格 endpoint/injection、共享作者 CROWN 驱动，物理 GPU3 与 CPU18–19。内层 [RESULT](RESULT.json) 为首周期 50/50 个 `h=0.01` 小步接受、wall **3.537455453 s**；外层 exit 0、wall **5.983031088 s**。[独立扫描](INDEPENDENT_INTERVAL_SCAN.json)核对[原始 50 条四态范围](ranges.bin)与[逐步观察](observations.jsonl)：有限、有序、步号连续、每步 endpoint 包含于同段 tube，首步 tube 覆盖完整初盒。另立全程的前 50 条保存范围与本门检直接逐条相同。

这是 `T=0.5` 数值前缀，不是 `T=5` 结果；没有运行性质 checker，也没有独立端到端浮点 NNCS 证明。新[500 步全程](../tora_reach_tanh_official2026_mat_u11_workingp3_full500_20261002_001/SUMMARY.md)另有独立原始收据。门检壁钟不作全程速度比较。
