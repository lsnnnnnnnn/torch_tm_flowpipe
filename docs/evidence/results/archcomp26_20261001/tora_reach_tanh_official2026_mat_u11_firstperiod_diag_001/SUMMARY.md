# TORA reach-tanh：官方文件控制器的一期诊断与旧全程复用门

隔离 run ID `tora_reach_tanh_official2026_mat_u11_firstperiod_diag_001`。固定官方 `.mat` 网络为三层 ReLU 隐层、tanh 输出；[显式构造的 ONNX](controller_plant_u.onnx)图内给 plant `u=11f(x)`，执行器外部缩放 `1/0`。[构造收据](controller_plant_u.onnx.json)的 22 点 NumPy 与 ONNX Runtime 最大绝对差为 `9.992007221626409e-16`。[配置](config.yaml)使用四态 TORA ODE、完整初盒、0.5 秒一期、50 个 0.01 秒小步，不评估 5 秒目标。论文合并文字写 sigmoid 隐层，与本官方文件网络不同；本条不是论文主表控制器。

在 GPU3、CPU18–19 的一次 Huan sparse plant 加共享作者 CROWN 驱动运行中，[RESULT](RESULT.json)为 50/50 小步接受，监督进程 wall `3.7501907348632812 s`。[原始 START](START.json)、[stdout](stdout.log)、[stderr](stderr.log)、[metrics](metrics.json)、[逐步观察](observations.jsonl)和[区间](ranges.bin)保留。[独立扫描](INDEPENDENT_INTERVAL_SCAN.json)确认 50 条区间有限、有序，endpoint 均包含于同一步 tube；`t=0.5` 的 `x1∈[-0.8862507517991547,-0.8585960832389572]`、`x2∈[-0.016283813893203318,0.012513910249841838]`。这只是短前缀，没有性质判定。

[只读旧记录复用审计](HISTORICAL_REUSE_AUDIT.json)逐项比较本次 50 步、四态、tube/endpoint 的上下界，与 2026-09-23 冻结 Huan 全程 CSV 的 **800/800 个数逐值完全相等**。旧 Huan、Xiangru、P3 原始 `result.json` 各记 500/500 接受并完成；旧 native 记 10 个控制期、500 条范围，作者原始日志写 `VERIFIED`。此前控制器审计也已确认旧 ONNX 与官方 `.mat` 的全部权重、偏置、ReLU/ReLU/ReLU/tanh 激活逐元素相同；旧 tanh 与本 profile 的 plant 缩放均为 `11f`。因此继续启动同合同的 500 步新作业会重复原实验，**没有启动**。旧 native `VERIFIED` 只按其终点 checker 解读，不能代替跨时间窗完整性质判定或端到端浮点 NNCS 证明；旧四方数据也不能直接冒充论文字面 sigmoid 隐层主表。

复用审计入口为 [compare_historical_first50_nohash.py](compare_historical_first50_nohash.py)，读取旧保存区间 CSV 和四份原始结果；全过程未做哈希校验。
