# TORA reach-sigmoid：官方文件控制器的完整数值时域

新隔离 run ID `tora_reach_sigmoid_official2026_mat_u11_full500_huan_002`。合同为固定官方 2026 `.mat` 四层 sigmoid `4→20→20→20→1`、ONNX 图内 `u=11f(x)`，执行器外部 `scale=1,offset=0`。四态 ODE、完整初盒、0.5 秒控制周期、0.01 秒 ODE 小步和 5 秒数值时域见 [配置](config.yaml)及 [PREFLIGHT](PREFLIGHT.json)。本 profile 与论文合并文字的输出激活不同，**不冒称论文主表结果**。

保存的 [START](START.json)、[RESULT](RESULT.json)、[stdout](stdout.log)、[stderr](stderr.log)、[metrics](metrics.json)、[逐步观察](observations.jsonl)、[原始区间](ranges.bin)来自一次 Huan sparse plant engine 加共享作者 CROWN 驱动运行，GPU3、CPU18–19、保存的 `nncs_env` PyTorch `2.5.1+cu121`。全初盒 10/10 控制期、500/500 小步接受，监督进程 wall `11.285899113863707 s`，作者驱动输出 `time cost: 8.944547`。一次性时间不能作四方速度排名。

[独立区间扫描](INDEPENDENT_INTERVAL_SCAN.json)从 500 条唯一 `(lane,step)` 原始记录核对了有限、有序与每步 endpoint 包含于同一步 tube。保存的 `t=5` 数值 endpoint 为 `x1∈[0.13465642242366743,0.16047432673915857]`、`x2∈[-0.8762353072866234,-0.8507148906441829]`，完全落入目标 `[-0.1,0.2]×[-0.9,-0.6]`；第 496–500 小步的保存 endpoint 也各自落入目标。这是**保存的数值区间中可作 5 秒内到达充分条件的观察**。本运行关闭了性质 checker，未取得作者性质判定，也没有端到端浮点 NNCS 证书。全过程未做哈希校验。

较早 `_001` 用错保存环境，0 个小步；[失败收据](../tora_reach_sigmoid_official2026_mat_u11_full500_huan_001/SUMMARY.md)独立保留。
