# TORA reach-sigmoid：Xiangru 官方合同完整数值时域

远端原始 run_dir：`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_xiangru_full500_001`。

用户选择的 2026 官方主合同：四层 sigmoid `.mat` 控制器、图内 `u=11f`、外部缩放 `1/0`；完整初盒、10 个 0.5 秒控制期、500 个 0.01 秒 ODE 小步。合同与 CPU 控制器检查见 [配置](config.yaml)和 [PREFLIGHT](PREFLIGHT.json)。较早的[独立一期门检](../tora_reach_sigmoid_official2026_mat_u11_xiangru_firstperiod_001/SUMMARY.md)已接受 50/50 小步。

[原始 START](START.json)、[RESULT](RESULT.json)、[stdout](stdout.log)、[stderr](stderr.log)、[metrics](metrics.json)、[逐步观察](observations.jsonl)和[区间](ranges.bin)来自一次 Xiangru sparse plant engine 加共享作者 CROWN 驱动运行，GPU3、CPU18–19。全初盒 10/10 期、500/500 小步接受，监督进程 wall `11.335332102142274 s`，驱动 `time cost: 8.812399`。一次时间不具四方排名资格。

[独立区间扫描](INDEPENDENT_INTERVAL_SCAN.json)确认 500 条唯一、有限、有序保存记录，所有 endpoint 在对应 tube 内。`t=5` 数值 endpoint 为 `x1∈[0.13465642242366743,0.16047432673915857]`、`x2∈[-0.8762353072866234,-0.8507148906441829]`，落入目标 `[-0.1,0.2]×[-0.9,-0.6]`；保存的第 496–500 步 endpoint 也落入目标。Huan 与 Xiangru 两份 500 条原始区间各 76000 字节，逐字节相等。这些是保存数值区间的充分条件观察；本运行未启用作者性质 checker，也没有端到端浮点 NNCS 证书。未做哈希校验。
