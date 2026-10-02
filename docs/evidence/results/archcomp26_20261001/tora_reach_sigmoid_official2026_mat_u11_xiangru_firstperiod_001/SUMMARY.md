# TORA reach-sigmoid：Xiangru 官方合同一期门检

远端原始 run_dir：`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_xiangru_firstperiod_001`。

用户选择的 2026 官方主合同：四层 sigmoid `.mat` 控制器、图内 `u=11f`、外部缩放 `1/0`，完整初盒、四态 TORA ODE，见 [配置](config.yaml)与 [CPU 预检](PREFLIGHT.json)。此 run ID 只执行 Xiangru plant engine 的第一个 0.5 秒控制期，不启用 5 秒目标 checker。

[原始 START](START.json)、[RESULT](RESULT.json)、[stdout](stdout.log)、[stderr](stderr.log)、[metrics](metrics.json)、[观察](observations.jsonl)和[区间](ranges.bin)显示 GPU3、CPU18–19 的 50/50 小步接受；监督进程 wall `3.521614311262965 s`。[独立扫描](INDEPENDENT_INTERVAL_SCAN.json)确认 50 条唯一、有限、有序区间，所有 endpoint 在相应 tube 内。保存 `t=0.5` 的 `x1∈[-0.8850144075262375,-0.8573825037301664]`、`x2∈[-0.0066841849575308195,0.02190112928341277]`。这不是完整时域或性质结果；该门检通过后在独立新 run ID 执行[全程](../tora_reach_sigmoid_official2026_mat_u11_xiangru_full500_001/SUMMARY.md)。未做哈希校验。
