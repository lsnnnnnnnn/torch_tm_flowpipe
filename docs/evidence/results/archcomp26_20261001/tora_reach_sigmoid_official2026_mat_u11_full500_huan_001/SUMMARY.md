# TORA reach-sigmoid 官方文件 profile：环境入口失败

新隔离 run ID `tora_reach_sigmoid_official2026_mat_u11_full500_huan_001`，完整初盒、拟运行 10 个 0.5 秒控制期。官方 `.mat` 四层 sigmoid，ONNX 图内 `u=11f(x)`，外部缩放 `1/0`；这不是已选定的论文主表控制器。

CPU 控制器预检通过并保存了 [PREFLIGHT](PREFLIGHT.json) 与 [START](START.json)，但执行器初始化发现误选的 `crownreach28` 环境为 PyTorch `2.8.0+cu128`，不符合保存入口要求的 `2.5.1+cu121`。原始 [RESULT](RESULT.json)、[stdout](stdout.log)、[stderr](stderr.log) 表明 **0 个 ODE 小步**，没有任何可达区间或性质结果。该目录保留失败收据，不重用。修正解释器后的独立新尝试为 [`_002`](../tora_reach_sigmoid_official2026_mat_u11_full500_huan_002/SUMMARY.md)。全过程未做哈希校验。
