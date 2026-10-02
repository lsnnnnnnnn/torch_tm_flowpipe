# TORA reach-sigmoid：P3 官方合同首控制期门检

远端原始 run_dir：`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_p3_firstperiod_001`。本地保存了同名目录的原始配置、模型、源码、`START.json`、`PREFLIGHT.json`、`RESULT.json`、日志、`metrics.json`、`observations.jsonl` 和 `ranges.bin`；按文件名及字节数与远端只读清单核对。

用户选择的主合同是 2026 官方四层 sigmoid `.mat` 控制器，ONNX 图内 `u=11f`，外部缩放 `1/0`，完整四态初盒和论文 TORA ODE。此 run ID 使用 working P3 / validation P4 的 plant engine、严格 endpoint/injection 与共享作者 CROWN 驱动，GPU3、CPU18–19，只执行第一个 0.5 s 控制期。见[启动收据](START.json)、[预检](PREFLIGHT.json)、[配置](config.yaml)和[运行结果](RESULT.json)。

运行接受 50/50 个 0.01 s 小步，外层 wall 为 **3.474431793 s**。[独立保存区间扫描](INDEPENDENT_INTERVAL_SCAN.json)重读了[全部 50 条原始范围](ranges.bin)和[观察记录](observations.jsonl)：四态 tube/endpoint 均有限、有序，同步 step/lane 连续，endpoint 均落在相应 tube 内，首段 tube 覆盖完整初盒。末端 `t=0.5` 的保存 endpoint 为 `x1∈[-0.8850332870274797,-0.8573636335322319]`、`x2∈[-0.006694021145015465,0.021910970492376655]`；[运行指标](metrics.json)的末端 hull 略窄，二者不是同一输出对象。

这是一期数值门检，不是 5 s 完整时域，也未运行性质 checker；当前末端不在目标 `x1∈[-0.1,0.2], x2∈[-0.9,-0.6]`。保存区间不足以构成端到端浮点 NNCS 证明。这一次 wall 不能用于四方完整时域速度排名。没有重新启动旧实验，也没有做内容摘要校验。
