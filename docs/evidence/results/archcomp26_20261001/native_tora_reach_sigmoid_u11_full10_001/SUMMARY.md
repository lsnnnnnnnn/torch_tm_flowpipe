# TORA reach-sigmoid：原生官方 u=11f 完整数值时域

远端原始 run_dir：`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_tora_reach_sigmoid_u11_full10_001`。复用独立[构建源码、模型和预检](../native_tora_reach_sigmoid_u11_build_001/PREFLIGHT.json)；该目录保存生成的[C++ 入口](../native_tora_reach_sigmoid_u11_build_001/archcomp/TORA/tora_sigmoid_u11.cpp)、[RPC 服务](../native_tora_reach_sigmoid_u11_build_001/archcomp/TORA/crown_paper.py)、原始冻结源码、构建命令和日志。与[一期门检](../native_tora_reach_sigmoid_u11_smoke1_001/SUMMARY.md)使用相同官方四层 sigmoid `.mat`、图内 `u=11f`、外部 `1/0`、完整初盒、四态论文 ODE 和物理 GPU1/CPU44–47/端口 5111；本次新进程独立运行 10 个 0.5 s 控制期。

[外层 RESULT](RESULT.json)为 exit 0，wall **8.991666202 s**；[原生日志](native.log)逐期输出 `Step 0` 至 `Step 9`，随后作者**终点 checker 输出 `VERIFIED`**，其 `time cost` 为 5.408 s。[控制器请求](controller_rpc.jsonl)共 10 条，首条与一期门检逐值相同，均接收新的 `11f` plant 控制值。端口在运行退出后不再监听。

[独立范围扫描](RANGE_SCAN.json)重读[全部 500 条原始范围](ranges.bin)：完整 1×500 网格、四态 tube/endpoint 均有限有序，无端点超出同段 tube。保存 `T=5` 的 endpoint 为 `x1=[0.1345319317307225,0.16059887233324702]`、`x2=[-0.8763648122763305,-0.8505857060659042]`，完整包含于官方目标 `[-0.1,0.2]×[-0.9,-0.6]`。终点包含为“5 秒内到达”的充分数值观察；作者 `VERIFIED` 仅按其终点 checker 语义记录，不自动成为独立端到端浮点 NNCS 证明。单次壁钟不能用于稳定四方速度排名；未做内容摘要校验。
