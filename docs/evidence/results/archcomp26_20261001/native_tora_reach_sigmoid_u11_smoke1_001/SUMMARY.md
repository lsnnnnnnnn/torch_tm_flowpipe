# TORA reach-sigmoid：原生官方 u=11f 一期门检

远端原始 run_dir：`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_tora_reach_sigmoid_u11_smoke1_001`。旧 2024 `22(f−0.5)` 入口没有重启；本次独立[构建源码和预检](../native_tora_reach_sigmoid_u11_build_001/PREFLIGHT.json)使用官方四层 sigmoid `.mat` 导出的图内 `11f` ONNX，RPC 外部缩放 `1/0`，论文四态 ODE 和完整单初盒，P6、0.01 s 小步、控制周期 0.5 s。新 RPC 端口 5111，物理 GPU1、CPU44–47。

[外层 RESULT](RESULT.json)为 exit 0，wall **4.227119009 s**；[原生日志](native.log)完成 `Step 0`，在短时域末端输出 `UNKNOWN`，这不是 `T=5` 的性质结论。[唯一一次控制器 RPC](controller_rpc.jsonl)输入是完整初盒，其仿射中心包络 `[5.078120866697472,5.078162989956009]` 含官方模型中心 `u=5.078156863296165`。[编译入口实际初盒](initial_boxes.json)也已保存。

[独立范围扫描](RANGE_SCAN.json)重读[原始二进制范围](ranges.bin)：50/50 个四态 tube/endpoint 记录，lane/步号完整，无非有限值、反向区间或 endpoint 超出同段 tube；`t=0.5` 保存 endpoint 的 `x1=[-0.885015359716206,-0.8573815513819983]`、`x2=[-0.006688532969114709,0.02190562985484164]`。一期通过后另立[全程 run ID](../native_tora_reach_sigmoid_u11_full10_001/SUMMARY.md)。这些数值收据不构成独立端到端浮点 NNCS 证明；未做内容摘要校验。
