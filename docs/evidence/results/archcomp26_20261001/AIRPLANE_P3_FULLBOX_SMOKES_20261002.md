# Airplane continuous P3：官方全初盒的两个隔离 smoke

两次都是新尝试，物理初集为官方单个完整盒 `x,y,z,r,p,q=0`、`u,v,w,phi,theta,psi∈[0,1]`，固定 2026 ONNX `[N,12]→[N,6]`，六控制按 `(Fx,Fy,Fz,Mx,My,Mz)` 注入。连续合同是 20×0.1 s 至 `T=2 s`，全时 `y,phi,theta,psi∈[-1,1]`；本轮各只尝试首个 0.1 s 周期的 10×0.01 s 三阶子步。未重跑旧单点实验，也未进行内容摘要或哈希校验。合同及 P3/native 前置缺口见[入口审计](../../../ARCHCOMP26_AIRPLANE_P3_NATIVE_ENTRY_AUDIT_20261002.md)。

[既有 CPU 合同预检](airplane_continuous_order3_fullbox_20261002/PREFLIGHT.json)与 P3 新增[六输出 CPU 注入预检](airplane_p3_solution_order_prep_20261002/CPU_PREFLIGHT.json)均通过，后者未初始化 GPU。每个运行的原始远端目录为 `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/<run_id>`；小原始证据及当时入口源已复制到同名本地目录和仓库镜像。

| 新 run ID / 验证设置 | 外层 wall | 数值结果 | 首个准确阻断与证据 |
|---|---:|---|---|
| [`airplane_p3_order3_smoke1_001`](airplane_p3_order3_smoke1_001/RESULT.json)：工作 P3、`solution_plus_one` 验证 P4 | 13.757866 s；failed / exit 1 | 0 个已接受段 | 第 1 个数值步之前构建验证 P4 表，`_radix_encode` 报 `9^20 >= 2^63`；[payload 结果](airplane_p3_order3_smoke1_001/payload/RESULT.json)、[stderr](airplane_p3_order3_smoke1_001/stderr.log)、[当时入口源](airplane_p3_order3_smoke1_001/source/archcomp26_airplane_continuous_p3_nohash.py) |
| [`airplane_p3_solution_order_smoke1_001`](airplane_p3_solution_order_smoke1_001/RESULT.json)：工作/严格验证均 P3 `solution_order` | 12.500084 s；failed / exit 1 | 首段 `accepted=false`；0/10 已接受 | [逐段观测](airplane_p3_solution_order_smoke1_001/payload/observations.jsonl)、[payload 结果](airplane_p3_solution_order_smoke1_001/payload/RESULT.json)、[方法记录](airplane_p3_solution_order_smoke1_001/payload/P3_METHOD.json) |

第二种是引擎已支持的严格同阶验证设置，并以新 run ID 重试；它成功越过先前的整数编码阻断，但在 `t∈[0,0.01]` 的唯一全初盒数值接受检查处停止。当前观察器只记录 `rejected_lane`，未给出内部验证失败细因，因此不能归因于物理性质、控制器、分母或舍入。两个 `ranges.bin` 都为空，均无可用全时 tube；没有安全或不安全性质结论，也不能将任一尝试记为完成 `T=2`。两次仅是不同配置的诊断，不能做速度/精度排名。

首拒后没有继续运行 full20。后续若要推进，需在独立新 run 中记录首步验证内部失败分类，或另立明确且可接受完整初盒的数值方案；本轮的原始失败记录应保留。神经网络包络和浮点链也尚无独立端到端证书。
