# 三工具 Taylor 模型深度研究：最终报告

## 交付状态

本报告对应通过验收的 `20260730T015245Z` 运行，权威分支为
`codex/torch-flowstar-diffreach-deep-study`。默认分支可能尚未包含这些结果。
最终验收为 **True**，十次重复门槛为
**True**，质量审计为
**True**。质量审计解析了
36 个 CSV、共
195551 行；生成
18 张强制图。数值运行冻结的完整测试为
354 passed / 5 skipped / 0 failed；报告过滤修正
`b0d20d4` 后，直接绑定 curated artifact 的最终代码测试为
350 passed / 10 skipped / 0 failed（sandbox 不暴露主机 CUDA/外部接口）。
该修正不改变数值 CSV；精确分组和日志摘要见
`FINAL_DELIVERY_TEST_RECORD.md`。

## 核心结论与撤回

早期“同阶冠军”结论正式撤回。Torch 的 complete total-degree 基、
DiffReach 的 affine/restricted quasi-quadratic 基以及 Flow* 最低合法
二阶基并不相同；三者的验证器、余项语义、重置方式和数值后端也不同。
因此不能把三个工具标签相同或相近的 `order` 当成同一数学对象，也不能
给出跨工具的统一赢家。

有效比较被拆成五类：共同初始盒和步长下的原始单步结果、共同仿射 carry、
共同逐分量盒 carry、准确标注真实基的原生低阶配置，以及同一工具、同一
系统、同一绝对时刻内部的宽度/运行时间 Pareto 前沿。Torch 的
legacy-tightened 端点只保留为该工具内部诊断，绝不与其他工具的 raw
端点比较。没有达到请求终点的配置只报告成功时域，不能参与终点宽度比较。

## Flow* 正确性修复

Riccati 反例的根因不是精度不足，而是缓存路径不一致：完整表达式求值在
变量叶节点执行截断并产生区间贡献，旧的“只重放余项”缓存没有记录该项。
stock 结果因此漏掉解析真值。记录/重放补丁以及独立的 full-Picard
重新验证都恢复了解析包含；主修复行
64 条，修复后解析违反
0 条，stock 反例保留
4 条。原始 Van der Pol
调度到 T=10 使用 290 段，根因补丁
到 T=10 使用 303 段；补丁改变
自适应接受决策，所以不要求两者调度相同。固定二阶、`h=0.05` 的 Riccati
压力点可能拒绝其候选余项；这属于“配置被拒绝”，不是 Flow* 整体失败或
崩溃。实验也从不在 `advance` 后覆盖 Flow* 余项。

此前 Van der Pol 自适应路径的折叠端点漏包也已闭环。审计同时比较
upstream stock、原始/同构生成 harness、变量叶截断补丁和自适应
full-Picard fallback，定位到折叠端点限制路径。对同一已验证 flowpipe
直接在 `tau=[h,h]` 上做原生区间求值时，所有确定性样本均被包含；raw
端点现将两条路径的 hull 差值显式加入独立余项。修复通过：
**True**；是否从权威结果排除：
**False**。

## 共同协议与基

共同 CIR v2 明确记录状态顺序、时间变量、归一化域、中心、状态缩放、
稀疏指数、系数区间、独立/结构化余项、tube/endpoint、请求与接受步长、
请求与成功时域、验证状态、运行时间拆分和来源 SHA。缺失能力写成
`unavailable`，不会伪装成零。

| tool | basis | status | reason |
| --- | --- | --- | --- |
| torch_common_engine | B1 | supported_experiment_adapter |  |
| torch_common_engine | B_DR | supported_experiment_adapter |  |
| torch_common_engine | B2 | supported_experiment_adapter |  |
| torch_common_engine | B3 | supported_experiment_adapter |  |
| diffreach | B1 | supported_native |  |
| diffreach | B_DR | supported_native |  |
| diffreach | B2 | capability_gap | no complete total-degree-2 native dictionary |
| diffreach | B3 | capability_gap | no quadratic state-cross dictionary with tau lift |
| flowstar | B1 | capability_gap | minimum legal fixed order is 2; no exact B1 selector |
| flowstar | B_DR | capability_gap | no exact restricted c/L/Lt dictionary selector |
| flowstar | B2 | supported_native |  |
| flowstar | B3 | capability_gap | order 3 is a strict cubic superset, not exact B3 |

B1 是 complete affine；B_DR 是受限的局部时间/状态结构；B2 是 complete
quadratic；B3 只增加“一次局部时间乘二次状态依赖”，包括
`tau*xi_i*xi_j`，但排除一般三次状态项和 `tau^3`。匹配基实验在同一个
算术引擎、同一验证器和同一重置下比较这些字典，用于隔离“基”的作用，
而不是伪造三个原生工具都支持同一个字典。

## 正确性、运行时间与有效性

本次运行完成 17017 个解析检查、
4095 个 CIR 点包含检查、
2010 个
原生/CIR 往返检查，以及
63813 个原生验证检查。
确定性轨迹采样检查了
24054 个候选，
它只用于发现错误，不是证明。任何采样失败的候选都从主要数值 Pareto
集合中排除，但仍保留成功时域和失败分类证据。

运行时间把编译/JIT、稳态完整配置执行、重复次数和内存分开。JAX 融合与
JIT、C++ 编译/进程启动属于后端因素；多项式项数、Picard/细化轮数、
区间求值、符号窗口和重置属于算法工作量。不存在跨工具 Pareto 排名，
也不把不同绝对时刻的宽度/时间点互相支配。

## BERN/IBF 可行性

BERN 只作为多项式范围查询候选，不是第四个可达性工具。clean-room CPU
float64 原型对 5 个解析案例都包含精确范围，并在
2 个消去型案例上更紧。当前证据是
“精确代数 Bernstein 包络 + 保守浮点余量候选”，不是完整舍入证明。
在继续之前需要稀疏维数限制和形式化向外舍入后端。它本身不提供局部时间
积分、截断余项、Picard 验证、端点代入或多步重置；本研究也没有神经网络
控制器，因此不能据此推出 CROWN/β-CROWN 或 NN 抽象的直接收益。

## 局限与后续

有效结论限于上述共同协议、准确标注的原生配置以及工具内部消融。无效结论
包括“同阶全局赢家”、不同绝对时刻的宽度排名、以及把 MPFR 证明强度与
float64 候选等同。建议 Torch 后续支持规范化 affine/QR 重置、受限
time-state 基、改进多项式范围界和 overflow 归因、暴露验证器时间，并在
任何证明级声明之前加入严格 directed-rounding/MPFR 后端。

完整英文数值表、十一项逐题回答和全部有效性限制见 `FINAL_REPORT.md`；
权威表格和图片见 `artifacts/authoritative/20260730T015245Z`。
