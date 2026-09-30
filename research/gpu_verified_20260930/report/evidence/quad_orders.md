# QUAD 的阶数：可以指定，当前证据到哪里

2026-09-27。以下仅依据本地冻结源码、配置和实际结果；路径相对本工作区。

**我们的实现不是只能用二阶。** Huan、Xiangru 的入口均支持 `--order`，并用它创建工作多项式的阶数及相应表。当前固定二阶来自对齐原始 QUAD 实验的配置。三阶已经实际运行；四阶及更高阶需要先做资格和资源检查，不能由“参数可填”推定可正确、有效地完成整个 benchmark。

## 实际比较使用什么阶数

| 对象 | 已核实设置 | 证据 |
|---|---|---|
| ARCH-COMP 原生 Flow* QUAD | 固定步长 0.005、固定二阶、余项预算每维 ±0.1 | `results/archcomp_review_20260923/sources/native/submit/CROWN-Reach/archcomp/Quadrotor/quad.cpp:68–74`；Xiangru 归档同位置一致。 |
| 当前对齐的 QUAD 配置 | 控制周期 0.1、ODE 步长 0.005、`ode_order: 2` | `results/quad_split_probe_20260925/quad_author_resolved.yaml:6–12`。 |
| 已归档 Huan / Xiangru 比较运行 | 两者实际记录 `order: 2` | `results/archcomp_review_20260923/evidence_v1/quad_huan_full_cache/metrics.json:12`、同级 `quad_xiangru_full_cache/metrics.json:12`。这里只核实阶数，不把这些旧运行升级为新的严格性结论。 |
| 仓库另一份通用 QUAD YAML | `ode_order: 4`，同时 `step_size: 0.005`、`ode_step_size: 0.01` | `results/archcomp_review_20260923/sources/xiangru/src/configs/quad.yaml:6–8`；native 副本同值。这不是当前对齐实验的配置，不能只取“四阶”作同条件比较。 |

Flow* 本身既能指定固定阶，也有阶数范围的重载：`results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/Continuous.cpp:98–149`。本次原生 QUAD 调用的是二参数固定阶接口，不是自适应阶数范围。Huan 的 `--order`、配置覆盖、表创建见 `results/archcomp_review_20260923/sources/huan/integrations/crown_reach/gpu_driver.py:730,842–843,868–874,927–931`；Xiangru 对应 `results/archcomp_review_20260923/sources/xiangru/src/flowstar_gpu/integrations/crown_reach.py:787,932–933,958–964,1019–1023`。这些入口在初始化时选定阶数。

## “三阶”到底增加了什么

这里的工作阶数限制的是**局部时间变量 τ 与归一化空间变量的总次数**，不是单独只看时间幂。例如 `τ z_i z_j` 的总次数是 3，二阶工作多项式不能保留它，三阶可以。这正是提高阶数可能减少截断余项的原因；也不保证最后的整体区间一定更窄。

基定义见 `results/flowstar_acceleration_20260921/qualified_private_a99d614/source/engine/src/flowstar_gpu/monomials.py:6–20`；Flow* 按总次数移出超过阶数的项，见 `results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/Polynomial.h:1530–1566`，总次数定义见同目录 `Term.h:22–23`。

还须区分三个数字：

| 当前生产推进设置 | 工作解多项式 | point code | 验证计算 |
|---|---:|---:|---:|
| P2 | 2 | 1 | 3 |
| P3 | 3 | 2 | 4 |

point code 是 RHS 编译使用的阶数；验证阶数决定验证余项时使用的计算表和展开。二者都不能直接称为“解的阶数”。只提高验证阶数不会给原二阶候选补上 `τ z_i z_j`。当前 `solution_plus_one` 路径及独立验证表见 `results/quad_recovery_plan_review_20260925/source_reconstruction/sparse_exec.py:2021–2037`；实际 P3 三个阶数记录见 `results/quad_targeted_recovery_20260927/evidence_promote/promote_long/RESULT.json:24–26`。早先只补六类三次项的局部 P3 探针是另一种候选实验，不能混称这个完整 P3 推进。

QUAD 有 16 个引擎状态变量：12 个物理量、1 个物理时钟、3 个保持控制量（原生 `quad.cpp:31–49`）。再加独立局部积分时间 τ，共 17 个多项式变量。完整总次数不超过 k 的理论基大小为 `C(17+k,k)`：

| k | 理论完整基项数 / 每个状态多项式 |
|---:|---:|
| 2 | 171 |
| 3 | 1,140 |
| 4 | 5,985 |
| 5 | 26,334 |

计数公式及 n 的含义见上述 `monomials.py:61–64,115–117`。**这不是实测非零项数、稀疏活跃支持大小或显存用量。** 乘法扩展表和验证表还会增加成本；不能直接按此表推算运行时间或 GPU 显存。

## 已有进展和下一步

**后续实际结果更新（同日）：** 下述从t0的P2/P3对照及40步/冷回放已完成，分别共同接受681/689步。共同681步x5宽度4.70043062→3.25079434、x6宽度3.28235196→2.23246671；共同advance时间134.287947→300.547394秒。完整条件、证据范围和全部结果见本目录`REPORT.md`及`evidence_order_long/compare_order_long/ORDER_COMPARISON.md`。这些是root1初始x5完整二分的两叶诊断，不是全部1024个root完成。

四阶仍配套五阶验证。原全局索引表及步长因子静态计数19.508438GiB，不能直接在当前设备按旧布局分配；新按活跃支持生成索引的实现已通过CPU与CUDA逐字节差分，真实P3首步已通过，正在做40步及原P3对照后才进入P4。没有把验证策略降阶以换取可运行。下文保留此前晚升阶诊断与其局限，计划语句对应当时阶段。

从真实已提交的第 681 步状态出发，保留全部多项式、普通余项和 SR 历史后提升到 P3，两叶共同通过 **682–688，共新增 7 步**，第 689 步两叶均拒绝。没有完成 700 或 1000 步，也没有推进原始全部 1024 个根盒。实际结果见 `results/quad_targeted_recovery_20260927/evidence_promote/promote_long/RESULT.json:2–20,27–33`。这是升阶确有帮助的直接证据，还不是完整 QUAD 成功或速度优势。

这段延伸保持第 680 步的旧控制，没有跨新控制更新边界。已确认旧历史控制边界使用普通 RN 端点计算，端点舍入未计入余项；后来升阶不能修复此前遗漏。来源与收费位置见 `results/quad_targeted_recovery_20260927/ENDPOINT_ROUNDOFF_FINDING.md:17–34`。因此旧 681→688 结果应作为历史状态上的诊断，而不是完整 NNCS 严格证明。

下一步应从 **t0** 用相同修复后的端点交接、相同模型/初值覆盖/控制证书策略/步长/预算，分别运行 P2 与 P3；先过 40 步和跨控制边界的冷回放门，再继续长程，比较同一步的时间与区间宽度。这是新实验路线，不把尚未完成的运行当作结果。端点修复之外，CROWN 浮点计算及一般 RN 控制注入仍需各自资格。四阶应在上述对照清楚、解析资格和资源检查通过后再做；不宜直接扩大阶数并把失败或额外成本归因于算法本身。
