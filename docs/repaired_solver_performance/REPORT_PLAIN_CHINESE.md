**修复版结果保持不变，prepared replay 有实际收益，但未达到 Brusselator 完整求解 1.5× 目标。**

主要相邻完整配对的求解倍率为 Brusselator **1.426×**、VDP **1.012×**。这是每系统一对完整复测；各窗口/前缀的三对重复及最初的完整样本同时保留，不能把一次完整配对解释为稳定精确倍率。状态为 `REPAIRED_REFERENCE_PRESERVED__USEFUL_SPEEDUP_BELOW_TARGET`；Brusselator 置信标签为 `borderline_single_pair`。

原来 generic `ordered_terms` 路径在同一步每次改变余项 R 后，都会重复准备不变的点多项式乘积、截断尾项和 polynomial-only range。本轮仅新增 opt-in 的 `prepared_remainder_replay`：首次接受后的本次 attempt 建一个局部计划，首轮按原操作顺序准备固定部分，随后重用。默认 reference 保留；VDP 已有专用缓存，继续走该缓存，不按 Brusselator 名称写死方程。

可复用性来自输入依赖：candidate、base、domain（包含实际 h）、basis、order、cutoff、range policy 及 RHS 有序图在本次 attempt 内固定。计划深复制数值和 basis 张量，核对对象/参数及输入张量版本；每轮仍实际调用 RHS 并核对操作及常数。新的 attempt/retry 重建，失败使半建记录失效，没有全局数值缓存，也不缓存某轮 R 的 proposal。详细生命周期见 [依赖表](REPLAY_DEPENDENCIES.md)。

每次仍重算当前 R 的交叉乘积、动态范围、raw/regular 两套完整误差账本、subset 判断、停止比率和原子提交。端点舍入修正、normal 左右映射、历史传播、拒绝回滚与非有限数检查保持原序；关闭 observer 也不省略动态验证账本。491 上限、0.99 停止比率、步长、阶数、预算和队列容量均沿用冻结合同。

两系统全部 fixed 运行均从原始初始条件实际执行：VDP 1000/0、Brusselator 1000/0。每系统四条完整流（初始 reference/优化、确认 reference/优化）逐位相同，覆盖每步 endpoint/tube 的 x/y 上下界、完整模型、普通余项、端点 E、重要状态/队列与 replay 决策。每种视图共 8000 个分量区间，两种视图的 16000 行宽度比全部为 1；并非只比较宽度或终点。见 [全程范围表](../../artifacts/runs/repaired_solver_performance_20260908T084224Z/full_width_equivalence.csv)。

真实时间使用 binary64 h 的精确和：VDP 为 `720575940379279375/72057594037927936`，Brusselator 为 `720575940379279375/36028797018963968`，分别是名义 T10/T20；没有缩短最后一步凑十进制终点。VDP 自适应实际得到 246 次接受、35 次拒绝，调度时钟为 10，实际步长和为 `5764607523034236731/576460752303423488`。全部 h、范围、模型、E、状态/队列和决策与修复档案逐位一致；自适应求解 185.071 秒、导出 5.108 秒，只作一致性结果，不作 fresh 自适应加速主张。

正式计时使用原 py11、CPU float64、单线程、CPU 3、production-no-observer。setup 是进程内初始化；每步 plan 准备和命中检查已经计入 solve，表中的 plan 秒数是 solve 的子项，不再相加。导出/trace/checkpoint 在 solve 外单列，whole process 还包含 Python 启动与全部 I/O。

| 主要完整复测 | setup 秒 | plan 秒（solve 内） | solve 秒 | export 秒 | whole 秒 | RSS MiB |
| --- | --- | --- | --- | --- | --- | --- |
| Brusselator reference | 0.0030 | 0.000 | 2044.604 | 94.804 | 2144.236 | 459.84 |
| Brusselator 优化 | 0.0030 | 65.030 | 1433.751 | 90.549 | 1528.972 | 460.27 |
| VDP reference | 0.0030 | 0.000 | 696.791 | 20.582 | 719.774 | 456.88 |
| VDP 优化 | 0.0029 | 0.000 | 688.645 | 20.656 | 711.717 | 456.88 |

初始化加求解的倍率为 Brusselator 1.426052×、VDP 1.011829×；峰值 RSS 比为 Brusselator 1.001×、VDP 1.000×，均未超过 1.5 倍。

初始 fresh reference 时主机 load 约 1–2；第一条完整优化开始时已升到约 40，未改动的 VDP/reference 路径也有时间波动。在首条完整优化结果产生前的 2026-09-08 10:29 UTC，已登记且只增加每系统一对相邻完整复测：Brusselator reference→优化，VDP 优化→reference。主要比较集在结果出现前确定，全部原始数据保留；不把所有差异都定量归因于负载，也没有停止他人任务。见 [预登记](../../artifacts/runs/repaired_solver_performance_20260908T084224Z/raw_minimal/timing_confirmation_preregistration.json)及 [逐运行负载/计时](../../artifacts/runs/repaired_solver_performance_20260908T084224Z/timings_raw.csv)。

| 初始完整样本（保留） | reference solve 秒 | 优化 solve 秒 | 倍率 | reference / 优化 export 秒 |
| --- | --- | --- | --- | --- |
| Brusselator | 1980.395 | 1833.675 | 1.080× | 92.740 / 118.904 |
| VDP | 706.034 | 685.036 | 1.031× | 20.758 / 20.824 |

相邻完整复测期间仍观察到主机负载波动；从首次明显升高后保存的定时观察中，1 分钟 load 范围为 0.70–23.40。相邻配对没有消除共享主机干扰，完整倍率仍是这组实际环境中的单次观察，不能据此证明稳定达到 1.5×。load 也不是 CPU 3 的精确争用时间，未用它事后修正求解秒数。见 [复测过程观察](../../artifacts/runs/repaired_solver_performance_20260908T084224Z/raw_minimal/confirmation/monitor.jsonl)。

窗口/前缀均为三对交替顺序，倍率定义为 reference solve / 优化 solve：

| 窗口或前缀 | 三次原始倍率 | median | min–max |
| --- | --- | --- | --- |
| Brusselator 1–20 | 1.2944 / 1.3484 / 1.3249 | 1.3249 | 1.2944–1.3484 |
| Brusselator 101–120 | 1.3651 / 1.3785 / 1.3849 | 1.3785 | 1.3651–1.3849 |
| Brusselator 981–1000 | 1.3363 / 1.3270 / 1.3356 | 1.3356 | 1.3270–1.3363 |
| VDP 1–20 | 1.0028 / 0.9995 / 1.0095 | 1.0028 | 0.9995–1.0095 |
| VDP 91–110 | 1.0127 / 1.0172 / 1.0092 | 1.0127 | 1.0092–1.0172 |
| Brusselator 1–100 | 1.3168 / 1.3935 / 1.3052 | 1.3168 | 1.3052–1.3935 |
| VDP 1–100 | 0.8739 / 0.9332 / 1.0035 | 0.9332 | 0.8739–1.0035 |

VDP 的 100 步前缀第一次约慢 14.4%，第二次约慢 7.2%，第三次接近持平，三次都保留。按各组 median 和主要完整配对的预设判据，VDP 通过“无稳定超过 10% 的减速”检查；这不改变第一次前缀样本确实慢于 10% 的事实。完整数据见 [原始计时](../../artifacts/runs/repaired_solver_performance_20260908T084224Z/timings_raw.csv)与 [median/min/max](../../artifacts/runs/repaired_solver_performance_20260908T084224Z/timing_summary.csv)。

同一批实际 R 的固定多项式方法调用数与互斥诊断秒数如下。调用计数包含嵌套方法，不能当作独立 RHS 次数；收紧轮数完全保留。

| 20 步窗口 | R 轮数（前后相同） | 固定方法调用：前→后 | 固定部分秒：前→后 |
| --- | --- | --- | --- |
| brusselator_0001_0020 | 148 | 5772 → 780 | 9.433 → 1.292 |
| brusselator_0101_0120 | 180 | 7020 → 780 | 11.605 → 1.341 |
| brusselator_0981_1000 | 180 | 7020 → 780 | 11.710 → 1.290 |
| van_der_pol_0001_0020 | 80 | 460 → 460 | 1.354 → 1.355 |
| van_der_pol_0091_0110 | 80 | 460 → 460 | 1.363 → 1.338 |

实现前的早期窗口预登记 selected fraction 为 0.3207，含准备估计的局部加速 3.689×，Amdahl 预测整体 1.305×、理想上限 1.472×。这已经支持有意义但可能不到 1.5× 的单机制优化。当前匹配诊断得到：

| Brusselator 窗口 | f | 局部 s（含准备/检查） | Amdahl 整体预测 | 理想上限 |
| --- | --- | --- | --- | --- |
| brusselator_0001_0020 | 0.3175 | 3.856× | 1.307× | 1.465× |
| brusselator_0101_0120 | 0.3618 | 4.226× | 1.382× | 1.567× |
| brusselator_0981_1000 | 0.3177 | 4.329× | 1.323× | 1.466× |

f 是修改前完整 post-accept 收紧阶段占比，s 是同一阶段含准备及检查的前后时间比。二者来自同一窗口中互斥 scope，父 scope 减去所有已计子 scope；不加总重叠 inclusive time。wrapper 相对独立无插桩诊断约增加 2%–4.5% 时间，cProfile 另跑，只使用 self time。它们不充当正式计时分母，也不由窗口推算完整 1000 步总时间。优化后历史处理占 Brusselator 三窗口 scoped solve 的约 51%–56%，VDP 约 41%–45%；区间对象构造及 torch.clone/any/stack/min/max 仍是热点，不能全解释成 Python 开销。

与 Flow* 的宽度差没有因为此次性能优化而改变。下面沿用已保存完整 Flow* 对象的共同 observer 结果，身份为 `REUSED_MATCHED_REFERENCE`，给出全时域的优化版/Flow* 宽度比；没有重新编译或运行 Flow*。

| 系统 / 范围 / 分量 | P50 | P95 | 最大值 | 最大值时刻 |
| --- | --- | --- | --- | --- |
| Brusselator / endpoint / x | 0.996 | 1.110 | 1.357 | 7.20 |
| Brusselator / endpoint / y | 0.991 | 1.006 | 1.038 | 13.46 |
| Brusselator / tube / x | 1.006 | 1.089 | 1.188 | 7.22 |
| Brusselator / tube / y | 1.002 | 1.138 | 1.203 | 7.64 |
| VDP / endpoint / x | 1.198 | 1.268 | 1.278 | 2.85 |
| VDP / endpoint / y | 1.165 | 1.251 | 1.266 | 3.49 |
| VDP / tube / x | 1.174 | 1.227 | 1.235 | 2.85 |
| VDP / tube / y | 1.184 | 1.231 | 1.262 | 3.49 |

历史原生 Flow* 求解约为 VDP 1.488 秒、Brusselator 13.570 秒，仍显示很大的成本量级差距。这些是历史计时，不能与本轮秒数相除宣称 fresh Flow* 倍率。本轮唯一正式速度分母是本轮最终修复 reference。

验证覆盖五个实际窗口共 100 步、668 次 proposal：每个已保存 R 由两条 evaluator 独立重算，并从相同初始余项分别运行完整 replay loop；proposal、点系数、全部 ledger、E 和决策逐位一致。边界测试覆盖不同 R、绑定参数/结构变化、原地修改隔离、零轮/固定点/不同分量收敛、subset 失败、真零/subnormal/NaN/Inf/overflow、失败回滚/retry、完整 checkpoint 恢复、开关及局部 B2 隔离。最终 verifier 还重算两套完整 reference 和自适应保存模型的端点 E、完整范围及 checkpoint 状态；固定优化模型通过全模型逐位桥接覆盖。

完整矩阵去重为 1094 passed / 11 skipped，加六种不同的重新散列语义篡改测试后为 **1100 passed / 11 skipped / 0 failed / 0 errors**。首次 root 矩阵因子进程找不到 shell 的 `python` 失败，保留原日志；仅将 PATH 指向同一既有 py11 后，在同一数值提交重跑全部 root 测试通过。旧停止 snapshot 在其原 snapshot 执行，所需历史 fixture 仅按既有精确路径/摘要复制。targeted 与独立 clone 的重复测试不再加总。起始旧端点回归和旧 verifier 均在干净父版本通过，未修改旧断言或 verifier。

源码身份分开记录：父提交 `7e41f33`；最终修复 runtime `0714e47`；初始 fresh reference `e2d00f1` 的 src 与其完全相同；完整矩阵 runtime `ae6e21a`；正式窗口/前缀/优化及确认复测 `f627d64` 的 src 与该 tested runtime 完全相同。后续 helper/report/package 提交不冒充早先运行源码。见 [SOURCE_MAP](../../artifacts/runs/repaired_solver_performance_20260908T084224Z/SOURCE_MAP.json)、[运行环境](../../artifacts/runs/repaired_solver_performance_20260908T084224Z/RUN_CONTEXT.json)、[实际合同](../../artifacts/runs/repaired_solver_performance_20260908T084224Z/EXECUTION_CONTRACT.json)与 [验证回执](../../artifacts/runs/repaired_solver_performance_20260908T084224Z/tests/final_evidence_verification.json)。

此次只权威验证 CPU binary64，没有建立整个求解器的形式化证明。局部 B2 能检查数值隔离，完整 solver 的历史队列、停止/拒绝和 scheduler 仍是 Python B1。下一轮唯一主方向是包含历史传播的真实 B 维短前缀原型，取得 B2/B8/B32 整步吞吐、内存和独立 B1 逐 lane 一致性数据；详见 [一页路线决定](ROUTE_DECISION.md)。GPU 后端仍未决定，本轮没有第三方复评或 CUDA 开发。

图表均提供 PNG/PDF：[全程宽度比](../../artifacts/runs/repaired_solver_performance_20260908T084224Z/figures/brusselator_common_width_equivalence.png)、[主要完整时间分项](../../artifacts/runs/repaired_solver_performance_20260908T084224Z/figures/full_timing_categories.png)、[初始完整时间分项](../../artifacts/runs/repaired_solver_performance_20260908T084224Z/figures/initial_full_timing_categories.png)、[固定准备次数与时间](../../artifacts/runs/repaired_solver_performance_20260908T084224Z/figures/fixed_preparation_repetition.png)、[剩余热点与 Amdahl](../../artifacts/runs/repaired_solver_performance_20260908T084224Z/figures/remaining_time_and_amdahl.png)，其余系统与视图在同一目录。

最终 package 单独提交并只推送本轮新分支，不合并 main。交付复核在最终推送 tip 的独立远端 clone 中执行相关局部测试及完整新 verifier；回执位置为本轮 ROOT 的 `DELIVERY_VERIFIED.json`，放在包外以保持被验证的提交不变。交付时核对远端、本地和 clone 三者一致，不重跑全部长实验。复核命令见 [实验说明](../../experiments/repaired_solver_performance/README.md)。
