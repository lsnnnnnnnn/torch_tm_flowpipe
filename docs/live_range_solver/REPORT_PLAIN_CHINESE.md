# 在线范围批量求解：真实连续前缀与完整流程实测

科学源码：`ef4e2f0c17518a4c4989071a596cd8631235c5ea`。机器状态：`LIVE_RANGE_SCHEDULER_VALIDATED__END_TO_END_PREFIX_SPEEDUP`。
本轮选择 A：两系统 B32 都达到预注册的在线完整前缀提速门槛。保留默认关闭接口，下一步可以研究更完整的边界/状态常驻；本轮没有迁移第二个数学算子。

## 1. 这次从“存好的请求”变成了什么在线流程？

每个任务从固定初始子盒建立自己的完整状态，执行原有连续 step。运行到已接入的范围调用时，它提交当时真正生成的系数与 domain，等待自己的 Future；收到通过身份检查的结果后才生成后续请求。服务只合并已经就绪、完整数学 key 相同的请求。整个过程中没有加载先前保存的答案。

接口默认关闭。worker 显式建立 prepared replay / packed boundary / 请求适配上下文；一个服务线程拥有 CUDA stream 和完成 event。GPU 只执行已有四个范围 kernel；方程、阶数、初始原盒、VDP h=0.01、Brusselator h=0.02、cutoff、E、收紧与历史传播未改。dense 范围及其余数学仍在 CPU。覆盖边界与空支持/外供表回退见 [调度合同](SCHEDULING_CONTRACT.md) 和 [coverage/coverage_summary.json](../../artifacts/runs/live_range_solver_20260909T053007Z/coverage/coverage_summary.json)。

L 是原有最快 packed 逐任务路线，保留已知旧幂缺陷；S 是校正后的 CPU 串行；Q 是同算术的在线 CPU；G 是同服务的 CUDA。S_gpu 只用于区分 CUDA 算术变化和调度错误。

## 2. 实际每批有几个任务，是否出现长时间等齐？

主输入仍是原盒子的固定 8×4 分区，B8 使用任务 0/4/8/12/16/20/24/28，B1 使用任务0。下面是三次正式 G 运行合并后的真实派发分布；B32 是活跃任务数，不等于每个范围组32条。

| 系统 | 任务数 | 平均组大小 | 最大组 | 单例请求占比 | 等待P50 ms | 等待P95 ms | 最长等待 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Brusselator | 1 | 1.00 | 1 | 100.0% | 0.020 | 0.032 | 0.061 |
| Brusselator | 8 | 1.80 | 8 | 31.8% | 5.336 | 22.654 | 125.507 |
| Brusselator | 32 | 3.64 | 31 | 9.2% | 27.167 | 121.010 | 1145.173 |
| VDP | 1 | 1.00 | 1 | 100.0% | 0.020 | 0.032 | 0.307 |
| VDP | 8 | 1.62 | 8 | 38.5% | 6.814 | 24.500 | 116.610 |
| VDP | 32 | 2.72 | 25 | 15.6% | 35.310 | 138.378 | 580.557 |

预注册的2/20 ms短前缀候选按总耗时选择20 ms并冻结。达到组上限、全部存活任务等待/结束或等待到期就派发。20 ms是可派发的期限，已有计算和单核调度仍会延迟实际服务；上表报告真实等待。小组、早完成、失败和取消任务均能退出，不要求所有任务的第k次调用对齐。完整直方图、flush原因和全部等待样本见 [actual_grouping.csv](../../artifacts/runs/live_range_solver_20260909T053007Z/actual_grouping.csv)、[flush_reasons.csv](../../artifacts/runs/live_range_solver_20260909T053007Z/flush_reasons.csv) 与各原始 run。

## 3. 怎么证明返回给正确任务、失败不污染别人？

请求带 run/task/epoch/接受边界generation/attempt/counter。提交、派发、回传、消费、commit均检查身份；后一请求必须引用已消费的前一结果。输入和结果私有复制，取消与整状态commit共用锁。checkpoint先取消未提交attempt，保存最后接受状态；恢复建立新epoch，旧worker的迟到结果无效。整包验证还从固定子盒或checkpoint重建每个诊断任务的首个状态，核对后续接受边界逐步递增与固定h，不能只靠记录内部相互一致的hash证明输入来源。

110个诊断案例覆盖两系统B1/B2两步、B8/B32二十步、两个不同任务各120步、原未分区B1二十步和历史窗口。CPU S/Q按完整segment及下一状态比较；G/S_gpu在B1/B2/B8全段、B32首两步比较；另比较B8拆成2×B4、4×B2、8×B1和逆序/延迟/合法flush改变。详细比较与历史队列99/0/1、999/0/1清空记录见 [same_backend_state_equivalence.json](../../artifacts/runs/live_range_solver_20260909T053007Z/same_backend_state_equivalence.json)、[history_reset_checks.json](../../artifacts/runs/live_range_solver_20260909T053007Z/history_reset_checks.json)。

四个真实故障专项（两系统×CPU/CUDA）各有10个接受步骤、一次拒绝重试、一次异步取消和一次旧返回丢弃；失败后恢复的完整状态及健康任务后续步骤与独立执行一致。发现的checkpoint私有诊断字段/字典顺序丢失已用安全JSON sidecar修复，原失败记录保留。另有不同h的测试任务：真实收紧次数2/5、请求数137/215，CPU/CUDA都与各自独立运行一致；这组输入只用于状态机验收，未进入性能表。

当前根测试按身份去重为1221 passed、2 optional skipped；完整遍历在`51cd57915e70`，最终科学源码28项受影响测试重验通过，已经包含在1221个身份内。父源码局部131项标为REUSED。原始值验证覆盖九类重新计算外层hash后的篡改，包括错任务返回、旧generation、取消后commit、范围端点、fallback、等待/传输遗漏、lane-step数和离线冒充在线。独立副本验收及最终推送/SHA核对尚待执行；本草稿不将其标为完成。

## 4. CPU/GPU的范围变化是什么，哪些不等于错误？

CPU/GPU各自用独立Fraction检查幂、每项和总和，并用原算子重新计算保存的实际请求。完整检查范围为B32首两步、所有两步小型/拆批用例、B8和原B1的首两步、每个三步历史窗口的前两步，以及120步预注册的1/2/60/100/119/120；故障边界前后成功返回和新结构/回退/纠正另完整核验。其余调用依靠未改变的局部算术合同，不宣称每条都重新做过Fraction审计。各路线计数见 [ARITHMETIC_AUDIT_SCOPE.json](../../artifacts/runs/live_range_solver_20260909T053007Z/ARITHMETIC_AUDIT_SCOPE.json)。

同一个CPU observer读取各自真实segment的endpoint/tube模型，产生以下宽度统计。宽度差异本身不能证明不安全，GPU更窄也不是安全证明；没有取CPU/GPU区间交集或以抽样替代包含保证。

| 系统 | 视图 | x/y条目数 | 宽度比P50 | 宽度比P95 | 宽度比max | 上下界最大绝对差 | 中心最大差 | 绝对差最差位置 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VDP | endpoint | 2138 | 1 | 1 | 1 | 4.44089e-16 | 4.44089e-16 | prefix-b8/12/step 10 |
| VDP | tube | 2138 | 1 | 1 | 1 | 4.44089e-16 | 4.44089e-16 | prefix-b8/28/step 8 |
| Brusselator | endpoint | 2138 | 1 | 1 | 1 | 8.88178e-16 | 4.44089e-16 | history/0/step 1001 |
| Brusselator | tube | 2138 | 1 | 1 | 1 | 4.44089e-16 | 4.44089e-16 | prefix-b8/8/step 14 |

每个坐标、时刻的上下界及精确累计时间见 [cross_backend_widths.csv](../../artifacts/runs/live_range_solver_20260909T053007Z/cross_backend_widths.csv)；按系统/案例/endpoint或tube/x或y细分的P50/P95/max及各项最差位置见 [width_summary_by_coordinate.csv](../../artifacts/runs/live_range_solver_20260909T053007Z/width_summary_by_coordinate.csv)。CPU宽度≤1e-10的条目只列绝对差，见 [near_zero_widths.csv](../../artifacts/runs/live_range_solver_20260909T053007Z/near_zero_widths.csv)。GPU/CPU>1.10共有0条，全部列于 [width_warnings.csv](../../artifacts/runs/live_range_solver_20260909T053007Z/width_warnings.csv)，定位记录见 [WIDTH_WARNING_INVESTIGATION.json](../../artifacts/runs/live_range_solver_20260909T053007Z/WIDTH_WARNING_INVESTIGATION.json)。接受/步长/验证与收紧停止计数/请求数比较中共有0步行为差异，首个实际分歧及全部条目见 [BEHAVIOR_SUMMARY.json](../../artifacts/runs/live_range_solver_20260909T053007Z/BEHAVIOR_SUMMARY.json)、[behavior_comparison.csv](../../artifacts/runs/live_range_solver_20260909T053007Z/behavior_comparison.csv)。

VDP连续120步覆盖SR100；历史99/100/101和Brusselator999/1000/1001来自安全完整checkpoint，明确标为RESUMED_LOCAL_WINDOW。没有新GPU从初始集跑满1000步，也没有证明整个ODE求解器形式化正确。

## 5. 整个前缀实际快多少，分组CPU是否本来就更慢？

以下墙钟包含建任务初始状态、worker/队列启动、全部非范围数学、必要检查、真实请求准备/等待/分组/传输/同步/回传和清理。S/Q/G每组三次顺序为S-Q-G、G-Q-S、Q-S-G；L每系统/批次一个有限同工作量样本。速度倍率按配对样本取中位数，>1表示G更快。

| 系统 | B | 每任务步数 | L秒(1次) | S秒中位 | Q秒中位 | G秒中位 | S/G | Q/G | min(S,Q)/G | L/G |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VDP | 1 | 2 | 0.878 | 1.035 | 1.070 | 0.954 | 1.091× | 1.122× | 1.091× | 0.921× |
| VDP | 8 | 20 | 78.326 | 96.077 | 93.584 | 75.568 | 1.271× | 1.223× | 1.223× | 1.036× |
| VDP | 32 | 20 | 416.845 | 389.619 | 338.178 | 304.436 | 1.327× | 1.106× | 1.106× | 1.369× |
| Brusselator | 1 | 2 | 1.293 | 1.561 | 1.603 | 1.453 | 1.076× | 1.103× | 1.076× | 0.890× |
| Brusselator | 8 | 20 | 114.899 | 141.555 | 135.249 | 111.774 | 1.257× | 1.213× | 1.213× | 1.028× |
| Brusselator | 32 | 20 | 532.549 | 579.601 | 499.856 | 430.507 | 1.313× | 1.113× | 1.113× | 1.237× |

两系统B32的三组配对倍率逐项如下，均为min(S,Q)/G；小于1表示该组G较慢。接纳要求每系统至少2/3组大于1且中位数≥1.10，完整结果包含各组波动。

| 系统B32 | S-Q-G组 | G-Q-S组 | Q-S-G组 | G胜出次数 | 配对中位数 | 实用提速门槛 |
| --- | --- | --- | --- | --- | --- | --- |
| VDP | 1.106× | 1.157× | 1.034× | 3/3 | 1.106× | 达到 |
| Brusselator | 1.168× | 1.113× | 0.975× | 2/3 | 1.113× | 达到 |

全部60次实测、min/max、吞吐和资源见 [timings_raw.csv](../../artifacts/runs/live_range_solver_20260909T053007Z/timings_raw.csv)、[end_to_end_summary.csv](../../artifacts/runs/live_range_solver_20260909T053007Z/end_to_end_summary.csv)、[resource_usage.csv](../../artifacts/runs/live_range_solver_20260909T053007Z/resource_usage.csv)。正式成功lane-steps按路线合计为 {"L": 1604, "S": 4812, "Q": 4812, "G": 4812}；失败/取消任务没有进入成功分子。B1只要求揭示开销，不要求GPU获胜。原未分区B1单列在 [original_b1_diagnostic.csv](../../artifacts/runs/live_range_solver_20260909T053007Z/original_b1_diagnostic.csv)，它带诊断成本，不能和子盒正式计时混用。

所有主表路线使用同一CPU2、Torch intra/inter-op各1、同一GPU0 Tesla V100及原py11环境。GPU冷启动编译/模块加载/原语自检另记；正式服务复用已经自检的模块，必要计算与检查仍计时。正式窗口不做独立Fraction审计、大输入hash或profiler；CPU保证幂包含所需的精确校正仍在正式分母内。冷启动、证据序列化、进程RSS与Torch显存口径见 [startup_and_serialization.csv](../../artifacts/runs/live_range_solver_20260909T053007Z/startup_and_serialization.csv)。显存数是Torch分配峰值，未包含驱动上下文；RSS是进程生命周期高水位。

等待秒数不能跨任务相加当作总耗时。[time_partition.csv](../../artifacts/runs/live_range_solver_20260909T053007Z/time_partition.csv)以真实时间区间的并集给出互斥墙钟分类，保留服务与worker步骤的重叠；worker步骤跨度含等待，服务kernel-and-sync是主机观察值，都不冒充纯CPU执行或纯设备kernel时间。

| 系统B32 | 实测S请求evaluator份额 | 消除该份额的上限参考 | 替换为实测G服务时间的参考 |
| --- | --- | --- | --- |
| VDP | 34.52% | 1.527× | 0.838× |
| Brusselator | 34.33% | 1.523× | 0.987× |

这里的份额只覆盖已接入请求的evaluator，不包含调用前packing和仍在CPU的dense范围；G服务还与worker跨度重叠。因此Amdahl值仅是有明确范围的参考，正式结论始终来自完整wall time，没有沿用旧两步17%的份额。

## 6. B1、B8、B32分别适不适合，是否真正胜过此前最快路径？

| 系统 | 批次 | min(S,Q)/G | L/G | 本轮判断 | 旧最快L有限比较 |
| --- | --- | --- | --- | --- | --- |
| VDP | B1 | 1.091× | 0.921× | 小幅/边界收益 | 尚未超过本轮 L 有限样本 |
| VDP | B8 | 1.223× | 1.036× | 在线 G 有实测优势 | 超过本轮 L 有限样本 |
| VDP | B32 | 1.106× | 1.369× | 在线 G 有实测优势 | 超过本轮 L 有限样本 |
| Brusselator | B1 | 1.076× | 0.890× | 小幅/边界收益 | 尚未超过本轮 L 有限样本 |
| Brusselator | B8 | 1.213× | 1.028× | 在线 G 有实测优势 | 超过本轮 L 有限样本 |
| Brusselator | B32 | 1.113× | 1.237× | 在线 G 有实测优势 | 超过本轮 L 有限样本 |

正式B32接纳规则是两系统各至少2/3配对>1且中位数≥1.10。逐配对数据与机器决策见 [paired_speedups.csv](../../artifacts/runs/live_range_solver_20260909T053007Z/paired_speedups.csv)、[PERFORMANCE_RESULT.json](../../artifacts/runs/live_range_solver_20260909T053007Z/PERFORMANCE_RESULT.json)；按耗时直接计算的G/S、G/Q、G/L及其吞吐反比同时列于 [relative_wall_time.csv](../../artifacts/runs/live_range_solver_20260909T053007Z/relative_wall_time.csv)。只胜Q但不胜S意味着调度成本吞噬收益；只胜严格CPU但未胜L也不能说超过此前最快实现。L有已知三次幂局限，本轮有限计时不构成其全部输入安全证据。

## 7. 下一步走哪条路线？

本轮选择 A：两系统 B32 都达到预注册的在线完整前缀提速门槛。保留默认关闭接口，下一步可以研究更完整的边界/状态常驻；本轮没有迁移第二个数学算子。

交付的是能真正连续推进的默认关闭在线接口、固定科学SHA的有限诊断和完整前缀实测。可重算原始证据入口为 [raw_minimal/INDEX.json](../../artifacts/runs/live_range_solver_20260909T053007Z/raw_minimal/INDEX.json)，文件摘要为 [SHA256SUMS](../../artifacts/runs/live_range_solver_20260909T053007Z/SHA256SUMS)；执行方法见 [实验README](../../experiments/live_range_solver/README.md)，逐条目标验收见 [GOAL_AUDIT](GOAL_AUDIT.md)。后续报告/包装提交只更新证据及说明，不改变这些数值源码和正式分母。
