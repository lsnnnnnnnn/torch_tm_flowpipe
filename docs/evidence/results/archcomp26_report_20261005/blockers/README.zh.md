# 16×4 实例状态、未完成原因与证据边界（2026-10-05）

这份审计回答“哪些已经能做、哪些仍未完成、为什么、还缺什么”。它只整理本地原始 START/RESULT、合同和已有范围审计，没有重跑实验、旧 checker 或范围扫描，没有联网。

64 格仍为 **38 本轮数值完整 + 8 经审计历史同合同完整 + 14 无完整数值时域 + 4 Airplane 离散合同阻塞**。五个 Oct5 新 P3 主选均完整且有保存输出等值收据，未扩大证书范围；历史 289 条尝试索引保持原计数。

“无完整时域”需要进一步区别：Airplane 主全盒无有效首步；Balancing raw4 数值拒绝；DP more 性质早停；TORA remain 作者两法拒绝部分盒后仍继续记录。Docking 四方已完成数值时域但性质 UNKNOWN，不应列为没跑完。

下表中的步数均为 ODE 小步（离散为转移次数），秒数是已保存的物理时间，不是运行耗时。全初集有效前缀与最后观察步分别记录；原生二进制无逐步 accepted 字段时仅称有效保存段。标签和保存盒不是独立 NNCS 证明。完整逐格字段及原始收据见 [64格 JSON](status_64cells.json)。

## 一眼可读的 16 实例覆盖表

| 实例 | P3 | Huan | Xiangru | Flow* native | 未完成原因或当前结论 |
|---|---|---|---|---|---|
| ACC safe-distance | 50/50（Oct5） | 50/50 | 50/50 | 50/50 | 全程，保存安全距离满足 |
| Airplane continuous | 全盒0有效段 | 全盒0有效段 | 全盒0有效段 | 全盒0有效段 | 资源/编码、数值自包含或观察器失败；64子盒仅首步 |
| Airplane discrete | 合同阻塞 | 合同阻塞 | 合同阻塞 | 合同阻塞 | 缺权威NN采样与状态更新执行顺序 |
| Attitude Control avoid | 60/60 | 60/60 | 60/60 | 60/60 | 全程，修正六维危险盒后保存tube分离 |
| Balancing reach | 86/2000 | 98/2000 | 98/2000 | 83/2000 | raw4数值早拒；论文五特征模型/映射另缺 |
| Docking constraint | 400/400 | 400/400 | 400/400 | 400/400 | 全程，四方UNKNOWN |
| Double Pendulum less-robust | 100/100 | 100/100 | 100/100 | 100/100 | 全程，保存tube安全 |
| Double Pendulum more-robust | 72/80 | 72/80 | 72/80 | 64/80 | 性质早停；共同Safe仅60步 |
| NAV standard | 600/600 | 600/600（历史） | 600/600（历史） | 600/600 | 官方具名合同全程；新旧样本分开 |
| NAV robust | 600/600（Oct5） | 600/600（历史） | 600/600（历史） | 600/600（历史） | 官方具名合同全程；新旧样本分开 |
| QUAD reach | 1000/1000（Oct5） | 1000/1000 | 1000/1000 | 1000/1000 | 全程终点入带；全时语义、证书与native生产门未闭合 |
| Single Pendulum reach | 100/100 | 100/100 | 100/100 | 100/100 | 具名两态全程；三返回量MATLAB合同缺件 |
| TORA remain | 200/200 | 全盒189/200；Safe184 | 全盒189/200；Safe184 | 200/200 | 主h=.1作者两法不完整；补充h=.05四方全程 |
| TORA reach-sigmoid | 500/500（Oct5） | 500/500 | 500/500 | 500/500 | 全程终点入带；GPU未运行性质checker |
| TORA reach-tanh | 500/500 | 500/500（历史） | 500/500（历史） | 500/500（历史） | 全程终点入带；当前P3与三历史方法 |
| Unicycle reach | 500/500（Oct5） | 500/500 | 500/500 | 500/500 | 全程；P3/native终点充分观察，H/X UNKNOWN |

## 1. ACC safe-distance

**明确合同：** `participant-order / v_lead-v_ego`；1 个初盒，50 个控制期 × 0.1 s，请求 T=5 s、50 个 h=0.1 s 小步。

四方均能完成具名 participant-order 合同。保存 tube 的安全距离半空间下界均为正；这不是数值失败。论文未明确相对速度符号，当前按参与者源码使用 v_lead-v_ego。

| 方法 | 全初集有效数值范围 | 性质/直接观察 | 准确停止原因或限制 |
|---|---|---|---|
| P3 | 50/50步；t=5s | 保存范围满足指定性质 | Oct5新完整候选；仅在已保存对象范围内与基线等值。 |
| Huan | 50/50步；t=5s | 保存范围满足指定性质 | 已完成所列具名数值合同；共同证据边界见下。 |
| Xiangru | 50/50步；t=5s | 保存范围满足指定性质 | 已完成所列具名数值合同；共同证据边界见下。 |
| Flow* native | 50/50步；t=5s | VERIFIED | 已完成所列具名数值合同；共同证据边界见下。 |

**性质边界：** 全部 50 段保存 tube 满足安全半空间；完整 T=5。

**还缺什么：** 若要求独立安全定理，仍需 NN 浮点包络、控制注入及 plant 运算的完整包含证明；已有保存盒检查不是该证明。

**合同来源：** [ARCHCOMP26_ACC_PARTICIPANT_CONTRACT_20261001.md](../../../../ARCHCOMP26_ACC_PARTICIPANT_CONTRACT_20261001.md)。

**逐方法原始收据：**

- P3：[2026-10-05_current_candidate:run_001/RESULT](../../../../../research/p3_speed_tightness_20261005/results/acc_p3_fast1_20261005_001/run_001/RESULT.json)；[2026-10-05_current_candidate:candidate/RESULT](../../../../../research/p3_speed_tightness_20261005/results/acc_p3_fast1_20261005_001/run_001/candidate/RESULT.json)；[2026-10-05_current_candidate:data/RESULT](../../../../../research/p3_speed_tightness_20261005/results/acc_p3_fast1_20261005_001/run_001/candidate/data/RESULT.json)；[已存输出比较及明确范围](../../../../../research/p3_speed_tightness_20261005/results/acc_p3_fast1_20261005_001/run_001/candidate/SAVED_COMPARISON.json)。
- Huan：[baseline_receipt:steady05_huan/RESULT](../../archcomp26_20261001/acc_fourway_campaign_001/steady05_huan/RESULT.json)；[baseline_receipt:data/RESULT](../../archcomp26_20261001/acc_fourway_campaign_001/steady05_huan/data/RESULT.json)。
- Xiangru：[baseline_receipt:steady05_xiangru/RESULT](../../archcomp26_20261001/acc_fourway_campaign_001/steady05_xiangru/RESULT.json)；[baseline_receipt:data/RESULT](../../archcomp26_20261001/acc_fourway_campaign_001/steady05_xiangru/data/RESULT.json)。
- Flow* native：[baseline_receipt:steady05_native/RESULT](../../archcomp26_20261001/acc_fourway_campaign_001/steady05_native/RESULT.json)。

**已有绝对宽度：** [acc_t5_endpoint_and_full_tube_long.csv](../../archcomp26_20261001/acc_fourway_saved_ranges_20261001/acc_t5_endpoint_and_full_tube_long.csv)。

**已有图：** [acc_four_method_t_safe_distance_margin_tube.png](../../archcomp26_20261001/plots/nohash_saved_20261001/acc_four_method_t_safe_distance_margin_tube.png)。图来自各自明确保存对象，不能补出缺失时间段。

## 2. Airplane continuous

**明确合同：** `official-2026 continuous / unsplit full box`；1 个初盒，20 个控制期 × 0.1 s，请求 T=2 s、200 个 h=0.01 s 小步。

未分割官方完整初盒没有四方可用首步流管。不同尝试分别遇实现编码/资源限制、Picard 自包含失败或 observer 的 tube/endpoint 不相容，必须分别归因。

| 方法 | 全初集有效数值范围 | 性质/直接观察 | 准确停止原因或限制 |
|---|---|---|---|
| P3 | 0/200步；t=0s | 性质未判定 | P4 验证表在首步前因 9^20 >= 2^63 编码溢出；同阶 P3 的首步 trace 为 x/y/z Picard 提议越出 ±0.01，四次重心化仍 FAILED_CONTRACTION。另扩大 xyz 余项到 ±0.1 后引擎接受，但五坐标 endpoint 超出 tube 1–2 ULP，observer 拒绝，0 个可用保存段。 |
| Huan | 0/200步；t=0s | 性质未判定 | order6 建表资源阻断，RSS超过54,006,540 KiB后终止，未进ODE；order3全盒首步 accepted=false、0/10接受。后一拒绝未记录内部失败坐标，不能猜成控制器错误或真实越界。 |
| Xiangru | 0/200步；t=0s | 性质未判定 | order3全盒首个 h=0.01 小步 accepted=false，0/10接受；内部失败分量未保存，不能猜成物理性质失败。Huan order6资源记录不能冒称为Xiangru实测。 |
| Flow* native | 0/200步；t=0s | UNCOMPLETED_SAFE (4) | order6±0.01、order3±0.01、order3±1 三个全盒 profile 首步均 UNCOMPLETED_SAFE、0段。后续 Real 首拒追踪分别定位 x/y/z/phi/theta 或加宽后的 x/y/phi/theta/psi Picard 不自包含。空 safety 表上的 BOX_SAFE 1 是空集结论。 |

**性质边界：** 主全盒无有效接受段，性质未判定；数值拒绝不是真实轨迹反例。

**还缺什么：** 需要能在完整初集覆盖上通过数值自包含与观察器一致性的明确方法；之后才可推进到 T=2 并逐段判性质。不能把分盒首步或无效候选界补成全程。

64个二分子盒的首步覆盖是有用补充：64/64盒各接受0.01秒，8盒保存tube在安全盒内、56盒UNKNOWN；没有完成首个0.1秒周期或后续199小步。高角111111独立首期数值诊断只接受4/10段至0.04秒，第5步x/y自包含失败。两者均不把主全盒格改成完成。

**合同来源：** [ARCHCOMP26_AIRPLANE_2026_ENTRY_AUDIT.md](../../../../ARCHCOMP26_AIRPLANE_2026_ENTRY_AUDIT.md)。

**逐方法原始收据：**

- P3：[baseline_receipt:airplane_p3_xyz_rem0p1_observer_smoke1_001/RESULT](../../archcomp26_20261001/airplane_p3_xyz_rem0p1_observer_smoke1_001/RESULT.json)；[baseline_receipt:payload/RESULT](../../archcomp26_20261001/airplane_p3_xyz_rem0p1_observer_smoke1_001/payload/RESULT.json)。
- Huan：[baseline_receipt:airplane_continuous_order3_huan_smoke1_001/RESULT](../../archcomp26_20261001/airplane_continuous_order3_huan_smoke1_001/RESULT.json)；[baseline_receipt:payload/RESULT](../../archcomp26_20261001/airplane_continuous_order3_huan_smoke1_001/payload/RESULT.json)。
- Xiangru：[baseline_receipt:airplane_continuous_order3_xiangru_smoke1_001/RESULT](../../archcomp26_20261001/airplane_continuous_order3_xiangru_smoke1_001/RESULT.json)；[baseline_receipt:payload/RESULT](../../archcomp26_20261001/airplane_continuous_order3_xiangru_smoke1_001/payload/RESULT.json)。
- Flow* native：[baseline_receipt:run/RESULT](../../archcomp26_20261001/native_airplane_rem1_first_reject_trace_20261002_006/run/RESULT.json)。

**原始问题定位和既有审计：** [archcomp26_20261001/AIRPLANE_P3_FULLBOX_SMOKES_20261002.md](../../archcomp26_20261001/AIRPLANE_P3_FULLBOX_SMOKES_20261002.md)；[airplane_p3_first_reject_trace_smoke1_001/SUMMARY.md](../../archcomp26_20261001/airplane_p3_first_reject_trace_smoke1_001/SUMMARY.md)；[airplane_p3_xyz_rem0p1_observer_smoke1_001/SUMMARY.md](../../archcomp26_20261001/airplane_p3_xyz_rem0p1_observer_smoke1_001/SUMMARY.md)；[airplane_continuous_order3_fullbox_20261002/SUMMARY.md](../../archcomp26_20261001/airplane_continuous_order3_fullbox_20261002/SUMMARY.md)；[native_airplane_fullbox_smokes_20261002/SUMMARY.md](../../archcomp26_20261001/native_airplane_fullbox_smokes_20261002/SUMMARY.md)；[native_airplane_first_reject_trace_20261002_005/README.md](../../archcomp26_20261001/native_airplane_first_reject_trace_20261002_005/README.md)；[native_airplane_rem1_first_reject_trace_20261002_006/README.md](../../archcomp26_20261001/native_airplane_rem1_first_reject_trace_20261002_006/README.md)；[native_airplane_binary6_numeric_cover_20261003_001/README.md](../../archcomp26_20261001/native_airplane_binary6_numeric_cover_20261003_001/README.md)；[native_airplane_binary6_numeric_cover_20261003_001/INDEPENDENT_AUDIT.json](../../archcomp26_20261001/native_airplane_binary6_numeric_cover_20261003_001/INDEPENDENT_AUDIT.json)；[native_airplane_binary6_period_numeric_20261003_001/README.md](../../archcomp26_20261001/native_airplane_binary6_period_numeric_20261003_001/README.md)。

**宽度为何空缺：** 主全盒没有有效已接受段；不能用未通过观察器的候选界或仅首步子盒补成T=2宽度。

## 3. Airplane discrete

**明确合同：** `authoritative participant discrete execution unresolved`；1 个初盒，20 个控制期 × 0.1 s，请求 T=2 s、20 次转移（检查 k=0…20）。

四方法均因执行合同材料不足而未在权威离散合同下运行，非算法已经失败。论文给 Euler 规则、0.1 秒和 20 次转移，但固定官方目录只有连续 dynamics.m，缺参与者实际 NN 采样与状态更新顺序。

| 方法 | 全初集有效数值范围 | 性质/直接观察 | 准确停止原因或限制 |
|---|---|---|---|
| P3 | 未启动权威合同 | 未运行 | 四方法均因执行合同材料不足而未在权威离散合同下运行，非算法已经失败。论文给 Euler 规则、0.1 秒和 20 次转移，但固定官方目录只有连续 dynamics.m，缺参与者实际 NN 采样与状态更新顺序。 |
| Huan | 未启动权威合同 | 未运行 | 四方法均因执行合同材料不足而未在权威离散合同下运行，非算法已经失败。论文给 Euler 规则、0.1 秒和 20 次转移，但固定官方目录只有连续 dynamics.m，缺参与者实际 NN 采样与状态更新顺序。 |
| Xiangru | 未启动权威合同 | 未运行 | 四方法均因执行合同材料不足而未在权威离散合同下运行，非算法已经失败。论文给 Euler 规则、0.1 秒和 20 次转移，但固定官方目录只有连续 dynamics.m，缺参与者实际 NN 采样与状态更新顺序。 |
| Flow* native | 未启动权威合同 | 未运行 | 四方法均因执行合同材料不足而未在权威离散合同下运行，非算法已经失败。论文给 Euler 规则、0.1 秒和 20 次转移，但固定官方目录只有连续 dynamics.m，缺参与者实际 NN 采样与状态更新顺序。 |

**性质边界：** 无四方离散性质结果；独立 CPU 的 1/20 安全端点诊断不能填四方法格。

**还缺什么：** 取得参与者离散转移及控制更新源码/等价权威执行记录，再建立四个离散入口并覆盖完整初盒 k=0…20；或用户明确另立 paper-Euler-controller-first 补充合同。

**合同来源：** [ARCHCOMP26_AIRPLANE_DISCRETE_EXECUTION_GATE_20261002.md](../../../../ARCHCOMP26_AIRPLANE_DISCRETE_EXECUTION_GATE_20261002.md)。

**逐方法原始收据：**

- P3：无权威合同运行收据；见上述执行门。
- Huan：无权威合同运行收据；见上述执行门。
- Xiangru：无权威合同运行收据；见上述执行门。
- Flow* native：无权威合同运行收据；见上述执行门。

**宽度为何空缺：** 四方法尚未在权威离散合同执行。

## 4. Attitude Control avoid

**明确合同：** `corrected closed unsafe box`；1 个初盒，30 个控制期 × 0.1 s，请求 T=3 s、60 个 h=0.05 s 小步。

四方均能完成修正危险盒后的数值合同，六维保存 tube 与闭危险盒不相交。旧 checker 把 x4 危险区间写成空集，其旧 VERIFIED 不作为本轮证据。

| 方法 | 全初集有效数值范围 | 性质/直接观察 | 准确停止原因或限制 |
|---|---|---|---|
| P3 | 60/60步；t=3s | 保存范围满足指定性质 | 已完成所列具名数值合同；共同证据边界见下。 |
| Huan | 60/60步；t=3s | 保存范围满足指定性质 | 已完成所列具名数值合同；共同证据边界见下。 |
| Xiangru | 60/60步；t=3s | 保存范围满足指定性质 | 已完成所列具名数值合同；共同证据边界见下。 |
| Flow* native | 60/60步；t=3s | VERIFIED | 已完成所列具名数值合同；共同证据边界见下。 |

**性质边界：** 全部 60 段六维保存 tube 避开修正后的闭危险盒。

**还缺什么：** 已有六维保存盒判交可支持数值描述；独立 NNCS 证明仍须补浮点 NN、注入和 plant 包含链。单轴危险投影图不能代替六维判交。

**合同来源：** [ARCHCOMP26_ATTITUDE_CONTROL_CONTRACT_20261001.md](../../../../ARCHCOMP26_ATTITUDE_CONTROL_CONTRACT_20261001.md)。

**逐方法原始收据：**

- P3：[baseline_receipt:later05_ours_p3/RESULT](../../archcomp26_20261001/attitude_corrected_fourway_campaign_20261002_001/later05_ours_p3/RESULT.json)；[baseline_receipt:payload/RESULT](../../archcomp26_20261001/attitude_corrected_fourway_campaign_20261002_001/later05_ours_p3/payload/RESULT.json)。
- Huan：[baseline_receipt:later05_huan/RESULT](../../archcomp26_20261001/attitude_corrected_fourway_campaign_20261002_001/later05_huan/RESULT.json)；[baseline_receipt:payload/RESULT](../../archcomp26_20261001/attitude_corrected_fourway_campaign_20261002_001/later05_huan/payload/RESULT.json)。
- Xiangru：[baseline_receipt:later05_xiangru/RESULT](../../archcomp26_20261001/attitude_corrected_fourway_campaign_20261002_001/later05_xiangru/RESULT.json)；[baseline_receipt:payload/RESULT](../../archcomp26_20261001/attitude_corrected_fourway_campaign_20261002_001/later05_xiangru/payload/RESULT.json)。
- Flow* native：[baseline_receipt:later05_native/RESULT](../../archcomp26_20261001/attitude_corrected_fourway_campaign_20261002_001/later05_native/RESULT.json)。

**已有绝对宽度：** [absolute_widths.csv](../../archcomp26_20261001/report_saved_projections_20261004_001/absolute_widths.csv)；[RUNS.csv](../../archcomp26_20261001/attitude_corrected_fourway_campaign_20261002_001/RUNS.csv)。

**已有图：** [attitude_fourway_t_x4.png](../../archcomp26_20261001/report_saved_projections_20261004_001/attitude_fourway_t_x4.png)。图来自各自明确保存对象，不能补出缺失时间段。

## 5. Balancing reach

**明确合同：** `balancing-fixed-repo-raw4`；1 个初盒，500 个控制期 × 0.02 s，请求 T=10 s、2000 个 h=0.005 s 小步。

有两个独立问题：论文 feature5 控制器缺文件/权威映射，尚不能执行；已冻结的 raw4 合同则确已四方尝试，但数值拒绝都发生在 8–10 秒性质窗之前。

| 方法 | 全初集有效数值范围 | 性质/直接观察 | 准确停止原因或限制 |
|---|---|---|---|
| P3 | 86/2000步；t=0.43s | FAILED_CONTRACTION | raw4 第 87 小步数值拒绝；仅 86/2000 个有效小步，未到8秒性质窗；不是性质反例。论文 feature5 模型/权威映射另缺。 |
| Huan | 98/2000步；t=0.49s | UNKNOWN_OR_INCOMPLETE | raw4 第 99 小步数值拒绝；仅 98/2000 个有效小步，未到8秒性质窗；不是性质反例。论文 feature5 模型/权威映射另缺。 |
| Xiangru | 98/2000步；t=0.49s | FAILED_CONTRACTION | raw4 第 99 小步数值拒绝；仅 98/2000 个有效小步，未到8秒性质窗；不是性质反例。论文 feature5 模型/权威映射另缺。 |
| Flow* native | 83/2000步；t=0.415s | UNCOMPLETED_SAFE (4) | raw4 第 84 小步数值拒绝；仅 83/2000 个有效小步，未到8秒性质窗；不是性质反例。论文 feature5 模型/权威映射另缺。 原生此前解析失败已修正；这里报告修正入口的数值首拒，不继续把旧语法错误当当前阻断。 |

**性质边界：** 性质窗尚未到达，0/400 个目标窗小步检查；UNKNOWN/incomplete。

**还缺什么：** 论文合同需五输入模型及尺度/顺序，或权威五特征到四输入映射。raw4 需明确的新数值方案通过拒绝并覆盖 T=10；不得把旧小初盒 T=1 或拒绝前缀代填。

**合同来源：** [ARCHCOMP26_BALANCING_EXECUTION_GATE_20261002.md](../../../../ARCHCOMP26_BALANCING_EXECUTION_GATE_20261002.md)。

**逐方法原始收据：**

- P3：[baseline_receipt:balancing_fixed_raw4_p3_full500_001/RESULT](../../archcomp26_20261001/balancing_fixed_raw4_p3_full500_001/RESULT.json)；[baseline_receipt:payload/RESULT](../../archcomp26_20261001/balancing_fixed_raw4_p3_full500_001/payload/RESULT.json)。
- Huan：[baseline_receipt:balancing_fixed_raw4_huan_full500_001/RESULT](../../archcomp26_20261001/balancing_fixed_raw4_huan/balancing_fixed_raw4_huan_full500_001/RESULT.json)。
- Xiangru：[baseline_receipt:balancing_fixed_raw4_xiangru_full500_001/RESULT](../../archcomp26_20261001/balancing_fixed_raw4_xiangru_20261002/balancing_fixed_raw4_xiangru_full500_001/RESULT.json)。
- Flow* native：[baseline_receipt:native_balancing_raw4_full500_001/RESULT](../../archcomp26_20261001/native_balancing_raw4_20261002/native_balancing_raw4_full500_001/RESULT.json)。

**原始问题定位和既有审计：** [native_balancing_raw4_20261002/SUMMARY.md](../../archcomp26_20261001/native_balancing_raw4_20261002/SUMMARY.md)；[balancing_fixed_raw4_xiangru_20261002/SUMMARY.md](../../archcomp26_20261001/balancing_fixed_raw4_xiangru_20261002/SUMMARY.md)；[balancing_raw4_fourway_saved_20261004_001/README.md](../../archcomp26_20261001/balancing_raw4_fourway_saved_20261004_001/README.md)。

**已有绝对宽度：** [absolute_widths.csv](../../archcomp26_20261001/balancing_raw4_fourway_saved_20261004_001/output/absolute_widths.csv)；[saved_bounds.csv](../../archcomp26_20261001/balancing_raw4_fourway_saved_20261004_001/output/saved_bounds.csv)。

**已有图：** [balancing_raw4_fourway_common_prefix.png](../../archcomp26_20261001/balancing_raw4_fourway_saved_20261004_001/output/balancing_raw4_fourway_common_prefix.png)。图来自各自明确保存对象，不能补出缺失时间段。

## 6. Docking constraint

**明确合同：** `official full-box q<=0`；1 个初盒，40 个控制期 × 1 s，请求 T=40 s、400 个 h=0.1 s 小步。

四方数值时域均完成，未解决的是性质：轴对齐 tube 对 q 的保守上界从首段就跨过 0，因此不能判全时 q<=0。原生外层 failed/exit2 对应 UNKNOWN，不能据此删除其完整 400 段。

| 方法 | 全初集有效数值范围 | 性质/直接观察 | 准确停止原因或限制 |
|---|---|---|---|
| P3 | 400/400步；t=40s | Unknown | 四方数值时域均完成，未解决的是性质：轴对齐 tube 对 q 的保守上界从首段就跨过 0，因此不能判全时 q<=0。原生外层 failed/exit2 对应 UNKNOWN，不能据此删除其完整 400 段。 |
| Huan | 400/400步；t=40s | Unknown | 四方数值时域均完成，未解决的是性质：轴对齐 tube 对 q 的保守上界从首段就跨过 0，因此不能判全时 q<=0。原生外层 failed/exit2 对应 UNKNOWN，不能据此删除其完整 400 段。 |
| Xiangru | 400/400步；t=40s | Unknown | 四方数值时域均完成，未解决的是性质：轴对齐 tube 对 q 的保守上界从首段就跨过 0，因此不能判全时 q<=0。原生外层 failed/exit2 对应 UNKNOWN，不能据此删除其完整 400 段。 |
| Flow* native | 400/400步；t=40s | UNKNOWN | 四方数值时域均完成，未解决的是性质：轴对齐 tube 对 q 的保守上界从首段就跨过 0，因此不能判全时 q<=0。原生外层 failed/exit2 对应 UNKNOWN，不能据此删除其完整 400 段。 |

**性质边界：** 完整 T=40 数值流管；四方法性质 UNKNOWN。

**还缺什么：** 需要更强的相关性/耦合性质判定、明示合法精化或可信反例来解决 UNKNOWN；不能由 q 上界为正断言真实轨迹违规。

**合同来源：** [ARCHCOMP26_DOCKING_BALANCING_SOURCE_CONTRACT_20261001.md](../../../../ARCHCOMP26_DOCKING_BALANCING_SOURCE_CONTRACT_20261001.md)。

**逐方法原始收据：**

- P3：[baseline_receipt:docking_p3_full40_001/RESULT](../../archcomp26_20261001/docking_p3_full40_001/RESULT.json)。
- Huan：[baseline_receipt:docking_huan_full40_001/RESULT](../../archcomp26_20261001/docking_huan_full40_001/RESULT.json)。
- Xiangru：[baseline_receipt:docking_xiangru_full40_001/RESULT](../../archcomp26_20261001/docking_xiangru_full40_001/RESULT.json)。
- Flow* native：[baseline_receipt:native_docking_full40_001/RESULT](../../archcomp26_20261001/native_docking_full40_001/RESULT.json)。

**已有绝对宽度：** [docking_fourway_saved_widths_20261002.csv](../../archcomp26_20261001/docking_fourway_saved_widths_20261002.csv)；[absolute_widths.csv](../../archcomp26_20261001/report_saved_projections_20261004_001/absolute_widths.csv)。

**已有图：** [docking_fourway_t_sx.png](../../archcomp26_20261001/report_saved_projections_20261004_001/docking_fourway_t_sx.png)；[docking_fullbox_4method_q_upper.png](../../archcomp26_20261001/plots/nohash_saved_20261001/docking_fullbox_4method_q_upper.png)。图来自各自明确保存对象，不能补出缺失时间段。

## 7. Double Pendulum less-robust

**明确合同：** `official less network / 225 boxes`；225 个初盒，20 个控制期 × 0.05 s，请求 T=1 s、100 个 h=0.01 s 小步。

四方能完成 225 盒×100 步。P3 主选是 affine-split4 控制残差入口；旧失败候选另存。保存 tube 都在安全盒内。

| 方法 | 全初集有效数值范围 | 性质/直接观察 | 准确停止原因或限制 |
|---|---|---|---|
| P3 | 100/100步；t=1s | 保存范围满足指定性质 | 已完成所列具名数值合同；共同证据边界见下。 |
| Huan | 100/100步；t=1s | 保存范围满足指定性质 | 已完成所列具名数值合同；共同证据边界见下。少量endpoint末位超出同段tube，分别保留。 |
| Xiangru | 100/100步；t=1s | 保存范围满足指定性质 | 已完成所列具名数值合同；共同证据边界见下。少量endpoint末位超出同段tube，分别保留。 |
| Flow* native | 100/100步；t=1s | VERIFIED | 已完成所列具名数值合同；共同证据边界见下。 |

**性质边界：** 完整 T=1；保存 tube 全在 [-1.7,2]^4。

**还缺什么：** Huan/Xiangru 少量 endpoint 末位超出同段 tube，须继续区分两种界并保留原值；独立浮点 NNCS 包含链仍未闭合，不能凭窄界确认正确性。

**合同来源：** [ARCHCOMP26_DOUBLE_PENDULUM_LESS_CONTRACT_20261001.md](../../../../ARCHCOMP26_DOUBLE_PENDULUM_LESS_CONTRACT_20261001.md)；[ARCHCOMP26_DP_P3_PARTITION_DIAGNOSTIC_20261002.md](../../../../ARCHCOMP26_DP_P3_PARTITION_DIAGNOSTIC_20261002.md)。

**逐方法原始收据：**

- P3：[baseline_receipt:attempt/RESULT](../../archcomp26_20261001/dp_p3_affine_split4_v1/full225_smoke_001/attempt/RESULT.json)；[baseline_receipt:data/RESULT](../../archcomp26_20261001/dp_p3_affine_split4_v1/full225_smoke_001/attempt/data/RESULT.json)。
- Huan：[baseline_receipt:huan_full20_001/RESULT](../../archcomp26_20261001/author_dp_less_v1/huan_full20_001/RESULT.json)；[baseline_receipt:payload/RESULT](../../archcomp26_20261001/author_dp_less_v1/huan_full20_001/payload/RESULT.json)。
- Xiangru：[baseline_receipt:xiangru_full20_001/RESULT](../../archcomp26_20261001/author_dp_less_v1/xiangru_full20_001/RESULT.json)；[baseline_receipt:payload/RESULT](../../archcomp26_20261001/author_dp_less_v1/xiangru_full20_001/payload/RESULT.json)。
- Flow* native：[baseline_receipt:native_dp_less_full20_001/RESULT](../../archcomp26_20261001/native_dp_less_full20_001/RESULT.json)。

**已有绝对宽度：** [endpoint_stats.csv](../../archcomp26_20261001/dp_less_fourway_split4_20261002/endpoint_stats.csv)；[tube_union.csv](../../archcomp26_20261001/dp_less_fourway_split4_20261002/tube_union.csv)。

**已有图：** [fourway_tube_union.png](../../archcomp26_20261001/dp_less_fourway_split4_20261002/fourway_tube_union.png)。图来自各自明确保存对象，不能补出缺失时间段。

## 8. Double Pendulum more-robust

**明确合同：** `official more network / 225 boxes`；225 个初盒，20 个控制期 × 0.02 s，请求 T=0.4 s、80 个 h=0.005 s 小步。

四方主选均是性质检查早停，不是已观察步的数值收缩失败：P3/Huan/Xiangru 数值接受 72/80 步后 Unsafe.；原生数值保存 64/80 步后 UNKNOWN。外层 completed 不等于完成 T=0.4。

| 方法 | 全初集有效数值范围 | 性质/直接观察 | 准确停止原因或限制 |
|---|---|---|---|
| P3 | 72/80步；t=0.36s | Unsafe. | 全部225盒在已保存 72/80 步数值接受，作者 Unsafe. 后停止；保存安全前缀仅60步。第61步开始跨带；第72步发生作者 Unsafe.，不是数值拒绝。 P3外层也completed/exit0而payload为incomplete；旧interval残差入口的第9步FAILED_CONTRACTION另存。 |
| Huan | 72/80步；t=0.36s | Unsafe. | 全部225盒在已保存 72/80 步数值接受，作者 Unsafe. 后停止；保存安全前缀仅60步。第61步开始跨带；第72步发生作者 Unsafe.，不是数值拒绝。 |
| Xiangru | 72/80步；t=0.36s | Unsafe. | 全部225盒在已保存 72/80 步数值接受，作者 Unsafe. 后停止；保存安全前缀仅60步。第61步开始跨带；第72步发生作者 Unsafe.，不是数值拒绝。 |
| Flow* native | 64/80步；t=0.32s | UNKNOWN | 全部225盒在已保存 64/80 步数值接受，作者 UNKNOWN 后停止；保存安全前缀仅60步。原生外层 completed/exit0 只表示进程结束。 |

**性质边界：** 共同数值保存前缀 64 步到 t=0.32；共同保存 Safe 前缀仅 60 步到 t=0.30。作者 Unsafe. 不升级为独立真实轨迹反例。

**还缺什么：** 需要独立可核的反例或更强性质分析以解释早停；若另做不因性质停止的数值诊断须单独命名，不追认现有前缀为完整结果。旧 P3 第9步数值拒绝与新主选应分开。

图保留64个共同数值步至0.32秒，包括61–64步的未知跨带段；不得把整条64步曲线称为安全前缀。P3没有单独保存lane到初始子盒的映射ledger，现有审计确认配置分区、225观察行与pooled首步覆盖，并未独立重建逐lane映射。严格内点区间见证仅到0.1954秒即Picard拒绝，尚未覆盖数值轨迹候选约0.325秒的越界时刻，不能作为已完成反例。

**合同来源：** [ARCHCOMP26_NEXT_CONTRACT_SOURCE_AUDIT_20261001.md](../../../../ARCHCOMP26_NEXT_CONTRACT_SOURCE_AUDIT_20261001.md)。

**逐方法原始收据：**

- P3：[baseline_receipt:attempt/RESULT](../../archcomp26_20261001/dp_more_p3_affine_split4_full20_20261003_001/attempt/RESULT.json)；[baseline_receipt:data/RESULT](../../archcomp26_20261001/dp_more_p3_affine_split4_full20_20261003_001/attempt/data/RESULT.json)。
- Huan：[baseline_receipt:huan_full20_001/RESULT](../../archcomp26_20261001/author_dp_more_v1/huan_full20_001/RESULT.json)；[baseline_receipt:payload/RESULT](../../archcomp26_20261001/author_dp_more_v1/huan_full20_001/payload/RESULT.json)。
- Xiangru：[baseline_receipt:xiangru_full20_001/RESULT](../../archcomp26_20261001/author_dp_more_v1/xiangru_full20_001/RESULT.json)；[baseline_receipt:payload/RESULT](../../archcomp26_20261001/author_dp_more_v1/xiangru_full20_001/payload/RESULT.json)。
- Flow* native：[baseline_receipt:native_dp_more_full20_001/RESULT](../../archcomp26_20261001/native_dp_more_full20_001/RESULT.json)。

**原始问题定位和既有审计：** [dp_more_p3_affine_split4_full20_20261003_001/README.md](../../archcomp26_20261001/dp_more_p3_affine_split4_full20_20261003_001/README.md)；[dp_more_p3_affine_split4_full20_20261003_001/INDEPENDENT_SAVED_INTERVAL_SCAN.json](../../archcomp26_20261001/dp_more_p3_affine_split4_full20_20261003_001/INDEPENDENT_SAVED_INTERVAL_SCAN.json)；[author_dp_more_v1/SUMMARY.md](../../archcomp26_20261001/author_dp_more_v1/SUMMARY.md)；[native_dp_more_full20_001/SUMMARY.md](../../archcomp26_20261001/native_dp_more_full20_001/SUMMARY.md)；[dp_more_fourway_saved_20261004_001/README.md](../../archcomp26_20261001/dp_more_fourway_saved_20261004_001/README.md)；[dp_more_validated_witness_20261002_001/AUDIT.md](../../archcomp26_20261001/dp_more_validated_witness_20261002_001/AUDIT.md)。

**已有绝对宽度：** [absolute_widths.csv](../../archcomp26_20261001/dp_more_fourway_saved_20261004_001/absolute_widths.csv)；[saved_bounds.csv](../../archcomp26_20261001/dp_more_fourway_saved_20261004_001/saved_bounds.csv)。

**已有图：** [dp_more_fourway_common_prefix.png](../../archcomp26_20261001/dp_more_fourway_saved_20261004_001/dp_more_fourway_common_prefix.png)。图来自各自明确保存对象，不能补出缺失时间段。

## 9. NAV standard

**明确合同：** `official point ONNX / author execution order`；640 个初盒，30 个控制期 × 0.2 s，请求 T=6 s、600 个 h=0.01 s 小步。

新 P3/原生和经合同审计的历史 Huan/Xiangru 均有完整 640×600 盒步；不是未完成实例。固定官方 point 模型有同合同记录，另一作者 Git LFS 仓库的模型身份尚未建立。

| 方法 | 全初集有效数值范围 | 性质/直接观察 | 准确停止原因或限制 |
|---|---|---|---|
| P3 | 600/600步；t=6s | VERIFIED | 已完成所列具名数值合同；共同证据边界见下。 |
| Huan | 600/600步；t=6s | VERIFIED | 已有经审计同合同历史全程；不是本轮新计时。 |
| Xiangru | 600/600步；t=6s | VERIFIED | 已有经审计同合同历史全程；不是本轮新计时。 |
| Flow* native | 600/600步；t=6s | VERIFIED | 已完成所列具名数值合同；共同证据边界见下。 |

**性质边界：** 保存逐盒二维 tube 避开闭障碍，T=6 所有终点盒进入目标；x/y 单轴图本身不能证明二维避障。

**还缺什么：** 如需声称复现另一 Git LFS 提交，需其可读取模型及明确映射；当前官方模型具名结果不因此作废。新旧混合时间不可拼为同资源 campaign，独立 NNCS 证明另缺。

**合同来源：** [ARCHCOMP26_NAV_AUTHOR_EXECUTION_CONTRACT_20261002.md](../../../../ARCHCOMP26_NAV_AUTHOR_EXECUTION_CONTRACT_20261002.md)。

**逐方法原始收据：**

- P3：[baseline_receipt:nav_author_standard_working_p3_full30_001/RESULT](../../archcomp26_20261001/nav_author_standard_working_p3_full30_001/RESULT.json)；[baseline_receipt:nav_author_standard_working_p3_full30_001/RESULT](../../archcomp26_20261001/nav_author_standard_working_p3_full30_001/SUPERVISOR_RESULT.json)。
- Huan：[historical_full_numerical_result:nav_standard_huan/RESULT](../../../../../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/nav_standard_huan/result.json)。
- Xiangru：[historical_full_numerical_result:nav_standard_xiangru/RESULT](../../../../../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/nav_standard_xiangru/result.json)。
- Flow* native：[baseline_receipt:nav_author_standard_native_full30_001/RESULT](../../archcomp26_20261001/nav_author_standard_native_full30_001/RESULT.json)。

**已有绝对宽度：** [absolute_widths.csv](../../archcomp26_20261001/nav_fourstate_saved_20261004_001/absolute_widths.csv)；[all_states_saved_curves.csv](../../archcomp26_20261001/nav_fourstate_saved_20261004_001/all_states_saved_curves.csv)。

**已有图：** [nav_saved_xy_tubes_with_working_p3.png](../../archcomp26_20261001/nav_current_p3_saved_20261004_001/nav_saved_xy_tubes_with_working_p3.png)；[nav_each_method_width_difference_vs_native_with_working_p3.png](../../archcomp26_20261001/nav_current_p3_saved_20261004_001/nav_each_method_width_difference_vs_native_with_working_p3.png)。图来自各自明确保存对象，不能补出缺失时间段。

## 10. NAV robust

**明确合同：** `official set ONNX / author execution order`；25 个初盒，30 个控制期 × 0.2 s，请求 T=6 s、600 个 h=0.01 s 小步。

新 P3 和经合同审计的历史 Huan/Xiangru/原生均有完整 25×600 盒步。robust 是控制器训练方式，不向当前 plant 擅加训练噪声。

| 方法 | 全初集有效数值范围 | 性质/直接观察 | 准确停止原因或限制 |
|---|---|---|---|
| P3 | 600/600步；t=6s | VERIFIED | Oct5新完整候选；仅在已保存对象范围内与基线等值。 |
| Huan | 600/600步；t=6s | VERIFIED | 已有经审计同合同历史全程；不是本轮新计时。 |
| Xiangru | 600/600步；t=6s | VERIFIED | 已有经审计同合同历史全程；不是本轮新计时。 |
| Flow* native | 600/600步；t=6s | VERIFIED | 已有经审计同合同历史全程；不是本轮新计时。 |

**性质边界：** 保存逐盒二维 tube 避障且 T=6 终点入目标。

**还缺什么：** 另一 Git LFS 模型身份仍需外部材料；历史结果保留历史身份。四态保存统计已经补齐，不能再写成四态宽度缺失；独立 NNCS 证明及同资源重复比较另缺。

**合同来源：** [ARCHCOMP26_NAV_AUTHOR_EXECUTION_CONTRACT_20261002.md](../../../../ARCHCOMP26_NAV_AUTHOR_EXECUTION_CONTRACT_20261002.md)。

**逐方法原始收据：**

- P3：[2026-10-05_current_candidate:run_001/RESULT](../../../../../research/p3_speed_tightness_20261005/results/nav_robust_p3_fast32_20261005_001/run_001/RESULT.json)；[2026-10-05_current_candidate:candidate/RESULT](../../../../../research/p3_speed_tightness_20261005/results/nav_robust_p3_fast32_20261005_001/run_001/candidate/RESULT.json)；[2026-10-05_current_candidate:data/RESULT](../../../../../research/p3_speed_tightness_20261005/results/nav_robust_p3_fast32_20261005_001/run_001/candidate/data/RESULT.json)；[已存输出比较及明确范围](../../../../../research/p3_speed_tightness_20261005/results/nav_robust_p3_fast32_20261005_001/run_001/candidate/SAVED_COMPARISON.json)。
- Huan：[historical_full_numerical_result:nav_robust_r2_huan/RESULT](../../../../../../../../results/archcomp_review_20260923/evidence_v2/timing_v1/nav_robust_r2_huan/result.json)。
- Xiangru：[historical_full_numerical_result:nav_robust_xiangru/RESULT](../../../../../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/nav_robust_xiangru/result.json)。
- Flow* native：[historical_full_numerical_result:nav_robust_native_historical_20260923/RESULT](../../archcomp26_20261001/nav_robust_native_historical_20260923/result.json)。

**已有绝对宽度：** [absolute_widths.csv](../../archcomp26_20261001/nav_fourstate_saved_20261004_001/absolute_widths.csv)；[all_states_saved_curves.csv](../../archcomp26_20261001/nav_fourstate_saved_20261004_001/all_states_saved_curves.csv)。

**已有图：** [nav_saved_xy_tubes_with_working_p3.png](../../archcomp26_20261001/nav_current_p3_saved_20261004_001/nav_saved_xy_tubes_with_working_p3.png)；[nav_each_method_width_difference_vs_native_with_working_p3.png](../../archcomp26_20261001/nav_current_p3_saved_20261004_001/nav_each_method_width_difference_vs_native_with_working_p3.png)。图来自各自明确保存对象，不能补出缺失时间段。

## 11. QUAD reach

**明确合同：** `2026 paper ODE / same-slope CROWN`；1024 个初盒，50 个控制期 × 0.1 s，请求 T=5 s、1000 个 h=0.005 s 小步。

四方均完成论文 ODE 的 1024×1000 盒步；T=5 x3 并集均入 [0.94,1.06]。尚未完成的是权威 reach-and-remain 全时间窗语义/检查、独立 NNCS 证明及 native octagon 生产资格。

| 方法 | 全初集有效数值范围 | 性质/直接观察 | 准确停止原因或限制 |
|---|---|---|---|
| P3 | 1000/1000步；t=5s | 终点入带；全时检查未闭合 | 四方均完成论文 ODE 的 1024×1000 盒步；T=5 x3 并集均入 [0.94,1.06]。尚未完成的是权威 reach-and-remain 全时间窗语义/检查、独立 NNCS 证明及 native octagon 生产资格。 |
| Huan | 1000/1000步；t=5s | 终点入带；全时检查未闭合 | 四方均完成论文 ODE 的 1024×1000 盒步；T=5 x3 并集均入 [0.94,1.06]。尚未完成的是权威 reach-and-remain 全时间窗语义/检查、独立 NNCS 证明及 native octagon 生产资格。 |
| Xiangru | 1000/1000步；t=5s | 终点入带；全时检查未闭合 | 四方均完成论文 ODE 的 1024×1000 盒步；T=5 x3 并集均入 [0.94,1.06]。尚未完成的是权威 reach-and-remain 全时间窗语义/检查、独立 NNCS 证明及 native octagon 生产资格。 |
| Flow* native | 1000/1000步；t=5s | VERIFIED | 四方均完成论文 ODE 的 1024×1000 盒步；T=5 x3 并集均入 [0.94,1.06]。尚未完成的是权威 reach-and-remain 全时间窗语义/检查、独立 NNCS 证明及 native octagon 生产资格。 |

**性质边界：** 四方仅有共同终点入带观察。P3/原生保存包络分别约自 3.95/3.87 秒持续入带；Huan/Xiangru 未保存逐步坐标，无法同样扫描。

**还缺什么：** 取得参与者 reach-and-remain checker 或等价权威记录；Huan/Xiangru 若要画全时曲线需新的逐步坐标记录。原生须在控制构造修正后重新建立 plant、多期与全时包含资格，不能由条件性首批构造门外推。

当前论文P3逐步**计算逐盒**tube/endpoint，再保存**pooled投影、接受计数和状态**；其新旧等值范围仅为这些保存对象和最终字段，未比较隐藏TM/SR或未保存的逐盒范围。Huan/Xiangru未保存逐步坐标，因此全时图只能给其终点，逐盒mean/max不能从并集宽度反推。

原生octagon生产门仍关闭。10月4日新隔离回放已通过1024盒、3072控制输出行、196608个精确顶点的最终余项构造检查，但前提是原实数仿射CROWN界有效，且没有调用NN/CROWN或ODE。它补齐保存首批控制构造的条件性包含，不认证后续控制、plant或全时性质；不能把原长作业的原库资格追认成修补库资格。

**合同来源：** [ARCHCOMP26_QUAD_PAPER_CONTRACT_DECISION_20261001.md](../../../../ARCHCOMP26_QUAD_PAPER_CONTRACT_DECISION_20261001.md)。

**逐方法原始收据：**

- P3：[2026-10-05_current_candidate:run_001/RESULT](../../../../../research/p3_speed_tightness_20261005/results/quad_paper_p3_private256_full1000_20261005_001/run_001/RESULT.json)；[2026-10-05_current_candidate:candidate/RESULT](../../../../../research/p3_speed_tightness_20261005/results/quad_paper_p3_private256_full1000_20261005_001/run_001/candidate/RESULT.json)；[2026-10-05_current_candidate:data/RESULT](../../../../../research/p3_speed_tightness_20261005/results/quad_paper_p3_private256_full1000_20261005_001/run_001/candidate/data/RESULT.json)；[已存输出比较及明确范围](../../../../../research/p3_speed_tightness_20261005/results/quad_paper_p3_private256_full1000_20261005_001/run_001/candidate/SAVED_COMPARISON.json)。
- Huan：[baseline_receipt:supervisor/RESULT](../../archcomp26_20261001/quad_paper_huan_full50_001/supervisor/RESULT.json)。
- Xiangru：[baseline_receipt:full50_001/RESULT](../../archcomp26_20261001/quad_paper_xiangru_v1/full50_001/RESULT.json)；[baseline_receipt:data/RESULT](../../archcomp26_20261001/quad_paper_xiangru_v1/full50_001/data/RESULT.json)。
- Flow* native：[baseline_receipt:native_quad_paper_full50_001/RESULT](../../archcomp26_20261001/native_quad_paper_full50_001/RESULT.json)。

**原始问题定位和既有审计：** [docs/ARCHCOMP26_NATIVE_OCTAGON_PRODUCTION_GATE_20261002.md](../../../../ARCHCOMP26_NATIVE_OCTAGON_PRODUCTION_GATE_20261002.md)；[quad_paper_fourway_saved_20261002/SUMMARY.md](../../archcomp26_20261001/quad_paper_fourway_saved_20261002/SUMMARY.md)；[native_quad_allbox_adaptive_remainder_replay_20261004_001/README.md](../../archcomp26_20261001/native_quad_allbox_adaptive_remainder_replay_20261004_001/README.md)；[native_quad_allbox_adaptive_remainder_replay_20261004_001/RESULT.json](../../archcomp26_20261001/native_quad_allbox_adaptive_remainder_replay_20261004_001/RESULT.json)。

**已有绝对宽度：** [terminal_12states_fourway.csv](../../archcomp26_20261001/quad_paper_fourway_saved_20261002/terminal_12states_fourway.csv)；[SCAN.json](../../archcomp26_20261001/native_quad_paper_full50_001/SCAN.json)。

**已有图：** [quad_paper_fourway_t_x3_pooled_tube.png](../../archcomp26_20261001/quad_paper_fourway_saved_20261002/quad_paper_fourway_t_x3_pooled_tube.png)；[quad_paper_fourway_endpoint_x1_x3.png](../../archcomp26_20261001/report_saved_projections_20261004_001/quad_paper_fourway_endpoint_x1_x3.png)。图来自各自明确保存对象，不能补出缺失时间段。

## 12. Single Pendulum reach

**明确合同：** `named two-physical-state profile`；1 个初盒，20 个控制期 × 0.05 s，请求 T=1 s、100 个 h=0.01 s 小步。

具名两物理态合同四方均完成且保存性质窗安全。它不能冒名为官方 MATLAB 三返回量执行：dynamics_sp.m 另有 dx3=1，却未给第三态初值、重置和该 MATLAB 闭环/checker 入口。

| 方法 | 全初集有效数值范围 | 性质/直接观察 | 准确停止原因或限制 |
|---|---|---|---|
| P3 | 100/100步；t=1s | 保存范围满足指定性质 | 具名两物理态合同四方均完成且保存性质窗安全。它不能冒名为官方 MATLAB 三返回量执行：dynamics_sp.m 另有 dx3=1，却未给第三态初值、重置和该 MATLAB 闭环/checker 入口。 |
| Huan | 100/100步；t=1s | 保存范围满足指定性质 | 具名两物理态合同四方均完成且保存性质窗安全。它不能冒名为官方 MATLAB 三返回量执行：dynamics_sp.m 另有 dx3=1，却未给第三态初值、重置和该 MATLAB 闭环/checker 入口。 |
| Xiangru | 100/100步；t=1s | 保存范围满足指定性质 | 具名两物理态合同四方均完成且保存性质窗安全。它不能冒名为官方 MATLAB 三返回量执行：dynamics_sp.m 另有 dx3=1，却未给第三态初值、重置和该 MATLAB 闭环/checker 入口。 |
| Flow* native | 100/100步；t=1s | VERIFIED | 具名两物理态合同四方均完成且保存性质窗安全。它不能冒名为官方 MATLAB 三返回量执行：dynamics_sp.m 另有 dx3=1，却未给第三态初值、重置和该 MATLAB 闭环/checker 入口。 |

**性质边界：** 具名两态 T=1 完整，t∈[0.5,1] 的保存 x1 tube 在 [0,1] 内；不覆盖未定义的三态执行身份。

**还缺什么：** 若要复现官方三返回量 MATLAB 提交，需第三态初始化/重置语义及实际闭环检查源码；不得擅自把该态当当前辅助时钟。具名两态数值结论可保留，独立证明另缺。

两态profile中的辅助时钟t(0)=0、t′=1不送入网络；这项显式选择不能补写未知MATLAB第三返回量的初值/重置。论文和规格的两个物理态合同与未提供的三返回量执行入口分开。

**合同来源：** [ARCHCOMP26_NEXT_CONTRACT_SOURCE_AUDIT_20261001.md](../../../../ARCHCOMP26_NEXT_CONTRACT_SOURCE_AUDIT_20261001.md)；[ARCHCOMP26_SINGLE_PENDULUM_THIRD_STATE_SOURCE_FOLLOWUP_20261003.md](../../../../ARCHCOMP26_SINGLE_PENDULUM_THIRD_STATE_SOURCE_FOLLOWUP_20261003.md)。

**逐方法原始收据：**

- P3：[baseline_receipt:later05_ours_p3/RESULT](../../archcomp26_20261001/sp_two_state_fourway_campaign_20261002_001/later05_ours_p3/RESULT.json)；[baseline_receipt:data/RESULT](../../archcomp26_20261001/sp_two_state_fourway_campaign_20261002_001/later05_ours_p3/data/RESULT.json)。
- Huan：[baseline_receipt:later05_huan/RESULT](../../archcomp26_20261001/sp_two_state_fourway_campaign_20261002_001/later05_huan/RESULT.json)；[baseline_receipt:data/RESULT](../../archcomp26_20261001/sp_two_state_fourway_campaign_20261002_001/later05_huan/data/RESULT.json)。
- Xiangru：[baseline_receipt:later05_xiangru/RESULT](../../archcomp26_20261001/sp_two_state_fourway_campaign_20261002_001/later05_xiangru/RESULT.json)；[baseline_receipt:data/RESULT](../../archcomp26_20261001/sp_two_state_fourway_campaign_20261002_001/later05_xiangru/data/RESULT.json)。
- Flow* native：[baseline_receipt:later05_native/RESULT](../../archcomp26_20261001/sp_two_state_fourway_campaign_20261002_001/later05_native/RESULT.json)。

**已有绝对宽度：** [endpoint_and_window_widths.csv](../../archcomp26_20261001/sp_two_state_fourway_saved_20261002/endpoint_and_window_widths.csv)；[saved_tubes.csv](../../archcomp26_20261001/sp_two_state_fourway_saved_20261002/saved_tubes.csv)。

**已有图：** [fourway_saved_bounds.png](../../archcomp26_20261001/sp_two_state_fourway_saved_20261002/fourway_saved_bounds.png)。图来自各自明确保存对象，不能补出缺失时间段。

## 13. TORA remain

**明确合同：** `official remain / fixed main h=0.1`；12 个初盒，20 个控制期 × 1 s，请求 T=20 s、200 个 h=0.1 s 小步。

主表固定 h=0.1：P3/原生完整，Huan/Xiangru 在第190步开始丢失已接受初盒，虽仍观察至第200步也没有全初集 T=20 流管。性质 Unknown 不是数值拒绝原因。

| 方法 | 全初集有效数值范围 | 性质/直接观察 | 准确停止原因或限制 |
|---|---|---|---|
| P3 | 200/200步；t=20s | 保存范围满足指定性质 | 固定 h=0.1 全部12盒×200步完整，保存 tube 全在安全盒内；不是未完成方法。 |
| Huan | 189/200步；t=18.9s；末次观察t=20，6/12盒 | Unknown. | 固定h=0.1第190步首次拒绝初盒；全盒接受前缀189步、安全前缀184步。仍观察200步，但总计2357/2400盒步接受、末步仅6/12，不能给全初集T=20终点。Huan首拒trace是盒2的x2 Picard提议越出±0.01；Xiangru未独立记录该内部坐标，不能冒称已逐内部trace验证。 |
| Xiangru | 189/200步；t=18.9s；末次观察t=20，6/12盒 | Unknown. | 固定h=0.1第190步首次拒绝初盒；全盒接受前缀189步、安全前缀184步。仍观察200步，但总计2357/2400盒步接受、末步仅6/12，不能给全初集T=20终点。Huan首拒trace是盒2的x2 Picard提议越出±0.01；Xiangru未独立记录该内部坐标，不能冒称已逐内部trace验证。 |
| Flow* native | 200/200步；t=20s | VERIFIED | 固定 h=0.1 全部12盒×200步完整，保存 tube 全在安全盒内；不是未完成方法。 |

**性质边界：** h=0.1 共同全盒接受且保存安全前缀为184步；H/X第185步跨带，第190步首拒，第200步仅6/12盒接受。

**还缺什么：** 固定 h=0.1 两作者需明确通过自包含的新方法/参数资格；原源码没有可直接启用的自映射重试开关。另列 h=0.05 已四方完整，但不能暗换冻结 h=0.1 主格。

另列h=0.05补充profile已四方400/400步、12×400有效盒步，保存tube均安全。它证明这些方法在该明示数值步长下可完成，并不改写h=0.1主格。Huan h=0.1首拒的实测原因是盒2 x2 Picard提议越出±0.01；扩大到±0.02的另一个profile仍在第192步拒绝，不能写成已解决。Xiangru具有相同外部拒绝记录，未单独记录上述内部trace，内部细因不可冒称逐项验证。

**合同来源：** [ARCHCOMP26_TORA_REMAIN_CONTRACT_20261001.md](../../../../ARCHCOMP26_TORA_REMAIN_CONTRACT_20261001.md)；[ARCHCOMP26_TORA_REMAIN_NUMERIC_STOP_AUDIT_20261002.md](../../../../ARCHCOMP26_TORA_REMAIN_NUMERIC_STOP_AUDIT_20261002.md)。

**逐方法原始收据：**

- P3：[baseline_receipt:full20_001/RESULT](../../archcomp26_20261001/p3_tora_remain_v1/full20_001/RESULT.json)；[baseline_receipt:payload/RESULT](../../archcomp26_20261001/p3_tora_remain_v1/full20_001/payload/RESULT.json)。
- Huan：[baseline_receipt:huan_full20_001/RESULT](../../archcomp26_20261001/author_tora_remain_v1/huan_full20_001/RESULT.json)；[baseline_receipt:payload/RESULT](../../archcomp26_20261001/author_tora_remain_v1/huan_full20_001/payload/RESULT.json)。
- Xiangru：[baseline_receipt:xiangru_full20_001/RESULT](../../archcomp26_20261001/author_tora_remain_v1/xiangru_full20_001/RESULT.json)；[baseline_receipt:payload/RESULT](../../archcomp26_20261001/author_tora_remain_v1/xiangru_full20_001/payload/RESULT.json)。
- Flow* native：[baseline_receipt:native_tora_remain_full20_001/RESULT](../../archcomp26_20261001/native_tora_remain_full20_001/RESULT.json)。

**原始问题定位和既有审计：** [tora_remain_fourway_common_prefix_20261002/SUMMARY.md](../../archcomp26_20261001/tora_remain_fourway_common_prefix_20261002/SUMMARY.md)；[author_tora_remain_h01_refusal_trace_20261002_001/README.md](../../archcomp26_20261001/author_tora_remain_h01_refusal_trace_20261002_001/README.md)；[author_tora_remain_h01_x2rem002_step192_trace_20261003_001/README.md](../../archcomp26_20261001/author_tora_remain_h01_x2rem002_step192_trace_20261003_001/README.md)；[tora_remain_h005_fourway_saved_20261002/SUMMARY.md](../../archcomp26_20261001/tora_remain_h005_fourway_saved_20261002/SUMMARY.md)。

**已有绝对宽度：** [RANGES.csv](../../archcomp26_20261001/tora_remain_fourway_common_prefix_20261002/RANGES.csv)。

**已有图：** [tora_remain_2026_fourway_t_x4_saved_tube.png](../../archcomp26_20261001/tora_remain_fourway_common_prefix_20261002/plots/tora_remain_2026_fourway_t_x4_saved_tube.png)。图来自各自明确保存对象，不能补出缺失时间段。

## 14. TORA reach-sigmoid

**明确合同：** `official 2026 sigmoid / u=11f`；1 个初盒，10 个控制期 × 0.5 s，请求 T=5 s、500 个 h=0.01 s 小步。

四方均完成官方 sigmoid、u=11f 的 500 步，保存 T=5 目标坐标入目标。三 GPU 明确没有执行性质 checker；不能给它们补写作者 VERIFIED。

| 方法 | 全初集有效数值范围 | 性质/直接观察 | 准确停止原因或限制 |
|---|---|---|---|
| P3 | 500/500步；t=5s | 未运行作者性质checker | Oct5新完整候选；仅在已保存对象范围内与基线等值。 |
| Huan | 500/500步；t=5s | 未运行作者性质checker | 已完成所列具名数值合同；共同证据边界见下。 |
| Xiangru | 500/500步；t=5s | 未运行作者性质checker | 已完成所列具名数值合同；共同证据边界见下。 |
| Flow* native | 500/500步；t=5s | VERIFIED | 已完成所列具名数值合同；共同证据边界见下。 |

**性质边界：** T=5 保存 x1/x2 终点全盒入目标；三 GPU property_evaluated=false，原生仅终点 VERIFIED。

**还缺什么：** 若需作者性质标签，需与选定到达语义一致的 checker；终点包含是五秒内到达的充分数值观察。原生终点 VERIFIED 仍不是独立端到端证明。

**合同来源：** [ARCHCOMP26_TORA_REACH_EXECUTION_GATE_20261002.md](../../../../ARCHCOMP26_TORA_REACH_EXECUTION_GATE_20261002.md)。

**逐方法原始收据：**

- P3：[2026-10-05_current_candidate:run_001/RESULT](../../../../../research/p3_speed_tightness_20261005/results/tora_sigmoid_official_u11_p3_fused1_20261005_001/run_001/RESULT.json)；[2026-10-05_current_candidate:candidate/RESULT](../../../../../research/p3_speed_tightness_20261005/results/tora_sigmoid_official_u11_p3_fused1_20261005_001/run_001/candidate/RESULT.json)；[2026-10-05_current_candidate:data/RESULT](../../../../../research/p3_speed_tightness_20261005/results/tora_sigmoid_official_u11_p3_fused1_20261005_001/run_001/candidate/data/RESULT.json)；[已存输出比较及明确范围](../../../../../research/p3_speed_tightness_20261005/results/tora_sigmoid_official_u11_p3_fused1_20261005_001/run_001/candidate/SAVED_COMPARISON.json)。
- Huan：[baseline_receipt:outer/RESULT](../../archcomp26_20261001/tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/later05_huan/outer/RESULT.json)；[baseline_receipt:payload/RESULT](../../archcomp26_20261001/tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/later05_huan/payload/RESULT.json)。
- Xiangru：[baseline_receipt:outer/RESULT](../../archcomp26_20261001/tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/later05_xiangru/outer/RESULT.json)；[baseline_receipt:payload/RESULT](../../archcomp26_20261001/tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/later05_xiangru/payload/RESULT.json)。
- Flow* native：[baseline_receipt:later05_native/RESULT](../../archcomp26_20261001/tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/later05_native/RESULT.json)。

**已有绝对宽度：** [tora_reach_sigmoid_official2026_mat_u11_terminal_fourway.csv](../../archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_terminal_fourway.csv)。

**已有图：** [tora_reach_sigmoid_u11_fourway_saved.png](../../archcomp26_20261001/tora_reach_sigmoid_u11_fourway_saved_figure_20261002/tora_reach_sigmoid_u11_fourway_saved.png)。图来自各自明确保存对象，不能补出缺失时间段。

## 15. TORA reach-tanh

**明确合同：** `official ReLU3/tanh / u=11f`；1 个初盒，10 个控制期 × 0.5 s，请求 T=5 s、500 个 h=0.01 s 小步。

新 working P3 与三历史作者方法均完成同合同500步，目标终点入带。旧 engine_linear_leaf_v2 我方记录另列，不能当最新 P3。

| 方法 | 全初集有效数值范围 | 性质/直接观察 | 准确停止原因或限制 |
|---|---|---|---|
| P3 | 500/500步；t=5s | 未运行作者性质checker | 已完成所列具名数值合同；共同证据边界见下。 |
| Huan | 500/500步；t=5s | 未运行作者性质checker | 已有经审计同合同历史全程；不是本轮新计时。 |
| Xiangru | 500/500步；t=5s | 未运行作者性质checker | 已有经审计同合同历史全程；不是本轮新计时。 |
| Flow* native | 500/500步；t=5s | VERIFIED | 已有经审计同合同历史全程；不是本轮新计时。 |

**性质边界：** 保存 T=5 x1/x2 终点入目标，是五秒内到达的充分数值观察；非独立证书。

**还缺什么：** 当前 P3 与历史 GPU 无作者性质 verdict；需按选定到达语义补相应判定才可声称作者检查通过。历史原生仅终点 VERIFIED；混代不作稳定速度排名。

**合同来源：** [ARCHCOMP26_TORA_REACH_EXECUTION_GATE_20261002.md](../../../../ARCHCOMP26_TORA_REACH_EXECUTION_GATE_20261002.md)。

**逐方法原始收据：**

- P3：[baseline_receipt:tora_reach_tanh_official2026_mat_u11_workingp3_full500_20261002_001/RESULT](../../archcomp26_20261001/tora_reach_tanh_official2026_mat_u11_workingp3_full500_20261002_001/RESULT.json)。
- Huan：[historical_full_numerical_result:tora_relu_tanh_huan/RESULT](../../../../../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/tora_relu_tanh_huan/result.json)。
- Xiangru：[historical_full_numerical_result:tora_relu_tanh_xiangru/RESULT](../../../../../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/tora_relu_tanh_xiangru/result.json)。
- Flow* native：[historical_full_numerical_result:tora_relu_tanh/RESULT](../../../../../../../../results/archcomp_review_20260923/evidence_v2/native_matched/tora_relu_tanh/result.json)。

**已有绝对宽度：** [absolute_widths.csv](../../archcomp26_20261001/tora_tanh_current_p3_saved_20261004_001/absolute_widths.csv)；[saved_bounds.csv](../../archcomp26_20261001/tora_tanh_current_p3_saved_20261004_001/saved_bounds.csv)。

**已有图：** [tora_tanh_current_fourway.png](../../archcomp26_20261001/tora_tanh_current_p3_saved_20261004_001/tora_tanh_current_fourway.png)；[tora_tanh_p3_generations.png](../../archcomp26_20261001/tora_tanh_current_p3_saved_20261004_001/tora_tanh_p3_generations.png)。图来自各自明确保存对象，不能补出缺失时间段。

## 16. Unicycle reach

**明确合同：** `paper-speed / constant w in [-1e-4,1e-4]`；1 个初盒，50 个控制期 × 0.2 s，请求 T=10 s、500 个 h=0.02 s 小步。

四方均完成用户选定的论文速度导数加常值扰动合同。P3/原生保存终点入目标；Huan/Xiangru 的保存 endpoint 不能整体落入目标，性质仍 UNKNOWN。

| 方法 | 全初集有效数值范围 | 性质/直接观察 | 准确停止原因或限制 |
|---|---|---|---|
| P3 | 500/500步；t=10s | ENDPOINT_SUFFICIENT_FOR_REACH | 四方均完成用户选定的论文速度导数加常值扰动合同。P3/原生保存终点入目标；Huan/Xiangru 的保存 endpoint 不能整体落入目标，性质仍 UNKNOWN。 |
| Huan | 500/500步；t=10s | UNKNOWN | 四方均完成用户选定的论文速度导数加常值扰动合同。P3/原生保存终点入目标；Huan/Xiangru 的保存 endpoint 不能整体落入目标，性质仍 UNKNOWN。 |
| Xiangru | 500/500步；t=10s | UNKNOWN | 四方均完成用户选定的论文速度导数加常值扰动合同。P3/原生保存终点入目标；Huan/Xiangru 的保存 endpoint 不能整体落入目标，性质仍 UNKNOWN。 |
| Flow* native | 500/500步；t=10s | VERIFIED | 四方均完成用户选定的论文速度导数加常值扰动合同。P3/原生保存终点入目标；Huan/Xiangru 的保存 endpoint 不能整体落入目标，性质仍 UNKNOWN。 |

**性质边界：** P3/原生最早保存全盒 endpoint 入目标分别约 t=9.72/9.80；H/X 的全部500保存 endpoint 无一次四态同时全入目标。

**还缺什么：** H/X 需更强到达性分析或可信反例；500个保存 endpoint 都未全入目标不能推出连续窗口不可达。常值 w 是具名选择，不覆盖论文未明确的任意时变扰动。

**合同来源：** [ARCHCOMP26_UNICYCLE_EXECUTION_GATE_20261002.md](../../../../ARCHCOMP26_UNICYCLE_EXECUTION_GATE_20261002.md)。

**逐方法原始收据：**

- P3：[2026-10-05_current_candidate:run_001/RESULT](../../../../../research/p3_speed_tightness_20261005/results/unicycle_p3_fused1_20261005_001/run_001/RESULT.json)；[2026-10-05_current_candidate:candidate/RESULT](../../../../../research/p3_speed_tightness_20261005/results/unicycle_p3_fused1_20261005_001/run_001/candidate/RESULT.json)；[2026-10-05_current_candidate:data/RESULT](../../../../../research/p3_speed_tightness_20261005/results/unicycle_p3_fused1_20261005_001/run_001/candidate/data/RESULT.json)；[已存输出比较及明确范围](../../../../../research/p3_speed_tightness_20261005/results/unicycle_p3_fused1_20261005_001/run_001/candidate/SAVED_COMPARISON.json)。
- Huan：[baseline_receipt:huan_full50_001/RESULT](../../archcomp26_20261001/unicycle_paper_speed_w_constant_v1/huan_full50_001/RESULT.json)；[baseline_receipt:payload/RESULT](../../archcomp26_20261001/unicycle_paper_speed_w_constant_v1/huan_full50_001/payload/RESULT.json)。
- Xiangru：[baseline_receipt:xiangru_full50_001/RESULT](../../archcomp26_20261001/unicycle_paper_speed_w_constant_v1/xiangru_full50_001/RESULT.json)；[baseline_receipt:payload/RESULT](../../archcomp26_20261001/unicycle_paper_speed_w_constant_v1/xiangru_full50_001/payload/RESULT.json)。
- Flow* native：[baseline_receipt:native_unicycle_paper_speed_full50_001/RESULT](../../archcomp26_20261001/native_unicycle_paper_speed_full50_001/RESULT.json)。

**已有绝对宽度：** [terminal_fourway.csv](../../archcomp26_20261001/unicycle_paper_speed_w_constant_v1/terminal_fourway.csv)。

**已有图：** [unicycle_paper_speed_fourway_saved.png](../../archcomp26_20261001/unicycle_paper_speed_w_constant_v1/plots/fourway_saved_20261002/unicycle_paper_speed_fourway_saved.png)。图来自各自明确保存对象，不能补出缺失时间段。

## 交付审计范围

只读取现有64格索引、五个Oct5主选的START/RESULT与保存比较收据、上述合同和已有审计。生成时核对16个实例×4种方法键唯一、64格齐全、所有引用本地路径存在；这些是文档结构检查，不是重新执行科学校验。

历史八格完整结果保持历史身份；后来单盒smoke不是其全程来源。64格JSON保留原始收据字段、原始性质字段、主选与基线两个来源，避免将矩阵派生分类冒充作者逐字verdict。所有状态均不宣称独立端到端NNCS证书。
