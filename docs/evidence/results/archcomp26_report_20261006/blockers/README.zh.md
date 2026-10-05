# 10 月 6 日 64 格状态与证据边界

数值完整、作者标签、保存几何观察和独立证明分别保留。原合同缺件、数值早停、缺保存数据及性质 UNKNOWN 的原因不因提速改变；本表仅更新当前 P3 收据。

## ACC safe-distance / P3

状态：full_numeric_horizon。四方均能完成具名 participant-order 合同。保存 tube 的安全距离半空间下界均为正；这不是数值失败。论文未明确相对速度符号，当前按参与者源码使用 v_lead-v_ego。

仍需：若要求独立安全定理，仍需 NN 浮点包络、控制注入及 plant 运算的完整包含证明；已有保存盒检查不是该证明。

当前收据：`research/p3_speed_tightness_20261005/results/acc_p3_fast1_20261005_001/run_001/RESULT.json`；`research/p3_speed_tightness_20261005/results/acc_p3_fast1_20261005_001/run_001/candidate/RESULT.json`；`research/p3_speed_tightness_20261005/results/acc_p3_fast1_20261005_001/run_001/candidate/data/RESULT.json`

## ACC safe-distance / Huan

状态：full_numeric_horizon。四方均能完成具名 participant-order 合同。保存 tube 的安全距离半空间下界均为正；这不是数值失败。论文未明确相对速度符号，当前按参与者源码使用 v_lead-v_ego。

仍需：若要求独立安全定理，仍需 NN 浮点包络、控制注入及 plant 运算的完整包含证明；已有保存盒检查不是该证明。

当前收据：`docs/evidence/results/archcomp26_20261001/acc_fourway_campaign_001/steady05_huan/RESULT.json`；`docs/evidence/results/archcomp26_20261001/acc_fourway_campaign_001/steady05_huan/data/RESULT.json`

## ACC safe-distance / Xiangru

状态：full_numeric_horizon。四方均能完成具名 participant-order 合同。保存 tube 的安全距离半空间下界均为正；这不是数值失败。论文未明确相对速度符号，当前按参与者源码使用 v_lead-v_ego。

仍需：若要求独立安全定理，仍需 NN 浮点包络、控制注入及 plant 运算的完整包含证明；已有保存盒检查不是该证明。

当前收据：`docs/evidence/results/archcomp26_20261001/acc_fourway_campaign_001/steady05_xiangru/RESULT.json`；`docs/evidence/results/archcomp26_20261001/acc_fourway_campaign_001/steady05_xiangru/data/RESULT.json`

## ACC safe-distance / Flow* native

状态：full_numeric_horizon。四方均能完成具名 participant-order 合同。保存 tube 的安全距离半空间下界均为正；这不是数值失败。论文未明确相对速度符号，当前按参与者源码使用 v_lead-v_ego。

仍需：若要求独立安全定理，仍需 NN 浮点包络、控制注入及 plant 运算的完整包含证明；已有保存盒检查不是该证明。

当前收据：`docs/evidence/results/archcomp26_20261001/acc_fourway_campaign_001/steady05_native/RESULT.json`

## Airplane continuous / P3

状态：no_usable_fullbox_step。P4 验证表在首步前因 9^20 >= 2^63 编码溢出；同阶 P3 的首步 trace 为 x/y/z Picard 提议越出 ±0.01，四次重心化仍 FAILED_CONTRACTION。另扩大 xyz 余项到 ±0.1 后引擎接受，但五坐标 endpoint 超出 tube 1–2 ULP，observer 拒绝，0 个可用保存段。

仍需：需要能在完整初集覆盖上通过数值自包含与观察器一致性的明确方法；之后才可推进到 T=2 并逐段判性质。不能把分盒首步或无效候选界补成全程。

当前收据：`docs/evidence/results/archcomp26_20261001/airplane_p3_xyz_rem0p1_observer_smoke1_001/RESULT.json`；`docs/evidence/results/archcomp26_20261001/airplane_p3_xyz_rem0p1_observer_smoke1_001/payload/RESULT.json`

## Airplane continuous / Huan

状态：no_usable_fullbox_step。order6 建表资源阻断，RSS超过54,006,540 KiB后终止，未进ODE；order3全盒首步 accepted=false、0/10接受。后一拒绝未记录内部失败坐标，不能猜成控制器错误或真实越界。

仍需：需要能在完整初集覆盖上通过数值自包含与观察器一致性的明确方法；之后才可推进到 T=2 并逐段判性质。不能把分盒首步或无效候选界补成全程。

当前收据：`docs/evidence/results/archcomp26_20261001/airplane_continuous_order3_huan_smoke1_001/RESULT.json`；`docs/evidence/results/archcomp26_20261001/airplane_continuous_order3_huan_smoke1_001/payload/RESULT.json`

## Airplane continuous / Xiangru

状态：no_usable_fullbox_step。order3全盒首个 h=0.01 小步 accepted=false，0/10接受；内部失败分量未保存，不能猜成物理性质失败。Huan order6资源记录不能冒称为Xiangru实测。

仍需：需要能在完整初集覆盖上通过数值自包含与观察器一致性的明确方法；之后才可推进到 T=2 并逐段判性质。不能把分盒首步或无效候选界补成全程。

当前收据：`docs/evidence/results/archcomp26_20261001/airplane_continuous_order3_xiangru_smoke1_001/RESULT.json`；`docs/evidence/results/archcomp26_20261001/airplane_continuous_order3_xiangru_smoke1_001/payload/RESULT.json`

## Airplane continuous / Flow* native

状态：no_usable_fullbox_step。order6±0.01、order3±0.01、order3±1 三个全盒 profile 首步均 UNCOMPLETED_SAFE、0段。后续 Real 首拒追踪分别定位 x/y/z/phi/theta 或加宽后的 x/y/phi/theta/psi Picard 不自包含。空 safety 表上的 BOX_SAFE 1 是空集结论。

仍需：需要能在完整初集覆盖上通过数值自包含与观察器一致性的明确方法；之后才可推进到 T=2 并逐段判性质。不能把分盒首步或无效候选界补成全程。

当前收据：`docs/evidence/results/archcomp26_20261001/native_airplane_rem1_first_reject_trace_20261002_006/run/RESULT.json`

## Airplane discrete / P3

状态：contract_blocked。四方法均因执行合同材料不足而未在权威离散合同下运行，非算法已经失败。论文给 Euler 规则、0.1 秒和 20 次转移，但固定官方目录只有连续 dynamics.m，缺参与者实际 NN 采样与状态更新顺序。

仍需：取得参与者离散转移及控制更新源码/等价权威执行记录，再建立四个离散入口并覆盖完整初盒 k=0…20；或用户明确另立 paper-Euler-controller-first 补充合同。

当前收据：合同未执行，无结果收据。

## Airplane discrete / Huan

状态：contract_blocked。四方法均因执行合同材料不足而未在权威离散合同下运行，非算法已经失败。论文给 Euler 规则、0.1 秒和 20 次转移，但固定官方目录只有连续 dynamics.m，缺参与者实际 NN 采样与状态更新顺序。

仍需：取得参与者离散转移及控制更新源码/等价权威执行记录，再建立四个离散入口并覆盖完整初盒 k=0…20；或用户明确另立 paper-Euler-controller-first 补充合同。

当前收据：合同未执行，无结果收据。

## Airplane discrete / Xiangru

状态：contract_blocked。四方法均因执行合同材料不足而未在权威离散合同下运行，非算法已经失败。论文给 Euler 规则、0.1 秒和 20 次转移，但固定官方目录只有连续 dynamics.m，缺参与者实际 NN 采样与状态更新顺序。

仍需：取得参与者离散转移及控制更新源码/等价权威执行记录，再建立四个离散入口并覆盖完整初盒 k=0…20；或用户明确另立 paper-Euler-controller-first 补充合同。

当前收据：合同未执行，无结果收据。

## Airplane discrete / Flow* native

状态：contract_blocked。四方法均因执行合同材料不足而未在权威离散合同下运行，非算法已经失败。论文给 Euler 规则、0.1 秒和 20 次转移，但固定官方目录只有连续 dynamics.m，缺参与者实际 NN 采样与状态更新顺序。

仍需：取得参与者离散转移及控制更新源码/等价权威执行记录，再建立四个离散入口并覆盖完整初盒 k=0…20；或用户明确另立 paper-Euler-controller-first 补充合同。

当前收据：合同未执行，无结果收据。

## Attitude Control avoid / P3

状态：full_numeric_horizon。四方均能完成修正危险盒后的数值合同，六维保存 tube 与闭危险盒不相交。旧 checker 把 x4 危险区间写成空集，其旧 VERIFIED 不作为本轮证据。

仍需：已有六维保存盒判交可支持数值描述；独立 NNCS 证明仍须补浮点 NN、注入和 plant 包含链。单轴危险投影图不能代替六维判交。

当前收据：`research/p3_speed_tightness_20261006/results/attitude_fused1_full60_001/RESULT.json`；`research/p3_speed_tightness_20261006/results/attitude_fused1_full60_001/candidate/RESULT.json`；`research/p3_speed_tightness_20261006/results/attitude_fused1_full60_001/candidate/data/RESULT.json`

## Attitude Control avoid / Huan

状态：full_numeric_horizon。四方均能完成修正危险盒后的数值合同，六维保存 tube 与闭危险盒不相交。旧 checker 把 x4 危险区间写成空集，其旧 VERIFIED 不作为本轮证据。

仍需：已有六维保存盒判交可支持数值描述；独立 NNCS 证明仍须补浮点 NN、注入和 plant 包含链。单轴危险投影图不能代替六维判交。

当前收据：`docs/evidence/results/archcomp26_20261001/attitude_corrected_fourway_campaign_20261002_001/later05_huan/RESULT.json`；`docs/evidence/results/archcomp26_20261001/attitude_corrected_fourway_campaign_20261002_001/later05_huan/payload/RESULT.json`

## Attitude Control avoid / Xiangru

状态：full_numeric_horizon。四方均能完成修正危险盒后的数值合同，六维保存 tube 与闭危险盒不相交。旧 checker 把 x4 危险区间写成空集，其旧 VERIFIED 不作为本轮证据。

仍需：已有六维保存盒判交可支持数值描述；独立 NNCS 证明仍须补浮点 NN、注入和 plant 包含链。单轴危险投影图不能代替六维判交。

当前收据：`docs/evidence/results/archcomp26_20261001/attitude_corrected_fourway_campaign_20261002_001/later05_xiangru/RESULT.json`；`docs/evidence/results/archcomp26_20261001/attitude_corrected_fourway_campaign_20261002_001/later05_xiangru/payload/RESULT.json`

## Attitude Control avoid / Flow* native

状态：full_numeric_horizon。四方均能完成修正危险盒后的数值合同，六维保存 tube 与闭危险盒不相交。旧 checker 把 x4 危险区间写成空集，其旧 VERIFIED 不作为本轮证据。

仍需：已有六维保存盒判交可支持数值描述；独立 NNCS 证明仍须补浮点 NN、注入和 plant 包含链。单轴危险投影图不能代替六维判交。

当前收据：`docs/evidence/results/archcomp26_20261001/attitude_corrected_fourway_campaign_20261002_001/later05_native/RESULT.json`

## Balancing reach / P3

状态：numerically_rejected_prefix。raw4 第 87 小步数值拒绝；仅 86/2000 个有效小步，未到8秒性质窗；不是性质反例。论文 feature5 模型/权威映射另缺。

仍需：论文合同需五输入模型及尺度/顺序，或权威五特征到四输入映射。raw4 需明确的新数值方案通过拒绝并覆盖 T=10；不得把旧小初盒 T=1 或拒绝前缀代填。

当前收据：`docs/evidence/results/archcomp26_20261001/balancing_fixed_raw4_p3_full500_001/RESULT.json`；`docs/evidence/results/archcomp26_20261001/balancing_fixed_raw4_p3_full500_001/payload/RESULT.json`

## Balancing reach / Huan

状态：numerically_rejected_prefix。raw4 第 99 小步数值拒绝；仅 98/2000 个有效小步，未到8秒性质窗；不是性质反例。论文 feature5 模型/权威映射另缺。

仍需：论文合同需五输入模型及尺度/顺序，或权威五特征到四输入映射。raw4 需明确的新数值方案通过拒绝并覆盖 T=10；不得把旧小初盒 T=1 或拒绝前缀代填。

当前收据：`docs/evidence/results/archcomp26_20261001/balancing_fixed_raw4_huan/balancing_fixed_raw4_huan_full500_001/RESULT.json`

## Balancing reach / Xiangru

状态：numerically_rejected_prefix。raw4 第 99 小步数值拒绝；仅 98/2000 个有效小步，未到8秒性质窗；不是性质反例。论文 feature5 模型/权威映射另缺。

仍需：论文合同需五输入模型及尺度/顺序，或权威五特征到四输入映射。raw4 需明确的新数值方案通过拒绝并覆盖 T=10；不得把旧小初盒 T=1 或拒绝前缀代填。

当前收据：`docs/evidence/results/archcomp26_20261001/balancing_fixed_raw4_xiangru_20261002/balancing_fixed_raw4_xiangru_full500_001/RESULT.json`

## Balancing reach / Flow* native

状态：numerically_rejected_prefix。raw4 第 84 小步数值拒绝；仅 83/2000 个有效小步，未到8秒性质窗；不是性质反例。论文 feature5 模型/权威映射另缺。 原生此前解析失败已修正；这里报告修正入口的数值首拒，不继续把旧语法错误当当前阻断。

仍需：论文合同需五输入模型及尺度/顺序，或权威五特征到四输入映射。raw4 需明确的新数值方案通过拒绝并覆盖 T=10；不得把旧小初盒 T=1 或拒绝前缀代填。

当前收据：`docs/evidence/results/archcomp26_20261001/native_balancing_raw4_20261002/native_balancing_raw4_full500_001/RESULT.json`

## Docking constraint / P3

状态：full_numeric_horizon。四方数值时域均完成，未解决的是性质：轴对齐 tube 对 q 的保守上界从首段就跨过 0，因此不能判全时 q<=0。原生外层 failed/exit2 对应 UNKNOWN，不能据此删除其完整 400 段。

仍需：需要更强的相关性/耦合性质判定、明示合法精化或可信反例来解决 UNKNOWN；不能由 q 上界为正断言真实轨迹违规。

当前收据：`research/p3_speed_tightness_20261006/results/docking_fused1_full400_001/RESULT.json`；`research/p3_speed_tightness_20261006/results/docking_fused1_full400_001/candidate/RESULT.json`；`research/p3_speed_tightness_20261006/results/docking_fused1_full400_001/candidate/data/RESULT.json`

## Docking constraint / Huan

状态：full_numeric_horizon。四方数值时域均完成，未解决的是性质：轴对齐 tube 对 q 的保守上界从首段就跨过 0，因此不能判全时 q<=0。原生外层 failed/exit2 对应 UNKNOWN，不能据此删除其完整 400 段。

仍需：需要更强的相关性/耦合性质判定、明示合法精化或可信反例来解决 UNKNOWN；不能由 q 上界为正断言真实轨迹违规。

当前收据：`docs/evidence/results/archcomp26_20261001/docking_huan_full40_001/RESULT.json`

## Docking constraint / Xiangru

状态：full_numeric_horizon。四方数值时域均完成，未解决的是性质：轴对齐 tube 对 q 的保守上界从首段就跨过 0，因此不能判全时 q<=0。原生外层 failed/exit2 对应 UNKNOWN，不能据此删除其完整 400 段。

仍需：需要更强的相关性/耦合性质判定、明示合法精化或可信反例来解决 UNKNOWN；不能由 q 上界为正断言真实轨迹违规。

当前收据：`docs/evidence/results/archcomp26_20261001/docking_xiangru_full40_001/RESULT.json`

## Docking constraint / Flow* native

状态：full_numeric_horizon。四方数值时域均完成，未解决的是性质：轴对齐 tube 对 q 的保守上界从首段就跨过 0，因此不能判全时 q<=0。原生外层 failed/exit2 对应 UNKNOWN，不能据此删除其完整 400 段。

仍需：需要更强的相关性/耦合性质判定、明示合法精化或可信反例来解决 UNKNOWN；不能由 q 上界为正断言真实轨迹违规。

当前收据：`docs/evidence/results/archcomp26_20261001/native_docking_full40_001/RESULT.json`

## Double Pendulum less-robust / P3

状态：full_numeric_horizon。四方能完成 225 盒×100 步。P3 主选是 affine-split4 控制残差入口；旧失败候选另存。保存 tube 都在安全盒内。

仍需：Huan/Xiangru 少量 endpoint 末位超出同段 tube，须继续区分两种界并保留原值；独立浮点 NNCS 包含链仍未闭合，不能凭窄界确认正确性。

当前收据：`docs/evidence/results/archcomp26_20261001/dp_p3_affine_split4_v1/full225_smoke_001/attempt/RESULT.json`；`docs/evidence/results/archcomp26_20261001/dp_p3_affine_split4_v1/full225_smoke_001/attempt/data/RESULT.json`

## Double Pendulum less-robust / Huan

状态：full_numeric_horizon。四方能完成 225 盒×100 步。P3 主选是 affine-split4 控制残差入口；旧失败候选另存。保存 tube 都在安全盒内。

仍需：Huan/Xiangru 少量 endpoint 末位超出同段 tube，须继续区分两种界并保留原值；独立浮点 NNCS 包含链仍未闭合，不能凭窄界确认正确性。

当前收据：`docs/evidence/results/archcomp26_20261001/author_dp_less_v1/huan_full20_001/RESULT.json`；`docs/evidence/results/archcomp26_20261001/author_dp_less_v1/huan_full20_001/payload/RESULT.json`

## Double Pendulum less-robust / Xiangru

状态：full_numeric_horizon。四方能完成 225 盒×100 步。P3 主选是 affine-split4 控制残差入口；旧失败候选另存。保存 tube 都在安全盒内。

仍需：Huan/Xiangru 少量 endpoint 末位超出同段 tube，须继续区分两种界并保留原值；独立浮点 NNCS 包含链仍未闭合，不能凭窄界确认正确性。

当前收据：`docs/evidence/results/archcomp26_20261001/author_dp_less_v1/xiangru_full20_001/RESULT.json`；`docs/evidence/results/archcomp26_20261001/author_dp_less_v1/xiangru_full20_001/payload/RESULT.json`

## Double Pendulum less-robust / Flow* native

状态：full_numeric_horizon。四方能完成 225 盒×100 步。P3 主选是 affine-split4 控制残差入口；旧失败候选另存。保存 tube 都在安全盒内。

仍需：Huan/Xiangru 少量 endpoint 末位超出同段 tube，须继续区分两种界并保留原值；独立浮点 NNCS 包含链仍未闭合，不能凭窄界确认正确性。

当前收据：`docs/evidence/results/archcomp26_20261001/native_dp_less_full20_001/RESULT.json`

## Double Pendulum more-robust / P3

状态：property_stopped_numeric_prefix。全部225盒在已保存 72/80 步数值接受，作者 Unsafe. 后停止；保存安全前缀仅60步。第61步开始跨带；第72步发生作者 Unsafe.，不是数值拒绝。 P3外层也completed/exit0而payload为incomplete；旧interval残差入口的第9步FAILED_CONTRACTION另存。

仍需：需要独立可核的反例或更强性质分析以解释早停；若另做不因性质停止的数值诊断须单独命名，不追认现有前缀为完整结果。旧 P3 第9步数值拒绝与新主选应分开。 当前P3未保存单独 lane→initial-subbox ledger；配置分区/225条观察与pooled首步覆盖不等于独立重建该映射。

当前收据：`docs/evidence/results/archcomp26_20261001/dp_more_p3_affine_split4_full20_20261003_001/attempt/RESULT.json`；`docs/evidence/results/archcomp26_20261001/dp_more_p3_affine_split4_full20_20261003_001/attempt/data/RESULT.json`

## Double Pendulum more-robust / Huan

状态：property_stopped_numeric_prefix。全部225盒在已保存 72/80 步数值接受，作者 Unsafe. 后停止；保存安全前缀仅60步。第61步开始跨带；第72步发生作者 Unsafe.，不是数值拒绝。

仍需：需要独立可核的反例或更强性质分析以解释早停；若另做不因性质停止的数值诊断须单独命名，不追认现有前缀为完整结果。旧 P3 第9步数值拒绝与新主选应分开。

当前收据：`docs/evidence/results/archcomp26_20261001/author_dp_more_v1/huan_full20_001/RESULT.json`；`docs/evidence/results/archcomp26_20261001/author_dp_more_v1/huan_full20_001/payload/RESULT.json`

## Double Pendulum more-robust / Xiangru

状态：property_stopped_numeric_prefix。全部225盒在已保存 72/80 步数值接受，作者 Unsafe. 后停止；保存安全前缀仅60步。第61步开始跨带；第72步发生作者 Unsafe.，不是数值拒绝。

仍需：需要独立可核的反例或更强性质分析以解释早停；若另做不因性质停止的数值诊断须单独命名，不追认现有前缀为完整结果。旧 P3 第9步数值拒绝与新主选应分开。

当前收据：`docs/evidence/results/archcomp26_20261001/author_dp_more_v1/xiangru_full20_001/RESULT.json`；`docs/evidence/results/archcomp26_20261001/author_dp_more_v1/xiangru_full20_001/payload/RESULT.json`

## Double Pendulum more-robust / Flow* native

状态：property_stopped_numeric_prefix。全部225盒在已保存 64/80 步数值接受，作者 UNKNOWN 后停止；保存安全前缀仅60步。原生外层 completed/exit0 只表示进程结束。

仍需：需要独立可核的反例或更强性质分析以解释早停；若另做不因性质停止的数值诊断须单独命名，不追认现有前缀为完整结果。旧 P3 第9步数值拒绝与新主选应分开。

当前收据：`docs/evidence/results/archcomp26_20261001/native_dp_more_full20_001/RESULT.json`

## NAV standard / P3

状态：full_numeric_horizon。新 P3/原生和经合同审计的历史 Huan/Xiangru 均有完整 640×600 盒步；不是未完成实例。固定官方 point 模型有同合同记录，另一作者 Git LFS 仓库的模型身份尚未建立。

仍需：如需声称复现另一 Git LFS 提交，需其可读取模型及明确映射；当前官方模型具名结果不因此作废。新旧混合时间不可拼为同资源 campaign，独立 NNCS 证明另缺。

当前收据：`research/p3_speed_tightness_20261006/results/nav_standard_fused512_full600_001/RESULT.json`；`research/p3_speed_tightness_20261006/results/nav_standard_fused512_full600_001/candidate/RESULT.json`；`research/p3_speed_tightness_20261006/results/nav_standard_fused512_full600_001/candidate/data/RESULT.json`

## NAV standard / Huan

状态：full_numeric_horizon。新 P3/原生和经合同审计的历史 Huan/Xiangru 均有完整 640×600 盒步；不是未完成实例。固定官方 point 模型有同合同记录，另一作者 Git LFS 仓库的模型身份尚未建立。

仍需：如需声称复现另一 Git LFS 提交，需其可读取模型及明确映射；当前官方模型具名结果不因此作废。新旧混合时间不可拼为同资源 campaign，独立 NNCS 证明另缺。

当前收据：`/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v1/suite_v1/nav_standard_huan/result.json`

## NAV standard / Xiangru

状态：full_numeric_horizon。新 P3/原生和经合同审计的历史 Huan/Xiangru 均有完整 640×600 盒步；不是未完成实例。固定官方 point 模型有同合同记录，另一作者 Git LFS 仓库的模型身份尚未建立。

仍需：如需声称复现另一 Git LFS 提交，需其可读取模型及明确映射；当前官方模型具名结果不因此作废。新旧混合时间不可拼为同资源 campaign，独立 NNCS 证明另缺。

当前收据：`/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v1/suite_v1/nav_standard_xiangru/result.json`

## NAV standard / Flow* native

状态：full_numeric_horizon。新 P3/原生和经合同审计的历史 Huan/Xiangru 均有完整 640×600 盒步；不是未完成实例。固定官方 point 模型有同合同记录，另一作者 Git LFS 仓库的模型身份尚未建立。

仍需：如需声称复现另一 Git LFS 提交，需其可读取模型及明确映射；当前官方模型具名结果不因此作废。新旧混合时间不可拼为同资源 campaign，独立 NNCS 证明另缺。

当前收据：`docs/evidence/results/archcomp26_20261001/nav_author_standard_native_full30_001/RESULT.json`

## NAV robust / P3

状态：full_numeric_horizon。新 P3 和经合同审计的历史 Huan/Xiangru/原生均有完整 25×600 盒步。robust 是控制器训练方式，不向当前 plant 擅加训练噪声。

仍需：另一 Git LFS 模型身份仍需外部材料；历史结果保留历史身份。四态保存统计已经补齐，不能再写成四态宽度缺失；独立 NNCS 证明及同资源重复比较另缺。

当前收据：`research/p3_speed_tightness_20261006/results/nav_robust_fused32_full600_001/RESULT.json`；`research/p3_speed_tightness_20261006/results/nav_robust_fused32_full600_001/candidate/RESULT.json`；`research/p3_speed_tightness_20261006/results/nav_robust_fused32_full600_001/candidate/data/RESULT.json`

## NAV robust / Huan

状态：full_numeric_horizon。新 P3 和经合同审计的历史 Huan/Xiangru/原生均有完整 25×600 盒步。robust 是控制器训练方式，不向当前 plant 擅加训练噪声。

仍需：另一 Git LFS 模型身份仍需外部材料；历史结果保留历史身份。四态保存统计已经补齐，不能再写成四态宽度缺失；独立 NNCS 证明及同资源重复比较另缺。

当前收据：`/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/timing_v1/nav_robust_r2_huan/result.json`

## NAV robust / Xiangru

状态：full_numeric_horizon。新 P3 和经合同审计的历史 Huan/Xiangru/原生均有完整 25×600 盒步。robust 是控制器训练方式，不向当前 plant 擅加训练噪声。

仍需：另一 Git LFS 模型身份仍需外部材料；历史结果保留历史身份。四态保存统计已经补齐，不能再写成四态宽度缺失；独立 NNCS 证明及同资源重复比较另缺。

当前收据：`/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v1/suite_v1/nav_robust_xiangru/result.json`

## NAV robust / Flow* native

状态：full_numeric_horizon。新 P3 和经合同审计的历史 Huan/Xiangru/原生均有完整 25×600 盒步。robust 是控制器训练方式，不向当前 plant 擅加训练噪声。

仍需：另一 Git LFS 模型身份仍需外部材料；历史结果保留历史身份。四态保存统计已经补齐，不能再写成四态宽度缺失；独立 NNCS 证明及同资源重复比较另缺。

当前收据：`docs/evidence/results/archcomp26_20261001/nav_robust_native_historical_20260923/result.json`

## QUAD reach / P3

状态：full_numeric_horizon。四方均完成论文 ODE 的 1024×1000 盒步；T=5 x3 并集均入 [0.94,1.06]。尚未完成的是权威 reach-and-remain 全时间窗语义/检查、独立 NNCS 证明及 native octagon 生产资格。

仍需：取得参与者 reach-and-remain checker 或等价权威记录；Huan/Xiangru 若要画全时曲线需新的逐步坐标记录。原生须在控制构造修正后重新建立 plant、多期与全时包含资格，不能由条件性首批构造门外推。 当前论文P3每步逐盒计算tube/endpoint后保存pooled投影、accepted_count及状态；没有逐盒范围档案，不能从pooled并集恢复逐盒mean/max或声称全逐盒等值。

当前收据：`research/p3_speed_tightness_20261006/results/quad_paper_joint_full1000_001/RESULT.json`；`research/p3_speed_tightness_20261006/results/quad_paper_joint_full1000_001/candidate/RESULT.json`；`research/p3_speed_tightness_20261006/results/quad_paper_joint_full1000_001/candidate/data/RESULT.json`

## QUAD reach / Huan

状态：full_numeric_horizon。四方均完成论文 ODE 的 1024×1000 盒步；T=5 x3 并集均入 [0.94,1.06]。尚未完成的是权威 reach-and-remain 全时间窗语义/检查、独立 NNCS 证明及 native octagon 生产资格。

仍需：取得参与者 reach-and-remain checker 或等价权威记录；Huan/Xiangru 若要画全时曲线需新的逐步坐标记录。原生须在控制构造修正后重新建立 plant、多期与全时包含资格，不能由条件性首批构造门外推。

当前收据：`docs/evidence/results/archcomp26_20261001/quad_paper_huan_full50_001/supervisor/RESULT.json`

## QUAD reach / Xiangru

状态：full_numeric_horizon。四方均完成论文 ODE 的 1024×1000 盒步；T=5 x3 并集均入 [0.94,1.06]。尚未完成的是权威 reach-and-remain 全时间窗语义/检查、独立 NNCS 证明及 native octagon 生产资格。

仍需：取得参与者 reach-and-remain checker 或等价权威记录；Huan/Xiangru 若要画全时曲线需新的逐步坐标记录。原生须在控制构造修正后重新建立 plant、多期与全时包含资格，不能由条件性首批构造门外推。

当前收据：`docs/evidence/results/archcomp26_20261001/quad_paper_xiangru_v1/full50_001/RESULT.json`；`docs/evidence/results/archcomp26_20261001/quad_paper_xiangru_v1/full50_001/data/RESULT.json`

## QUAD reach / Flow* native

状态：full_numeric_horizon。四方均完成论文 ODE 的 1024×1000 盒步；T=5 x3 并集均入 [0.94,1.06]。尚未完成的是权威 reach-and-remain 全时间窗语义/检查、独立 NNCS 证明及 native octagon 生产资格。

仍需：取得参与者 reach-and-remain checker 或等价权威记录；Huan/Xiangru 若要画全时曲线需新的逐步坐标记录。原生须在控制构造修正后重新建立 plant、多期与全时包含资格，不能由条件性首批构造门外推。

当前收据：`docs/evidence/results/archcomp26_20261001/native_quad_paper_full50_001/RESULT.json`

## Single Pendulum reach / P3

状态：full_numeric_horizon。具名两物理态合同四方均完成且保存性质窗安全。它不能冒名为官方 MATLAB 三返回量执行：dynamics_sp.m 另有 dx3=1，却未给第三态初值、重置和该 MATLAB 闭环/checker 入口。

仍需：若要复现官方三返回量 MATLAB 提交，需第三态初始化/重置语义及实际闭环检查源码；不得擅自把该态当当前辅助时钟。具名两态数值结论可保留，独立证明另缺。

当前收据：`research/p3_speed_tightness_20261006/results/sp_two_state_fused1_full100_001/RESULT.json`；`research/p3_speed_tightness_20261006/results/sp_two_state_fused1_full100_001/candidate/RESULT.json`；`research/p3_speed_tightness_20261006/results/sp_two_state_fused1_full100_001/candidate/data/RESULT.json`

## Single Pendulum reach / Huan

状态：full_numeric_horizon。具名两物理态合同四方均完成且保存性质窗安全。它不能冒名为官方 MATLAB 三返回量执行：dynamics_sp.m 另有 dx3=1，却未给第三态初值、重置和该 MATLAB 闭环/checker 入口。

仍需：若要复现官方三返回量 MATLAB 提交，需第三态初始化/重置语义及实际闭环检查源码；不得擅自把该态当当前辅助时钟。具名两态数值结论可保留，独立证明另缺。

当前收据：`docs/evidence/results/archcomp26_20261001/sp_two_state_fourway_campaign_20261002_001/later05_huan/RESULT.json`；`docs/evidence/results/archcomp26_20261001/sp_two_state_fourway_campaign_20261002_001/later05_huan/data/RESULT.json`

## Single Pendulum reach / Xiangru

状态：full_numeric_horizon。具名两物理态合同四方均完成且保存性质窗安全。它不能冒名为官方 MATLAB 三返回量执行：dynamics_sp.m 另有 dx3=1，却未给第三态初值、重置和该 MATLAB 闭环/checker 入口。

仍需：若要复现官方三返回量 MATLAB 提交，需第三态初始化/重置语义及实际闭环检查源码；不得擅自把该态当当前辅助时钟。具名两态数值结论可保留，独立证明另缺。

当前收据：`docs/evidence/results/archcomp26_20261001/sp_two_state_fourway_campaign_20261002_001/later05_xiangru/RESULT.json`；`docs/evidence/results/archcomp26_20261001/sp_two_state_fourway_campaign_20261002_001/later05_xiangru/data/RESULT.json`

## Single Pendulum reach / Flow* native

状态：full_numeric_horizon。具名两物理态合同四方均完成且保存性质窗安全。它不能冒名为官方 MATLAB 三返回量执行：dynamics_sp.m 另有 dx3=1，却未给第三态初值、重置和该 MATLAB 闭环/checker 入口。

仍需：若要复现官方三返回量 MATLAB 提交，需第三态初始化/重置语义及实际闭环检查源码；不得擅自把该态当当前辅助时钟。具名两态数值结论可保留，独立证明另缺。

当前收据：`docs/evidence/results/archcomp26_20261001/sp_two_state_fourway_campaign_20261002_001/later05_native/RESULT.json`

## TORA remain / P3

状态：full_numeric_horizon。固定 h=0.1 全部12盒×200步完整，保存 tube 全在安全盒内；不是未完成方法。

仍需：固定 h=0.1 两作者需明确通过自包含的新方法/参数资格；原源码没有可直接启用的自映射重试开关。另列 h=0.05 已四方完整，但不能暗换冻结 h=0.1 主格。

当前收据：`docs/evidence/results/archcomp26_20261001/p3_tora_remain_v1/full20_001/RESULT.json`；`docs/evidence/results/archcomp26_20261001/p3_tora_remain_v1/full20_001/payload/RESULT.json`

## TORA remain / Huan

状态：partial_lanes_after_rejection。固定h=0.1第190步首次拒绝初盒；全盒接受前缀189步、安全前缀184步。仍观察200步，但总计2357/2400盒步接受、末步仅6/12，不能给全初集T=20终点。Huan首拒trace是盒2的x2 Picard提议越出±0.01；Xiangru未独立记录该内部坐标，不能冒称已逐内部trace验证。

仍需：固定 h=0.1 两作者需明确通过自包含的新方法/参数资格；原源码没有可直接启用的自映射重试开关。另列 h=0.05 已四方完整，但不能暗换冻结 h=0.1 主格。

当前收据：`docs/evidence/results/archcomp26_20261001/author_tora_remain_v1/huan_full20_001/RESULT.json`；`docs/evidence/results/archcomp26_20261001/author_tora_remain_v1/huan_full20_001/payload/RESULT.json`

## TORA remain / Xiangru

状态：partial_lanes_after_rejection。固定h=0.1第190步首次拒绝初盒；全盒接受前缀189步、安全前缀184步。仍观察200步，但总计2357/2400盒步接受、末步仅6/12，不能给全初集T=20终点。Huan首拒trace是盒2的x2 Picard提议越出±0.01；Xiangru未独立记录该内部坐标，不能冒称已逐内部trace验证。

仍需：固定 h=0.1 两作者需明确通过自包含的新方法/参数资格；原源码没有可直接启用的自映射重试开关。另列 h=0.05 已四方完整，但不能暗换冻结 h=0.1 主格。

当前收据：`docs/evidence/results/archcomp26_20261001/author_tora_remain_v1/xiangru_full20_001/RESULT.json`；`docs/evidence/results/archcomp26_20261001/author_tora_remain_v1/xiangru_full20_001/payload/RESULT.json`

## TORA remain / Flow* native

状态：full_numeric_horizon。固定 h=0.1 全部12盒×200步完整，保存 tube 全在安全盒内；不是未完成方法。

仍需：固定 h=0.1 两作者需明确通过自包含的新方法/参数资格；原源码没有可直接启用的自映射重试开关。另列 h=0.05 已四方完整，但不能暗换冻结 h=0.1 主格。

当前收据：`docs/evidence/results/archcomp26_20261001/native_tora_remain_full20_001/RESULT.json`

## TORA reach-sigmoid / P3

状态：full_numeric_horizon。四方均完成官方 sigmoid、u=11f 的 500 步，保存 T=5 目标坐标入目标。三 GPU 明确没有执行性质 checker；不能给它们补写作者 VERIFIED。

仍需：若需作者性质标签，需与选定到达语义一致的 checker；终点包含是五秒内到达的充分数值观察。原生终点 VERIFIED 仍不是独立端到端证明。

当前收据：`research/p3_speed_tightness_20261006/results/tora_sigmoid_cutoff1e8_full500_001/RESULT.json`；`research/p3_speed_tightness_20261006/results/tora_sigmoid_cutoff1e8_full500_001/candidate/RESULT.json`；`research/p3_speed_tightness_20261006/results/tora_sigmoid_cutoff1e8_full500_001/candidate/data/RESULT.json`

## TORA reach-sigmoid / Huan

状态：full_numeric_horizon。四方均完成官方 sigmoid、u=11f 的 500 步，保存 T=5 目标坐标入目标。三 GPU 明确没有执行性质 checker；不能给它们补写作者 VERIFIED。

仍需：若需作者性质标签，需与选定到达语义一致的 checker；终点包含是五秒内到达的充分数值观察。原生终点 VERIFIED 仍不是独立端到端证明。

当前收据：`docs/evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/later05_huan/outer/RESULT.json`；`docs/evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/later05_huan/payload/RESULT.json`

## TORA reach-sigmoid / Xiangru

状态：full_numeric_horizon。四方均完成官方 sigmoid、u=11f 的 500 步，保存 T=5 目标坐标入目标。三 GPU 明确没有执行性质 checker；不能给它们补写作者 VERIFIED。

仍需：若需作者性质标签，需与选定到达语义一致的 checker；终点包含是五秒内到达的充分数值观察。原生终点 VERIFIED 仍不是独立端到端证明。

当前收据：`docs/evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/later05_xiangru/outer/RESULT.json`；`docs/evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/later05_xiangru/payload/RESULT.json`

## TORA reach-sigmoid / Flow* native

状态：full_numeric_horizon。四方均完成官方 sigmoid、u=11f 的 500 步，保存 T=5 目标坐标入目标。三 GPU 明确没有执行性质 checker；不能给它们补写作者 VERIFIED。

仍需：若需作者性质标签，需与选定到达语义一致的 checker；终点包含是五秒内到达的充分数值观察。原生终点 VERIFIED 仍不是独立端到端证明。

当前收据：`docs/evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/later05_native/RESULT.json`

## TORA reach-tanh / P3

状态：full_numeric_horizon。新 working P3 与三历史作者方法均完成同合同500步，目标终点入带。旧 engine_linear_leaf_v2 我方记录另列，不能当最新 P3。

仍需：当前 P3 与历史 GPU 无作者性质 verdict；需按选定到达语义补相应判定才可声称作者检查通过。历史原生仅终点 VERIFIED；混代不作稳定速度排名。

当前收据：`research/p3_speed_tightness_20261006/results/tora_tanh_cutoff1e8_full500_001/RESULT.json`；`research/p3_speed_tightness_20261006/results/tora_tanh_cutoff1e8_full500_001/candidate/RESULT.json`；`research/p3_speed_tightness_20261006/results/tora_tanh_cutoff1e8_full500_001/candidate/data/RESULT.json`

## TORA reach-tanh / Huan

状态：full_numeric_horizon。新 working P3 与三历史作者方法均完成同合同500步，目标终点入带。旧 engine_linear_leaf_v2 我方记录另列，不能当最新 P3。

仍需：当前 P3 与历史 GPU 无作者性质 verdict；需按选定到达语义补相应判定才可声称作者检查通过。历史原生仅终点 VERIFIED；混代不作稳定速度排名。

当前收据：`/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v1/suite_v1/tora_relu_tanh_huan/result.json`

## TORA reach-tanh / Xiangru

状态：full_numeric_horizon。新 working P3 与三历史作者方法均完成同合同500步，目标终点入带。旧 engine_linear_leaf_v2 我方记录另列，不能当最新 P3。

仍需：当前 P3 与历史 GPU 无作者性质 verdict；需按选定到达语义补相应判定才可声称作者检查通过。历史原生仅终点 VERIFIED；混代不作稳定速度排名。

当前收据：`/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v1/suite_v1/tora_relu_tanh_xiangru/result.json`

## TORA reach-tanh / Flow* native

状态：full_numeric_horizon。新 working P3 与三历史作者方法均完成同合同500步，目标终点入带。旧 engine_linear_leaf_v2 我方记录另列，不能当最新 P3。

仍需：当前 P3 与历史 GPU 无作者性质 verdict；需按选定到达语义补相应判定才可声称作者检查通过。历史原生仅终点 VERIFIED；混代不作稳定速度排名。

当前收据：`/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/native_matched/tora_relu_tanh/result.json`

## Unicycle reach / P3

状态：full_numeric_horizon。四方均完成用户选定的论文速度导数加常值扰动合同。P3/原生保存终点入目标；Huan/Xiangru 的保存 endpoint 不能整体落入目标，性质仍 UNKNOWN。

仍需：H/X 需更强到达性分析或可信反例；500个保存 endpoint 都未全入目标不能推出连续窗口不可达。常值 w 是具名选择，不覆盖论文未明确的任意时变扰动。

当前收据：`research/p3_speed_tightness_20261005/results/unicycle_p3_fused1_20261005_001/run_001/RESULT.json`；`research/p3_speed_tightness_20261005/results/unicycle_p3_fused1_20261005_001/run_001/candidate/RESULT.json`；`research/p3_speed_tightness_20261005/results/unicycle_p3_fused1_20261005_001/run_001/candidate/data/RESULT.json`

## Unicycle reach / Huan

状态：full_numeric_horizon。四方均完成用户选定的论文速度导数加常值扰动合同。P3/原生保存终点入目标；Huan/Xiangru 的保存 endpoint 不能整体落入目标，性质仍 UNKNOWN。

仍需：H/X 需更强到达性分析或可信反例；500个保存 endpoint 都未全入目标不能推出连续窗口不可达。常值 w 是具名选择，不覆盖论文未明确的任意时变扰动。

当前收据：`docs/evidence/results/archcomp26_20261001/unicycle_paper_speed_w_constant_v1/huan_full50_001/RESULT.json`；`docs/evidence/results/archcomp26_20261001/unicycle_paper_speed_w_constant_v1/huan_full50_001/payload/RESULT.json`

## Unicycle reach / Xiangru

状态：full_numeric_horizon。四方均完成用户选定的论文速度导数加常值扰动合同。P3/原生保存终点入目标；Huan/Xiangru 的保存 endpoint 不能整体落入目标，性质仍 UNKNOWN。

仍需：H/X 需更强到达性分析或可信反例；500个保存 endpoint 都未全入目标不能推出连续窗口不可达。常值 w 是具名选择，不覆盖论文未明确的任意时变扰动。

当前收据：`docs/evidence/results/archcomp26_20261001/unicycle_paper_speed_w_constant_v1/xiangru_full50_001/RESULT.json`；`docs/evidence/results/archcomp26_20261001/unicycle_paper_speed_w_constant_v1/xiangru_full50_001/payload/RESULT.json`

## Unicycle reach / Flow* native

状态：full_numeric_horizon。四方均完成用户选定的论文速度导数加常值扰动合同。P3/原生保存终点入目标；Huan/Xiangru 的保存 endpoint 不能整体落入目标，性质仍 UNKNOWN。

仍需：H/X 需更强到达性分析或可信反例；500个保存 endpoint 都未全入目标不能推出连续窗口不可达。常值 w 是具名选择，不覆盖论文未明确的任意时变扰动。

当前收据：`docs/evidence/results/archcomp26_20261001/native_unicycle_paper_speed_full50_001/RESULT.json`
