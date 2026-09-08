# 有序取范围的数值与生命周期合同

科学提交 `5f37cbe0427c480ef0ebbbcaba292143bc8f4ede` 基于
`f6af6f67565a954d0c68f88a50cc9c181a2d0b08`。正式两种模式都显式开启
`prepared_remainder_replay(True)`；新的 `packed_boundary_execution` 使用独立
ContextVar，公共默认值为 False，退出上下文恢复先前值。

生产接入点只有 `Polynomial.evaluate_interval`、`evaluate_interval_normal` 和
`accepted_boundary_sr._interval_polynomial_range`。这是一条统一的取范围执行链。
接受标量 CPU binary64 稀疏输入；其他 dtype/device/shape 回退原 evaluator。
normal 调用带外部 step_exp_table 时也保留原路径。所有原 range policy、分区、
cutoff、端点误差及收紧设置不变。

normal 入口沿用原坐标约定：指定的状态变量视为 `[-1, 1]` 归一化坐标，
存储 domain 中对应的状态标签不参与状态因子计算；时间和其他变量使用实际
domain（或回退路径中的原 step_exp_table）。一般非对称物理范围使用普通入口。
局部 normal 对照中的非单位存储标签用于检查这条既有规则，没有声称它包围
标签所描述的任意非归一化物理区域。

| 对象/操作 | 合同 |
|---|---|
| 系数 | 原 binary64 值，私有 `[B, output, terms]` 上下界；不改支持集或零项 |
| 支持顺序 | 普通/normal 保留 Polynomial 的项顺序；区间系数入口保留原 sorted 顺序 |
| domain | 原 `[B, variables]` 上下界；每个 batch lane 独立 |
| 变量幂 | 每个实际 `(variable, power, batch)` 使用原标量 Interval.pow_int；只在一次 evaluate 内复用 |
| 项乘法 | 保留变量阶段先后次序，四个乘积、原 min/max、nextafter 及有效性检查 |
| normal | 原时间因子先行，原奇偶状态因子随后，其他变量按原序执行 |
| 求和 | 从原零区间开始，按原项序逐项加，每次 nextafter 与检查，不用 parallel sum/cumsum |
| 真零/微量/非有限 | 不加零项裁剪阈值；NaN/无效区间仍拒绝，Inf/溢出与原路径一致；subnormal 单列测试 |
| 结构缓存 | 有界 LRU128，键为不可变支持、变量数、normal 变量/时间索引；值仅索引、幂次、顺序元数据 |
| 数值缓存 | 不跨 evaluate 缓存系数、domain、h、余项、区间总值或变量幂；无 Python id 数值缓存 |
| 所有权 | 输入不原地写入，项上下界为私有 clone；返回结果不共享输入可变存储 |
| 状态 | accepted-only commit、失败回滚、generation/reset、队列容量和 checkpoint/resume 原样保留 |
| 误差 | 普通余项、已传播历史、本步新增 E 和 cutoff 支付归属不变，不重复追加误差 |

独立项和 batch/output 维度一起执行张量乘法；相互依赖的逐项累加继续保持原顺序。
局部接口支持 B2，测试使用不同系数和 domain，检查无混合；完整 solver 仍只要求 B1，
不据此声称完整批量加速或 GPU 吞吐。

独立精确有理数 oracle 验证每项和最终区间的包含性，同时检查新旧逐位一致。
41 项局部测试还覆盖 order4/6、不同输出、非对称范围、严重相消、空项、非有限值、
回退、共享存储和上下文隔离。代表真实状态通过安全 JSON checkpoint 加载；
两条执行路径分别连续推进，直接比对完整 dataclass、张量和诊断对象，仅排除
`host_to_device_s`、`dense_kernel_s`、`device_to_host_s` 三项耗时计数。
随后分别检查失败尝试不污染已有历史、重复测量不改结果，以及恢复后实际继续一步。

验证范围是上述操作、局部输入和记录运行；不是整个求解器的形式化证明。

计时范围另行冻结：正式 solve 计时包含计划构造、打包、必要复制、安全检查和拒绝尝试；
初始化、外部 JSON/范围测量/检查点导出分别计时。profile 的 Interval/Tensor/copy
计数使用独立遍历；创建 Interval 的成本嵌在 A–H 内，不另加到总秒数。
Tensor 原始计数是 ATen 返回 Tensor 的次数；派生数扣除 schema 可确认的原地自身返回，
包含 view，不能当成 storage 分配字节数。
