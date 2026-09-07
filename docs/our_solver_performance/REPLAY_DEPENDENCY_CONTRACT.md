# 参考安全前提失败，单步准备计划未实现

本文件保留待检验的依赖边界，不是已完成的 prepared-plan 数据流证明。目标第 2.2 节的端点包含性失败先于第 4 节性能授权，因此没有新缓存、接口、执行模式、数学修复或布局变更。

| 工作 | 可能复用的前提 | 每轮必须变化或保留的内容 | 本轮结论 |
|---|---|---|---|
| 基底、乘法/积分索引及表达式结构 | 同一固定候选、阶数与表达式，实际索引依赖均相同 | 下一步、重试或合同变化必须重新确定有效期 | 未测，不增加新缓存 |
| 候选多项式的乘积、截断与范围 | 实际数据流完全不依赖当前余项、混合状态或动态支持决策 | 静态误差的数值即使复用，每次输出仍应记账一次 | 未证明，不能凭名字判静态 |
| 当前余项及余项交叉项 | 无此静态前提 | 当前余项、依赖余项的范围、proposal 全部重算 | 原逻辑保留 |
| 包含与停止判断 | 无此静态前提 | 每次逐操作向外舍入、subset、STOP_RATIO、491 次上限、整向量提交 | 原逻辑保留 |
| 最后提交误差与跨步历史 | 依赖最后真实提交和队列状态 | 误差分解、owner、SR reset、失败回滚及 checkpoint | 原逻辑保留 |

如果后续重新开启性能任务，第一版生命周期仍须限定单步/单次固定候选尝试，冻结不可变数值数组并分离动态缓冲。ODE、order、basis、步长与时间表、domain、range policy、cutoff、validation epsilon、候选系数和实际本步依赖需要在创建和使用边界核验，不能用可变 tensor 的 `id()` 代替内容安全，也不能每轮全量哈希吞掉收益。这些约束本轮没有通过实现测试，不能标为完成。

没有新 profile，因而 f、s、`1/((1-f)+f/s)` 及 `1/(1-f)` 数值上限均为空。没有据此宣称机制值得或不值得做；正确分类是参考缺陷停止，而不是无优化授权的热点定位结论。

VDP 专用缓存和严格结构保护保持原样，未把专用公式用于 Brusselator。默认路径、CPU float64、表达式顺序、range policy、舍入位置、cutoff、历史容量和全部提交政策没有变化。

实际端点反例的输入和完整返回表示见 `raw_minimal/endpoint_reference.json`。精确输入采用 `Fraction(binary64(0.01))`，不采用 `Fraction(1,100)`。两条入口的普通余项都不足以覆盖代入失去的系数误差；本轮没有将其补记、更换求和顺序或调整边界。

源码入口附录：

- 发布：`src/torch_tm_flowpipe/flowpipe.py:6090` 的 `segment_tm.substitute_const(...).drop_variable(...).apply_cutoff(...)`。
- 代入：`src/torch_tm_flowpipe/taylor_model.py:319`、`src/torch_tm_flowpipe/polynomial.py:394`。
- dense 内部：`src/torch_tm_flowpipe/batched_dense_tm.py:2610`，由接受分支第 4941 行调用。
- 待测性能假设仍指向 `_post_accept_refine_raw_remainder` 与 `_dense_flowstar_raw_compat_image`，本轮没有修改或深入实施。
