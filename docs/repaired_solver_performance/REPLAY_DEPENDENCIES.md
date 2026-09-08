本轮只复用同一接受 attempt 内固定的多项式计算。默认开关关闭；通过
`prepared_remainder_replay(True)` 上下文启用。上下文只保存布尔开关，数值计划
只存在于 `_post_accept_refine_raw_remainder` 的局部变量中。VDP 的已有 canonical
closure 缓存继续原路径。本实现没有把 Brusselator 方程写进求值器。

| 量 | 实际依赖 | 创建、读取、失效 |
|---|---|---|
| RHS 操作序列 | immutable PolynomialODE 的有序项，或纯算术函数的源语法 | 第一次实际 replay 记录原运算顺序；随后执行同一序列；退出 attempt 丢弃 |
| 输入点多项式、base、domain、basis | 本步候选、归一化输入、阶数、h、尺度 | 接受后独立深拷贝；调用方原地修改不能改计划；绑定身份和 tensor version 检查拒用变化输入 |
| 中间点多项式及乘法截断界 | 固定左右点系数、basis、domain、degree、range policy | 第一次实际运算调用原 BatchedPolynomial 方法；以后读取私有结果；没有重排 scatter/add/mul |
| cutoff mask 与迁移界 | **实际固定系数**、threshold、domain、range policy | 在本步实际系数上首次计算；不是只依据 basis 缓存 mask |
| polynomial-only range | 对应固定点多项式、domain、policy、context | 原范围方法首次执行；不同 context 分别缓存；动态 remainder 不在这些输入中 |
| regular RHS 的积分多项式、overflow、poly_diff range | 上述固定点系数、积分变量、base、cutoff | 同一计划内第一次计算；其伴随普通余项和 ledger 不缓存 |
| raw 与 regular RHS 普通余项、全部 ledger 类别 | 本轮 R、固定多项式界、原运算顺序 | **每轮**调用原 BatchedTaylorModel 和 DenseRemainderLedger 方法；包含所有交叉项、向外舍入和构造检查 |
| proposal、decomposition、包含检查、stop ratio、提交结果 | 本轮动态输出、validation epsilon、当前已保留 R | 每轮调用原 raw-compat 装配和原 atomic decision；任何失败仍原子回退 |
| endpoint E、normal state、历史队列和 owner | 最后接受余项、完整段、边界状态 | 不进入计划；原流程每步执行 |

首轮计算就是该步骤的第一个真实 proposal，不额外生成一个被丢弃的 R0 结果。
记录后立即释放首轮动态中间量；常量节点只保留真实零余项。所有保存的多项式
和区间缓存都来自私有快照，未使用 `.detach()` 冒充复制。内部固定张量在计划
seal 后记录版本，下一轮读前检查；调用方改变返回的内部临时对象会被检测。
检查只读 Tensor 版本元数据，不扫描系数数组；成本计入求解。

Python 函数只准入直接返回 `TMVector` 的字面算术图（状态索引、常数、加减乘、
取负）；拒用闭包、分支、外部数值、属性读取和其他函数调用。此准入针对依赖，
不识别系统名。任意黑盒 callable 保留原路径。函数 code/全局 TMVector 身份、
ODE 结构、candidate/base 对象及其版本、tau/order/cutoff/epsilon/policy 均绑定；
retry、改变 h、candidate 或尺度时由新 attempt 自然重建。
RHS 首次调用沿用原接口约定：先传 state，遇到 TypeError 再传 state 与 None；
两次都失败时保留原单参数异常。未使用但必填的 control 参数也有逐轮回归测试。

计划中的固定张量和原动态运算具有 B 维。局部 B2 用不同 R 验证不串 lane。
现有完整 flowpipe/normal/history 仍为 Python B1；这不是完整 batch 求解器。

开发探针先测到 Brusselator 1–3 和 101–103 步各重放 24/27 次。后者固定
多项式切片约 1.72 秒，总约 5.42 秒，含轻量函数计时开销。因此进入原型，
但没有预先认定完整 1.5× 能达到。正式互斥占比、Amdahl 和完整实测将从本轮
原始测量生成。生产计时没有 profiler，初始化与导出另列，首轮准备始终在 solve 中。
