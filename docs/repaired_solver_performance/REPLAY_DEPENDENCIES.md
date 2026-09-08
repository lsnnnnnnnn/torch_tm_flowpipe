本轮机制是 `prepared_remainder_replay`：在首次接受后的同一次 attempt 中，将只依赖点多项式的工作准备一次。进入上下文才启用；默认 reference 和 VDP 的既有 canonical closure/cache 保留。

| 保存的量 | 依赖 | 创建、读取和失效 |
|---|---|---|
| 输入点系数、domain、basis 路由 | 本次 candidate、base、order、实际 h、设备/精度 | 创建计划时深复制，包括 basis 张量；不共享外部可变存储。每轮只读取私有副本。新的 attempt/retry 创建新计划。 |
| 有序操作记录和标量常数 | RHS 当前执行的加减乘顺序、输入节点、常数精确值 | 首轮实际执行时记录，结束后封为 tuple。每轮仍调用 RHS，逐操作核对类型、输入节点和常数；结构变化拒用旧计划。没有按系统名称写死 Brusselator 方程。 |
| 乘积的保留点系数和截断尾项范围 | 两个输入点多项式、degree、固定 domain、range policy | 首轮调用原 `BatchedPolynomial.mul_trunc`；随后读取相同结果。原 scatter 合并和向外范围计算顺序不变。 |
| 两个乘法操作数的 polynomial-only range | 点系数、domain、range policy、原调用 context | 首轮分别按原 left/right context 求范围；每轮与当前另一操作数的 R 相乘。它们没有包含某一轮的 R。 |
| cutoff 的保留点系数和移出范围 | 当前图节点的固定系数、cutoff、domain、policy | 对实际系数计算 mask，再保存该节点的结果。不是按 basis 缓存一个可用于变化系数的 mask。 |
| regular RHS 积分点系数与 overflow range | regular RHS 固定点系数、积分变量、order、domain | 首轮按原积分路径准备；每轮仍执行当前 R 的时间乘法、ledger 缩放和 overflow 加入。 |
| polynomial difference 及其未膨胀范围 | base、regular RHS 点系数、candidate、domain、cutoff | 首轮从实际 regular tmp 得到；每轮仍按原顺序执行 validation epsilon 膨胀、余项合并和最终包含检查。 |

动态工作每轮完整保留：当前 R，P×R、R×P、R×R，raw 和 regular 两条路径的全部 ledger 类别与求和，普通余项的时间缩放，subset/停止比率/原子提交、非有限数处理，以及验证分解的包含断言。regular RHS 的动态账本在 production-no-observer 中也继续计算。没有利用“报告关闭”省去可能影响错误处理的运算。

计划绑定 RHS、base、candidate 对象身份及 order、tau、cutoff、epsilon。原输入张量的版本检查是固定数量的元数据读取，不逐元素扫描大张量；domain 中包含实际 h，base 包含当前尺度和初始余项。外部张量即使存在别名，也与保存的数值深复制隔离。内部固定张量只通过私有执行方法使用，公开 `image(R)` 返回新 proposal 和复制后的账本，不返回缓存张量。失败的执行使计划失效；下一 attempt 不能继续使用半建好的记录。

`ContextVar` 只保存本调用上下文的布尔开关，不保存数值。退出上下文或异常退出均恢复先前设置。计划是 `_post_accept_refine_raw_remainder` 内的局部对象，零轮收紧不创建计划。默认分支继续调用原 evaluator；491 上限、0.99 停止比率、whole-vector commit、normal 左右映射、SR 队列和端点修正均未改变。

本轮只权威验证 CPU binary64。张量布局保留 B 维，局部 B2 用不同余项和交错的两个计划检查隔离；完整 solver 仍为原有 Python B1 流程。这不是完成 batch 或 GPU 后端的声明。

准备计数包含首次构造和原多项式操作，命中检查也在求解计时内。`prepared_plan_setup_s` 是 solve 内的子项，不能与 solve 再相加。单独 profiler 的各类秒数采用嵌套 scope 减去已测子 scope；inclusive 列只用于定位，不能加总为占比。
