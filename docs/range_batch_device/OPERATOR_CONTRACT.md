# 局部请求批量范围合同

`evaluate_range_requests(requests, backend="cpu" | "cuda")` 输入不同 ID 的
`RangeRequest`，返回每 ID 的私有上下界张量与状态。它是一个数学算子，未实现
并发 flowpipe 调度器。默认求解器与 prepared/packed 两个公开开关均保持原样。
真实求解器适配显式使用三个上下文：`prepared_remainder_replay(True)`、
`packed_boundary_execution(True)`、`range_request_execution(backend)`。

输入为 CPU binary64：系数 `[output, terms]`，domain `[variables]`。
指数为有序不可变 tuple，整数幂 0..64，不重复支持。保留显式零项及符号零。
各输出共用该请求的有序支持；不同输出结构不能分为一组。普通项按变量序相乘；
normal 先时间、再状态奇偶因子、最后其余变量。normal 状态 domain 必须包含于
`[-1,1]`，其计算使用整个归一化状态范围。超出该保证直接拒绝，不将物理范围
误当归一化范围。分组键包含支持顺序、变量数、类型、状态/时间角色、外供表
语义、输出结构、dtype/device。不合并不同顺序，不用补零强凑共同支持。

外供时间幂表通过原 normal evaluator 逐请求回退。逐个核对表的有限性与其对
指定时间域精确幂的包含；使用其实际数值，不能静默忽略。回退路径也核对最终
精确区间和，必要时向外修正。回退不计为 GPU kernel 处理请求。

CPU 在每次调用内重算变量幂，保留旧标量 `Interval.pow_int` 结果；使用 Python
任意精度整数/有理数校验两个数学端点，仅在原端点失守时向外校正。其余张量
项乘法和逐项 nextafter 加法保持原操作顺序。修正以 `corrected` 明确标记。
旧三次幂反例在测试及原始证据中固定；不修改原 Interval、原 RangePlan 或旧证据。
因此新 CPU 的承诺是：旧幂边界成立时逐位保持；失守时遵循新的严格包含行为。
真实请求上的逐位关系必须单独实测。

精确端点校正的依据是：binary64 输入是精确二进有理数，整数乘方及比较由
任意精度整数完成；转回 binary64 后再次以精确比较决定是否 nextafter。
区间幂的奇次单调、偶次跨零取零、否则绝对值极值均直接由实数函数得出。
CPU 基本乘加以 RN 后各向外一步构造区间；每项区间包含数学项，按原项序
相加后的区间包含区间和，从而包含多项式在域上的所有值。这里不声称区间和
消除了变量依赖，也不声称整个求解器经过形式化证明。

CUDA 只采用一套方案：现有 NVRTC 编译独立 `.cu`，四个内部 kernel 分别验证
输入、计算幂、计算项、按项序累加。双精度加乘使用 `__dadd_rd/ru` 与
`__dmul_rd/ru`，关闭 FMA 合并与 FTZ，无 fast-math，无经验 epsilon。
绝对值的整数幂从 1 开始按固定次数 directed multiply；对负奇次幂用精确
取负交换上下界，对偶次跨零域设下界为 0。0 次为 1，1 次原样返回。
基本运算正确舍入的上下界通过有限次数归纳覆盖精确结果；四端点 directed
乘积的最小下界/最大上界包围每次区间乘法。总和从 0 开始顺序 directed add，
不做树形 reduction。GPU 区间无需与 CPU 区间逐位相同或互相包含。

设备状态按请求独立；状态读写在并行 kernel 中使用原子操作。mask 在读取
该 lane 数值前检查。非有限/倒置输入、幂/乘积/和无法获得有限界时明确失败，
无穷区间不作为证书。失败请求后续不再参与算术；其未定义中间输出不被使用。
每次执行由设备写出四个 kernel 回执，回传并检查。设备代码的导入或编译成功
不足以替代实际执行与精确 oracle 检查。

ID 必须非空且全局唯一，否则无法无歧义 scatter 而整次提交拒绝。其余非法
请求独立返回 `invalid`，不影响正常请求；提交时关闭的 mask 与取消请求
分别返回 `masked` / `cancelled`，不参加检查或计算。同步接口不提供运行中
调度取消。未声明并发写同一个输入张量的支持；两个独立线程上下文与输入
不会共享可变数值。调用期间不修改输入；输出与各请求、输入均不共享可写存储。

隐式缓存只存不可变 RangePlan 指数/阶段元数据和已编译设备代码。系数、domain、
h、余项、幂与最终范围不进入缓存。`ResidentGroup` 是调用者显式持有的输入
张量对象，其常驻成本与复用次数另表记录，不能当作当前 CPU 调用者免费拥有。

设备依据为本机实际编译版本与实测证据。参考：
[NVIDIA CUDA 双精度舍入原语](https://docs.nvidia.com/cuda/archive/12.1.1/cuda-math-api/group__CUDA__MATH__INTRINSIC__DOUBLE.html)、
[PyTorch 数值精度说明](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)。
文档仅说明原语合同，本轮 kernel 的正确性另以真实执行和精确有理数检验确认。
