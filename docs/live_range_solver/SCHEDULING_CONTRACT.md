# 在线范围调度合同

默认关闭。`LiveRangeService` 不改变求解器公开默认值。调用者先统一配置 Torch
线程数，再创建服务、注册独立状态、启动 worker；每个 worker 使用自己的
`RangeTask.execution()` 上下文，显式开启 prepared replay / packed boundary。
`begin_attempt()` 返回上一个接受状态；真实求解器计算成功后才 `commit()` 整个
下一状态。数学函数、端点 E、完整余项、左右映射与历史传播继续用已有实现。

每条请求包含 run/task/epoch/accepted-boundary generation/attempt/counter。
generation 是接受边界版本，区别于历史队列内部 SR 清空 generation。
attempt 是一次调用现有完整 step 的外层尝试；该 step 内部的同步收紧/验证循环
保留原顺序，并在完整 segment 中记录次数与停止原因。重试必须 begin 新 attempt。
同任务只允许一条未消费请求；后一条请求只能在前一条 Future 返回并通过身份检查后
提交。生产保留身份检查，完整状态/输入摘要只在诊断遍历启用。

提交复制系数、domain、外供表。支持、项序、类型、变量角色、输出形状、dtype/device
与外供表语义组成原完整 key，不补零、不按调用序号配齐。结果按完整身份检查并
复制给调用者。服务不会缓存系数、domain、幂、答案或未来请求。

每服务一个独立线程提交 GPU0，显式 CUDA stream、完成 event 和四个设备回执。
原语自检含极小数、相消与已知三次幂例，首次失败不放行；同进程相同模块后续复用。
所有 kernel 算术不变，只增加已验证回执的记录。CUDA 错误默认影响当前组任务；
显式启用 hardware_fallback 才以同一私有请求 CPU 重算，并记录失败耗时、重算耗时
与请求数。正常外供表另记 CPU fallback，绝不计作 GPU 完成请求。

FIFO 按最老就绪 key 服务。达到组上限、全部存活任务等待/结束，或者就绪等待
超过 20 ms 即派发。超时请求优先于新满组，防止饥饿。20 ms 是派发资格上限，
不是在 CPU 正忙或某组执行中保证返回的实时 deadline；实际等待完整记录。
20 ms 在预注册 2/20 ms 两候选中由两系统 B8 两步 Q/G 总耗时选定，随后冻结。

取消与 commit 共用同一把锁。取消解除当前等待，丢弃整个未提交 attempt，之后
迟到/重复/旧 epoch 返回无效。错误只使相应任务失败，不返回上一次范围作为答案。
checkpoint_state 先取消未提交计算，复制最后接受状态；由已有安全 JSON checkpoint
接口保存。载入后 register 会创建新 epoch，旧 worker 无法提交到新任务。

接入覆盖 scalar CPU64 的 Polynomial standard/normal 和 accepted-boundary SR
interval-coefficient packed 调用。外供幂表显式走原 evaluator 加既有精确校验/校正。
空支持保持精确零，无需设备请求。非 scalar/CPU64 sparse 输入继续原回退；dense
Taylor-model 内部范围/余项算术仍在 CPU，本服务没有将所有范围调用都迁到 GPU。

S/Q 使用同一新请求 CPU 算术，G/S_gpu 使用同一 CUDA 算术；同后端比较完整
segment 和下一输入。CPU/GPU 各自独立 Fraction 检查，不能用宽度比较替代包含证明。
L 保留已知旧 pow_int 缺陷，仅作有限实测速率参照。
