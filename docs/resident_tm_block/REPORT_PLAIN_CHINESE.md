# 常驻 Taylor-model 复合块：结果首页

## 一句话结论

本轮做成了一个真实被在线求解器消费的 CUDA 连续数学块：它把“当前端点 Taylor map 去常数后，与上一接受边界的归一化 right map 做依赖保持复合”的完整 Horner 图放进一次设备执行，块内中间多项式不回主机。独立 Fraction 检查、在线状态比较和任务生命周期检查均通过；局部 B32 块快了 42.31×/49.57×。但是同一批 32 个不同子盒、每盒 20 步的完整求解只快了 1.1376×/1.1606×，未达到两系统均至少 1.20× 的工程目标。因此候选继续默认关闭，不扩大迁移，也不新跑两个 1000 步全时域。

最终事实状态是：

```text
block_numerical_contract: pass
actual_device_residency: measured
online_integration: pass
local_speed_effect: measured
end_to_end_effect: small_gain
full_horizon_new_candidate: not_run
matched_flowstar_gap: measured
entire_solver_formal_proof: false
full_gpu_engine: false
```

## 先回答六个问题

### 1. 以前为什么只把范围运算送 GPU 不够？

上一轮 packet 已经把物理提交减少约 79%，但每个当前 Taylor 操作仍在 CPU 建请求、生成/消费结果、做稀疏多项式复合、截断、cutoff、余项传播、Picard 验证、接受后历史更新和对象重建。减少 CUDA launch 没有消灭这些数学工作。

本轮在实现前用八个新鲜在线窗口测量，并对重叠 worker 区间取并集，避免把并发等待相加成虚假的墙钟。被选中的 normal composition 占合计墙钟 36.3412%；B32 中 VDP 为 41.5907%，Brusselator 为 32.9197%。更大的 dense Picard/validation 包络占 85.3136%，但它同时包含固定点、验证策略、完整余项 ledger 和边界提交，已经接近迁移另一套求解器，不是一个有限可审计的块，所以没有选择它。

按 36.3412% 的非重叠份额，要让全程缩短 20%，选中块至少要快 2.224×。这个估算说明目标在理论上可能，但不保证在线调度、对象边界和块外成本会随之消失。

### 2. 本轮究竟放到了 GPU 哪一段？

输入边界是当前端点 Taylor map 去常数后的两个分量、上一接受边界的两个归一化 right maps、阶数、cutoff、定义域和 split 元数据；输出是已经完成复合、逐次截断、cutoff、全部余项交叉项和系数舍入付款的 Taylor map。

CUDA 内按两个归一化变量做递归 Horner。每个 `(任务, 输出分量)` 由一个线程执行；乘法产生的保留项、超阶项、自然/分割范围、左右多项式乘对方余项、余项乘余项、外层余项，都在同一 kernel 的局部数组中继续被后续 Horner 层使用。块内主机数值往返为零；一个服务组只有一份数值 payload 与 split 元数据上传、一次 kernel、一次同步，再返回最终数值和执行回执。

这不是完整 GPU 引擎。每个接受边界最终仍要回到 CPU，周围的 Picard、验证、历史队列、调度和对象语义仍在主机。正式 Gr 运行每次都有 640 个 resident 请求、1280 个输出分量且零结构/硬件 fallback；动态到达使 VDP 每次约 475–510 个 resident 组，Brusselator 约 429–465 个组，而不是把全部 640 步强行同步成一个组。

### 3. 为什么没有漏掉舍入、截断和余项？

点系数沿冻结的 binary64 RN Horner 次序计算；NVRTC 使用 `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`。每个系数区间的加、乘、减端点分别使用 CUDA 的向下/向上舍入 intrinsic。每次 TM 乘法都支付四类 remainder：高于目标总次数的截断尾项、左多项式范围乘右 remainder、右多项式范围乘左 remainder、两 remainder 的乘积；cutoff 删除的系数区间也按定义域求范围后付款。最后再把每个保留系数的精确区间与 RN 点系数之差求成区间，并只支付一次其多项式范围。

`1.1 × 1.1` 是具体反例：旧稀疏 CPU 块返回点系数但其 remainder 不包含这个保留乘法的精确误差；新块返回的系数误差区间 `[-0, 2^-52]` 包含它。独立 `fractions.Fraction` oracle 没有调用 CUDA interval helper，也没有拿旧 CPU 输出当真值；order 4/6 共检查 2752 个点系数、2752 个保留系数误差区间和 128 个最终 remainder。

任务隔离由完整 basis/变量角色/图/舍入/cutoff 指纹、run/task/epoch/generation/attempt/request 身份和 owned copy 保证。不同图不会混组，当前数值重新生成有效掩码；错指纹和非有限输入失败关闭。取消后的迟到结果被丢弃，整段成功后才在同一锁下原子提交接受状态。

### 4. 同一批真实任务的新旧完整时间是多少？

正式矩阵固定为原 8×4 的 32 个不同子盒、每盒 20 步、单 CPU 核、单 BLAS/Torch 线程、Tesla V100。Gp 是上一轮 packet 路线；Gr 只在 Gp 上显式打开本块。五个配对块预先固定并交替顺序，没有为越过阈值追加样本。每个运行都成功完成 640 lane-step。

| 系统 | Gp 墙钟五次 (s) | Gr 墙钟五次 (s) | 配对 `Gp/Gr` 中位数 | Gr 胜场 | 1.20×目标 |
|---|---|---|---:|---:|---|
| Van der Pol | 279.094, 282.464, 282.683, 284.355, 288.124 | 247.362, 246.334, 249.890, 249.952, 250.043 | 1.137641 | 5/5 | 未达到 |
| Brusselator | 411.505, 409.170, 409.232, 413.401, 410.322 | 354.576, 356.392, 350.282, 355.409, 355.177 | 1.160553 | 5/5 | 未达到 |

Gr 墙钟中位数为 249.890 秒和 355.177 秒。较快严格 CPU Q 的三次中位数为 336.347 秒和 485.912 秒；其相对各 Gr 样本倍率的中位数为 1.34598× 和 1.36809×，所以没有稳定 CPU 回退。S 每系统仅做一次有限核对（389.747/578.234 秒），不是配对性能分母。

局部真实 B32 输入上，含请求构建、打包、H2D、kernel、D2H、回执检查和 CPU 对象重建的新块相对旧块中位数为 42.3135× 与 49.5727×，两系统都是 5/5。可是正式 Gr 中，resident kernel 设备时间每次合计中位数只有约 0.502/0.521 秒，打包约 8.331/7.116 秒，返回后检查和重建约 9.007/9.588 秒；完整墙钟仍有约 230/336 秒在这个 kernel 以外。实现前的 Amdahl 份额也高估了可串行消掉的关键路径，因为在线 worker 的复合、等待和边界工作存在重叠。局部目标达成，整体目标没有达成，这两个结论不能互相替代。

### 5. 与相同工作量 Flow* 相比还差多少？

新建的测试驱动不改 Flow* 算法，在同一个 CPU 核、单线程、同一进程内顺序处理相同 32 个外向 binary64 子盒和每盒 20 步。VDP 使用 h=0.01、order 4、cutoff 1e-10、初始 remainder ±1e-4、SR100；Brusselator 使用 h=0.02、order 6、其余相同、SR1000。新进程外部墙钟包含启动、动态装载、求解、最终 endpoint 记录和退出：

| 系统 | Flow* 完整墙钟 (s) | Flow* reach (s) | Gr 中位墙钟 (s) | `Gr/Flow*` |
|---|---:|---:|---:|---:|
| Van der Pol | 0.702908 | 0.683962 | 249.890 | 355.510× |
| Brusselator | 8.024463 | 7.951032 | 355.177 | 44.262× |

两次 Flow* 都完成 640 lane-step，32 个任务的 native status 都是 `COMPLETED_SAFE`。该数值只是现实速度尺度，不把 Flow* 当作浮点数学真值。Flow* 依赖不能暴露或强制本项目特有的“接受后 raw-remainder 收紧、最多 491 次 replay、0.99 停止、整向量原子提交”控制，因此这里只能严格匹配方程、盒、步数、h、阶数、cutoff、初始 remainder 和 SR 容量；历史/接受语义不宣称完全相同。

使用的是 GPL-3.0 Flow* stock 基点 `b85a321...`，其上只有提交 `722a561...` 增加两个 const 只读 term accessor；静态库 SHA256 为 `274e9bd...`。本轮只增加测试驱动，未修改算法，计时区也没有调用全模型导出。仓库不打包 Flow* 二进制或源代码，只保存身份、构建命令、输出和哈希。

### 6. 是否值得继续扩大这个方向？

这个块值得保留为“数学合同已建立、真实在线可用”的 opt-in 原型，但本轮数据不支持沿同一边界继续扩大：两系统都稳定小幅变快，却都离 1.20× 门槛明显，且与 Flow* 仍有 44–356× 的现实差距。默认保持关闭；不能把局部 42–50× 写成全求解器倍率，也不再回头微调 packet 或等待参数来凑阈值。

下一轮若继续，合理的有限问题是评估既有完整设备架构中是否有可隔离、可独立验证、能跨多个求解阶段保留状态的部分可复用；这不是自动授权迁移整个引擎。新的原始未分区 1000 步候选长跑按预先规则标记 `NOT_RUN_PERFORMANCE_NOT_USEFUL`。

## 正确性与在线接入结果

诊断共 24 个新运行、4208 个完成的 route lane-step。两系统各覆盖 B1/B2 三步、B8/B32 二十步、两个不同任务连续 120 步；VDP 120 步跨过 SR100。另用父轮真实 step99/step999 完整 checkpoint 各恢复一个三步窗口，明确只叫恢复窗口，不冒充从初始盒新跑到 T10/T20。

Gp/Gr 的 12 组同设备比较中，接受决定、状态步、h、拒绝/验证次数、ordered support 和 ledger 类别均无差异。完整二进制状态预期不相等，因为新块补付了旧路径漏掉的保留系数舍入；因此报告数值差而不是声称逐位一致。同设备 endpoint/tube x/y 共 8416 行，复用父严格 CPU 的对应宽度共 8368 行，没有一行非零宽度比超过 1.10。最坏同设备比为 Brusselator 任务 31、连续第 120 步、endpoint x：`1.0000051904426903`；VDP 最坏为恢复窗口 `1.0000021107233557`。

这些结果证明的是本轮有限合同和实际运行的包含/一致行为，不是整个求解器的形式化证明，也没有消除默认路径历史上其他已知局限。

## 哪些是新数据，哪些是复用数据

本轮新数据包括：八个选择窗口、64-lane Fraction 局部矩阵、两系统各五次局部 B32 对、24 个在线诊断、20 个 Gp/Gr 正式 GPU 运行、6 个 Q 和 2 个 S 运行，以及两个 matched-workload Flow* 运行。

复用数据只有两类：父轮严格 CPU 诊断用于宽度尺度；父轮从原始未分区初始盒完成的 Gp 1000 步及其 step99/step999 checkpoint 用于历史恢复与说明。父轮长跑源码为 `1d870939...`、修订验证器为 `0b0da13d...`。它们没有被贴到新候选名下，也没有作为本轮新长跑计数。

第一次诊断求解本身完成，但旧验证器假定精简事件中的 `request` 必为字典，遇到合法的 `null` 后报 schema 异常。提交 `8953b5ea...` 只把读取改为 `(request or {})`；没有放宽数值、包含、身份或生命周期条件。旧失败输入、日志和修订范围都保存在 `raw_minimal/validator_amendment_before_fix/`。

Flow* 驱动第一次试跑又暴露出相邻数学分区边界各有独立向外舍入值，不能压成共享 binary64 边界。该试跑在正式结果前被拒绝，摘要和旧驱动哈希保存在 `raw_minimal/flowstar_shared_boundary_rejected/`；最终驱动逐盒对照 `PARTITION_PLAN.json` 的四个十六进制端点后才接受结果。

## 证据入口与复现

主证据目录是 `artifacts/runs/resident_tm_block_20260914T032650Z/`。重点文件：

- `RESULT.json`：机器可读最终事实状态；
- `SOURCE_MAP.json`：科学运行源码、验证器修订、Flow* 和复用父证据身份；
- `local_exact_checks.json`：独立 Fraction 包含矩阵与漏付反例；
- `online_state_comparison.json`：真实连续状态、决策、support、ledger 和宽度；
- `timings_raw.csv`、`paired_speedups.csv`：全部规定正式样本；
- `device_residency_and_calls.csv`：传输、kernel、同步、回执和 host 边界成本；
- `matched_flowstar.csv`：同工作量 Flow* 尺度与合同缺口；
- `full_horizon_result.json`：为什么没有新增 1000 步；
- `tests/affected.xml` 与 `tests/commands.json`：269 项受影响测试；
- `SHA256SUMS`：交付内文件完整性。

具体数值规则见 `NUMERICAL_CONTRACT.md`。候选实际科学运行提交是 `8953b5ea24e69d11ac5c2890a4cdb663305e9e79`；Flow* 测量驱动提交是 `b2dfba33...`。完整性能没有在封装后重跑。
