# GPU 范围工作包与所有权合同

本机制默认关闭，只由 `LiveRangeService(packet_mode=True)` / 路线 `Gp` 显式开启。
`S`、`Q`、父在线 GPU 路线 `G0`（兼容旧名 `G`）及独立 GPU 路线的数学输入和
默认行为不随之改变。Gp 仍只执行已有的稀疏范围运算；端点代入、Picard、历史矩阵、
dense 范围和求解器其余阶段继续在 CPU。

## 数学不变量

一个 `RangeWorkPacket` 可以包含多个结构 key，但每条请求仍保留自己的：有序支持
（含显式零项）、输出数、变量数、standard/normal/interval-coefficient 类型、时间/
状态变量角色、domain、幂条目、逐项乘法链和从零开始的顺序 directed 累加。不同支持
不补零、不做 union、不重排，也不共用求和次序。normal 的阶段顺序仍是时间幂、状态
奇偶因子、其余变量。外供时间幂表不进入 packet，继续由已检查的原 CPU fallback
逐请求执行。空支持仍产生精确零。

设备收到两个连续输入：一个 `int64` 元数据区和一个 binary64 数值区。请求、power、
coefficient-cell、output、variable 和 term-operation 描述符都带 checked offset/length；
四个 kernel 分别验证、算幂、按原链算项、按原项序求和。请求状态互相独立。坏数值、
溢出或局部坏描述符只使所属请求失败；坏包头或无法归属请求的描述符使整包失败，结构
损坏不能成为成功区间。每包的 buffer epoch、状态和四阶段回执都从零重新初始化并回传。

## 在线选择和公平性

20 ms 仍只是父合同冻结的派发资格规则，不重新搜索。超时、单 key 满 32 或全部存活
任务等待时才取得派发资格。Gp 随后从当时已经就绪的请求中按提交时间选最多 32 条；
最老请求必先进入，未来请求不会预先执行。每任务仍只允许一条未消费请求。失败、取消、
重试、旧 generation、旧 epoch、迟到及重复返回继续由完整 run/task/epoch/generation/
attempt/counter 身份检查处理。

## 所有权和生命周期

调用者的 coefficients、domain 和外供表在服务提交前先复制。packet 再把当次实际数值
复制为连续 CPU payload；GPU 不缓存上一步数值、幂或答案。服务线程独占 CUDA stream
和一个 `RangePacketExecutor`。executor 串行持有 pinned staging 与 device scratch，直到：

1. 本包 H2D 完成；
2. 四阶段 kernel 完成并回传 epoch/status/receipt；
3. 必要的结果 D2H 完成；
4. 每条健康结果已复制为私有 CPU tensor。

之后 scratch 才可用于下一包。排队取消不进入包；在途取消只使身份失效，不能提前复用
正在读写的缓冲；完成未消费的结果也已是私有 CPU 副本。交付时任务再克隆一次，调用者
修改输出不能影响同包其他任务或后续包。包内没有可写结果别名，也不把数值缓存称为
“完整边界状态常驻 GPU”。

## 固定资源上限和超限行为

默认上限为：32 请求、262144 个有序 term、262144 个 coefficient cell、2048 variables、
131072 powers、1048576 term operations、2048 outputs；metadata/numeric/output 分别不超过
64/64/128 MiB，单包三者合计不超过 128 MiB。scratch 只按二次幂增长到这些上限，不缩小，
每次增长和实际 pinned/device 字节均计数。多请求超限时只在请求边界安全切包；一个合法
请求本身超过 packet 上限时显式走父 G0 CUDA 算子并记录 `oversize_parent`，不会先 OOM
再偷偷改变正式配置。

正常非诊断包实际执行两次 H2D（metadata、numeric）和两次 D2H（header/status/receipt、
最终数值），以及一组四阶段 kernel。诊断包仍是相同四次 kernel，只扩大数值 D2H 以取回
每项和每个幂。纯设备 event 时间、host enqueue、host completion sync、打包、scatter、
包信封复查、分配和字节/操作计数分别记录；scratch 同时报告当前逻辑容量与历次实际
申请的累计字节，不能把 kernel event 时间替代完整 wall。

## 失败策略与可声明范围

GPU/驱动/packet 级异常默认使当前包请求显式 `backend_error`；只有调用者开启
`hardware_fallback` 才用同一私有输入在 CPU 重算，并单列失败耗时和重算耗时。局部请求
继续由独立 Fraction 检查幂、项和总和；同 GPU 的 G0/Gp 应逐位一致。这个合同不证明整个
ODE 求解器形式化正确，也不把 GPU 范围参与的求解链称为全 GPU 引擎。
