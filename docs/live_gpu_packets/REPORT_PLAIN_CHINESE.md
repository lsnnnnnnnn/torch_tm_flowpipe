# 异构 GPU 范围工作包：完整在线实测与长时域结果

科学运行源码：`1d870939fcdb72bd999730cf09de0b14d1cd71c4`。结论：未观察到足够的完整 wall 收益；保留正确实现和测量，不默认启用。

## 先说结果

Gp 的正确性门通过；正式性能由冻结的 5 个配对区块、两系统各 S/Q/G0/Gp、每次
B32×20=640 个成功 lane-step 得出，没有看到结果后追加样本。逐系统结果是：

- VDP：G0/Gp 中位数 1.050×，Gp 胜 5/5；较快严格 CPU/Gp 中位数 1.202×，Gp 胜 5/5；scratch 末 1/4 无继续增长=True，最大逻辑设备容量 0.05 MiB。
- Brusselator：G0/Gp 中位数 1.037×，Gp 胜 5/5；较快严格 CPU/Gp 中位数 1.190×，Gp 胜 5/5；scratch 末 1/4 无继续增长=True，最大逻辑设备容量 0.05 MiB。

机器判定为 `no_gain`，工程目标
`engineering_target_met=false`。这个字段同时要求
两系统 G0/Gp 中位数至少 1.15、至少 4/5 配对胜出，并按冻结定义排除相对同区块较快
严格 CPU 的稳定回退。接口继续默认关闭。

两条新鲜长跑都从原始未分区 B1 初始盒开始，没有读保存答案或周期性 CPU 纠偏：VDP
完成 1000 步到 T10，Brusselator
完成 1000 步到 T20。每一步的
endpoint/tube x/y、余项类别、接受与收紧、精确累计时间、历史 owner/清空、完整状态指纹
和分项时间都在压缩原始流中；checkpoint 也被重新加载核对。

长跑计算源码仍是上述科学 SHA。Brusselator 完成后，冻结离线验证器因把 VDP 的
`c3_cross_step_sr_v1` owner schema 错用于两个系统而拒绝；Brusselator 的既有运行合同和
测试明确使用 `accepted_boundary_sr_v1`。`full_horizon/VERIFIER_AMENDMENT.json` 保留原失败、
两个计算进程的成功收据、逐文件旧/新哈希和零次重跑事实；修正版按 plant 显式映射后重新
逐行验证两个既有完整流。求解器、CUDA kernel、packet 与 scheduler 运行时字节均未改变。

## 实现究竟改变了什么

Gp 只改变范围请求的物理组织。一次取得派发资格后，它按提交时间从当时已就绪的不同
结构 key 中最多取 32 条，生成一个有界连续 packet；请求自己的支持顺序、输出数、变量
角色、domain、power/term 链和从零开始的 directed 顺序累加都没有合并或重排。一包实际
执行 validate、power、term、sum 四个 kernel，正常路径各两次 H2D/D2H。外供幂表仍明确
走 CPU fallback；单请求超限明确走父 G0；未来请求不预执行。

输入在排队前私有复制，packet 数值再次连续化。服务线程独占 stream，直到 kernel、回执、
D2H 和私有 scatter 全部结束才复用 scratch。取消、旧 generation/epoch、错 ID 和完成未消费
结果都不能把缓冲别名交给下一任务。设备端还检查全局布局、每请求精确分区以及 power、
cell、operation 的请求所有权；跨请求但仍处于全局合法范围的 offset 也不能成功。

## 为什么值得做这个 packet

父在线 GPU 的冻结生命周期记录显示如下；这是静态机会测量，不是把重叠快照相加成
反事实加速：

- VDP B8：父派发 11946 次，其中真实 KEY_SPLIT 89.93%，就绪请求 P50/P95=6.0/7。
- VDP B32：父派发 26916 次，其中真实 KEY_SPLIT 98.29%，就绪请求 P50/P95=21.0/30。
- Brusselator B8：父派发 15027 次，其中真实 KEY_SPLIT 88.17%，就绪请求 P50/P95=6/8。
- Brusselator B32：父派发 32218 次，其中真实 KEY_SPLIT 95.31%，就绪请求 P50/P95=23.0/31。

本轮的新计数表逐运行记录实际 kernel、传输操作、字节、device/pinned 分配、packet 数和
超限/fallback。成本表把请求 ownership copy 的任务和/区间并集、锁等待、选择、信封检查、
packet build、host staging、纯设备 H2D/kernel/D2H、host 同步与 scatter 分开；不会把并发
Future 等待相加冒充 wall，也不会把 GPU event 时间冒充完整求解时间。

## 正确性与范围比较

受影响测试为 220 项，失败 0、错误
0；真实双请求设备固件记录四个 GPU 回执。诊断门覆盖
26 个在线 run，并用独立 Fraction 对 49162
条捕获请求重新检查幂、项和有序和；同 GPU 完整 segment 比较
2122 条。非法数值/溢出只污染所属请求，结构损坏不变成
成功区间。

新 Gp 长跑与 REUSED 的完整严格 CPU/stock Flow* 对象都由当前严格 CPU observer 重新观察，
共 16000 行比较；reference 宽度近零另列，GPU/reference>1.10 的
3550 行逐位置保留。宽度比是测量，不是安全证明；没有选较窄
端点、求交或用抽样替代包含检查。

## 边界与复现

这不是完整 GPU ODE 引擎：GPU 只参与既有范围算子，Picard、端点代入、历史和其余求解器
数学仍在 CPU。它也不是整个 ODE 求解器的形式化证明，更没有宣称修复全仓 `pow_int`。
正式 wall 包含初始状态、worker/服务、复制、等待、安全检查、传输、同步、求解和清理；
只有编译与一次性设备自检在外，首包 scratch 分配仍在内。

原始索引、冻结计划、派生表、逐文件 SHA256 和执行方法分别见
`RAW_INDEX.json`、`PLAN_FROZEN.json`、本目录各 CSV/JSON、`SHA256SUMS` 与
`experiments/live_gpu_packets/README.md`。独立 clone 的验证器会重算派生决定、核对两个
1000 步流及 checkpoint，并真实运行两系统 B2×2 的 G0/Gp（合计 16 lane-step）；它不会
把这称为重跑两个完整长时域。推送与三方 SHA 的实际回执保存在 `acceptance/` 及最终交付记录。
