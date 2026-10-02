# ARCH-COMP26 Single Pendulum 第三态与 MATLAB 执行身份：定点来源复核

只读日期：2026-10-03。本页接续[既有合同审计](ARCHCOMP26_NEXT_CONTRACT_SOURCE_AUDIT_20261001.md)，仅追查 MATLAB 第三态初值、闭环入口/重置及性质检查时序；未启动实验或进行内容摘要校验。

## 已核实的一手材料

| 来源 | 实际所见 | 对执行身份的限度 |
| --- | --- | --- |
| [2026 AINNCS 报告 §3.5，印刷页 93](https://easychair.org/publications/paper/GsKW/download) | 明定 `x1=θ`、`x2=θ̇` 两个物理态，初集 `[1,1.175]×[0,0.2]`，控制周期 `0.05 s`；连续性质为闭窗 `t∈[0.5,1]` 内 `x1∈[0,1]`，离散对应 `k=10..20`。 | 没有定义第三态、它的初值或参与者 MATLAB 程序的检查调用次序。报告给出数学性质，不能证明某一提交的 checker 已按整个闭窗执行。 |
| [固定官方 `dynamics_sp.m`](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Single_Pendulum/dynamics_sp.m)、[规格](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Single_Pendulum/specifications.txt) | 前两项导数是论文两态方程；MATLAB 文件另返回 `dx(3,1)=1`，注释掉的旧行是 `dx(3,1)=20`。规格只列两个初态区间、控制周期和性质闭窗。 | `dx3=1` 说明第三分量在该函数中以单位速率变化，却没有给出 `x3(0)`、控制周期之间是否重置、是否送入 NN，以及它是否或怎样用于性质判断。不能从导数单独推出这些字段。 |
| [固定提交的递归仓库目录](https://api.github.com/repos/Kiguli/ARCH-COMP2026/git/trees/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26?recursive=1) | `benchmarks/Single_Pendulum/` 只有 `.h5/.mat/.nnet/.onnx` 四种控制器格式、`dynamics_sp.m` 与 `specifications.txt`；递归列表未截断。 | 没有官方 MATLAB 闭环调用者、初集建立/重置程序或性质 checker 可供复核。 |
| [原始基准仓库递归目录](https://api.github.com/repos/amaleki2/benchmark_closedloop_verification/git/trees/master?recursive=1)、[`example4.py`](https://github.com/amaleki2/benchmark_closedloop_verification/blob/master/example4.py)、[`pendulum/pendulum.py`](https://github.com/amaleki2/benchmark_closedloop_verification/blob/master/pendulum/pendulum.py) | 原始仓库没有 Single Pendulum MATLAB 闭环 `.m` 入口。`example4.py` 用二维随机初态建立 `Pendulum1Env`，每次从当前二维 `p1.x` 求 NN 再调用 `step`，共 150 次；`Pendulum1Env` 指定 `n_pend=1`，其 `reset` 恢复二维 `x_0`。 | 这是原始作者的 Python 两态示例，初态抽样和 150 步也不同于 2026 规格；它支持两态来源背景，但不能认证 2026 参与者的三态 MATLAB 初值、重置或 checker 时序。 |

报告所指[2026 重复性归档](https://gitlab.com/goranf/ARCH-COMP/-/tree/master/2026/AINNCS)于本次公开读取时尚无 `master/2026/AINNCS` 目录；[GitLab API 对该路径返回 404](https://gitlab.com/api/v4/projects/goranf%2FARCH-COMP/repository/tree?path=2026%2FAINNCS&ref=master&per_page=100)，`master` 根目录只列出 2017–2025 年。由此路径目前无法取得参与者实际提交，但不能据此断定不存在未公开的执行记录。

## 当前结论与具体缺件

2026 **论文两物理态**合同及其全时闭窗性质已由论文和官方规格确定；本轮四方两态数值结果仍只属于具名的 `paper-two-state` profile。若要进一步称为某个 **2026 MATLAB 三态提交**的忠实复现，还须取得该提交的实际闭环入口或等价权威运行记录，其中至少能读出：

1. 第三态 `x3` 的初值、每个控制周期内与周期间的重置/推进规则；
2. NN 实际输入投影和控制器在周期边界的调用先后；
3. 性质检查器如何覆盖 `t=0.5`、`t=1` 及两者之间所有连续时刻，是否使用 `x3` 作为时钟。

现有权威材料没有给出以上三项；`dx3=1` 与原始两态示例都不能代填。没有调整主表身份或运行任何新任务。
