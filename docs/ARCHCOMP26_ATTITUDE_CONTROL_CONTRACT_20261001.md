# ARCH-COMP26 Attitude Control `avoid`：新执行合同与旧 checker 错误

状态（2026-10-01）：官方动力学、初集、控制器函数和性质极性已核对。**保存的参与者 checker 把第四维 unsafe 区间写成空集；其旧 `VERIFIED` 不能证明本页的官方性质。** 新实验须使用修正后的区间并另立运行目录。本页针对采样保持的 Attitude Control；论文中 immrax 的连续反馈结果属于另一语义。

## 来源与任务

- 论文：[ARCH-COMP26 AINNCS 报告第 3.8 节、表 2、第 4.2.8 节](https://easychair.org/publications/paper/GsKW/download)。同一段先定义“不得到达 unsafe 集”，又写“要证明该规格不成立”；但表 2 将 CROWN-Reach 的 `avoid` 结果列为 verified。
- 固定官方仓库 `Kiguli/ARCH-COMP2026` 的 `d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26`：[Specifications.txt](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Attitude-Control/Specifications.txt)、[dynamics.m](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Attitude-Control/dynamics.m)、[torch ONNX](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Attitude-Control/attitude_control_3_64_torch.onnx)、[另一种 ONNX 表示](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Attitude-Control/model.onnx)。
- 保存的作者参与者在本机 `results/archcomp_review_20260923/sources/xiangru/submit/CROWN-Reach/archcomp/AttitudeControl/{crown.py,attitude_control.cpp}`；native 保存版同名 C++。服务器旧匹配构建源为 `N/runs/archcomp_review_20260923/suite_build/archcomp/attitude_control/matched_threads4.cpp`，其中 `N=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z`。

| 字段 | 2026 采样保持合同 |
| --- | --- |
| 物理状态 | `x1..x6 = (ω1,ω2,ω3,ψ1,ψ2,ψ3)` |
| 控制 | 6 输入、3 输出神经网络；`u1..u3` 每 `0.1 s` 更新并保持 |
| 初始集合 | `[-.45,-.44] × [-.55,-.54] × [.65,.66] × [-.75,-.74] × [.85,.86] × [-.65,-.64]`，全盒一分区 |
| 时域与作者数值设置 | `[0,3] s`，30 控制期；每期两段 `h=.05`，Taylor order 3，cutoff `1e-6`，初始余项 `[-.01,.01]`，符号余项队列 1000 |
| 官方 unsafe 集 `U` | `[-.2,0] × [-.5,-.4] × [0,.2] × [-.7,-.6] × [.7,.8] × [-.4,-.2]` |
| 性质 | 对整个闭区间 `[0,3]` 证明 `Reach(t) ∩ U = ∅`；`VERIFIED` 代表证明避免 `U` |
| 绘图 | 论文用 `ω1/ω2` 投影；二维图不能替代六维 unsafe 判定 |

官方 MATLAB 前三条 ODE 是 `x1'=0.25(u1+x2*x3)`、`x2'=0.5(u2-3*x1*x3)`、`x3'=u3+2*x1*x2`。后三条是所链 `dynamics.m` 的 Rodrigues 参数多项式，与保存参与者 C++ 的前六条表达式逐式一致；作者另增 `t'=1` 和 `u1'=u2'=u3'=0`，使输入每期保持。新 runner 仍须将实际编译源与六条官方导数逐式比对。

## 两份 ONNX 的关系与选择

官方副本本机位于 `results/archcomp26_20261001/official_attitude_control_3_64_torch.onnx`、`results/archcomp26_20261001/official_model.onnx`，服务器在 `N/runs/archcomp26_20261001/attitude_prep_001/`。本次没有计算任何内容摘要。

| 文件 | 输入/输出 | 图 | 选择 |
| --- | --- | --- | --- |
| `attitude_control_3_64_torch.onnx` | float32 `[1,6]→[1,3]` | `Gemm→Sigmoid` 三次，末层 `Gemm` | 保存的参与者 `crown.py` 明确加载；新四方主控制器 |
| `model.onnx` | float32 `[B,1,1,6]→[B,3]` | `Sub→Conv→Sigmoid` 三次，末层 `Conv→Flatten` | 同函数的另一种表示；使用时标明 reshape/后端 |

用现有 ONNX/NumPy 环境解析两图：输入 `Mean=[0,0,0,0,0,0]`；四层对应的 float32 权重和偏置逐元素**精确相等**；各 `Gemm` 为 `alpha=beta=1, transB=1`，首层 `Conv` 为 `1×6`，后续 `1×1`，无填充、步幅 1、分组 1。按相应 reshape，两图是同一个四层仿射／sigmoid 函数。初盒中心两种手动图求值同为 `[2.9856860637664795, 0.5552264451980591, -0.6381427645683289]`。这是**数学函数等价**，不声称不同 ONNX 后端的浮点执行逐位相同。

服务器官方 torch 文件与三份保存的 2024 参与者模型（`CROWN-Reach_Development`、其 native 分支、`xiangru_upstream`）用 `filecmp.cmp(..., shallow=False)` 直接逐字节比较，均为 `True`，各 54,656 字节。另一官方 ONNX 为 37,226 字节。选 torch 文件的依据是参与者执行入口明确指定，加上它与固定 2026 文件直接字节一致。

## 极性与必须修复的 unsafe 约束

官方实例名为 `avoid`。作者 C++ 调用 `result.unsafetyChecking(unsafeSet,...)`，其 `result.isSafe()` 输出 `VERIFIED`；故新主实验固定为**避免 `U`**。论文“要证明该规格不成立”的一句与官方性质、作者 checker 和 2026 表格冲突，记录为文本矛盾，不用来反转实际 checker。

保存的 native 和 Xiangru C++、以及两份保存的 YAML 都在 `x4` 处写成：

    -x4 - 0.4 <= 0     即 x4 >= -0.4
     x4 + 0.6 <= 0     即 x4 <= -0.6

两条不可能同时成立，故旧 `unsafeSet` 是**空集**，旧 `VERIFIED` 对官方 `x4∈[-.7,-.6]` 无证明价值。官方区间要求首条改为 `-x4 - 0.7 <= 0`，第二条不变。其余五维的 10 条约束与官方上下界一致。更改只影响性质 checker，不改 ODE、模型或流管；必须使用新 run ID，旧时间/宽度只作为历史执行记录。

Flow* `unsafetyChecking` 按每条 `expression <= bound` 判交集。服务器实际 `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.h:1148` 的 `reach(..., result, ...)` 在同一个 `result.flowpipes` 中追加 ODE 小段；`Continuous.cpp:5191` 的 `Result_of_Reachability::unsafetyChecking` 遍历整个列表。保存的 Attitude C++ 在 30 期内复用 `result` 且不调用 `clear()`。修正约束后，这一 checker 才有全时域覆盖的实现路径。新运行还须检查 60 个小段完整产生、无拒绝、性质状态及六维范围；作者浮点 `VERIFIED` 不等于独立端到端证书。

## 新运行门

1. 隔离新构建只改本节明确的 `x4` unsafe 下界及必要的进程/记录行为；不覆写旧源码、旧日志。
2. 一周期 smoke 只验证端口、控制器和流管管线；前 `0.1 s` 的安全不算 `T=3` 的 `VERIFIED`。
3. 完整运行前重新查 GPU、CPU 和端口；记录 30 次 RPC、全盒 60 段、性质及 wall。只有完整 `T=3` 后填新主结果，失败保留实际前缀。

## 2026-10-01 新运行结果

在独立新目录，四种方法均先过一周期 smoke，随后分别完整运行全初盒 `T=3`。完整结果如下，均为**单次**样本：

| 方法 | 完整期/小段 | 进程 wall | 正确官方 unsafe 性质 | 末端六维宽度 |
| --- | --- | ---: | --- | --- |
| 原生 Flow* | 30/30、60/60，30 RPC | 6.281498344 s | 修正 checker `VERIFIED`；保存六维 tube 盒 60/60 与 unsafe 盒分离 | `[.004194134463,.005980153698,.006193976913,.032292131362,.017635942833,.018637403948]` |
| Huan | 30/30、60/60 accepted | 7.094217066 s | 作者 checker 无失败输出；保存六维 tube 盒 60/60 分离 | `[.004174214013,.005935658769,.006172913343,.032070326164,.017361515808,.018479363610]` |
| Xiangru | 30/30、60/60 accepted | 6.933375641 s | 同 Huan；两方 `ranges.bin` 直接字节相同 | 同 Huan |
| 我们 P3 | 30/30、60/60 accepted | 13.017036134 s | 作者 checker 无失败输出；保存六维 tube 盒 60/60 分离 | `[.004078890759,.005800308639,.005992562328,.031363063673,.017096770656,.018023016452]` |

完整局部证据与范围数据在[四方综合摘要](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp26_20261001/ATTITUDE_AVOID_4METHODS_SUMMARY.md)、[原生摘要](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp26_20261001/native_attitude_avoid_full30_001/SUMMARY.md)、[作者方法摘要](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp26_20261001/author_attitude_avoid_v1/SUMMARY.md)及[P3 摘要](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp26_20261001/p3_attitude_avoid_v1/SUMMARY.md)。[六态绝对 lo/hi/width 对照 CSV](evidence/archcomp26_attitude_avoid_4methods_abs_bounds_20261001.csv)直接来自四份完整二进制范围记录；原生和 GPU 的范围观察器不同，Huan 与 Xiangru 记录直接字节相同。旧空集 checker 的 `VERIFIED` 与这四份修正后完整结果不可混同。四方同一台机器但使用不同启动时段；单次时间不能排名，作者 Huan/Xiangru 也不是两套独立 NNCS 证明。四方数值链尚无独立端到端浮点 NNCS 正确性证书。
