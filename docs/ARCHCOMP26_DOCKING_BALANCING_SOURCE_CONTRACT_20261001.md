# ARCH-COMP 2026 Docking 与 Balancing/CartPole 来源合同审计

审计日期：2026-10-01。本轮只读官方论文、固定官方仓库文本/ONNX、仓库所附 Docking 说明 PDF 和已保存参与者入口；未启动数值实验、复用旧成绩或做内容摘要校验。本文是执行前合同，不是新的四方结果；共享 attempt/matrix 未修改。

## 一手来源和名称映射

- [2026 AINNCS 报告](https://easychair.org/publications/paper/GsKW/download)：Docking §3.10 印刷页 98（PDF 14）；CartPole §3.12 印刷页 99–100（PDF 15–16）。公式页已按 PDF 版面查看。本地阅读副本为 `output/flowstar_latest_20260930/reference/ARCH_COMP26_AINNCS.pdf`。
- 固定官方提交 `d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26` 的[顶层 README](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/README.md)将 `Docking/constraint` 指向 `benchmarks/Docking`，把 `Balancing/reach` 指向 `benchmarks/CartPole`；后者是同一实例的表名与目录名，并非另一个模型。
- Docking：[规格](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Docking/specification.txt)、[动力学](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Docking/dynamics.m)、[ONNX](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Docking/model.onnx)、[仓库附带的九页说明 PDF](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Docking/spacecraft_docking_benchmark_description.pdf)。说明 PDF 的页 1–4 给出状态、控制、预/后处理、采样语义，页 5 给出验证初盒；它虽生成于 2022 年，确实是固定 2026 仓库所附的文件。固定 Docking 目录仅有上述五个文件，没有执行脚本。
- CartPole：[规格](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/CartPole/specifications.txt)、[动力学](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/CartPole/dynamics.m)、[ONNX](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/CartPole/model.onnx)、[目录 README](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/CartPole/README.md)。固定目录只有这四个文件；README 明说原提案为 2024 CartPole，但文件仍被 2026 顶层表引用。

## Docking：可冻结的四态连续安全合同

论文印刷页 98、固定 MATLAB 与说明 PDF 页 1、3–4 互相支持：物理状态顺序为 `(sx,sy,vx,vy)`，网络原始输入就是这四态，网络输出 `(Fx,Fy)`；`m=12, n=0.001027`。取两个控制在一周期内常值，动力学逐式为：

```text
sx' = vx
sy' = vy
vx' = 2*n*vy + 3*n^2*sx + Fx/m
vy' = -2*n*vx + Fy/m
```

完整初盒是 `[70,106] × [70,106] × [-0.28,0.28] × [-0.28,0.28]`。固定规格说控制周期为 **1 s**；论文页 98 说验证 **40 s**，所以新连续 sample-and-hold profile 有周期 `k=0..39`、末时刻 `t=40`。纸面性质及规格相同，对**整个** `t∈[0,40]` 检查：

```text
sqrt(vx^2 + vy^2) <= 0.2 + 0.002054*sqrt(sx^2 + sy^2)
```

说明 PDF 页 1 的 docking-success 距离条件及页 8 的 RL 训练终止条件**不是**此 2026 验证实例的性质；不能另加 `||r||≤0.5` 的终点目标。1 s 是 NN 更新周期，不是 Flow*/GPU 内部积分步长。

固定官方 ONNX 已在服务器内存中解析且通过图结构检查：float32 `[N,4]→[N,2]`，输入名 `input_1`，输出名 `postprocess_filter_std_clip`。图序列为 `4→4` 预处理 MatMul、`4→256→256→4` 带两层 hidden Tanh 的主体、`4→2` 后处理 MatMul、最终 Tanh；预处理矩阵是 `diag(0.01,0.01,2,2)`，后处理矩阵选主体输出的第 1、3 列。说明 PDF 页 3 明说这些变换已在网络中，输入就是原始状态，输出就是物理控制；外部若再归一化或裁剪会改变合同。说明 PDF 页 4 也把输出力范围列为 `[-1,1] N`，与末端 Tanh 对应。固定官方 ONNX 与服务器旧 `/srv/local/shengenli/CROWN-Reach/ARCH-COMP2024/benchmarks/Docking/model.onnx` 已直接逐字节比较相同；这只确立模型文件对应，不使旧数值结果自动有效。

**可执行的新比较定义**：四方法从同一完整初盒出发，在每个整数 `t=k` 用该周期起点可达集的原始四态送上述固定 ONNX，得到两路控制包络/关系，保持至 `k+1`；各自用明确记录的数值方法积分这四条连续 ODE，保存每段全时 tube 与端点。在每段 tube 上证明径向速度不等式，`t=0` 与 `t=40` 均覆盖。若分箱，先保存覆盖完整初盒、不重叠遗漏的共同 ledger；**单个完整盒**是无需额外分区决定的基准 profile。内部 ODE 步长、Taylor 阶数、NN bounder、CPU/GPU、停止/计时规则是方法设置，须逐入口记录；不能把 1 s 误作其共同 ODE 步长。对性质判定，可对耦合 TM/支持函数直接界定 `sqrt(vx²+vy²)−0.2−0.002054 sqrt(sx²+sy²)`；单纯轴对齐盒代入也可作保守预检，但若不定不能据此断言真实违例。

**实际启动缺件**：冻结旧 CROWN-Reach 14 例源码树没有 Docking 的 C++/RPC 提交入口；旧 GPU 树只在通用 `crown_reach.py` 的说明中提到 Docking 耦合性质及可选细化检查，没有已核实的 Docking 实例配置或四方运行。四方法仍需新隔离入口/模型路径、实际 ONNX 转换与两输出注入的静态/短程预检、完整初盒 ledger、全时间管上的非线性性质 checker、每期 accepted 与早停证据，以及数值/资源预算。本审计不把通用 checker 的文字说明当成已执行的 Docking 性质证明。

## Balancing/CartPole：两个不能合并的控制/时间边界

报告页 99 的四态顺序是 `(x1,x2,x3,x4)` = 车位置、车速度、杆角度、角速度，初盒为 `[-0.1,0.1]×[-0.05,0.05]×[-0.1,0.1]×[-0.05,0.05]`，控制间隔 `0.02 s`。固定规格也给出这个完整盒、`T=10 s` 与周期 `0.02 s`，即 **500** 周期。固定 MATLAB 四条 ODE 为：

```text
x1' = x2
x2' = 2*f
x3' = x4
x4' = (0.08*0.41*(9.8*sin(x3)-2*f*cos(x3))-0.0021*x4)/0.0105
```

报告页 99 式 (17)、(19)把控制器写成 **五个特征** `f(x1,x2,sin(x3),cos(x3),x4)`；但固定 MATLAB 文件直接写明控制器取 `(x1,x2,x3,x4)`、输出可直接用。固定 ONNX 实际 float32 `[1,4]→[1,1]`，两层 64 宽 Tanh 和标量 Tanh 输出，图内没有把四原态扩成五特征的 `sin/cos` 结点。历史 `CartPole/crown.py:35–45` 也用 `(-1,4)` 输入，`balancing.cpp:18–30,139–153` 把四态原样送 NN。固定官方 ONNX 与服务器旧 `/srv/local/shengenli/CROWN-Reach/ARCH-COMP2024/benchmarks/CartPole/model.onnx` 已直接逐字节比较相同。**因此论文五特征函数不能直接用这个四输入模型执行**；没有来源证明一种外部五到四映射，应保留为 paper-feature-5 冲突，不猜映射。

报告页 100 明确性质为闭时间窗 `∀t∈[8,10]`，`x1,x3,x4∈[-0.001,0.001]`；固定规格则写 `For t > 8.0 s`，在 `T=10` 下是 `(8,10]`，不含精确 `t=8`。两者只有边界点差异，但严格验证不应偷偷合并。**可执行的仓库四原态 profile** 可明确选择固定 MATLAB/ONNX 四输入、上述 ODE 与初盒，周期起点原始四态送 NN、输出在 `0.02 s` 内保持，500 周期；性质按固定规格 `8<t≤10` 检查三态。另设纸面闭窗 `[8,10]` 作为独立 checker profile，且标明它仍采用仓库四输入网络，而非已经解决论文五特征控制器。若要声称完整复现论文版本，仍需作者给出五特征与四输入 ONNX 的权威映射或对应五输入控制器，以及边界点选用依据。

旧 `balancing.cpp` 在历史小初盒 `[-0.0375,-0.03125]×[-0.015625,-0.0125]×[-0.00625,0]×[-0.007375,-0.00625]` 只设 `steps=50`，到 `T=1`；其 `safeSet` 构造被注释，`iter>400` 的触发条件在 50 期内不可达。旧 GPU `arch_cartpole_official_f32.yaml` 同为该小盒、50 周期，目标约束只在终点使用。它们都不是 2026 完整盒、500 周期、`8<t≤10` 或 `[8,10]` 的结果。旧 MatLab/Flow* 算法参数仅是历史方法参数，不可直接写成官方四方要求。

**实际启动缺件**：四个新入口需固定 2026 四输入 ONNX 与原始四态映射、完整初盒共同覆盖 ledger、500 个周期及两种性质窗口之一的连续 tube checker；尤其 `t=8` 所在边界必须按所命名 profile 明确处理。还需各方法的模型加载/注入、内步长、资源预算和失败保留方案。若主表要求严格论文五特征语义，目前缺权威模型/映射，不能启动该身份；仓库四原态 profile 则是可单独命名的新比较合同，不能冒充五特征论文执行。
