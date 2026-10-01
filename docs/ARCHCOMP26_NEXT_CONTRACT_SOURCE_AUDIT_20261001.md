# ARCH-COMP 2026：Single Pendulum、Airplane 与 Double Pendulum more-robust 合同审计

审计日期：2026-10-01。结论只涉及**新四方比较的合同定义**；本次审计没有启动数值任务、重跑旧实验、修改冻结证据或执行哈希校验。本文件不把旧结果登记为 2026 四方结果。仓库原有 `benchmarks/archcomp26/manifest.json` 与执行矩阵仍是恢复前快照；其摘要绑定 launcher 仍不可用于用户要求的无哈希新运行。用户已授权在独立目录恢复实验，实时进度另见 [接续记录](GOAL_EXECUTION_PROGRESS_20261001.md)。

## 可定位的一手来源

- [ARCH-COMP26 AINNCS 报告](https://easychair.org/publications/paper/GsKW/download)，印刷页 89（PDF 5：所有离散模型采用 forward Euler）、93（PDF 9：Single Pendulum）、94–95（PDF 10–11：Double Pendulum）、95–96（PDF 11–12：Airplane）、111（PDF 27：CROWN-Reach 对 more-robust 的角点反例说明）。本地阅读副本为 `output/flowstar_latest_20260930/reference/ARCH_COMP26_AINNCS.pdf`。公式所在 PDF 页 9、11、12 已按版面查看，避免把文本提取的上下标及分式误读成另一个式子。
- 固定的 [2026 官方 benchmark README](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/README.md)，以及下文每项指向同一提交 `d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26` 的具体规格和动力学文件。本审计直接读取了这些文本文件及三个 benchmark 目录的文件清单；没有把当前浮动分支当成来源。
- 历史执行源码均为本地冻结副本 `results/archcomp_review_20260923/sources/native/submit/CROWN-Reach/archcomp/` 下的文件。它们展示旧 CROWN/Flow* 的变量、控制更新和初值选择，但使用 `ARCH-COMP2024` 控制器路径，不能仅凭同名文件推定为 2026 控制器。旧我方配置在本仓库 `research/gpu_verified_20260930/report/configs/` 下，同样只是回归材料。

## 1. Single Pendulum：两物理态与第三导数

[报告 §3.5，页 93](https://easychair.org/publications/paper/GsKW/download) 明确定义 `x1=theta, x2=theta_dot`，取 `m=L=0.5, c=0, g=1` 后两条物理方程是

```text
x1' = x2
x2' = 2 sin(x1) + 8 T
```

[官方 `specifications.txt`](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Single_Pendulum/specifications.txt) 给出完整初始盒 `[1,1.175] × [0,0.2]`、控制周期 `0.05 s`，以及 `0.5 ≤ t ≤ 1` 上 `x1∈[0,1]`；离散版本的对应索引是 `k=10..20`。报告方程与 [官方 `dynamics_sp.m`](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Single_Pendulum/dynamics_sp.m) 的前两项逐式一致。MATLAB 文件还返回 `dx(3)=1`（前一行注释掉的是 `dx(3)=20`），但规格没有第三初值，报告没有定义第三物理态，且前两条导数不引用 `x3`。

冻结历史 `SinglePendulum/single_pendulum.cpp:18–30,41–49` 使用四个 Flow* 变量 `(x1,x2,t,u)`，取 `t(0)=u(0)=0`、`t'=1,u'=0`；`crown.py:35–45` 将网络输入形状固定为 `(1,2)`；C++ `:63–109` 仅将 `x1,x2` 送入 NN，先更新 `u`，再推进 `0.05 s`。这支持一个**明确命名的新 paper-two-state profile**：物理系统只有 `(x1,x2)`，控制器在每周期起点读这两态并在周期内保持输出；实现如需时钟，令辅助 `t(0)=0,t'=1`，时钟不进入 NN、物理宽度或物理初始盒。它不是对官方 MATLAB 第三状态初值的推断。不能把三态 MATLAB 扩展与两态论文表述不加标记地合并为“原提交执行完全相同”。

历史 C++ `:122–133` 把 `-t+0.5, x1, -x1+1` 同列传给一次 `unsafetyChecking`。本地冻结包没有该 Flow* 函数的实现，不能由这些字符串独自认定它精确实现了 `x1<0 OR x1>1` 的时间窗违例。新四方 checker 必须独立明确：在 `t∈[0.5,1]` 的**全部连续时间**证明 `0≤x1≤1`，任何已验证违反轨迹另列 falsification；恰好 `t=0.5` 和 `t=1` 都包括。旧 20 周期结果及其 verdict 不自动升格。

**尚缺材料**：固定 2026 ONNX 在目标执行环境中的可读取副本及其实际输入/输出和预处理记录；四个新入口各自证明只把两物理态送 NN、使用同一完整初始盒与周期起点更新；新 checker 对上述闭时间窗和两侧违反的实现/收据。若要宣称复现某个 2026 提交工具的三态 MATLAB 运行，还须取得它如何给第三态赋初值、何时重置时钟、是否把它传给控制器/性质检查器的实际执行源码或记录；现有官方文件没有这些字段。

## 2. Airplane：连续 2 秒与离散 20 次转移

[报告 §3.7，页 95–96](https://easychair.org/publications/paper/GsKW/download) 和 [官方实例规格](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Airplane/specifications.txt) 一致：物理顺序为 `(sx,sy,sz,vx,vy,vz,phi,theta,psi,r,p,q)`，`sx=sy=sz=r=p=q=0`，其余六态各在 `[0,1]`；控制间隔 `0.1 s`，20 步等于 **2 s**；安全性质是 `sy,phi,theta,psi∈[-1,1]`。连续性质覆盖闭区间 `t∈[0,2]`，离散性质覆盖 `k=0..20`。固定 [顶层 README](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/README.md) 的 continuous `[0,20]` 与报告及实例文件冲突，应作为冲突元数据记录，不能拿来延长运行到 20 秒。

取 `s=sin,c=cos`，令 `u=(Fx,Fy,Fz,Mx,My,Mz)`。下列逐式抄录把 [官方 `dynamics.m`](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Airplane/dynamics.m) 的 12 个输出位置与报告式 (10)–(12) 及历史 `Airplane/airplane.cpp:20–53` 对齐；下式的 `vx,vy,vz` 对应旧 C++ 的 `u,v,w`，不能把 `u` 同时当速度和六维控制向量。

```text
sx'    = cψ cθ vx + (-sψ cφ + cψ sθ sφ) vy + (sψ sφ + cψ sθ cφ) vz
sy'    = sψ cθ vx + ( cψ cφ + sψ sθ sφ) vy + (-cψ sφ + sψ sθ cφ) vz
sz'    = -sθ vx + cθ sφ vy + cθ cφ vz
vx'    = -sθ + Fx - q vz + r vy
vy'    =  cθ sφ + Fy - r vx + p vz
vz'    =  cθ cφ + Fz - p vy + q vx
phi'   = p + tanθ(sφ q + cφ r)
theta' = cφ q - sφ r
psi'   = (sφ q + cφ r)/cθ
r'     = Mz
p'     = Mx
q'     = My
```

其中参数取报告给定的 `m=Ix=Iy=Iz=g=1,Ixz=0`。历史 C++ 前 12 条与 MATLAB 的变量映射和输出顺序一致，`Airplane/crown.py:35–45` 显示旧 NN 接收 12 态、输出六控制，`airplane.cpp:118–169` 显示周期起点 NN、保持控制并推进 `0.1 s`。然而历史 C++ `:64–72` 只用 `vx=vy=vz=1,phi=theta=psi=0.9` 的**单点**，旧我方 `arch_airplane_boxes.json` 也只有该单点，不能把旧范围、宽度或时间当作官方六维全盒结果。单点若配对正确控制器和可信轨迹，可成为全盒的反例见证；旧材料没有证明其 2024 控制器等于固定 2026 控制器，故旧 verdict 仍不能移入新格。

离散公式在报告印刷页 89 已给出 forward Euler 总规则，连同上述明确的 `f` 和 `Δ=0.1`，数学上可以写成 `x[k+1]=x[k]+0.1 f(x[k],u[k])`。但报告该处简写为 `f(x)`，固定官方 `Airplane` 目录只列四种控制器格式、`dynamics.m` 和 `specifications.txt`，**没有离散转移程序**。从历史连续 C++ 推出 `u[k]=NN(x[k])` 再做 Euler，是可显式选择的*新比较约定*，不是已核实的 2026 离散提交运行顺序。连续 Flow* ODE 积分不能填离散格；离散格的 ODE 步长、Taylor 阶数和余项字段应记为不适用。

**尚缺材料**：连续四方格需要固定 2026 控制器在执行环境中的模型及 I/O/预处理记录、覆盖全部 `[0,1]^6` 的共同分区与逐盒 ledger、连续全时域 checker 和四个周期起点 NN/保持控制入口。离散格若要宣称“复现官方 2026 执行”，还缺实际提交的离散转移/控制先后顺序源码或同等权威记录；若只建明确命名的新 Euler 比较格，则须先冻结 `u[k]=NN(x[k])`、`x[k+1]` 更新顺序、在 `k=0..20` 检查性质的可执行实现，并确保四方法都有真正的离散入口。

## 3. Double Pendulum more-robust：可优先冻结的下一数学合同

[报告页 94–95](https://easychair.org/publications/paper/GsKW/download) 与 [官方 `Specifications.txt`](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Double_Pendulum/Specifications.txt) 明确将 `controller_double_pendulum_more_robust.onnx` 配 **Specification 2**：物理顺序 `(theta1,theta2,theta1_dot,theta2_dot)`，参数 `m=L=0.5,c=0,g=1`，完整初始盒 `[1,1.3]^4`，控制周期 `0.02 s`，20 周期到 `0.4 s`，在**所有** `t∈[0,0.4]` 要求四态都在 `[-1.5,1.5]`。网络文件是固定官方目录中的独立 [`more_robust.onnx`](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Double_Pendulum/controller_double_pendulum_more_robust.onnx)，不能以 less-robust 文件替代。固定官方 [`more_robust.nnet`](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Double_Pendulum/controller_double_pendulum_more_robust.nnet) 的文本头声明四输入、两输出，但它不能替代对所选 ONNX 本身的图解析。离散 `k=0..20` 是报告提到的另一种模型版本，本条选连续 sample-and-hold profile。

设 `d=theta1-theta2, v1=theta1_dot, v2=theta2_dot`，`T1,T2` 为周期内保持的控制。固定 [官方 `dynamics_dp.m`](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Double_Pendulum/dynamics_dp.m) 的四条表达式可无歧义地合并为：

```text
A = 4*T1 + 2*sin(theta1) - v2^2*sin(d)/2
B = v1^2*sin(d) + 8*T2 + 2*sin(theta2)
    - cos(d)*(-v2^2*sin(d)/2 + 4*T1 + 2*sin(theta1))
theta1' = v1
theta2' = v2
v1' = A + cos(d)*B / (2*(cos(d)^2/2 - 1))
v2' = -B / (cos(d)^2/2 - 1)
```

冻结历史 `DoublePendulum/double_pendulum_more_robust.cpp:18–36` 的四条物理导数在 `th1,th2,u1,u2 ↔ theta1,theta2,v1,v2` 改名后逐式相同，且 `:80–131` 是 NN 先于 `0.02 s` plant advance。它额外存 `t'=1,T1'=T2'=0`，这些是实现辅助变量。关键不匹配位于历史 `:47–50`：四态初值全是 `[1.3,1.3]`；历史 `crown_more_robust.py:35–36` 还选择 `controller_double_pendulum_less_robust.onnx`。旧我方 `arch_double_pendulum_more_robust.yaml:13–40,62` 和相邻 boxes 文件复制了该角点与错误网络。因此旧时间、宽度、结论**没有 more-robust 2026 合同身份**。报告印刷页 111 说 CROWN-Reach 用角点 `(1.3,1.3,1.3,1.3)` falsify more-robust；这说明角点可作为正确网络下的反例候选，不能证明冻结旧代码使用了正确网络，也不能把角点当成全初始盒的宽度实验。

**推荐决定**：在本次三组候选中，先将本条冻结为下一份**新四方共享数学/控制合同**。它的论文、实例文件和动力学在关键字段上直接吻合，且有精确命名的官方 controller；旧错配可以通过全新配置而非修补冻结证据来隔离。建议的新比较 profile 如下，所有新增字段都作为新实验选择记录，不伪称为官方 more-robust 源码规定：

| 字段 | 新四方共享定义 |
| --- | --- |
| Plant 与 NN | 上述四态连续 ODE；固定 2026 `controller_double_pendulum_more_robust.onnx`，按 `(theta1,theta2,v1,v2)` 输入、`(T1,T2)` 输出。实际 ONNX 图的输入/输出及任何转换仍须运行前读取确认。 |
| 初始集与分区 | **完整** `[1,1.3]^4`，建议复用历史 less-robust C++ `double_pendulum_less_robust.cpp:77–99` 的轴向 `5×5×3×3=225` 网格：前两轴切点 `1+3i/50, i=0..5`，后两轴切点 `1+i/10, i=0..3`。四方共用同一份新导出的二进制端点/盒序 ledger；此分区是显式比较选择。 |
| 控制与终点 | 在 `t_j=j/50, j=0..19` 的每周期起点，用该周期初始可达集算一次逻辑控制更新，输出保持到 `t_{j+1}`；20 次更新，目标终点 `t_20=0.4 s`。每个方法实际 backend 调用数可不同，须另记。 |
| 性质与宽度观察 | 对每一盒每一段的完整时间管检查四态 `[-1.5,1.5]^4`；共享端点网格为 `j=0..20`，段管网格为 `j=0..19`。物理宽度只列四态，分别报告 endpoint 与 segment tube；验证须覆盖全盒全程。 |
| 停止条件 | 一个属于全初始盒、使用正确 controller 的可核反例可报告 `falsified/early_stopped`；其时间和共同有效前缀与全程完成分开，不能作为全盒宽度或四方法全时长速度排名。 |

旧 native 的 `0.005` 内部 ODE 步、order 4、cutoff 与 SR queue 是**方法设置**，不纳入共享数学合同，也不默认为其他三方法的算法选择。

**启动前尚缺的具体材料**：

1. 固定官方 more-robust ONNX 在目标环境中可读取的副本、实际四输入/两输出及任何转换/预处理记录；四个方法的命令都必须指向它。现有旧脚本和 YAML 指向 less-robust，不能直接复用。
2. 将上表建议的 225 盒网格正式写入新实例合同，导出全部盒的统一实际端点及顺序 ledger，并在四个方法执行前核对各入口使用该同一份清单。历史 more-robust 只有一个角点。
3. 每个方法独立的新入口/执行计划：周期起点 4 态输入、两输出 `(T1,T2)` 注入、周期内保持、20 个 `0.02 s` 周期、全时间管的八侧闭区间检查；各自数值设置、实际 NN 调用、资源与计时边界单独冻结。旧 native 的 `0.005` 内部 ODE 步、order 4、cutoff 和 SR queue 是方法设置，不能冒充官方要求四方相同算法。
4. 对“验证”保留全盒全时域证明；对“证伪”保存属于全盒的具体初值、所用**正确** controller、覆盖越界时刻的可检查轨迹/证书。没有证书的模拟值只能作为候选。

以上足以指定**下一合同的内容和缺件**，不足以声称这些实例的任一新方法已经完成。后续材料未到位时，`double-pendulum-more-robust`、Single Pendulum 和两条 Airplane 实例在旧矩阵中均保持 `unresolved/not_started`；新尝试另以独立记录为准。
