# ARCH-COMP 2026 NAV 作者执行合同补充审计

审计日期：2026-10-02。本文件补充[先前的固定官方来源审计](ARCHCOMP26_NAV_PAPER_SOURCE_CONTRACT_20261001.md)，记录后来找到的控制器作者训练与执行源码，以及两次仅用 CPU 的首周期中心点诊断。没有重启历史长实验；诊断没有覆盖初始集合，也没有验证时域性质。

## 新的一手来源与可确定的接口

| 来源 | 直接证据 | 可得结论 |
| --- | --- | --- |
| [2026 固定官方 NAV README](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/NAV/README.md) 与 [dynamics.m](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/NAV/dynamics.m) | `x1'=x3 cos(x4)`, `x2'=x3 sin(x4)`, `x3'=u1`, `x4'=u2`；0.2 秒控制周期、6 秒时域、两种同名网络及性质。 | 按可执行方程，`x3` 是速度，`x4` 是朝向；`u1` 是速度率，`u2` 是朝向率。 |
| [控制器作者的 NAV 训练脚本](https://github.com/ManuelWendl/VerifiablyRobustSetBasedRL/blob/master/code/scripts/benchmark_rl_agentDDPG_NavTask.m)（2026-10-02 读取） | 使用与官方相同的四维动力学；actor 为 `4→64→32→2`，输出 tanh；环境 `dt=.2`。 | 训练环境的状态与控制量顺序与官方 `dynamics.m` 对齐。脚本的 40 步是训练 episode 上限，不是 2026 比赛的 30 周期。 |
| [作者 CORA NAV 验证例](https://github.com/ManuelWendl/VerifiablyRobustSetBasedRL/blob/master/code/cora/examples/contDynamics/neurNetContrSys/example_neuralNet_reach_11_NAV.m)（2026-10-02 读取） | 同一动力学直接加载 `nn-nav-point.onnx` 或 `nn-nav-set.onnx`，构造 `neurNetContrSys(sys, nn, 0.2)`；`tFinal=6`，初始盒、障碍与终点目标也相同。 | 作者提供了两个文件名对应的闭环可执行入口，没有在网络和 plant 间交换第 3、4 列或第 1、2 输出。 |
| [作者训练循环](https://github.com/ManuelWendl/VerifiablyRobustSetBasedRL/blob/master/code/cora/nn/rl/agents/%40agentRL/train.m)及[环境步进](https://github.com/ManuelWendl/VerifiablyRobustSetBasedRL/blob/master/code/cora/nn/rl/%40ctrlEnvironment/step.m)（2026-10-02 读取） | actor 直接读取 `observation`；环境将模拟状态的前四列原序返回。 | 在已查看的训练/环境路径上，没有状态列重排。 |
| [2026 AINNCS 论文](https://easychair.org/publications/paper/GsKW/download) §3.11、[已有论文页审计](ARCHCOMP26_NAV_PAPER_SOURCE_CONTRACT_20261001.md) | 文字按 `(x,y,θ,v)` 介绍状态，却给出 `(v cos θ,v sin θ,u1,u2)` 的导数列。论文还写两层各 64 个隐藏单元。 | 若把文字顺序当作向量顺序，后两项语义会与官方及作者源码冲突；提供的 ONNX 实际为 64、32 两层。论文没有逐行标出导数列对应的状态名。 |

固定官方 README 指明 `nn-nav-point.onnx` 为标准点训练控制器，`nn-nav-set.onnx` 为集合训练的 robust 控制器。这里的 **robust 是控制器训练方式**：固定官方 plant 方程没有外加扰动项。作者训练脚本中的 `noise=.1` 是训练输入扰动设置，不应移入比赛 plant 或初始集合。

作者仓库里的两个 ONNX 是 Git LFS 文件；本审计只读取作者源码，**没有取得并逐字节比较**作者仓库模型内容与固定官方模型。因此“作者例使用同名文件及相同结构”是接口证据，不是两个仓库二进制内容相同的证明。新实验须直接使用固定官方 2026 提供的 point/set ONNX；不依赖这种未做的跨仓库等同性判断。

## 建议冻结的新 NAV 执行合同

对 `nav-standard` 和 `nav-robust`，建议主表明确采用**固定官方可执行方程 + 控制器作者可执行接口**：

1. 网络输入原序 `[x, y, v, θ]`，网络输出原序 `[u_v, u_θ]`；`x'=v cos θ, y'=v sin θ, v'=u_v, θ'=u_θ`。控制量每 0.2 秒更新一次并保持，时域 30 周期至 6 秒。
2. 初始集合 `x,y∈[2.9,3.1]`，`v=θ=0`。整个 `[0,6]` 避开闭盒 `x,y∈[1,2]`；在 `t=6` 落入闭盒 `x,y∈[-0.5,0.5]`。第 3、4 状态在目标与障碍中不受限。
3. 标准配置用固定官方 `nn-nav-point.onnx`，robust 用固定官方 `nn-nav-set.onnx`。参考 CROWN-Reach 的已保存分块：标准 `40×16×1×1=640`，robust `5×5×1×1=25`；全盒最终成绩必须覆盖各自所有分块。已保存入口均用 Flow* 固定 ODE 步长 0.01、阶数 4，不能把历史性能直接写进新的结果格。
4. 论文文字的 `(x,y,θ,v)` 顺序与 `64/64` 层宽作为来源冲突单列，不用它们悄悄重排固定官方网络输入，也不改写官方 ONNX。

这套合同有明确的官方 plant 与作者闭环入口依据，可开始**新 run ID 的单分块完整时域 smoke**，但它只有单分块覆盖，不能作为完整 NAV 格成绩。若按保存分块顺序选择首盒，标准首盒是 `x∈[2.9,2.905], y∈[2.9,2.9125]`，robust 首盒是 `x,y∈[2.9,2.94]`，两者均有 `v=θ=0`。需要在实际执行入口中检查：原始固定官方网络文件、输入/输出原序、600/600 ODE 子步、全时域障碍、终点目标、完成时域 guard、数值 accepted/rejected 状态与原始区间轨迹。现有 Huan/Xiangru/原生旧入口可作为实现参考；PyTorch/GPU NAV 新入口尚需适配并审计。当前未占用 GPU，也未启动单分块完整时域作业。

## CPU 首周期中心点诊断

[诊断脚本](../tools/archcomp26_nav_author_cpu_preflight_nohash.py)在服务器的现有 `nncs_env` 内，用纯 CPU 对固定官方网络的 `MatMul/Add/Relu/Tanh` 算子链求值。独立的 ONNX ReferenceEvaluator 对两次控制输出均给出逐值相同结果。随后用四状态作者方程、常值控制及 RK4 以 0.01 秒步长积分一个 0.2 秒周期；将步长折半至 0.005 秒所得终点最大绝对差约 `2.6×10⁻¹²`。这些是数值一致性检查，不是区间包络。

| 固定官方控制器 | 新 run ID / 证据 | 中心点 `x(0)` | `u(0)` | `x(0.2)` |
| --- | --- | --- | --- | --- |
| 标准 point | [`nav_author_cpu_step1_standard_001`](evidence/results/archcomp26_20261001/nav_author_cpu_step1_standard_001/RESULT.json)；[ONNX 参考求值](evidence/results/archcomp26_20261001/nav_author_cpu_step1_standard_001/REFERENCE_CHECK.json) | `(3,3,0,0)` | `(-1, 0.9973941445350647)` | `(2.9801985196229714, 2.9973508507508595, -0.2, 0.1994788289070130)` |
| robust set | [`nav_author_cpu_step1_robust_001`](evidence/results/archcomp26_20261001/nav_author_cpu_step1_robust_001/RESULT.json)；[ONNX 参考求值](evidence/results/archcomp26_20261001/nav_author_cpu_step1_robust_001/REFERENCE_CHECK.json) | `(3,3,0,0)` | `(-0.9905992150306702, 0.9934086203575134)` | `(2.9803831040335997, 2.997386158101895, -0.1981198430061341, 0.1986817240715027)` |

两份 `RESULT.json` 都明确标为 `complete_horizon=false`、`initial_set_covered=false`、`property_evaluated=false`、`formal_certificate=false`。不能从中心点轨迹推出避障、到达、任意一个分块的安全性，或四方法成绩。
