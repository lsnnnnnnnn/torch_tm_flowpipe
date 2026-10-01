# ARCH-COMP 2026 NAV 作者执行合同补充审计

审计日期：2026-10-02。本文件补充[先前的固定官方来源审计](ARCHCOMP26_NAV_PAPER_SOURCE_CONTRACT_20261001.md)，记录后来找到的控制器作者训练与执行源码、两次仅用 CPU 的首周期中心点诊断，以及两次 GPU 首盒首周期 smoke。没有重启历史长实验；新运行没有覆盖完整初始集合或完整时域性质。

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

这套合同有明确的官方 plant 与作者闭环入口依据，可开始**新 run ID 的单分块完整时域 smoke**，但它只有单分块覆盖，不能作为完整 NAV 格成绩。若按保存分块顺序选择首盒，标准首盒是 `x∈[2.9,2.905], y∈[2.9,2.9125]`，robust 首盒是 `x,y∈[2.9,2.94]`，两者均有 `v=θ=0`。需要在实际执行入口中检查：原始固定官方网络文件、输入/输出原序、600/600 ODE 子步、全时域障碍、终点目标、完成时域 guard、数值 accepted/rejected 状态与原始区间轨迹。现有 Huan/Xiangru/原生旧入口可作为实现参考；PyTorch/GPU NAV 新入口尚需适配并审计。下述 GPU 作业只执行到 `t=0.2`，尚未启动单分块完整 6 秒作业。

## CPU 首周期中心点诊断

[诊断脚本](../tools/archcomp26_nav_author_cpu_preflight_nohash.py)在服务器的现有 `nncs_env` 内，用纯 CPU 对固定官方网络的 `MatMul/Add/Relu/Tanh` 算子链求值。独立的 ONNX ReferenceEvaluator 对两次控制输出均给出逐值相同结果。随后用四状态作者方程、常值控制及 RK4 以 0.01 秒步长积分一个 0.2 秒周期；将步长折半至 0.005 秒所得终点最大绝对差约 `2.6×10⁻¹²`。这些是数值一致性检查，不是区间包络。

| 固定官方控制器 | 新 run ID / 证据 | 中心点 `x(0)` | `u(0)` | `x(0.2)` |
| --- | --- | --- | --- | --- |
| 标准 point | [`nav_author_cpu_step1_standard_001`](evidence/results/archcomp26_20261001/nav_author_cpu_step1_standard_001/RESULT.json)；[ONNX 参考求值](evidence/results/archcomp26_20261001/nav_author_cpu_step1_standard_001/REFERENCE_CHECK.json) | `(3,3,0,0)` | `(-1, 0.9973941445350647)` | `(2.9801985196229714, 2.9973508507508595, -0.2, 0.1994788289070130)` |
| robust set | [`nav_author_cpu_step1_robust_001`](evidence/results/archcomp26_20261001/nav_author_cpu_step1_robust_001/RESULT.json)；[ONNX 参考求值](evidence/results/archcomp26_20261001/nav_author_cpu_step1_robust_001/REFERENCE_CHECK.json) | `(3,3,0,0)` | `(-0.9905992150306702, 0.9934086203575134)` | `(2.9803831040335997, 2.997386158101895, -0.1981198430061341, 0.1986817240715027)` |

两份 `RESULT.json` 都明确标为 `complete_horizon=false`、`initial_set_covered=false`、`property_evaluated=false`、`formal_certificate=false`。不能从中心点轨迹推出避障、到达、任意一个分块的安全性，或四方法成绩。

## Huan 首初盒首控制周期有限 smoke

[独立运行档案与命令](evidence/results/archcomp26_20261001/nav_author_smoke_v1/README.md)记录了固定官方 ONNX、原始历史初盒台账、生成配置、GPU2/CPU10–13、300 秒上限与端口情况。两个新 run ID 均只选各自初始分区表的第 0 盒、首个 `0.2 s` 控制周期；端点目标只在 `t=6` 成立，因此短 smoke 不检查目标。CROWN 在进程内执行，没有 TCP RPC 端口。

| 实例 | 新 run ID | 20 个 `0.01 s` 子步 | 末端 x、y 包络 | 进程 wall |
| --- | --- | --- | --- | --- |
| standard，point ONNX | [`nav_author_standard_huan_smoke1_001`](evidence/results/archcomp26_20261001/nav_author_standard_huan_smoke1_001/RESULT.json) | 20/20 accepted；20 个保存 tube 均与障碍盒分离 | x `[2.8801981464,2.8851982812]`，y `[2.8973532532,2.9098525077]` | 4.1424 s |
| robust，set ONNX | [`nav_author_robust_huan_smoke1_001`](evidence/results/archcomp26_20261001/nav_author_robust_huan_smoke1_001/RESULT.json) | 20/20 accepted；20 个保存 tube 均与障碍盒分离 | x `[2.8803819925,2.9203823466]`，y `[2.8973937349,2.9373917875]` | 4.4177 s |

保存轨迹的[独立 standard 审计](evidence/results/archcomp26_20261001/nav_author_standard_huan_smoke1_001/INDEPENDENT_SAVED_RANGE_AUDIT.json)与[robust 审计](evidence/results/archcomp26_20261001/nav_author_robust_huan_smoke1_001/INDEPENDENT_SAVED_RANGE_AUDIT.json)逐行确认步号连续、每条区间有限且上下界有序、障碍投影分离。结果只支持**该两盒在已保存首周期 tube 上的数值观察**；完整 640/25 分块、30 周期与最终到达性质都未由新作业覆盖。

## 旧 Huan 全程记录与当前可执行合同逐项比对

2026-10-02 只读重查服务器旧运行的 `contracts/nav_{standard,robust}.yaml`、两份 `*_boxes.json`、旧 ONNX 文件与固定官方 2026 ONNX 副本；对模型做直接逐字节比较，**没有计算文件摘要**。旧进程参数、600 步接受状态与作者 checker 输出取自本地保存的原始 `result.json`、`*_watch/process.json`、`*_watch/stdout.log`。官方要求由上文 README、`dynamics.m` 与作者闭环例界定；不能把下表中的数值调参误称为官方强制值。

| 字段 | 旧 standard / robust Huan 全程记录 | 对当前固定官方加作者可执行 profile 的判定 |
| --- | --- | --- |
| 模型文件 | 旧路径分别为 `ARCH-COMP2024/benchmarks/NAV/networks/nn-nav-point.onnx`、`nn-nav-set.onnx`；各自与服务器 `nav_prep_001/official_nn-nav-{point,set}.onnx` **直接字节相同**。 | 同一固定 2026 point/set 模型内容；路径年份差异不改变模型。作者仓库 Git LFS 模型另未比较。 |
| 状态、网络输入及输出 | 旧 YAML 为七变量 `[x1,x2,x3,x4,t,u1,u2]`；网络取前四维原序，输出写入 `u1,u2`，无列置换，输出 scale=1、offset=0。 | 对应 `[x,y,speed,heading]` → `[speed_rate,heading_rate]`；与作者可执行训练/验证入口一致。论文文字的 `[x,y,heading,speed]` 是已注明的来源冲突。 |
| 动力学 | `x1'=x3 cos(x4), x2'=x3 sin(x4), x3'=u1, x4'=u2`；附加 `t'=1,u1'=u2'=0` 用于采样保持。 | 四物理态 RHS 与固定官方 `dynamics.m` 相同；附加三态不改变物理 plant。 |
| 初集与分区 | 两者均 `x1,x2∈[2.9,3.1],x3=x4=0`；保存分区台账分别恰有 `40×16=640`、`5×5=25` 个笛卡尔盒，坐标连续覆盖初集。 | 与官方初集及论文报告的 CROWN-Reach 分区相同。 |
| 采样、总时域 | 旧 YAML 均 `steps=30,step_size=.2`；旧进程各记录 600/600 个 ODE 小步。 | 与官方 `T=6`、0.2 秒采样一致。 |
| 性质 | 旧 YAML 全时障碍为 `x1,x2∈[1,2]`，末时目标为 `x1,x2∈[-.5,.5]`；旧 driver 的 unsafe group 在流管小步上判断，target group 在最终状态判断。 | 与固定官方性质相同；本轮 `t=.2` smoke 暂不检查只在 `t=6` 定义的终点目标。 |
| 数值设置和控制器传输 | 旧 Huan `h=.01`、order 4、cutoff `1e-6`、余项 `[-.1,.1]`、SR queue 1000；保存 argv 为 `sparse`、`strict`、CROWN `box`/`same-slope`、`rpc-float32`、`native` 输入布局。 | 本轮短 smoke 继承这些旧设置。它们是明确数值 profile，不是官方 plant/性质的额外规定。 |

旧 standard 原始 [result](<../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/nav_standard_huan/result.json>)、[watch 输出](<../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/nav_standard_huan_watch/stdout.log>)记录 640 盒 × 600 小步，384,000/384,000 接受、exit 0、`VERIFIED`，`T=6` 保存末端 hull 的 `(x1,x2)` 分别为 `[-0.0880373374786,-0.00771573664868]` 与 `[0.0889568833234,0.353429667069]`。旧 robust 原始 [result](<../../../../results/archcomp_review_20260923/evidence_v2/timing_v1/nav_robust_r2_huan/result.json>)、[watch 输出](<../../../../results/archcomp_review_20260923/evidence_v2/timing_v1/nav_robust_r2_huan_watch/stdout.log>)记录 25 盒 × 600 小步，15,000/15,000 接受、exit 0、`VERIFIED`，末端 `(x1,x2)` 为 `[0.107480570599,0.19778359055]` 与 `[-0.063191845023,-0.0483889391416]`。旧 standard 保存有范围文件，robust 本地镜像只有结果/输出而无逐小步范围；此处没有独立重算两条旧记录的全时避障。两条 `result.json` 均明示 `end_to_end_strict_certificate=false`。

**结论：** 在已查的模型、物理动力学、接口、初集、分区、采样、时域、性质和数值设置上，旧 Huan 两条完整运行与当前命名的作者可执行 profile **没有实质合同差异**。旧结果可作为该 profile 的历史全程证据按原始资格引用；不为填新矩阵重复启动 640/25 全盒 Huan 作业，也不把旧 14.72/11.66 秒视为本轮统一资源的正式排名时间。尚缺的是 robust 历史逐步范围的本地副本、旧两条运行的独立端到端浮点 NNCS 证书，以及其它方法在同一 profile 下的完整可比数据；这些缺口不等于需要重跑已完成的 Huan 全程。

成本评估：两次短 smoke 的进程时间包含 Python/CUDA/网络初始化，不可线性乘以 640/25 和 30 推算全盒用时。旧记录的 driver wall 分别约 14.72 s 和 11.66 s；它们可用于判断同一合同在历史环境可执行，不能由此推断当前完整作业时间或证明强度。本轮没有重复启动历史全时域实验。
