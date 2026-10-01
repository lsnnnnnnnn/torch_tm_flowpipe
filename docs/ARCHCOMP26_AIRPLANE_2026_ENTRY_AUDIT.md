# ARCH-COMP26 Airplane 连续/离散入口审计（2026-10-01）

本审计只准备新实验合同；**没有启动 Airplane 数值任务**，没有重跑旧作业，也没有计算或校验内容摘要。旧冻结矩阵与旧实验数据未改动。当前两格仍是未尝试，不能用旧单点结果填入。

## 固定来源与实际控制器

- 固定官方提交 `d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26` 的 [Airplane 目录](https://github.com/Kiguli/ARCH-COMP2026/tree/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Airplane) 只有四种控制器格式、`dynamics.m` 和 `specifications.txt`；递归树查询返回 `truncated=false`，**没有离散转移实现文件**。本分支保留固定版本的[规格](evidence/results/archcomp26_20261001/airplane_prep_001/specifications.txt)与[动力学](evidence/results/archcomp26_20261001/airplane_prep_001/dynamics.m)；[官方 ONNX](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Airplane/controller_airplane.onnx)仍由官方仓库提供，本地实验副本不重复发布。报告公式的先前逐式核对见[合同来源审计](ARCHCOMP26_NEXT_CONTRACT_SOURCE_AUDIT_20261001.md)。
- 用目标服务器现有 `nncs_env` 的 `onnx.load`、`onnx.checker.check_model` 直接检查这个实际文件：输入 `sequential_1_input: FLOAT[N,12]`，输出 `dense_4: FLOAT[N,6]`，图为 `12→100→100→20→6`，4 次 MatMul、4 次 Add、3 次 ReLU，首个节点就是 MatMul，没有图内显式输入归一化。[检查记录](evidence/results/archcomp26_20261001/airplane_prep_001/CONTROLLER_INSPECTION.json)保存文件路径、大小和图结构；它不是摘要绑定的运行凭据。
- 固定官方 ONNX 与服务器旧 `CROWN-Reach-GPU/ARCH-COMP2024/benchmarks/Airplane/controller_airplane.onnx` 在检查时直接 `cmp -s`，返回 0。这个事实消除了**所比两份文件的字节差异**，不自动修复旧初值、检查器或运行入口。旧 [`crown.py`](../../../../results/archcomp_review_20260923/sources/native/submit/CROWN-Reach/archcomp/Airplane/crown.py) 用 12 输入、6 输出、`output_scale=1`、`output_offset=0`，把模型转为 float64 CROWN，但旧 C++ RPC `asFloat()` 和完整浮点包络仍未被独立证明。

## 共同数学内容与旧代码差异

物理顺序固定为 `(sx,sy,sz,vx,vy,vz,phi,theta,psi,r,p,q)`。官方初盒是前三个位置与最后三个角速度为 0，中间六态 **每个 `[0,1]`**；本轮先用[单个未分割的完整初盒 ledger](evidence/results/archcomp26_20261001/airplane_prep_001/INITIAL_FULL_BOX.json)，避免猜测分区数。控制周期 0.1 s，共 20 次，终点 2 s。性质只约束 `sy,phi,theta,psi ∈[-1,1]`；连续版在闭区间 `t∈[0,2]` 的全部时刻检查，离散版在 `k=0..20` 检查。官方顶层 README 的连续 `[0,20]` 与报告及实例规格的 2 s 冲突；本轮选后者并保留冲突记录。

旧 native 与 Xiangru [`airplane.cpp`](../../../../results/archcomp_review_20260923/sources/native/submit/CROWN-Reach/archcomp/Airplane/airplane.cpp) 的 12 条 ODE 和控制输出顺序 `(Fx,Fy,Fz,Mx,My,Mz)` 与固定 `dynamics.m` 一致；其每周期的实际顺序是**从 `initial_set` 的 12 物理态求 NN 输入盒 → RPC CROWN → 把六个输出写入 held-control Taylor 模型 → 连续积分 0.1 s**（第 118–169 行）。然而第 66–69 行只运行 `vx=vy=vz=1, phi=theta=psi=0.9` 单点；旧 Huan [YAML](../research/gpu_verified_20260930/report/configs/arch_airplane.yaml) 和 [盒 ledger](../research/gpu_verified_20260930/report/configs/arch_airplane_boxes.json) 也都是该单点。旧代码在 `COMPLETED_UNKNOWN` 分支只设置最终状态，没有更新 `initial_set` 且没有退出，后续周期可能从旧集合再次出发；新入口必须在任何非安全/未接受状态立刻停并保存前缀，不能原样复用该循环。

## 两格最小入口与缺件

| 格 | 可明确采用的最小新入口 | 执行前仍缺的实现/记录 | 当前资格 |
|---|---|---|---|
| `airplane-continuous` | 官方完整初盒 **1 个 lane**；辅助时钟 `t(0)=0,t'=1`，六控制初值 0 仅作实现辅助。对 `j=0..19`，在 `t=0.1j` 从当时的 12 态集合算一次 `U_j=NN(X_j)`，按 `Fx,Fy,Fz,Mx,My,Mz` 注入并保持 0.1 s；每段用官方 `dynamics.m` 对应的连续 ODE 推进。首态及每段完整 tube 检查四个安全坐标，保存每段 tube 与 endpoint；任何 `UNKNOWN`/拒绝即停，不宣称全程。旧方法设置 `h=0.01,order=6,cutoff=1e-6` 可作为**明确标名的数值候选**，不是官方强制值。 | 新的完整初盒配置/入口及逐盒保存；四方法各自确认六控制顺序、资源、NN 调用与停止语义；全时 tube 检查与无哈希运行收据。现有旧 C++ 单点初始化及 `UNKNOWN` 续跑问题不能直接用于正式格。 | **未尝试**。连续合同内容足够明确，可先做 CPU 配置预检，再做独立 smoke/full；此文件不授权冒称旧单点成绩。 |
| `airplane-discrete` | 官方报告给出 forward Euler 总规则 `X_{k+1}=X_k+0.1 f(X_k,U_k)`，安全索引 `k=0..20`。若另立并冻结**新比较约定**，可明确写成：先检查 `X_0`，每个 `k=0..19` 在 `X_k` 求 `U_k=NN(X_k)`，用同一 `X_k,U_k` 同时评估 12 条 `f`，一次 Euler 更新到 `X_{k+1}`，再检查安全；控制输出映射同上。这个顺序来自显式新选择与旧连续入口的类比，**未核成官方 2026 离散提交顺序**。 | 固定官方目录没有离散转移程序；仍需取得 2026 参与者实际离散控制/更新顺序，或明确把上述新比较约定列为四方共同合同并分别实现真正的离散算子。其区间舍入/NN 输出包络、每个 `k` 的状态记录、检查器和证据也缺。ODE 步长与 Taylor 阶数在离散格不适用。 | **未尝试、执行语义未冻结**；不能以连续 ODE 结果或旧 C++ 积分结果代填。 |

**可执行的下一步**：连续格以这个 1-lane 全初盒 ledger、固定官方 ONNX、12 态/6 控制映射建立一个隔离无哈希 CPU-only 预检：解析配置，核初盒与控制顺序，构造 20×0.1 s 的调用日程，验证初态及全时安全谓词，禁止数值推进。预检通过后才在新目录做一步 smoke 与 full。离散格先取得实际更新顺序材料或明确命名新 Euler 比较 profile；在此之前不启动数值运行。两个格各自记录 `START`/`RESULT`、保存区间、accepted/status、全时域资格及模型路径；路径记录并不构成内容摘要绑定。
