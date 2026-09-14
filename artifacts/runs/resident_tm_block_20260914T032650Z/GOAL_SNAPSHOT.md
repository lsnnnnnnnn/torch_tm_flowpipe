# Codex Goal：从“GPU 算一个范围”转向“GPU 连续算完一段 Taylor-model 计算”

日期：2026-09-11
仓库：`https://github.com/lsnnnnnnnn/torch_tm_flowpipe`

## 0. 本轮主线与交付

**停止继续优化 range packet 的等待时间、分包和缓冲区。保留现有 CPU 参考、在线范围服务和 GPU 长时域结果，选择实际求解路径上一段有连续依赖的 Taylor-model 计算，将它实现为一个有独立数值检查、真正保留中间设备数据的 GPU 计算块，并接入真实求解前缀。**

本轮不只是写架构建议，不只跑另一个离线范围 microbenchmark，也不从零重写完整 ODE 求解器。需要交付一个可运行、默认关闭的复合计算块原型、真实接入结果和是否值得扩大迁移的结论。若集成没有收益，保留正确原型和负结果，不继续叠加第二项无关优化。

这里“复合计算块”的白话含义是：原来 CPU 要连续做若干多项式运算，期间反复请求范围计算；现在输入一次，在 GPU 上完成这段相互依赖的运算，最后一次返回完整结果。**不是把许多互不相关的范围请求再次合包。**

当前主目标仍是：在可靠且不明显损失宽度的前提下，让多项式 plant-only 求解器具备有实际价值的批量/GPU 执行能力。不得只为跨过某个倍率门槛而改变方程、初始集、阶数、步长或必要的误差处理。

本轮明确授权比上一轮更大的数学执行范围，但只允许一个连贯的计算块。上一轮“GPU 只做范围”的约束不再限制这个新原型；它仍适用于未改的基线。

## 1. 已有事实：不要重新做成一轮历史审计

| 角色 | 固定提交 |
|---|---|
| 本轮父交付 | `372cede015105a3bcdf56b514b65f7dd0551fc43` |
| 上轮实际科学运行代码 | `1d870939fcdb72bd999730cf09de0b14d1cd71c4` |
| 上轮修正后的离线验证器 | `0b0da13da2c563ddf287fa79731e2c8653d46852` |
| 之前在线 GPU 交付 | `e4d920aa710bd407666c60285632a4c294edb9da` |
| 历史 stock Flow* 对照 | `b85a3211748cb77b736fe4ad42ee02d8d2b81148` |

父分支：`codex/live-gpu-range-packets-and-long-horizon-20260910T023603Z`。
父证据：`artifacts/runs/live_gpu_packets_20260910T023603Z/`。

已完成而不能重复冒充新成果的内容：

- packet 数学和任务隔离检查通过；220 项受影响主测试及 2 项验证器衔接测试通过。这不是全仓所有历史测试的总数。
- 五个固定配对区块中，Gp/G0 的速度改善为：VDP `G0/Gp=1.0495176363`，Brusselator `1.0368809219`；两系统各 5/5 次 Gp 更快，但未达 1.15 工程目标。`no_gain` 是机器分级，不等于实测零收益。Gp 默认仍关闭。
- 相对同区块较快严格 CPU，Gp 倍率中位数为 VDP `1.2020491744`，Brusselator `1.1900009062`。这是同轮 B32×20 前缀，不是 Flow* 对照，也不是完整 T10/T20 倍率。
- 新 GPU 范围链已从原始未分区初始盒跑满两套 1000 步：VDP T10，Brusselator T20；没有周期性 CPU 纠偏，没有读取旧答案推进。
- 长跑步内求解时钟合计约 VDP 508.416 秒、Brusselator 1224.525 秒；完整调用约 534.782/1367.953 秒，包含报告明确列出的记录成本。这不是新鲜同轮 CPU/Flow* 配对，不可直接计算稳定加速。
- 全程 GPU/CPU 宽度比最大值：VDP 约 `1+6.6e-14`；Brusselator 四通道中最大约 `1+6.56e-7`。不能写成逐位一致。
- 3550 条 `width_ratio > 1.10` 全部来自 stock Flow* 对照：VDP 3356 条、Brusselator 194 条；相对修复 CPU 的警戒为零。它们不是 3550 条 GPU 漏包或新增 GPU 宽度故障。
- 全程和局部证据不是整个求解器的形式化证明；原默认路径的已知 `pow_int` 局限不能被抹掉。

实际诊断说明合包没有做假：VDP B32 派发 25508→5257，Brusselator 36960→7773，平均包约 13.45/13.76 条。请求总数仍约 70704/106919，分别完成 640 个 lane-step。**物理提交减少约 79%，任务内部需要逐次生成和消费的范围结果没有消失。**这些是带记录诊断，不当作正式性能样本。

下一步的假设是：只迁移范围已经不能覆盖大部分主机端工作，需要更大粒度的连续数学执行。尚未识别出唯一主导 CPU 函数；本轮必须用有限测量选择具体块，不能直接声称 GIL、PCIe 或某一个函数已经被证明是根因。

## 2. 工作树与环境

优先从以下已知目录建立新工作树；不存在时只检查同轮明确的 package 目录和自己的主仓库，不全盘扫描旧 clone：

```bash
set -euo pipefail
SRC=/srv/local/shengenli/live_gpu_packets_20260910T023603Z/repo
BASE=372cede015105a3bcdf56b514b65f7dd0551fc43
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/srv/local/shengenli/resident_tm_block_${RUN_ID}"
BRANCH="codex/resident-tm-block-${RUN_ID}"
git -C "$SRC" status --short --branch
git -C "$SRC" cat-file -e "$BASE^{commit}"
mkdir -p "$ROOT"
git -C "$SRC" worktree add -b "$BRANCH" "$ROOT/repo" "$BASE"
```

保持 py11、现有 PyTorch/NVRTC、V100/GPU0。保留原线程预算：`taskset -c 2` 是绑定逻辑 CPU 编号 2，**不是给两个 CPU 核心**；Torch intra/inter-op 和 BLAS 线程均为 1。初始化阶段统一设置，不能让多个 worker 竞争修改全局配置。

不升级环境、不删除旧修改、不 reset/clean 旧工作树、不强推、不合并 main。读远端仅核对父身份；出现新提交也不要在正式实验中途换基线。

先阅读这些具体文件：

```text
docs/live_gpu_packets/REPORT_PLAIN_CHINESE.md
experiments/live_gpu_packets/README.md
experiments/live_gpu_packets/long_horizon.py
experiments/live_gpu_packets/compare_horizons.py
src/torch_tm_flowpipe/live_range_service.py
src/torch_tm_flowpipe/range_packets.py
src/torch_tm_flowpipe/range_packet_cuda_kernel.cu
src/torch_tm_flowpipe/taylor_model.py
src/torch_tm_flowpipe/polynomial.py
src/torch_tm_flowpipe/batched_dense_tm.py
src/torch_tm_flowpipe/flowpipe.py
src/torch_tm_flowpipe/prepared_remainder_replay.py
```

从父包复用 `RESULT.json`、`packet_cost_breakdown.csv`、`full_horizon_matrix.csv`、`full_width_summary.csv`、原始检查点和既有安全测试。不要开始就重跑 40 次父计时或两个父 1000 步。旧验证器的源锁定要求在旧工作树保留，不删断言以适应新代码。

## 3. 不可改变的数学与实验设置

以父冻结的 `experiments/endpoint_roundoff_repair/frozen.py` 和真实 runner 为唯一参数来源：

- VDP：原表达式，order 4，binary64 h=0.01，SR100。
- Brusselator：原表达式，order 6，binary64 h=0.02，SR1000。
- 原始盒、固定 8×4 分区、指定 B8 子集、余项预算、cutoff、验证 epsilon、接受后收紧、491 次上限和 0.99 停止规则全部不变。
- prepared replay、现有有序张量取范围、端点舍入误差处理都保留。
- 禁止为了批量将误差历史合并为普通独立区间，或删掉左右映射、端点误差和截断误差。
- 不接 CROWN/auto_LiRPA/NN controller，不加第三个 ODE，不改 Float32，不更换为另一种更容易的积分方法。

CPU 仍是独立实现参照，不是无条件真值。新块要对相应数学操作建立独立包含证据，而不是只对照 CPU 输出。

## 4. 第一阶段：有限地定位一个“连续计算块”

### 4.1 只测新的架构决策需要的成本

复用父数据，新增少量真实窗口：每系统 B8 的 5–10 步、B32 的 3–5 步，以及长历史与历史清空前后各一个可恢复短窗口。旧 B1 checkpoint 不能冒充 32 个不同的晚期轨迹；若需异构晚期状态就实际生成，并分开标注。

拆开并记录：

1. 多项式/Taylor-model 组合、相乘、截断及相关范围；
2. 候选构造与余项计算中未迁移的数学；
3. 历史误差整理与矩阵传播；
4. 数据表示转换、Python 对象构造和小张量操作；
5. 请求准备、队列等待与结果消费；
6. 实际设备执行、传输、主机同步。

区分线程 CPU 时间、互斥墙钟区间和含等待的函数跨度。多个任务的 Future 等待不能相加当作总时间。CUDA event 区间可能包含主机提交间隙；没有活动级跟踪时，不把它命名为纯 kernel active time。正式性能不带重型 profiler。

### 4.2 优先选择的块

**优先检查真实跨步组合/代入路径中的一段 Taylor-model 运算：多项式相乘或组合 → 保留规定阶数 → 计算被截断/删除部分和交叉误差的范围 → 合并余项。**

至少涵盖两个有数据依赖的实质操作，具有明确的完整输入、完整输出；中间多项式或区间结果能在设备上被下一操作直接消费。不接受把原四个 range kernel 换名为新块。

必须指明真实调用链、函数及入口/出口，证明两个系统确实使用它。先查看 `TaylorModel.__mul__`、`Polynomial.mul_truncate`、normal insertion/组合的实际调用，而不是默认通用辅助函数一定是生产大头。

若优先块在真实运行中很少出现，允许在同一轮改选一个相邻的通用复合块，例如完整的一次 RHS/余项图求值。最多比较两个连贯候选，选一个实施；不能叠加历史传播、端点、RHS 和所有组成模块为完整求解器重写。

选择记录需要回答：本块现在花多少实际时间，有多少 Python/范围往返，迁移后哪些中间对象留在设备，哪些 CPU 工作仍然存在。以这些成本估计可能收益，注明只是预测。

不得因为“没有一个函数名占比超过任意 30%”就停止：一段调用链的成本可能分散在许多小函数里。应按一段实际连续计算累计不重叠工作。也不要为了满足交付而迁移没有实际价值的冷门 helper。若两候选均缺少可信收益空间，提交有实测依据的架构否定与最小可运行接口，而非再增加第三个候选或重开调度参数搜索。

## 5. 第二阶段：定义数值合同与最小设备执行图

### 5.1 有哪些输入/输出

输入至少包含块实际使用的：点系数或区间系数、有序单项式结构、domain、当前余项及类别、阶数、cutoff、变量角色、必要的时间幂数据、任务和 attempt 身份。

输出必须是原调用方真正需要的多项式/Taylor model、余项及误差记录，不只是一个更窄的盒子或一个 success flag。CPU 接口可以在块结束后重建对象；这个成本计入正式时间。

不在本轮把完整历史队列搬到 GPU，除非它本来就是选中且唯一的计算块的必要输入。其余状态、接受/拒绝和跨步 commit 沿用已验收路径。

### 5.2 一次传入，连续完成，最后传出

- 执行图可以有多个 kernel；不强求一个巨大 fused kernel。
- 中间系数、截断项、区间和依赖范围留在 GPU，不能每算一个临时范围就返回 Python 再继续。
- 块内不得为了查看每个数值或控制每个项而反复 `.item()`、`.cpu()`、同步和 CPU 循环。
- 必需的块结束检查、失败掩码和最终返回允许同步，必须计时。
- 结构按签名编译/复用，系数和余项每次重新上传。结构签名不得只用矩阵形状；应含有序支持、表达式、角色和计算规则。
- cutoff、零项、数据依赖分支不能沿用过去一次输入的决定。用设备掩码或明确重新构建结构；结构失配安全回退且记录次数、成本。
- 同批任务支持不同可分组或使用分段描述。补零不能变相增加实际运算、改变累加顺序或忽略某个任务的项。
- 控制缓存和设备工作区大小，完成事件之前不得复用在途内存。

只把内存留下而数学仍在 CPU 不算 device-resident 计算块。请给出实际输入/输出字节、块内中间字节、GPU 执行次数、同步数和替代了多少原范围请求。

### 5.3 浮点与遗漏项必须有依据

以两个 Taylor model 相乘为说明（实际执行次序以已选块为准）：

```text
A = p + I，B = q + J
输出为 r + R
```

至少需保证：

```text
p*q - r 的范围
+ p(D)*J
+ q(D)*I
+ I*J
```

被 R 覆盖。其中 `p*q-r` 不仅包含高阶截断和 cutoff 删除项，**也包含保留点系数在乘加时产生的舍入误差**。不能只支付被删除项，继续把保留系数当作精确值。组合块则逐操作维护对应函数包含关系，不能用一个最终盒子测试替代。

不强制采用上面公式重写旧的联合误差算法；若旧块采用更紧的联合误差计算，保持其数学结构，不能为方便 GPU 改成更松的独立区间表达。

推荐固定顺序、显式 double 舍入加乘和区间保护；若使用不同归约顺序，必须先给出适用的、数值上向外计算的误差上界。禁用未记账的 FMA 合并/fast-math；处理次正规数、非有限数和溢出；启动时在实际设备验证所依赖原语。

新块出现的局部舍入遗漏应在本轮修复，而不是机械停在“又发现一个反例”。同一计算块涉及多个函数不等于超出预算。修复需要改变旧系数/余项时，明确标注安全修复与性能改动，旧有缺陷不能成为强制逐位复制的目标。独立于本块、影响前置保证的新缺陷则保留反例、限定结果，不能跨全仓展开无边界审计。

Fraction/高精度用于独立检查与诊断，不放入正式 GPU 热路径充当每次结果的 CPU 纠偏。CPU 正式对照原本必要的保证计算不能为赢得比赛而删掉。

## 6. 第三阶段：局部和连续接入检查

### 6.1 有限而针对性的局部矩阵

覆盖正常与取消输入：不同变量数、支持大小、实际阶数；点/区间系数；同类项合并与严重相消；截断/cutoff 门槛两侧；极小数、负数、奇偶幂、零及非有限输入；B1/B2/B8/B32；异构有效掩码和切批。

用独立 Fraction 操作构造精确输入多项式的相乘/组合与误差系数，逐系数或以可验证的系数误差范围证明包含，不靠随机轨迹抽样。保留之前端点、历史矩阵和三次幂的已有回归，但不要重新研究其历史来源。

### 6.2 同设备调度与跨设备数值分开

- 同一新 GPU 块，单请求与批量、换任务顺序与切批，逐任务的数值和失败结果应一致。不能用“CPU/GPU 本来不同”解释同 GPU 调度错误。
- CPU/GPU 各自对精确合同检查。两者正确区间不必相互包含；禁止择窄端点或无依据求交。
- 若数值运算顺序完全不变，争取逐位一致；若为了正确支付误差而变化，不要求复制旧错误，分别报告系数、余项与最终宽度变化。
- checkpoint/取消/重试仍只提交整段成功状态；新块返回在取消后无效；每任务独立中间数据和误差归属。

### 6.3 必须真实在线接入

先 B1/B2 各 2–5 步，再两系统 B8/B32 各 20 步，再每系统两个不同任务各 120 步。使用当前输入构建/提交计算，输出继续推进下一步，不加载保存答案。

VDP 120 步须跨过 SR100。Brusselator 清空附近可复用本轮真实 step999 checkpoint 跑局部窗口，标为恢复窗口，不冒充新从初始集达到 T20。保留原自适应接口的受影响回归。

比较每段 endpoint/tube x/y、当前余项、相关误差类别、支持和多项式、收紧和接受行为、下一步完整状态。宽度异常与包含性失败分开。

GPU/对应 CPU 的非零宽度超过 1.10 时列出最差位置并定位是必要舍入保护还是表示退化。这个比率是实用性警戒，不是安全证明；不能仅因小于 1.10 就认为正确。

## 7. 第四阶段：实际性能与 Flow* 距离

### 7.1 本轮比较路线

| 路线 | 用途 |
|---|---|
| 现有 Gp | 本轮同设备性能基线；显式开启 packet，默认值不因此改变 |
| 新常驻块 + Gp | 唯一候选；块外继续同一路范围后端 |
| 当前较快严格 CPU Q（同时保留少量 S 检查） | 检查总收益，而非只与 GPU 自己比较 |
| stock Flow*，同方程/分区/步数/阶数/历史合同 | 补一个现实速度尺度，不能用旧未分区 T10 时间除 B32 前缀时间 |

未修改 G0/Gp 的数学结果证据复用。不要为了候选快，给它更多 CPU 核、移走初始化、或者让分母额外导出全部状态。

### 7.2 正式矩阵收敛，不再复制所有历史实验

主测试是原 32 个不同子盒、每个 20 步。正确性和原型调优使用另一组预先注明的短窗口；冻结实现与参数后，五个配对区块新跑“Gp/新块”，交替顺序。所有样本报告，不为凑阈值追加。

CPU Q 每系统三个同资源前缀测量，S 每系统一个有限核对，明确哪些可形成配对、哪些仅比较分布。不重复声称曾经的约 1.20 倍为本轮新增收益。

Flow* 复用已有编译和只读导出接口，在同一 CPU 预算上顺序处理相同 32 个子盒和 20 步，记录完整工作量、实际参数、冷启动与求解。不能拆得更细，也不能降低阶数。优先同进程循环；若只能逐进程运行，把启动成本另列并保留总时间。必要接口构建限于测试驱动，不改 Flow* 算法。若依赖不支持完全相同合同，如实写不支持，不伪造近似相同结果。

这组 Flow* 数据用于看清实际差距，不是把它视为浮点数学真值，也不是要求本轮必须追平它才能算进展。

B1/B8 只做少量诊断与计时，避免把主矩阵膨胀成数百次实验。

### 7.3 计时与目标

完整墙钟包含真实初始状态建立、执行计划创建/必要更新、打包、上传、块内计算、返回、重建 CPU 对象、未迁移数学、身份检查、worker/服务和清理。冷编译与一次性设备自检单列；本次输入准备不能移到计时外。

报告四个层次：常驻块内部时间、带往返的块时间、真实完整前缀时间、与同工作量 Flow* 的现实距离。只报告成功完成的规定 lane-step；取消/失败不能成为减少工作量的“加速”。

目标分开：

- **架构目标**：真实替代一个连续数学块，中间结果在设备上被后续操作使用；块内主机数值往返显著减少。
- **局部目标**：带往返的复合块在 B32 争取至少 3× 相对原 CPU 块，给出实际是否达到。
- **整体目标**：新块相对现有 Gp，两系统前缀各争取中位数 ≥1.20 且至少 4/5 组胜出；相对严格 CPU 无稳定回退。

目标不是预先承诺。实施前用新的非重叠成本份额估算 Amdahl 空间；若所选块理论上无法带来约 20% 改善，应在两个候选内调整选择，不在计时后改阈值。

正确但收益较小，就写正确、收益较小；kernel 快而完整前缀不快，则明确瓶颈仍在块外或边界。不得自动授权迁移整个引擎，也不要再次转回小包/等待参数调优。

## 8. 长时域：复用已完成证据，候选有实质价值后才新增全程

父版本的原始未分区 GPU 范围链 T10/T20 已真实完成，连同 CPU/Flow* 的完整对象和比较全部保留，不能重新计为本轮成果。

新块局部和 120 步检查通过、且完整前缀取得实质收益后，才对最终候选新跑原始未分区 B1 的两个固定 1000 步，分别保存四项完整上下界、求解决策、余项和状态。以这些结果决定能否成为下一轮数值基线。

若原型正确但无工程收益，停止扩大迁移，新的完整候选时域明确标 NOT_RUN_PERFORMANCE_NOT_USEFUL；不要浪费大量时间重复已知旧长跑，也不要把旧长跑贴在新块下面。

需要同 GPU 完整逐位对照时必须有真实父长跑锚点；本轮父 Gp 已有完整状态检查点及范围，可复用其支持的比较范围。父未保存的中间对象不得声称已逐位比较。

宽度分别比较“新块 vs 父 Gp”“新块 vs 修复 CPU”“新块 vs Flow*”；近零范围用绝对差。达到 T10/T20 不等于宽度相等，更不等于速度相等。不得通过换 h/order/初始盒救活候选。

## 9. 与 Huan/Xiangru 的关系

本轮不是宣布永远弃用他们。现有服务、独立反例和严格修复资产都保留。

允许只读查看服务器已有的严格修复代码或底层 GPU 运算，复用一个适用的实现思路/局部算子，但必须说明版本、修改、许可证和验证责任。禁止重新 clone 十个仓库、重跑历史 450 条 dirty 实验、重审整篇证明或重新搭一套 NNCS。

若直接复用已有严格实现显著降低选中块的工程成本，可以采用；不能仅因“我们自己的”就拒绝复用，也不能仅因“教授/同学的”就免检。不偷偷把整个候选引擎切成其他后端。

## 10. 测试、证据和交付控制

不要再写一个庞大的通用审计框架。复用当前 runner、状态比较、checkpoint、独立 Fraction 检查和性能汇总。

建议最小交付：

```text
docs/resident_tm_block/REPORT_PLAIN_CHINESE.md
docs/resident_tm_block/NUMERICAL_CONTRACT.md
experiments/resident_tm_block/
artifacts/runs/resident_tm_block_<RUN_ID>/
  SOURCE_MAP.json
  BLOCK_SELECTION.json
  local_exact_checks.json
  device_residency_and_calls.csv
  online_state_comparison.json
  timings_raw.csv
  paired_speedups.csv
  matched_flowstar.csv
  full_horizon_result.json
  RESULT.json
  tests/commands.json
  tests/*.xml
  raw_minimal/
  SHA256SUMS
```

性能原始值、数学输入/输出和必要源码身份足够复算，不为每个普通通过再增加一组无关哈希。加入少量有针对性的错 lane、错程序版本、漏付一笔舍入误差、取消后提交、计时分母变化和假设备执行测试。

测试计数说明本轮受影响集合，不与上轮完整根测试数量直接比较。独立副本重算保存证据，并真实执行两系统 B2 两步的父/新路径；明确未重复运行完整计时和 1000 步。实际运行源码、验证器修订和最终封装分别记录；验证器的 schema 错误可以修正，保留旧失败和修改范围，不能通过放宽数学条件让旧数据过关。

在新分支提交、推送并核对远端，不修改 main。禁止输出凭据。最终主结果不能只有 PASS，至少分开：

```text
block_numerical_contract: pass/fail/not_established
actual_device_residency: measured/not_implemented
online_integration: pass/fail/not_run
local_speed_effect: measured
end_to_end_effect: useful/small_gain/no_gain/regression
full_horizon_new_candidate: achieved/not_achieved/not_run
matched_flowstar_gap: measured/not_established
entire_solver_formal_proof: false
full_gpu_engine: false
```

## 11. 最终中文报告首页必须讲清楚

用读者不用记内部代号的语言回答：

1. 以前为什么只把范围运算送 GPU 不够？实际还在 CPU 做什么？
2. 本轮究竟把哪段连续工作放到了 GPU？中间结果怎样继续被使用？
3. 为什么没有漏掉乘加舍入、截断和余项，任务间如何独立？
4. 同一批真实任务的新旧完整时间各是多少，不能只放 kernel 倍率。
5. 与相同工作量 Flow* 相比还差多少？哪些时间是历史复用？
6. 本轮是否值得继续扩大 GPU 计算范围？如果没有，是否应转向既有完整设备架构的有限复用，而不是再磨 packet？

本轮的成功不是又增加一个 C 编号，而是拿到一个可运行、能被实际求解消费的连续设备计算块，并用正确性、全程宽度与实际时间判断它的价值。

## 12. 依据

父提交 `372cede015105a3bcdf56b514b65f7dd0551fc43` 的报告、结果和源码是项目事实依据，尤其：

- `docs/live_gpu_packets/REPORT_PLAIN_CHINESE.md`
- `artifacts/runs/live_gpu_packets_20260910T023603Z/RESULT.json`
- `artifacts/runs/live_gpu_packets_20260910T023603Z/packet_cost_breakdown.csv`
- `artifacts/runs/live_gpu_packets_20260910T023603Z/full_width_summary.csv`
- `artifacts/runs/live_gpu_packets_20260910T023603Z/full_horizon_matrix.csv`
- `src/torch_tm_flowpipe/range_packets.py`
- `src/torch_tm_flowpipe/taylor_model.py`

外部工程原则仅供实现参考，不代替项目实测或数学保证：

- NVIDIA CUDA C++ Best Practices Guide，10.1 Data Transfer Between Host and Device：减少往返，让中间数据在设备上直接继续计算。
  `https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device`
- NVIDIA CUDA Math API，Double Precision Intrinsics：显式舍入的 double 加、乘等原语。
  `https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html`
