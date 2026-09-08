# Codex Goal：在端点修复版上减少重复计算，用完整求解时间检验提速

日期：2026-09-08  
主仓库：`https://github.com/lsnnnnnnnn/torch_tm_flowpipe`

## 0. 一句话目标与本轮边界

**端点舍入误差已经修复。本轮不要重新审计同一个反例，也不要恢复旧的不安全版本；在最新修复版本上找出同一步内反复准备的不变工作，实现一个有实际依据的执行结构优化，并证明完整范围和接受/拒绝行为保持不变、完整求解实际更快。**

项目主线仍是：在两个已冻结的多项式动力学系统上，得到可靠、足够紧、能进一步利用批量/GPU 的 PyTorch 求解器。现在恢复性能研究，不再增加 C5/C6 一类紧致性修补。

这不是正式放弃 Huan/Xiangru，也不是永久确定自研 GPU 路线。当前 GPU 后端选择仍未决定。本轮只改进我们自己的执行结构：这样既能减少独立对照的成本，也能判断将来哪些计算值得张量化或复用。**不以别人有缺陷为由宣称我们全局正确，不以这次 CPU 优化成败自动决定最终 GPU 后端。**

本轮不是“再交一份审计报告”的任务。已知反例是应当通过的起始回归；只有与本轮真实相关的新安全失败才能阻断受影响工作。尽量复用刚完成的证明和证据，工作重心必须是实际程序执行和测量。

---

## 1. 已知成果、当前起点与不能混淆的版本

已推送分支：

```text
codex/endpoint-roundoff-repair-and-revalidation-20260908
7e41f33f515f5315b0dec7003a5b06ed0a79afd5
```

| 身份 | 固定提交 | 用途 |
|---|---|---|
| 本轮代码/证据起点 | `7e41f33f515f5315b0dec7003a5b06ed0a79afd5` | 新 worktree 父提交 |
| 修复版最终数值运行时代码 | `0714e475ed9e73bec31619c9c690d1fd63de3d36` | 本轮未优化 reference 的数值语义 |
| 上轮完整长跑源码 | `196a50e9131336d68df07ad0af353deca0092d19` | 旧修复实测，非本轮新计时 |
| 初始端点修复 | `ca330de4ac46e823fb8e36e8982762f0437f71be` | 只用于了解修复历史 |
| 存在端点缺陷的旧 CPU | `4939fb288c941a67f55cc191f4d75f8594692f47` | 仅历史档案，不作新优化的正确性基准 |
| 原生 Flow* | `b85a3211748cb77b736fe4ad42ee02d8d2b81148` | 历史同合同对照，明确 reused |

上轮长跑在 `196a50e...`；随后 `0714e...` 完成设备/精度兼容及端点类别重组的补充。`runtime_bridge.json` 记录了 CPU binary64 的版本衔接和实际回放。**本轮必须以最终修复代码为起点，不能漏掉后续类别重组修复。**

上轮结果，只作已知规模与回归锚点，不作本轮 fresh 性能分母：

| 运行 | 接受 / 拒绝 | 求解秒 | 导出秒 |
|---|---:|---:|---:|
| VDP 固定 1000 步，名义 T10 | 1000 / 0 | 687.521 | 18.744 |
| Brusselator 固定 1000 步，名义 T20 | 1000 / 0 | 2004.428 | 90.527 |
| VDP 原自适应 T10 | 246 / 35 | 188.943 | 4.768 |

完整矩阵 1066 passed / 11 skipped；另有 7 项独立 artifact tests，共 1073 passed / 11 skipped / 0 failed。独立 clone 的 124 项是重复局部检查，不再加总。端点修复没有建立整个求解器的形式化证明，GPU 算术也不在本轮权威范围。

## 2. 工作目录与发布纪律

已知源目录：

```text
/srv/local/shengenli/endpoint_roundoff_repair_20260908/repo
/srv/local/shengenli/endpoint_roundoff_repair_20260908/package
```

先只读确认其中实际 Git 路径、HEAD、clean 状态。若目录不存在，仅在已知自己的仓库中取得固定对象，不全盘扫描、重复盘点旧 clone、更新第三方仓库。

示例操作（SRC 取上述实际存在的仓库）：

```bash
set -euo pipefail
SRC=/srv/local/shengenli/endpoint_roundoff_repair_20260908/repo
BASE=7e41f33f515f5315b0dec7003a5b06ed0a79afd5
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/srv/local/shengenli/repaired_solver_performance_${RUN_ID}"
BRANCH="codex/repaired-solver-prepared-replay-performance-${RUN_ID}"
git -C "$SRC" status --short --branch
git -C "$SRC" cat-file -e "$BASE^{commit}"
mkdir -p "$ROOT"
git -C "$SRC" worktree add -b "$BRANCH" "$ROOT/repo" "$BASE"
```

需要网络时只 fetch 固定父分支/对象，不追随 main 替换本轮起点。新增 worktree 的 Git 元数据变化与修改旧工作树文件分开记录。不得 reset/clean/stash 旧目录，不 force push，不合并 main。

实际保存：ROOT、BRANCH、BASE、Python 路径、包导入路径、线程/CPU affinity、运行环境。继续使用原 py11、CPU float64、单线程；不要同时升级依赖。生产运行必须来源于 clean scientific SHA，之后 package commit 单独记录。

## 3. 启动检查要短：已知端点修复是前提，不是新的研究任务

阅读并复用：

```text
docs/endpoint_roundoff_repair/ENDPOINT_CONTRACT.md
docs/endpoint_roundoff_repair/REPORT_PLAIN_CHINESE.md
artifacts/runs/endpoint_roundoff_repair_20260908/RESULT.json
artifacts/runs/endpoint_roundoff_repair_20260908/raw_minimal/runtime_bridge.json
experiments/endpoint_roundoff_repair/frozen.py
experiments/endpoint_roundoff_repair/run.py
experiments/endpoint_roundoff_repair/README.md
```

在干净父版本中运行一次当前局部回归及现有证据 verifier：

```bash
PYTHONPATH=src:. python -m pytest -q \
  tests/test_our_reference_endpoint_containment.py \
  tests/test_endpoint_roundoff_repair.py \
  tests/test_endpoint_roundoff_carry.py
PYTHONPATH=src:. python -m experiments.endpoint_roundoff_repair.verify \
  artifacts/runs/endpoint_roundoff_repair_20260908
```

以上是验收起点，不重新生成反例、不要扩大到第三方审计。旧 verifier 含固定源码身份；未来优化源码改变后，应在父版本 worktree 检查旧包，不能修改旧 verifier 或旧证据来迁就新实现。新优化用本轮的新比较器验证。

已知历史 fixture 已恢复，沿用上轮记录的精确路径和摘要；只复制所需文件到测试环境，不全盘恢复，不删断言。明确区分历史 fixture/environment 失败与新数值失败。

---

## 4. 冻结的是修复后的数学语义

两套 fixed 合同必须从已提交 `MATCHED_CONTRACTS.json` 和 `frozen.py` 直接读取，并保存实际生效配置，不凭摘要重新组装。

| 参数 | Van der Pol | Brusselator |
|---|---|---|
| 方程 | x'=y；y'=y-x-x²y | x'=1+x(xy-4)；y'=x(3-xy) |
| 初始 x | [1.1,1.4] | [1.48,1.52] |
| 初始 y | [2.35,2.45] | [2.98,3.02] |
| 阶数 | 4 | 6 |
| 固定 h | 0.01 | 0.02 |
| 固定步数 | 1000 | 1000 |
| 历史队列容量 | 100 | 1000 |
| 余项预算 | 每分量 [-1e-4,1e-4] | 同左 |
| cutoff / validation epsilon | 1e-10 / 1e-12 | 同左 |

同时保留各自原有表达式计算顺序、range policy、normal 左右映射、收紧规则、491 上限和 0.99 停止比率。VDP 原自适应合同保持 h_min=0.002、h_max=0.1 和原 scheduler。

必须始终启用：时间幂/乘法/合并舍入的端点修正、完整余项、历史传播、失败回滚和所有包含检查。端点 ad-hoc tightening、拆分初始盒子和改变数学预算仍禁止。

固定步长的真实时间按 binary64 h 的精确和记录；不要为了凑十进制 T10/T20 截短最后一步。自适应的调度时钟和实际 h 之和分开，不重新设计 scheduler。

**本轮是性能优化，所以候选应与最终修复 reference 保持逐位数值行为；上一轮是安全修复，允许改变旧数值。这两种规则不要混用。**

---

## 5. 先测量：此前的性能假设还没有真正执行

前一轮性能任务在发现端点缺陷后就停止了，没有新的 profile，也没有 prepared plan 实现。不能把当时的计划写成已经确定的热点。

当前源码的线索是：`_post_accept_refine_raw_remainder` 在同一步固定 candidate polynomial 的条件下，逐轮更新余项；Van der Pol 专用计算已经有 `_VDPRefinementStaticCache`，generic `ordered_terms` 路径仍通过 `_dense_flowstar_raw_compat_image` 重放。需要用实际测量回答其中是否重复做了不随本轮余项改变的工作。

上轮完整收紧次数：VDP fixed 4386、Brusselator 8806、VDP adaptive 1014。它们只是调用次数，不是时间占比，也不意味着这些轮次可以删除。

### 5.1 复用检查点，别为了 profile 反复长跑

优先复用上轮完整状态 checkpoint，并先用最终修复代码恢复后推进一个小步核对。如果某窗口缺少合法完整 checkpoint，可在本轮的一次顺序 reference 长跑中顺便捕获。

不允许从发布的上下界盒子重新初始化，冒充同一个中间状态；必须保存多项式、余项、normal state、历史队列和调度状态。

至少覆盖：

- Brusselator 早期 1–20、中期约 101–120、晚期约 981–1000。
- VDP 早期 1–20，以及 91–110（跨历史容量清空边界）。
- 自适应接受/拒绝的少量现有代表步骤，只做一致性检查，不展开新的 scheduler 研究。

### 5.2 区分数值求解、性能分析器和证据导出

实际性能分母用 production-no-observer，并包含算法必需的检查和所有准备成本。profile、trace、序列化、绘图、checkpoint 文件写盘分别报告；不能把 profile 开销当成生产时间。

使用 cProfile 的 self time、轻量函数计时或 PyTorch profiler。不要加总重叠的 inclusive percentages。形状/堆栈跟踪会有额外成本，所以正式计时另跑不带 profiler 的路径。

输出互斥时间分类和调用次数，至少区分：

```text
构造候选多项式
初次余项计算
后续逐轮收紧中的固定部分
后续逐轮收紧中的动态部分
多项式乘法/截断与范围计算
跨步历史处理
边界重建与端点误差修正
Python 对象/小张量调度
记录、导出与其他
```

说明每种时间分类如何避免重复计算；给出随步数、历史长度和收紧轮数变化的实际秒数。不要预设“所有慢都是 Python”。

## 6. 本轮主候选：同一步固定计算只准备一次

首选验证的优化机制为：

> 将一次已接受步骤中不会随收紧余项变化的计算准备一次，之后每轮只更新真正依赖当前余项的计算。

这必须是从实际数据流中抽取，不是照搬 VDP 专用缓存、把系统名称换成 Brusselator。

### 6.1 固定与动态分类必须写明依赖

对每个拟复用的量记录：由哪些输入决定；为什么这些输入在同一 attempt/replay 生命周期内不会变；哪里创建、读取、失效。

可能固定的量（须逐项证明，不默认全部成立）：

- basis/exponent/乘法索引、积分索引和 RHS 的有序结构；
- 本次候选点多项式及其独立乘积、截断尾项；
- 仅由该候选多项式、固定 domain、order、h 和 cutoff 决定的中间结果；
- 与当前 remainder 无关的真实 polynomial-only range。

必须动态重算的量：

- 当前 remainder，任何含它的乘积、范围与交叉项；
- 当前 proposal、subset 判断、stop ratio 和最终提交结果；
- 依赖当前 owner、SR history、边界尺度或本次输入的值；
- 动态 cutoff 判定：如果系数可变，就不能仅凭 basis 固定而缓存 mask；
- 与实际分派、错误处理或新 remainder 有关的账本。

**以依赖关系为准，不以变量名有 `poly` 或 `static` 为准。**

### 6.2 单个机制可以涉及必要的多个函数

允许定义本次 attempt 内的不可变 prepared object，配合动态 replay evaluator。为实现这一机制，修改相关多个函数是正常的，不按函数数量机械限制成“只能改一个函数”。

但是不能同时改 RHS 的数学表达式、历史队列策略、range 算法、收紧停止策略或引入第二套 GPU 后端。

生产计算不要使用 Fraction；它保留为独立检查工具。不用 NumPy/C++ 重写整套核，不修改 PyTorch 版本。

### 6.3 先作因果与收益估计，不在明显不值得的位置投入整轮

记录未优化同一步耗时：

```text
准备工作 × 实际重复次数 + 动态更新 + 其他开销
```

新路径估计：

```text
准备一次 + 相同次数的动态更新 + cache 安全检查/分派开销 + 其他开销
```

候选先在少量真实输入上原型测试。若 selected fraction 为 f，局部加速为 s，计算：

```text
predicted_total_speedup = 1 / ((1-f) + f/s)
max_possible_speedup = 1 / (1-f)
```

f 必须是同一个优化前 production workload 的互斥占比，s 必须包含准备与必要检查；不得把局部 2.6x 标成整体预测，不能超出 Amdahl 上限。

不设“必须单函数占30%”这种机械门槛。允许一个连贯的 fixed/dynamic 执行切片；但如果预测不足以产生有意义收益，应如实说明。

若 prepared replay 假设被实际测量否定，**本轮最多转向一个由同一 profile 支持的通用重复准备/调度热点**。先登记选择理由与预期，不串联多个小优化来凑数字。若没有值得实现的候选，提交可复核的否定结论，不搭建无用框架。

---

## 7. 实现约束：省重复准备，不省保证

新增路径 opt-in，现有端点修复始终启用。默认 reference 不变；命名描述实际优化，例如 `prepared_remainder_replay`，不再增加 C5/C6 科学代号。

要求：

1. 同一个输入下，保留原加法/乘法/合并/向外舍入顺序和精度；不借性能任务重排数学运算。
2. prepared object 不得记录某轮 R0 的结果供下一轮 R1 误用；不要做全局数值缓存。
3. first acceptance 后创建/绑定本步计划最容易说明；需要更早准备时同样不得改变首次判断。
4. 计划只活在本次 step attempt 内，candidate/domain/h/order/ODE/scale 改变或 retry 时失效重建。
5. 明确不共享可变张量；`.detach()` 不等于不可变副本。共享 immutable storage 时必须能阻止原地修改。
6. cache 命中/安全检查成本进入时间统计；不为每次命中扫描全部大张量而把收益耗掉，也不能省略必要约束。
7. final remainder、全部误差类别、endpoint correction、normal state 和历史状态与参考同输入逐位一致。
8. 失败的 proposal、非有限数和拒绝步骤遵守原流程，上一份已接受状态不变。
9. report/counters 开关不改变数学结果；不通过减少记录的验证步骤制造速度。
10. 旧 scalar/reference 路径保留，用来逐轮比较和复现，不删除原功能。

本轮默认不接受“相差很小所以视为一致”。如果实现必须改变归约顺序，应先明确这已超出当前执行优化合同，不用容差放行；保存原型与具体原因，不偷偷当性能成功。

## 8. 正确性验证要落到每轮 proposal，不只比较最终盒子

### 8.1 同输入逐轮回放

从真实早期/中期/晚期步骤捕获固定候选及实际 R 序列。未优化 evaluator 与新 evaluator 在同一 R 上比较：

```text
proposal 全部上下界
保留点系数
所有 remainder ledger 类别与总和
是否提交与停止原因
端点修正 E
```

还要分别从同一初值独立运行完整 replay loop，比较实际序列；不能只给新 evaluator 喂旧序列，就忽略新 loop 可能错误更新状态。

### 8.2 必须有的边界测试

- 原两条 endpoint 安全断言正常通过，已有一般 Fraction/两步传递测试直接复用；
- 同一候选连续使用不同 R，不能复用旧 proposal 或旧动态范围；
- 变更 candidate、domain、h、order、cutoff、ODE结构后，拒用/重建旧计划；
- in-place mutation 检测或安全隔离；
- 0 次收紧、一步固定点、不同分量有不同收敛情况、proposal subset失败；
- 真零、subnormal、NaN/Inf、overflow 的原处理不变；
- 拒绝/重试后状态和队列不污染；
- 从修复版 checkpoint 恢复，启用新优化仍得到相同下一步；
- 小型 B2 的结构/动态量不串任务。如真实 solver仍 B1，不把该局部测试宣称完整 batch 能力；
- 新开关关闭时与父版本一致。

新增失败不能通过改旧数学期望、提高 tolerance、删检查或 xfail 解决。历史-only verifier仍在对应旧 snapshot 执行，不与新优化测试混淆。

---

## 9. 正式实测：新分母必须是最新修复版

### 9.1 时间口径

每条正式记录包含：

```text
source SHA / actual import path / config digest
CPU threads / affinity / Python / torch
setup/plan construction
numerical solve（包含每步准备、保证检查、端点修正）
export/trace/checkpoint I/O
whole process
peak RSS
accepted/rejected / actual sum(h)
```

不能以旧 6083s 审计运行、旧 4061s 未张量化版本或有端点缺陷的 4939fb 作为速度分母。上轮 2004.43s/687.52s 也只是历史规模；**正式速度比来自本轮同环境、最终修复 reference 与优化版的匹配测量。**

新 prepared plan 的建立不能移到计时外。可另列长期复用的结构索引初建成本，但必须提供含初始化的完整求解时间。

正式计时不带 profiler。采用相同单线程和 CPU 亲和性，不清理/终止他人任务，不读取凭据或完整进程参数。记录明显 contention，必要时重复匹配测量。

### 9.2 测量顺序与有限重复

开发阶段先少量局部步骤，不反复1000步。

1. representative 20-step windows：reference/optimized 至少 3 对，交替顺序。
2. 稳定后 Brusselator 100-step prefix 至少 3 对；VDP代表prefix至少3对。
3. 两系统 reference/optimized 各一条新的完整固定 1000 步运行。
4. VDP新优化执行一次完整自适应T10，和修复版自适应档案按相同h/状态序列核对。用于速度结论则必须新跑配对；本轮不要求自适应 speed claim。
5. 若完整速度距离接纳阈值不到10%，增加一对完整测量或明确标 `borderline_single_pair`，不能用小数点硬签稳定通过。

可在同一份新科学提交中用开关区分 reference/optimized，前提是reference路径已核对与最终修复源码一致。完整两条均从初始条件实际执行，不能用缓存的求解输出伪造 baseline/optimized。

### 9.3 结果必须同时回答三个维度

**时域：** VDP fixed 1000/0、Brusselator fixed 1000/0，原自适应保持同一规则和同输入决策。性能优化应与修复 reference 的 accepted/rejected/h序列一致；不是强制手写246/35。

**范围：** 每一步 endpoint x/y 与 tube x/y 的上下界，修复 reference 与优化版逐位一致；同样记录普通余项、端点E、重要状态/队列及replay停止计数。所有正式4000分量范围均覆盖，不能只测最后一步。

**时间：** 本轮目标为 Brusselator完整求解至少1.5x；VDP不出现稳定超过10%的减速。保留原始重复值、median/min/max，不宣称一次测量就是稳定加速。

1.5x是本轮性能目标，不是数学真理。正确但仅1.2x也应保留并解释为“有收益但未达目标”，不能改成无数值价值或造假通过。VDP已有专用缓存，因此不要求两个系统都同倍率加速；不得给Brusselator写死方程来达到目标。

峰值RSS最好不超过reference的1.5倍；超出则列为time-memory tradeoff，不能隐瞒。

## 10. 对 Flow* 的关系与图表

Flow* 原对照默认复用，不重编译、不改其数值代码。历史CPU与Flow*约百倍速度差只作背景，本轮能正式主张的加速是“优化版 / 本轮修复reference”。若要称本轮Flow*速度比，则必须同环境新计时并说明；不是本轮硬要求。

至少给出：

- 修复reference与优化版全程四项宽度比（预期1）；
- 两者完整求解时间、初始化和导出分开柱图；
- 代表步骤中“固定准备重复多少次”修改前后次数/时间；
- 优化后剩余热点和Amdahl预测与实测差异。

旧Flow*全程宽度可以与新优化结果用既有共同observer再次比较，明确 `REUSED_MATCHED_REFERENCE`。不要求优化版为了更紧而改变数值；保留VDP约一至两成典型宽度差、Brusselator个别时间点较大的差距，不再只讲终点。

---

## 11. 本轮明确不做的事情

- 不再修端点算法、减少端点误差或撤销其中的向外计算以提速。
- 不修新的 tightness 算子，不调队列容量、步长、阶数或refinement次数。
- 不重新审计Xiangru/Huan、不扫描旧clone、不恢复历史dirty补丁、不自动联系维护者。
- 不移植整个第三方引擎，不增加第三系统或控制器。
- 不写完整CUDA后端，不把CPU需要先追平C++当作GPU前提。
- 不借性能工作把逐lane循环称为真正并行batch。
- 不再建立庞大的通用审计平台、十几份重复报告；复用现有run/export/test能力。
- 不将任何已知局部通过表述为整个solver形式化证明。

可只读引用现有第三方架构记录说明未来选择，但不运行第三方实验。本轮不会永久决定GPU自研/采用路线。

## 12. 停止、回退和最终决策

本轮最多提交一个连贯的性能机制。允许在它内部做合理调试和必要的相邻执行修改，不因为函数数大于1就停止。

- 已知端点反例应通过；只有实际新回归/当前依赖链新安全问题才阻断相关实现。
- 若优化改变数学输出，首先在候选中定位并修正或回退，不把基线改到“配合”候选。
- 若发现独立基线缺陷，不掩盖，也不就地扩大为整套安全审计：保存最小复现、影响范围、已完成的profile与实现；暂停依赖它的performance claim。
- 若候选无实际收益，保留反证和优化后profile，不继续叠第二种机制。

最终状态只需以下少数几种，报告先用中文解释：

```text
REPAIRED_REFERENCE_PRESERVED__PREPARED_REPLAY_SPEED_TARGET_MET
REPAIRED_REFERENCE_PRESERVED__USEFUL_SPEEDUP_BELOW_TARGET
REPAIRED_REFERENCE_PRESERVED__NO_USEFUL_SPEEDUP
OPTIMIZATION_REGRESSION__REFERENCE_RETAINED
NEW_RELEVANT_SAFETY_BLOCKER__PARTIAL_WORK_PRESERVED
```

没有性能实现时，不能用测试数量授予前两种状态。状态从原始数值比较和时间推导。

本轮结束后必须留一页路线决定：

1. 本轮到底减少了什么，剩余时间主要在哪里？
2. 这个执行计划是否天然能容纳B维，哪些还只是Python B1？
3. 下一步更值得在这个计划上做batch，还是另行评估已保存严格GPU引擎？依据是什么，缺什么数据？
4. CPU再减少一半仍可能远慢于Flow*，不能靠局部成功声称GPU选型完成。

**下一轮只根据实测提出一个主方向；不要在本轮同时开展GPU后端复评和完整CUDA开发。**

---

## 13. 紧凑交付清单

建议文档与代码位置：

```text
docs/repaired_solver_performance/REPORT_PLAIN_CHINESE.md
docs/repaired_solver_performance/REPLAY_DEPENDENCIES.md
experiments/repaired_solver_performance/
artifacts/runs/repaired_solver_performance_<RUN_ID>/
```

最小必要结果：

```text
SOURCE_MAP.json
RUN_CONTEXT.json
EXECUTION_CONTRACT.json
profile_windows.csv
replay_work_counts.csv
candidate_decision.json
same_input_replay_equivalence.json
full_width_equivalence.csv
full_run_summaries.json
timings_raw.csv
timing_summary.csv
remaining_hotspots.csv
RESULT.json
raw_minimal/
tests/commands.json
tests/*.xml / *.log / *.exit
figures/
SHA256SUMS
```

复用已有上下界导出与common observer。新输出带 `FRESH_REPAIRED_REFERENCE`、`FRESH_OPTIMIZED`；旧修复数据 `REUSED_ENDPOINT_REPAIRED`；缺陷旧CPU `OLD_UNREPAIRED_ARCHIVE`；Flow* `REUSED_MATCHED_REFERENCE`。不得混用身份。

新增有边界的verifier，从raw记录重算：数学等价、时域/步数、实际h、replay决策、求解/导出时间、倍率、计数和最终状态。源码身份和环境检查与科学包含证明分开。

少量有意义的篡改测试：改一个端点误差、一个proposal、一个h、一个mode、一个计时或把reused改fresh，重算外层hash仍须被拒绝。不要为凑数量复制同一种测试。

报告用白话明确回答：

> 原来具体重复了什么？为什么可以少做？哪些保证仍每次执行？新旧全程范围是否一样？T10/T20是否仍完成？完整速度是多少？有没有靠隐藏准备/导出来加速？还差Flow*多少？GPU路线是否已决定？

测试先targeted再现有完整矩阵。旧停止snapshot和已修复当前测试各自归属正确版本，去重计数；原失败断言不得改弱。必要历史fixture按已恢复摘要复用，不新做大恢复任务。

科学运行在clean commit；最终package不冒充科学版本。结束后只push本轮新分支，不合并main。最后确认：远端tip、本地HEAD、独立clone一致；独立clone实际跑相关局部测试并验证新证据，说明没有重跑所有长实验。

## 14. 成功的真正含义

**不是“我们又通过了更多检查”，而是：在端点已经修复、两个系统完整结果重新成立的基础上，把同一步里重复的不变计算真正省掉，并用相同输出和完整求解时间证明收益。**

若本轮没有足够收益，给出有实测依据的结论，避免再盲目投入。无论结果怎样，不能由一轮CPU优化自动否定或接纳Huan/Xiangru的GPU架构。
