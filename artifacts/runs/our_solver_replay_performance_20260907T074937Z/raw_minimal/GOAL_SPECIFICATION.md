# Codex Goal：结束当前后端接纳，回到我们自己的求解器——减少重复计算并验证实际提速

编写日期：2026-09-07。主仓库：`lsnnnnnnnn/torch_tm_flowpipe`。

## 0. 一句话任务与项目主线

**保持现在已经得到的安全范围和完成时域不变，找出我们自己的求解器中反复重做的不变计算，只做一次有证据支持的执行结构优化，用完整求解时间证明是否有用。**

最终目标仍然是：得到范围足够紧、保证不漏掉轨迹、能利用批量计算和 GPU 提速的 PyTorch 求解器。本轮不是回头增加紧致性修补，不是又一次 Huan/Xiangru 审计，也不是一次性开发完整 CUDA 引擎。

现成后端接纳尝试在 `f23c375...` 合法结束。保留候选、旧修复、反例和维护者说明，但本轮不再读第三方源码、增加第四个反例或扩展第三方补丁预算。发现三个局部问题不等于整个架构不值得使用；只是这版代码不能直接接纳，而且继续修已经超出我们的当前主线。

本轮对“提速”的解释必须简单：例如“同一步中一份不会改变的多项式过去算九次，现在只准备一次；每次真正变化的误差仍重新算”。不能只交更多检查器、测试数和状态代号。

---

## 1. 固定起点，保护旧工作

### 1.1 源码身份

| 角色 | 固定版本 |
|---|---|
| 本轮代码与证据起点 | `f23c375f7935dc2f85a37a0ad69814ab052168bb` |
| 前一轮接纳审计 | `3ca31b8cafc0ad4bf36214417c5a971e145f3ae0` |
| 已运行的优化 CPU 数值参考 | `4939fb288c941a67f55cc191f4d75f8594692f47` |
| 已有 CPU 独立任务包装 | `7608dd52e48af3ce8ae2e0a8343aae125c63b7f4` |
| 原生 Flow* 对照 | `b85a3211748cb77b736fe4ad42ee02d8d2b81148` |

本轮 reference 是**本轮起点的现有 CPU 运行路径**，先核对其与 4939fb 既有数值行为一致。不能把早期未优化的 `f34b5fa`、包含全部导出的 6083 秒或旧失败方法当作有利的性能分母。

已知本轮之前的工作目录：

```text
/srv/local/shengenli/strict_backend_reevaluation_20260907T061907Z/our_audit
/srv/local/shengenli/xiangru_adoption_20260907T032448Z/our_optimized
/srv/local/shengenli/torch_tm_flowpipe_c4_perf_batch_20260829
```

只读核对这些位置；不重新盘点十个 clone，不全盘搜索、不读取进程完整参数或凭据。不要在旧目录 pull、reset、clean、stash、重装包或重编译。

在已确认的我们自己的审计仓库建立新 worktree；新增 worktree 的正常 Git 元数据变化与修改旧工作树文件分开记录：

```bash
set -euo pipefail
AUDIT=/srv/local/shengenli/strict_backend_reevaluation_20260907T061907Z/our_audit
BASE=f23c375f7935dc2f85a37a0ad69814ab052168bb
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/srv/local/shengenli/our_solver_replay_performance_${RUN_ID}"
BRANCH="codex/our-solver-prepared-replay-performance-${RUN_ID}"
git -C "$AUDIT" cat-file -e "$BASE^{commit}"
mkdir -p "$ROOT"
git -C "$AUDIT" worktree add -b "$BRANCH" "$ROOT/repo" "$BASE"
```

若该目录不存在，只到上述已知的我们自己仓库中定位/fetch 固定提交，再创建 worktree；不得切换成某个未知的更新 main。保存实际 ROOT、BRANCH、BASE、Python 路径、源码导入路径、线程数、CPU 亲和性。

本轮允许修改**我们自己的性能实现**，但不允许改变其数学合同或覆盖旧结果。默认路径不变，新路径 opt-in。每次正式运行固定 clean scientific SHA；报告封装 SHA 另列。

### 1.2 复用既有材料，不重新跑接纳

直接读取并保留：

```text
docs/backend_reevaluation/REPORT_PLAIN_CHINESE.md
docs/backend_reevaluation/REPAIR_REUSE_PLAN.md
docs/backend_reevaluation/GOAL_AUDIT.md
artifacts/runs/backend_reevaluation_20260907T061907Z/
artifacts/runs/xiangru_adoption_20260907T032448Z/
```

旧完整宽度、原生 Flow* 时间和反例标为 `REUSED`。旧 verifier 最多验证一次，不为启动本轮重跑第三方测试、旧 GPU 反例或全部旧性能实验。

---

## 2. 先处理两件小事；不能把它们变成整轮任务

### 2.1 明确测试现状

上一轮：根目录 954 passed / 2 skipped；七组隔离测试 59 passed / 9 skipped / 1 failed。不同项目用例合计 **1013 passed / 11 skipped / 1 failed**；新增九项已在根目录内，不重复相加。

唯一已知失败是：

```text
test_frozen_historical_result_manifest_is_unchanged
experiments/three_way_common_contract/results/20260724T132534Z
```

这是历史资料目录缺失，不是观察到数值不相等。不得删断言、改期望或临时改成 skip 来取得全绿。

仅可从已知自己的旧工作树位置读取该**精确目录**，核对测试原先固定的摘要后复制到新测试环境。优先查：

```text
/srv/local/shengenli/torch_tm_flowpipe_three_tool_study/experiments/three_way_common_contract/results/20260724T132534Z
```

找不到就保持原失败与 `historical_fixture_available=false`，不开展全盘恢复，也不阻断与它无关的当前数值测试及性能实验。最终明确区分“当前求解器回归通过”和“全项目是否全绿”。任何新增或不同原因失败都不能归为这个例外。

### 2.2 把新教训加入我们自己的小回归

我们在上一轮仅对历史传播和归一化做了三方检查，第三个端点例子只比较了当前候选与旧修复，不能默认我们的端点函数免检。

在我们的实际端点代入入口加入一个小测试：

```text
P(t) = -1 + 100*t
ordinary remainder = [0,0]
t = 实际 binary64(0.01)
精确结果 = 3/144115188075855872
```

必须检查**代入后的多项式加完整余项**是否包含该值，不能只对比 point coefficient，也不能使用十进制精确 1/100 偷换输入。端点例子不算增加第三个动力学 benchmark。

历史矩阵和归一化原有回归直接复用。若我们的真实入口出现精确漏包，保存反例并以 `OUR_REFERENCE_DEFECT_FOUND__PERFORMANCE_STOP` 停止性能改动；不能一边修正确性一边宣布 bitwise 提速。

---

## 3. 两道题保持不变：比较的是同样的工作

从上一轮 `MATCHED_CONTRACTS.json`、实际执行合同及 `run_ours.py` 读取全部参数，不凭报告摘要重建。

| 项目 | Brusselator | Van der Pol |
|---|---|---|
| 方程 | x'=1+x(xy-4), y'=x(3-xy) | x'=y, y'=y-x-x²y |
| 初始 x | [1.48,1.52] | [1.1,1.4] |
| 初始 y | [2.98,3.02] | [2.35,2.45] |
| 阶数 | 6 | 4 |
| 固定步长 | 0.02 | 0.01 |
| 完整步数 / 标称时域 | 1000 / T20 | 1000 / T10 |
| 历史容量 | 1000 | 100 |
| 预设余项 | 每分量 [-1e-4,1e-4] | 同左 |
| cutoff / validation epsilon | 1e-10 / 1e-12 | 同左 |

CPU float64，B1，原范围求值策略、原细分设置、原舍入位置、原表达式求值顺序均不变。不额外加入时钟，不收紧或修补最终报告区间。

Van der Pol 另保留已有自适应回归：T10，246 accepted / 35 rejected。**它只做能力不回归检查，不能拿 246 步时间与固定 1000 步时间比较。**

实际时间按原 binary64 步长和既有精确时间标签处理，不擅自裁短最后一步。初始区间的十进制/二进制语义及范围策略必须直接继承原执行合同。

---

## 4. 唯一优先假设：同一步里，不变的多项式工作是不是被反复重做？

### 4.1 为什么从这里开始

我们当前每步已经确认安全后，会多次代回方程，把余项继续收紧。这个过程中的**候选多项式保持不变，当前余项却会改变**。

已审查的源码入口：

```text
src/torch_tm_flowpipe/batched_dense_tm.py
  _post_accept_refine_raw_remainder
  _dense_flowstar_raw_compat_image
```

VDP 专用 `canonical_factorized_joint_closure` 已有 `_VDPRefinementStaticCache`；一般 `ordered_terms` 路径没有使用这个专用缓存。循环会反复构造 candidate_with_remainder 并调用余项图像计算。

这给出一个**待检验的执行效率假设**：一般路径可能每轮重新建立相同的多项式运算图、乘法索引、保留系数和与余项无关的范围。不要提前宣布它是主要瓶颈，也不要将 VDP 专用公式直接搬给 Brusselator。

### 4.2 先读旧 profile，再补最少量的新测量

从自己的性能分支已有 profile/runner 开始，不新建全套 profiling 平台。注意旧 profile 可能来自 SR 张量优化之前，不能直接把旧百分比当作当前事实。

对当前 CPU 优化版本测量四类短窗口：

```text
Brusselator：起始 20 步；历史队列较长时连续 20 步
Van der Pol：起始 20 步；跨 SR100 清空边界的固定步长窗口
```

优先使用已存在且验证过的 accepted checkpoint。没有必要 checkpoint 时，只生成各系统一次连续前缀，在接受边界保存；不猜造 queue 或从普通区间重启假装同一 checkpoint。

必要时另外读取三个真实 accepted step 的“候选多项式 + 原余项收紧输入序列”，用于同一步重复执行诊断。诊断可在固定输入序列上做，但最终性能必须来自原样完整求解器。

### 4.3 时间如何划分

避免重复累计嵌套调用的 inclusive 时间。用互斥计时和 profiler 辅助区分：

- 构造本步候选多项式；
- 第一次安全检查；
- 接受后继续收紧的所有计算；
- 多项式乘法、截断及与当前余项无关的范围求值；
- 依赖当前余项的交叉项和包含检查；
- 跨步历史传播和边界缩放；
- Python/Interval/TaylorModel 构造、张量分配与调度；
- 观察、导出、写文件。

若按父阶段计时，子分类只是该阶段的分解，不能与父阶段相加。记录调用次数、每步 replay 数、queue 长度，以及 production 总时间。

计时跑不打开 profiler 的 shape/stack tracing。诊断跑可以打开，但不能把诊断开销算成优化收益；PyTorch profiler 的 tracing 本身会增加开销并可能保留张量引用。

### 4.4 是否授权实现

输出一页结论和一个小表：重复的工作是什么、为什么输入没变、占多少时间、预计可省多少。

用

```text
S_pred = 1 / ((1-f) + f/s)
```

区分被优化部分占比 f、该部分预期加速 s、整体预期加速。局部函数快 3 倍不等于完整程序快 3 倍；永远不能超过 `1/(1-f)` 的上限。

只有该机制覆盖实际热点且有希望达到至少约 1.5 倍完整求解提速，才投入实现。没有达到时，可再检查**一个**由当前 profile 指出的通用连续执行阶段；同样先给出依赖分类和整体收益预测。两个候选都不值得做，就交付明确的瓶颈定位，不堆多个小优化，也不转去写 CUDA。

---

## 5. 只实施一个机制：本步准备一次，变动余项每轮重新算

若第 4 节支持优先假设，建立本步内的 prepared evaluation plan。名字可按仓库习惯选择；它是执行计划，不是新 Taylor-model 数学方法，不命名成 C5/C6。

逻辑：

```text
本步候选多项式固定
→ 准备不依赖当前余项的计算结果/索引一次
→ 第一次接受仍按原逻辑
→ 每次收紧只输入当前余项，重算所有依赖它的项
→ 仍执行原来的逐操作向外舍入、包含检查和停止判断
→ 丢弃本步计划；下一接受边界或重试按新输入准备
```

### 5.1 静态与动态必须按实际数据流证明

可能静态的项目：基底和指数表、固定表达式结构、乘法/积分索引、候选多项式系数、只由该多项式决定的乘积/截断项及相应范围。

必然动态的项目：当前余项、乘法中的余项交叉项、依赖余项的范围、当前 proposal、subset margin、是否接受和停止、最后一次提交的误差分解。

**仅因为某项叫 polynomial/range/cutoff，不能把它判为静态。** 支持集合、cutoff 决策、细分方案或中间系数只要依赖当前余项或混合状态，就必须重新算或使用旧路径。静态的误差值也可能每次输出都需要保留一次：缓存的是数值工作，不是取消它的记账。

第一版计划生命周期限定在**单个固定候选的步骤/尝试内**，不做跨步全局数值缓存。历史状态、缩放、步长改变后不得复用旧计划。可继续复用已有、已经证明安全的通用基底索引缓存。

### 5.2 必须保持的语义

- 原求和次序与每个 outward rounding 位置不变；不把顺序和改成未证明的 reduction。
- 阶数、cutoff、range policy、validation epsilon 不变。
- 第一次接受/拒绝、每轮 proposal、提交顺序和 stop reason 不变。
- 不减少 replay 次数，也不修改 STOP_RATIO、491 上限或原子整向量提交。
- 最后余项、误差分解及历史 owner 对应最后一次真正提交；不遗漏、不重复入账。
- 不改变 SR queue 政策、队列清空、失败回滚和 checkpoint 恢复。
- 计划的不可变数组与动态缓冲区分离，禁止用 in-place 写回污染前一轮或另一次尝试。
- 新模式关闭时逐位保持原行为；原 scalar/reference 路径保留。
- 原 VDP 专用保护不放宽；对它没有收益就走原路径，不为“通用”改已有公式。

这轮只使用我们自己的 Python/PyTorch 代码，不引入第三方引擎、C++ 求解器绑定或新的 CUDA kernel。可让数据布局未来方便批量化，但不同时开发完整 batch scheduler。

### 5.3 计划接口至少要检查

在创建与使用边界检查实际依赖：ODE 结构、order、basis、步长与时间表、domain、range policy、cutoff、验证参数、候选多项式身份及其他确实使用的本步状态。

这些检查不能每轮通过大量全量哈希消耗掉收益。可在准备阶段核验并持有不可变对象，使用明确的本步有效期；测试中主动变更其中一项验证拒绝或重新准备。不得依赖 Python id() 对可变 tensor 内容作正确性保证。

---

## 6. 局部正确性：先证明复用没有改变计算

至少覆盖：

1. 一般多项式 RHS 的 1D/2D/3D 小例子；Fraction 精确期望与安全包含检查。
2. 同一候选、多组不断变化的余项；新旧每轮 proposal 与最终误差分解逐位相同。
3. 固定真实步骤的 8/9 次等原有 replay 序列；不得硬编码实际次数。
4. 余项从较宽变窄、改变符号、包含零、接近抵消、极小数和非有限输入。
5. 改变候选系数、domain、步长、order、cutoff 或 range policy，旧计划必须失效。
6. subset 失败和 evaluation error，保留原最后 certified vector，失败尝试不污染历史。
7. 缓冲区别名、重复使用计划、前一步计划误用于下一步，均不能静默出错。
8. 新模式关闭精确相同；checkpoint/resume 和 VDP 队列清空零回归。

局部 oracle 证明的是被优化操作的合同；新旧输出相同证明行为保持。两者都不能被写成全求解器形式证明。

若为了快必须改变求和次序、舍入模型或误差大小，本轮停止这项实现。不要把“差异很小”当作语义不变；这将是另一个需单独授权的算法任务。

---

## 7. 从短测量到完整求解，必须有真实结果

### 7.1 计时口径

reference 与 optimized 都从 clean 固定源码启动，使用相同 py11、单线程、CPU affinity、配置与 initial state。确认它们实际导入的是各自路径，不能两个 runner 都误导入旧 editable 包。

正式 solver 时间包括每步计划准备、动态计算、验证、reset 和必要的运行开销。不能把新计划的准备时间移到计时外，却把旧方法对应计算留在计时内。进程启动、独立导出、文件序列化另列。

所有正式计时均关闭诊断 profiler；预热及环境检查不篡改解题状态。对照顺序交替，不终止其他用户任务。环境有明显争用就注明，不能声称独占机器。

### 7.2 小规模计时

从相同初态/检查点比较：

```text
Brusselator 起始 100 步：reference/optimized 各至少 3 次
Brusselator 较晚连续 20 步：各至少 3 次
VDP 起始 100 步：各至少 3 次
```

报告每次原始时间、median 与离散程度；不要只取最快一次。不再要求所有前缀都必须恰好 2 倍，但必须说明早晚效果差别。

### 7.3 完整正确性与速度

短跑语义全部通过后，reference 与 optimized 各做一次新鲜完整固定步长运行：

```text
Brusselator：1000 accepted，T20
VDP：1000 accepted，T10
```

逐步比较四项上下界、candidate/final remainder、replay 次数/停止理由；关键边界保存 queue/checkpoint 比较，覆盖 VDP SR100 reset 与 Brusselator SR1000 最终 reset。不只核对最后一行。

再做一次优化版本 VDP 原自适应合同回归，和已有证据比较：T10，246/35，原轨迹输出与决策不变。此项不是新性能对照。

原生 Flow* 的上轮完整原始结果直接复用，标为 REUSED；本轮没有改双方题目，不必再新 clone/编译 Flow* 来重复同一张表。

若完整单次加速接近门槛、与短跑统计矛盾，补一次配对确认；无重复证据时只能称“单次完整运行观察到”，不得称稳定性能保证。

### 7.4 最终判定

正确性为硬前提：任何新数值差异、丢包、状态污染或新增回归失败，都不能宣布优化可用。

性能目标：

- 主目标 Brusselator 完整生产求解观察到至少 1.5× 加速，并有重复短跑支持，不靠导出差异。
- VDP 三次前缀中位数不慢超过 5%；不要求已有缓存的 VDP 也达到 1.5×。
- 峰值内存不超过 reference 的 1.5×；如预先有明确的时间换空间方案须在实现前记录，不事后放宽。
- 全程四项安全范围保持旧参考结果，不用更窄或更松的区间换速度。

达标：`OUR_SOLVER_EXECUTION_OPTIMIZED__SEMANTICS_PRESERVED`。

正确但未达目标：`OUR_SOLVER_OPTIMIZATION_CORRECT__LIMITED_SPEED_GAIN`，保留原默认、如实报告收益，不叠第二种优化。

没有值得做的单一机制：`OUR_SOLVER_HOTSPOT_LOCALIZED__NO_CHANGE_AUTHORIZED`，需给出具体时间归因和为何结构性重构才有意义，不能只写“不知道”。

新路径有错误：`OUR_SOLVER_OPTIMIZATION_REGRESSION_STOP`。

原 CPU 已有漏包：`OUR_REFERENCE_DEFECT_FOUND__PERFORMANCE_STOP`。

---

## 8. 报告必须回答什么

正文用简单语言，内部函数名和 SHA 放附录。不得全篇只写 C3/C4、gate、owner、image 而不解释。

至少回答：

1. 这轮有没有让我们的程序真的变快，还是只是定位问题？
2. 原来在重复做什么，为何可以不重做？哪些误差必须每轮重新算？
3. 改前改后 T20/T10 实际各需多少秒？早期和晚期分别怎样？
4. 整个过程中 x/y 的终点范围和小段范围是否保持不变？
5. 相对 Flow* 的宽度和速度差距还剩多少？旧数据与新数据分开。
6. 历史文件缺失测试是否恢复；当前数值测试是否全过；有哪些未执行或已知失败？
7. 新布局未来为什么有利于批量/GPU，而不只是另一套 Python 包装？
8. 下一轮只该处理哪个剩余执行瓶颈？不要自动恢复第三方接纳或控制器任务。

至少提供一张 runtime 前后表、一张剩余时间分布图，以及“全程四项范围未改变”的机器表。复用旧宽度图可标明 REUSED，不必再排一份几十页报告。

---

## 9. 精简交付与验证

建议文档：

```text
docs/our_solver_performance/REPORT_PLAIN_CHINESE.md
docs/our_solver_performance/REPLAY_DEPENDENCY_CONTRACT.md
```

建议结果：

```text
artifacts/runs/our_solver_replay_performance_<RUN_ID>/
  SOURCE_MAP.json
  EXECUTION_CONTRACT.json
  profile_windows.csv
  repeated_work_attribution.csv
  optimization_decision.json
  local_equivalence.json
  prefix_timings.csv
  full_timings.csv
  full_width_equivalence.csv
  adaptive_regression.json
  test_results.json
  RESULT.json
  raw_minimal/
  tests/
  SHA256SUMS
```

复用旧 runner、canonical exporter 和 verifier 公共函数；新增检查器只需重算本轮数值保持、计时比率及决策。不要开发通用审计平台或给所有 tensor 做永久日志。

增加必要篡改测试：一项区间端点、一项 replay 决策、一项 source 身份、一项运行时间或最终状态被改后应拒绝。所有测试日志和退出码原样保留。

完整测试至少运行根目录与项目规定的隔离组。历史资料例外严格限于第 2.1 节；不能借它掩盖新增失败。报告总计去重，不把专项测试再次计入总 passed。

独立 clone 验证已提交证据即可，明确不等于又跑全部长实验。只 push 我们自己的新分支；不自动合并 main、不改变默认后端、不向第三方发送材料。

最终回复给用户：一句白话结论、两系统速度表、数值保持与测试状态、实现 SHA、证据 SHA、远端分支和准确结果路径。

---

## 10. 本轮不做的事与下一步边界

禁止：继续检查/修补当前 Xiangru/Huan，重新审论文，找历史 dirty patch，扩第三方预算，第三个 benchmark，CROWN/NN controller，SR 容量调参，增加 refinement 次数，新的 C5/C6，降低精度或取消向外舍入，完整 CUDA/全 batch 调度重写。

当前候选接纳结论保持不变；将来只有维护者提供了修复版本且用户另行授权，才重开。

本轮结束后，根据新 profile 决定下一阶段：可复用的张量执行计划是否已具备真正批量化条件；或者数据表示是否仍让 Python 小操作主导。**这轮的代码必须在我们的求解器里产生可测价值，不再把“又审完一个别人模块”当作项目进展。**

GPU 是最终目标，不是当前暂停之后被放弃的目标。先把“同样计算反复在 Python 调度”的问题处理清楚，是为了后续把真正值得并行的计算搬上 GPU；不是要求 CPU B1 必须完全追平 C++ Flow* 才允许讨论 GPU。

## 来源索引（本任务依据）

- `f23c375.../docs/backend_reevaluation/REPORT_PLAIN_CHINESE.md`
- `f23c375.../docs/backend_reevaluation/REPAIR_REUSE_PLAN.md`
- `f23c375.../artifacts/runs/backend_reevaluation_20260907T061907Z/tests/TEST_EXECUTIONS.json`
- `4939fb.../src/torch_tm_flowpipe/batched_dense_tm.py` 中 `_post_accept_refine_raw_remainder`
- `3ca31b8.../artifacts/runs/xiangru_adoption_20260907T032448Z/` 的冻结运行、全程宽度和单次计时
- PyTorch 官方 profiler 文档的 tracing overhead 说明；按服务器已安装版本使用支持的 API，不升级环境来适配新文档。
