# Codex Goal：减少“本步结果交给下一步”时的对象与小张量开销

日期：2026-09-08  
仓库：`https://github.com/lsnnnnnnnn/torch_tm_flowpipe`

## 0. 本轮要真正完成什么

**保留已经通过全程对照的 prepared replay 优化，在此基础上，优化“每一步结束后，整理结果并交给下一步”的一个主要执行环节。范围、误差处理和步进规则不变；目标是让完整求解继续变快，而不是再改宽度算法，也不是再生成一轮后端审计。**

项目主线仍是：获得可靠、范围足够紧、能够进一步利用批量/GPU 的 PyTorch 多项式动力学求解器。当前工作是改善执行结构，不要求 CPU 先追平 C++ 才能考虑 GPU；也不以这一轮成败决定永久采用或放弃 Huan/Xiangru。

这轮应当交付一次实质的执行改进，或以已经拆开的真实成本明确说明为什么没有值得做的局部改进。不要为了刚好达到某个倍率调整数学设置，也不要把“没有达到 1.5 倍”当成正确实现毫无价值。

### 已接受的上一轮结果

- 端点舍入修复保留，原端点反例正常通过。
- 新增的 prepared replay 在同一次已接受尝试中复用固定多项式工作；当前余项、交叉项、账本和包含判断仍逐轮计算。
- Brusselator 完整求解：1991.943555 → 1454.661096 秒，1.369352×。
- VDP 完整固定步长：711.377453 → 693.657346 秒，1.025546×。
- 两系统各 1000 步的数值对象、四项范围、端点误差和关键状态逐位一致。
- VDP 自适应仍为 246 接受 / 35 拒绝，结果序列与端点修复档案一致。
- Brusselator 100 步三对倍率为 1.3392、1.3324、1.3145；VDP 为 0.9837、0.9861、1.0161。**不能把 VDP 完整单次的 1.026× 写成稳定提速。**
- 正式状态：`REPAIRED_REFERENCE_PRESERVED__USEFUL_SPEEDUP_BELOW_TARGET`。
- 完整测试去重 1108 passed / 11 skipped；独立副本的 105 项是重复检查，不再相加。
- 整个求解器尚未形式化证明；保持“已检查操作与这些运行”的准确范围。

## 1. 固定起点：本轮分母必须已经包含 prepared replay

| 身份 | 固定提交 |
|---|---|
| 本轮仓库与证据父提交 | `f6af6f67565a954d0c68f88a50cc9c181a2d0b08` |
| 当前数值代码 / 上轮正式运行源码 | `1551ab57aef7324f91882beeba9d368f36b3cdd5` |
| 上轮未优化端点修复语义 | `0714e475ed9e73bec31619c9c690d1fd63de3d36` |
| 端点修复证据包 | `7e41f33f515f5315b0dec7003a5b06ed0a79afd5` |
| 历史 Flow* 同合同对照 | `b85a3211748cb77b736fe4ad42ee02d8d2b81148` |

父分支：

```text
codex/repaired-solver-prepared-replay-performance-20260908T034636Z
```

本轮正式比较必须是：

```text
baseline：当前修复代码 + prepared_remainder_replay(True) + 新边界优化关闭
candidate：同一代码和全部设置 + prepared_remainder_replay(True) + 新边界优化开启
```

**不准拿 prepared replay 关闭的约 1992 秒作新优化分母，否则会把上一轮收益再次算进去。**

默认公共入口是否开启 prepared replay 不在本轮偷偷变更。正式实验显式记录两个开关；原 reference 作为回退/对照继续保留。

### 新 worktree

只从已经知道的自己的工作目录读取，不重新盘点 10 个旧 clone，不追随第三方最新分支：

```bash
set -euo pipefail
SRC=/srv/local/shengenli/repaired_solver_performance_20260908T034636Z/repo
BASE=f6af6f67565a954d0c68f88a50cc9c181a2d0b08
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/srv/local/shengenli/boundary_execution_${RUN_ID}"
BRANCH="codex/packed-boundary-execution-${RUN_ID}"
git -C "$SRC" status --short --branch
git -C "$SRC" cat-file -e "$BASE^{commit}"
mkdir -p "$ROOT"
git -C "$SRC" worktree add -b "$BRANCH" "$ROOT/repo" "$BASE"
```

若 SRC 不存在，从已知自己的主仓库取得固定对象即可，不全盘搜索。禁止覆盖、reset、clean、stash 旧工作树，禁止 force push 或合并 main。

继续使用原 py11、CPU binary64、单线程和相同 CPU affinity；不要同时升级依赖。保存真实 ROOT、BRANCH、Python、导入路径和生效配置。

## 2. 起始验证：复用既有证据，不重开安全审计

阅读：

```text
docs/repaired_solver_performance/REPORT_PLAIN_CHINESE.md
docs/repaired_solver_performance/REPLAY_DEPENDENCIES.md
src/torch_tm_flowpipe/prepared_remainder_replay.py
experiments/repaired_solver_performance/profile.py
experiments/repaired_solver_performance/run.py
experiments/repaired_solver_performance/compare.py
experiments/repaired_solver_performance/verify.py
artifacts/runs/repaired_solver_performance_20260908T034636Z/RESULT.json
artifacts/runs/repaired_solver_performance_20260908T034636Z/remaining_hotspots.csv
```

父版本上运行一次相关局部回归；上轮 verifier 按它自己的 README 在父版本独立工作树执行，未来新源代码改变后，不修改旧 verifier 或旧证据让它适应新 SHA。

建议局部集：

```bash
PYTHONPATH=src:.:tests python -m pytest -q \
  tests/test_prepared_remainder_replay.py \
  tests/test_our_reference_endpoint_containment.py \
  tests/test_endpoint_roundoff_repair.py \
  tests/test_endpoint_roundoff_carry.py
```

已恢复的历史夹具、DiffReach 环境按前轮明确记录的路径和摘要复用。无关旧资料缺失单列，不演变为恢复整个历史项目。

## 3. 为什么不继续优化同一段收紧循环

上轮优化后，Brusselator 的固定准备和收紧动态开销在几个窗口中各只有约 3%–4%。同一机制已把中/晚期 20 步的固定多项式乘法从 1440 次降到 120 次，收紧轮数仍为 180。

剩余互斥分类中，`boundary_and_endpoint_correction` 占比：

- Brusselator 早期 63.35%，中期 62.97%，晚期 68.87%；
- VDP 早期 49.90%，跨历史容量边界窗口 54.52%。

**这些是带计时器的代表窗口占比，不是全程无观察器精确份额；更不等于“新增端点误差保护占了 69%”。**该分类混合了边界代入、normal 重建、历史操作、Python 稀疏区间调用及尚未拆分的子过程。

另外，上轮 `profile.py` 在窗口结束后额外推进一个步骤采集 cProfile。例如 Brusselator 981–1000 窗口后的 cProfile 对应第 1001 步；第 1000 步刚清空历史队列。**不能把这一个重建步骤当作第 995 步长队列状态的代表。**保留旧文件原意，本轮补采窗口内真实代表步骤。

## 4. 第一项实质工作：把这个大分类拆开，确认实际生产入口

### 4.1 不重新跑一张庞大的全仓 profile 表

复用上轮完整 checkpoint。Brusselator 优先使用 step100、step980；VDP 优先 step90。若要观察 step995，从 step980 完整状态推进，不从发布的盒子重新初始化。

至少覆盖：

- 两系统初始 1–20 步；
- Brusselator 101–120、981–1000；
- VDP 91–110，跨 SR100 清空；
- 在相同输入上单独观察 Brusselator step995 和 reset 后步骤、VDP reset 前后，明确记录历史长度。

只增加为判定主导边界工作所需的少量观察，不重新寻找 tightness 原因。

### 4.2 分开以下工作，使用互斥时间

```text
A. 稠密 / 稀疏表示转换、项表构造、系数复制
B. 端点点系数代入（幂、乘法、同类项合并）
C. 已修复的独立端点误差计算与账本合计
D. 多项式取范围：相同 domain 的幂、各项区间、按序求和
E. 代入 / 组合 / cutoff 中固定结构的解释与执行
F. 中心、尺度、左右映射构造
G. 历史状态检查、pack/unpack、传播与 commit
H. 其他未归属边界工作
```

记录每类实际秒数、调用次数、创建的 Interval/Tensor 数、必要复制次数。对象分配是嵌在数学操作内的开销时，不能既计入 D 又另计一遍总时间。

计时器必须挂到真实调用绑定：如果 flowpipe 已通过 `from module import function` 保存旧函数引用，只替换 module.function 不会测到它。零调用数首先核对实际入口，不能据此判定这个阶段不存在。

同时给出窗口总时间、各子类总和和剩余值；不能把 inclusive time 直接相加。正式性能运行不带 profiler。

### 4.3 使用现有证据作为线索，而非预先指定根因

上轮额外一步的 cProfile 出现大量 `Interval.__init__`、Tensor.clone、torch.any/min/max/stack 和 `Polynomial.evaluate_interval` 调用。这支持优先检查“许多小对象和重复幂计算”，但不支持直接认定它们就是整个 T20 的唯一瓶颈。

## 5. 第二项实质工作：实现一个有依据的边界执行改进

### 5.1 优先考虑的统一机制

**将选定边界子过程中的逐项 Python Interval/多项式操作，转换为明确的张量输入与可重复使用的结构计划；只减少对象重建、重复的结构准备和不必要的数据往返，保留原数学操作。**

优先候选为：多项式取范围/代入过程的张量化执行与同一不可变输入下的必要结构复用。如果 profile 指向其他子过程，则可以选择 G 的历史 pack/unpack 或 A 的必要转换减量；必须说明选择依据。只选一个连贯机制，不把 A–G 全部重写。

一个机制可能跨多个函数。允许完成其必要调用链，不机械限制“只能改一个函数”。

### 5.2 原型授权必须联系完整时间

在修改生产路径前记录：

- 被选部分占各窗口 baseline 的份额 f；
- 原型单独完成相同工作的倍率 s，包含准备/转换/验证；
- 预测整体倍率 `1 / ((1-f) + f/s)`；
- 无限加速的上限 `1 / (1-f)`。

首选在两个系统共用、占主要成本、预计完整运行能改善至少约 20% 的执行子图。完整 1.5× 是目标，不是给局部实现下的先验保证。

如果第一个方案在短原型中明确无收益，可在同一边界分类内检查一个替代子过程，再选其中一个。不得继续无限试方案，也不得叠第二个优化凑数。确无高杠杆对象时，提交分解和短实验结果，不创建无依据的“fast mode”。

### 5.3 数据与运算合同

必须保留：

1. 实际 binary64 系数、domain、h、幂次和支持集顺序。
2. 原端点误差修正，以及它进入完整余项和下步历史的方式。
3. 乘法、加法和 reduction 的既定顺序与向外舍入位置。
4. 现有 range policy、子域覆盖、cutoff 判定和移入余项的量。
5. 本步普通余项、已传播历史、本步新增误差的各自归属。
6. 真零与极小非零、负数、相消和非有限输入的原有处理。
7. accepted-only commit、拒绝回滚、队列清空和 checkpoint/resume。

不准将一个整体数学上等价但舍入顺序不同的式子，未经独立检查就作为等价优化。不准把更窄输出当作正确证据。

### 5.4 建议布局与缓存范围

仅在获授权子过程内，可以采用：

```text
coefficients: [B, output, terms]
exponents:   [terms, variables]
domain:      [B, variables, lower_or_upper]
```

本轮完整 solver 仍只要求 B1；局部子过程自然支持 B2 时，必须使用不同输入检查任务间无混合。这不是完整 batch 求解器，也不作 GPU 吞吐声明。

结构索引可按支持集、运算顺序、dtype/device 构造；依赖系数、domain、h 或当前 remainder 的数值，只能在这些输入明确不变且私有的生命周期内复用。

**不允许按 Python id 跨步永久缓存数值，不允许只缓存一个已经算过的总区间并用于变动后的输入。**减少 clone 必须有不可变性/独占所有权证据，不能让一个状态被下一步原地污染。

如果向量化变量幂会改变原浮点输出，应保留原求幂路线或回退；不要用“只差一点”取代包含检查。恒定结构计划不是新增范围算法。

### 5.5 与上一轮优化的关系

prepared replay 在本轮 baseline 和 candidate 中均开启。不得修改它的收紧次数、准入合同或既有缓存内容来制造附加收益。边界新开关默认关闭并与 prepared 开关分开；模式身份写入每份运行记录。

## 6. 正确性验证：验证新路径，不重复整个项目的证明史

### 6.1 同一个输入，旧边界与新边界对照

对早期、中期、长历史、清空前后取样的完整边界对象进行对照，检查：

```text
保留系数和单项式支持
每一项必要区间、最终范围
普通余项及其类别总和
端点新增 E 与 cutoff 支付
center/scales、左右映射
历史内容 / generation / reset 状态
最终 next-step 输入
```

CPU binary64 主路径要求逐位一致。主目标是执行结构优化；若必须改变数学计算或舍入语义才能通过，应将该方案留为研究原型，不作为本轮生产优化接纳。

### 6.2 一般局部输入

至少包含：实际 order4/6、不同输出分量、B1/B2、非对称变量范围、正负系数、不同时间值、严重相消、零项/空项、subnormal、溢出、NaN/Inf。

继续使用端点/区间的独立精确有理数 oracle；新测试不能只比较新旧的同一个 helper。若旧 helper 本身有已确认缺陷，不能通过复制旧输出放行。

### 6.3 实际状态传递

新旧两条路径都做连续步骤，确认返回状态实际成为下步输入，特别覆盖：

- VDP step99/100/101；
- Brusselator step999/1000 与随后的一个仅用于回归的边界使用；
- 已有历史的失败尝试，队列和接受状态不变；
- 同一结果重复测量/导出不追加误差；
- checkpoint 恢复后继续一步与不中断运行一致；
- 开关关闭时父版本的数值结果保持。

这些步骤用于本轮改动验证，不把已通过的第三方反例重新扩成审计项目。

## 7. 三套数值合同继续冻结

从已有 `frozen.py`、reference config 和 MATCHED_CONTRACTS 读取，不重新凭摘要拼装参数。

| 项目 | VDP 固定 | Brusselator 固定 |
|---|---|---|
| 方程 | x'=y；y'=y-x-x²y | x'=1+x(xy-4)；y'=x(3-xy) |
| 初始 x/y | [1.1,1.4] / [2.35,2.45] | [1.48,1.52] / [2.98,3.02] |
| order / h / steps | 4 / .01 / 1000 | 6 / .02 / 1000 |
| 历史容量 | 100 | 1000 |
| 余项预算 | 每分量 [-1e-4,1e-4] | 同左 |
| cutoff / validation epsilon | 1e-10 / 1e-12 | 同左 |

同时保留 VDP 原自适应规则（h_min=.002，h_max=.1）。表达式顺序、范围策略、收紧 491 上限 / 0.99 停止条件、完整误差链不变；不用更高阶、更小步、更大余项预算、更多分区或 endpoint ad-hoc tightening 获得成功。

固定步长实际时间为 binary64 h 的精确和；不裁剪最后一步凑十进制整数。自适应时钟与实际步长和分开记录。

## 8. 正式性能与全程范围验证

### 8.1 开发先短跑

先做同一完整 checkpoint 的边界微实验，再做 20 步窗口，再做初始 100 步。每个正式短窗口至少 3 对，交替 baseline/candidate 顺序。比较时不含 profiler、绘图或外部 JSON 导出，**但必须包含新计划构造、必要输入转换和安全检查**。

### 8.2 定稿后完整运行

科学代码定稿、干净提交后，各做：

```text
Brusselator baseline 与 candidate：固定完整 1000 步
VDP baseline 与 candidate：固定完整 1000 步
VDP candidate：原自适应 T10 数值回归
```

不额外反复重跑 old-unrepaired 版本；Flow* 数值和原时间可作为已标注来源的历史对照。只有明确做了新的公平同机计时，才称作新的 Flow* 性能比较。

baseline 与 candidate 优先同一个 scientific SHA，通过显式新开关选择；同时检查 baseline 与父版本已开启 prepared 的保存对象相同。这样不混入脚本或 source 漂移。

### 8.3 比较全过程，不只最终盒子

对两个系统每一步：

- endpoint x/y 上下界；tube x/y 上下界；
- published/common 两种测量结果；
- 完整多项式、余项、端点 E；
- 必要 normal/history 状态和收紧/接受/拒绝序列；
- 时间位置和 checkpoint。

新旧阶段缓存结构和诊断计数可以不同，但数值载荷及 solver 决策必须逐位一致。公开对 Flow* 的宽度关系应因本轮纯执行优化保持；不把对旧值 equality 当作整个 solver 的形式证明。

### 8.4 性能分级

目标：

- Brusselator 完整求解相对**已开启 prepared replay 的新 baseline**争取 ≥1.5×；
- VDP 不出现超过 10% 的稳定减速；
- 峰值 RSS 不超过 baseline 的 1.5×，否则单列并解释；
- 改动确实减少被选工作的调用/对象构造/数据搬运，不是计时边界变化。

完整各一次，只能称单次配对。短窗口用至少三对支持方向性；若完整倍率接近目标或短窗口方向矛盾，在同一冻结源码上补一对反向顺序完整运行。不得选择性删去较慢记录。

达到目标：`BOUNDARY_EXECUTION_PRESERVED__TARGET_SPEEDUP_OBSERVED`。

确有收益但未达目标：`BOUNDARY_EXECUTION_PRESERVED__USEFUL_BELOW_TARGET`。

正确但基本无收益：`BOUNDARY_EXECUTION_PRESERVED__NO_MATERIAL_SPEEDUP`。

有本轮引入的数值/所有权回归：先在本轮修复；尚未修复不得启用候选或宣称成功。

目标不是让 Codex 不断调试到某个倍率。正确且有用的实现可以保留为 opt-in，但结论必须与实测一致。

## 9. 明确禁止的偏离

- 不再优化 prepared replay 的最后几％来凑 1.5×。
- 不删端点保护、不删 nextafter、不跳 validation、不调整 refinement 或历史容量。
- 不用另一种更松/更紧的范围方法替换原方法。
- 不把所有旧证据全量重打包；新 verifier 只检查本轮变化和已引用的基线。
- 不恢复 Huan/Xiangru 后端接纳、全库安全审计或新 clone 盘点。
- 不重写完整 CUDA 引擎，不接 CROWN/NN controller，不增加第三个长跑系统。
- 不把标量循环套一个 B 维容器称为批量加速。
- 不使用随当前余项变化的缓存；不借共享可变存储节省复制。
- 不删除失败断言、降低数学标准或把具体新失败变成 expected failure。

**GPU 路线仍未定；本轮范围限制不代表永久拒绝复用第三方实现。**已经有价值的 GPU 架构、旧严格修复和反例保留。当前选择下一项工作依据是已测主导开销，而不是作者或投入了多少历史代码。

## 10. 交付：少而完整，围绕实际代码与数据

代码置于自己的主仓库新分支，不改 main。结构计划/张量化 helper 是生产模块；实验和验证器放在单独实验目录。尽量复用现有运行、比较和计时器，避免再造大框架。

建议：

```text
docs/boundary_execution/REPORT_PLAIN_CHINESE.md
docs/boundary_execution/NUMERICAL_AND_LIFETIME_CONTRACT.md
experiments/boundary_execution/{profile,run,compare,verify}.py
artifacts/runs/boundary_execution_<RUN_ID>/
```

核心输出：

```text
SOURCE_MAP.json                 # scientific/package、实际开关、来源与配置
boundary_cost_breakdown.csv      # 实际入口、历史长度、子类时间与调用数
implementation_decision.json     # 选择哪个子图、f/s/预测、没有实现什么
local_and_state_equivalence.json # 局部/误差/回滚/恢复对照
full_width_equivalence.csv       # 逐步四通道，两种 observer
runtime_pairs.csv               # 每次原始计时与环境
RESULT.json                     # 从数据计算，含置信边界
raw_minimal/                    # 复算以上判断所需原始对象
tests/                          # 命令、JUnit、退出码
SHA256SUMS
```

测试计数按身份去重；独立副本重复检查不再次相加。新 verifier 从原始数据重算时域、范围一致、速度和状态，而不是只检查已写好的 True。篡改测试限于本轮关键项：数值/误差、实际开关、分母/时间、来源、结果身份；不再扩成通用审计系统。

旧 verifier 仍在自己的固定快照执行。不因为新优化修改源文件，就去篡改旧失败或旧成功档案。

最终 push 前确认 clean，remote/local full SHA 一致；独立副本做证据复核和相关局部测试。独立副本没有重新长跑，就不要宣称重新长跑。

## 11. 白话报告必须回答

1. 这次到底省掉了什么工作？一段没有代码背景的人也能读懂的解释。
2. 上轮 63%–69% 的大分类这次拆成了什么？真正大的是什么，不是什么？
3. 为什么省掉这些准备/对象不会省掉必要的误差和安全判断？
4. 两系统全程范围、时域和自适应行为有没有变？
5. baseline 是否已开启上一轮 prepared replay，有没有重复计算旧收益？
6. 完整时间和短窗口重复结果各是多少，是否达到目标？
7. 现在剩余瓶颈是什么？哪些已经是张量操作，哪些仍然是逐项 Python？
8. 这项结果是否为下一轮批量/GPU 子过程提供了可用接口？若没有，具体还差什么？

结尾只提出一个下一步，不再同时启动自研完整 GPU、第三方接纳和 CPU 多轮微调。CPU 不必先与 Flow* 同速；但下一步必须以实际热点、可靠接口和可测收益为依据。

## 12. 本轮成功的含义

不是又实现 C5/C6，也不是把宽度调得更好看。

**真正成功是：继“同一步固定计算只准备一次”之后，进一步减少跨步交接中大量细碎对象和重复操作，在完整数值行为不变的条件下节省实际求解时间，并明确这些工作将来怎样利用张量批量执行。**

只找到 profile 而没有实质改动时，必须明确是“定位完成、优化未实施”；不能把局部5×写成完整5×。获得1.3×但没达到1.5×时，按实测保留有用成果，不伪造最高状态。