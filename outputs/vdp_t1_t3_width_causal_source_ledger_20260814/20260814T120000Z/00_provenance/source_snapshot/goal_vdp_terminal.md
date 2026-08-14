# Codex 大 Goal：解释并修复 Torch 在 T≈1、T≈3 开始累积的 Flow* 宽度差距

## 一句话目标

不要再把主要精力放在第一个小步的正确性整理上。本轮必须直接回答并推进用户最初的问题：

> 在相同 Van der Pol、相同初始集合、相同 order/step/validator 合同下，为什么 Torch 在 `T≈1` 已经比 Flow* 宽、到 `T≈3` 差距明显增大，之后被非线性反馈放大并在约 `t=6.397` 停止；然后实现一个有独立 soundness contract、会被下一步实际 Picard 输入消费、保留跨步依赖的最小 tensor-friendly symbolic/source carry，使这条宽度增长曲线发生可归因的改善。

“代码写完”“测试通过”“第一步更窄”“T=10 跑完”都不能单独算成功。成功必须同时包括：长期差距的逐步因果解释、sound 的跨步机制、真实 consumer 证明和冻结合同上的 T=1/T=3 对照。

## 仓库和起点

- repo：`git@github.com:lsnnnnnnnn/torch_tm_flowpipe.git`
- 从远端分支 `codex/step1-stage-oracle-sound-carry-candidate-20260813` 开始
- 起点 tip：`08dd34e44f7cfc3fb456bf947959304599f07451`
- 新分支：`codex/t1-t3-width-causal-symbolic-carry-20260814`
- pinned Flow* SHA：`b85a3211748cb77b736fe4ad42ee02d8d2b81148`

先核对远端 SHA、工作树、Flow* identity 和既有 evidence checksum。若身份漂移，停止并报告，不得在不一致的基线上继续。

## 用户真正关心的现象

本轮所有设计都要围绕下面这条已观察到的曲线，不要转向旁支：

1. 在匹配的 fixed-schedule/common-prefix 比较中，`T=1` 时 Torch 的四个 width excess 已全部为正，约 `0.00272–0.00890`。
2. 到 `T=3`，四个 excess 已约为 `0.0470–0.0488`。
3. 到约 `t=6.32`，excess 增长到 `0.763–1.468`，更宽的 prestate 进入 Van der Pol 非线性 `x²y`，形成进一步反馈。
4. 权威 Torch complete-O4 baseline 通过 307 个 accepted segments，到 `t=6.397083942944808`，下一步 fail closed；stock Flow* 到 T=10。
5. 图中看似 Flow* width 到 0 的点已经核实：它只是坐标投影下的正最小值，四个 Flow* minimum 都大于 `0.0086`，并非数值 0、除零或爆宽原因。本轮只做一次原始数据回归确认，不要重新花一轮研究它。

最终报告必须用普通语言解释：为什么差距在早期已出现、为什么到 T=3 累积、为什么后期加速，以及候选究竟改变了哪一个跨步集合表示。

## 已闭合事实：继承，不得反复重做

1. stock benchmark、copied probe 与实际 `ode.reach` 路径在 pinned VDP 合同上已有 1000-step observer 等价证据。
2. step 1 common mathematical input 已闭合；complete-O4 support 35 项。
3. 独立 Fraction oracle 已证明第 4 次 Flow* staged 与 Torch complete-O4 精确 Picard 多项式相等：x 13 项、y 18 项。
4. degree-100 exact Taylor/Cauchy/sensitivity oracle 已证明当前两边 step-1 最终 segment/endpoint box 都包含真解；Torch step-1 endpoint 较窄在该合同上是 sound 的。
5. 精确十进制合同与 point binary64 初始化有约 `1e-16` 的端点缺口。这是必须修的小型接口问题，但它的数量级不能解释 T=1/T=3 的明显差距。
6. 图中 Flow* apparent zero 已闭合为 `FLOWSTAR_WIDTH_MINIMUM_POSITIVE_NOT_NUMERICALLY_NEAR_ZERO`。
7. 已排除：输出解析错误、坐标 roundtrip 遗漏、outer remainder 重复加、单纯 degree>4 overflow 主导、epsilon/padding 主导。
8. 已观察的主要机制候选是：同一 parameterization remainder 在不同 nonlinear monomial paths 中被独立 intervalize，造成相关性丢失；这不是同一余项被显式重复加两次。
9. R35 same-prestate 诊断中，before step 320 的同一 prestate 下 CDR 接受、CNI 拒绝；但该结果不是 authoritative dense complete-O4/Flow* carry parity，不能直接当最终修复证明。

## 已失败或不足的方案：禁止换名字重跑

### 1. 完整 endpoint polynomial 直接跨步 carry

`normalized_insertion_complete_polynomial` 已试过。它在单边界 bit-preserving，但 fresh horizon 只到约 `0.04345`，反而破坏短期证书。原因是它把全部历史依赖和 accumulated remainder 直接送入后续 nonlinear Picard products，没有 Flow* 式 structured preconditioning/source separation。

不得再次实现“完整多项式原样 clone 到下一步”并宣称是新 symbolic carry。

### 2. `normalized_insertion_structured_total_delta_k16`

它通过 307/307 历史 accepted-prefix replay，但在冻结 terminal prestate 上仍与 ordinary-only/historical controls 一样拒绝，y margin 约 `-1.99996e-5`，没有闭合 terminal。

不得简单改 K16→K32、调宽 target、降低 h_min、加 fallback 或换 padding 后继续宣传。

### 3. Horner 和旧 diagnostic queue

Horner 只把 accepted steps 632→636；diagnostic queue 从 step 2 改 segment width，但不改 endpoint、scale 或 horizon。它们可作为定位对照，不能直接晋升 production fix。

## 本轮的核心科学问题

必须回答以下五个问题：

1. 从相同初始集合出发，Torch 相比 Flow* 的每一步“新增 excess”到底在哪个边界阶段产生？
2. `T=1` 的早期差距与 `T=3` 的累计差距，是否由同一 source/materialization 机制反复作用？
3. 为什么同一机制在 `t≈6.32` 附近被 `x²y` 放大到 order-one，而不是线性累积？
4. Flow* 跨 accepted boundary 保留的 `Phi_L/J`、ordinary remainder、nonlinear residual 和 composition grouping，怎样抽象成不依赖 VDP、可 tensorize、可独立验证的 complete-O4 source-ledger contract？
5. 采用该 contract 后，T=1/T=3 的每步 width increment 是否因同一 prestate 的因果干预而下降，而不是因为改了 schedule、h、target、range reporting 或输出定义？

## Gate A：冻结两条比较合同，不混淆语义

### A1：主因果合同

主实验使用历史宽度图对应的 `binary64_literal_matched_contract`，以便一字不差复现 T=1/T=3 现象。固定：

- Van der Pol `x'=y`, `y'=(1-x²)y-x`
- 初始程序 literals 对应的 x/y boxes
- order 4、complete total-degree support
- cutoff `1e-10`
- candidate remainder `1e-4`
- 相同 fixed schedule/common accepted prefix
- float64、CPU B1 为权威科学 lane
- endpoint、last-segment tube、prefix tube 分开，禁止混用

如果原生 Flow* 与 Torch adaptive schedule 不同：

- 用 fixed-schedule/common-prefix lane 做因果宽度比较；
- 用各自 native adaptive lane报告实际 horizon；
- 不得把 native 与 fixed-schedule 数字放进同一个 ratio。

### A2：精确输入清洁 lane

对 `exact_decimal_contract` 实现最小 outward 初始化补偿：Torch 与 Flow* audit driver 的 normalized initial TM 都包含 x `[11/10,7/5]`、y `[47/20,49/20]`。使用 exact-rational witness 和实际 consumer 测试。

这只是 Gate A 的短前置，不是本轮主产出。修完并跑 step-1 containment 后立即返回 T=1/T=3 主线；不得再扩展成新的十进制/MPFR 大项目。

## Gate B：重建 T=0→1→3→6.32 的逐 accepted-step 宽度账本

从原始 artifact 重新生成一张机器可读 joined ledger。每个共同 accepted time、x/y、endpoint/segment 分别记录：

- raw lower、upper、width，禁止只存 ratio；
- Flow* width、Torch width、absolute excess、relative ratio；
- 本步相对上一步的 width increment；
- prestate center/scale；
- retained polynomial natural range；
- polynomial truncation、integration overflow、cutoff；
- parameterization/right-map ordinary remainder；
- raw Picard image、refinement image、subset margin；
- endpoint `tau=h` substitution/merge 后 range；
- accepted/rejected、retry、h、order；
- live source count、source fingerprint、structured/ordinary width mass；
- 实际 source file/function/line 和 observer on/off parity。

至少强制输出这些 checkpoint：

- step 1 和 step 2；
- T 最接近 `0.5`、`1.0`、`2.0`、`3.0` 的共同 accepted rows；
- first ratio > `1.1`、`1.5`、`2`、`5` 的 rows；
- `t≈4.205867`、`t≈6.225303`；
- `t≈6.32`；
- Torch terminal prestate `6.397083942944808` 及 rejected attempt。

必须重新得到或解释不一致：T=1 excess `0.00272–0.00890`、T=3 excess约 `0.0470–0.0488`、6.32 excess `0.763–1.468`。

Flow* apparent-zero 只做以下回归：从 raw lo/hi 直接重算四个 minimum，确认均大于 `0.0086`，并在图上同时画 lower/upper/width。通过后标记 closed，不再把它列为爆宽候选。

## Gate C：定位“每一步新增 excess”的第一消费点

不能只比较最终 width，也不能把第一处 bit difference 当成因果结论。对以下三个边界窗口做 actual consumer audit：

- early：step 1→2；
- accumulated：T≈1 和 T≈3 前一 accepted boundary；
- nonlinear acceleration：t≈6.32 前一 boundary。

每个窗口冻结同一个完整 prestate、同一个 h/order/target/cutoff，依次做：

1. legacy Torch normalized insertion；
2. 把同一 represented set materialize 成 ordinary-only 的控制；
3. 保留 source identities/linear images、但 outward-collapse nonlinear residual 的 source-ledger operator；
4. 能无损表达时运行 Flow* operator；不能表达时明确 `UNAVAILABLE`，不得用有损 adapter 伪装。

对每项做 tamper-consumer test：改变真正 carry payload 必须改变下一步实际 Picard input；只改变 metadata 不得改变结果，也不得骗过 verifier。

本 Gate 要输出的是“第一个改变下一步 raw Picard image/subset margin 的 carry field/source class”，而不是笼统写 `dependency loss`。

## Gate D：先定义 authoritative complete-O4 source-ledger contract

这是本轮最重要的理论/实现桥梁。不要直接抄 Flow* 内部对象，也不要继续依赖固定 R35 adapter。为 dense complete-O4 定义通用 boundary state：

1. accepted endpoint retained polynomial，变量/域/总阶明确；
2. `tau=h` 精确 substitution 与同 source monomial merge；
3. ordinary remainder `R_o`；
4. 有身份的 structured sources `z_k∈[-1,1]` 及其系数/linear images；
5. nonlinear source interactions 的 outward residual ledger；
6. truncation、integration overflow、cutoff 各自 ownership；
7. normalization/rebase map；
8. source create/propagate/consume/merge/retire lineage；
9. retry atomicity：rejected candidate 不得修改 accepted boundary state；
10. deterministic finite-shape collapse rule。

必须给出集合语义，而不只是字段定义。例如：

`X = P(old_sources) + R_o + Σ Phi_k z_k + R_nonlinear`

并对 boundary operator 的每个动作证明：新表示包含旧 accepted endpoint represented set。证明可以由 exact-rational fixtures + outward MPFR/interval obligations组成，不要求 proof assistant，但不能靠 sampling。

### 有限 source policy

不能靠无限增加 source 数逃避依赖爆炸。优先研究“one-generation/有限代 structured source carry”：

- 保留会在下一步不同 nonlinear paths 中重复出现的历史 source identity；
- linear part 分别传播一次；
- 新的 `integration_overflow`、`polynomial_truncation` 或命名 nonlinear residual 可保留一代；
- 到期/超 K 时确定性 outward-collapse 到 ordinary remainder；
- collapse 前后必须有 containment witness；
- 不允许按 VDP 状态编号或经验大小偷选 source。

该 contract 必须能表达 dense complete-O4 actual path，不得再出现 `DENSE_CNI_PARITY_NOT_EXPRESSIBLE` 后仍继续跑候选。

## Gate E：独立 micro-oracle 验证

在任何 VDP horizon run 前，至少完成：

1. affine source 经过两步仍保留精确系数；
2. shared-source cancellation：`x=1+u,y=1+u` 时 carry lane 保留 `x-y=0` 相关性，legacy rebox 显示预期 excess；
3. quadratic/cubic `x²y` fixture，验证同一 source 在不同 monomial paths 不会被无身份地重复 intervalize；
4. ordinary×structured、structured×structured、非对称 remainder；
5. degree-4 truncation、integration overflow、cutoff ownership；
6. duplicate exponent merge、`tau=h` substitution；
7. source retire/collapse 前后 containment；
8. retry/rejection 后 state/source hash 不变；
9. B1/B8/B64 batch permutation equivariance；
10. CPU/CUDA 同决策检查；CUDA 只作实现一致性，不宣称形式化 directed rounding；
11. exact-decimal initialization compensation 被实际路径消费；
12. observer default-off 与 legacy bitwise/hex parity。

micro-oracle 必须检查集合包含和真实消费者，不能只检查 queue 长度、source id 非空或最终采样无 violation。

## Gate F：只实现一个被证据授权的 candidate

Gate B–E 全部通过后，才实现一个 production candidate。候选名称和单一 changed variable 必须预注册。

候选应当是：

> dense complete-O4 bounded source-ledger carry：accepted endpoint 在边界上保留必要的 source ownership/linear images，nonlinear residual outward-collapse，下一步 Picard 实际消费该 state；不是完整 endpoint polynomial 生搬硬套，也不是只在 metadata 里放 queue。

除 carry mode 外全部冻结：ODE、初始集合、order/support、candidate remainder、cutoff、validator、h_min/h_max、scheduler、range policy、endpoint/tube semantics、dtype/device。

禁止：

- 同时改 Horner、validator、target、cutoff、h_min 或 adaptive controller；
- fallback 到 legacy 后仍记 candidate accepted；
- 看到 terminal 失败后再试 K32/K64、加 padding或调阈值；
- 只凭更窄或更长 horizon 宣称 sound；
- benchmark-specific VDP 分支。

若 Gate D/E 无法闭合，正确结果是 `AUTHORITATIVE_SOURCE_LEDGER_CONTRACT_INCOMPLETE__NO_CANDIDATE`，不要实现半成品。

## Gate G：直接检验 T=1/T=3 主问题

候选通过 micro-oracle 后，按下面顺序运行 paired experiment，每行只允许 carry mode 不同：

1. frozen same-prestate one-step checkpoints：step 2 前、T≈1 前、T≈3 前、t≈6.32 前、terminal 前；
2. fresh fixed-schedule requests：T=`0.1,0.5,1,2,3`；
3. fresh native adaptive requests：T=`1,3,6,6.5`；
4. 只有前面全部无 soundness/consumer/ledger failure，才运行 T=`7.5,10`。

主结果不是只有最终 horizon。必须比较 legacy vs candidate：

- T=1 四个 width excess；
- T=3 四个 width excess；
- 每步 excess increment 曲线；
- first ratio >1.1/1.5/2/5 的时间是否后移；
- t≈6.32 的 prestate scale、dominant parameterization-remainder source、raw y image、subset margin；
- highest continuously validated time；
- accepted/rejected schedule；
- source 数量、collapse 次数、峰值内存、CPU runtime。

必须回答两个因果问题：

1. 在相同 frozen prestate 上，source-ledger carry 是否直接减小下一步 raw Picard/remainder width并改善 margin？
2. fresh T=1/T=3 的改善是否由这些逐步变化累计得到，而不是 schedule 不同造成？

如果只延长 horizon、但 T=1/T=3 excess 不变，不能声称解决了原问题；如果 T=1/T=3 变好但 terminal 不变，也要如实报告“早期依赖积累改善、terminal 仍由其他项限制”。

## Gate H：性能和 PyTorch/GPU目标

用户最初用 PyTorch 的动机包括 tensorization、batching 和未来 GPU 加速，不是做一个更慢的 Python Flow* 复制品。但性能必须在正确性后测：

- B1 CPU 是科学正确性权威 lane；
- B8/B64/B256/B512 测 carry kernel 的 batch scaling；
- 有 V100 时测 CUDA，并同步计时、记录 host/device transfer 和峰值显存；
- 分开报告 carry kernel、Picard/range/validation、adaptive outer loop；
- 不得把单步 batched kernel 吞吐写成完整 multi-step solver speedup；
- 当前已有 V100 在相关 O4 路径比 CPU 慢的负结果，除非新结果在相同合同下推翻，否则不得声称 GPU speedup；
- 目标是给出瓶颈与可 tensorize 结构，而不是强行证明 Torch 比 C++ Flow* 快。

## Gate I：结论、证据和 push

### 强制报告

更新 `handoff.md` 和一份白话报告，至少写清：

1. T=1、T=3、6.32 的真实宽度差及其逐步来源；
2. apparent Flow* zero 为什么不是 0；
3. 第一个被因果干预验证的跨步 source/consumer；
4. 新 source-ledger contract 的集合语义；
5. 为什么它不同于已经失败的完整多项式 carry 和 total-delta K16；
6. frozen-prestate 与 fresh-horizon 分别说明什么；
7. T=1/T=3 是否真正改善；
8. terminal/T=10 是否改善；
9. 哪些是 formal/discrete、directed-numerical、deterministic empirical、sampling-only；
10. 仍未闭合的问题和下一步。

### 测试和证据

- `compileall`
- focused tests
- full pytest
- independent oracle/verifier
- tamper tests
- candidate-vs-legacy regression
- detached fresh-clone scientific SHA 验收
- clean worktree

evidence package 保留源码、命令、机器可读 ledger、必要压缩 trace、图和校验和；不要提交 clone、ELF、重复原始日志。新增 package 目标小于 25 MiB，超过须逐文件说明。

先提交 scientific commit 并 push；从 GitHub detached fresh clone 科学 SHA 做验收；再用单独 attestation commit 记录结果。最终报告 branch、scientific SHA、attestation/final SHA、远端一致性和 evidence path。

## 允许的最终结论

只能选择与证据一致的一项：

1. `T1_T3_WIDTH_CAUSE_CLOSED__SOURCE_LEDGER_CARRY_ACCEPTED`
2. `T1_T3_WIDTH_CAUSE_CLOSED__EARLY_GAP_IMPROVED__TERMINAL_STILL_OPEN`
3. `T1_T3_WIDTH_CAUSE_CLOSED__CANDIDATE_SOUND_BUT_NO_CAUSAL_IMPROVEMENT`
4. `AUTHORITATIVE_SOURCE_LEDGER_CONTRACT_INCOMPLETE__NO_CANDIDATE`
5. `SOUNDNESS_OR_CONSUMER_GATE_FAILED__HORIZON_RUNS_NOT_AUTHORIZED`

不要再以“精确输入修复完成”“step 1 oracle 完成”“测试全部通过”作为本轮主结论。它们只是前置门。主结论必须回到用户最开始的问题：T=1 的差距从哪里来，为什么 T=3 变大，为什么后期爆开，以及跨步依赖保留是否真实改变了这条曲线。
