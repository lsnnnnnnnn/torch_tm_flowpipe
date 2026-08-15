# 下一轮大 Goal：闭合 T=1/T=3 剩余差距，并验证固定形状 G2 共享源列 carry

把下面整段作为一次新的 Codex 大任务执行。不要把它拆成只写报告、只加 metadata 或只做小型 toy test 的任务。

---

## 0. 仓库、起点与最终目标

仓库：`git@github.com:lsnnnnnnnn/torch_tm_flowpipe.git`

从远端分支 `codex/t1-t3-width-causal-symbolic-carry-20260814` 的最终 tip
`771948ef7592d5b5c81e35e36ba4aa067674821e` 建立新分支：

`codex/vdp-residual-gap-g2-shared-column-carry-20260815`

上一轮科学代码 SHA 是：

`8ac2962bf691dd81ae5d06a9ea146bb011b7ec42`

本项目的原始目标不是复刻一个 Python CPU 版 Flow\*，也不是为了给某个 diagnostic queue 找一个好看的局部数字。原始目标是：

1. 在 PyTorch 原生 Taylor-model flowpipe 中，理解并减小长时间传播时的不确定性依赖丢失；
2. 保持 sound containment、可微/批处理/GPU 架构方向和真实生产 Picard 消费路径；
3. 在完全匹配的 Van der Pol 合同下，最终至少恢复并超过 legacy 的连续 validated horizon，并以 `T=10` 为终局目标；
4. 所有结论必须区分 formal/discrete、directed numerical、deterministic empirical 和 sampling-only 证据，不得把局部缩窄写成 soundness、总根因闭合或 GPU speedup。

这一轮的大目标是：

> **先把 T=1/T=3 的 Flow\*–Torch 总差距分解到可以被同一 prestate 的真实算子反事实验证；再只实现一个固定形状、两代共享源变量的 G2 候选，检验它能否显著减少剩余差距并恢复 native horizon。**

不要先写候选、后找理由。先完成归因门，再进入 G2。

---

## 1. 必须先纠正的当前科学状态

不得沿用上一轮的：

`T1_T3_WIDTH_CAUSE_CLOSED__EARLY_GAP_IMPROVED__TERMINAL_STILL_OPEN`

作为无条件总根因结论。先新增一份 erratum/status 文档，不重写或删除旧证据，当前准确状态应是：

`BOUNDED_SOURCE_MATERIALIZATION_CONTRIBUTION_CONFIRMED__TOTAL_T1_T3_CAUSE_OPEN__G1_TERMINAL_REGRESSION`

必须在新报告和 `handoff.md` 当前状态区明确记录以下事实：

- G1 的 affine source 确实进入了下一步真实 dense Picard 的 polynomial input，不是 metadata；payload tamper 会改变 consumer，metadata tamper 不会。
- G1 的固定步长缩窄数字真实，但只占 legacy 相对 Flow\* excess 的很小部分：
  - T=1：约 `0.147%–0.465%`；
  - T=3：约 `0.211%–0.227%`；
  - T=6.32：约 `0.261%–0.307%`。
- 第一步在没有旧 `J/Phi_L` source 穿过 accepted boundary 时，Torch 与 Flow\* 已经存在局部 operator 差异；因此 accepted-boundary materialization 不能单独解释全部 T=1 差距。
- 在 frozen legacy prestates 的局部反事实中，G1 source identity 明显优于把同一 affine source set 完全 ordinary-materialize，但现有 legacy rebox 在这些隔离 prestates 上仍略优于 G1；因此不能把 G1 全轨迹的微小缩窄直接等同于“已重现 Flow\* carry”。
- native G1 停在 `6.382737816137232`，早于 legacy 的 `6.397083942944808`；G1 不是可接受的生产改进。
- 所谓 13 个“independent micro-oracles”直接使用了项目的 `Polynomial`、`TaylorModel`、dense Picard 和 source-ledger 实现。应改称 project-core-backed exact/discrete micro-oracles；它们有价值，但不是独立实现。
- Flow\* lossless same-prestate operator cell 仍是 `UNAVAILABLE`，所以总根因不得标记为 closed。

`handoff.md` 的“当前结论”只允许有一个，不要继续在文件顶部宣布新候选、后面又保留 `NO_FIX_AUTHORIZED` 而不解释作用域。历史结论移入明确标记的 historical section 或链接到旧报告；当前状态必须无歧义。

---

## 2. 冻结合同：任何比较都不得偷偷换条件

必须冻结并把机器可读 contract 写进证据包：

- ODE：标准二维 Van der Pol，`x' = y`，`y' = (1-x^2)y-x`；
- 初始集合：与 Flow\* exact-decimal outward initialization 完全一致；
- order：complete O4；
- fixed schedule 主比较：`h=0.01`；
- target remainder、cutoff、range method、validator、endpoint/tube 语义与上一轮 authoritative lane 相同；
- CPU float64 B1 是 soundness/scientific authoritative lane；
- fixed-schedule curve 与 native adaptive horizon 必须分开报告；
- 不允许通过改 order、target remainder、cutoff、adaptive step policy、range subdivision budget 或初始箱来让候选“成功”；
- 不允许把 Flow\* coordinate projection 图上的 apparent zero 当原始宽度；必须使用 raw lower/upper subtraction；
- 不允许混用 endpoint width、segment-tube width、remainder width、normal scale 或图像像素宽度。

必须保留 legacy、G1 和未来 G2 三条独立 lane。默认 legacy 行为不得改变；G1 只作为冻结研究基线。

---

## 3. Gate A：真正闭合 T=1/T=3 的剩余差距归因

### A1. 建立 lossless common prestate/operator fixture

必须把以下状态作为一等机器可读对象保存，而不是 component box：

- 时间 `t` 与步长 `h` 的精确/hex 表示；
- center、normalized base TM/right map、domain；
- Flow\* 的 `Phi_L/J` 共享源列及其 generation/order；
- 普通 remainder 与 complete-O4 owner ledger，二者不得合并成一个无来源区间；
- 所有 polynomial term 的 canonical exponent/coefficient；
- validator target 与 range contract；
- 序列化前后 byte/canonical-hash equality。

优先直接在 Flow\* 进程内从 native checkpoint 做 one-step replay；不要因为跨语言序列化麻烦就退化成 box adapter。若需要 common schema，必须证明 Flow\* native state → schema → Flow\* one-step 的结果与不经过 schema 的 native one-step 一致。

### A2. 必须补齐四个 same-prestate operator cells

在 step 1、step 2、T=1 前、T=3 前和 T=6.32 前至少五个冻结位置，运行并保存：

1. Flow\* operator on Flow\* prestate；
2. Torch operator on Torch prestate；
3. Torch operator on Flow\* prestate；
4. Flow\* operator on Torch prestate。

这里的“on”必须保留完整相关性对象，不能只把 lower/upper component box 喂给另一边。

若两个工具的内部维度不能直接一致，定义一个双方都能无损消费的 affine-column + ordinary-ledger 子合同，并分别证明 round trip 与 native one-step parity。不得静默补零、丢 queue、把 shared column 拆成独立区间或用 sampled fit 替代。

如果任一交叉 cell 仍做不到，结论必须是：

`LOSSLESS_CROSS_OPERATOR_CELL_UNAVAILABLE__TOTAL_CAUSE_OPEN`

可以继续做 Torch 内部机制研究，但不允许写 `T1_T3_WIDTH_CAUSE_CLOSED`。

### A3. 把差距分成可反事实验证的机制，不要强行做虚假的可加分摊

至少区分：

- local Picard polynomial construction/grouping；
- local remainder refinement/complete ledger；
- local range extraction/validator；
- accepted-boundary parameterization/rebox；
- old-source retirement/nonlinear collapse；
- adaptive scheduler feedback（只在 native lane）。

对每个冻结位置保存 actual-consumer 输出：canonical coefficients、owner ledger、raw Picard image、subset margin、endpoint/tube raw bounds、下一 boundary state hash。

这些机制一般非线性且次序相关。不要声称简单相减能得到唯一可加百分比。采用预注册的 sequential intervention order，并至少增加反向 order；若两种 order 不同，报告 interaction 区间。可以使用完整 factorial/Shapley 分解，但必须在运行前冻结因素和顺序，且保留所有原始 cell。

最终必须回答：

1. 第一步差异中，local operator 的哪一层首先产生实际 coefficient/range delta？
2. 到 T=1 和 T=3 时，boundary materialization、local operator 和二者 interaction 分别能解释多少 observed excess？
3. 仍未被反事实复现的 residual 是多少？
4. 哪个机制控制 T=6.32 之后 validator margin 的恶化？

只有当相同 prestate 的完整 operator cells 能在 raw bounds/ledger 上重现 observed delta，且没有未解释 residual 时，才允许使用 `CAUSE_CLOSED`。

---

## 4. Gate B：把 G1 的 `ordinary mass = 2.1933445893` 拆到真正 owner

上一轮只把主导项命名为 `ordinary_parameterization_and_retired_nonlinear_source_collapse`，这个粒度不够指导下一步。

在真实 accepted-boundary transition 中增加只读 owner accounting，至少按以下维度记录：

- validated dense ledger 的每个原始 category；
- insertion truncation、cutoff、integration overflow；
- old-source term 按 source generation；
- old-source term按 component 和 monomial class：linear、quadratic、cubic、含 `x^2y` 的 mixed path；
- 含 oldest source 与 current source 的 mixed term；
- scale/range reboxing 产生的额外宽度；
- fresh structured source mass 与 ordinary collapsed mass。

每个 owner 必须有 canonical support hash、outward interval、width 和 containment witness。若 owner intervals 因 dependency/interaction 不能相加成 total，要明确保存 overlap/interaction，不得伪造 exact additive ledger。

在 step 1→2、T=1 前、T=3 前、T=6.32 前及 G1 自己的 terminal prestate 上，对每个主要 owner 做 actual-consumer intervention：

- 保持其他输入完全相同；
- 只把该 owner 从 ordinary interval 改为共享 source variable；
- 运行真实下一步 dense Picard；
- 记录 raw image、subset margin、endpoint/tube、payload hash；
- payload tamper 必须改变 consumer，metadata tamper 必须不改变。

这一门的产物必须能明确指出：G2 应保留哪类 source、保留几代，以及为什么不是事后按最好数字选出来的。

---

## 5. Gate C：只实现一个 G2——固定 `3d` 变量、两代共享源 carry

若 Gate A/B 的证据没有否定该方向，则实现唯一候选：

`normalized_insertion_bounded_shared_source_o4_g2`

不得同时实现多个 K、多个 generation 数或多个 owner 组合再挑最好结果。

### C1. 固定形状

对 state dimension `d`，accepted boundary 永远使用：

- `d` 个 normalized base variables；
- `d` 个 retained previous-generation source variables；
- `d` 个 fresh complete-ledger source variables；

总计固定 `3d` variables；VDP 中始终为 6。不得随 horizon 增长。

### C2. 共享列语义

同一个 source variable 必须能同时出现在 x、y 两个 state polynomial 中。它代表一列跨 component、跨 nonlinear path 的共享不确定性，而不是每次出现都重新 materialize 的独立 interval。

fresh ledger 初次仍可按 component 建立 `d` 个 source；经过一个 Picard generation 后，它们在两个 component 中形成共享列，下一 boundary 必须继续保留这一完整 polynomial identity一代。

### C3. 两代轮换与 collapse

在 boundary `n+1`：

- 保留 current-generation source-bearing polynomial terms；
- 添加 boundary `n+1` 的 fresh source bank；
- 只 retire oldest generation；
- 所有含 oldest source 的项，包括 oldest×current mixed terms，必须 canonical merge 后一次 outward-collapse 到 ordinary ledger；
- 不含 oldest、只含 current source 的项继续保留；
- source bank 轮换后 shape 仍为 `3d`。

不得把 retained source 的贡献同时放进 scale/remainder，防止 double count。不得使用 fallback；合同不满足时 fail closed。

### C4. accepted/rejected atomicity

- 只有 accepted step 可以 commit generation rotation；
- rejected retry 前后的 state object/fingerprint 必须完全相同；
- checkpoint/resume 必须恢复所有 source IDs、generation、coefficients、owner ledger 和 hashes；
- fresh run 与 resume run 必须 bitwise/canonical equal。

### C5. 独立 correctness oracle

新增一个真正独立的 oracle executable/script。它不得 import `torch_tm_flowpipe` 的 `Polynomial`、`TaylorModel`、dense Picard、source ledger 或 interval 运算实现。

独立 oracle 至少用 exact rational arithmetic 自己实现：

- canonical monomial merge；
- affine shared-column substitution；
- `x^2y` 的二次/三次 shared-source 展开；
- 两代 source rotation；
- oldest/current mixed-term retirement；
- degree-4 truncation owner accounting；
- rejected retry immutability fixture；
- exported black-box coefficient table 与独立结果逐项相等。

区间 containment 使用独立 exact Bernstein/rational enclosure 或另一套明确 directed-rounding 实现；sampling 只能作为补充，不能标为 proof。

项目内部 tests 可以继续使用 project core，但必须与独立 oracle 分开命名和报告。

---

## 6. 实验矩阵

所有运行都必须 fresh，不得从上轮 CSV 复制候选结果。

### Fixed schedule，`h=0.01`

对 legacy、G1、G2 运行：

- T=0.1、0.5、1、2、3、6.32；
- 保存每一步四个 raw channel：endpoint x/y、segment-tube x/y；
- 保存 excess、ratio、increment、candidate-vs-baseline reduction；
- 保存 owner mass、active variables、collapse count、term count、runtime 和 peak memory；
- ratio crossings 在 `1.1/1.5/2/5` 上单独报告。

### Native adaptive

对 legacy、G1、G2 fresh 运行：

- T=1、3、6、6.5、7.5、10；
- 保存 accepted/rejected attempts、最后连续 validated time、失败前 prestate、raw Picard image、subset margin 和 rejection reason；
- 不得把“请求 T=10”写成“到达 T=10”。

### Batch/GPU

- CPU float64 B1 是权威科学结果；
- CUDA 只报告 implementation consistency 和真实性能；
- 若测试 V100/A100，必须同步计时并分别报告 H2D、kernel、D2H、完整 solver runtime 和 transfer count；
- 只有完整多步 solver 在同一 batch workload 上更快，才能宣称 GPU speedup；kernel-only throughput 不得外推。

---

## 7. 预注册成功、部分成功和失败条件

### Correctness gate（任何一项失败都不得跑科学 acceptance）

- 独立 exact oracle 全通过；
- project unit/property tests 全通过；
- complete ledger containment、source rotation、no-double-count、retry atomicity、checkpoint/resume 全通过；
- 默认 legacy bitwise/canonical parity；
- 三类 evidence tamper 均被 verifier 拒绝；
- fresh detached clone 重现 compileall、focused、full tests 和 verifier；
- CUDA 结果不得被描述为 formal directed rounding。

### G2 production-success gate

只有同时满足以下条件，才可称 G2 成功：

1. fixed T=1 四个 channel 均不比 G1 宽；
2. fixed T=3 和 T=6.32 四个 channel 相对 legacy excess 均至少减少 10%；
3. 任何 `0.01` fixed prefix 上不出现新的 containment/validator failure；
4. native 连续 validated horizon 至少不低于 legacy 的 `6.397083942944808`；
5. 失败模式若仍存在，最后 y subset margin 必须优于 legacy，且不能通过改 solver contract 获得；
6. 无任何 soundness oracle 反例。

如果只比 G1 好、但仍早于 legacy 停止，结论只能是：

`G2_MECHANISM_IMPROVED__PRODUCTION_GATE_NOT_MET`

如果 native 到达 T=10，且所有 correctness gate 通过，才允许：

`G2_VDP_T10_VALIDATED`

如果 G2 更宽、提前失败、变量/term 爆炸或独立 oracle 失败，停止调参并发布负结果：

`G2_SHARED_COLUMN_CARRY_REJECTED`

不得换成三代、K16、不同 owner subset 或不同 solver 参数继续扫。

### Total-cause closure gate

G2 成功不自动等于根因闭合。只有 Gate A 的 lossless 四个 cross-operator cells 齐全，observed raw deltas 被完整反事实重现，且未解释 residual 为零或被严格包含在已声明的 rounding envelope 中，才能写：

`T1_T3_TOTAL_CAUSE_CLOSED`

否则保持 `TOTAL_T1_T3_CAUSE_OPEN`。

---

## 8. 明确禁止重复的路线

- 不要回到完整 endpoint-polynomial carry；它曾在 `t=0.0434546875` 失败。
- 不要复活 `normalized_insertion_structured_total_delta_k16` 或调 K；它虽 replay 307/307，但 terminal y margin 仍为 `-1.999959117e-5`。
- 不要做只存在 diagnostics/queue metadata、没有进入下一 dense Picard polynomial input 的“candidate”。
- 不要用 ordinary component boxes 冒充 Flow\* 的 `Phi_L/J` 共享状态。
- 不要通过增加 range subdivision、改变 remainder target、降低步长、改变 order/cutoff 来掩盖 carry 问题。
- 不要用采样点未发现反例来证明 containment。
- 不要把微小正 reduction、局部 kernel throughput 或更小的某一个 channel 写成整体成功。
- 不要在 verifier 中硬编码期望 conclusion 后再把 verifier PASS 当成科学结论证明；verifier 必须从原始机器可读证据重算判定。

---

## 9. 必需产物

至少交付：

1. `docs/VDP_G1_CAUSAL_CLAIM_ERRATUM_20260815.md`
2. `docs/VDP_T1_T3_RESIDUAL_CAUSAL_DECOMPOSITION_20260815.md`
3. `docs/COMPLETE_O4_G2_SHARED_COLUMN_CONTRACT_20260815.md`
4. `docs/VDP_G2_SHARED_COLUMN_RESULT_20260815.md`
5. lossless common prestate schema、round-trip tests 和五个 checkpoint 的四格 operator matrix；
6. owner-resolved ordinary-mass ledger 与 actual-consumer intervention CSV/JSON；
7. 独立 exact oracle 源码及 raw output；
8. legacy/G1/G2 fixed 和 native 原始 traces；
9. CPU/GPU 性能与 transfer ledger；
10. 小于 25 MiB 的自包含 evidence package，含 source snapshot、commands、environment、raw minimal fixtures、manifest、`SHA256SUMS`、verifier 和 tamper tests；
11. 更新后的单一当前状态 `handoff.md`；
12. fresh-clone acceptance JSON。

证据包必须可从仓库内已有文件重算关键结论，不能只记录 `/srv/local/...` 路径而不包含相应 raw/minimal fixture。所有表中的 width 都必须能从同包 lower/upper 独立复算。

报告结尾必须用普通语言分别回答：

- 这一轮真正证明了什么；
- 哪些仍只是机制证据或经验现象；
- T=1/T=3 总差距还剩多少未解释；
- G2 是否比 G1 和 legacy 真正更好；
- native 是否超过 `6.397083942944808`，是否到达 T=10；
- 下一步是否还值得继续 shared-source carry，还是应该转向 local Picard/range operator。

---

## 10. Git 与发布要求

- 从指定 final tip 开新分支，先记录 base SHA；
- 科学代码、测试、报告和 evidence 在一个可复现 scientific SHA 上冻结；
- push scientific SHA；
- 从 GitHub fresh clone，detached checkout 该 SHA，运行 compileall、focused/full tests、独立 oracle、package verifier；
- 再提交只包含 attestation/handoff/evidence-finalization 的 child commit，不得修改 `src/`、核心 `experiments/` 或 tests；
- 再 push，并用 `git ls-remote` 验证远端 tip；
- 最终工作树必须 clean；
- 最终回复给出 branch、scientific SHA、attestation SHA、远端 SHA、测试数量、四个 checkpoint 表、native horizon 表和唯一当前结论。

不要用“做了很多工作”作为完成标准。完成标准是：**归因结论不再越界、G2 进入真实 consumer、独立 soundness oracle 成立、固定曲线有显著收益，并且 native horizon 至少恢复 legacy；否则诚实冻结负结果。**
