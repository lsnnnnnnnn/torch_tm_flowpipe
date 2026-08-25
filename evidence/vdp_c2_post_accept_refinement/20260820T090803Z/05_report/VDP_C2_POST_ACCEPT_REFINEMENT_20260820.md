# VDP C2 接受后余项联合收缩报告

## 结论

最终决策：`C2_T1_T3_GATE_PASSED__T10_FAILED`

C2 在冻结 VDP 合约上通过 soundness、provenance、同输入因果门、T1/T3 八项原始目标门、T6.32 四通道不回归门、native 推进门与 CPU runtime 门。native C2 从 C1 的 6.589638579126679 推进到 6.714914669607182，但没有达到 T=10，因此只报告 request T=10 未完成，不宣称 reached T=10。

科学代码 SHA 为 `29c9ee8f1fe96b860052b86a2b37d79a37bbb2ca`；全部正式运行来自该 SHA 的干净 detached worktree，tracked diff SHA256 为 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

## 冻结合约与 Phase 0

冻结系统为 `x'=y; y'=y-x-x^2*y`，二状态、binary64、初值盒 `x∈[1.1,1.4]`、`y∈[2.35,2.45]`，完整总次数 O4，固定 `h=0.01`，目标余项半径 `1e-4`，cutoff `1e-10`，validation epsilon `1e-12`，native `h_min=0.002`、`h_max=0.1`，原 scheduler 与 dependency-preserving normal insertion 不变。

C1/C2 只对精确 VDP 结构指纹开放：状态维度、每项幂次、项次序、系数类型和每个 binary64 bit 均匹配才进入专用路径；其他 `PolynomialODE`、子类、自定义 evaluator、额外项、不同系数或维度均 fail closed。专项 binary64 snapshots 证明 default、legacy/H1、H1+H2 与 C1 在 C2 关闭时不变。矩阵字段明确区分 `historical_h1_candidate`、`gate_b_h1_h2_candidate`、`production_c1_candidate` 与 `production_c2_candidate`，verifier 禁止含混的 `candidate_width` 字段。

## Flow* 合同与 C2 语义

Flow* 源码固定在 `b85a3211748cb77b736fe4ad42ee02d8d2b81148`。逐行提取结果为 `MAX_REFINEMENT_STEPS=490`、零基含端点最多 491 次 replay、`STOP_RATIO=0.99`、`old.widthRatio(new)=new_width/old_width`；旧/新宽度同为零时 MPFR 得到 NaN，比较为 false。首次 self-map 失败会在 refinement 前返回。首次成功后的 replay 固定候选多项式、`intDifferences` 与 remainder-independent polynomial ranges，并随当前 remainder 重算 remainder-dependent 项。

Flow* 原实现按分量顺序更新，后续分量失败时可能留下 hybrid vector；Torch C2 采用更强的整向量原子规则。首次验证完全沿用 C1 的候选多项式、R1 自映射、接受判定和 scheduler 决策；只有首次验证接受后才进入 post-accept refinement replay。每个 proposal 仅在 x/y 全部有限且逐分量 subset 时整向量提交，否则保留完整旧向量。segment、endpoint、下一步 reset carry、最终 decomposition 与 remainder ledger 均取最后一次 committed vector。没有 hull/endpoint repair、endpoint tightening、sampling containment、partial/Gauss-Seidel commit 或跨时间常值余项抵消。

## Gate A：同输入因果门

Gate A 通过：

- C1/C2 首次接受决定、候选多项式和 scheduler 相同；候选多项式 SHA256 为 `805e9b93c16285662bc077e84d4e266cefeea30fd8daebee62688c4ed91e9b22`。
- C2 原子提交 4 次；每次均有全分量 subset 证明并被独立 exact `Fraction`/sparse tensor-product Bernstein oracle 包含，最终 ledger 指向第 4 次提交，停止原因为 `stop_ratio`。
- x raw image 从 C1 `2.0000060000000026e-6` 收缩到 C2 `1.0674997070575039e-7`。相对 Flow* runtime cross-check `3.30228001377617e-7`，该变化与消费已收窄 y remainder 一致；C2 甚至越过 Flow* 宽度，剩余差为 `-2.234780306718666e-7`，这不被当作 Flow* 等价性或 soundness 依据。
- y raw image 从 `2.6020086299715395e-5` 收缩到 `1.067417758129762e-5`，无回退；step-1 endpoint/segment 的 x/y 四通道也全部不宽于 C1。
- tamper verifier 的 10 个案例全部被拒绝，其中明确包括交换分量、partial commit、stale remainder-dependent cache 与错误 stop ratio。

CPU float64 outward lane 与独立 exact oracle 是权威 soundness 依据；CUDA 只用于实现一致性和性能测量。

## 固定时域科学矩阵

“恢复率”为 `(legacy-C2)/(legacy-Flow*)`。H1 与 H1+H2 来自已单独验证的历史包；legacy、C1、C2 是同一科学 SHA 的新鲜运行；Flow* 是 pinned stock 宽度。

| 时域/通道 | Flow* | legacy | H1 | H1+H2 | C1 | C2 | 恢复率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 endpoint x | 0.0795178281 | 0.0879559235 | 0.0879439409 | 0.0878008220 | 0.0876341299 | 0.0869021026 | 12.4888% |
| T1 endpoint y | 0.1115769156 | 0.1142921745 | 0.1142770205 | 0.1140809159 | 0.1138421712 | 0.1133006471 | 36.5169% |
| T1 segment x | 0.0837525716 | 0.0922215177 | 0.0922095309 | 0.0920663603 | 0.0918996051 | 0.0911674344 | 12.4465% |
| T1 segment y | 0.1196669930 | 0.1285652386 | 0.1285492647 | 0.1283427687 | 0.1280913649 | 0.1275210031 | 11.7353% |
| T3 endpoint x | 0.1385053267 | 0.1872595897 | 0.1865686202 | 0.1847205549 | 0.1824935618 | 0.1764272636 | 22.2182% |
| T3 endpoint y | 0.1088516715 | 0.1558642556 | 0.1541143565 | 0.1509403569 | 0.1468418604 | 0.1371266849 | 39.8565% |
| T3 segment x | 0.1639208754 | 0.2127350396 | 0.2120438511 | 0.2101951972 | 0.2079674955 | 0.2018992652 | 22.1980% |
| T3 segment y | 0.1256837926 | 0.1727677949 | 0.1710170720 | 0.1678415885 | 0.1637411732 | 0.1540214651 | 39.8146% |
| T6.32 endpoint x | 0.1530755556 | 0.9165121029 | 0.7919255932 | 0.6805171232 | 0.5489073248 | 0.2782209781 | 83.6076% |
| T6.32 endpoint y | 0.1222956280 | 1.5898587283 | 1.2745154469 | 1.0373859831 | 0.7737280035 | 0.2918636063 | 88.4456% |
| T6.32 segment x | 0.1783273000 | 0.9420414425 | 0.8174173573 | 0.7059752141 | 0.5743256105 | 0.3035572989 | 83.6025% |
| T6.32 segment y | 0.1398213090 | 1.6080698025 | 1.2925904925 | 1.0553588399 | 0.7915870848 | 0.3095143915 | 88.4425% |

T1/T3 的 8/8 项恢复率均至少 10%；T6.32 的 4/4 项均严格不宽于 C1。

## 步数、拒绝与性能

| 请求/模式 | accepted | rejected attempts | reached | CPU runtime | C2/legacy |
|---|---:|---:|---:|---:|---:|
| T1 legacy / C1 / C2 | 100 / 100 / 100 | 0 / 0 / 0 | 1 / 1 / 1 | 41.929 / 55.382 / 69.161 s | 1.6495 |
| T3 legacy / C1 / C2 | 300 / 300 / 300 | 0 / 0 / 0 | 3 / 3 / 3 | 134.954 / 182.708 / 217.579 s | 1.6122 |
| T6.32 legacy / C1 / C2 | 632 / 632 / 632 | 0 / 0 / 0 | 6.32 / 6.32 / 6.32 | 300.932 / 393.877 / 476.675 s | 1.5840 |
| native T10 legacy / C1 / C2 | 307 / 247 / 233 | 48 / 39 / 37 | 6.397084 / 6.589639 / 6.714915 | 142.965 / 160.793 / 178.979 s | 1.2519 |

固定时域没有拒绝。native 三条 lane 的首次 scheduler 拒绝都在 `t=0,h=0.1`，limiting component 均为 y；legacy 的 limiting side 为 lower，C1/C2 为 upper。它们随后缩步并继续。最终 C2 在 `t=6.714914669607182,h=0.003950348390361663` 再次由 y-upper 限制，subset margin 为 `-2.9756743616707034e-6`，下一次 scheduler retry 为 `0.0019751741951808317<h_min`，所以停止分类为 `scheduler_h_min_after_first_self_map_subset_failure`。首次 self-map 未 subset，production 没有进入或提交接受后 refinement。

CPU/V100 T0.1 的四个发布宽度 delta 全为 0，满足 `1e-12`。CPU runtime/RSS 为 `5.803s/422727680 bytes`，V100 为 `17.000s/1020002304 bytes`；这只证明普通 float64 实现一致性，不构成 CUDA directed-rounding proof 或 speedup 声明。

## 测试、证据与停止规则

最终加固工作树上的 targeted 测试为 `48 passed`；全套为 `827 passed, 2 skipped`。JUnit 由 manifest/hash verifier 覆盖。专项测试覆盖结构 fail-closed、C2-off binary64 snapshots、首次失败不可救活、原子提交、ratio 边界、零宽度、subnormal、非有限值、0/1/491 replay、最终 decomposition/ledger、checkpoint resume、CPU determinism 与 CUDA consistency。

证据包包含 provenance、历史 H1/H2 与 C1 verifier 结果、Gate A、exact oracle、pinned Flow* 源码摘录与 blob、逐次 refinement/remainder ledgers、step1/fixed/native 原始 summaries/attempts/segments/configs/commands/profiles/decisions、terminal checkpoint/diagnostic、JUnit、manifest 与 `SHA256SUMS`。机械派生的 `range_trace.jsonl`、空 Horner trace 与 owner trace 被显式排除；保留的数据足以重算所有发布判据。

依据停止规则，下一轮只应追踪 terminal y-upper 的最早来源；本轮不并行开启 GPU/CROWN/新 benchmark。
