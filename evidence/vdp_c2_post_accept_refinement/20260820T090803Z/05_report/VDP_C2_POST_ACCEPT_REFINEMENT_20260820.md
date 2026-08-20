# VDP C2 接受后余项联合收缩报告

## 结论

最终决策：`C2_SOUND_AND_T1_T3_TARGET_MET__T10_NOT_MET`

C2 在冻结 VDP 合约上通过 soundness、provenance、同输入因果门、T1/T3 八项原始目标门、T6.32 四通道不回归门、native 下限门与 CPU runtime 门。native C2 从 C1 的 6.589638579126679 提升到 6.714914669607182，但没有达到 T10，因此不宣称 T10 成功。

科学代码 SHA 为 `29c9ee8f1fe96b860052b86a2b37d79a37bbb2ca`；全部正式运行来自该 SHA 的干净 detached worktree，tracked diff SHA256 为 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

## 冻结合约与阶段边界

冻结系统为 `x'=y; y'=y-x-x^2*y`，二状态、binary64、初值盒 `x∈[1.1,1.4]`、`y∈[2.35,2.45]`，完整总次数 O4，固定组 `h=0.01`，目标余项半径 `1e-4`，cutoff `1e-10`，validation epsilon `1e-12`，当前 dependency-preserving normal insertion 与原 scheduler 不变。

C2 只对这个精确结构指纹开放：状态维度、每项幂次、系数符号和每个 binary64 bit 均匹配才进入 C1/C2 专用路径；任意其他 `PolynomialODE`、子类、自定义 evaluator、项、系数或维度均 fail-closed。legacy、H1/H2 与 C1 默认路径保持原行为，C2 是显式 opt-in mode。

阶段定义如下：

- 首次验证：完全沿用 C1 的候选多项式、R1 自映射、接受判定与 scheduler 决策。C2 不参与是否接受当前步。
- post-accept refinement：仅在首次验证已接受后，以最后保留余项为输入重算同一 sound map；只有 x/y 整个向量的所有分量同时有限且逐分量 subset 时才原子提交。首次提案失败时结果与 C1 相同，任何后续混合成功/失败也不允许部分提交。
- 发布：segment、endpoint、下一步 reset carry 与最终 remainder ledger 都取最后一次已提交余项；候选多项式不变。

## Flow* 语义与独立 oracle

Flow* 源码固定在 `b85a3211748cb77b736fe4ad42ee02d8d2b81148`。提取的宏与循环语义为 `MAX_REFINEMENT_STEPS=490`、零基含端点最多 491 次 replay、停止比 `0.99`、`widthRatio=new_width/old_width`；旧宽度为零时遵循源端 NaN/比较语义。Flow* 原实现可以按分量顺序更新；Torch C2 采用更强的整向量原子规则，失败时保留旧 enclosure。

独立 oracle 不导入 production Bernstein 转换器，使用每个输入 binary64 的精确 `Fraction`、稀疏张量积 Bernstein 与严格 containment 检查。CPU binary64 outward 路径加该独立 Fraction/Bernstein oracle 构成 soundness 依据。

## Gate A：同输入因果门

Gate A 通过，且没有 repair、endpoint tightening、sampling containment 或 metadata 数值决策：

- C1/C2 首次接受决定相同；候选多项式 bitwise 相同，SHA256 为 `805e9b93c16285662bc077e84d4e266cefeea30fd8daebee62688c4ed91e9b22`。
- C2 原子提交 4 次；每次提交均被独立 exact oracle 包含，停止原因为 `stop_ratio`，最终 ledger 精确指向第 4 次提交。
- x raw image：C1 `2.0000060000000026e-6`，C2 `1.0674997070575039e-7`，相对 Flow* runtime cross-check `3.30228001377617e-7`，消除 C1/Flow* 差距的 `113.383697%`，超过 50% 门槛。
- y raw image：C1 `2.6020086299715395e-5`，C2 `1.067417758129762e-5`，无回归。
- step1 四个发布通道全部不宽于 C1：endpoint x `0.3008966037317445≤0.30089849698777377`，endpoint y `0.12130934198471711≤0.12132468789343599`，segment x `0.32524532867626627≤0.32524722193229555`，segment y `0.14940252431227655≤0.14941787022099542`。

6 个 tamper 案例全部被拒绝：删除、重排、伪造 subset、复用迭代号、修改 stop reason、替换最终 ledger。

## 固定时域科学矩阵

表中的“原目标消除率”为 `(legacy-C2)/(legacy-Flow*)`；“C2 增量”为 `C1-C2`。所有宽度都是原始 endpoint 或最后 segment 通道，不使用修复 hull 或 tightening。

| 时域/通道 | Flow* | legacy | C1 | C2 | C2 增量 | 原目标消除率 |
|---|---:|---:|---:|---:|---:|---:|
| T1 endpoint x | 0.0795178281 | 0.0879559235 | 0.0876341299 | 0.0869021026 | 0.0007320273 | 12.4888% |
| T1 endpoint y | 0.1115769156 | 0.1142921745 | 0.1138421712 | 0.1133006471 | 0.0005415240 | 36.5169% |
| T1 segment x | 0.0837525716 | 0.0922215177 | 0.0918996051 | 0.0911674344 | 0.0007321707 | 12.4465% |
| T1 segment y | 0.1196669930 | 0.1285652386 | 0.1280913649 | 0.1275210031 | 0.0005703617 | 11.7353% |
| T3 endpoint x | 0.1385053267 | 0.1872595897 | 0.1824935618 | 0.1764272636 | 0.0060662982 | 22.2182% |
| T3 endpoint y | 0.1088516715 | 0.1558642556 | 0.1468418604 | 0.1371266849 | 0.0097151755 | 39.8565% |
| T3 segment x | 0.1639208754 | 0.2127350396 | 0.2079674955 | 0.2018992652 | 0.0060682303 | 22.1980% |
| T3 segment y | 0.1256837926 | 0.1727677949 | 0.1637411732 | 0.1540214651 | 0.0097197081 | 39.8146% |
| T6.32 endpoint x | 0.1530755556 | 0.9165121029 | 0.5489073248 | 0.2782209781 | 0.2706863467 | 83.6076% |
| T6.32 endpoint y | 0.1222956280 | 1.5898587283 | 0.7737280035 | 0.2918636063 | 0.4818643972 | 88.4456% |
| T6.32 segment x | 0.1783273000 | 0.9420414425 | 0.5743256105 | 0.3035572989 | 0.2707683115 | 83.6025% |
| T6.32 segment y | 0.1398213090 | 1.6080698025 | 0.7915870848 | 0.3095143915 | 0.4820726933 | 88.4425% |

T1/T3 的 8/8 项均达到至少 10% 原目标消除率；T6.32 的 4/4 项均严格不宽于 C1。固定组全部按请求步数完成，拒绝数均为零。

## native T10 与真实终端诊断

native 新鲜结果为：legacy `6.397083942944808`，C1 `6.589638579126679`，C2 `6.714914669607182`。C2 超过冻结 C1 下限，但没有达到 T10。

诊断直接载入 C2 真实失败前 checkpoint；checkpoint 的有效 reset mode 是 `normalized_insertion_dependency_preserving`。失败尝试 `h=0.003950348390361663`，scheduler 记录下一重试 `0.0019751741951808317<h_min=0.002`，因此分类为 `scheduler_h_min_after_first_self_map_subset_failure`。

首次 image 的限制分量/侧为 y-upper，subset margin `-2.9756743616707034e-6`；x margin 为正。首次 self-map 不是 subset，所以 production 没有进入或提交接受后 refinement。独立 exact oracle 包含该首次 image。若从未提交的 I1 理论上再 replay，y 收缩但 x 提案不 subset，整向量原子规则仍不能提交，故 `another_sound_replay_would_contract=false`。最大 additive ledger 类别为 `composition_overflow`，这只表示区间所有权，不表示因果排名。

## 性能、CPU/V100 与测试

CPU 串行 runtime（legacy/C1/C2）为：T1 `41.929/55.382/69.161s`，T3 `134.954/182.708/217.579s`，T6.32 `300.932/393.877/476.675s`。C2/legacy 比分别为 `1.6495`、`1.6122`、`1.5840`；native 请求比为 `1.2519`，全部不超过 2×。

CPU/V100 T0.1 实现一致性在四个发布宽度上 delta 全为 0，满足 `1e-12`。CPU runtime/RSS 为 `5.803s/422727680 bytes`，V100 为 `17.000s/1020002304 bytes`；V100 在该小批量运行更慢。因此 CUDA 仅用于实现一致性与实测 runtime/memory，不作 directed-rounding soundness 或 speedup 声明。

同一干净 detached SHA 上的 targeted 测试为 `46 passed`；全套为 `825 passed, 2 skipped`。覆盖零/非对称/固定点、混合 subset 原子性、ratio 边界、零宽度、subnormal、cutoff、阶数溢出、非有限值、0/1/上限 replay 次数、checkpoint resume、CPU bitwise determinism、V100 consistency、最终 ledger 与缓存边界。

## 证据与打包策略

证据保留 provenance、历史 verifier 结果、Gate A、exact oracle、Flow* 固定源码契约、逐次 refinement ledger、原 remainder ledger、step1/fixed/native 原始 summaries/attempts/segments/configs/commands/profiles/decisions、terminal checkpoint、terminal diagnostic、JUnit、manifest 与 `SHA256SUMS`。

为避免把 366MB 机械派生 trace 纳入 Git，manifest 明确披露排除 `range_trace.jsonl`、空的 `horner_stage_trace.jsonl` 与 `owner_ledger.jsonl`；上述原始 attempts、segments、两类 ledger、checkpoint 和 run index 均保留。该排除不改变任何科学判断或可验证公式。
