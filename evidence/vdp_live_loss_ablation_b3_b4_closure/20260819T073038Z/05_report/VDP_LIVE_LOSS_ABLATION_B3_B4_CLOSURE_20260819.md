# VDP live-loss ablation 与 C1 joint closure（2026-08-19）

## 结论

本轮纠正了 H2 的因果归因，并交付了一个 opt-in、sound 的生产候选：
`flowstar_raw_remainder_compat_factorized_joint_closure`。它只替换 C1
`factor_times_y` 的 remainder materialization；候选多项式、validator、scheduler、reset、
endpoint 均未更改，也没有 repaired hull、endpoint repair 或采样式 containment claim。

科学 SHA 为 `dbe03dcdfbf2f36b1d58013373d1d235ace1a48e`。Gate A/B/C 全部通过，
但总体 T=1/T=3 成功门槛没有全部通过，T=10 stretch goal 也失败。因此准确判定是：

> C1 operator sound 且在生产矩阵中有实质收益；T6.32、native floor、runtime 通过；
> T1/T3 全四通道 10% 总目标失败；T10 stretch 失败。

## 必须分开的五种“第一/最大”

自动 production-event ledger 给出的结论是：

| 概念 | 结果 | 含义 |
|---|---|---|
| syntactic first strict surplus | `torch.attempt1.ordinary.c0.before_validation_eps` | 更早出现，但 raw-compat 数值路径会 discard，不能解释最终宽度 |
| first live strict surplus | `torch.i4.c1.y_rhs.y_minus_x` | 到 subset 有 consumer chain，但同输入 isolated marginal 为零 |
| first live material surplus | `torch.i4.c1.y_rhs.x_squared_generic` | 第一个具有正 same-input marginal 的 live stage |
| largest causal marginal contributor | `torch.i4.c1.y_rhs.distributed_final` | Gate B byte-identical counterfactual 中，对最终 subset 宽度贡献最大 |
| native terminal largest additive ledger category | `composition_overflow` | C1 完整 joint residual 的所有权类别，不是因果排名 |

因此，`raw.B1.x_squared` 既不是严格意义的 syntactic first，也不是最大 causal
contributor。旧的硬编码 first-loss 归因不再成立。

## Gate A：真实 consumer chain

审计从实际生产调用记录了 30 个严格有序事件，包含源码函数/行、父子 stage、production
interval、精确 binary64-rational/Bernstein necessary enclosure、surplus、discard site、
validation-eps payment 和到最终 subset 的 consumer chain。10 个 eps payment 均为真实、唯一
执行事件，最终组合 enclosure 包含完整精确链。

Verifier 的五种篡改均被拒绝：删除早期 stage、重排、把 dead 标为 live、恢复
`raw.B1.x_squared` 硬编码归因、复用 eps payment。

## Gate B：H2 四格消融

四格使用 byte-identical step1 prestate：

- L0：distributed + generic square；
- L1：factorized + generic square；
- L2：distributed + joint square；
- L3：factorized + joint square。

step1 raw-y width 中，factorization main reduction 约 `4.00762e-4`，joint-square main
reduction约 `4.92251e-8`，交互接近零。也就是说，factorization 的实际收益约比 joint
square 大四个数量级；combined H2 的收益不能全部归因给 shared-remainder square。

## Gate C：单一 C1 算子

C1 对

`(1-(P_x+r_x)^2)*(P_y+r_y)-(P_x+r_x)-retained_y_rhs_polynomial`

构造一次联合系数多项式，并在 `(u_x,u_y,tau,r_x,r_y)` 上进行 outward tensor-product
Bernstein enclosure。dropped polynomial equal-exponent routes 与所有 P×R/R×P/R×R 关系在
最后一次 materialization 前保持共享。

- 代数路线：3,778；唯一系数：338；Bernstein 系数：2,730；
- exact oracle width：`0.0026020080299317804`；
- production width：`0.0026020080299710784`；
- production 完整包含独立 `Fraction` oracle；
- 去除 H2-vs-exact excess 的比例约 `0.99999999993`（远高于 10% micro gate）；
- retained Picard polynomial bitwise 不变；四个 segment/endpoint 通道无回退。

零余项、非对称余项、underflow、cutoff boundary、order overflow、checkpoint/resume 均有
回归测试。

## 科学矩阵

下表的“追回”定义为 `(legacy-candidate)/(legacy-Flow*)`。Flow* 数据只用于宽度交叉比较，
不作为 Torch 算子的 soundness oracle。

| Horizon | channel | candidate width | 追回 legacy excess | 门槛 |
|---|---:|---:|---:|---:|
| T=1 | endpoint-x | 0.08763412990388875 | 3.81% | 失败 |
| T=1 | endpoint-y | 0.11384217117563089 | 16.57% | 通过 |
| T=1 | segment-x | 0.09189960514713102 | 3.80% | 失败 |
| T=1 | segment-y | 0.12809136488991996 | 5.33% | 失败 |
| T=3 | endpoint-x | 0.18249356179174203 | 9.78% | 失败 |
| T=3 | endpoint-y | 0.14684186043233360 | 19.19% | 通过 |
| T=3 | segment-x | 0.20796749551800348 | 9.77% | 失败 |
| T=3 | segment-y | 0.16374117322801585 | 19.17% | 通过 |
| T=6.32 | endpoint-x | 0.5489073247629823 | 48.15% | 不宽于 H1+H2 |
| T=6.32 | endpoint-y | 0.7737280035448646 | 55.61% | 不宽于 H1+H2 |
| T=6.32 | segment-x | 0.5743256104684731 | 48.15% | 不宽于 H1+H2 |
| T=6.32 | segment-y | 0.7915870847892559 | 55.61% | 不宽于 H1+H2 |

fixed T=6.32 为 632 accepted、0 rejected。candidate/legacy runtime 为
`374.7869/262.8566 = 1.4258x`；native 请求为 `156.2001/135.4005 = 1.1536x`，均低于
2x。candidate fixed 峰值 RSS 为 646,877,184 bytes。

native T=10 请求在 `6.589638579126679` 停止，高于当前 H1+H2 floor
`6.482041958201616`，但未到 T=10。共 247 accepted、39 rejected attempts；最后限制为 y
上侧，subset margin `-1.1839005828144817e-5`。这明确是 stretch failure。

V100 T=0.1 与 CPU 的状态、步数、拒绝数、四通道宽度逐位一致。该结果只声明实现一致性，
不声明 V100 directed-rounding soundness 或 speedup。

## 不混淆 additive ownership 与因果 marginal

native 终止点最大的 additive ledger 类别是 `composition_overflow`，宽度
`0.00020872547742954457`。这是 C1 将完整 joint residual 一次性拥有在该类别下的结果；它
不能被解释成“composition overflow 是下一处最大损失”。真正的 causal 排名只来自 Gate B
byte-identical counterfactual，最大项仍是 `distributed_final`。

## 验证

预打包代码/科学全量测试为 `786 passed, 2 skipped`；加入 package self-test 后的最终全量为
`787 passed, 2 skipped`。legacy/H1/H2 step1 四通道与 Phase 0 bitwise 一致。最终 evidence
verifier 会校验 manifest、SHA256、clean scientific SHA、Gate A/B/C、tamper rejection、
矩阵公式和所有成功/失败 gate。
