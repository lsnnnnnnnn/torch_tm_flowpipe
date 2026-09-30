# 原生full1024初始affine覆盖缺口：独立源码与Fraction审查

结论：固定源码中的初始化欠包机制已由原/新两份实际CPU构造器MPFR系数独立审查确认。它先把中心向最近值舍入，再只按右端点算半径；中心向上舍入时，没有覆盖左端所需距离。全部8dd原盒的独立有理数回放得到384个欠包维度，涉及338个lane，物理x1/x2/x3各128个；其余行覆盖。没有修改入口、input boxes或库，也没有执行SSH/GPU/NN/植物积分。

## 冻结身份

`boxes.json` SHA `8dd18155d7d63b311fece5732251758a4984c0abb428377c135e68c244cedd30`；当前原生编译计划SHA `623b2418464fe7a26bc9df1c1477d32a90b8f099eb45c753c4f7095be049412c`；当前原库SHA `274e9bd11c9de21e95d72e935d9e45fd4fe9479ebb986751dd7f90522e416e3e`。读取的7份本地源码虽保存在较早var_tail_fix证据目录，但每份SHA都与当前**原库**BUILD_PLAN内对应源码完全相同；没有拿修改后的库替换本次对照。

| 源文件 | 本地实际SHA256，等于当前原库pin |
|---|---|
| Continuous.cpp | `4d42b818521b2f434297852af12b7c6b1fa7b56ad7e6d32b5626ea3d0387b8f1` |
| TaylorModel.h | `ed127748245341a1c0e740d879739ca71d3d6215941ab6c092121e74bfe657c7` |
| Polynomial.h | `9ded467c66f4aab77eeb1ad603445744f5a214bd5fedb97dbd69a97b7ac734dc` |
| Term.h | `cdcafe5689d2dcae9f0fb44a7f0eb3c4eae3612a9a340e111529bd0a4928c33f` |
| Interval.cpp | `fd1d760eb239d13a783354a7e5aa7242e328cc2e9eb91fb55344d9543e981324` |
| Interval.h | `cfc13c5583ed9b9bb0c75eb43531eec97692e67565de9c1cc68c49a86c7c0017` |
| include.h | `691afce6f5e065ab2cddee5b81941c6270ce966566045c6bf3270d91d6c250aa` |

## 实际初始化数学

对原始binary64端点l,u，53位MPFR执行：

```
c = RN53(RN53(l + u) / 2)
r = RU53(u - c)
x_i(z) = c + r*z_i + [0,0], z_i ∈ [-1,1]
```

box端点在53位MPFR中可以精确嵌入；c/r直接存入tmvPre的常数/自身变量系数，没有另存中心误差余项。tmv为identity，初始domain时间[0,0]、其余16维[-1,1]。原代码对半径上侧距离用了RU，但没有计算另一侧c-l，因而没有完整支付中心舍入造成的偏移。

- [Continuous.cpp:260–269](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/Continuous.cpp:260)：Flowpipe(box): tmvPre为box affine，tmv为独立变量identity。
- [TaylorModel.h:2838–2875](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/TaylorModel.h:2838)：实际center/radius到独立affine，domain t=0 / spatial[-1,1]。
- [TaylorModel.h:2921–2928](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/TaylorModel.h:2921)：tmv identity。
- [TaylorModel.h:336–339](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/TaylorModel.h:336)：线性TaylorModel不另赋余项，默认Interval0。
- [TaylorModel.h:440–443](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/TaylorModel.h:440)：observer评估expansion后加remainder。
- [TaylorModel.h:4134–4144](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/TaylorModel.h:4134)：Horner使用DATA_TYPE2=Interval做实际范围运算。
- [Polynomial.h:222–235](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/Polynomial.h:222)：线性系数直接形成degree1 term。
- [Polynomial.h:350–396](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/Polynomial.h:350)：Horner结构拆分。
- [Polynomial.h:499–503](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/Polynomial.h:499)：intEval调用Horner。
- [Interval.cpp:15–33](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/Interval.cpp:15)：全局precision53及Real复制/二进制输入。
- [Interval.cpp:659–664](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/Interval.cpp:659)：默认remainder=[0,0]。
- [Interval.cpp:683–688](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/Interval.cpp:683)：binary64端点定向嵌入。
- [Interval.cpp:851–858](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/Interval.cpp:851)：observer输出double向外取整。
- [Interval.cpp:927–938](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/Interval.cpp:927)：c=RN midpoint；r只取RU(up-c)。
- [Interval.cpp:403–423](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/Interval.cpp:403)：通用Real乘加减除为RN，未自动收费。
- [Interval.cpp:1400–1412](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/Interval.cpp:1400)：Interval加法定向。
- [Interval.cpp:1537–1559](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/Interval.cpp:1537)：Interval乘Real定向。
- [Interval.cpp:156–198](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/Interval.cpp:156)：已有Real定向加减API可供独立证明门。
- [include.h:36–37](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/flowstar/include.h:36)：normal precision53；未设置任意高精度。

## 精确反例与全量结果

第一处反例为0-based lane2、物理x3（随后lane3/x3、lane16/x2）。请求端点是l=-0.30000000000000004、u=-0.2，对应hex `-0x1.3333333333334p-2`、`-0x1.999999999999ap-3`。

存入模型的预测c=-0.25、r=0.04999999999999999（`0x1.9999999999998p-5`）。中心比精确中点大1/2^55；实际affine下界c-r恰好是binary64的-0.3，缺少的请求下端长度为1/2^54=5.551115123125783e-17，即请求下端附近1 ULP。上端保持-0.2。384行都是同一个输入模式。

本地计算使用Fraction对53位有效位做精确ties-to-even/向上/向下量化，未依赖被审Flow*实现。所有中间c/r和观测边界均在binary64 normal范围或为0，并核精确float往返。不是把十进制-.3和原binary64-.30000000000000004当作相同输入。

native_matched代理报告的实际initial.jsonl观测为同样384端点/338lane和相同首例、1ULP缺口；该报告与本独立预测一致。最初源码/Fraction分析阶段尚未读取实际MPFR导出，所以冻结的FRACTION_REVIEW保留`actual_MPFR_coefficients_read=false`。后续已单独完成actual原/新系数独审，见下节ACTUAL_NATIVE_PAIR_REVIEW；两个阶段证据没有混写。

## 为什么不是仅observer问题

初始intEval的Horner路径使用Interval运算，Interval加/乘与最终inf/sup转换均向外舍入。对于这个反例，c、r、c-r和c+r都可精确表示，向外observer会得到[-0.3,-0.2]；它如实包含已经欠包的affine。扩大打印范围、改JSON精度或只放宽审计容忍度不能恢复实际初始集合。一般而言observer覆盖输入只是必要证据，不能替代actual affine系数精确包含门，因为观测过宽可能遮住真实affine缺口。

## 最小独立修复及证明门（CPU构造器原型已通过）

保留原boxes字节、53位precision和原libflowstar.a，单独命名新的初始化入口变体；保留实际c，令 `r_new=max(RU(c-l),RU(u-c))`，构造相同独立affine结构、identity tmv、零余项及原domain。该半径是围绕已存c能覆盖两侧的最小53位向上表示，不需要笼统多扩若干ULP。当前反例的r_new=0.050000000000000044，hex `0x1.99999999999a0p-5`；它比旧r大8个radius ULP，直接nextafter旧r一次仍欠包。

独立Fraction检查已证明这个公式覆盖全部16,384行，只改变384个半径。这个Fraction结果本身只证明方案可行性；native代理随后以独立公共API helper完成实际CPU构造器原型，以下初始系数门已独立通过。完整闭环新入口尚不能据此标为通过。实际资格门应包含：

1. 对原盒binary64位串、库/全部相关header/source和precision重新锁定；改动仅限独立初始化变体，不覆写原运行。
2. 实际构造后导出每维c/r的精确MPFR二进制significand/exponent、R、domain和support，检查只有常数和自身变量、radius≥0、tmv identity、R=0。逐16,384行以Fraction证明`c-r≤l≤u≤c+r`；另可使用已有Real API检查`RU(c-r)≤l`和`RD(c+r)≥u`，这是对真实值的保守证明方向。
3. 负例必须让原384行失败；只nextafter一次也失败；两侧RU半径实际构造通过。不得把observer容忍度或打印放宽当作修复。
4. 再从零跑独立short40并审计真实NN/ranges/接受记录，才考虑full1000。新行必须标注“初始覆盖修复变体”，保留本次原库同输入复现失败的结论。它不会自动修好其余RN controller bias、线性注入、endpoint等路径，也不能升级为端到端严格证书。

可重算脚本：`check_center_form_fraction.py` SHA `fc5c2ca536545fb9c42a79bc0134b5b98abbff3ccaecc11cae43c87a2110a190`；完整384行证据：`FRACTION_REVIEW.json` SHA `879401d11942bdd681931d783a771141ff63789785696bfc8c56f1ee8ee7c998`。

## 后续实际MPFR原/新构造器独审

已读取原ACTUAL_AFFINE SHA `2ab06876d59b3299a679f68643ee8bfc73b9969e1e98faf8bd286cd23d66dc64`和新ACTUAL_AFFINE SHA `3d573b61070711c05e99cf92c1059cfd8fb736475a480375e1f538487c48fd5e`。逐一核对CPU10编译/执行退出0、源/helper SHA、同原库274e9bd、同8dd文件，以及新构造器收据对旧probe build的引用。没有执行被审C++ helper或native代理的Python checker，只本地精确解码实际MPFR binary_significand/exponent2。

- 原16,384个actual坐标全部与先前独立源码/Fraction模型一致；确有384行欠包，涉及338lane，失效lane/dimension集合与原short40失败收据完全一致。
- 新16,384个actual坐标逐项满足精确`c-r≤l≤u≤c+r`；每个新半径都等于围绕原c的最小53位向上双侧距离。
- 仅384个radius各增加8ULP；其余16,000个完整coordinate导出记录逐字段相同。所有center、tmv identity及其系数、support、domain、零余项均不变。原/新observer都包住自己的精确affine像；新observer同时覆盖请求盒。
- domain/R导出为向外double，不冒称单独的MPFR interval端点导出。它们的精确性还由原构造器literal±1/0和新helper的`Interval::operator==`（内部双端mpfr_cmp）检查、只替换pre对应行不触碰domain/tmv的源码支持。

独立可复算检查器`check_actual_affine_pair.py` SHA `677fe63d33a7166a53fc683b19f495dbdab3fba5a8aef1833c332945cfcf9bf5`；结果`ACTUAL_NATIVE_PAIR_REVIEW.json` SHA `0f4d440c459e48580fba8aadd152e340f8c65c20fdf6ccc17a60503a6a4653eb`。资格仅限初始CPU构造器，不把它写成新short40/full1000通过，不升级完整NNCS证书。

## GPU这次运行的对照检查

另已取得当前GPU14运行自身INITIAL_COVERAGE.json，完整source_identity与full1000 INPUT一致，实际16,384次Fraction初始覆盖门passed，actual pre/tmv摘要与独立8dd重构一致。GPU取两侧距离max再next_up，因此没有这个native单侧半径缺口。详情见GPU_INITIAL_REVIEW.md及GPU_INITIAL_LOCAL_REVIEW.json。这里沿用初始化当时的actual系数门，不声称此前终态审计读取过初始plant PT。
