# 上次 slides 之后：QUAD 之外的基准证据整理

整理日期：2026-09-29；仅只读本地资料并生成本文，没有 SSH、实验、源码修改或重新计时。实验保持暂停。

## 1. 时间边界与应讲清楚的增量

按 [上次 slides](/Users/shengenli/Documents/ChatGPT/verification/output/progress_review_20260923/slides.tex) 为起点；虽然新旧目录均含 20260923，ARCH-COMP suite 是上次交付之后补做的更广四方对照。旧 slides 的 VDP/Brusselator、TORA homogeneous、Single Pendulum、CartPole 不是这次新加的算法成果。新 suite 采用官方 C++ 合同重新对齐；不得把旧 slides 的 1.802 倍 TORA 数值与新套件的计时与观察口径拼成同一趋势。

本次最重要的增量是：官方 14 个配置（本文列 QUAD 外 13 个）完成四方筛查；其中 7 个达到完整时域且各方完成 5 次正式计时。Airplane 从积分前元数据内存失败推进到真实 NNCS 的第 78 步性质早停，并解释了停止差异。ACC 找到原生参照真实漏包并在隔离库修复；这撤销了“GPU 精度比正确 Flow* 差约 8 倍”的解释。Unicycle 已找到首个宽差及部分观察器保守性的原因，但三 GPU 同注入单步归因尚未完成。

**当前新 QUAD P3/三角复用引擎并未重跑这些完整基准。** 下述五次时间与主要宽度来自冻结历史 `fff9d0f`/`d5f0b68`/`1c16d4e`；Airplane、ACC 的新候选和 Unicycle 首步诊断分别标记，不能把它们当作一套最新引擎的统一回归成绩。

## 2. 四方比较的合同与限制

- 三 GPU 主比较为共同 box/same-slope、官方输入布局、原 float32 RPC 传输数值、strict 植物模式；这是共享对照入口，不等于作者独立默认 hybrid/parity 入口。Huan/Xiangru 独立默认单摆入口另曾完成，不能用其模式替代主表。
- 原生使用 CROWN-Reach 中冻结的 Flow* 库和原 C++ 植物/NN 服务，固定完整 h 的整数步包装器。批量正式计时用 4 worker，不用早期自动 80 线程结果做性能排名。ACC 为共同且明确的仿射特征适配器。
- 正式计时：V100-SXM2-16GB，GPU3、CPU14–17、Torch 单线程；四方轮换顺序，1 轮排除预跑 + 5 轮正式运行；7 配置 × 4 方 × 6 轮 = 168 进程，其中 140 样本进入统计。共享服务器，非整机独占。
- 表中时间为新进程 wall time，含导入/缓存扩展加载、NN 初始化、求解与总结；不含首次编译，不是单 kernel 时间。宽度来自独立带观察器运行，不能把其单次秒数混入正式中位数。
- Huan/Xiangru 29 核心 Python 文件中 27 相同，CUDA 内嵌数学代码相同；13 个可比配置（含 QUAD、不含 Airplane）的范围文件完全相同。相同结果有源码和字节证据，不是两个独立算法互相证明正确。
- strict 开关和宽度接近不证明 soundness；后续已查到 ACC 原生反例、Unicycle 原生初盒 1 ULP 缺口，以及冻结三 GPU 通用倒数路径反例。通用倒数反例不等于已证某个完整 benchmark 实际漏解；原报告应保留这种范围限定。

来源：[REPORT.md](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/REPORT.md)；[作者倒数路径实际 CPU 反例](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/reciprocal_author_counterexamples_cpu_20260928/REPORT.md)。

## 3. 完整时域的五次正式计时

单位秒；每格为中位数 [最小, 最大]。ACC* 的旧原生结果有后续正确性反例，此行仅保留历史执行时间，不能构成正确参照下的速度/精度验收。

| Benchmark | Flow* | Huan | Xiangru | 我们 | Flow* / 我们 |
|---|---:|---:|---:|---:|---:|
| Attitude | 6.315304 [6.111067, 6.354589] | 6.989571 [6.960294, 7.058767] | 7.028113 [6.839285, 7.375633] | 7.120503 [6.978546, 7.186483] | 0.886918× |
| Unicycle | 10.721960 [10.553988, 10.778262] | 9.795658 [9.622197, 10.058480] | 9.580615 [9.544477, 10.054354] | 6.908679 [6.852378, 7.041852] | 1.551955× |
| TORA ReLU/tanh | 8.632177 [8.617337, 8.676372] | 12.421312 [12.137125, 12.502958] | 12.331677 [12.225945, 12.705246] | 7.684916 [7.659163, 7.874790] | 1.123262× |
| TORA sigmoid | 9.098132 [9.033632, 9.121180] | 13.164734 [12.969238, 13.259016] | 13.273851 [13.188094, 13.515251] | 8.207989 [8.126622, 8.310452] | 1.108448× |
| NAV robust | 68.163522 [66.916274, 78.318447] | 14.646262 [14.070551, 15.074267] | 14.813850 [14.044636, 14.915904] | 9.729652 [9.348544, 9.966918] | 7.005751× |
| Single Pendulum | 4.907801 [4.773311, 4.916538] | 5.596916 [5.527276, 5.624499] | 5.618161 [5.503686, 5.819269] | 5.176120 [5.068480, 5.374686] | 0.948162× |
| ACC* | 8.259421 [8.224243, 8.341850] | 8.034780 [7.875580, 8.078416] | 7.959671 [7.862016, 8.080713] | 7.792836 [7.740355, 7.868385] | 1.059874× |

原始数据：[timing.csv](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/timing.csv)；[168 运行样本](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/timing_samples.csv)。
可用于主讲的具体结论：NAV robust 68.163522 → 9.729652 s（7.005751×，显示可用 GPU 加速）；TORA ReLU/tanh 与 sigmoid 分别约 1.123262× / 1.108448×；Unicycle 约 1.551955×，但最终 x4 宽度为 3.499392 倍，不能只讲速度。Attitude 及单摆完整进程仍慢于 Flow*。

## 4. 未完成四方全时域：原始停止与步数

每格是接受/保存的全分区前缀、终止类别、单次带观察器 wall 秒。它们不是五次正式计时，不能计算完整时域排名。原生“保存前缀”不推断未导出的内部进度。

| Benchmark / 目标 | Flow* | Huan | Xiangru | 我们 |
|---|---|---|---|---|
| DP more robust / 80步 | 21步；性质早停 FALSIFIED；5.107280s | 22步；性质早停 Unsafe；5.630889s | 22步；性质早停 Unsafe；5.755396s | 22步；性质早停 Unsafe；5.359457s |
| DP less robust / 100步 | 25步；300s 求解超时；303.658638s | 100步；完整；8.303232s | 100步；完整；8.506425s | 100步；完整；8.761649s |
| NAV standard / 600步 | 220步；300s 求解超时；303.930472s | 600步；完整；18.117416s | 600步；完整；17.907006s | 600步；完整；14.153986s |
| CartPole / 200步 | 141步；数值拒绝；8.941050s | 156步；数值拒绝；13.765621s | 156步；数值拒绝；13.991427s | 156步；数值拒绝；12.647434s |
| Airplane / 200步 | 78步；性质早停 FALSIFIED；8.464804s | 0（未形成范围）步；积分前 RSS 守卫；35.481602s | 0（未形成范围）步；积分前 RSS 守卫；35.792755s | 0（未形成范围）步；积分前 RSS 守卫；35.673258s |
| TORA homogeneous / 200步 | 200步；完整；8.195928s | 189步；数值拒绝；8.902766s | 189步；数值拒绝；8.818860s | 200步；完整；7.876134s |

来源：[diagnostic_attempts.csv](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/diagnostic_attempts.csv)。DP more robust 的官方 NN 服务器实际加载 less-robust 模型，忠实记录；未将早停分类升级为新的正式安全/不安全证明。H/X homogeneous 2279/2400 lane-steps、全分区前缀 189；不是 200 步全成功。CartPole 已用正确 float32 NN 重跑（GPU 接受156步，原生保存141步），错误初轮不入主表。

## 5. 宽度：共同前缀与两个不同统计量

宽度 = upper − lower；endpoint 是步末、tube 是整步。观测为局部 tmvPre 的区间 hull，不能拼接旧 slides 的组合 tmv 观察量。分区宽度比先对每个坐标/视角在所有共同有效步和分区上取 p95/max，再取最差坐标/视角；union 则先合并全部分区为整体 hull。p95 不是置信区间。以下比值均 GPU/旧 native；Airplane 初轮无 GPU 范围，另列后续数据。

### 5.1 分区内宽度比（p95 / max）

| Benchmark | 共同步 / 时间 | Huan | Xiangru | 我们 |
|---|---|---:|---:|---:|
| Attitude | 60 / 3s | 1.1195988 / 1.1455754 | 1.1195988 / 1.1455754 | 1.1207865 / 1.1456466 |
| DP more robust | 21 / 0.105s | 1.3446626 / 1.4002776 | 1.3446626 / 1.4002776 | 1.3446381 / 1.4002687 |
| DP less robust | 25 / 0.25s | 1.0945835 / 1.1130817 | 1.0945835 / 1.1130817 | 1.1060886 / 1.1243569 |
| NAV robust | 600 / 6s | 1.0112475 / 1.0387696 | 1.0112475 / 1.0387696 | 1.0298648 / 1.0465015 |
| NAV standard | 220 / 2.2s | 1.0261561 / 1.4863705 | 1.0261561 / 1.4863705 | 9.0921053 / 17.064516 |
| Unicycle | 500 / 10s | 2.8405832 / 3.4434523 | 2.8405832 / 3.4434523 | 2.8756543 / 3.4993923 |
| TORA ReLU/tanh | 500 / 5s | 1.0070272 / 1.0077032 | 1.0070272 / 1.0077032 | 1.0072094 / 1.0081448 |
| TORA sigmoid | 500 / 5s | 1.0091702 / 1.0107409 | 1.0091702 / 1.0107409 | 1.0091702 / 1.0107409 |
| CartPole | 141 / 0.705s | 1.071974 / 1.2030585 | 1.071974 / 1.2030585 | 1.071974 / 1.2030715 |
| Single Pendulum | 100 / 1s | 1.0834202 / 1.0921442 | 1.0834202 / 1.0921442 | 1.0531892 / 1.0581209 |
| TORA homogeneous | 189 / 18.9s | 2.5743687 / 7.8995985 | 2.5743687 / 7.8995985 | 1.4401228 / 1.9658069 |
| ACC* | 50 / 5s | 7.3096151 / 7.9448411 | 7.3096151 / 7.9448411 | 7.3026165 / 7.9397267 |

### 5.2 整体 union 宽度比（p95 / max）

| Benchmark | Huan | Xiangru | 我们 |
|---|---:|---:|---:|
| Attitude | 1.1195988 / 1.1455754 | 1.1195988 / 1.1455754 | 1.1207865 / 1.1456466 |
| DP more robust | 1.3446626 / 1.4002776 | 1.3446626 / 1.4002776 | 1.3446381 / 1.4002687 |
| DP less robust | 1.0311083 / 1.0394422 | 1.0311083 / 1.0394422 | 1.0317188 / 1.0394415 |
| NAV robust | 1.0047324 / 1.0123134 | 1.0047324 / 1.0123134 | 1.0079598 / 1.012313 |
| NAV standard | 1.0038687 / 1.0097081 | 1.0038687 / 1.0097081 | 1.0048974 / 1.0095881 |
| Unicycle | 2.8405832 / 3.4434523 | 2.8405832 / 3.4434523 | 2.8756543 / 3.4993923 |
| TORA ReLU/tanh | 1.0070272 / 1.0077032 | 1.0070272 / 1.0077032 | 1.0072094 / 1.0081448 |
| TORA sigmoid | 1.0091702 / 1.0107409 | 1.0091702 / 1.0107409 | 1.0091702 / 1.0107409 |
| CartPole | 1.071974 / 1.2030585 | 1.071974 / 1.2030585 | 1.071974 / 1.2030715 |
| Single Pendulum | 1.0834202 / 1.0921442 | 1.0834202 / 1.0921442 | 1.0531892 / 1.0581209 |
| TORA homogeneous | 3.0016734 / 7.8995985 | 3.0016734 / 7.8995985 | 1.4008632 / 1.8235283 |
| ACC* | 7.3096151 / 7.9448411 | 7.3096151 / 7.9448411 | 7.3026165 / 7.9397267 |

精确 CSV：[width_summary.csv](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/width_summary.csv)。NAV standard 的我方 cell 最大 17.064516 倍对应 step59、lane626、x3：native 3.441691376e-15，我方 5.8730798e-14；整体 union 最坏仅 1.00958805。不能把该极小分母比值解说成“整体膨胀17倍”。Unicycle 的 3.499392 倍对应约 0.0352 的实质宽度，不属于此类末位现象。ACC* 旧分母漏包，因此旧 7.94 倍只描述输出、不支持精度结论。

### 5.3 共同前缀最后一步的绝对 endpoint union 宽度

下面直接从归档 `width_trajectory_*_matched.csv.gz` 抽取，物理坐标按官方配置次序。完整案例是最终时刻；其余只是表中共同前缀最后时刻，不能冒充完整终点。数值显示10位有效数字，原 gzip CSV 保留完整二进制64文本。Huan/Xiangru 分列但值相同。

| Benchmark / 步 | Flow* | Huan | Xiangru | 我们 |
|---|---|---|---|---|
| Attitude / 60 | [0.004194134463, 0.005980153698, 0.006193976913, 0.03229213136, 0.01763594283, 0.01863740395] | [0.004174214013, 0.005935658769, 0.006172913343, 0.03207032616, 0.01736151581, 0.01847936361] | [0.004174214013, 0.005935658769, 0.006172913343, 0.03207032616, 0.01736151581, 0.01847936361] | [0.004398304598, 0.006299372384, 0.006555204266, 0.03384847706, 0.01848375419, 0.01971918617] |
| DP more robust / 21 | [3.80609534e-06, 7.522141192e-06, 4.407068939e-05, 9.565700683e-05] | [3.255760261e-08, 4.029821099e-08, 5.513193027e-07, 7.481526322e-07] | [3.255760261e-08, 4.029821099e-08, 5.513193027e-07, 7.481526322e-07] | [2.737195159e-08, 3.416114414e-08, 5.486477652e-07, 7.437120154e-07] |
| DP less robust / 25 | [0.3989302374, 0.326588365, 0.6066063165, 0.5372534131] | [0.3986209495, 0.3262640348, 0.6040297561, 0.5347722464] | [0.3986209495, 0.3262640348, 0.6040297561, 0.5347722464] | [0.3992846159, 0.327082814, 0.6071061661, 0.5376248335] |
| NAV robust / 600 | [0.09099849755, 0.01507648988, 0.02489177998, 0.2941088597] | [0.09030302286, 0.01480298716, 0.02461457813, 0.2924140666] | [0.09030302286, 0.01480298716, 0.02461457813, 0.2924140666] | [0.09134371892, 0.01511633265, 0.02495126959, 0.2947860326] |
| NAV standard / 220 | [0.4330085412, 0.1660777786, 0.3054833759, 0.6188299386] | [0.4329277545, 0.1660152493, 0.3054564813, 0.6187841293] | [0.4329277545, 0.1660152493, 0.3054564813, 0.6187841293] | [0.4329757561, 0.1660973374, 0.3054856481, 0.6187799965] |
| Unicycle / 500 | [0.0403853379, 0.06082287733, 0.04165892139, 0.03520954223] | [0.0954093678, 0.1409441537, 0.1313936842, 0.1212423802] | [0.0954093678, 0.1409441537, 0.1313936842, 0.1212423802] | [0.09669186312, 0.1428876202, 0.1337684916, 0.1232120025] |
| TORA ReLU/tanh / 500 | [0.0251280695, 0.02745753526, 0.02203482393, 0.02161695002] | [0.0250795086, 0.02741051979, 0.02199292419, 0.02159111089] | [0.0250795086, 0.02741051979, 0.02199292419, 0.02159111089] | [0.02509965709, 0.02739114704, 0.02201384029, 0.02165234232] |
| TORA sigmoid / 500 | [0.02487466578, 0.02635338883, 0.02610609014, 0.02799617606] | [0.02480363422, 0.02628227864, 0.02602510721, 0.02794208213] | [0.02480363422, 0.02628227864, 0.02602510721, 0.02794208213] | [0.02489772349, 0.0262875855, 0.02608527732, 0.02804372738] |
| CartPole / 141 | [0.3692052792, 1.830964122, 2.116055089, 15.83408371] | [0.3684962987, 1.821561803, 2.064329903, 14.39007037] | [0.3684962987, 1.821561803, 2.064329903, 14.39007037] | [0.3684543433, 1.821221842, 2.063535278, 14.38119426] |
| Single Pendulum / 100 | [0.1391393524, 0.1444355487] | [0.1482616762, 0.1551736948] | [0.1482616762, 0.1551736948] | [0.1438742238, 0.1502647624] |
| TORA homogeneous / 189 | [0.1416534512, 0.1358758305, 0.5933792014, 0.5906356258] | [0.2870019596, 0.3714916143, 3.124087479, 4.148934951] | [0.2870019596, 0.3714916143, 3.124087479, 4.148934951] | [0.1674881631, 0.1657311502, 0.9090997401, 1.000535148] |
| ACC* / 50 | [20.99393312, 0.1975723307, 0.0006846514465, 5.028368774, 1.79563214, 1.063699852] | [21.01204096, 0.2017538906, 0.0007410763492, 5.497975787, 2.01551073, 1.181106183] | [21.01204096, 0.2017538906, 0.0007410763492, 5.497975787, 2.01551073, 1.181106183] | [21.01045505, 0.2016286635, 0.0007376997796, 5.333353383, 1.948655935, 1.159430455] |

## 6. ACC 后续纠错：需要替换旧解读

旧 native 的合法初始点 v(0)=32、a(0)=0 在 t=.1/.2 有独立解析包络，而保存端点与它们不相交。直接 ODE.reach / 匹配包装器、2维/8维最小例复现，问题定位于 refinement 的 VAR 截断尾项缓存/重放；不能用“Flow* 更窄”当真值。只在隔离库补回尾项，原作者库和原结果保留。修复库 `73a23102…` 完整 ACC 50步、早期6个解析见证通过，尚未通用回归。

| 方法 | step2 a_lead endpoint 绝对宽度 | 计时口径/秒 |
|---|---:|---|
| 旧 native（漏包） | 0.00028282614978181186 | 旧正式五次见上表；不作正确参照 |
| VAR-tail 修复 native 实验版 | 0.002247354391217149 | 8.296836939，新进程单次诊断 |
| Huan strict 历史 | 0.0022470088317284587 | 5.659108131，历史 driver |
| Xiangru strict 历史 | 0.0022470088317284587 | 5.573999452，历史 driver |
| 我方本轮控制组 | 0.002245562328254458 | 5.157350691 driver；8.3066进程 |
| 我方 solution_order 候选 | 0.0007721378347164132 | 5.448104681 driver；8.4740进程 |

候选保持解阶3、h=.1、T5/B1/SR50，仅将 RHS 验证阶数升至解阶3，无重试；关键 endpoint 宽度比控制组减约65.6%，driver单次慢约5.6%，不是加速结论。与修复 native 的50×6 endpoint比，H/X最大约1.0000000000035；tube最大仍约1.42045，需另查。候选和修复参照都未成为覆盖全基准的默认验收版本。
原始来源：[REPORT.md](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/REPORT.md)；[ACC_REPAIRED_ANALYSIS.json](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/ACC_REPAIRED_ANALYSIS.json)；[ACC_ANALYSIS_V2.json](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/ACC_ANALYSIS_V2.json)；[ACC_REPAIRED_WIDTHS.csv](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/ACC_REPAIRED_WIDTHS.csv)。

修复参照下的逐项最大宽度比（覆盖50步×6维，endpoint与tube分开；小于1仍不是正确性证明）：

| 方法 | endpoint最大比 | tube最大比 |
|---|---:|---:|
| 修复native | 1 | 1 |
| Huan | 1.0000000000034808 | 1.4204541210857169 |
| Xiangru | 1.0000000000034808 | 1.4204541210857169 |
| 我方控制组 | 0.99999382236969947 | 1.4204438330975624 |
| 我方solution_order候选 | 0.99999356475164225 | 1.4099253717240527 |

## 7. Airplane 后续：已越过内存阻塞，未完成200步

原任务 B1、19总变量/12物理维、P6/h=.01、T2/200步不变。将稠密元数据改为按实际单项式支持集构建，并接入稀疏范围与带舍入的endpoint转换；显式候选 b049d443，不是偷偷替换H/X原入口。模型、初盒、NN/RPC32合同保持。原H/X仍无积分范围。

| 阶段 | 实际接受步 | 原因 | 完整进程秒（单次，非正式排名） |
|---|---:|---|---:|
| 原四方 suite Huan / Xiangru / ours | 0 / 0 / 0 | 元数据建表RSS守卫 | 35.481602 / 35.792755 / 35.673258 |
| 我方稀疏元数据 NNCS | 79 | 整步 hull Unsafe 早停 | 46.628120 |
| 仅额外采集 GPU 见证 | 79 | 同上；79步范围与前次完全同字节 | 45.689562 |
| 隔离时间二分检查 | 78 | 后半步违反约束，Unsafe 早停 | 45.312770 |
| 原生 Flow* 历史 | 78 | 递归时间划分 FALSIFIED 早停 | 8.464804 |

已取回并检查原生和实际 GPU 的 step78 y−1 多项式：整步跨0，后半步下界严格为正；GPU独立Fraction后半步界约[.004196709424782507,.009004833661608235]。有界两分检查在本例让停止步一致；不声称与原生递归检查在任意系统等价。提前停止没有改善前78步宽度，范围与原候选前缀完全同字节。

共同78步：tube最坏比1.0272159799109835；step78 y tube native0.009514599684767755、ours0.009582362199862193。endpoint含极小分母，最坏比223，不可脱离绝对值：step78 y endpoint native4.1096015479524795e-12、ours8.713696431073004e-12。所有12维完整表见来源。此病例初始点箱使端点接近浮点误差尺度，跨实现中心亦不同；不能由tube接近宣称完整NNCS已证明。

首次稀疏NNCS peak RSS1.872482GiB / GPU2.480469GiB；最终二分诊断GPU采样峰2,663,383,040B，无资源中止。通用runner/watch因没到200步记incomplete/error，但实际是接受后的性质早停，不是数值拒绝。仍没有完整200步、H/X恢复、四方正式五次时间。
来源：[REPORT.md](/Users/shengenli/Documents/ChatGPT/verification/results/airplane_nncs_recovery_20260923/REPORT.md) → [REPORT.md](/Users/shengenli/Documents/ChatGPT/verification/results/airplane_stop_witness_20260924/REPORT.md) → [REPORT.md](/Users/shengenli/Documents/ChatGPT/verification/results/airplane_gpu_stop_20260924/REPORT.md)；[WIDTHS_COMMON_PREFIX.csv](/Users/shengenli/Documents/ChatGPT/verification/results/airplane_nncs_recovery_20260923/evidence/WIDTHS_COMMON_PREFIX.csv)。

## 8. Unicycle 后续：不要把9/29首步诊断写成新500步成功

9/23 suite 四方确已完整500步及五次计时。9/29新工作是归因，不是重跑新QUAD引擎的500步。初盒/扰动/变量置换核对正确；native真实50次输入的独立NN replay在同输入时f64与RPC32均相等，但不能据此说闭环两边每周期输入盒相同。

- 首步 native→三GPU 已出现共同宽差。x3/x4 tube可在同一个GPU多项式上解释为逐项区间化丢失共享时间相关性；额外下界宽度分别0.0010869132244709406和0.00013452625651671978。这不是删误差项，也不是整条轨迹原因已找到。
- 我方相对H/X首3步仅末位差，以1e-12绝对阈值观察到step4 x1额外宽1.0968519248422126e-6；尚在第一次控制期。没有step3/4内部证据，不能把cutoff猜测当结论。
- 原生CPU首步已在原库/原RPC响应asFloat模式复现，16bounds/8width与旧首步逐位一致；保存initial、before/after injection、after advance、after endpoint五个实际TM/SR边界。一步0.114133202秒不含NN重算，不是全程计时。
- 实际旧native初始化x2下端点少覆盖1ULP（8.881784197001252e-16）。本次保留该问题；不能拿同RPC域重放替代初始集合严格覆盖，也不能直接把这一末位缺口当作最终3.5倍宽差解释。
- 三GPU对齐首步计划只启动ours一次，在植物计算前库加载失败；Huan/Xiangru没有启动。watch3.132135秒是失败启动时间，不是单步数值性能。缺Ninja为CPU实证前置阻断；原三SO直接导入成功、SHA不变。没有修加载/新v2/数值重跑。

来源：[UNICYCLE_FIRST_GAP_REVIEW_20260929.md](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/UNICYCLE_FIRST_GAP_REVIEW_20260929.md)；[REVIEW.md](/Users/shengenli/Documents/ChatGPT/verification/results/unicycle_first_gap_20260929/native_v1_local/REVIEW.md)；[LOADER_FAILURE_AND_MINIMAL_PROPOSAL.md](/Users/shengenli/Documents/ChatGPT/verification/results/unicycle_first_gap_20260929/gpu_preparation_v1/LOADER_FAILURE_AND_MINIMAL_PROPOSAL.md)。

## 9. 剩余其他基准的真实进展

NAV robust、异构TORA、Attitude、单摆已有上面冻结suite结果；本次本地资料未找到它们在新QUAD候选上的全程复验。NAV standard与DP less robust三GPU完成，但原生4worker仍在300s求解时限停止，不能将其截断时间当完整Flow*时间。DP more robust仍为早停、模型命名/实际加载差异保留。CartPole仅完成原配置/NN dtype纠正后的失败对照，尚未解释所有拒绝差异。TORA homogeneous我方与native完整而H/X拒绝，仍无完整四方速度结论。VCAS只有NN服务/随机模拟，没有同类C++可达性入口，未冒充第15个实验。


## 10. 各任务精确配置索引

共同 cutoff=1e-6；SR默认1000、ACC50。余项估计按各配置逐项保留，不统一写成±.1。下面是最终执行配置，初盒、分区、模型/盒SHA见原CSV。

| Benchmark | B | 物理维 | h / 解阶 | 控制周期 × 次数 | T / ODE步 | 余项估计 |
|---|---:|---:|---|---|---|---|
| Attitude | 1 | 6 | 0.05 / 3 | 0.1 × 30 | 3.0 / 60 | [-0.01, 0.01] |
| DP more robust | 1 | 4 | 0.005 / 4 | 0.02 × 20 | 0.4 / 80 | [-0.01, 0.01] |
| DP less robust | 225 | 4 | 0.01 / 4 | 0.05 × 20 | 1.0 / 100 | [-0.01, 0.01] |
| NAV robust | 25 | 4 | 0.01 / 4 | 0.2 × 30 | 6.0 / 600 | [-0.1, 0.1] |
| NAV standard | 640 | 4 | 0.01 / 4 | 0.2 × 30 | 6.0 / 600 | [-0.1, 0.1] |
| Unicycle | 1 | 4 | 0.02 / 2 | 0.2 × 50 | 10.0 / 500 | [-0.01, 0.01] |
| TORA ReLU/tanh | 1 | 4 | 0.01 / 6 | 0.5 × 10 | 5.0 / 500 | [-0.01, 0.01] |
| TORA sigmoid | 1 | 4 | 0.01 / 6 | 0.5 × 10 | 5.0 / 500 | [-0.01, 0.01] |
| CartPole | 1 | 4 | 0.005 / 6 | 0.02 × 50 | 1.0 / 200 | [-0.1, 0.1] |
| Airplane | 1 | 12 | 0.01 / 6 | 0.1 × 20 | 2.0 / 200 | [-0.01, 0.01] |
| Single Pendulum | 1 | 2 | 0.01 / 2 | 0.05 × 20 | 1.0 / 100 | [-0.01, 0.01] |
| TORA homogeneous | 12 | 4 | 0.1 / 3 | 1.0 × 20 | 20.0 / 200 | [-0.01, 0.01] |
| ACC* | 1 | 6 | 0.1 / 3 | 0.1 × 50 | 5.0 / 50 | [-0.1, 0.1] |

[configurations.csv](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/configurations.csv) 同时给出各初盒、模型SHA、boxesSHA和实际YAML路径。Unicycle额外w∈[-1e-4,1e-4]。ACC候选仍是解阶3，只改变验证阶策略，不与其它配置的解阶混为一谈。

## 11. 可直接用于slides的主线与数据边界

1. 从少数案例扩展到官方14配置的四方筛查，7项有正式五次时间；最亮眼是NAV robust约7×且整体范围接近。
2. 异构TORA时间略快、范围接近；Unicycle速度更快但范围宽，Attitude/单摆启动开销下仍不占优。
3. 发现“原生更窄不一定更正确”：ACC原生尾项缺失，修复后关键宽度与H/X接近；这是一项正确性纠错，不包装成GPU更好看的结果。
4. Airplane由积分前内存失败变成真实NNCS接受并可解释早停，未完成全程和四方性能。
5. Unicycle完成历史复核、首步原生状态采集及环境定位，三GPU内部归因未完成；当前新QUAD优化尚未全套回归。

以下源SHA仅绑定本次读取的本地文件，不声称重新下载/审计服务器全部大范围。终点表重读12份现存CSV并检查三GPU重复native列一致、H/X终点一致。原始二进制范围多数保留远端；可复算的CSV/JSON与manifest本地已有。

| 本地来源 | SHA256 |
|---|---|
| [oldslides](/Users/shengenli/Documents/ChatGPT/verification/output/progress_review_20260923/slides.tex) | `f96380982cd462e1f48121f57b431a00a1715b7caede750767f17254e31f917e` |
| [main](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/REPORT.md) | `ea322d459fa53284fdf4992fe393d36544dc08c909f04ad5b7a4bddb2599887f` |
| [timing](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/timing.csv) | `0075fa5557e72bc9633aa2e4ba3950b0764bcdca38253e1b4220b883c26adfcf` |
| [samples](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/timing_samples.csv) | `a37013ba290f5601f6a691a5305e349020a3382219a195630d24816471accc51` |
| [width](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/width_summary.csv) | `63b08b7f6c16e76b26a12588d51db60ee97a48d6f5ca34494c09e71e8004ba76` |
| [diag](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/diagnostic_attempts.csv) | `35cecb669a47b2daf5990098c23810919b2178076b7e1b3bbb955fa3f90ba19b` |
| [config](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/configurations.csv) | `7abe1a1e1545dccc717b65825161921f9ae73956c87d55417df73f3d16f98220` |
| [accfix](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/ACC_REPAIRED_ANALYSIS.json) | `f3cdfd7fc856c12a2e203581feea13a60596750840463feb691aeebfa316a057` |
| [accnew](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/ACC_ANALYSIS_V2.json) | `aa2e12ff5ec27029c9db28e078556bf8d80906e8dc013a3442521c7ef6f52a27` |
| [airwidth](/Users/shengenli/Documents/ChatGPT/verification/results/airplane_nncs_recovery_20260923/evidence/WIDTHS_COMMON_PREFIX.csv) | `dc29535672696c3b02516e56ebf0ace1fb4dd9c0031f3641302546e6dde5b1d9` |
| [airgpu](/Users/shengenli/Documents/ChatGPT/verification/results/airplane_gpu_stop_20260924/REPORT.md) | `9d5504a410e65baa28e623511ddeac0d2a98f1a24fc3d1598f3a7cb549161349` |
| [unigap](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/UNICYCLE_FIRST_GAP_REVIEW_20260929.md) | `8045690da759d37ecb11fa6e75fa6b6b4fe6fe87399198bf323393dbf669a680` |
| [unicpu](/Users/shengenli/Documents/ChatGPT/verification/results/unicycle_first_gap_20260929/native_v1_local/REVIEW.md) | `5475d55afb6bdbc8f70339f3de75df273a8e82aeb88f2021f2f67c92c91796a3` |
| [unienv](/Users/shengenli/Documents/ChatGPT/verification/results/unicycle_first_gap_20260929/gpu_preparation_v1/LOADER_FAILURE_AND_MINIMAL_PROPOSAL.md) | `8c1648b062dd84b9b741d1f3c7cd45448a43806564bd64b95596dfcec09390d7` |

完整轨迹CSV（endpoint/tube、每步整体上下界/宽度，保留原全精度）：

- [width_trajectory_attitude_control_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_attitude_control_matched.csv.gz)；SHA `edc8dd2732ddea1242a6721076326104683096dbce85629d60209b80a8753841`。
- [width_trajectory_double_pendulum_more_robust_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_double_pendulum_more_robust_matched.csv.gz)；SHA `36b104f4a940d5f06b44d35ba8e25253499bb1602c41f61ecaa034cfd7873cac`。
- [width_trajectory_double_pendulum_less_robust_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_double_pendulum_less_robust_matched.csv.gz)；SHA `cac4bd3a63d09619f16f4b6e19d3fdbc9f9744e99f75cc834bca6de82e96531f`。
- [width_trajectory_nav_robust_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_nav_robust_matched.csv.gz)；SHA `bc2316cd3817cd4575c5417a6131bdffad8c533068cf53a2e7faf307b3929808`。
- [width_trajectory_nav_standard_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_nav_standard_matched.csv.gz)；SHA `1f950a2c1cd7d7cfddfae4fc2cf080499aa0a1f0861fee81b927305fa26d4cdc`。
- [width_trajectory_unicycle_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_unicycle_matched.csv.gz)；SHA `faf582bcbd28286aeee8ef413268b7de4f28f21564a71618df23bbc2cb864751`。
- [width_trajectory_tora_relu_tanh_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_tora_relu_tanh_matched.csv.gz)；SHA `67b0087b23f659bdd75d7c5c8aaca96220828b81c2e8226c63e154905165781f`。
- [width_trajectory_tora_sigmoid_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_tora_sigmoid_matched.csv.gz)；SHA `1db4abcdd6e7ac5c87709cf5331976e86341618f7e7d772fafd9691923c59b2b`。
- [width_trajectory_cartpole_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_cartpole_matched.csv.gz)；SHA `b572693f1981bedb6b5a47fa73d4e3b2399787f5a55d7c54596eecf353a3dd3f`。
- [width_trajectory_single_pendulum_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_single_pendulum_matched.csv.gz)；SHA `7407b0cc59e878a926cfae85b40cd42ff8580da56d3c329efdb87fb4deea2dc0`。
- [width_trajectory_tora_homogeneous_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_tora_homogeneous_matched.csv.gz)；SHA `3643f8e0bdb8c82d00662d16570f9e9ca37446614e2f288c701c53bc716e64d7`。
- [width_trajectory_acc_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_acc_matched.csv.gz)；SHA `d520c3f0e3a9c900375ef2cd594dc8988a228f9588b10658d8a3f1b4ce079726`。
