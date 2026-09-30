# QUAD：相对于2026-09-23 slides的新增证据

范围：仅本地归档提取，不SSH、不启动实验。本材料基线是 `output/progress_review_20260923/slides.tex`（其中原QUAD仍列为规模/配置限制），不是把9月23日前已经展示的VDP/Bruss/TORA成绩再次算作新成果。来源ID、SHA与原始表列完整保存在同目录 `quad_evidence.json`。

## 建议主线（3–4页）

1. **先把任务算完整，再优化速度。** 完整1024初始分区的修正P2由旧共同前缀677推进到799；显式P3变体解决后期拒绝，高预算运行完成1000；最后直接sin/cos复用把完整时间减半且所有输出字节不变。
2. **Huan“很快”已真实复现。** 原作者parity＋只改临时SR存储的分块版，5次全部完整，进程中位75.250秒。它不是strict误差合同；不能用strict失败否定parity，也不能用parity完成追认严格证明。
3. **现在能展示具体同一步宽度，不能展示四方完整strict速度排名。** 第580步有相同8dd盒/native-f64的Huan strict、我们P2/P3均宽/max；第100步有四方历史union参考；T5只有Huan parity和我们P3系列可展示。
4. **性能结果最扎实的部分是P3对P3。** 1000步、50NN、同模型/同阶数/同SR/同资源守卫，watch3153.449→1533.752秒（2.056×），1000个observer和101个其他PT文件逐字节相同。不是全部算法都更快，也不是新的NNCS端到端证书。

## 统一配置与必须保留的差别

名义任务是QUAD 12物理状态＋时间/3控制变量（16维）；x1–x6初始[-.4,.4]，其余0；8×8×8×2×1×1=1024分区；h=.005，T=5，1000步，控制周期.1（每20步），50次NN，SR容量1000，cutoff1e-6、余项cap±.1，冻结网络SHA fabd84e…，box/same-slope。当前我们P2/P3和Huan同盒复现用8dd初始数组/native-f64；历史ARCH Huan/Xiangru/Flow*用f90数组/RPC-float32。4224初盒端点存在末位差（最大8.33e-17），因此后者不是逐位同盒对照。

计时环境记录为Tesla V100 16GB / PyTorch2.5.1+cu121；我方完整P3使用GPU3、CPU14–17，torch线程1；新native使用GPU2、CPU6–9和4植物worker。P3新旧完整版均GPU14GiB守卫/allocator13.5GiB/RSS11.5GiB；原生新full使用RSS128GiB/GPU14GiB，和旧11.5GiB RSS失败不是相同资源条件。

Huan作者P2验证RHS1；我方修正P2为working2/point1/validation3，P3为working3/point2/validation4，另含严格倒数、端点/控制注入舍入收费及完整SR K20重组。不能仅凭“P2”名称判为相同算法。Huan与Xiangru历史同值来自共享默认核心和同一NN驱动；不能当作两套独立算法的交叉证明。一般倒数反例针对所归档版本，不代表已证明该条QUAD轨迹漏包。

## 可直接上slides：完成程度与计时

不同完成范围的秒数分开解释，**不可在此整表计算排名或加速比**。

| 方法/模式 | 全1024共同接受步 | 时间/秒 | 时间口径/状态 |
|---|---:|---:|---|
| Huan parity + B-only SR chunk | 1000 | 75.250099 | 5-repeat full process median [Q1,Q2] |
| Huan strict + B-only SR chunk | 596 | 33.764603 | failure process total, not T5 [Q1] |
| Huan ARCH strict | 596 | 58.763553 | failure process total, not T5 [Q1] |
| Xiangru ARCH strict | 596 | 61.563960 | failure process total, not T5 [Q1] |
| Ours corrected P2 full batch | 799 | 658.477174 | failure watch total, not T5 [Q6,Q18] |
| Ours P3 old GPU11.5GiB guard | 589 | 1767.419438 | resource-stopped watch total, not T5 [Q6] |
| Ours P3 GPU14GiB | 1000 | 3153.449456 | single full watch total [Q8,Q20] |
| Ours P3 + direct trig reuse | 1000 | 1533.752052 | single full watch total [Q11,Q12,Q14,Q23] |
| Flow* historical stream/full SR | 160 | 1728.820754 | RSS-guard process total;160 complete exported steps [Q1] |
| Flow* new initial-affine-cover variant | 400 | — | last local read-only snapshot at2026-09-29T12:19:53Z; terminal unknown [Q15] |

Huan parity另有作者内部中位70.727803秒、driver调用中位72.173857秒；保留三种计时边界。其5次所有非计时指标相同。当前P3完整性能是单次完成性运行，包含日志、检查、初始化/图准备等，另有native在不同GPU/CPU并行，不能替代隔离重复计时。旧P3的589停止是11.5GiB人为显存守卫，GPU预算提高到14GiB后完成；这一步本身不是数学修复或提速。最新native400仅是北京时间2026-09-29 20:19:53的存档快照，不能说现在仍在400或已经结束。

## 四方同一t=0.5：endpoint总体union宽度（描述性参考）

旧100步P3与后来完整P3的observer100 PT SHA均为27a7f006…，表中36个P3 endpoint统计也与完整P3表精确相同。均值不是hull；以下都是最大上界−最小下界。ARCH三方与我们初盒/RPC及数学阶数仍不同，没有匹配的同前缀计时。不绘制把它们视为公平算法排名的柱图。[Q4/Q5]

| 维度 | 我们P3 | 我们P2 | Huan ARCH strict | Xiangru ARCH strict | Flow* ARCH |
|---|---:|---:|---:|---:|---:|
| x1 | 1.2094829 | 1.21096548 | 1.21269684 | 1.21269684 | 1.21501121 |
| x2 | 1.20869062 | 1.21173752 | 1.21619678 | 1.21619678 | 1.21431033 |
| x3 | 0.493475432 | 0.4983923 | 0.501518598 | 0.501518598 | 0.496225632 |
| x4 | 0.825957841 | 0.830255405 | 0.834660163 | 0.834660163 | 0.83485789 |
| x5 | 0.857812185 | 0.874356179 | 0.892896338 | 0.892896338 | 0.867656637 |
| x6 | 1.52604061 | 1.53374833 | 1.53895609 | 1.53895609 | 1.52948766 |
| x7 | 0.0123784314 | 0.0148458469 | 0.0174706922 | 0.0174706922 | 0.0124067843 |
| x8 | 0.0116692321 | 0.014057863 | 0.0165277957 | 0.0165277957 | 0.0117015144 |
| x9 | 0.000141791266 | 0.000168953418 | 0.000215318062 | 0.000215318062 | 0.000129809914 |
| x10 | 0.210014865 | 0.213901938 | 0.218655262 | 0.218655262 | 0.210297889 |
| x11 | 0.197841215 | 0.201549386 | 0.205980906 | 0.205980906 | 0.198103366 |
| x12 | 4.45014772e-308 | 4.45014772e-308 | 0 | 0 | 0 |

## 更严格的同盒t=2.9：每分区endpoint宽度均值

三者均在第580步完整接受1024分区；8dd盒和native-f64一致，算法仍不同。Huan这里只存mean/max，没有总体union或tube；Flow*全1024没有580导出，所以不补值。Huan这一时点没有逐步计时，不拿其599尝试失败总33.765秒当580步耗时。下表适合主slides选x5/x6/x7，其余放附录。[Q6/Q7/Q19]

| 维度 | Huan strict mean | 我们P2 mean | 我们P3 mean |
|---|---:|---:|---:|
| x1 | 2.95059418 | 2.2072176 | 1.822787 |
| x2 | 4.93156584 | 3.75722031 | 3.20070648 |
| x3 | 0.800737104 | 0.19008228 | 0.0922670086 |
| x4 | 2.39121652 | 1.27872459 | 0.838570382 |
| x5 | 4.51963104 | 2.00271607 | 1.4013097 |
| x6 | 2.62793953 | 0.468764091 | 0.215671747 |
| x7 | 0.402382703 | 0.0639981183 | 0.0187245045 |
| x8 | 0.37797573 | 0.059692855 | 0.0178334829 |
| x9 | 0.114547624 | 0.0126888279 | 0.00435001944 |
| x10 | 4.71870951 | 0.676147481 | 0.301811652 |
| x11 | 4.45710235 | 0.606008787 | 0.26108896 |
| x12 | 0 | 4.45014772e-308 | 4.45014772e-308 |

来源历史P3在589守卫后停止，但该580实际observer已本地重算；后来完整高预算P3前589份observer与旧49da运行字节相同，新trig全1000与完整P3相同。这里不能移植旧589运行的1621.385秒prefix580 advance当作新trig的580耗时。P2同前580 advance289.214933秒仅为已存的不同变体诊断分项，不计算速度比。

## 完整t=5的endpoint总体union宽度

相同名义T5和请求盒，但Huan parity与我方P3的余项/阶数/边界合同不同。它们可并列展示真实绝对值，**更窄不代表更严格或更正确**。Huan strict、Xiangru strict、Flow*的完整T5均缺失，不填数字。x12常量舍入底噪，不生成宽度比。[Q3/Q8/Q9/Q12]

| 维度 | Huan parity | 我们P3（含新trig） |
|---|---:|---:|
| x1 | 6.49970944 | 6.62362135 |
| x2 | 7.05301872 | 7.25355368 |
| x3 | 0.0476872323 | 0.0689723821 |
| x4 | 1.42263174 | 1.50395626 |
| x5 | 1.6005968 | 1.72207246 |
| x6 | 0.123350957 | 0.179927098 |
| x7 | 0.00701656253 | 0.0108531397 |
| x8 | 0.00611990429 | 0.00971082782 |
| x9 | 0.00444940679 | 0.00614348764 |
| x10 | 0.0861795769 | 0.15565646 |
| x11 | 0.0710206706 | 0.130841171 |
| x12 | 0 | 4.45014772e-308 |

P2/P3共同第799步另有72项mean/max/union×tube/endpoint数据[Q10]：例如endpoint mean x6从P2的3.869302188到P3的0.196513515；x7从0.535247686到0.014142749。原P2第800开始拒绝，不能将799之后幸存分区hull包装成全初始集覆盖。

## Native Flow*最新修复入口（单独标注变体）

原始8dd/native-f64原库短程40步虽然接受，实际仿射初值有338个分区/384下端点欠包1ULP，故未准入完整运行。独立修复保留输入盒/中心/原库，只用两侧向外距离最大值确定半径；实际16384坐标Fraction包含检查通过，只有384半径扩大。修复后short40的40960范围记录、2NN、SR20/40全部通过；两次同实际NN输入经原driver重放，86016个T/L/U binary64结果全相等，限定为这两次输入。旧库RN边界/VAR尾项等仍未取得全链路证明。[Q16/Q17/Q22]

同40步可示例：endpoint均宽x3，我们P3=0.2144798713、native修复P2=0.2165987523；总体hull分别0.8208427869/0.8229603592。完整72项在Q17。40步外层wall为我们诊断P3 155.451914秒、native修复171.138231秒；观察/保存开销不同，不报公平加速比，更不能外推T5。新native全程原作业仅存档400/1000步，终态与T5时间/宽度未知。

## 审核/归档边界，建议末页一句话

已完成的是“条件性数值轨迹与实现等价核验”：初始化Fraction包含、真实边界冷恢复、完整steps/acceptance、实际PT/SR对照均有保存证据；不是用少跑分区、去掉误差项、抬高余项上限换成功。但控制器CROWN/same-slope浮点证书仍为条件性，`end_to_end_strict_certificate=false`/`fullbatch_qualification=false`保留，原driver的VERIFIED不能扩成完整NNCS证明。[Q12/Q21/Q23]

最新trig远端CPU实际核验4530文件，1000observer+101其他PT字节相同、两个最终plant/14SR组件/进度相同。8关键JSON已本地下齐并SHA核验；完整35成员小包本地仅partial，候选8 PT/SR/plant保持远端，不能写成“所有原始张量已下载本地”。旧完整P3本地8 actual observer已重算576宽度，其表通过全1000实际配对适用于trig。实验保持暂停。

## 原始来源清单

- **Q1** `results/quad_targeted_recovery_20260927/fullbatch_baseline_review_20260928/RESULT.json`；SHA256 `71c122d0ced5a5445a7b043dc3c4c16d77368c3abced7373f645b794432ddfe4`。
- **Q2** `results/archcomp_failure_20260923/evidence/huan_parity_repeats/CAMPAIGN.json`；SHA256 `8651b97943518f6d16e6d73787df624292b3b903ddb9001c5d578cb823144947`。
- **Q3** `results/archcomp_failure_20260923/evidence/huan_box_parity_chunk/metrics.json`；SHA256 `084ce1510d71572dd1d2415ed5a8175a003d2e29dba0cc835b8fff7db92c5186`。
- **Q4** `results/quad_targeted_recovery_20260927/p3_common100_comparison_20260929/RESULT.json`；SHA256 `1ed8863105268f56af41c698d51fc2c398818e8db067da31362559fc45b49dbf`。
- **Q5** `results/quad_targeted_recovery_20260927/p3_common100_comparison_20260929/WIDTHS.csv`；SHA256 `27b454cc97c5d2442d58240b26c2df9ea5940322812f02404bdbeb792bcf5be1`。
- **Q6** `results/quad_targeted_recovery_20260927/p3_common580_comparison_20260929/RESULT.json`；SHA256 `06354feb33ccdd2fbfad22e1eaa80217b221fdc1a9e2a48556371f41f7fd541e`。
- **Q7** `results/quad_targeted_recovery_20260927/p3_common580_comparison_20260929/WIDTHS.csv`；SHA256 `64f846958294058e0f713037b8402a68763a6c866e5b98c90fca61f6254f628f`。
- **Q8** `results/quad_targeted_recovery_20260927/gpu14_full1000_summary_20260929/RESULT.json`；SHA256 `bb9ed40ae642162a9e9376fc0d04bb88ad0f201fb93702bd459229f23d8a84fa`。
- **Q9** `results/quad_targeted_recovery_20260927/gpu14_full1000_summary_20260929/WIDTHS.csv`；SHA256 `f673126d8e8e31b9414dd247dda95e5058e7aa554c754548c6ae0a6913d3bc08`。
- **Q10** `results/quad_targeted_recovery_20260927/gpu14_full1000_summary_20260929/COMMON799_WIDTHS.csv`；SHA256 `8eca27ab682df8cf49cac05ae14901fe0c13af05f73638f8201fa2fd3837b1d5`。
- **Q11** `results/quad_targeted_recovery_20260927/evidence_trig_terminal_20260929/full1024_p3_trig_gpu14_1000_v1/RESULT.json`；SHA256 `310bb9acb6780845c2d336e96de344e25bfbb2916cd9b721362296fc75c8ae98`。
- **Q12** `results/quad_targeted_recovery_20260927/evidence_trig_terminal_20260929/TRIG_GPU14_TERMINAL_REVIEW_20260929/RESULT.json`；SHA256 `517bd183820d43af794d92b64434d84e05e9fb4fa120927f2e9bf3c41968add2`。
- **Q13** `results/quad_targeted_recovery_20260927/evidence_trig_terminal_20260929/LOCAL_VERIFICATION.json`；SHA256 `1b4a6b2d9a0425083e67f3ec5e4dbad68ec22c766aad2f1016422e515a0750de`。
- **Q14** `results/quad_targeted_recovery_20260927/evidence_trig_terminal_20260929/TIME_COMPARISON.csv`；SHA256 `614e60f787350946d1684f551a129a09c961d0c2d13fdd8907f92dc3574731f1`。
- **Q15** `results/quad_targeted_recovery_20260927/native_full1024_preparation_20260929/evidence_full_cover/PAUSE_HANDOFF_READONLY_20260929_v1.json`；SHA256 `3f240c731298dce6b2174d496ba052a35a5d1c69b2bce7daf7a4f137636fd865`。
- **Q16** `results/quad_targeted_recovery_20260927/native_cover_P3_common40_20260929/RESULT.json`；SHA256 `47a6e63460c24fcdaafe21c373f97e39497dc03f8bc609d4765bd16aadba6d33`。
- **Q17** `results/quad_targeted_recovery_20260927/native_cover_P3_common40_20260929/WIDTHS.csv`；SHA256 `65ecb45faf3e786e2faf45ef8ae5f3480d0240a3920a41ef05510b7a2f21ad9f`。
- **Q18** `results/quad_targeted_recovery_20260927/evidence_fullbatch_long_logs_20260928/full1024_p2_1000_v1/RESULT.json`；SHA256 `bbe8e4fd243c02202767e2619ed03ac8df0661b29e9f5a7a00a308d317ef4dbd`。
- **Q19** `results/quad_targeted_recovery_20260927/p3_common580_comparison_20260929/INDEPENDENT_REVIEW.md`；SHA256 `d3b7be88fb009e4deb81fee7a4b2efa9fe5bab2a8a0b3ebc7b5d6682142d4777`。
- **Q20** `results/quad_targeted_recovery_20260927/gpu14_full1000_summary_20260929/INDEPENDENT_REVIEW.json`；SHA256 `d854d2f6cd3f467e1f2a12ab3a2735e6c87300ef464cb50d3d93e66dc3337494`。
- **Q21** `results/quad_targeted_recovery_20260927/native_initial_coverage_review_20260929/GPU_INITIAL_REVIEW.md`；SHA256 `28a5e6e0ac3487711c3e7600d414eacbc64e1d257dc4c5748ab68bd8de04104d`。
- **Q22** `results/quad_targeted_recovery_20260927/native_initial_coverage_review_20260929/REVIEW.md`；SHA256 `0436dca7cca6168b9d494a382d2e2f0fe1d48b12bb311b5af19a2ab20ab5301a`。
- **Q23** `results/quad_targeted_recovery_20260927/evidence_trig_terminal_20260929/ROOT_AUDIT_REVIEW.json`；SHA256 `5ce9c48388673c495ff47e4bf4dc8dc775a93d00030760899bf3f2182977c0ad`。
