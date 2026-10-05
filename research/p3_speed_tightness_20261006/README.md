# 10月6日：先提速，再测试区间收紧

仅新增独立候选和派生比较；冻结基线、原记录不改，无旧实验/checker重跑、内容摘要或CUDA扩展JIT。绘图只用Python；共享服务器单次计时不证明稳定排名。

20项终态含14项全程通过、3项40步通过、1项40步完成但包装门失败、2项首步前失败。QUAD当前采用joint full，同一次运行保存时间与收紧范围；其余当前选择以[selection.json](report_data/selection.json)为准。四方结果见[正式报告包](../../docs/evidence/results/archcomp26_report_20261006/README.md)，全部时间层、资源与字段来源见[当前索引](report_data/current/RUN_INDEX.json)。

## 已保存候选

时间为process / driver秒；wrapper/payload另存JSON，不代填。0步与40步不进全程比较。链接原RESULT，同目录candidate/存数值及比较收据。“同”不包含隐藏TM/SR身份。

| 原始候选 | 完成步数 | process / driver | 类型及采用情况 |
| --- | --- | --- | --- |
| [Attitude fused1](results/attitude_fused1_full60_001/RESULT.json) | 60 | 10.447 / 6.264 | 实现；当前选择，保存范围同 |
| [Docking fused1](results/docking_fused1_full400_001/RESULT.json) | 400 | 12.700 / 8.538 | 实现；当前选择，原UNKNOWN保留 |
| [SP两态 fused1](results/sp_two_state_fused1_full100_001/RESULT.json) | 100 | 5.180 / 1.289 | 实现；当前选择，具名order2 |
| [NAV robust fused32](results/nav_robust_fused32_full600_001/RESULT.json) | 600×25盒 | 14.961 / 10.999 | 实现；当前选择，全部逐盒范围同 |
| [NAV standard private256](results/nav_standard_private256_full600_001/RESULT.json) | 600×640盒 | 34.221 / 30.022 | 实现；全程同但慢于旧driver25.865，不选 |
| [NAV standard fused512](results/nav_standard_fused512_full600_001/RESULT.json) | 600×640盒 | 24.031 / 19.864 | 实现；当前选择，全部逐盒范围同 |
| [TORA tanh fused1](results/tora_tanh_fused1_full500_001/RESULT.json) | 500 | 8.791 / 4.563 | 实现；保存同，保留速度优先备选 |
| [TORA tanh cutoff](results/tora_tanh_cutoff1e8_full500_001/RESULT.json) | 500 | 9.242 / 4.963 | 数值；当前折中选择，4000项宽度不增 |
| [TORA sigmoid cutoff](results/tora_sigmoid_cutoff1e8_full500_001/RESULT.json) | 500 | 9.995 / 5.763 | 数值；当前折中选择，4000项宽度不增 |
| [TORA sigmoid order4](results/tora_sigmoid_order4_cutoff1e8_full500_001/RESULT.json) | 500 | 10.497 / 6.237 | 数值；高阶备选，额外收益有限 |
| [TORA sigmoid order6](results/tora_sigmoid_order6_cutoff1e8_full500_001/RESULT.json) | 500 | 15.511 / 10.696 | 数值；不选默认，增加耗时明显 |
| [QUAD fused v1](results/quad_paper_fused256_40_001/RESULT.json) | 0 | 2.371 / — | 实现；prepare准入误拒绝 |
| [QUAD fused v2](results/quad_paper_fused256v2_40_001/RESULT.json) | 40 | 49.669 / 43.136 | 实现；数值完整、包装门失败，更慢不推广 |
| [QUAD trig 40](results/quad_paper_trigpower40_001/RESULT.json) | 40 | 45.555 / 38.149 | 实现；短测保存同，全程资格另立新ID |
| [QUAD trig full1000](results/quad_paper_trigpower_full1000_001/RESULT.json) | 1000×1024盒 | 975.862 / 968.516 | 实现；保存同，保留速度优先备选 |
| [QUAD control v1](results/quad_paper_anchored_control40_001/RESULT.json) | 0 | 6.886 / — | 数值；首步前调用不存在的接口 |
| [QUAD control v2](results/quad_paper_anchored_controlv2_40_001/RESULT.json) | 40 | 46.209 / 39.576 | 数值；短测改善，另有完整候选 |
| [QUAD control v2 full1000](results/quad_paper_anchored_controlv2_full1000_001/RESULT.json) | 1000×1024盒 | 1008.895 / 1001.431 | 数值；完整收紧备选，未同时提速 |
| [QUAD joint 40](results/quad_paper_joint40_001/RESULT.json) | 40 | 45.707 / 38.273 | 联合；双门通过，保存同control短测 |
| [QUAD joint full1000](results/quad_paper_joint_full1000_001/RESULT.json) | 1000×1024盒 | 978.067 / 969.549 | 联合；当前选择，自身时间及收紧范围 |

## 采用依据和边界

实现候选保留两轮原式、包含判断、合法回退和首live原实现比较。NAV robust补32行；standard的256行虽少padding却多dispatch，512行更快。QUAD融合v2的38次快路、2次合法回退不是数值错误；当时包装门要求40次全快，故保留失败状态。后续只读比较确认40行和最终科学字段相同，但process49.669慢于旧private256短测45.708，不推广。

QUAD trig只复用同一次执行内、相同输入的sin/cos幂链，保留独立累加、尾界和舍入。40步driver从旧private256的39.289降至38.149，process45.708至45.555；后者只是很小的单样本差异。新GPU门计时计入process/payload、在driver之外。Python复用计数是eager/capture构图次数，不是逐步CUDA重放数。QUAD保存的是每步1024盒的12态pooled投影，不是每盒完整轨迹。

新增full1000于2026-10-05 18:04:33.750 UTC结束，process1004.197→975.862（单次下降2.82%），driver997.675→968.516（下降2.92%），仍不接近作者完整进程速度。wrapper974.732、payload974.703分别保留；GPU门0.938秒计入外层。其1,216,787字节、1000行观察与原full50直接相同；config同、argv仅输出路径迁移。保存宽度未改变，不把提速写成收紧。

数值候选不声称输出相同。TORA cutoff从1e-6改1e-8；sigmoid高阶另改work/point/validation为4/3/5或6/5/7。两个cut版4000项宽度不增加，但较窄不等于包含。sigmoid上版速度优先process/driver为9.392/5.255，当前cut版为9.995/5.763，二者优势不能拼成同一次运行。

sigmoid两个高阶各有3800项窄、4项等、196项宽；宽项来自x3/x4的tube和endpoint在第2–50步，各49步。最大绝对增加分别约6.57e-14、1.54e-13，仅末位尺度，全部最终四态仍更窄。计数不是跨物理量紧度分数。order4比cut多约0.502秒process，order6多约5.516秒，额外收益有限，故保留备选。精确增量见[宽度评估](tightness/CANDIDATE_WIDTH_ASSESSMENT.md)。

control v2保留same-slope注入多项式，用同一输入域的额外two-slope仿射包络收紧余项。40步960项为878窄、82等、0宽，942项包含于参考、18项不包含；实际NN调用4次，payload旧计数器仅记2次。包络有效性仍是条件，独立端到端浮点NNCS证书为false。短测不外推全程。

另立control full1000已完整通过，process/driver1008.895/1001.431，比原private256的1004.197/997.675略慢，不能与trig时间拼为一次“更快且更紧”。24000项保存宽度为21990窄、2002等、8宽；宽项仅x9第43–46步的tube/endpoint，最大绝对增加1.1101261e-10、最大相对增加0.0007664052%。终点x1–x11较窄、x12相等，12态终点均包含于参考；x3和x11终点分别缩小2.543565%和4.756848%。全程23956项区间包含于参考，不能声称每步全包含；计数不合成跨物理单位评分。

joint将trig和control v2放入同一次运行。40步45.707/38.273、两套资格门通过；与control v2短测观察/config直接字节相同、final_hull相同，见[短测比较](results/posthoc/joint40_vs_control40_saved_comparison.json)。另立full1000实际978.067/969.549，相较上一版private256的1004.197/997.675，单次process下降2.60%、driver下降2.82%。其自身完整范围与control full相同，因此上述终点收紧来自同一次联合运行，未拼接两项优点。它较trig-only的process慢2.205秒（0.226%），保留trig为速度优先备选；仍远慢于Huan/Xiangru，8项早期增宽亦未消失。

P3是方法族列名：SP保留具名两物理态order2、point1、validation3、native-f64及shared-driver原参考注入，不冒称QUAD自定义严格注入或官方MATLAB三返回量合同。Docking的UNKNOWN、缺件合同、早停前缀均不因加速消失。

engine_linear_leaf_v2旧快路径仍在；历史快数字的控制定义、阶数或输出工作量不同，不能删严格步骤、SR历史或绘图输出移植。见[历史快分支表](../../docs/evidence/results/archcomp26_report_20261005/timing/historical_fast_branch.csv)。

## 完整QUAD资格

`quad_paper_trigpower_full1000_001`的原始RESULT和完整观察已回收并独立只读核对：1000步每步1024盒接受、无拒绝，所有物理区间有限且有序；50次刷新[0,20,…,980]；每行SR长度等于步数且epoch0，最后观察1000/0，正常末尾重置为0/1000。private/weighted/trig恢复均通过，新GPU门8个case和4次图捕获通过；未调用旧检查器。

`quad_paper_anchored_controlv2_full1000_001`已独立只读核对1000步全接受、SR历史长度与末尾重置，实际100次NN调用、50次注入、153600控制行和完整24000项比较均已保存。它的包络变化与运行耗时同时记账。

`quad_paper_joint_full1000_001`原始终态为2026-10-05 18:41:26.009 UTC完成、exit0，无超时。独立只读审计核对1000步全1024盒接受、SR每步长度1…1000/epoch0及正常末重置0/1000、50次基础刷新、actual100 NN/50注入/153600控制行；private/weighted/trig/control四恢复、trig新GPU门8case/4capture与两套资格均通过。

联合观察1,216,906字节、宽度CSV8,019,632字节与control full直接相同，config同、final_hull同；此外按原始bounds精确有理数独立重算并核对1000×12×2有序宽度项。仅覆盖保存的pooled投影和字段，不证明隐藏逐盒TM/SR同一。wrapper976.924、payload975.966、tinyGPUgate0.934秒分别记录，gate在driver计时外。见[独立保存证据审计](results/posthoc/jointfull_independent_saved_audit.json)与[联合对control完整比较](results/posthoc/jointfull_vs_controlfull_saved_comparison.json)。所有本轮候选均已有终态；生产门与独立证书仍未开放。

## 独立入口，仅供另立新候选

仅供授权新候选；不安排重复旧实验。资源先预检，`NEW_RUN`须不存在且不得用已有ID。`SRC`为新源码快照，`ADAPTERS`为10月5日依赖，`GATE`为已通过门目录；依赖同冻，监督器另存START/RESULT/退出码。

```sh
: "${SRC:?}" "${ADAPTERS:?}" "${GATE:?}" "${NEW_RUN:?}"
test ! -e "$NEW_RUN" || exit 1
RESEARCH_ROOT=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z
PYTHON_EXE="$RESEARCH_ROOT/nncs_env/bin/python"
env CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 \
  timeout --signal=TERM --kill-after=10 120 taskset -c 10-13 \
  "$PYTHON_EXE" -B "$SRC/run_sigmoid_cutoff_candidate.py" \
  --adapters "$ADAPTERS" --plant-cutoff 1e-8 --gate "$GATE" --output "$NEW_RUN"
```

其他入口均用python -B并补`--gate "$GATE" --output "$NEW_RUN"`：

- NAV robust：`speed/run_fused_batch_candidate.py --instance nav --mode full600 --base-wrapper "$NAV_BASE"`，GPU1/CPU6–9，120秒。
- NAV standard：`speed/v3/run_nav640_fused_candidate.py --base-wrapper "$SRC/expansion/run_nav_standard_candidate.py" --adapters "$ADAPTERS"`，GPU1/CPU6–9，180秒。
- Attitude：`expansion/run_b1_candidate.py --instance attitude --adapters "$ADAPTERS" --weighted-mode fused`，GPU2/CPU10–13，120秒。
- tanh提速：`expansion/run_b1_candidate.py --instance tora-tanh --adapters "$ADAPTERS" --weighted-mode fused`，GPU3/CPU18–19（taskset参数`18,19`），120秒。
- Docking/SP：`expansion/run_b1_more_candidate.py --instance docking`（或single-pendulum），加`--adapters "$ADAPTERS" --weighted-mode fused`；Docking GPU3/CPU14–17，SP GPU2/CPU10–13，120秒。
- tanh收紧：`tightness/run_tanh_cutoff_candidate.py --base-wrapper "$SRC/expansion/run_b1_candidate.py" --adapters "$ADAPTERS"`，GPU3/CPU18–19（taskset参数`18,19`），120秒。
- QUAD：`speed/run_trig_power_candidate.py`或`tightness/v2/run_quad_control_candidate.py`，加`--mode batch2`（或full50）及`--base-wrapper "$QUAD_BASE"`；GPU3/CPU14–17，短测180秒/full1800秒。
- QUAD联合：`speed/joint/run_quad_joint_candidate.py --mode full50 --base-wrapper "$QUAD_BASE" --trig-dir "$TRIG_SOURCE" --control-dir "$CONTROL_V2_SOURCE"`；依赖显式冻结目录，GPU3/CPU14–17、1800秒，仍须上述gate和全新output。参见[joint初始接口与资格（准备时快照）](speed/joint/README.md)。
- sigmoid高阶：`tightness/run_sigmoid_order4_candidate.py`或`tightness/order6/run_sigmoid_order6_candidate.py`，加`--adapters "$ADAPTERS"`，GPU2/CPU10–13，120秒；仅备选接口，不安排重复实测。

`NAV_BASE`/`QUAD_BASE`为对应冻结包装器路径，取同类原START确认；方法资源不得互换。更多取舍见[候选备注](REPORT_CANDIDATE_NOTES.md)。正式选择以JSON为准，原RESULT、START、日志和失败记录不可编辑。
