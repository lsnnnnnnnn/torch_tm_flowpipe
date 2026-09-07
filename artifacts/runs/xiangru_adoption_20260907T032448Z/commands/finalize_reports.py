from pathlib import Path
import csv, json
root=Path(__file__).resolve().parent
repo=root/'our_audit';art=repo/'artifacts/runs/xiangru_adoption_20260907T032448Z';docs=repo/'docs/xiangru_adoption'
load=lambda p:json.loads(p.read_text())
def read(name):
 with (art/name).open() as f:return list(csv.DictReader(f))
source=load(art/'source_manifest.json')
ours={name:load(art/('raw_minimal/our_'+name+'_full/summary.json')) for name in ['brusselator','vdp']}
native={name:load(art/('raw_minimal/native_'+name+'/summary.json')) for name in ['brusselator','vdp']}
summary=read('width_summary.csv');checks=read('width_at_checkpoints.csv');timing=read('timings_raw.csv')
metrics={('endpoint','x'):'终点 x',('endpoint','y'):'终点 y',('tube','x'):'整段 x',('tube','y'):'整段 y'}
plants={'brusselator':'Brusselator','van_der_pol':'Van der Pol'}
rel='../../artifacts/runs/xiangru_adoption_20260907T032448Z/'
ref_table='| 系统 | 我们实际完成 | Flow* 实际完成 | 我们求解秒 | Flow* 求解秒 |\n|---|---:|---:|---:|---:|\n'
for name,label in [('brusselator','Brusselator'),('vdp','Van der Pol')]:
 o,n=ours[name],native[name]
 ref_table+=f"| {label} | {o['accepted_horizon']:g}（{o['accepted_steps']} 步） | {n['accepted_horizon']:g}（{n['accepted_steps']} 步） | {o['solve_seconds']:.6f} | {n['solve_seconds']:.6f} |\n"
width_table='| 系统 | 范围 | 我们 / Flow* 时间加权中位数 | P95 | 最大比率 | 最差时刻 |\n|---|---|---:|---:|---:|---:|\n'
for row in summary:
 if row['comparison']=='ours_vs_flowstar' and row['view']=='common':
  width_table+=f"| {plants[row['plant']]} | {metrics[row['range_kind'],row['state']]} | {float(row['weighted_median']):.6f} | {float(row['weighted_p95']):.6f} | {float(row['max_ratio']):.6f} | {float(row['max_at_time']):g} |\n"
checkpoint_table='| 系统 | 时刻 | 范围 | 我们宽度 | Flow* 宽度 | 我们 / Flow* |\n|---|---:|---|---:|---:|---:|\n'
for row in checks:
 if row['comparison']=='ours_vs_flowstar' and row['view']=='common' and row['selection']=='specified_checkpoint':
  value=lambda k:format(float(row[k]),'.9g') if row[k] else '未完成'
  checkpoint_table+=f"| {plants[row['plant']]} | {row['requested_time']} | {metrics[row['range_kind'],row['state']]} | {value('numerator_width')} | {value('denominator_width')} | {value('ratio')} |\n"
process_table='| 系统 | 工具 | 完整进程秒（含导出） | 纯导出秒 | 峰值 CPU MiB |\n|---|---|---:|---:|---:|\n'
for row in timing:
 if row['measurement']=='fresh_process_including_export':
  process_table+=f"| {plants[row['plant']]} | {'我们的 CPU 参考版' if row['mode']=='our_cpu_reference' else '原生 Flow*'} | {float(row['process_seconds']):.2f} | {float(row['export_seconds']):.6f} | {float(row['peak_rss_bytes'])/2**20:.2f} |\n"
batch_table='| 系统 | 路线 | B1 完整 warm / fresh | B8 完整 warm / fresh | B32 完整 warm / fresh |\n|---|---|---|---|---|\n'
for plant in plants.values():
 for lane in ['Xiangru 普通对齐版','Xiangru 严格误差版']:
  batch_table+=f'| {plant} | {lane} | 未测 / 未测 | 未测 / 未测 | 未测 / 未测 |\n'
status={'schema':'xiangru_adoption_decision/1','decision':'DO_NOT_ADOPT_YET__CONFIRMED_CORRECTNESS_DEFECT','candidate_sha':source['candidate']['actual_runtime_sha'],'candidate_numerical_patch':False,'optional_backend_added':False,'default_backend_changed':False,'our_numerical_src_changed':False,
'gates':{'correctness_and_function':'FAIL_CONFIRMED_EXACT_LOCAL_UNDERENCLOSURES','full_prefix_width':'NOT_EVALUATED_AFTER_CORRECTNESS_FAILURE','main_batch_speed':'NOT_EVALUATED_AFTER_CORRECTNESS_FAILURE'},
'reasons':['Historical point-matrix propagation misses preserved exact Fraction witness on CPU/CUDA, parity/strict.','Independent centered two-state affine normalization underenclosure on CPU/CUDA; zero constant coefficients and symmetric remainders meet production invariants; fixing both requires more than one numerical mechanism.'],
'candidate_complete_solve_seconds':None,'candidate_complete_horizons':{'brusselator':None,'van_der_pol':None},'candidate_algorithm_accepted_diagnostic_horizons':{'brusselator':.04,'van_der_pol':.02},
'actual_cpu_references':{name:{'our_accepted_horizon':ours[name]['accepted_horizon'],'flowstar_accepted_horizon':native[name]['accepted_horizon'],'our_solve_seconds':ours[name]['solve_seconds'],'flowstar_solve_seconds':native[name]['solve_seconds']} for name in ours},
'access_or_hardware_missing':False,'adaptive_plus_history_candidate':'UNSUPPORTED','source_and_environment_blocked':False,'third_party_code_published':False}
(art/'ADOPTION_RESULT.json').write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n')
baseline=f"""当前不推荐采用这版 Xiangru 严格误差后端。新克隆已取得，普通和严格模式均在实际 CPU/CUDA 路线上重现历史矩阵漏包；归一化缩放又有一个不依赖历史矩阵、满足零常数项约束的仿射集合漏包。候选数值源码未改动。两类机制超出本轮一个数值修补预算。

四条路线使用已提交的两状态合同：Brusselator 六阶、h=0.02、历史容量 1000；VDP 四阶、h=0.01、历史容量 100。没有分区初始盒子、扩大范围细分预算或额外修补 endpoint。我们的 validation_eps 是向外误差扩张，候选时间容差没有同等数值预算含义。

{ref_table}
这是本轮新鲜固定步长测量。我们的 VDP 自适应 T10 是保留的历史能力，未冒充新的固定步长结果。候选只执行两步诊断（Brusselator T0.04，VDP T0.02），不具备严格接纳资格；按目标第 17 节停止受影响长跑。

{width_table}
以上是完整共同对象测量下的 CPU 参考比较，四项分别列出。候选 / 我们、候选 / Flow*、严格 / 普通的完整四项宽度均缺失，不能给出全程最大倍率或“接近 Flow*”标签。两步数据仅作为诊断保留，不能据此评价候选完整时域的宽度。

每个 CPU 参考只有一次新鲜测量，求解与导出分开计时。严格与普通全程代价、预热 GPU 完整时间和 B32 吞吐没有实测，所以无法把候选差异归因于批量并行、初始化或实际求解速度。没有从两步冷启动外推完整速度。首个 CUDA segment 扩展构建为 43.218453059 秒；tape 扩展在短诊断中另行构建。

当前应保留我们的独立 CPU 参考、VDP 自适应能力与原生 Flow* 对照，暂停从零 GPU 开发。后续候选须先闭合两个误差来源，再重新通过全程宽度和实际批量速度门槛。完整表、图和方法见 [REPORT_PLAIN_CHINESE.md](REPORT_PLAIN_CHINESE.md)。
"""
(docs/'BASELINE_COMPARISON.md').write_text(baseline)
report=f"""当前选择是暂不采用 Xiangru 的可选 GPU 后端，决定为 `DO_NOT_ADOPT_YET__CONFIRMED_CORRECTNESS_DEFECT`。指定远端与 V100 均可用，阻碍是两处精确集合包含性失败。没有修改我们的数值 src、默认后端或旧工作树，也没有恢复从零 GPU 开发。

新目录为 `/srv/local/shengenli/xiangru_adoption_20260907T032448Z/xiangru_upstream`，远端分支 `2026_experiment` 在 2026-09-07 03:25:55 UTC 取得的完整 SHA 为 `{source['candidate']['actual_runtime_sha']}`，实际 detached 运行。旧 Xiangru 的 `84184de6c2b3f1ff2da6755f732d91925037025d` 没有本轮新增的仓库内 GPU 引擎。源码、完整对照 SHA、许可、环境和实际导入位置见 [SOURCE_AND_ENVIRONMENT.md](SOURCE_AND_ENVIRONMENT.md)；旧副本盘点见 [OLD_CLONES_INVENTORY.md](OLD_CLONES_INVENTORY.md)。

{ref_table}
上述两组都是本轮连续固定步长请求。VDP 的 T1、T3、T6.32 从同一次运行读取；自适应 T10 仅列为我们已有的能力。候选当前仍不支持自适应与历史保存同时开启，因此未新做配对自适应跑法。

![实际执行时域]({rel}figures/horizons.png)

候选普通、严格模式的两步数值诊断均被算法接受，但不能据此称为保证范围：历史矩阵连乘在两种模式、两种设备上均遗漏精确分量 `-4503599627370497/38685626227668133590597632`。独立归一化反例是两个仿射 Taylor model `1.1*u_i + [-2^-52,2^-52]`，各 u_i∈[-1,1]；多项式常数项为零，余项对称且包含零，符合生产调用链的约束，返回的缩放表示没有包含原精确像集。后者不使用历史矩阵，所有数值均为有限正常量级。反例只证明相应局部包含关系失败，不声称这些操作数已出现在两条冻结长轨迹中。

密集与默认稀疏步进均调用同一历史传播函数；新 clone 的 CUDA profiler 确认矩阵与区间 kernel 确实在 GPU 执行。稀疏缩放使用默认 eager helper，B1/B8/B32 都低于编译胶合路径的 B64 门槛。这些检查排除了旧 editable 包和 CPU fallback 冒充新严格生产路径的可能。早期常数单点帮助函数诊断未强制上述零常数项约束，保留为辅助数据；最终归一化结论只采用满足生产约束的仿射反例。完整误差记账仍未闭合，局部测试数量和与其他工具的接近程度不能替代包含关系依据。[反例数据]({rel}known_witness_replay.json)、[独立缩放反例]({rel}preconditioning_centered_exact.json)、[不修补的理由](CONDITIONAL_PATCH.md)。

每条已接受时间小段分别保存终点 x/y 与整段 x/y 的上下界，整段范围包含步内轨迹移动。发布视角保存各工具实际返回范围；共同视角复用我们的 canonical exporter，并对已经组合完整左右映射的数学对象作精确 Fraction 区间单项式求值，最后向外转为 binary64。它不增加 cutoff、不额外收紧 endpoint、不把不同工具的内部变量强行认作同一变量。Flow* 只增加 const term 只读访问器；普通误差和历史误差按各对象原本的归属导出一次。候选自带 io.metrics 测 tmvPre，故另外保留真正组合后的导出，不混用两类对象。

固定步长以实际 binary64 h 的精确有理数累加对齐时间，四条路线的行按真实时间区间关联，没有插值。完整导出仍可能因各自内部表示和区间依赖不同而得到不同宽度；我们的 CPU 参考和 Flow* 都不是其他实现的数学真值机。比率分母不大于 1e-12 时保留绝对差而不计算倍率。本轮阈值未改。

{width_table}
这是我们 / Flow* 的完整共同前缀摘要。候选的全程四项宽度和严格 / 普通代价没有建立；表中的候选两步行统一标为 DIAGNOSTIC_ONLY。完整请求时域一直显示到 T20/T10，空白尾段保留。[逐步双视角表]({rel}widths_full_prefix.csv)、[时间加权全程摘要]({rel}width_summary.csv)、[指定及最差时刻表]({rel}width_at_checkpoints.csv)。

![Brusselator 上下界]({rel}figures/brusselator_bounds.png)

![VDP 上下界]({rel}figures/van_der_pol_bounds.png)

图中旧停止点仅作历史说明：Brusselator 是加入接受后收紧之前、容量 1000 的 CPU 版本停止于 T7.14；VDP 是更早 CPU 版本的自适应停止点 T6.714914669607182，步长策略不同。这些标线不是本轮固定步长失败，也不是旧 Xiangru 的匹配实验。其原始来源与摘要见 [历史标线记录]({rel}raw_minimal/historical_plot_markers.json)。

![Brusselator 严格宽度比]({rel}figures/brusselator_strict_ratios.png)

![VDP 严格宽度比]({rel}figures/van_der_pol_strict_ratios.png)

{checkpoint_table}
表内采用共同测量的未取整数据计算；此处仅展示有限位数。全程每项最大比率及发生时间在前表，原始 CSV 还包含两端偏移、中心偏移、不相交时间与超出工程阈值的持续时间。更窄不代表更正确。

{process_table}
主要 CPU 测量为单线程、CPU 2；相关测试在其他核运行，因此不声称整机独占。求解时间包含原求解器安全验证和接受后收紧，纯导出另列；完整进程时间含 Python/程序启动和导出。没有重复实测中位数或波动可报告，不能把单次结果称为预热稳定吞吐。新 segment CUDA 扩展首次构建 43.218453059 秒，tape 的编译混在有明确标记的短诊断冷启动中，没有作为单独编译计时捏造数值。

{batch_table}
性能阶段按正确性前提停止。我们的 CPU 与 Flow* 的 B8/B32 配对串行批次、20 步五次预筛和严格 B32 完整三次重复均未开启；没有用 B×B1 估计跨过速度门槛。未发生 B32 OOM，未降为 B16。主批量 GPU 显存峰值和端到端适配器开销也没有测量，因为候选未接入。[全部时间行与缺失原因]({rel}timings_raw.csv)、[计时摘要]({rel}timing_summary.csv)。

两步重复盒子 B2/B8 与 B1 的 40 次逐任务比较均逐位一致；预注册不同盒子的前两个任务另有 8 次逐位一致比较。局部失败任务的已发布状态冻结，仍活动任务与独立运行一致；内部全局历史队列会继续为失败 lane 写入废弃槽位，所以没有宣称可恢复暂停状态。NaN 输入没有发布成功结果，状态维度错配被拒绝；没有发现公开的 checkpoint/resume 承诺，未新增该功能。既有小容量队列测试通过，但候选容量 100/1000 的完整生产长前缀和长期隔离仍因正确性门槛而未执行。[批量表]({rel}batch_equivalence.csv)、[状态原始记录]({rel}candidate_state_checks_with_distinct.json)、[缺失清单]({rel}missing_items.md)。

[最终路线图](ADOPTION_DECISION.md)保留 CPU 独立检验与原生自适应能力，本轮只采用合同、导出、反例和复核工具。将来须先闭合历史矩阵与缩放两处误差，再按原门槛执行完整严格宽度和 B32 速度测试。当前没有授权、源码或 GPU 硬件缺失，也没有向 Xiangru/Huan 推送代码。

证据校验从原始完整对象和逐步记录重新计算时域、四项宽度、P95、最差时刻、mode、设备和计时。另有四项直接篡改测试：修改宽度、完成时域、mode 或运行时间，重新生成外层哈希后仍必须拒绝。哈希只提供可追溯性。[测试结果与复核范围](VALIDATION.md)。fresh clone 检查仅重新验证已提交证据，没有重跑长时间数值实验。
"""
(docs/'REPORT_PLAIN_CHINESE.md').write_text(report)
print('Wrote decision, baseline, and full report from completed records.')
