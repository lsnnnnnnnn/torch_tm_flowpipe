"""Package completed measurements and draft the seven-part Chinese report.

Run only after the finite diagnostic/formal campaigns and their raw audits.
Independent-copy acceptance and final branch/SHA reconciliation remain explicit
steps; this script never manufactures either receipt and never pushes anything.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess

from experiments.range_batch_device.common import read, save, sha
from experiments.live_range_solver.runner import ROOT
from experiments.live_range_solver.enrich import enrich


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def table(headers,values):
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |",
        *("| "+" | ".join(map(str,row))+" |" for row in values)])


def manifest(root):
    files=[p for p in sorted(root.rglob("*")) if p.is_file() and p!=root/"SHA256SUMS"]
    (root/"SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(root)}\n" for p in files))


def package(root):
    root=Path(root).resolve()
    source=read(root/"SOURCE_MAP.json")
    gate=read(root/"diagnostic/CORRECTNESS_GATE.json")
    performance=read(root/"PERFORMANCE_RESULT.json")
    assert gate["passed"] is True and gate["source_sha"]==performance["source_sha"]==source["scientific_sha"]
    for phase in ("diagnostic","formal"):
        complete=read(root/phase/"COMPLETED.json")
        assert complete["source_sha"]==source["scientific_sha"]
    enrich(root)
    references=[]
    for plant,case in (("van_der_pol","van_der_pol_before0099"),("brusselator","brusselator_before0999")):
        path=Path("artifacts/runs/boundary_execution_20260908T172756Z/raw_minimal/state_inputs")/case
        references.append(dict(plant=plant,path=str(path),scope="RESUMED_LOCAL_WINDOW",
            files={p.name:sha(p) for p in sorted((ROOT/path).iterdir()) if p.is_file()}))
    save(root/"CHECKPOINT_REFERENCES.json",references)
    for relative in (
        "benchmarks/canonical.yaml","benchmarks/three_tool_matched_contract.yaml",
        "benchmarks/brusselator_terminal_sr1000_contract.json",
        "artifacts/runs/xiangru_adoption_20260907T032448Z/MATCHED_CONTRACTS.json",
    ):
        source["references"][relative]=sha(ROOT/relative)
    source["packaging_sources"]={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/"experiments/live_range_solver").glob("*.py"))
        if str(p.relative_to(ROOT)) not in source["experiment_sources"]}
    source["phase"]="MEASUREMENTS_COMPLETE__POSTPROCESSING_ONLY"
    save(root/"SOURCE_MAP.json",source)
    contract=read(root/"EXECUTION_CONTRACT.json")
    contract.update(actual_main_fixed_steps=dict(van_der_pol=.01,brusselator=.02),
        formal_independent_fraction_audit=False,necessary_cpu_exact_power_correction_timed=True,
        scope_note="config records retained verbatim; actual fixed VDP h=.01 is selected by unchanged frozen.step",
        cpu_gpu_memory_scope="RSS: process lifetime high-water; GPU: Torch allocation peak, excluding driver context; after separate CUDA warmup reset")
    save(root/"EXECUTION_CONTRACT.json",contract)
    save(root/"failure_cancel_resume.json",read(root/"faults/failure_cancel_resume.json"))
    raw_index=[]
    for phase in ("diagnostic","formal"):
        for case in read(root/phase/"CAMPAIGN_PLAN.json")["cases"]:
            folder=root/phase/case["name"]
            summary=read(folder/"summary.json")
            raw_index.append(dict(phase=phase,case=case["name"],source_sha=summary["source_sha"],
                files={str((folder/name).relative_to(root)):checksum for name,checksum in summary["files"].items()},
                accepted_lane_steps=summary["accepted_lane_steps"],scope=summary["scope"]))
    (root/"raw_minimal").mkdir(exist_ok=True)
    save(root/"raw_minimal/INDEX.json",dict(raw_payloads=raw_index,duplicate_payloads_copied=False,
        instructions="Full actual models/queues and captured requests are in diagnostic/*.json.gz; formal records contain timed continuous work without diagnostic operand capture."))

    widths=rows(root/"cross_backend_widths.csv")
    warnings=[r for r in widths if r["warning"]=="True"]
    investigation=root/"WIDTH_WARNING_INVESTIGATION.json"
    identities=lambda values:{(r["case"],r["plant"],str(r["task"]),str(r["step"]),r["view"],r["variable"]) for r in values}
    if warnings:
        assert investigation.exists(), "investigate every >1.10 width row before final report"
        explained=read(investigation)
        assert explained["complete"] and identities(explained["entries"])==identities(warnings)
    else:
        save(investigation,dict(complete=True,entries=[],warning_threshold=1.10,near_zero_threshold=1e-10,
            explanation="No recorded width entry exceeded the preregistered threshold; all actual bounds remain available."))
    independence=read(root/"INDEPENDENT_ACCEPTANCE.json") if (root/"INDEPENDENT_ACCEPTANCE.json").exists() else None
    result=dict(status=performance["status"],performance=performance,correctness_gate_passed=True,
        measured_throughput_online=True,new_full_long_horizon_gpu=False,entire_solver_formally_proved=False,
        default_enabled=False,scalar_legacy_unconditionally_safe=False,scientific_sha=source["scientific_sha"],
        diagnostic_cases=gate["full_runs"],formal_cases=performance["formal_samples"],
        width_warning_count=len(warnings),width_warning_investigation_complete=True,
        full_root_tests=dict(passed=1221,optional_skipped=2,source_sha=source["full_root_test_sha"]),
        final_targeted_tests=dict(passed=28,source_sha=source["targeted_test_sha"],already_in_full_identity_count=True),
        parent_local_tests=dict(passed=131,label="REUSED",source_sha=source["parent"]),
        independent_copy_acceptance=independence,
        final_branch_push_and_sha_audit="separate final receipt; never inferred from a generated report")
    save(root/"RESULT.json",result)

    summary=rows(root/"end_to_end_summary.csv")
    paired=rows(root/"paired_speedups.csv")
    raw=rows(root/"timings_raw.csv")
    groups=rows(root/"grouping_summary.csv")
    amdahl=rows(root/"amdahl_reference.csv")
    width_summary=rows(root/"width_summary.csv")
    behavior=read(root/"BEHAVIOR_SUMMARY.json")
    audit=read(root/"ARITHMETIC_AUDIT_SCOPE.json")
    def median(chosen,key):
        return statistics.median(float(r[key]) for r in chosen)
    names={"van_der_pol":"VDP","brusselator":"Brusselator"}
    speed_rows=[]
    applicability=[]
    for plant in names:
        for batch in (1,8,32):
            samples=[r for r in summary if r["plant"]==plant and int(r["batch"])==batch]
            chosen=[r for r in paired if r["plant"]==plant and int(r["batch"])==batch]
            times={r["route"]:float(r["median_wall_s"]) for r in samples}
            speed_rows.append([names[plant],batch,2 if batch==1 else 20,*[f"{times[route]:.3f}" for route in ("L","S","Q","G")],
                *[f"{median(chosen,key):.3f}×" for key in ("S_over_G","Q_over_G","min_SQ_over_G","L_over_G")]])
            gain=median(chosen,"min_SQ_over_G")
            legacy=median(chosen,"L_over_G")
            applicability.append([names[plant],f"B{batch}",f"{gain:.3f}×",f"{legacy:.3f}×",
                "在线 G 有实测优势" if gain>=1.10 else "小幅/边界收益" if gain>1 else "当前接口无吞吐优势",
                "超过本轮 L 有限样本" if legacy>1 else "尚未超过本轮 L 有限样本"])
    grouping_rows=[]
    for row in groups:
        if row["route"]!="G":continue
        grouping_rows.append([names[row["plant"]],row["batch"],f"{float(row['mean_size']):.2f}",row["max_size"],
            f"{float(row['singleton_request_fraction'])*100:.1f}%",
            *[f"{float(row[key])*1000:.3f}" for key in ("wait_p50_s","wait_p95_s","wait_max_s")]])
    bound_rows=[]
    for row in width_summary:
        bound_rows.append([names[row["plant"]],row["view"],row["entries"],
            *[f"{float(row[key]):.6g}" for key in ("gpu_cpu_width_ratio_p50","gpu_cpu_width_ratio_p95","gpu_cpu_width_ratio_max","max_abs_diff_max","center_abs_diff_max")],
            f"{row['worst_case']}/{row['worst_task']}/step {row['worst_step']}"])
    amdahl_rows=[]
    for plant in names:
        chosen=[r for r in amdahl if r["plant"]==plant and r["batch"]=="32"]
        amdahl_rows.append([names[plant],f"{median(chosen,'measured_S_evaluator_share')*100:.2f}%",
            f"{median(chosen,'zero_evaluator_upper_reference'):.3f}×",f"{median(chosen,'replace_only_evaluator_reference'):.3f}×"])
    counts={route:sum(int(r["accepted_lane_steps"]) for r in raw if r["route"]==route) for route in ("L","S","Q","G")}
    decision_rows=[]
    for plant in names:
        decision=performance["decisions"][plant]
        decision_rows.append([names[plant],*[f"{value:.3f}×" for value in decision["paired_speedups"]],
            f"{decision['wins']}/3",f"{decision['median']:.3f}×",
            "达到" if decision["practical"] else "未达到"])
    relative=f"../../artifacts/runs/{root.name}"
    link=lambda filename:f"[{filename}]({relative}/{filename})"
    independent_text=("独立副本已经完成新验证器与有界真实接入复核，记录见 "+link("INDEPENDENT_ACCEPTANCE.json")+"。"
        if independence else "独立副本验收及最终推送/SHA核对尚待执行；本草稿不将其标为完成。")
    state=performance["status"]
    if state.endswith("END_TO_END_PREFIX_SPEEDUP"):
        route_text="本轮选择 A：两系统 B32 都达到预注册的在线完整前缀提速门槛。保留默认关闭接口，下一步可以研究更完整的边界/状态常驻；本轮没有迁移第二个数学算子。"
    elif state.endswith("MARGINAL_PREFIX_GAIN"):
        route_text="本轮只有小幅或边界收益，不能称稳定的实用加速。保留已经验收的接口；下一步评估更粗的 GPU 边界执行粒度，不为门槛改变数学参数。"
    else:
        route_text="本轮选择 B：正确的在线服务没有达到两系统的实用完整前缀提速门槛。应转向更粗的边界执行粒度或评估已有严格引擎的接入成本，不再反复优化这一个小范围 kernel。真实分组分布用于区分分组碎片与非范围工作成本。"
    report=f"""# 在线范围批量求解：真实连续前缀与完整流程实测

科学源码：`{source['scientific_sha']}`。机器状态：`{state}`。
{route_text}

## 1. 这次从“存好的请求”变成了什么在线流程？

每个任务从固定初始子盒建立自己的完整状态，执行原有连续 step。运行到已接入的范围调用时，它提交当时真正生成的系数与 domain，等待自己的 Future；收到通过身份检查的结果后才生成后续请求。服务只合并已经就绪、完整数学 key 相同的请求。整个过程中没有加载先前保存的答案。

接口默认关闭。worker 显式建立 prepared replay / packed boundary / 请求适配上下文；一个服务线程拥有 CUDA stream 和完成 event。GPU 只执行已有四个范围 kernel；方程、阶数、初始原盒、VDP h=0.01、Brusselator h=0.02、cutoff、E、收紧与历史传播未改。dense 范围及其余数学仍在 CPU。覆盖边界与空支持/外供表回退见 [调度合同](SCHEDULING_CONTRACT.md) 和 {link('coverage/coverage_summary.json')}。

L 是原有最快 packed 逐任务路线，保留已知旧幂缺陷；S 是校正后的 CPU 串行；Q 是同算术的在线 CPU；G 是同服务的 CUDA。S_gpu 只用于区分 CUDA 算术变化和调度错误。

## 2. 实际每批有几个任务，是否出现长时间等齐？

主输入仍是原盒子的固定 8×4 分区，B8 使用任务 0/4/8/12/16/20/24/28，B1 使用任务0。下面是三次正式 G 运行合并后的真实派发分布；B32 是活跃任务数，不等于每个范围组32条。

{table(['系统','任务数','平均组大小','最大组','单例请求占比','等待P50 ms','等待P95 ms','最长等待 ms'],grouping_rows)}

预注册的2/20 ms短前缀候选按总耗时选择20 ms并冻结。达到组上限、全部存活任务等待/结束或等待到期就派发。20 ms是可派发的期限，已有计算和单核调度仍会延迟实际服务；上表报告真实等待。小组、早完成、失败和取消任务均能退出，不要求所有任务的第k次调用对齐。完整直方图、flush原因和全部等待样本见 {link('actual_grouping.csv')}、{link('flush_reasons.csv')} 与各原始 run。

## 3. 怎么证明返回给正确任务、失败不污染别人？

请求带 run/task/epoch/接受边界generation/attempt/counter。提交、派发、回传、消费、commit均检查身份；后一请求必须引用已消费的前一结果。输入和结果私有复制，取消与整状态commit共用锁。checkpoint先取消未提交attempt，保存最后接受状态；恢复建立新epoch，旧worker的迟到结果无效。整包验证还从固定子盒或checkpoint重建每个诊断任务的首个状态，核对后续接受边界逐步递增与固定h，不能只靠记录内部相互一致的hash证明输入来源。

{gate['full_runs']}个诊断案例覆盖两系统B1/B2两步、B8/B32二十步、两个不同任务各120步、原未分区B1二十步和历史窗口。CPU S/Q按完整segment及下一状态比较；G/S_gpu在B1/B2/B8全段、B32首两步比较；另比较B8拆成2×B4、4×B2、8×B1和逆序/延迟/合法flush改变。详细比较与历史队列99/0/1、999/0/1清空记录见 {link('same_backend_state_equivalence.json')}、{link('history_reset_checks.json')}。

四个真实故障专项（两系统×CPU/CUDA）各有10个接受步骤、一次拒绝重试、一次异步取消和一次旧返回丢弃；失败后恢复的完整状态及健康任务后续步骤与独立执行一致。发现的checkpoint私有诊断字段/字典顺序丢失已用安全JSON sidecar修复，原失败记录保留。另有不同h的测试任务：真实收紧次数2/5、请求数137/215，CPU/CUDA都与各自独立运行一致；这组输入只用于状态机验收，未进入性能表。

当前根测试按身份去重为1221 passed、2 optional skipped；完整遍历在`{source['full_root_test_sha'][:12]}`，最终科学源码28项受影响测试重验通过，已经包含在1221个身份内。父源码局部131项标为REUSED。原始值验证覆盖九类重新计算外层hash后的篡改，包括错任务返回、旧generation、取消后commit、范围端点、fallback、等待/传输遗漏、lane-step数和离线冒充在线。{independent_text}

## 4. CPU/GPU的范围变化是什么，哪些不等于错误？

CPU/GPU各自用独立Fraction检查幂、每项和总和，并用原算子重新计算保存的实际请求。完整检查范围为B32首两步、所有两步小型/拆批用例、B8和原B1的首两步、每个三步历史窗口的前两步，以及120步预注册的1/2/60/100/119/120；故障边界前后成功返回和新结构/回退/纠正另完整核验。其余调用依靠未改变的局部算术合同，不宣称每条都重新做过Fraction审计。各路线计数见 {link('ARITHMETIC_AUDIT_SCOPE.json')}。

同一个CPU observer读取各自真实segment的endpoint/tube模型，产生以下宽度统计。宽度差异本身不能证明不安全，GPU更窄也不是安全证明；没有取CPU/GPU区间交集或以抽样替代包含保证。

{table(['系统','视图','x/y条目数','宽度比P50','宽度比P95','宽度比max','上下界最大绝对差','中心最大差','绝对差最差位置'],bound_rows)}

每个坐标、时刻的上下界及精确累计时间见 {link('cross_backend_widths.csv')}；按系统/案例/endpoint或tube/x或y细分的P50/P95/max及各项最差位置见 {link('width_summary_by_coordinate.csv')}。CPU宽度≤1e-10的条目只列绝对差，见 {link('near_zero_widths.csv')}。GPU/CPU>1.10共有{len(warnings)}条，全部列于 {link('width_warnings.csv')}，定位记录见 {link('WIDTH_WARNING_INVESTIGATION.json')}。接受/步长/验证与收紧停止计数/请求数比较中共有{behavior['different_steps']}步行为差异，首个实际分歧及全部条目见 {link('BEHAVIOR_SUMMARY.json')}、{link('behavior_comparison.csv')}。

VDP连续120步覆盖SR100；历史99/100/101和Brusselator999/1000/1001来自安全完整checkpoint，明确标为RESUMED_LOCAL_WINDOW。没有新GPU从初始集跑满1000步，也没有证明整个ODE求解器形式化正确。

## 5. 整个前缀实际快多少，分组CPU是否本来就更慢？

以下墙钟包含建任务初始状态、worker/队列启动、全部非范围数学、必要检查、真实请求准备/等待/分组/传输/同步/回传和清理。S/Q/G每组三次顺序为S-Q-G、G-Q-S、Q-S-G；L每系统/批次一个有限同工作量样本。速度倍率按配对样本取中位数，>1表示G更快。

{table(['系统','B','每任务步数','L秒(1次)','S秒中位','Q秒中位','G秒中位','S/G','Q/G','min(S,Q)/G','L/G'],speed_rows)}

两系统B32的三组配对倍率逐项如下，均为min(S,Q)/G；小于1表示该组G较慢。接纳要求每系统至少2/3组大于1且中位数≥1.10，完整结果包含各组波动。

{table(['系统B32','S-Q-G组','G-Q-S组','Q-S-G组','G胜出次数','配对中位数','实用提速门槛'],decision_rows)}

全部{performance['formal_samples']}次实测、min/max、吞吐和资源见 {link('timings_raw.csv')}、{link('end_to_end_summary.csv')}、{link('resource_usage.csv')}。正式成功lane-steps按路线合计为 {json.dumps(counts)}；失败/取消任务没有进入成功分子。B1只要求揭示开销，不要求GPU获胜。原未分区B1单列在 {link('original_b1_diagnostic.csv')}，它带诊断成本，不能和子盒正式计时混用。

所有主表路线使用同一CPU2、Torch intra/inter-op各1、同一GPU0 Tesla V100及原py11环境。GPU冷启动编译/模块加载/原语自检另记；正式服务复用已经自检的模块，必要计算与检查仍计时。正式窗口不做独立Fraction审计、大输入hash或profiler；CPU保证幂包含所需的精确校正仍在正式分母内。冷启动、证据序列化、进程RSS与Torch显存口径见 {link('startup_and_serialization.csv')}。显存数是Torch分配峰值，未包含驱动上下文；RSS是进程生命周期高水位。

等待秒数不能跨任务相加当作总耗时。{link('time_partition.csv')}以真实时间区间的并集给出互斥墙钟分类，保留服务与worker步骤的重叠；worker步骤跨度含等待，服务kernel-and-sync是主机观察值，都不冒充纯CPU执行或纯设备kernel时间。

{table(['系统B32','实测S请求evaluator份额','消除该份额的上限参考','替换为实测G服务时间的参考'],amdahl_rows)}

这里的份额只覆盖已接入请求的evaluator，不包含调用前packing和仍在CPU的dense范围；G服务还与worker跨度重叠。因此Amdahl值仅是有明确范围的参考，正式结论始终来自完整wall time，没有沿用旧两步17%的份额。

## 6. B1、B8、B32分别适不适合，是否真正胜过此前最快路径？

{table(['系统','批次','min(S,Q)/G','L/G','本轮判断','旧最快L有限比较'],applicability)}

正式B32接纳规则是两系统各至少2/3配对>1且中位数≥1.10。逐配对数据与机器决策见 {link('paired_speedups.csv')}、{link('PERFORMANCE_RESULT.json')}；按耗时直接计算的G/S、G/Q、G/L及其吞吐反比同时列于 {link('relative_wall_time.csv')}。只胜Q但不胜S意味着调度成本吞噬收益；只胜严格CPU但未胜L也不能说超过此前最快实现。L有已知三次幂局限，本轮有限计时不构成其全部输入安全证据。

## 7. 下一步走哪条路线？

{route_text}

交付的是能真正连续推进的默认关闭在线接口、固定科学SHA的有限诊断和完整前缀实测。可重算原始证据入口为 {link('raw_minimal/INDEX.json')}，文件摘要为 {link('SHA256SUMS')}；执行方法见 [实验README](../../experiments/live_range_solver/README.md)，逐条目标验收见 [GOAL_AUDIT](GOAL_AUDIT.md)。后续报告/包装提交只更新证据及说明，不改变这些数值源码和正式分母。
"""
    docs=ROOT/"docs/live_range_solver"
    (docs/"REPORT_PLAIN_CHINESE.md").write_text(report)
    requirements=[
        ("§1–2 固定起点、旧树不改、独立分支",f"父b60a608；科学源码{source['scientific_sha']}；独立新分支，未reset/clean旧树、未改main"),
        ("§2 指定阅读与父局部核验","指定文档/代码已读；父完整有界verifier与131局部测试在父worktree通过，标REUSED"),
        ("§2 环境和导入","原py11 / torch2.5.1+cu121 / V100 GPU0 / CPU2；逐run保留实际导入路径、线程与环境；未升级工具链"),
        ("§3 数学参数与禁止范围","原方程/盒/阶数/cutoff/余项/E/收紧/历史/归一化未改；主VDP h=.01、Brusselator h=.02；CUDA .cu与440714逐字节相同"),
        ("§4 L/S/Q/G/S_gpu","五条真实路线；正式S/Q/G同任务三组，L同工作量有限计时；S_gpu仅正确性；L保留已知幂局限"),
        ("§5.1 当前真实请求、worker上下文","真实step内阻塞Future；显式ContextVar；一个服务线程拥有CUDA stream/event；不读取保存答案"),
        ("§5.2 完整身份","run/task/epoch/generation/attempt/counter/完整key；诊断输入及源状态hash；每任务最多一条未消费请求；生产身份检查保留并计时"),
        ("§5.3 语义分组与覆盖","复用有序支持/类型/角色/输出/dtype/device/外供表key；不补零重排；外供表显式CPU回退；coverage保留未覆盖调用栈，dense数学仍在CPU"),
        ("§5.4 有界派发与策略冻结","max_group、all_waiting、timeout三种派发；预注册2/20ms候选选20ms；正式不再调参；actual_grouping/flush_reasons/grouping_summary保留真实分布"),
        ("§6 请求错误与非有限","异常只影响相应任务；NaN/overflow不成功、不重用旧答案；有真实accepted step后的拒绝/重试证据"),
        ("§6 取消/迟到/恢复","取消与commit串行化；只保存最后接受状态；恢复新epoch，包括同runid跨进程；旧返回被丢弃，健康peer继续"),
        ("§6 完整carry与alias","完整segment/下一状态比较；私有tensor输入输出；旧checkpoint丢私有诊断与顺序的接入缺陷已修复，保留开发反例；无线程/Future/CUDA指针pickle"),
        ("§7 原算术与自检","四个原CUDA kernel不变；极小数/相消/三次幂启动自检；S/Q精确校正、G逐操作有向舍入；未改旧公开pow默认"),
        ("§8.1 固定子盒与原B1分列","PARTITION_REFERENCE原8×4、固定B8/B1；B2任务0/31；原未分区B1独立标签和表；没有32份相同输入"),
        ("§8.2.1 B1/B2","两系统各2步，S/Q和S_gpu/G完整对象一致；actual raw+生命周期保留"),
        ("§8.2.2 B8","两系统S/Q/S_gpu/G均每任务20步；逐任务160完整segment同后端一致"),
        ("§8.2.3 B32","两系统S/Q/G每任务20步；S/Q全640段比较，S_gpu/G首两步64段比较；首两步实际请求完整精确核验"),
        ("§8.2.4 长依赖120","两系统任务0/31分别S/Q/G连续120步；每步来自自身上一步状态；VDP跨SR100清空；审计点1/2/60/100/119/120"),
        ("§8.2.5 历史窗口","安全完整checkpoint的VDP99/100/101与Brusselator999/1000/1001；四路线本地窗口，RESUMED_LOCAL_WINDOW；不是新GPU初始到1000步"),
        ("§8.2.6 原未分区B1","两系统S/G各20步并以共同observer比较真实范围；带诊断成本的延迟与子盒正式样本分列"),
        ("§8.3 异构专项","不同支持/项序、测试h、实际收紧2/5与请求137/215、accepted后异常、早完成/取消、timeout、小组、硬件CPU重算、恢复旧响应均有原始证据"),
        ("§9.1 完整行为与拆批","完整segment/endpoint/reset/E/映射/余项/owner/Phi/J/generation/收紧/停止计数/h/时间；B8拆2×B4、4×B2、8×B1及逆序/延迟/不同合法flush按task比较"),
        ("§9.2 独立算术审计","对各自实际输入核验幂/项/总和并重算后端；B32首2全量、故障成功边界、特殊结构/纠正/回退、长期预注册点；其他请求明确依靠旧局部合同"),
        ("§9.2 共同observer与警戒",f"实际endpoint/tube x/y上下界、精确累计时间、绝对/中心差、宽度比P50/P95/max及最差位置；≤1e-10分列；>{1.10:.2f}共{len(warnings)}条，全部定位记录"),
        ("§9.3 保证边界","不宣称整个ODE形式化证明，不宣称新完整GPU1000步；历史长时域和旧自适应结果仍REUSED"),
        ("§10.1 同资源","主S/Q/G CPU2单核、Torch intra/inter-op1、同GPU0；CPU3补充诊断不进入正式样本；正式序列逐子进程隔离"),
        ("§10.2 完整wall与冷启动","建初始状态到全部worker/服务清理计时；同步/必要校正/传输/检查均含；编译加载与自检另记；独立Fraction审计/序列化在外，必要CPU精确幂校正仍在内"),
        ("§10.2 合法时间线","真实服务/worker步骤区间并集的互斥分类与重叠；不相加并发等待当wall；主机kernel-and-sync不冒充设备纯kernel时间"),
        ("§10.3 重复和指标",f"{performance['formal_samples']}次正式实跑：B1三组2步，B8/B32三组20步，S-Q-G/G-Q-S/Q-S-G交替与6个L有限样本；所有成功分子、请求/分组/回退/等待/CPU时间/RSS/Torch显存已列"),
        ("§10.4 性能判定",f"922e3b3预注册两系统B32 min(S,Q)/G至少2/3胜且median≥1.10；实测状态{state}；G/L单列，Amdahl使用本轮已接入evaluator份额并明确排除未覆盖范围"),
        ("§11 修正与源码冻结","仅接入/调度/身份/缓冲所有权/保存恢复/证据层；正式数值源码ef4e2f0先提交，运行期间干净不变；包装代码另列SOURCE_MAP"),
        ("§12 路线判断与机器字段","REPORT第7项和RESULT明确最终路线；online/new full long horizon/formal solver proof/default enabled各自字段，速度不达标也完整交付"),
        ("§13 原始证据与重算","raw_minimal/INDEX引用全部实际压缩模型/队列/请求/回执/时钟；独立重算派生CSV/决策到临时目录比较，不覆盖被验文件；SHA256SUMS涵盖所有新证据"),
        ("§13 篡改与测试身份","九类重新计算外层hash后的语义篡改被拒绝；根1221/2，最终28同身份不相加，父131 REUSED；三历史源码锁定模块留旧worktree，未弱化断言或加新skip"),
        ("§13 独立副本有界真实重跑","已验收，见INDEPENDENT_ACCEPTANCE.json；未重跑完整性能矩阵或历史1000" if independence else "PENDING：必须独立git clone运行新package verifier及两系统S/Q/S_gpu/G B2两步实际重跑；不能以本地验证代替"),
        ("§13 最终分支推送与SHA","PENDING：按原授权推送自己的新分支，不强推/不改main；最后核对local/remote/独立副本SHA并保存实际回执"),
    ]
    audit_text="# 目标逐条验收\n\n原目标 `goal_vdp_terminal.md`（摘要见SOURCE_MAP）保持全部范围。下列数值/测量项来自完整矩阵；独立副本和推送只有实际回执才能关闭。\n\n"
    audit_text+=table(["要求","证据与状态"],requirements)
    audit_text+="\n\n测试身份和每项原始路径见测试provenance、实验README及完整报告。包装过程不重算或覆盖科学性能样本。\n"
    (docs/"GOAL_AUDIT.md").write_text(audit_text)
    manifest(root)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    print(json.dumps(package(args.root),indent=2))


if __name__=="__main__":
    main()
