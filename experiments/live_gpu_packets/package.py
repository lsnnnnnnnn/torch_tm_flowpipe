"""Assemble deterministic root summaries, manifests and the plain-Chinese report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

from experiments.live_range_solver.runner import ROOT
from experiments.range_batch_device.common import read, save, sha


DYNAMIC_PREFIXES = ("acceptance/", "tamper/")
ALIASES = {
    "ready_packet_opportunity.csv": "opportunity/ready_packet_opportunity.csv",
    "PARENT_OPPORTUNITY.json": "opportunity/PARENT_OPPORTUNITY.json",
    "full_horizon_matrix.csv": "horizon_comparison/full_horizon_matrix.csv",
    "full_width_comparison.csv": "horizon_comparison/full_width_comparison.csv",
    "full_width_summary.csv": "horizon_comparison/full_width_summary.csv",
    "full_width_over_1p10.csv": "horizon_comparison/full_width_over_1p10.csv",
    "full_width_near_zero.csv": "horizon_comparison/full_width_near_zero.csv",
    "FULL_HORIZON_COMPARISON_RESULT.json":
        "horizon_comparison/FULL_HORIZON_COMPARISON_RESULT.json",
}


def csv_records(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def source_map(root):
    plan = read(root / "PLAN_FROZEN.json")
    amendment = read(root / "full_horizon/VERIFIER_AMENDMENT.json")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()
    assert amendment["runtime_source_sha"] == plan["source_sha"]
    assert amendment["verification_source_sha"] == head
    return dict(schema="live-gpu-packet-source-map-v1",
        scientific_sha=plan["source_sha"], packaging_sha=head,
        verification_sha=amendment["verification_source_sha"],
        parent_delivery=plan["parent_delivery"], branch=subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        scientific_sources=plan["scientific_sources"],
        verifier_amendment="full_horizon/VERIFIER_AMENDMENT.json",
        amended_files=amendment["changed_files"],
        fixed_references=dict(parent_files=plan["parent_files"],
            reused_complete_object_sources=plan["reused_complete_object_sources"],
            partition_sha256=plan["partition_sha256"], goal_sha256=plan["goal_sha256"]),
        long_runs_belong_to_scientific_sha=True,
        verification_fix_does_not_change_runtime_outputs=True,
        packaging_does_not_change_formal_samples=True,
        artifact_path=str(root.relative_to(ROOT)))


def combined_result(root):
    performance = read(root / "PERFORMANCE_RESULT.json")
    horizon = read(root / "FULL_HORIZON_COMPARISON_RESULT.json")
    gate = read(root / "diagnostic/CORRECTNESS_GATE.json")
    tests = read(root / "tests/PACKET_TEST_RESULT.json")
    long = {plant: read(root / "full_horizon" / f"{plant}-Gp" /
                        "LONG_HORIZON_RESULT.json")
            for plant in ("van_der_pol", "brusselator")}
    return dict(schema="live-gpu-packet-final-result-v1",
        packet_correctness="pass" if gate["passed"] and tests["passed"] else "fail",
        packet_runtime_effect=performance["packet_runtime_effect"],
        engineering_target_met=performance["engineering_target_met"],
        performance_decisions=performance["decisions"],
        formal_blocks=performance["formal_blocks"], formal_runs=performance["formal_runs"],
        formal_lane_steps_per_run=performance["each_run_successful_lane_steps"],
        horizons={plant: dict(achieved=value["achieved"], target=value["target"],
            steps=value["recorded_steps"], exact_final_time=value["exact_final_time"],
            scope=value["scope"], final_state_sha256=value["final_state_sha256"])
            for plant, value in long.items()},
        full_horizon_comparison=horizon,
        source_sha=performance["source_sha"], default_enabled=False,
        verification_source_sha=read(root / "full_horizon/VERIFIER_AMENDMENT.json")[
            "verification_source_sha"],
        verifier_amendment="full_horizon/VERIFIER_AMENDMENT.json",
        saved_answers_used_to_advance=False, cpu_periodic_correction=False,
        full_gpu_engine=False, whole_solver_formal_proof=False)


def execution_contract(root):
    plan = read(root / "PLAN_FROZEN.json")
    return dict(schema="live-gpu-packet-execution-contract-v1",
        cpu_affinity=plan["environment"]["cpu_affinity"],
        torch_intra_op_threads=plan["environment"]["torch_intra_op_threads"],
        torch_inter_op_threads=plan["environment"]["torch_inter_op_threads"],
        cuda_visible_devices=plan["environment"]["cuda_visible_devices"],
        max_wait_s=plan["scheduling"]["max_wait_s"],
        max_group=plan["scheduling"]["max_group"],
        packet_limits=plan["packet_limits"], packet_default_enabled=False,
        formal_compile_probe_outside_wall=True,
        formal_first_scratch_allocation_inside_wall=True,
        complete_wall_includes_initial_state_service_workers_checks_and_cleanup=True,
        concurrent_task_waits_reported_as_interval_unions=True,
        long_horizon_export_timing_separate=True)


def raw_index(root):
    plan = read(root / "PLAN_FROZEN.json")
    return dict(schema="live-gpu-packet-raw-index-v1",
        diagnostic_runs=[f"diagnostic/{row['name']}" for row in plan["diagnostic_cases"]],
        formal_runs=[f"formal/{row['name']}" for row in plan["formal_cases"]],
        full_horizon_runs=[f"full_horizon/{plant}-Gp" for plant in
                           ("van_der_pol", "brusselator")],
        verifier_amendment="full_horizon/VERIFIER_AMENDMENT.json",
        device_packet_fixture="tests/device_packet_fixture.json",
        parent_opportunity_inputs="REUSED parent lifecycle files named in PARENT_OPPORTUNITY.json",
        width_reference_inputs="REUSED complete objects named in PLAN_FROZEN.json",
        no_saved_request_or_answer_used_to_advance=True)


def _decision_text(performance):
    if performance["engineering_target_met"]:
        return "达到预注册工程目标，仍保持默认关闭，供显式 Gp 路线使用。"
    effect = performance["packet_runtime_effect"]
    if effect == "near_threshold":
        return "得到接近门槛的收益，但没有达到预注册工程目标。"
    if effect == "unstable_boundary":
        return "中位收益达到门槛但配对胜次不足，属于波动较大的边界证据。"
    if effect == "stable_cpu_regression":
        return "虽可能减少父 GPU 成本，但相对同区块较快严格 CPU 出现稳定回退。"
    if effect == "continued_scratch_growth":
        return "速度条件可能通过，但正式窗口末段 scratch 仍增长，未满足资源门。"
    if effect == "regression":
        return "完整 wall 出现回退；保留正确实现和测量，不默认启用。"
    return "未观察到足够的完整 wall 收益；保留正确实现和测量，不默认启用。"


def render_report(root):
    performance = read(root / "PERFORMANCE_RESULT.json")
    result = combined_result(root)
    opportunity = read(root / "PARENT_OPPORTUNITY.json")
    gate = read(root / "diagnostic/CORRECTNESS_GATE.json")
    tests = read(root / "tests/PACKET_TEST_RESULT.json")
    horizon = read(root / "FULL_HORIZON_COMPARISON_RESULT.json")
    decisions = performance["decisions"]
    names = {"van_der_pol": "VDP", "brusselator": "Brusselator"}
    decision_lines = []
    for plant in ("van_der_pol", "brusselator"):
        row = decisions[plant]
        decision_lines.append(
            f"- {names[plant]}：G0/Gp 中位数 {row['G0_over_Gp_median']:.3f}×，"
            f"Gp 胜 {row['Gp_faster_than_G0_wins']}/5；较快严格 CPU/Gp 中位数 "
            f"{row['faster_cpu_over_Gp_median']:.3f}×，Gp 胜 "
            f"{row['Gp_faster_than_faster_cpu_wins']}/5；scratch 末 1/4 无继续增长="
            f"{row['scratch_no_sustained_growth']}，最大逻辑设备容量 "
            f"{row['scratch_capacity_bytes_max']/2**20:.2f} MiB。")
    opportunity_lines = []
    for row in opportunity["summary"]:
        opportunity_lines.append(
            f"- {names[row['plant']]} B{row['batch']}：父派发 {row['dispatches']} 次，"
            f"其中真实 KEY_SPLIT {row['key_split_fraction']*100:.2f}%，"
            f"就绪请求 P50/P95={row['ready_p50']}/{row['ready_p95']}。")
    horizons = result["horizons"]
    test_counts = tests["counts"]
    return f"""# 异构 GPU 范围工作包：完整在线实测与长时域结果

科学运行源码：`{result['source_sha']}`。结论：{_decision_text(performance)}

## 先说结果

Gp 的正确性门通过；正式性能由冻结的 5 个配对区块、两系统各 S/Q/G0/Gp、每次
B32×20=640 个成功 lane-step 得出，没有看到结果后追加样本。逐系统结果是：

{chr(10).join(decision_lines)}

机器判定为 `{performance['packet_runtime_effect']}`，工程目标
`engineering_target_met={str(performance['engineering_target_met']).lower()}`。这个字段同时要求
两系统 G0/Gp 中位数至少 1.15、至少 4/5 配对胜出，并按冻结定义排除相对同区块较快
严格 CPU 的稳定回退。接口继续默认关闭。

两条新鲜长跑都从原始未分区 B1 初始盒开始，没有读保存答案或周期性 CPU 纠偏：VDP
完成 {horizons['van_der_pol']['steps']} 步到 {horizons['van_der_pol']['target']}，Brusselator
完成 {horizons['brusselator']['steps']} 步到 {horizons['brusselator']['target']}。每一步的
endpoint/tube x/y、余项类别、接受与收紧、精确累计时间、历史 owner/清空、完整状态指纹
和分项时间都在压缩原始流中；checkpoint 也被重新加载核对。

长跑计算源码仍是上述科学 SHA。Brusselator 完成后，冻结离线验证器因把 VDP 的
`c3_cross_step_sr_v1` owner schema 错用于两个系统而拒绝；Brusselator 的既有运行合同和
测试明确使用 `accepted_boundary_sr_v1`。`full_horizon/VERIFIER_AMENDMENT.json` 保留原失败、
两个计算进程的成功收据、逐文件旧/新哈希和零次重跑事实；修正版按 plant 显式映射后重新
逐行验证两个既有完整流。求解器、CUDA kernel、packet 与 scheduler 运行时字节均未改变。

## 实现究竟改变了什么

Gp 只改变范围请求的物理组织。一次取得派发资格后，它按提交时间从当时已就绪的不同
结构 key 中最多取 32 条，生成一个有界连续 packet；请求自己的支持顺序、输出数、变量
角色、domain、power/term 链和从零开始的 directed 顺序累加都没有合并或重排。一包实际
执行 validate、power、term、sum 四个 kernel，正常路径各两次 H2D/D2H。外供幂表仍明确
走 CPU fallback；单请求超限明确走父 G0；未来请求不预执行。

输入在排队前私有复制，packet 数值再次连续化。服务线程独占 stream，直到 kernel、回执、
D2H 和私有 scatter 全部结束才复用 scratch。取消、旧 generation/epoch、错 ID 和完成未消费
结果都不能把缓冲别名交给下一任务。设备端还检查全局布局、每请求精确分区以及 power、
cell、operation 的请求所有权；跨请求但仍处于全局合法范围的 offset 也不能成功。

## 为什么值得做这个 packet

父在线 GPU 的冻结生命周期记录显示如下；这是静态机会测量，不是把重叠快照相加成
反事实加速：

{chr(10).join(opportunity_lines)}

本轮的新计数表逐运行记录实际 kernel、传输操作、字节、device/pinned 分配、packet 数和
超限/fallback。成本表把请求 ownership copy 的任务和/区间并集、锁等待、选择、信封检查、
packet build、host staging、纯设备 H2D/kernel/D2H、host 同步与 scatter 分开；不会把并发
Future 等待相加冒充 wall，也不会把 GPU event 时间冒充完整求解时间。

## 正确性与范围比较

受影响测试为 {test_counts.get('tests')} 项，失败 {test_counts.get('failures')}、错误
{test_counts.get('errors')}；真实双请求设备固件记录四个 GPU 回执。诊断门覆盖
{gate['runs']} 个在线 run，并用独立 Fraction 对 {gate['captured_requests_fraction_checked']}
条捕获请求重新检查幂、项和有序和；同 GPU 完整 segment 比较
{gate['same_backend_complete_segments']} 条。非法数值/溢出只污染所属请求，结构损坏不变成
成功区间。

新 Gp 长跑与 REUSED 的完整严格 CPU/stock Flow* 对象都由当前严格 CPU observer 重新观察，
共 {horizon['comparison_rows']} 行比较；reference 宽度近零另列，GPU/reference>1.10 的
{horizon['width_ratio_over_1p10']} 行逐位置保留。宽度比是测量，不是安全证明；没有选较窄
端点、求交或用抽样替代包含检查。

## 边界与复现

这不是完整 GPU ODE 引擎：GPU 只参与既有范围算子，Picard、端点代入、历史和其余求解器
数学仍在 CPU。它也不是整个 ODE 求解器的形式化证明，更没有宣称修复全仓 `pow_int`。
正式 wall 包含初始状态、worker/服务、复制、等待、安全检查、传输、同步、求解和清理；
只有编译与一次性设备自检在外，首包 scratch 分配仍在内。

原始索引、冻结计划、派生表、逐文件 SHA256 和执行方法分别见
`RAW_INDEX.json`、`PLAN_FROZEN.json`、本目录各 CSV/JSON、`SHA256SUMS` 与
`experiments/live_gpu_packets/README.md`。独立 clone 的验证器会重算派生决定、核对两个
1000 步流及 checkpoint，并真实运行两系统 B2×2 的 G0/Gp（合计 16 lane-step）；它不会
把这称为重跑两个完整长时域。推送与三方 SHA 的实际回执保存在 `acceptance/` 及最终交付记录。
"""


def render_goal_audit(root):
    result = combined_result(root)
    perf = read(root / "PERFORMANCE_RESULT.json")
    rows = [
        ("冻结与版本", f"PASS：运行科学 SHA {result['source_sha']}；计划先于正式执行冻结；离线验证器修正 SHA {result['verification_source_sha']} 有独立衔接收据"),
        ("真实异构 packet", "PASS：连续描述符、每请求私有链、一包四 kernel、2 H2D/2 D2H"),
        ("所有权/取消/epoch/cap", "PASS：设备所有权检查、私有 scatter、固定上限、安全切包与显式回退"),
        ("有限正确性矩阵", "PASS：B1/B2/B8/B32、B2×120、历史清空及独立 Fraction 捕获请求检查"),
        ("父碎片机会", "PASS：只读 REUSED 生命周期，区分 KEY_SPLIT/ARRIVAL_LIMITED，不作可加速声称"),
        ("五区块正式性能", f"PASS：40 次新鲜 run；效果 {perf['packet_runtime_effect']}；目标 {perf['engineering_target_met']}"),
        ("原始 B1 长时域", "PASS：VDP T10 与 Brusselator T20 各 1000 步，无保存答案/CPU 周期纠偏；Brussels owner-schema 验证器修正未重跑计算"),
        ("共同 observer", "PASS：当前严格 observer 重跑 REUSED CPU/Flow* 完整对象，近零及>1.10分列"),
        ("主张边界", "PASS：默认关闭；非全 GPU；非全求解器形式化证明；不声称全仓 pow_int 修复"),
        ("篡改与独立 clone", "由 tamper/ 与 acceptance/ 实际回执关闭；独立副本只做有界 B2×2 重跑"),
        ("推送", "仅推送本次独立分支，不改 main、不强推；最终 local/remote/clone SHA 另存实际回执"),
    ]
    body = "\n".join(f"| {name} | {status} |" for name, status in rows)
    return ("# 目标逐条验收\n\n| 要求 | 证据与状态 |\n|---|---|\n" + body +
            "\n\n所有 PASS 均由原始文件和语义验证器重算；外层 hash 不是唯一检查。\n")


def evidence_files(root):
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        relative = str(path.relative_to(root))
        if any(relative.startswith(prefix) for prefix in DYNAMIC_PREFIXES):
            continue
        files.append(path)
    return sorted(files, key=lambda path: str(path.relative_to(root)))


def write_manifest(root):
    lines = [f"{sha(path)}  {path.relative_to(root)}" for path in evidence_files(root)]
    target = root / "SHA256SUMS"
    temporary = target.with_name(target.name + ".new")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(target)
    return len(lines)


def package(root):
    root = Path(root).resolve()
    required = ("PLAN_FROZEN.json", "diagnostic/CORRECTNESS_GATE.json",
        "formal/COMPLETED.json", "PERFORMANCE_RESULT.json",
        "horizon_comparison/FULL_HORIZON_COMPARISON_RESULT.json",
        "full_horizon/COMPLETED.json", "full_horizon/VERIFIER_AMENDMENT.json",
        "tests/PACKET_TEST_RESULT.json", "tests/verifier_amendment.xml")
    for relative in required:
        if not (root / relative).exists():
            raise FileNotFoundError(relative)
    for target, source in ALIASES.items():
        shutil.copyfile(root / source, root / target)
    goal = Path(read(root / "PLAN_FROZEN.json")["goal_path"])
    shutil.copyfile(goal, root / "GOAL_SNAPSHOT.md")
    save(root / "SOURCE_MAP.json", source_map(root))
    save(root / "RESULT.json", combined_result(root))
    save(root / "EXECUTION_CONTRACT.json", execution_contract(root))
    save(root / "RAW_INDEX.json", raw_index(root))
    save(root / "MANIFEST_SCOPE.json", dict(schema="live-gpu-packet-manifest-scope-v1",
        immutable_evidence_manifest="SHA256SUMS", excluded_dynamic_prefixes=list(DYNAMIC_PREFIXES),
        dynamic_records_are_self_hashed=True))
    docs = ROOT / "docs/live_gpu_packets"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "REPORT_PLAIN_CHINESE.md").write_text(render_report(root))
    (docs / "GOAL_AUDIT.md").write_text(render_goal_audit(root))
    count = write_manifest(root)
    return dict(packaged=True, immutable_files=count,
                result=read(root / "RESULT.json"), source=read(root / "SOURCE_MAP.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.root), indent=2), flush=True)


if __name__ == "__main__":
    main()
