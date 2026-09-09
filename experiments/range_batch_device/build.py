"""Package this operator's evidence, without copying the parent long-run archives."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import shutil

from .common import ROOT,RUN,PARENT,read,save,sha,head,numerical_sources,read_records
from .derive import timing_summary,decision,projected,csv_write,width_differences
from .test_accounting import account
from .verify import NUMERICAL_SHA,BASE,write_manifest


def cost_rows(root=RUN):
    rows=read(root/"raw/timing/costs.json");groups=defaultdict(list)
    for r in rows:groups[r["plant"],r["batch"],r["backend"]].append(r)
    out=[]
    for (plant,batch,backend),records in sorted(groups.items()):
        row=dict(plant=plant,batch=batch,backend=backend,scope="SEPARATE_INSTRUMENTED_TRAVERSAL")
        for key in ("sparse_input_packing_s","grouping_and_packing_s","compute_and_transfers_s","scatter_and_wrap_s","fallback_s", "h2d_and_structure_s","kernel_and_sync_s","d2h_and_checks_s"):
            row[key]=sum(r.get(key,0.) for r in records)
        out.append(row)
    return out


def report(summary,result,tests):
    cpu=read(RUN/"cpu_batch_equivalence.json");gpu=read(RUN/"cuda_arithmetic_checks.json")
    widths=width_differences()
    table={(r["plant"],r["batch"],r["mode"]):r for r in summary}
    names={"van_der_pol":"VDP","brusselator":"Brusselator"}
    end_rows=[];pure_rows=[];group_rows=[]
    for plant in names:
        for b in (1,8,32):
            s=table[plant,b,"S_scalar_full"];c=table[plant,b,"B_cpu_full"];g=table[plant,b,"GPU_full_roundtrip"]
            cc=table[plant,b,"CPU_tensor_compute"];r=table[plant,b,"GPU_resident"]
            end_rows.append(f"| {names[plant]} | {b} | {c['requests']} | {s['median_s']:.6f} | {c['median_s']:.6f} | {g['median_s']:.6f} | {c['median_s']/g['median_s']:.3f}× |")
            pure_rows.append(f"| {names[plant]} | {b} | {cc['median_s']:.6f} | {r['median_s']:.6f} | {cc['median_s']/r['median_s']:.3f}× | {r['peak_allocated_bytes']/2**20:.2f} |")
            group_rows.append(f"| {names[plant]} | {b} | {c['groups']} | {c['effective_mean_group_size']:.3f} | {c['median_group_size']} | {c['max_group_size']} | {c['singleton_request_fraction']:.2%} |")
    status=result["status"]
    if status.endswith("CUDA_LOCAL_PILOT_USEFUL"):
        next_step="真实 B32 请求的完整往返成本也通过了预定门槛，值得下一轮设计能汇合独立任务的边界调度，并评估更完整边界状态常驻。此次未验证调度实现和全求解器吞吐；B1 的标量旧路径仍有其价值，不将新接口默认打开。"
    elif status.endswith("CUDA_RESIDENCY_REQUIRED"):
        next_step="设备常驻计算有明确价值，逐次传输不足以稳定兑现同等收益。下一轮应决策完整边界状态常驻或复用现成严格 GPU 架构；不建议逐次 B1 跨设备，也不继续围绕 CPU B1 1.5× 做小修。"
    else:
        next_step="当前测量未达到预先固定的设备收益门槛。保留数学接口和负结果，先解决实际分组/调度与数据表示的缺口；不迁移第二个算子寻找漂亮数字。"
    projected_rows=[]
    for p in projected():
        if p["batch"]==32:
            projected_rows.append(f"| {names[p['plant']]} | {p['requests']} | {p['observed_range_fraction']:.2%} | {p['projected_cpu_aggregate_work_speedup']:.3f}× | {p['projected_gpu_aggregate_work_speedup']:.3f}× | {p['ideal_range_elimination_upper_bound']:.3f}× |")
    build=read(RUN/"raw/timing/environment.json")["build"]
    text=f"""# 真实任务批量范围与 CUDA 局部原型

正式状态：`{status}`。

1. **现在是否真的能让不同任务一起算？** 能。新 `evaluate_range_requests`
   按完整语义分组，组内不同任务共用张量乘法或 CUDA kernel，并按请求 ID 返回
   私有结果。两系统分别用固定 8×4 初始子盒的 32 个不同任务，各取真实两步。
   它是“并发任务请求采集与回放”，尚无完整并发 flowpipe 调度器。另有 12 个
   已存完整状态的早期、中间窗口、末期和历史清空附近请求，明确标为
   `OFFLINE_REQUEST_REPLAY`。中间窗口使用实际 step101/121，未假称几何中点。

2. **实际能组成多大的组？** 下表以同一名义步骤、同一请求依赖阶段分组，
   每阶段每任务至多一个请求。不同支持顺序不合组，不补零，不提前执行后续请求。
   B1 固定任务 0，B8 固定任务 0/4/8/12/16/20/24/28，B32 使用全部任务。

| 系统 | 任务数 | 组数（两步） | 平均有效组大小 | 组大小中位数 | 最大组 | 单例请求比例 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(group_rows)}

3. **怎样保证不漏误差、不串任务？** 有序支持、类型、时间/状态角色、外供表、
   输出结构及 dtype/device 都进入键。normal 域必须在归一化范围内；外供表使用
   原 evaluator 并校验实际表值。结构缓存不保存系数、domain、h 或幂；输入不写入，
   每个 ID 的输出单独复制，mask/取消/非法/溢出分别处理。CPU 保留旧逐操作次序，
   对幂端点使用精确有理数校正；发现并固定了旧三次幂“一 ULP 仍不足”的反例。
   该修正仅在新入口，原 CPU B1 算术与默认开关未改。本轮 {cpu['requests']:,} 个
   真实请求与旧 B1 全部逐位相同，每个后端分别检查 {cpu['powers']:,} 个幂范围和
   {cpu['terms']:,} 个项及全部总和。GPU 使用显式上下舍入加乘与有限次整数幂，
   关闭 FMA 合并/FTZ。GPU 有 {gpu.get('legacy_bitwise_different',0):,} 个结果与 CPU
   边界不同，全部通过自己的精确包含检查；不要求两个正确区间互相包含。
   最大端点绝对差为 {widths['max_absolute_endpoint_difference']:.6g}；对 CPU 正宽区间，
   GPU/CPU 宽度比中位数为 {widths['gpu_to_cpu_width_ratio_for_positive_cpu_width']['median']:.12g}，
   min/max 为 {widths['gpu_to_cpu_width_ratio_for_positive_cpu_width']['min']:.6g} / {widths['gpu_to_cpu_width_ratio_for_positive_cpu_width']['max']:.6g}。
   零宽与极小算术误差区间另记，不能把这些算术差异解释成新 tightness 算法。
   这证明的是声明范围内的算子合同，不是整个求解器的形式化证明。

4. **三种现实的时间各是多少？** 以下秒数是同一任务集两步产生的全部范围请求
   合计 wall time 的五区块中位数。完整表包含原稀疏输入打包、检查、分组、计算、
   scatter/包装；GPU 往返再包含 H2D/D2H 和同步。所有请求均有限成功。

| 系统 | 任务数 | 请求数 | 旧 B1 逐请求 S / 秒 | 分组 CPU B / 秒 | GPU 完整往返 / 秒 | B / GPU |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(end_rows)}

下面单列显式保留输入的计算表，不将其当作当前 CPU 调用者的完整成本。
GPU resident 包含 kernel 输入检查、分配、四个 kernel 的完成和同步；CPU 纯计算
仍包含每次重算/校正幂和有限性检查，但输入已分组、已打包。

| 系统 | 任务数 | CPU 张量计算 / 秒 | GPU resident / 秒 | 计算倍率 | GPU 峰值分配 MiB |
|---|---:|---:|---:|---:|---:|
{chr(10).join(pure_rows)}

实际 GPU 为 V100-SXM2-16GB (sm70)，驱动 580.159.03；Python 3.11.15、torch
2.5.1+cu121。本原型实际使用已安装 NVRTC {build['nvrtc_version'][0]}.{build['nvrtc_version'][1]}，
首次编译/加载 {build['compile_and_load_s']:.6f} 秒；系统另有 nvcc 12.6，未用于此构建。
正式测量 CPU2、单线程，每组 resident 输入显式复用 6 次（1 次预热、5 次测量）。
五区块交替模式顺序，无 profiler；每行 min/max、请求/秒、峰值显存、原始时钟与
分组/打包/传输分项均在 CSV/JSON。B1 数据完整展示，即使某个比较更慢也保留。

5. **快的是请求还是整个求解器？** 主表是完整局部请求，纯计算表才是常驻算子。
   CPU 用真实完整状态连续核对 VDP99/100/101、Brusselator999/1000/1001，以及
   两系统各两个独立任务的两步；完整 segment 逐位相同，失败回滚、checkpoint
   恢复、重复观察均保持。CUDA 仅把 B1 范围操作 offload，每系统实际推进 20 步，
   首两步及后续均消费前一步完整状态，端点 E/历史路径没有断开。这些是接入诊断，
   没有测得完整 GPU solver 或多任务 solver 的吞吐。

下面只是 `PROJECTED`：将这批真实请求的次数和测得完整请求成本放回同一任务的
步骤工作量。非范围部分仍按串行相加；分组/打包/传输开销另列在原始投影数据中。
范围份额来自独立观察遍历，整步分母来自另一无观察器遍历，因此仍是估计。

| 系统 / B32 | 同批请求数 | 观察范围份额 | CPU 工作量投影 | GPU 往返工作量投影 | 范围全消除上限 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(projected_rows)}

6. **下一步选择什么？** {next_step}
   设备门槛在正式计时前冻结：两系统五个配对区块都达到 resident ≥2× 或完整
   offload ≥1.2× 才接受对应用途。没有选择最好区块，没有扩到第三个系统或算子。

7. **哪些旧成果保留、哪些没有新测？** 端点代入 E、prepared replay、原 CPU B1
   RangePlan、历史容量、收紧规则、ODE、阶数、步长、cutoff 及误差预算均保留。
   子盒仅为独立任务诊断，不替代原 B1 benchmark；不拿子盒宽度与未切分 Flow*
   比紧度。旧两系统各 1000 步、自适应 246 接受/35 拒绝、1121/575 秒均为
   `REUSED`，没有新鲜长跑或 Flow* 计时，也没有把旧秒数作为本轮分母。

当前测试按身份去重：{tests['unique_totals'].get('passed',0)} passed / {tests['unique_totals'].get('skipped',0)}
个原有可选 skipped。首次根测试有一个历史 subprocess 因 PATH 找不到 python
失败；补齐现有 py11 PATH 后该模块 3 项通过，原失败及修复后记录都保留，重跑不
重复加总。父快照验证器与 122 项局部测试单列 REUSED；独立副本复查也不重复计数。

数值源码：`{NUMERICAL_SHA}`。测试源码与证据代码另见 SOURCE_MAP；封装提交以
独立副本 verifier 输出的 checkout SHA 为准，避免自引用哈希。新 verifier 从
原始请求/结果重算分组、精确包含、时间分类、倍率和最终状态，并默认重放上述
140 个有限请求采集步骤及接入检查；这不是重做 1000 步长跑。

复算入口：[README](../../experiments/range_batch_device/README.md)。
原始与派生证据：[RESULT](../../artifacts/runs/range_batch_device_20260909T030609Z/RESULT.json)、
[完整计时](../../artifacts/runs/range_batch_device_20260909T030609Z/timing_summary.csv)、
[传输与打包](../../artifacts/runs/range_batch_device_20260909T030609Z/transfer_and_packing_costs.csv)、
[算子合同](OPERATOR_CONTRACT.md)。
"""
    (ROOT/"docs/range_batch_device/REPORT_PLAIN_CHINESE.md").write_text(text)


def main():
    samples,summary=timing_summary()
    assert read(RUN/"raw/timing/COMPLETE.json")["samples"]==150
    tests,commands=account()
    result=decision()
    save(RUN/"RESULT.json",result);save(RUN/"PROJECTED_AMDAHL.json",projected())
    save(RUN/"width_difference_summary.json",width_differences())
    csv_write(RUN/"timing_summary.csv",summary)
    columns=["plant","batch","block","position","mode","seconds","requests","valid_requests","terms","actual_kernel_invocations","peak_allocated_bytes"]
    csv_write(RUN/"timing_samples.csv",[{k:r[k] for k in columns} for r in samples])
    csv_write(RUN/"transfer_and_packing_costs.csv",cost_rows())
    save(RUN/"tests/commands.json",commands);save(RUN/"tests/TEST_ACCOUNTING.json",tests)
    source_root=ROOT.parent/"parent_checks"
    if source_root.exists():
        (RUN/"parent_checks").mkdir(exist_ok=True)
        for path in source_root.iterdir():shutil.copyfile(path,RUN/"parent_checks"/path.name)
    save(RUN/"parent_checks/REFERENCE.json",dict(source_sha=BASE,cwd="/srv/local/shengenli/boundary_execution_20260908T172756Z/package",
        scope="source-locked parent README verifier and six local regression modules",count_in_current_totals=False,
        verifier_argv=["taskset","-c","2","/srv/local/shengenli/miniforge3/envs/py11/bin/python","-m","experiments.boundary_execution.verify",str(PARENT.relative_to(ROOT))],
        tests=["test_boundary_range_plan.py","test_boundary_execution_evidence.py","test_prepared_remainder_replay.py","test_our_reference_endpoint_containment.py","test_endpoint_roundoff_repair.py","test_endpoint_roundoff_carry.py"]))
    report(summary,result,tests)
    code_paths=sorted((ROOT/"experiments/range_batch_device").glob("*.py"))+[ROOT/"experiments/range_batch_device/README.md",ROOT/"docs/range_batch_device/OPERATOR_CONTRACT.md"]
    test_paths=sorted((ROOT/"tests").glob("test_range_*.py"))
    save(RUN/"SOURCE_MAP.json",dict(base_sha=BASE,numerical_sha=NUMERICAL_SHA,numerical_sources=numerical_sources(),
        test_source_sha=head(),new_test_sources={str(p.relative_to(ROOT)):sha(p) for p in test_paths},
        evidence_code_sha=head(),evidence_code_sources={str(p.relative_to(ROOT)):sha(p) for p in code_paths},
        goal_sha256=sha(RUN/"GOAL_FROZEN.md"),report_sha256=sha(ROOT/"docs/range_batch_device/REPORT_PLAIN_CHINESE.md"),
        all_baselines=dict(prepared_remainder_replay=True,packed_boundary_execution=True),
        old_full_runs="REUSED",new_full_horizon_claimed=False,default_switches_changed=False,
        parent_report=dict(path=str((PARENT/"RESULT.json").relative_to(ROOT)),sha256=sha(PARENT/"RESULT.json")),
        frozen_baseline_numeric_sha="5f37cbe0427c480ef0ebbbcaba292143bc8f4ede",
        fresh_timing_source_sha=read(RUN/"raw/timing/environment.json")["source_sha"],
        final_evidence_commit="checkout SHA printed by verifier; no self-referential manifest"))
    write_manifest(RUN)
    print(json.dumps(dict(result=result,tests=tests),indent=2))


if __name__=="__main__":main()
