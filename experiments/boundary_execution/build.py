"""Package only the frozen boundary experiment and the references it uses."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import torch
from experiments.boundary_execution.analyze import (
    ROOT, SCIENTIFIC_SHA, PARENT_SHA, PARENT_RUNTIME_SHA, WINDOWS, read_json,
    decision, runtime_rows, profile_cost_rows, remaining_cost_rows, full_width_rows,
)
from experiments.boundary_execution.verify import (
    git, sha, source_files, check_local, check_states, implementation_decision,
)

RUN_NAME = 'boundary_execution_20260908T172756Z'
PARENT_BUNDLE = 'artifacts/runs/repaired_solver_performance_20260908T034636Z'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    with Path(path).open('w') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_manifest(root):
    files = sorted(p for p in Path(root).rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (Path(root)/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(root)}\n' for p in files))


def test_accounting(root):
    groups, unique = {}, {}
    for command in read_json(Path(root)/'tests/commands.json'):
        assert command['exit_code'] == 0
        totals = Counter()
        for case in ET.parse(Path(root)/command['xml']).iter('testcase'):
            assert case.find('failure') is None and case.find('error') is None
            status = 'skipped' if case.find('skipped') is not None else 'passed'
            totals[status] += 1
            if command.get('count_in_final_matrix'):
                identity = case.get('classname'), case.get('name')
                assert identity not in unique
                unique[identity] = status
        groups[command['name']] = dict(totals)
    result = {'unique_totals': dict(Counter(unique.values())), 'groups': groups,
              'duplicate_candidate_and_independent_checks_added': False}
    write_json(Path(root)/'tests/TEST_ACCOUNTING.json', result)
    return result


def source_map(workspace, package_code_sha):
    expected = source_files(SCIENTIFIC_SHA)
    assert all(sha(ROOT/p) == digest for p, digest in expected.items())
    package_paths = git('ls-tree', '-r', '--name-only', package_code_sha,
                        'experiments/boundary_execution', 'docs/boundary_execution',
                        'tests/test_boundary_range_plan.py', 'tests/test_boundary_execution_evidence.py').decode().splitlines()
    package_paths = [p for p in package_paths if p != 'docs/boundary_execution/REPORT_PLAIN_CHINESE.md']
    reused = [f'{PARENT_BUNDLE}/RESULT.json', f'{PARENT_BUNDLE}/remaining_hotspots.csv',
              'artifacts/runs/endpoint_roundoff_repair_20260908/raw_minimal/historical_fixture_recovery.json',
              'experiments/endpoint_roundoff_repair/frozen.py', 'experiments/xiangru_adoption/common.py',
              'experiments/repaired_solver_performance/compare.py',
              'experiments/repaired_solver_performance/verify.py',
              'experiments/repaired_solver_performance/README.md']
    for run in ['brusselator_full_optimized', 'vdp_full_optimized', 'vdp_adaptive_optimized']:
        reused += [f'{PARENT_BUNDLE}/raw_minimal/formal/{run}/{name}'
                   for name in ['summary.json', 'models.jsonl.gz', 'bounds.csv', 'endpoint_audit.jsonl']]
    for checkpoint in ['formal/brusselator_full_reference/checkpoint_0980', 'checkpoint_0090']:
        reused += [f'{PARENT_BUNDLE}/raw_minimal/{checkpoint}/{name}'
                   for name in ['terminal_state.json', 'terminal_state_manifest.json']]
    first = read_json(workspace/'formal/brusselator_early_pair1_baseline/source.json')
    return dict(scientific_sha=SCIENTIFIC_SHA, parent_sha=PARENT_SHA, parent_runtime_sha=PARENT_RUNTIME_SHA,
        package_code_sha=package_code_sha, scientific_sources=expected,
        package_sources={p: hashlib.sha256(git('show', f'{package_code_sha}:{p}')).hexdigest() for p in package_paths},
        reused_parent_files={p: hashlib.sha256(git('show', f'{PARENT_SHA}:{p}')).hexdigest() for p in reused},
        prepared_replay_in_both_modes=True, boundary_default_enabled=False,
        modes={'baseline': {'prepared_remainder_replay': True, 'packed_boundary_execution': False},
               'candidate': {'prepared_remainder_replay': True, 'packed_boundary_execution': True}},
        production_source_root=first['source_root'], production_imported_package=first['imported_package'],
        python=first['python'], python_version=first['python_version'], torch=first['torch_version'],
        dtype=first['dtype'], device=first['device'], affinity=first['affinity'], threads=first['threads'],
        branch='codex/packed-boundary-execution-20260908T172756Z', root=str(workspace),
        goal_sha256=sha(workspace/'GOAL_FROZEN.md'),
        source_identity_note='Scientific SHA fixes numerical code and runner; package_code_sha fixes analysis and verifier. The final delivery commit is reported by git/verification, never embedded self-referentially.',
        existing_parent_evidence='Referenced in place with parent commit hashes; no wholesale repackaging or modified old verifier.',
        new_flowstar_timing=False, whole_solver_formal_proof_claimed=False)


def report(root, result, costs, remaining, accounting):
    by_window = defaultdict(lambda: defaultdict(float))
    for row in costs:
        by_window[row['window']][row['category']] += float(row['seconds'])
    table = ['| 窗口 | 总秒数 | A 转换 | B 点代入 | C 独立误差/账本 | D 取范围 | E 组合/cutoff | F 映射 | G 历史 | H 其他 | 边界外 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for window in WINDOWS:
        v = by_window[window]; total = sum(v.values())
        table.append('| '+window+' | '+f'{total:.3f}'+' | '+' | '.join(f'{100*v[c]/total:.2f}%' for c in [*'ABCDEFGH', 'outside_boundary'])+' |')
    full_table = ['| 系统/完整配对 | baseline 秒 | candidate 秒 | 倍率 | 峰值 RSS 比 |', '|---|---:|---:|---:|---:|']
    for plant, pairs in result['full_pairs'].items():
        for i, pair in enumerate(pairs, 1):
            full_table.append(f"| {plant} / {i} | {pair['baseline_solve_seconds']:.6f} | {pair['candidate_solve_seconds']:.6f} | {pair['speedup']:.6f}× | {pair['peak_rss_ratio']:.4f} |")
    short_table = ['| 窗口/前缀 | 三对倍率（按执行顺序） |', '|---|---|']
    for name, values in result['short_speedups'].items():
        short_table.append('| '+name+' | '+'、'.join(f'{v:.6f}×' for v in values)+' |')
    remaining_table = ['| 候选真实步骤 / 进入时历史长度 | 观察整步秒数 | D 取范围 | G 历史 | 边界外 |', '|---|---:|---:|---:|---:|']
    for name in ['brusselator_before0995', 'brusselator_before1001', 'van_der_pol_before0100']:
        rows = [r for r in remaining if r['case'] == name]
        values = {r['category']: r for r in rows}
        remaining_table.append(f"| {name} / {rows[0]['history_length']} | {float(rows[0]['step_seconds']):.3f} | "+
            ' | '.join(f"{100*values[c]['fraction']:.2f}%" for c in ['D', 'G', 'outside_boundary'])+' |')
    artifact = '../../artifacts/runs/'+RUN_NAME
    totals = accounting['unique_totals']
    conclusion = {
        'BOUNDARY_EXECUTION_PRESERVED__TARGET_SPEEDUP_OBSERVED': '完整配对观察到达标提速；结论范围限于保存的实际运行。',
        'BOUNDARY_EXECUTION_PRESERVED__USEFUL_BELOW_TARGET': '完整配对观察到有用提速，未达到 1.5× 目标；保留默认关闭的可选路径。',
        'BOUNDARY_EXECUTION_PRESERVED__NO_MATERIAL_SPEEDUP': '数值行为保持，但完整测量未支持有实质收益的结论。',
    }[result['status']]
    text = f'''# 本步结果交给下一步：有序张量取范围

{conclusion}

正式状态：`{result['status']}`。新优化默认关闭，通过独立开关选择。本轮只改取范围这一条执行链。

1. **省掉了什么工作？** 原来每个多项式项都要反复创建小区间对象、复制系数，并重新算相同变量的幂。现在一次把独立项放进张量，在这一次调用内复用变量幂；最后仍按原来的先后次序累加各项。结构计划只保存“哪些项需要哪次运算”，不保存上一步的数值结果。真实 Brusselator step20 的同输入原型里，Interval 创建从 17410 降到 3320，显式复制从 69640 降到 13796。三对局部倍率约 2.35×，对应窗口预测约 1.42×；这些不是完整求解倍率。

2. **原来 63%–69% 的大分类拆成了什么？** 下表是 baseline 代表窗口的互斥计时，分母为整个步骤；各类及边界外相加为总时间。真正大的部分是 D 取范围，以及长历史时增长的 G。C 独立端点误差与账本约占整步 1%–2%，不能把整个边界大分类当成新增端点保护的成本。

{chr(10).join(table)}

所有计时器核对了实际调用绑定，包括 flowpipe 保存的 from-import 别名。Brusselator step995 的输入历史长度是 994；step1001 是 0，分别记录。VDP reset 前后也分开观察。Interval 次数与张量计数来自相同状态的独立计数遍历，不进入正式计时分母；张量数按 ATen 返回对象计数并扣除原地运算返回自身，包含 view，不等于独立存储分配数或字节数。显式 clone/copy 另列，未重复叠加到互斥时间。

3. **为什么没有省掉误差和安全判断？** 每个变量幂继续走原来的标量 Interval.pow_int，独立项乘法保留四个候选乘积、min/max、nextafter 和区间有效性检查。跨项加法仍逐项按原支持顺序执行，每次都向外舍入。normal 的时间因子、状态因子及其他变量顺序保持。原端点 E、cutoff 支付、普通余项、历史传播和收紧循环没有改。仅不可变支持结构进入有界缓存；系数、domain、h、余项和变量幂都不会跨调用缓存。输出使用私有存储。

41 项新局部测试覆盖 order4/6、B1/B2、不同输出、非对称范围、带符号/相消/空项、subnormal、溢出、NaN/Inf、回退及所有权；每项和总和另用精确有理数验证。六组真实完整状态序列覆盖 14 个位置，含 VDP99/100/101、Brusselator999/1000/1001。两种模式都检查已有历史的失败回滚、重复测量及 checkpoint 恢复后实际继续一步。

4. **全程范围、时域和自适应行为有变化吗？** 两系统各 1000 步的完整 endpoint/tube 多项式、余项、端点 E、中心/尺度、normal/history 摘要与求解决策逐位一致；checkpoint 的完整状态由安全加载器复核，代表位置再直接对照完整边界对象。published/common 两种 observer 的 endpoint x/y 与 tube x/y 共 16000 行比较，全部上下界逐位一致；verifier 从系数重新计算两种 observer。baseline 与父版本已经开启 prepared replay 的保存数值对象相同，原已发布的 Flow* 宽度关系因此保留。

固定时域仍为实际 binary64 h 的精确和，不裁剪末步。VDP 原自适应是 {result['adaptive_accepted']} 接受 / {result['adaptive_rejected']} 拒绝；实际 h 和为 `{result['adaptive_exact_h_sum']}`，调度时钟为 `{result['adaptive_scheduler_time_hex']}`，与父档案的序列一致。本轮没有重新计时 Flow*，也没有声称整个求解器已形式化证明。

5. **baseline 是否已含上一轮优化？** 是。每份 summary/source/执行合同都记录 `prepared_remainder_replay=True`；baseline 的 `packed_boundary_execution=False`，candidate 为 True。两者在同一科学提交 `{SCIENTIFIC_SHA}`、CPU binary64、单线程、CPU2 affinity 上运行。比较采用本轮新 baseline 的实际计时，未使用旧约 1992 秒作分母，prepared replay 的公开默认值和准入合同未改。

6. **完整时间、短窗口重复结果和目标怎样？** 正式 solve 时间包括计划构造、必要输入转换、安全检查、拒绝尝试及实际求解；初始化、外部导出和整个进程时间分别保存。

{chr(10).join(full_table)}

{chr(10).join(short_table)}

所有配对均保留。每系统完整配对次数及置信范围见 RESULT；只有一对时只能称单次完整配对，三对短实验支持方向，不能据此声称稳定的完整倍率。正式运行前已将接近 1.5× 定义为 [1.35,1.65]；若接近目标或短/长方向冲突，自动补反向完整配对，原先较窄的临时区间也原样保留。VDP 未出现超过 10% 的稳定减速判定为 `{result['vdp_no_stable_slowdown_over_10_percent']}`，完整 RSS 比均不超过 1.5 判定为 `{result['peak_rss_within_1p5']}`。最终按实际数据给出上述状态，不从局部倍率外推成功。

7. **剩余瓶颈是什么？** 正式长跑结束后，又在同一冻结数值源码的三个真实输入上各观察一次候选步骤，并单独计数：

{chr(10).join(remaining_table)}

这些是带观察器的代表步骤，不是全程精确份额。D 内独立项乘法已经张量化，变量幂仍调用标量区间路线，跨项累加仍是有序 Python 循环；稀疏输入打包和结果区间包装也仍存在。G 仍包含 Python 历史验证、pack/unpack 和已有张量传播。主积分与组合部分也仍混有 Python 稀疏项处理及既有稠密张量计算。此次未叠加历史重写或 prepared replay 第二项优化。

8. **是否给批量/GPU 提供了可用接口？** 局部 RangePlan 接口接受系数 `[B, output, terms]` 和 domain `[B, variables]`，独立任务确实共用张量乘法，B2 用不同输入检查无混合。一个计划共享同一支持结构；完整批量接入还需按支持结构分组，处理各任务的步长、接受进度和历史状态，不能用补零重排改变运算链。完整 solver 仍为 B1；接口目前限 CPU64。标量变量幂和逐项累加保留了经过检查的舍入语义，迁移设备前还需处理这些同步与执行成本，并在目标设备独立检查逐操作包含性。CPU 不必先追平 Flow* 才考虑 GPU；本轮没有 GPU 吞吐结论。

根测试、候选开启复查及本轮证据篡改测试按身份去重为 {totals.get('passed', 0)} passed / {totals.get('skipped', 0)} skipped；候选开启的 99 项和独立副本复查均不重复加总。父版本的 75 项及旧 verifier 结果单列复用。相关命令、JUnit、退出码和排除范围均可在证据包检查；新 verifier 只复核本轮变化及引用的父数值锚点，不重跑完整 ODE。

两项既有跳过分别是未配置 FLOWSTAR_ROOT 的可选后端身份测试、未配置 DIFFREACH_PYTHON 的可选环境导入测试；本轮没有增加预期失败或跳过新路径断言。

下一步只做一项：用真实独立输入测量 RangePlan 的局部批量扩展，包含打包和检查，确认收益能覆盖剩余成本后再决定设备迁移。

原始与派生证据：[RESULT]({artifact}/RESULT.json)、[成本分解]({artifact}/boundary_cost_breakdown.csv)、[完整范围]({artifact}/full_width_equivalence.csv)、[运行时间]({artifact}/runtime_pairs.csv)、[复核说明](../../experiments/boundary_execution/README.md)。
'''
    path = ROOT/'docs/boundary_execution/REPORT_PLAIN_CHINESE.md'
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
    return sha(path)


def refresh(root):
    root = Path(root)
    accounting = test_accounting(root)
    result = decision(root/'raw_minimal/formal')
    costs = profile_cost_rows(root/'raw_minimal/profiles')
    remaining = remaining_cost_rows(root/'raw_minimal')
    report_sha = report(root, result, costs, remaining, accounting)
    source = read_json(root/'SOURCE_MAP.json'); source['generated_report_sha256'] = report_sha
    write_json(root/'SOURCE_MAP.json', source)
    write_manifest(root)


def build(workspace, package_code_sha, output):
    workspace, output = Path(workspace), Path(output)
    assert read_json(workspace/'formal/SCHEDULE_COMPLETED.json')['source_sha'] == SCIENTIFIC_SHA
    assert not git('diff', package_code_sha, '--', 'experiments/boundary_execution',
                   'tests/test_boundary_range_plan.py', 'tests/test_boundary_execution_evidence.py').strip()
    assert (workspace/'remaining_profiles/COMPLETE.json').is_file()
    output.mkdir(parents=True, exist_ok=False); raw = output/'raw_minimal'; raw.mkdir()
    for name in ['profiles', 'prototypes', 'input_checkpoints', 'state_inputs', 'state_checks', 'formal', 'parent_checks', 'remaining_profiles']:
        shutil.copytree(workspace/name, raw/name)
    for name in ['GOAL_FROZEN.md', 'tensor_count_semantics.json', 'formal_repetition_policy.json']:
        shutil.copy2(workspace/name, raw/name)
    shutil.copy2(workspace/'implementation_decision.json', raw/'implementation_decision_preproduction.json')
    shutil.copytree(workspace/'tests', output/'tests')
    write_json(output/'SOURCE_MAP.json', source_map(workspace, package_code_sha))
    write_csv(output/'boundary_cost_breakdown.csv', profile_cost_rows(raw/'profiles'))
    write_csv(output/'remaining_hotspots.csv', remaining_cost_rows(raw))
    write_csv(output/'full_width_equivalence.csv', full_width_rows(raw/'formal'))
    write_csv(output/'runtime_pairs.csv', runtime_rows(raw/'formal'))
    write_json(output/'RESULT.json', decision(raw/'formal'))
    write_json(output/'implementation_decision.json', implementation_decision(raw))
    # Build recomputes local rational checks; the independent verifier also
    # replays actual boundary steps and allocation counters from safe inputs.
    write_json(output/'local_and_state_equivalence.json', {'local': check_local(raw, replay=False), 'state': check_states(raw, replay=False)})
    refresh(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--package-code-sha')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/runs'/RUN_NAME)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.refresh:
        refresh(args.output)
    else:
        assert args.workspace and args.package_code_sha
        print(build(args.workspace, args.package_code_sha, args.output))
