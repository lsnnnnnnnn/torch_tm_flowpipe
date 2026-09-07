"""Freeze this comparison before executing candidate numerical experiments."""
from pathlib import Path
from fractions import Fraction
from datetime import datetime, timezone
import ast
import hashlib
import json
import math
import subprocess

REPO = Path(__file__).resolve().parents[2]
RUN = REPO.parent
PATHS = json.loads((RUN / 'session_paths.json').read_text())
OUT = REPO / 'artifacts' / 'runs' / ('xiangru_adoption_' + PATHS['run_id'])
DOC = REPO / 'docs' / 'xiangru_adoption'

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def number(text):
    v = float(text)
    return {'decimal': text, 'nearest_binary64_hex': v.hex(),
            'nearest_binary64_fraction': str(Fraction(v)), 'exact_decimal_fraction': str(Fraction(text))}

def box(spec):
    result = []
    for lo, hi in spec:
        lower, upper = float(lo), float(hi)
        if Fraction(lower) > Fraction(lo): lower = math.nextafter(lower, -math.inf)
        if Fraction(upper) < Fraction(hi): upper = math.nextafter(upper, math.inf)
        result.append({'lower': number(lo), 'upper': number(hi),
                       'enclosing_binary64': [lower, upper],
                       'enclosing_binary64_hex': [lower.hex(), upper.hex()]})
    return result

def main():
    if (OUT / 'MATCHED_CONTRACTS.json').exists():
        raise RuntimeError('Pre-registration already exists; do not overwrite it after results.')
    OUT.mkdir(parents=True)
    DOC.mkdir(parents=True, exist_ok=True)
    ref = json.loads((REPO / 'artifacts/runs/c4_reference_performance_batch_20260829/REFERENCE_CONFIG.json').read_text())
    plants = {}
    for name, rhs, initial, order, step, horizon, capacity in [
        ('brusselator', ['1 + x*(x*y - 4)', 'x*(3 - x*y)'],
         [['1.48', '1.52'], ['2.98', '3.02']], 6, '0.02', '20', 1000),
        ('van_der_pol', ['y', 'y - x - x*x*y'],
         [['1.1', '1.4'], ['2.35', '2.45']], 4, '0.01', '10', 100),
    ]:
        # Same full widths, translated centers. These are independent tasks,
        # never subdivisions of the original initial box. Formula fixed now.
        distinct = []
        for i in range(32):
            dx = Fraction(i % 8 - 3, 1000)
            dy = Fraction(i // 8 - 1, 1000)
            distinct.append([[str(float(Fraction(v) + delta)) for v in pair]
                             for pair, delta in zip(initial, [dx, dy])])
        plants[name] = {
            'state_names': ['x', 'y'], 'state_dimension': 2, 'clock_state': False,
            'rhs_expression_strings': rhs,
            'rhs_expression_trees': [ast.dump(ast.parse(s, mode='eval'), include_attributes=False) for s in rhs],
            'initial_decimal_box': initial, 'initial_representation': box(initial),
            'initial_semantics': 'Exact decimal target; each backend encloses it outward in its own scalar format. Candidate uses recorded outward binary64 endpoints.',
            'order': order, 'step': number(step), 'requested_horizon': number(horizon),
            'requested_steps': int(Fraction(horizon) / Fraction(step)),
            'target_remainder_radius': number('1e-4'), 'cutoff': number('1e-10'),
            'validation_eps_ours': number('1e-12'), 'sr_capacity': capacity,
            'initial_partitions': 1, 'endpoint_repair': False, 'endpoint_tightening': False,
            'our_frozen_configuration': ref['plants'][name],
            'candidate_settings': {'mode': 'explicit parity OR strict', 'step_min': -1,
                                   'max_refinement_steps': 490, 'stop_ratio': 0.99,
                                   'validate_intervals': False},
            'different_task_boxes_decimal': distinct,
            'different_task_boxes_binary': [box(b) for b in distinct],
            'checkpoints': [1, 3, 6.32, 10] if name == 'van_der_pol' else [1, 5, 10, 15, 20],
        }
    contract = {
        'schema': 'xiangru_matched_adoption/1', 'preregistered_utc': datetime.now(timezone.utc).isoformat(),
        'goal_source_sha256': hashlib.sha256((RUN.parent / 'codex/goal_vdp_terminal.md').read_bytes()).hexdigest(),
        'plants': plants,
        'vdp_adaptive': {'h_min': number('0.002'), 'h_max': number('0.1'),
                         'horizon': number('10'), 'strategy': 'existing flowstar_compat growth/shrink strategy',
                         'candidate_adaptive_plus_sr': 'unsupported per Settings.__post_init__; verify exception; no replacement implementation'},
        'validation_eps_semantics': {'ours': 'Outward absolute inflation of raw RHS/integrated remainder and subset-check intervals; inspect flowpipe.py _flowstar_raw_remainder_compat_check.',
                                     'candidate': 'No corresponding error-budget field; THRESHOLD_HIGH=1e-12 is reach-loop time bookkeeping only.',
                                     'flowstar': 'No added validation_eps analogue; native directed interval computations.'},
        'modes': ['our_cpu_reference', 'native_flowstar', 'xiangru_parity', 'xiangru_strict'],
        'hardware_protocol': {'gpu_index': 0, 'cpu_threads': 1, 'cpu_affinity': [2],
                              'dtype': 'float64', 'cuda_sync_before_after': True,
                              'warm_cold_export_separate': True, 'skip_nonidle_repeats': True},
        'batch_protocol': {'main': 32, 'auxiliary': [1, 8], 'equivalence': [1, 2, 8],
                           'oom_fallback': 'Only confirmed B32 OOM permits B16, retain failed B32 record.',
                           'short_steps': 20, 'short_repeats': 5,
                           'full_strict_repeats': 3, 'full_flowstar_serial_repeats': 3,
                           'our_full_b1_repeats': 1, 'distinct_boxes': 'Fixed list above, prefixes for smaller B.',
                           'no_answer_caching': True,
                           'continue_full_condition': 'Key strict correctness passes, no confirmed underenclosure; short main-batch complete-solve median beats serial Flowstar median and gives >=5x throughput vs measured our serial short batch.'},
        'width_protocol': {'published': 'Actual endpoint and each segment tube, separately x/y.',
                           'common': 'Read-only export of complete published Taylor model or legal factorized map; same outward normal-domain interval evaluator; never align unrelated internal coordinates.',
                           'reuse': 'our brusselator_canonical_exchange and candidate io.oracle/adapters inspected first; metrics on uncomposed tmvPre alone are insufficient.',
                           'alignment': 'Match actual local-time domains; fixed-step exact accumulated binary64 step as mathematical schedule. Reach-loop epsilon recorded separately. No interpolated enclosure.',
                           'statistics': ['time_weighted_median', 'time_weighted_p95', 'max_and_time', 'checkpoints', 'bound_shifts', 'center_shifts', 'disjoint', 'missing_intervals'],
                           'zero_denominator_threshold': 1e-12, 'zero_denominator_policy': 'absolute difference, no infinite ratio',
                           'missing_export': 'Report published bounds and timing only; common width gate remains unverified.'},
        'adoption_thresholds': {'correctness': 'Strict 2D Brusselator T20, strict fixed VDP >=6.32, exact/local/history/rollback/isolation pass; actual CPU/CUDA source identifiable.',
                                'width_vs_ours_each_step_each_of_four': {'multiplier': 1.10, 'absolute_tolerance': 1e-12},
                                'near_flowstar_each_of_four': {'time_weighted_p95_max': 1.10, 'maximum_ratio_max': 1.25},
                                'strict_main_batch_speedup_vs_ours_min': 5,
                                'strict_main_batch_full_time_ratio_vs_flowstar_max': 1.0},
        'stop_rules': ['Confirmed underenclosure stops affected long runs and speed promotion; unaffected comparisons continue.',
                       'No numerical patch before first baseline report; at most one localized numerical mechanism.',
                       'No adaptive+SR implementation, order/step/budget tuning, or extra systems.',
                       'Optional adapter only if correctness, width and main-batch speed gates pass. Default unchanged.'],
    }
    write(OUT / 'MATCHED_CONTRACTS.json', contract)
    write(OUT / 'session_paths.json', PATHS)
    inventory = json.loads((RUN / 'old_clones_inventory.json').read_text())
    write(OUT / 'raw_minimal/old_clones_inventory.json', inventory)
    source = {'candidate': json.loads((RUN / 'xiangru_source.json').read_text()),
              'ours_evidence_base': 'df50c55ef640b0ca9d90b132c88235b8ec07510b',
              'ours_numerical_reference': '4939fb288c941a67f55cc191f4d75f8594692f47',
              'ours_cpu_batch_wrapper': '7608dd52e48af3ce8ae2e0a8343aae125c63b7f4',
              'flowstar': 'b85a3211748cb77b736fe4ad42ee02d8d2b81148',
              'ours_remote_main_observed': 'b2f34f5b2077e34662a2559d8c09b1d264bd7d98',
              'ours_remote_main_used_as_reference': False,
              'candidate_attribution': {'upstream': 'huanzhang12/flowstar-gpu',
                  'import_sha': '7a8002df4d45297c915c67f75247a0b88a8e0b03',
                  'declared_license': 'GPL-3.0-only', 'author': 'Huan Zhang'},
              'third_party_source_publication': 'No third-party source or source diffs in this evidence branch.'}
    write(OUT / 'source_manifest.json', source)
    lines = ['# 旧副本盘点', '', '只读检查，没有切分支、拉取、清理、安装或重编译旧目录。进程检查仅检查本用户可读的 cwd 和进程名；不读取完整参数。', '',
             '| 角色 | 绝对路径 | HEAD | 分支 | 工作树 | 本用户 cwd 任务 |', '|---|---|---|---|---|---|']
    for r in inventory['repositories']:
        state = r['status']['stdout'].splitlines()[1:]
        lines.append(f"| {r['role']} | `{r['path']}` | `{r['head']['stdout']}` | {r['branch']['stdout'] or 'detached'} | {'有修改/未跟踪内容' if state else '干净'} | {len(r['active_own_cwd_processes'])} |")
    lines += ['', '完整的已修改文件名、脱敏 remote、worktree 清单和旧证据位置保存在本轮 raw_minimal/old_clones_inventory.json。',
              '主要旧 Xiangru 为 84184de6；新版新增仓库内 GPU 引擎、相关测试和来源许可记录。旧 Xiangru 工作树干净，没有必须移植的未提交修补。原生 Flow* 和用户主工作树有本地改动，均原样保留。']
    (DOC / 'OLD_CLONES_INVENTORY.md').write_text('\n'.join(lines) + '\n')
    (DOC / 'MATCHED_CONTRACTS.md').write_text('# 预注册比较合同\n\n本轮权威合同为 `artifacts/runs/xiangru_adoption_' + PATHS['run_id'] + '/MATCHED_CONTRACTS.json`。在任何本轮数值运行前提交。\n\nBrusselator：真实两状态，六阶、0.02、1000 步到 T20、历史容量 1000。Van der Pol：真实两状态，四阶、0.01 连续请求 T10，读取 T1/T3/T6.32；自适应另外列出。保留原有范围预算及接受后误差收紧，禁用额外终点修补。\n\n初始值按精确十进制定义，完整列出最近 binary64 与向外包住十进制的 binary64；不同任务使用预先固定的平移盒子，同样宽度，不切分初始集。B32 主测试，只有确实 OOM 才降 B16。严格正确性失败时停止受影响长跑。\n\n我们的 validation_eps 是误差范围的绝对向外扩张；候选时间偏移 1e-12 不等价。候选自带指标直接测 tmvPre，不能代替已组合状态的共同测量。共同测量缺失时明确留空，不作宽度通过声明。\n\n预热后完整求解、进程启动、首次编译、导出各自计时；CPU 单线程固定亲和性，GPU 用设备 0 并同步。所有接纳阈值完整保存在 JSON，不根据结果调整。\n')
    print(OUT)

if __name__ == '__main__': main()
