"""Recompute this change's numerical, lifetime and timing evidence."""
import argparse
from collections import Counter
from datetime import datetime
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import torch
import torch_tm_flowpipe as core
from torch_tm_flowpipe.prepared_remainder_replay import prepared_remainder_replay
from torch_tm_flowpipe.packed_boundary_range import packed_boundary_execution, is_enabled
from experiments.endpoint_roundoff_repair.frozen import ROOT, step
from experiments.boundary_execution.analyze import (
    SCIENTIFIC_SHA, PARENT_SHA, PARENT_RUNTIME_SHA, WINDOWS, read_json, read_csv,
    verify_run, full_width_rows, runtime_rows, profile_cost_rows, remaining_cost_rows, decision,
)
from experiments.boundary_execution.compare import compare_pair, compare_parent_archive
from experiments.boundary_execution.oracles import verify_range_file
from experiments.boundary_execution.profile import state_identity, Attribution, TensorCounts
from experiments.boundary_execution.prototype import capture_ranges
from experiments.boundary_execution.prototype_range import make_plan as prototype_plan
from experiments.boundary_execution.state_equivalence import verify_sequence


REQUIRED = ('SOURCE_MAP.json', 'boundary_cost_breakdown.csv', 'implementation_decision.json',
    'local_and_state_equivalence.json', 'full_width_equivalence.csv', 'runtime_pairs.csv',
    'RESULT.json', 'tests/commands.json', 'SHA256SUMS')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(root):
    declared = {}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts and name not in declared
        path = root/relative
        assert path.is_file() and not path.is_symlink() and sha(path) == digest, f'manifest: {name}'
        declared[name] = digest
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'}
    assert set(declared) == actual, 'manifest has missing or extra files'
    return len(declared)


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def source_files(commit):
    paths = git('ls-tree', '-r', '--name-only', commit, 'src/torch_tm_flowpipe').decode().splitlines()
    return {p: hashlib.sha256(git('show', f'{commit}:{p}')).hexdigest()
            for p in paths if p.endswith('.py') and len(Path(p).parts) == 3}


def check_source(root):
    source = read_json(root/'SOURCE_MAP.json')
    assert source['scientific_sha'] == SCIENTIFIC_SHA
    assert source['parent_sha'] == PARENT_SHA and source['parent_runtime_sha'] == PARENT_RUNTIME_SHA
    assert source['prepared_replay_in_both_modes'] is True and source['boundary_default_enabled'] is False
    assert source['whole_solver_formal_proof_claimed'] is False and source['new_flowstar_timing'] is False
    expected = source_files(SCIENTIFIC_SHA)
    assert source['scientific_sources'] == expected
    assert Path(core.__file__).resolve() == (ROOT/'src/torch_tm_flowpipe/__init__.py').resolve()
    assert all(sha(ROOT/p) == digest for p, digest in expected.items()), 'imported numerical source differs'
    changed = set(git('diff', '--name-only', PARENT_SHA, SCIENTIFIC_SHA, '--', 'src/torch_tm_flowpipe').decode().splitlines())
    assert changed == {'src/torch_tm_flowpipe/__init__.py', 'src/torch_tm_flowpipe/polynomial.py',
                      'src/torch_tm_flowpipe/accepted_boundary_sr.py', 'src/torch_tm_flowpipe/packed_boundary_range.py'}
    assert not git('diff', PARENT_SHA, SCIENTIFIC_SHA, '--', 'src/torch_tm_flowpipe/prepared_remainder_replay.py').strip()
    assert sha(root/'raw_minimal/GOAL_FROZEN.md') == source['goal_sha256']
    package = source['package_code_sha']
    assert subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', package, 'HEAD']).returncode == 0
    package_paths = git('ls-tree', '-r', '--name-only', package, 'experiments/boundary_execution',
        'docs/boundary_execution', 'tests/test_boundary_range_plan.py', 'tests/test_boundary_execution_evidence.py').decode().splitlines()
    package_paths = set(package_paths) - {'docs/boundary_execution/REPORT_PLAIN_CHINESE.md'}
    assert set(source['package_sources']) == package_paths
    assert 'experiments/boundary_execution/verify.py' in package_paths
    for path, digest in source['package_sources'].items():
        assert hashlib.sha256(git('show', f'{package}:{path}')).hexdigest() == digest
        assert sha(ROOT/path) == digest
    for path, digest in source['reused_parent_files'].items():
        assert hashlib.sha256(git('show', f'{PARENT_SHA}:{path}')).hexdigest() == digest
        assert sha(ROOT/path) == digest
    assert not is_enabled(), 'public boundary mode must default off'
    assert source['generated_report_sha256'] == sha(ROOT/'docs/boundary_execution/REPORT_PLAIN_CHINESE.md')
    assert source['modes'] == {'baseline': {'prepared_remainder_replay': True, 'packed_boundary_execution': False},
                               'candidate': {'prepared_remainder_replay': True, 'packed_boundary_execution': True}}
    return source


def same_csv(path, rows):
    assert read_csv(path) == [{k: str(v) for k, v in row.items()} for row in rows], f'derived CSV: {path}'


def check_tests(root):
    commands = read_json(root/'tests/commands.json')
    unique, groups = {}, {}
    for command in commands:
        assert command['exit_code'] == 0
        assert int((root/command['exit']).read_text()) == 0
        assert (root/command['log']).is_file()
        assert not git('diff', SCIENTIFIC_SHA, command['source_sha'], '--', 'src/torch_tm_flowpipe').strip()
        totals = Counter()
        for case in ET.parse(root/command['xml']).iter('testcase'):
            status = ('failed' if case.find('failure') is not None else 'error' if case.find('error') is not None
                      else 'skipped' if case.find('skipped') is not None else 'passed')
            assert status not in {'failed', 'error'}
            totals[status] += 1
            if command.get('count_in_final_matrix'):
                identity = case.get('classname'), case.get('name')
                assert identity not in unique, 'test identity counted twice'
                unique[identity] = status
        groups[command['name']] = dict(totals)
    assert {'root', 'candidate_endpoint_and_boundary', 'boundary_evidence'} <= set(groups)
    assert groups['boundary_evidence'] == {'passed': 6}
    candidate = next(c for c in commands if c['name'] == 'candidate_endpoint_and_boundary')
    assert 'experiments.boundary_execution.enable_candidate' in candidate['argv']
    fixtures = read_json(root/'tests/historical_fixture_recovery.json')
    assert sha(ROOT/fixtures['reused_record']) == fixtures['reused_record_sha256']
    assert len(fixtures['files']) == 25 and fixtures['all_hashes_match'] is True
    result = {'unique_totals': dict(Counter(unique.values())), 'groups': groups,
              'duplicate_candidate_and_independent_checks_added': False}
    assert result == read_json(root/'tests/TEST_ACCOUNTING.json')
    return result


def check_profiles(raw):
    parent = source_files(PARENT_SHA)
    for window in [*WINDOWS, 'brusselator_after_reset']:
        for suffix in ['wall', 'counts']:
            path = raw/'profiles'/f'{window}_{suffix}'
            source = read_json(path/'source.json')
            assert source['source_hashes'] == parent
            assert source['prepared_remainder_replay'] is True and source['boundary_execution'] is False
            assert source['affinity'] == [2] and source['threads'] == 1
            binding = next(b for b in source['bindings'] if b['operation'].endswith('accepted_boundary_sr.prepare_accepted_boundary_sr'))
            assert 'torch_tm_flowpipe.flowpipe.prepare_accepted_boundary_sr' in binding['bindings']
            operations = read_csv(path/'operations.csv')
            for name in ('prepare_accepted_boundary_sr', 'commit_accepted_boundary_sr'):
                calls = sum(int(r['calls']) for r in operations if r['operation'] == f'torch_tm_flowpipe.accepted_boundary_sr.{name}')
                assert calls == (1 if window == 'brusselator_after_reset' else 20), 'actual binding missed'
    late = read_json(raw/'profiles/brusselator_late_wall/cprofile_state.json')
    reset = read_json(raw/'profiles/brusselator_after_reset_wall/cprofile_state.json')
    assert late['step'] == 994 and late['history_length'] == 994
    assert reset['step'] == 1000 and reset['history_length'] == 0


def check_checkpoint_inputs(raw):
    """Tie safe complete states to original profile states and parent objects."""
    parent = ROOT/'artifacts/runs/repaired_solver_performance_20260908T034636Z/raw_minimal/formal'
    parent_audits = {plant: {r['step']: r for r in
        (json.loads(line) for line in (parent/name/'endpoint_audit.jsonl').read_text().splitlines())}
        for plant, name in [('brusselator', 'brusselator_full_optimized'), ('van_der_pol', 'vdp_full_optimized')]}
    sources = read_json(raw/'input_checkpoints/SOURCES.json')
    for name, saved in sources.items():
        checkpoint = raw/'input_checkpoints'/name
        assert saved['sha256'] == {p.name: sha(p) for p in checkpoint.iterdir() if p.is_file()}
        loaded = core.load_terminal_checkpoint(checkpoint)
        identity = state_identity(loaded.current, loaded.normal_state)
        audit = parent_audits[loaded.contract['plant']][identity['step']]
        for key in ('current', 'normal_pre', 'normal_right'):
            assert identity[key] == audit['state_hashes'][key]
        assert identity['queue'] == audit['queue_sha256']
        assert loaded.scheduler['time_exact'] == audit['t_end_exact']
    safe_sources = read_json(raw/'state_inputs/SOURCE.json')
    assert len(safe_sources) == 12
    for record in safe_sources:
        checkpoint = raw/'state_inputs'/Path(record['path']).name
        assert record['manifest'] == read_json(checkpoint/'terminal_state_manifest.json')
        loaded = core.load_terminal_checkpoint(checkpoint)
        identity = state_identity(loaded.current, loaded.normal_state)
        payload = read_json(checkpoint/'terminal_state.json')
        provenance = payload['provenance']
        assert provenance['numeric_source'] == PARENT_SHA
        assert provenance['prepared_remainder_replay'] is True and provenance['packed_boundary_execution'] is False
        profile = raw/'profiles'/Path(provenance['capture_profile']).name
        assert sha(profile/Path(record['profile_input']).name) == provenance['private_input_sha256']
        saved = next(r['before'] for r in read_json(profile/'states.json') if r['before']['step'] == identity['step'])
        assert identity == saved, 'checkpoint must be the complete captured pre-step state'
        assert loaded.scheduler['time_exact'] == str(identity['step'] * Fraction(.02 if loaded.contract['plant'] == 'brusselator' else .01))


def check_run_checkpoints(path):
    path = Path(path)
    summary = read_json(path/'summary.json')
    audit = {r['step']: r for r in (json.loads(line) for line in (path/'endpoint_audit.jsonl').read_text().splitlines())}
    bounds = {int(r['step']): r for r in read_csv(path/'bounds.csv')}
    checkpoints = sorted(path.glob('checkpoint_[0-9]*'))
    assert checkpoints
    for checkpoint in checkpoints:
        loaded = core.load_terminal_checkpoint(checkpoint)
        identity = state_identity(loaded.current, loaded.normal_state)
        index = identity['step']; entry = audit[index]
        assert checkpoint.name == f'checkpoint_{index:04d}'
        assert loaded.contract == summary['config']
        assert read_json(checkpoint/'terminal_state.json')['provenance'] == read_json(path/'source.json')
        for key in ('current', 'normal_pre', 'normal_right'):
            assert identity[key] == entry['state_hashes'][key]
        assert identity['queue'] == entry['queue_sha256'] and identity['history_length'] == entry['queue_size']
        assert identity['reset_count'] == entry['queue_reset_count']
        assert loaded.scheduler['time_exact'] == entry['t_end_exact']
        assert loaded.scheduler['accepted_steps'] == index
        assert loaded.scheduler['scheduler_time_hex'] == bounds[index]['scheduler_time_hex']
        assert [v.hex() for v in loaded.normal_state.center] == entry['center_hex']
        assert [v.hex() for v in loaded.normal_state.scales] == entry['scales_hex']
    return len(checkpoints)


def implementation_decision(raw):
    """Keep the dated pre-implementation decision and its pre-run amendment."""
    initial = read_json(raw/'implementation_decision_preproduction.json')
    policy = read_json(raw/'formal_repetition_policy.json')
    assert initial['parent_sha'] == PARENT_SHA
    assert initial['mechanism'] == 'ordered_packed_interval_polynomial_range' and initial['selected_category'] == 'D'
    assert initial['production_modified_at_decision'] is False and initial['admitted_for_opt_in_implementation'] is True
    assert initial['prepared_remainder_replay_in_both_modes'] is True and initial['default_boundary_enabled'] is False
    assert initial['prototype_source_sha256'] == sha(ROOT/'experiments/boundary_execution/prototype_range.py')
    reports = {p.parent.name: read_json(p) for p in sorted((raw/'prototypes').glob('*/result.json'))}
    assert initial['same_input_microexperiments'] == reports
    assert initial['alternatives_attempted'] == 0
    assert initial['formal_additional_pair_rule']['brusselator_target_near_interval'] == [1.45, 1.55]
    assert policy['before_formal_runs'] is True and policy['near_target_interval'] == [1.35, 1.65]
    assert policy['reference_target'] == 1.5 and policy['relative_margin'] == .1
    science_time = int(git('show', '-s', '--format=%ct', SCIENTIFIC_SHA))
    assert datetime.fromisoformat(initial['recorded_before_production_modification_utc']).timestamp() < science_time
    first_run = min(c['wall_started_utc'] for c in read_json(raw/'formal/commands.json'))
    assert datetime.fromisoformat(policy['recorded_utc']).timestamp() < first_run
    return {'preproduction_decision': initial, 'formal_repetition_policy': policy,
            'policy_note': 'The later pre-run policy supersedes only the provisional repetition interval; both originals are preserved.'}


def check_parent_checks(raw):
    for command in read_json(raw/'parent_checks/commands.json'):
        assert command['source_sha'] == PARENT_SHA and command['exit_code'] == 0
        assert int((raw/'parent_checks'/command['exit']).read_text()) == 0
        assert command['count_in_current_test_total'] is False
        if command['name'] == 'parent_verifier':
            assert 'experiments.repaired_solver_performance.verify' in command['argv']
            result = read_json(raw/'parent_checks'/command['log'])
            assert result['passed'] is True and result['tests'] == {'passed': 1108, 'skipped': 11}
        else:
            cases = list(ET.parse(raw/'parent_checks'/command['xml']).iter('testcase'))
            assert len(cases) == 75 and all(c.find('failure') is None and c.find('error') is None for c in cases)


def check_local(raw, *, replay=True):
    reports = []
    for path in sorted((raw/'prototypes').glob('*/result.json')):
        report = read_json(path)
        checkpoint = raw/'state_inputs'/f"{report['plant']}_before{report['step']:04d}"
        loaded = core.load_terminal_checkpoint(checkpoint)
        assert state_identity(loaded.current, loaded.normal_state) == report['input_state']
        records = [json.loads(line) for line in gzip.open(path.parent/'range_inputs.jsonl.gz', 'rt')]
        original_input = raw/'profiles'/Path(report['profile']).name/f"input_step{report['step']:04d}.pt"
        assert sha(original_input) == report['input_sha256']
        if replay:
            captured = []
            with prepared_remainder_replay(True), packed_boundary_execution(False), capture_ranges(captured):
                result = step(report['plant'], loaded.current, loaded.normal_state, report['step'])
            assert result.status == 'validated'
            assert [record for _, _, record in captured] == records, 'range input not from the saved real step'
            for mode in (0, 1):
                attribution = Attribution()
                prototype_plan.cache_clear()
                with attribution, TensorCounts(attribution):
                    attribution.stack.append({'category': 'D', 'children': 0.})
                    try:
                        values = [record[mode]() for record in captured]
                    finally:
                        attribution.stack.pop()
                assert all([float(v.lo).hex(), float(v.hi).hex()] == record['bounds'] for v, record in zip(values, records))
                allocation = report['allocations'][mode]
                assert allocation['interval_constructions'] == sum(attribution.intervals.values())
                assert allocation['tensor_outputs'] == sum(attribution.tensors.values())
                assert allocation['explicit_copy_events'] == sum(attribution.copies.values())
        counts = verify_range_file(path.parent/'range_inputs.jsonl.gz')
        assert all(t['calls'] == counts['range_calls'] and t['seconds'] > 0 for t in report['timings'])
        ratios = [next(t['seconds'] for t in report['timings'] if t['repeat'] == i and t['mode'] == 'baseline') /
                  next(t['seconds'] for t in report['timings'] if t['repeat'] == i and t['mode'] == 'candidate') for i in (1, 2, 3)]
        assert report['local_speedups'] == ratios and report['s'] == median(ratios)
        profile = raw/'profiles'/Path(report['profile']).name
        assert sha(profile/'breakdown.csv') == report['profile_sha256']
        costs = read_csv(profile/'breakdown.csv')
        fraction = sum(float(r['seconds']) for r in costs if r['category'] == 'D') / sum(float(r['seconds']) for r in costs)
        assert report['f'] == fraction
        assert report['predicted_whole_window_speedup'] == 1 / ((1-fraction) + fraction/report['s'])
        assert report['infinite_speedup_limit'] == 1 / (1-fraction)
        assert report['implementation_sha256'] == sha(ROOT/'experiments/boundary_execution/prototype_range.py')
        assert report['harness_sha256'] == sha(ROOT/'experiments/boundary_execution/prototype.py')
        reports.append({'case': path.parent.name, **counts})
    assert len(reports) == 8
    return reports


def check_states(raw, *, replay=True):
    summaries = []
    for path in sorted((raw/'state_checks').glob('*/result.json')):
        report = read_json(path)
        checkpoint = raw/'state_inputs'/Path(report['checkpoint']).name
        stored = [json.loads(line) for line in gzip.open(path.parent/'boundary_objects.jsonl.gz', 'rt')]
        assert [r['step'] for r in stored] == report['steps']
        if replay:
            with tempfile.TemporaryDirectory(prefix='boundary-resume-') as tmp:
                actual = verify_sequence(checkpoint, steps=len(stored), output=Path(tmp))
            assert actual == stored, 'boundary objects, ownership or next input changed'
        summaries.append({'case': path.parent.name, 'steps': report['steps'], 'history_lengths': report['history_lengths'],
                          'full_objects_rollback_resume_measurement_verified': True})
    by_name = {r['case']: r for r in summaries}
    expected = {'brusselator_early': [2, 3], 'brusselator_middle': [101, 102],
                'brusselator_long': [995, 996], 'brusselator_reset': [999, 1000, 1001],
                'vdp_early': [2, 3], 'vdp_reset': [99, 100, 101]}
    assert {name: r['steps'] for name, r in by_name.items()} == expected
    assert by_name['vdp_reset']['steps'] == [99, 100, 101]
    assert by_name['vdp_reset']['history_lengths'] == [98, 99, 0]
    assert by_name['brusselator_reset']['steps'] == [999, 1000, 1001]
    assert by_name['brusselator_reset']['history_lengths'] == [998, 999, 0]
    return summaries


def check_result(root):
    actual = decision(Path(root)/'raw_minimal/formal')
    assert actual == read_json(Path(root)/'RESULT.json'), 'derived result identity or timing changed'
    return actual


def verify(root):
    root = Path(root)
    assert all((root/name).is_file() for name in REQUIRED)
    files = manifest(root)
    source = check_source(root)
    raw = root/'raw_minimal'; formal = raw/'formal'
    commands = read_json(formal/'commands.json'); names = [c['name'] for c in commands]
    assert len(names) == len(set(names))
    assert read_json(formal/'SCHEDULE_COMPLETED.json') == {'source_sha': SCIENTIFIC_SHA, 'jobs': len(commands)}
    for window, (plant, start) in WINDOWS.items():
        for i in (1, 2, 3):
            pair = [f'{window}_pair{i}_{m}' for m in (['baseline', 'candidate'] if i % 2 else ['candidate', 'baseline'])]
            assert names.index(pair[0]) < names.index(pair[1])
            for name in pair:
                s = read_json(formal/name/'summary.json')
                assert s['plant'] == plant and s['start_boundary'] == start and s['accepted_steps'] == 20
    for plant in ('brusselator', 'van_der_pol'):
        for i in (1, 2, 3):
            pair = [f'{plant}_prefix100_pair{i}_{m}' for m in (['baseline', 'candidate'] if i % 2 else ['candidate', 'baseline'])]
            assert names.index(pair[0]) < names.index(pair[1])
            for name in pair:
                s = read_json(formal/name/'summary.json')
                assert s['plant'] == plant and s['start_boundary'] == 0 and s['accepted_steps'] == 100
    checkpoint_count = 0
    for command in commands:
        assert command['exit_code'] == 0 and int((formal/(command['name']+'.exit')).read_text()) == 0
        assert sha(ROOT/'experiments/boundary_execution/launch.py') == command['launcher_sha256']
        assert command['prepared_remainder_replay'] is True
        assert command['packed_boundary_execution'] is (command['mode'] == 'candidate')
        s = verify_run(formal/command['name'], command, recompute_common=False)
        assert s['source_files'] == source['scientific_sources']
        assert s['runner_sha256'] == sha(ROOT/'experiments/boundary_execution/run.py')
        checkpoint_count += check_run_checkpoints(formal/command['name'])
    result = check_result(root)
    expected = []
    for window in WINDOWS:
        for repeat in (1, 2, 3):
            expected += [f'{window}_pair{repeat}_{mode}' for mode in (['baseline', 'candidate'] if repeat % 2 else ['candidate', 'baseline'])]
    for plant in ('brusselator', 'van_der_pol'):
        for repeat in (1, 2, 3):
            expected += [f'{plant}_prefix100_pair{repeat}_{mode}' for mode in (['baseline', 'candidate'] if repeat % 2 else ['candidate', 'baseline'])]
    for plant, prefix in [('brusselator', 'brusselator'), ('van_der_pol', 'vdp')]:
        expected += [f'{prefix}_full_{mode}' for mode in ('baseline', 'candidate')]
        if len(result['full_pairs'][plant]) == 2:
            expected += [f'{prefix}_full_reverse_{mode}' for mode in ('candidate', 'baseline')]
    assert names == expected + ['vdp_adaptive_candidate'], 'all planned records and reversal order required'
    same_csv(root/'runtime_pairs.csv', runtime_rows(formal))
    same_csv(root/'full_width_equivalence.csv', full_width_rows(formal))
    for name in ('brusselator_full_baseline', 'vdp_full_baseline', 'vdp_adaptive_candidate'):
        verify_run(formal/name, recompute_common=True)
    parent = ROOT/'artifacts/runs/repaired_solver_performance_20260908T034636Z/raw_minimal/formal'
    bridges = [compare_parent_archive(formal/name, parent/old) for name, old in (
        ('brusselator_full_baseline', 'brusselator_full_optimized'), ('vdp_full_baseline', 'vdp_full_optimized'),
        ('vdp_adaptive_candidate', 'vdp_adaptive_optimized'))]
    assert bridges == read_json(formal/'parent_archive_equivalence.json')
    check_profiles(raw)
    check_checkpoint_inputs(raw)
    check_parent_checks(raw)
    same_csv(root/'boundary_cost_breakdown.csv', profile_cost_rows(raw/'profiles'))
    same_csv(root/'remaining_hotspots.csv', remaining_cost_rows(raw))
    for path in (raw/'remaining_profiles').glob('*/source.json'):
        record = read_json(path)
        assert record['scientific_sources'] == source['scientific_sources']
        loaded = core.load_terminal_checkpoint(raw/'state_inputs'/path.parent.name)
        assert record['before'] == state_identity(loaded.current, loaded.normal_state)
        profile = Path(read_json(raw/'state_inputs'/path.parent.name/'terminal_state.json')['provenance']['capture_profile']).name
        original = next(r for r in read_json(raw/'profiles'/profile/'states.json') if r['before'] == record['before'])
        assert record['after'] == original['after']
    assert read_json(root/'implementation_decision.json') == implementation_decision(raw)
    local, states = check_local(raw), check_states(raw)
    assert read_json(root/'local_and_state_equivalence.json') == {'local': local, 'state': states}
    tests = check_tests(root)
    return {'passed': True, 'files': files, 'scientific_sha': SCIENTIFIC_SHA,
            'package_checkout_sha': git('rev-parse', 'HEAD').decode().strip(),
            'status': result['status'], 'tests': tests['unique_totals'],
            'recomputed_range_calls': sum(r['range_calls'] for r in local),
            'rationally_checked_terms': sum(r['rationally_checked_terms'] for r in local),
            'full_boundary_positions': sum(len(r['steps']) for r in states),
            'safe_checkpoints_recomputed': checkpoint_count,
            'long_ODE_runs_repeated_by_verifier': False, 'whole_solver_formal_proof_claimed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    print(json.dumps(verify(args.package), indent=2), flush=True)
