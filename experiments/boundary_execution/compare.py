"""Bitwise comparison with the prepared baseline and explicit switch identity."""
import json
from pathlib import Path

from experiments.repaired_solver_performance.compare import _digest, compare_repaired_archive


def check_switches(record):
    assert record['mode'] in {'baseline', 'candidate'}
    assert record['prepared_remainder_replay'] is True, 'prepared replay must be the denominator'
    assert record['packed_boundary_execution'] is (record['mode'] == 'candidate'), 'boundary switch identity'


def compare_pair(baseline, candidate):
    paths = [Path(baseline), Path(candidate)]
    summaries = [json.loads((p / 'summary.json').read_text()) for p in paths]
    a, b = summaries
    assert a['mode'] == 'baseline' and b['mode'] == 'candidate'
    assert a['data_origin'] == 'FRESH_PREPARED_BASELINE'
    assert b['data_origin'] == 'FRESH_PACKED_BOUNDARY_CANDIDATE'
    for p, s in zip(paths, summaries):
        for record in (s, s['config'], json.loads((p/'source.json').read_text())):
            check_switches(record)
        assert s['completed'] and s['source_clean_at_start'] and s['source_clean_at_end']
        assert s['observer_mode'] == 'production_no_observer' and s['plan_construction_included_in_solve']
    for key in ('scientific_sha', 'source_files', 'plant', 'adaptive', 'accepted_steps', 'rejected_attempts',
                'accepted_horizon_exact', 'start_boundary', 'starting_time_exact', 'scheduler_time_hex',
                'refinement_totals', 'final_queue_size', 'final_queue_reset_count', 'threads',
                'interop_threads', 'affinity', 'python', 'torch_version'):
        assert a[key] == b[key], key
    configs = [dict(s['config']) for s in summaries]
    for config in configs:
        config.pop('mode')
        config.pop('packed_boundary_execution')
    assert configs[0] == configs[1], 'mathematical configuration drift'
    hashes = {}
    for name in ('models.jsonl.gz', 'bounds.csv', 'endpoint_audit.jsonl'):
        left, right = (_digest(p/name) for p in paths)
        assert left == right, f'bitwise numerical payload mismatch: {name}'
        hashes[name] = left
    return dict(baseline=paths[0].name, candidate=paths[1].name, accepted_steps=a['accepted_steps'],
        all_steps_bitwise_equal=True, hashes=hashes,
        baseline_solve_seconds=a['solve_seconds'], candidate_solve_seconds=b['solve_seconds'],
        speedup=a['solve_seconds']/b['solve_seconds'], peak_rss_ratio=b['peak_rss_bytes']/a['peak_rss_bytes'])


def compare_parent_archive(fresh, archive):
    """The parent optimized archive is a numerical anchor, never a timing pair."""
    archived = json.loads((Path(archive)/'summary.json').read_text())
    assert archived['mode'] == 'optimized'
    assert archived['scientific_sha'] == '1551ab57aef7324f91882beeba9d368f36b3cdd5'
    result = compare_repaired_archive(fresh, archive)
    result['archive_label'] = 'REUSED_PREPARED_PARENT_NUMERICAL_OBJECTS'
    return result
