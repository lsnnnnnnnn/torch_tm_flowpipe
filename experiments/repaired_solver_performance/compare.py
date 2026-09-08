"""Immediate gates on each actual paired production run, without tolerances."""
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path


def _digest(path):
    digest=hashlib.sha256()
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rb') as source:
        while chunk:=source.read(1024*1024):digest.update(chunk)
    return digest.hexdigest()


def compare_pair(reference,optimized):
    reference,optimized=Path(reference),Path(optimized)
    a=json.loads((reference/'summary.json').read_text())
    b=json.loads((optimized/'summary.json').read_text())
    assert a['mode']=='reference' and a['data_origin']=='FRESH_REPAIRED_REFERENCE'
    assert b['mode']=='optimized' and b['data_origin']=='FRESH_OPTIMIZED'
    for s in [a,b]:
        assert s['completed'] and s['source_clean_at_start'] and s['source_clean_at_end']
        assert s['observer_mode']=='production_no_observer' and s['plan_construction_included_in_solve']
    for key in ['scientific_sha','plant','adaptive','accepted_steps','rejected_attempts','accepted_horizon_exact',
                'start_boundary','starting_time_exact','scheduler_time_hex','refinement_totals',
                'final_queue_size','final_queue_reset_count','threads','interop_threads','affinity','python','torch_version']:
        assert a[key]==b[key],key
    ca,cb=(dict(s['config']) for s in [a,b])
    ca.pop('mode');cb.pop('mode')
    assert ca==cb,'mathematical contract changed'
    hashes={}
    for name in ['models.jsonl.gz','bounds.csv','endpoint_audit.jsonl']:
        left,right=_digest(reference/name),_digest(optimized/name)
        assert left==right,f'bitwise paired output changed: {name}'
        hashes[name]=left
    return {'reference':reference.name,'optimized':optimized.name,'all_steps_bitwise_equal':True,
            'accepted_steps':a['accepted_steps'],'rejected_attempts':a['rejected_attempts'],
            'mathematical_hashes':hashes,'reference_solve_seconds':a['solve_seconds'],
            'optimized_solve_seconds':b['solve_seconds'],'speedup':a['solve_seconds']/b['solve_seconds'],
            'peak_rss_ratio':b['peak_rss_bytes']/a['peak_rss_bytes']}


def compare_repaired_archive(fresh,archive):
    """Final repaired runtime bridge already verified on the clean parent.

The old archive is used for numerical regression only, never fresh timing.
New state-hash fields have no historical counterpart; compare the full saved
models and every historical audit field that does have a counterpart.
"""
    fresh,archive=Path(fresh),Path(archive)
    a=json.loads((fresh/'summary.json').read_text())
    b=json.loads((archive/'summary.json').read_text())
    for key in ['plant','adaptive','accepted_steps','rejected_attempts','accepted_horizon_exact',
                'scheduler_time_hex','refinement_totals','final_queue_size','final_queue_reset_count']:
        assert a[key]==b[key],key
    for name in ['models.jsonl.gz','bounds.csv']:
        assert _digest(fresh/name)==_digest(archive/name),name
    with (fresh/'endpoint_audit.jsonl').open() as x,(archive/'endpoint_audit.jsonl').open() as y:
        for left,right in zip_longest(x,y):
            assert left is not None and right is not None
            left,right=json.loads(left),json.loads(right)
            for key,value in right.items():assert left[key]==value,key
    return {'fresh':fresh.name,'archive':archive.name,'archive_label':'REUSED_ENDPOINT_REPAIRED',
            'archive_scientific_sha':b['scientific_sha'],'all_saved_numerical_fields_equal':True,
            'archive_timing_used_as_fresh_denominator':False}
