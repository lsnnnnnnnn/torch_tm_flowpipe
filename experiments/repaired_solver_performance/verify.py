"""Bounded performance evidence verifier, separate from the inherited safety proof."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import torch

from experiments.repaired_solver_performance.analyze import (
    BASE_SHA, RUNTIME_SHA, SCIENTIFIC_SHA, FULL_NAMES, WINDOWS, REUSED_INPUT_PATHS, read_json, read_csv,
    verify_run, timing_tables, full_width_rows, profile_tables, decision, flowstar_widths,
)
from experiments.repaired_solver_performance.compare import compare_pair, compare_repaired_archive
from experiments.repaired_solver_performance.replay import verify_record, equal
from experiments.endpoint_roundoff_repair.frozen import ROOT, MATCHED_SHA256


REQUIRED=('SOURCE_MAP.json','RUN_CONTEXT.json','EXECUTION_CONTRACT.json','profile_windows.csv',
          'replay_work_counts.csv','candidate_decision.json','same_input_replay_equivalence.json',
          'full_width_equivalence.csv','full_run_summaries.json','timings_raw.csv','timing_summary.csv',
          'remaining_hotspots.csv','RESULT.json','tests/commands.json','SHA256SUMS')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(root):
    root=Path(root);declared={}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        path=Path(name)
        assert not path.is_absolute() and '..' not in path.parts and name not in declared
        assert (root/path).is_file() and not (root/path).is_symlink()
        assert sha(root/path)==digest,f'manifest mismatch: {name}'
        declared[name]=digest
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS'}
    assert actual==set(declared),'manifest missing or extra files'
    return len(declared)


def roles(source):
    assert source['base_sha']==BASE_SHA and source['scientific_sha']==SCIENTIFIC_SHA and source['runtime_sha']==RUNTIME_SHA
    assert source['data_roles']=={'reference':'FRESH_REPAIRED_REFERENCE','optimized':'FRESH_OPTIMIZED',
        'repaired_archive':'REUSED_ENDPOINT_REPAIRED','old_unrepaired':'OLD_UNREPAIRED_ARCHIVE',
        'flowstar':'REUSED_MATCHED_REFERENCE'}
    assert source['repaired_archive_sha']=='196a50e9131336d68df07ad0af353deca0092d19'
    assert source['repaired_reference_runtime_sha']=='0714e475ed9e73bec31619c9c690d1fd63de3d36'
    assert source['flowstar_sha']=='b85a3211748cb77b736fe4ad42ee02d8d2b81148'
    assert source['old_unrepaired_sha']=='4939fb288c941a67f55cc191f4d75f8594692f47'
    assert source['matched_contracts_sha256']==MATCHED_SHA256


def check_source_map(source):
    roles(source)
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args])
    paths=git('ls-tree','-r','--name-only',SCIENTIFIC_SHA,'src/torch_tm_flowpipe').decode().splitlines()
    paths=[p for p in paths if p.endswith('.py') and len(Path(p).parts)==3]
    expected={p:hashlib.sha256(git('show',f'{SCIENTIFIC_SHA}:{p}')).hexdigest() for p in paths}
    assert expected==source['runtime_sources']
    assert all(sha(ROOT/p)==digest for p,digest in expected.items()),'imported numeric source differs from scientific snapshot'
    assert not git('diff',RUNTIME_SHA,SCIENTIFIC_SHA,'--','src/torch_tm_flowpipe').strip()
    assert source['goal_sha256']=='36e0e623e3cf09fe4abf7f83ca59ab243006c126260caa171e8fcd48d944ca6e'
    assert source['package_is_separate_from_scientific_commit'] and not source['native_flowstar_rerun']
    assert set(source['reused_input_hashes'])==set(REUSED_INPUT_PATHS)
    for name,digest in source['reused_input_hashes'].items():
        assert hashlib.sha256(git('show',f'{BASE_SHA}:{name}')).hexdigest()==digest
        assert sha(ROOT/name)==digest,'inherited input modified'
    return expected


def same_csv(path,rows):
    assert read_csv(path)==[{k:str(v) for k,v in row.items()} for row in rows],f'derived CSV changed: {path}'


def check_replay_file(path,*,recompute=True):
    rounds=0;records=[]
    with gzip.open(path,'rt') as handle:
        for line in handle:
            assert len(records)<20 and len(line)<10_000_000,'replay evidence exceeds bounded local scope'
            record=json.loads(line)
            assert record['plant'] in {'brusselator','van_der_pol'}
            order=6 if record['plant']=='brusselator' else 4
            for model in [record['base'],record['candidate']]:
                assert model['basis_dim']==3 and model['order']==order
                assert model['coefficients']['shape']==[1,2,math.comb(order+3,3)]
                assert model['coefficients']['dtype']=='float64'
                assert len(model['coefficients']['values'])==2*math.comb(order+3,3)
            assert record['kwargs']['order']==order and record['kwargs']['tau_index']==2
            assert record['kwargs']['cutoff_threshold']==1e-10 and record['kwargs']['validation_eps']==1e-12
            assert record['kwargs']['observer_mode']=='full_evidence'
            assert 0<=len(record['reference']['rows'])<=491
            equal(record['reference'],record['optimized'])
            if recompute:rounds+=verify_record(record)
            else:rounds+=len(record['reference']['rows'])
            records.append(record)
    return records,rounds


def check_tests(root):
    commands=read_json(root/'tests/commands.json')
    unique={};groups={}
    for command in commands:
        assert command['exit_code']==0
        assert re.fullmatch('[0-9a-f]{40}',command['source_sha'])
        assert re.fullmatch('[a-z0-9_]+',command['name'])
        for key in ['log','exit','xml']:
            if key in command:assert command[key]==f"tests/{command['name']}.{key}"
        if command['name']=='old_stop_snapshot':assert command['source_sha']=='73c3b48a3dadd81cd03ecc827bde1228ef7f6ac8'
        elif command['name'].startswith('parent_') or command['name']=='repaired_parent_evidence':assert command['source_sha']==BASE_SHA
        else:
            assert not subprocess.check_output(['git','-C',str(ROOT),'diff',SCIENTIFIC_SHA,command['source_sha'],'--','src/torch_tm_flowpipe']).strip()
        assert (root/command['log']).is_file()
        assert int((root/command['exit']).read_text())==0
        if 'xml' not in command:continue
        counts=Counter()
        tree=ET.parse(root/command['xml'])
        for case in tree.iter('testcase'):
            status=('failed' if case.find('failure') is not None else 'errors' if case.find('error') is not None
                    else 'skipped' if case.find('skipped') is not None else 'passed')
            assert status not in {'failed','errors'}
            counts[status]+=1
            if command.get('count_in_final_matrix'):
                identity=(case.get('classname'),case.get('name'))
                assert identity not in unique,f'duplicate counted test: {identity}'
                unique[identity]=status
        groups[command['name']]=dict(counts)
    required={'root','old_stop_snapshot','torch_basis','diffreach_followup','first_order_three_way',
              'diffreach_support','common_contract','comparison_repair','deep_study','repaired_parent_evidence'}
    assert required <= {c['name'] for c in commands if c.get('count_in_final_matrix')}
    expected={'unique_totals':dict(Counter(unique.values())),'groups':groups,'repeated_local_and_clone_tests_added_again':False}
    assert read_json(root/'tests/FINAL_TEST_ACCOUNTING.json')==expected
    return expected


def check_captured_checkpoints(root):
    from torch_tm_flowpipe import load_terminal_checkpoint, tmvector_hashes, accepted_boundary_sr_queue_sha256
    raw=root/'raw_minimal/formal'
    for path,index,plant,run,order in [
        (root/'raw_minimal/checkpoint_0090',90,'van_der_pol','vdp_full_reference',4),
        (raw/'brusselator_full_reference/checkpoint_0980',980,'brusselator','brusselator_full_reference',6),
    ]:
        restored=load_terminal_checkpoint(path,expected_dtype='float64',expected_order=order)
        assert restored.normal_state.step_index==index and restored.contract['plant']==plant
        audit=json.loads((raw/run/'endpoint_audit.jsonl').read_text().splitlines()[index-1])
        assert restored.scheduler['time_exact']==audit['t_end_exact']
        assert accepted_boundary_sr_queue_sha256(restored.normal_state.symbolic_queue)==audit['queue_sha256']
        for name,tm in [('current',restored.current),('normal_pre',restored.normal_state.tmv_pre),
                        ('normal_right',restored.normal_state.tmv_right)]:
            assert tmvector_hashes(tm)==audit['state_hashes'][name]
        if index==90:
            old=read_json(raw/'vdp90_source.json')
            assert old['manifest']==restored.manifest and old['provenance']==restored.provenance
            assert old['reused_complete_state']
        else:assert restored.provenance['scientific_sha']==SCIENTIFIC_SHA


def verify(root):
    root=Path(root)
    assert all((root/p).is_file() for p in REQUIRED),'required deliverable missing'
    assert any((root/'figures').glob('*.png')) and any((root/'figures').glob('*.pdf'))
    files=manifest(root)
    source=read_json(root/'SOURCE_MAP.json');runtime=check_source_map(source)
    assert sha(root/'raw_minimal/GOAL_FROZEN.md')==source['goal_sha256']
    context=read_json(root/'RUN_CONTEXT.json')
    assert context['scientific_sha']==SCIENTIFIC_SHA and context['runtime_sha']==RUNTIME_SHA
    assert context['production_affinity']==[2] and context['threads']==1
    raw=root/'raw_minimal/formal';commands=read_json(raw/'commands.json')
    names=[c['name'] for c in commands];assert len(names)==len(set(names))
    completion=read_json(raw/'SCHEDULE_COMPLETED.json')
    assert completion=={'source_sha':SCIENTIFIC_SHA,'jobs':len(commands)}
    for window in WINDOWS:
        for repeat in [1,2,3]:
            expected=[f'{window}_pair{repeat}_{mode}' for mode in
                      (['reference','optimized'] if repeat%2 else ['optimized','reference'])]
            assert names.index(expected[0])<names.index(expected[1]),'pair order changed'
            compare_pair(raw/f'{window}_pair{repeat}_reference',raw/f'{window}_pair{repeat}_optimized')
    for plant in ['brusselator','van_der_pol']:
        for repeat in [1,2,3]:
            modes=['reference','optimized'] if repeat%2 else ['optimized','reference']
            assert names.index(f'{plant}_prefix100_pair{repeat}_{modes[0]}')<names.index(f'{plant}_prefix100_pair{repeat}_{modes[1]}')
            compare_pair(raw/f'{plant}_prefix100_pair{repeat}_reference',raw/f'{plant}_prefix100_pair{repeat}_optimized')
    for prefix in ['brusselator','vdp']:compare_pair(raw/(prefix+'_full_reference'),raw/(prefix+'_full_optimized'))
    summaries={}
    for command in commands:
        assert command['exit_code']==0 and command['source_sha']==SCIENTIFIC_SHA
        assert int((raw/(command['name']+'.exit')).read_text())==0
        assert (raw/command['log']).is_file()
        if command['module']=='run':
            summary=verify_run(raw/command['name'],command,recompute_common=False)
            assert summary['source_files']==runtime
            assert Path(summary['imported_package'])==Path(summary['source_root'])/'src/torch_tm_flowpipe/__init__.py'
            summaries[command['name']]=summary
        else:
            identity=read_json(raw/command['name']/'source.json')
            assert identity['sha']==SCIENTIFIC_SHA and identity['source_clean_at_start']
            assert identity['affinity']==[2] and identity['threads']==1
            assert identity['python']==context['python']
            assert identity['imported_solver']==str(Path(command['cwd'])/'src/torch_tm_flowpipe/batched_dense_tm.py')
            if command['module']=='profile':
                assert identity['mode']==command['mode'] and identity['profile_is_not_production_timing']
                assert identity['runtime_hashes']=={Path(p).name:runtime[p] for p in
                    ['src/torch_tm_flowpipe/batched_dense_tm.py','src/torch_tm_flowpipe/prepared_remainder_replay.py']}
            else:
                assert command['module']=='replay' and identity['profile_or_evidence_not_production_timing']
                assert identity['dense_source_sha256']==runtime['src/torch_tm_flowpipe/batched_dense_tm.py']
    assert read_json(root/'EXECUTION_CONTRACT.json')=={name:s['config'] for name,s in summaries.items()}
    assert read_json(root/'full_run_summaries.json')=={name:summaries[name] for name in FULL_NAMES}
    expected_result=decision(raw)
    assert read_json(root/'RESULT.json')==expected_result
    raw_time,summary_time=timing_tables(raw)
    same_csv(root/'timings_raw.csv',raw_time);same_csv(root/'timing_summary.csv',summary_time)
    same_csv(root/'full_width_equivalence.csv',full_width_rows(raw))
    profiles,work,remaining,amdahl=profile_tables(raw)
    same_csv(root/'profile_windows.csv',profiles);same_csv(root/'replay_work_counts.csv',work)
    same_csv(root/'remaining_hotspots.csv',remaining)
    candidate=read_json(root/'candidate_decision.json')
    assert candidate['mechanism']=='attempt_local_prepared_polynomial_replay' and candidate['fallback_hotspot_used'] is False
    assert candidate['same_workload_predictions']==amdahl
    replay_reports=[]
    for name,(plant,start) in WINDOWS.items():
        records,rounds=check_replay_file(raw/(name+'_replay')/'replays.jsonl.gz')
        assert [r['step'] for r in records]==list(range(start,start+20))
        assert all(r['plant']==plant for r in records)
        states=read_json(raw/(name+'_replay')/'states.json')
        assert [s['step'] for s in states]==list(range(start,start+20))
        audit={x['step']:x for x in map(json.loads,(raw/(name+'_pair1_reference')/'endpoint_audit.jsonl').read_text().splitlines())}
        for row in states:
            current=audit[row['step']]
            assert row['h_hex']==current['h_hex'] and row['step_rejections']==current['step_rejections']
            assert row['endpoint_error']==current['substitution_error_hex'] and row['queue']==current['queue_sha256']
        replay_reports.append({'window':name,'steps':len(records),'proposal_rounds':rounds,'independent_loops_and_same_inputs_bitwise_equal':True})
    for name,start in [('vdp_adaptive_replay_initial',1),('vdp_adaptive_replay_after99',100)]:
        records,rounds=check_replay_file(raw/name/'replays.jsonl.gz')
        assert [r['step'] for r in records]==list(range(start,start+3))
        replay_reports.append({'window':name,'steps':len(records),'proposal_rounds':rounds,'independent_loops_and_same_inputs_bitwise_equal':True})
    assert read_json(root/'same_input_replay_equivalence.json')==replay_reports
    for name,start in [('vdp_adaptive_replay_initial',1),('vdp_adaptive_replay_after99',100)]:
        states=read_json(raw/name/'states.json')
        audit={x['step']:x for x in map(json.loads,(raw/'vdp_adaptive_optimized/endpoint_audit.jsonl').read_text().splitlines())}
        assert [r['step'] for r in states]==list(range(start,start+3))
        for row in states:
            current=audit[row['step']]
            assert row['h_hex']==current['h_hex'] and row['step_rejections']==current['step_rejections']
            assert row['endpoint_error']==current['substitution_error_hex'] and row['queue']==current['queue_sha256']
    # Complete canonical models have already been shown byte-identical in each
    # pair, so one independent common-observer evaluation proves both columns.
    for name in ['brusselator_full_reference','vdp_full_reference','vdp_adaptive_optimized']:
        verify_run(raw/name,recompute_common=True)
    old=ROOT/'artifacts/runs/endpoint_roundoff_repair_20260908/raw_minimal'
    archive=[compare_repaired_archive(raw/name,old/archive) for name,archive in
             [('brusselator_full_reference','brusselator_full'),('vdp_full_reference','vdp_full'),('vdp_adaptive_optimized','vdp_adaptive')]]
    assert read_json(raw/'repaired_archive_equivalence.json')==archive
    check_captured_checkpoints(root)
    flow,flow_summary=flowstar_widths(raw)
    same_csv(root/'reused_flowstar_widths.csv',flow)
    assert read_json(root/'reused_flowstar_width_summary.json')==flow_summary
    tests=check_tests(root)
    return {'passed':True,'files':files,'fresh_production_runs':len(summaries),
        'fixed_endpoint_components':4000,'fixed_tube_components':4000,
        'independently_recomputed_proposal_rounds':sum(r['proposal_rounds'] for r in replay_reports),
        'state':expected_result['status'],'tests':tests['unique_totals'],
        'source_identity_and_environment_checked_separately':True,
        'whole_solver_formal_proof_claimed':False,'long_ODE_runs_repeated_by_verifier':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('package',type=Path)
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    print(json.dumps(verify(args.package),indent=2),flush=True)
