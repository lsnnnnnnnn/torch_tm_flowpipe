"""Recompute this repair's evidence; this command does not rerun long ODE solves."""
import argparse,csv,hashlib,json,subprocess,xml.etree.ElementTree as ET
from fractions import Fraction as F
from pathlib import Path
import torch

from experiments.endpoint_roundoff_repair.audit import prove_run,require
from experiments.endpoint_roundoff_repair.report import derive,ROLES
from experiments.endpoint_roundoff_repair.local_oracles import CASES,check_case
from experiments.our_solver_performance.reference_endpoint import ENTRIES,endpoint_witness
from experiments.xiangru_adoption.common import measure

SCIENTIFIC='196a50e9131336d68df07ad0af353deca0092d19'
DELIVERY='0714e475ed9e73bec31619c9c690d1fd63de3d36'
COMPATIBILITY='8c867d55c52f07cd824f381e0826efe17097e7ff'
PARENT='73c3b48a3dadd81cd03ecc827bde1228ef7f6ac8'
ARCHIVE=Path('artifacts/runs/xiangru_adoption_20260907T032448Z')
RUNS=('vdp_prefix20','brusselator_prefix20','vdp_prefix120','vdp_full','brusselator_full','vdp_adaptive')
GROUPS={'root','old_stop_snapshot','torch_basis','diffreach_followup','first_order_three_way',
        'diffreach_support','common_contract','comparison_repair','deep_study'}
SUCCESS='ENDPOINT_REPAIR_CLOSED__FULL_HORIZONS_REVALIDATED'
CHANGED='ENDPOINT_REPAIR_CLOSED__FULL_RUN_BEHAVIOR_CHANGED'


def read(path):return json.loads(path.read_text())
def canonical(value):return json.loads(json.dumps(value,allow_nan=False))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args])


def check_manifest(root):
    seen=set()
    for line in (root/'SHA256SUMS').read_text().splitlines():
        digest,rel=line.split('  ',1)
        require(rel not in seen and not Path(rel).is_absolute() and '..' not in Path(rel).parts,'manifest path')
        seen.add(rel);require(sha(root/rel)==digest,'artifact hash: '+rel)
    require(seen=={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS'},'complete manifest')


def test_matrix(root):
    commands=read(root/'tests/delivery/full_commands.json')
    require({c['group'] for c in commands}==GROUPS,'complete independent test matrix')
    identities=set();groups=[];endpoint=[]
    for c in commands:
        require(c['source_sha']==(PARENT if c['group']=='old_stop_snapshot' else DELIVERY),'test source identity')
        require(c['source_clean_at_start'] and c['exit_code']==0,'test command exit/clean source')
        require(int((root/c['exit']).read_text())==0,'actual test process exit')
        require((root/c['log']).stat().st_size>0,'test log missing')
        counts={'passed':0,'skipped':0,'failed':0,'errors':0}
        cases=ET.parse(root/c['xml']).getroot().findall('.//testcase')
        require(bool(cases),'empty JUnit group')
        for case in cases:
            identity=(case.get('classname'),case.get('name'))
            require(identity not in identities,'duplicate test accounting')
            identities.add(identity)
            status='failed' if case.find('failure') is not None else 'errors' if case.find('error') is not None else 'skipped' if case.find('skipped') is not None else 'passed'
            counts[status]+=1
            if 'test_our_reference_endpoint_containment' in str(identity[0]):endpoint.append(status)
        require(not counts['failed'] and not counts['errors'],'regression failure')
        groups.append({'group':c['group'],**counts,'source_sha':c['source_sha']})
    require(endpoint==['passed','passed'],'original endpoint safety assertions must pass normally')
    return {'groups':groups,'unique_totals':{k:sum(g[k] for g in groups) for k in ['passed','skipped','failed','errors']},
            'original_endpoint_tests_passed':2,'local_targeted_tests_added_again':False}


def derive_status(runs,tests,carry):
    if tests['original_endpoint_tests_passed']!=2:return 'ENDPOINT_REPAIR_FAILED__COUNTEREXAMPLE_REMAINS'
    if not carry['all_actual_paths_closed']:return 'ENDPOINT_REPAIR_PARTIAL__CARRY_OR_OTHER_SAFETY_BLOCKER'
    if all(r['completed'] and r['failure'] is None for r in runs.values()):return SUCCESS
    return CHANGED


def validate_local(root):
    before=read(root/'before_endpoint_witness.json')
    h=float.fromhex('0x1.47ae147ae147bp-7');exact=-1+100*F(h)
    require(before['source_sha']==PARENT and before['time_fraction']==str(F(h)) and before['exact_value']==str(exact),'BEFORE actual binary64 interpretation')
    for row in before['rows']:
        require(not any(F(a)<=exact<=F(b) for a,b in measure(row['output_model'])),'BEFORE must reproduce original counterexample')
    after=read(root/'after_endpoint_oracles.json')
    require(after['source_sha']==SCIENTIFIC and after['source_clean'],'local scientific source')
    require(len(after['original_witnesses'])==2,'two real endpoint entries')
    for expected,entry in zip(after['original_witnesses'],ENTRIES,strict=True):
        _,endpoint,q=endpoint_witness(entry)
        observed=[[float(b.lo).hex(),float(b.hi).hex()] for b in endpoint.range_box()]
        require(q==exact and expected['exact']==str(exact),'AFTER rational witness value')
        require(expected['entry']==entry and expected['full_bounds_hex']==observed,'production witness replay differs')
        require(expected['contains']==[True,True] and all(F(float.fromhex(lo))<=q<=F(float.fromhex(hi)) for lo,hi in observed),'original counterexample remains')
    require(after['general_cases']==canonical([check_case(case) for case in CASES]),'independent general coefficient oracles')
    compat=read(root/'raw_minimal/compatibility_endpoint_oracles.json')
    require(compat['source_sha']==COMPATIBILITY and compat['general_cases']==after['general_cases'] and compat['original_witnesses']==after['original_witnesses'],'CPU binary64 compatibility oracles changed')


def compare_csv(path,expected):
    rows=list(csv.DictReader(path.open()))
    require(len(rows)==len(expected),'table row count: '+path.name)
    for row,actual in zip(rows,expected,strict=True):
        require(set(row)==set(actual),'table columns: '+path.name)
        for key,value in actual.items():
            require(row[key]==('' if value is None else str(value)),f'table recomputation: {path.name} {key}')


def verify(root,repo,*,manifest=True,full_math=True):
    if manifest:check_manifest(root)
    source=read(root/'SOURCE_MAP.json')
    require(source['scientific_sha']==SCIENTIFIC and source['delivery_runtime_sha']==DELIVERY,'repair source identity')
    require(source['data_roles']=={'old_cpu':ROLES['old'],'new_cpu':ROLES['repaired'],'flowstar':ROLES['flowstar']},'reused data relabeled fresh')
    for path,digest in source['source_files'].items():
        require(hashlib.sha256(git(repo,'show',SCIENTIFIC+':'+path)).hexdigest()==digest,'scientific source content')
    for path,digest in source['scientific_runner_files'].items():
        require(hashlib.sha256(git(repo,'show',SCIENTIFIC+':'+path)).hexdigest()==digest,'scientific runner content')
    for path,digest in source['delivery_runtime_files'].items():
        require(sha(repo/path)==digest and hashlib.sha256(git(repo,'show',DELIVERY+':'+path)).hexdigest()==digest,'delivery runtime content')
    for path,digest in source['reused_files'].items():require(sha(repo/path)==digest,'frozen historical file changed')
    require(git(repo,'show',PARENT+':tests/test_our_reference_endpoint_containment.py')==(repo/'tests/test_our_reference_endpoint_containment.py').read_bytes(),'original safety assertions modified')
    validate_local(root)
    tests=test_matrix(root)
    require(read(root/'test_results.json')==tests,'test totals recomputation')
    tables,runs,impact=derive(root,repo/ARCHIVE)
    for name,rows in tables.items():compare_csv(root/name,rows)
    require(read(root/'full_run_summaries.json')==runs,'run summaries changed')
    require(read(root/'historical_claim_impact.json')==impact,'historical claim impact changed')
    carry=read(root/'two_step_carry_checks.json')
    require(carry['all_actual_paths_closed'] and carry['scientific_sha']==SCIENTIFIC,'actual carry closure missing')
    for short in ['vdp','brusselator']:
        resume=read(root/f'raw_minimal/{short}_resume_after120.json')
        require(resume['passed'] and resume['input_state_unchanged'] and resume['continuous_and_resumed_queue_identical'] and all(resume['bit_identical_models'].values()),'checkpoint resume evidence')
        delivery=read(root/f'raw_minimal/{short}_delivery_resume_after120.json')
        require(delivery['scientific_sha']==DELIVERY and delivery['passed'] and delivery['input_state_unchanged']
                and delivery['continuous_and_resumed_queue_identical'] and all(delivery['bit_identical_models'].values()),'delivery checkpoint replay')
    require({r['path'] for r in carry['paths']}=={'published','dense_internal','normal_sr','g1','g2','s1_current','s1_total_delta'},'direct consumer coverage')
    for row in carry['paths']:require(row['two_steps_passed'] and row['first_error_consumed_by_second'],'two-step path proof')
    if full_math:
        from experiments.endpoint_roundoff_repair.carry import collect
        replay=collect(repo)
        require(replay['paths']==carry['paths'] and replay['failed_attempts']==carry['failed_attempts'],'actual consumer/rollback replay changed')
    ownership=read(root/'error_ownership.json')
    require(ownership['paths']==carry['paths'] and ownership['new_error_category']=='endpoint_substitution_roundoff', 'error ownership differs from actual consumers')
    bridge=read(root/'raw_minimal/runtime_bridge.json')
    require(bridge['scientific_run_sha']==SCIENTIFIC and bridge['delivery_runtime_sha']==DELIVERY,'runtime bridge identity')
    require(git(repo,'diff','--name-only',SCIENTIFIC,DELIVERY,'--','src').decode().splitlines()==bridge['changed_runtime_files'],'runtime bridge source scope')
    require(not bridge['frozen_full_runs_were_rerun_at_delivery_sha'],'a replay must not be labeled a fresh long run')
    result=read(root/'RESULT.json')
    require(result['status']==derive_status(runs,tests,carry),'final status is not derived from actual evidence')
    require(result['scientific_sha']==SCIENTIFIC and result['delivery_runtime_sha']==DELIVERY,'result source identity')
    require(result['whole_solver_formally_proved'] is False and result['new_performance_implementation'] is False,'claim exceeds scope')
    proofs=[]
    matched=read(repo/ARCHIVE/'MATCHED_CONTRACTS.json')
    for name in RUNS:
        run=root/'raw_minimal'/name
        raw_source=read(run/'source.json');summary=read(run/'summary.json');contract=read(run/'execution_contract.json')
        require(raw_source['scientific_sha']==SCIENTIFIC and raw_source['source_files']==source['source_files'],'run scientific content differs')
        require(raw_source['source_clean_at_start'] and summary['source_clean_at_end'],'run not from clean source')
        require(raw_source['data_origin']==ROLES['repaired'],'raw run role')
        require(int((root/f'raw_minimal/{name}.exit').read_text())==0,'long-run process exit')
        require(contract['endpoint_substitution_roundoff_required'] and not contract['endpoint_ad_hoc_repair'],'required correction or optional tightening changed')
        require(contract['config']==matched['plants'][summary['plant']]['our_frozen_configuration'],'frozen numerical configuration changed')
        proof=prove_run(run,partial=not full_math)
        proofs.append(proof)
        if full_math and name in ['vdp_full','brusselator_full','vdp_adaptive']:
            require(read(root/f'raw_minimal/{name}_delivery_replay.json')==canonical(proof),'saved independent full-run audit differs')
    return {'passed':True,'verification_kind':'saved-evidence recomputation; no full ODE rerun',
            'status':result['status'],'scientific_sha':SCIENTIFIC,'delivery_runtime_sha':DELIVERY,
            'exact_endpoint_component_checks':sum(p['full_function_containment_checks'] for p in proofs),
            'exact_coefficient_checks':sum(p['exact_coefficient_containment_checks'] for p in proofs),
            'production_endpoint_replays':sum(p['production_endpoint_replays'] for p in proofs),
            'width_rows_recomputed':len(tables['widths_full_prefix.csv']),'test_matrix':tests}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--output',type=Path)
    args=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    result=verify(args.root.resolve(),args.repo.resolve())
    if args.output:dump(args.output,result)
    print(json.dumps(result,indent=2))
