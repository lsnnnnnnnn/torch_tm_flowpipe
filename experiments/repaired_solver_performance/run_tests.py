"""Run the inherited finite test matrix with its historical snapshot routing."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from experiments.endpoint_roundoff_repair.frozen import ROOT
from experiments.repaired_solver_performance.analyze import BASE_SHA, read_json, write_json

PARENT=Path('/srv/local/shengenli/endpoint_roundoff_repair_20260908/repo')
OLD_STOP=Path('/srv/local/shengenli/our_solver_replay_performance_20260907T074937Z/repo')
OLD_STOP_SHA='73c3b48a3dadd81cd03ecc827bde1228ef7f6ac8'
TARGETED=['tests/test_prepared_remainder_replay.py','tests/test_our_reference_endpoint_containment.py',
          'tests/test_endpoint_roundoff_repair.py','tests/test_endpoint_roundoff_carry.py',
          'tests/test_brusselator_c4_generic_refinement.py','tests/test_vdp_c2_post_accept_refinement.py']


def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def fixtures(output):
    recovery=ROOT/'artifacts/runs/endpoint_roundoff_repair_20260908/raw_minimal/historical_fixture_recovery.json'
    recorded=read_json(recovery)
    source=Path(recorded['source'])
    target=ROOT/'experiments/three_way_common_contract/results/20260724T132534Z'
    assert len(recorded['files'])==25
    for row in recorded['files']:
        relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
        content=(source/relative).read_bytes()
        assert len(content)==row['size'] and hashlib.sha256(content).hexdigest()==row['sha256']
        destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert destination.read_bytes()==content
        else:destination.write_bytes(content)
    write_json(output/'historical_fixture_recovery.json',{
        'reused_record':str(recovery.relative_to(ROOT)),'reused_record_sha256':hashlib.sha256(recovery.read_bytes()).hexdigest(),
        'source':str(source),'restored_to':str(target),'files':recorded['files'],'all_hashes_match':True})


def run_group(output,name,root,python,paths,*,cpu,counted,expected_sha=None,environment=None):
    sha=git(root,'rev-parse','HEAD')
    if expected_sha:assert sha==expected_sha
    assert not git(root,'status','--porcelain'),f'test source is dirty: {root}'
    output.mkdir(parents=True,exist_ok=True)
    commands_path=output/'commands.json'
    commands=read_json(commands_path) if commands_path.exists() else []
    assert name not in {c['name'] for c in commands},'inspect recorded job; never silently repeat it'
    overrides={'PYTHONPATH':str(root/'src')+':'+str(root),'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1',
        'PATH':str(Path(python).parent)+os.pathsep+os.environ.get('PATH',''),
        'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
        'DIFFREACH_ROOT':'/srv/local/shengenli/DiffReach','JAX_PLATFORMS':'cpu',**(environment or {})}
    argv=['taskset','-c',str(cpu),python,'-m','pytest','-q','-p','no:cacheprovider',*paths,
          '--junitxml='+str(output/(name+'.xml'))]
    record={'name':name,'source_sha':sha,'source_clean_at_start':True,'cwd':str(root),'argv':argv,
        'environment':overrides,'affinity':[cpu],'count_in_final_matrix':counted,
        'log':'tests/'+name+'.log','xml':'tests/'+name+'.xml','exit':'tests/'+name+'.exit'}
    started=time.perf_counter()
    with (output/(name+'.log')).open('x') as log:
        process=subprocess.Popen(argv,cwd=root,env={**os.environ,**overrides},stdout=log,stderr=subprocess.STDOUT)
        record.update(pid=process.pid,proc_start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21])
        commands.append(record);write_json(commands_path,commands)
        print(json.dumps({'test_group':name,'pid':process.pid,'status':'RUNNING'}),flush=True)
        code=process.wait()
    (output/(name+'.exit')).write_text(str(code)+'\n')
    record.update(exit_code=code,seconds=time.perf_counter()-started,source_clean_at_end=not git(root,'status','--porcelain'))
    write_json(commands_path,commands)
    assert code==0 and record['source_clean_at_end'],f'{name} failed; see its actual log'
    assert git(root,'rev-parse','HEAD')==sha
    print(json.dumps({'test_group':name,'status':'PASSED','seconds':record['seconds']}),flush=True)


def accounting(bundle):
    commands=read_json(bundle/'tests/commands.json');unique={};groups={}
    for command in commands:
        if 'xml' not in command:continue
        counts=Counter()
        for case in ET.parse(bundle/command['xml']).iter('testcase'):
            status=('failed' if case.find('failure') is not None else 'errors' if case.find('error') is not None
                    else 'skipped' if case.find('skipped') is not None else 'passed')
            counts[status]+=1
            if command.get('count_in_final_matrix'):
                identity=(case.get('classname'),case.get('name'))
                assert identity not in unique,f'duplicate counted test: {identity}'
                unique[identity]=status
        groups[command['name']]=dict(counts)
    result={'unique_totals':dict(Counter(unique.values())),'groups':groups,'repeated_local_and_clone_tests_added_again':False}
    write_json(bundle/'tests/FINAL_TEST_ACCOUNTING.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu',type=int,default=5)
    p.add_argument('--phase',choices=['targeted','matrix','evidence'],required=True)
    p.add_argument('--artifact',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.phase=='targeted':
        run_group(args.output,'targeted',ROOT,sys.executable,TARGETED,cpu=args.cpu,counted=False)
    elif args.phase=='evidence':
        assert args.artifact and (args.artifact/'SHA256SUMS').is_file()
        run_group(args.output,'prepared_evidence',ROOT,sys.executable,
            ['experiments/repaired_solver_performance/tests'],cpu=args.cpu,counted=True,
            environment={'PREPARED_REPLAY_EVIDENCE':str(args.artifact.resolve())})
    else:
        fixtures(args.output)
        inherited=read_json(ROOT/'artifacts/runs/endpoint_roundoff_repair_20260908/tests/delivery/full_commands.json')
        assert len(inherited)==9
        for entry in inherited:
            name=entry['group'];root=OLD_STOP if name=='old_stop_snapshot' else ROOT
            argv=entry['argv'];start=argv.index('no:cacheprovider')+1
            paths=[a for a in argv[start:] if not a.startswith('--junitxml=')]
            run_group(args.output,name,root,argv[3],paths,cpu=args.cpu,counted=True,
                expected_sha=OLD_STOP_SHA if name=='old_stop_snapshot' else None)
        run_group(args.output,'repaired_parent_evidence',PARENT,sys.executable,
            ['experiments/endpoint_roundoff_repair/tests'],cpu=args.cpu,counted=True,expected_sha=BASE_SHA)
        write_json(args.output/'MATRIX_COMPLETED.json',{'source_sha':git(ROOT,'rev-parse','HEAD'),'inherited_groups':9,
            'historical_parent_artifact_group':True,'long_experiments_rerun':False})


if __name__=='__main__':main()
