"""Verify saved arithmetic, matched timings and provenance without long solves."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import torch
from torch_tm_flowpipe import accepted_boundary_sr_queue_sha256,load_terminal_checkpoint
from torch_tm_flowpipe.batched_dense_tm import sparse_tmvector_to_dense
from torch_tm_flowpipe.terminal_checkpoint import tmvector_hashes
from experiments.endpoint_roundoff_repair.audit import decode,encode
from experiments.endpoint_roundoff_repair.frozen import ROOT
from experiments.xiangru_adoption.common import measure
from experiments.repaired_solver_performance.compare import exact,read,read_run,require
from experiments.repaired_solver_performance.results import BASE,CANDIDATE,OLD_PACKAGE,PLANTS,REFERENCE,REPAIRED,RUNTIME,WINDOWS,csv_rows,derive
from experiments.repaired_solver_performance.verify_replay import verify_replay_file
from experiments.repaired_solver_performance.diagnostics import flowstar_tables,profile_tables


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(root):
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (root/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(root)}\n' for p in files))


def verify_manifest(root):
    listed={}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        sha,name=line.split('  ',1)
        require(name not in listed and not Path(name).is_absolute() and '..' not in Path(name).parts,'unsafe or duplicate manifest path')
        listed[name]=sha
    actual={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS'}
    require(listed==actual,'package SHA256 manifest differs')
    return len(actual)


def git_blob(repository,sha,name):
    return subprocess.check_output(['git','-C',str(repository),'show',f'{sha}:{name}'])


def verify_sources(root,repository):
    source=read(root/'SOURCE_MAP.json')
    require(source['base_commit']==BASE and source['final_repaired_runtime']==REPAIRED,'fixed repaired parent identity')
    require(source['full_reference_scientific']==REFERENCE and source['production_candidate_scientific']==CANDIDATE,'fresh scientific identities')
    require(source['confirmation_scientific']==CANDIDATE and source['primary_full_timing_set']=='confirmation','primary full timing source identity')
    require(source['tested_candidate_runtime']==RUNTIME,'actual matrix runtime identity')
    require(source['branch']=='codex/repaired-solver-prepared-replay-performance-20260908T084224Z','new branch identity')
    require(source['reused']['endpoint_repaired']['data_origin']=='REUSED_ENDPOINT_REPAIRED','reused endpoint archive cannot become fresh')
    require(source['reused']['flowstar']['data_origin']=='REUSED_MATCHED_REFERENCE','reused Flow* cannot become fresh')
    for key,expected in [('endpoint_repaired','196a50e9131336d68df07ad0af353deca0092d19'),('flowstar','b85a3211748cb77b736fe4ad42ee02d8d2b81148')]:
        entry=source['reused'][key]
        require(entry['scientific_sha']==expected and entry['package_commit']==BASE,'reused scientific/package source identity')
        require(entry['files'],'empty reused evidence anchors')
        for name,sha in entry['files'].items():
            require(hashlib.sha256(git_blob(repository,BASE,name)).hexdigest()==sha==digest(repository/name),'reused bytes differ from committed parent')
    # Runtime bridge checks are identity checks, separate from arithmetic below.
    require(not subprocess.check_output(['git','-C',str(repository),'diff',REFERENCE,REPAIRED,'--','src/torch_tm_flowpipe']),'reference runtime differs from final endpoint repair')
    require(not subprocess.check_output(['git','-C',str(repository),'diff',RUNTIME,CANDIDATE,'--','src/torch_tm_flowpipe']),'production runtime differs from tested candidate')
    for name,sha in source['candidate_runtime_files'].items():
        require(hashlib.sha256(git_blob(repository,CANDIDATE,name)).hexdigest()==sha==digest(repository/name),'current verifier/runtime source changed')
    require(set(source['candidate_runtime_files'])=={str(p.relative_to(repository)) for p in (repository/'src/torch_tm_flowpipe').glob('*.py')},'runtime file identity coverage')
    for name,sha in source['frozen_execution_files'].items():
        require(hashlib.sha256(git_blob(repository,BASE,name)).hexdigest()==sha==digest(repository/name),'frozen setup/RHS/contract file changed')
    context=read(root/'RUN_CONTEXT.json')
    require(context['BASE']==BASE and context['reference_scientific_sha']==REFERENCE and context['candidate_scientific_sha']==CANDIDATE,'run context scientific identity')
    require(context['primary_full_reference_scientific_sha']==CANDIDATE and context['primary_full_timing_set']=='confirmation','primary matched source context')
    require(context['formal_affinity']==[3] and context['threads']==context['interop_threads']==1 and context['device']=='cpu' and context['dtype']=='float64','run context environment')
    require(not context['dependency_upgrades'] and not context['old_worktree_files_modified'],'run context scope')
    contracts={plant:read(root/'raw_minimal/fresh_reference'/plant/'execution_contract.json') for plant in PLANTS}
    contracts['van_der_pol_adaptive']=read(root/'raw_minimal/production/adaptive_van_der_pol_prepared_remainder_replay/execution_contract.json')
    require(read(root/'EXECUTION_CONTRACT.json')==contracts,'top-level effective configuration differs from raw runs')
    for path in [*(root/'raw_minimal/fresh_reference').glob('*/source.json'),*(root/'raw_minimal/production').glob('*/source.json'),*(root/'raw_minimal/confirmation').glob('*/source.json')]:
        run=read(path)
        for name,sha in run['source_files'].items():
            require(hashlib.sha256(git_blob(repository,run['scientific_sha'],name)).hexdigest()==sha,'actual recorded imported source file mismatch')
    return dict(parent=BASE,reference=REFERENCE,production_candidate=CANDIDATE,tested_runtime=RUNTIME,reused_labels_checked=True)


def verify_test_records(root):
    directory=root/'tests/full_matrix'
    commands=read(directory/'final_commands.json')
    require(len(commands)==9,'full matrix group coverage')
    totals=dict(passed=0,skipped=0,failed=0,errors=0)
    groups=[]
    for command in commands:
        group=command['group'];stem='root_rerun' if group=='root' else group
        require(command['exit_code']==int((directory/f'{stem}.exit').read_text())==0,'matrix process exit')
        require(command['source_clean_at_start'] and command['source_clean_at_end'],'dirty matrix source')
        expected='73c3b48a3dadd81cd03ecc827bde1228ef7f6ac8' if group=='old_stop_snapshot' else RUNTIME
        require(command['source_sha']==expected,'test snapshot ownership')
        suites=ET.parse(directory/f'{stem}.xml').getroot().iter('testsuite')
        row=dict(passed=0,skipped=0,failed=0,errors=0)
        for suite in suites:
            n=int(suite.get('tests',0));s=int(suite.get('skipped',0));f=int(suite.get('failures',0));e=int(suite.get('errors',0))
            row['passed']+=n-s-f-e;row['skipped']+=s;row['failed']+=f;row['errors']+=e
        for k,v in row.items():totals[k]+=v
        groups.append(dict(group=group,source_sha=expected,**row))
    require(totals==read(directory/'FINAL_TEST_ACCOUNTING.json')['unique_totals'],'matrix count derivation')
    require(totals==dict(passed=1094,skipped=11,failed=0,errors=0),'matrix completeness')
    return dict(unique_totals=totals,groups=groups,targeted_and_clone_repeats_not_added_again=True)


def verify_artifact_test_records(root):
    directory=root/'tests';command=read(directory/'artifact_tests_command.json')
    require(command['exit_code']==int((directory/'artifact_tests.exit').read_text())==0,'artifact test process failed')
    require(command['source_clean_at_start'] and command['source_clean_at_end'],'dirty artifact test source')
    require(read(directory/'commands.json')['artifact_tests']==command,'artifact test command record differs')
    counts=dict(tests=0,failures=0,errors=0,skipped=0)
    for suite in ET.parse(directory/'artifact_tests.xml').getroot().iter('testsuite'):
        for key in counts:counts[key]+=int(suite.get(key,0))
    require(counts==dict(tests=6,failures=0,errors=0,skipped=0),'six distinct semantic artifact cases must pass')
    return dict(passed=6,skipped=0,failed=0,errors=0,source_sha=command['source_sha'])


def verify_endpoints(path):
    """Reuse repaired endpoint operations; recompute every saved object and E.

    The previous Fraction/carry proofs and original safety tests are reused.
    This is an exact execution-equivalence replay, not a new solver proof.
    """
    run=read_run(path);order=run['contract']['config']['order'];cutoff=run['contract']['config']['cutoff']
    count=checkpoints=0
    with gzip.open(path/'models.jsonl.gz','rt') as stream:
        for row,audit,line in zip(run['bounds'],run['audits'],stream,strict=True):
            record=json.loads(line);h=float.fromhex(row['h_hex'])
            require(record['step']==int(row['step']) and record['t_end_exact']==row['t_end_exact'],'saved model step/time')
            tube=decode(record['models']['tube'],order)
            endpoint,errors=tube.substitute_const_with_roundoff(tube.n_vars-1,h)
            require([[float(e.lo).hex(),float(e.hi).hex()] for e in errors]==audit['substitution_error_hex'],'saved endpoint E recomputation')
            dense=sparse_tmvector_to_dense(tube,order=order)
            _,lo,hi,_,_=dense.poly.substitute_const_and_drop_with_roundoff(tube.n_vars-1,h,dense.domain_lo,dense.domain_hi)
            require([[float(lo[0,j]).hex(),float(hi[0,j]).hex()] for j in range(2)]==audit['internal_substitution_error_hex'],'saved dense endpoint E recomputation')
            endpoint=endpoint.drop_variable(endpoint.n_vars-1).apply_cutoff(cutoff)
            require(exact(encode(endpoint,'endpoint'))==exact(record['models']['endpoint']),'published endpoint/remainder replay')
            for kind,model in record['models'].items():
                for component,(lower,upper) in zip(('x','y'),measure(model),strict=True):
                    require(float(row[f'common_{kind}_{component}_lo']).hex()==lower.hex() and float(row[f'common_{kind}_{component}_hi']).hex()==upper.hex(),'common complete-object range replay')
                for component,bounds in zip(('x','y'),decode(model,order).range_box(),strict=True):
                    require(float(row[f'published_{kind}_{component}_lo']).hex()==float(bounds.lo).hex() and float(row[f'published_{kind}_{component}_hi']).hex()==float(bounds.hi).hex(),'published full range replay')
            checkpoint=path/f'checkpoint_{int(row["step"]):04d}'
            if checkpoint.is_dir():
                state=load_terminal_checkpoint(checkpoint,expected_contract=run['contract'],expected_order=order,expected_dtype='float64')
                require(state.scheduler['time_exact']==row['t_end_exact'],'checkpoint actual clock')
                require(accepted_boundary_sr_queue_sha256(state.normal_state.symbolic_queue)==audit['queue_sha256'],'checkpoint queue versus executed queue')
                for key,tm in [('reset_tm_hashes',state.current),('left_tm_hashes',state.normal_state.tmv_pre),('right_tm_hashes',state.normal_state.tmv_right)]:
                    require(exact(tmvector_hashes(tm))==exact(audit[key]),'checkpoint versus executed normal state')
                checkpoints+=1
            count+=1
    return dict(run=path.name,steps=count,endpoint_components=2*count,checkpoints=checkpoints,passed=True)


def replay_summary(root,*,recompute):
    directory=root/'raw_minimal/replay_windows';results=[]
    for plant,start in WINDOWS:
        path=directory/f'{plant}_{start:04d}.jsonl.gz'
        summary=read(path.with_suffix('.summary.json'))
        require(summary['source_sha'] in (RUNTIME,CANDIDATE),'actual replay scientific SHA')
        require(summary['start_step']==start and summary['steps']==20,'representative replay coverage')
        result=verify_replay_file(path,recompute=recompute)
        audits=read_run(root/'raw_minimal/fresh_reference'/plant)['audits']
        with gzip.open(path,'rt') as stream:
            for line in stream:
                row=json.loads(line);audit=audits[row['step']-1]
                require([[e[k]['values'] for k in ('lo','hi')] for e in row['endpoint_error']]==audit['substitution_error_hex'],'replay endpoint E versus actual full run')
                require(len(row['reference_proposals'])==audit['refinement_counters']['post_accept_replay_calls'],'replay count versus full run')
        # Stored summary has no switch whose value depends on verifier mode.
        result.pop('recomputed');results.append(result)
    return dict(windows=results,steps=sum(r['steps'] for r in results),proposals=sum(r['proposals'] for r in results),
                same_input_and_independent_loops_bit_identical=True,full_E_and_replay_counts_linked=True)


def csv_equal(path,expected):
    actual=csv_rows(path)
    normalized=[{k:'' if v is None else str(v) for k,v in row.items()} for row in expected]
    require(actual==normalized,f'derived table differs: {path.name}')


def verify(root,*,recompute=True,repository=ROOT):
    root=Path(root);repository=Path(repository)
    file_count=verify_manifest(root)
    identity=verify_sources(root,repository)
    tests=verify_test_records(root)
    # Cheap structural and cross-run checks precede the bounded evaluator work.
    data=derive(root,repository)
    for name,key in [('timings_raw.csv','timings_raw'),('timing_summary.csv','timing_summary'),('full_width_equivalence.csv','widths')]:
        csv_equal(root/name,data[key])
    require(exact(read(root/'full_run_summaries.json'))==exact(data['full_summaries']),'full result derivation')
    require(exact(read(root/'RESULT.json'))==exact(data['result']),'final status must follow measured data')
    require(exact(read(root/'raw_minimal/archive_bridges.json'))==exact(data['archive_bridges']),'archive runtime bridge')
    diagnostics=profile_tables(root)
    require(read(root/'raw_minimal/matched_profiles/method.json')['driver_sha256']==digest(root/'raw_minimal/profile_current.py'),'diagnostic driver identity')
    for key in ('profile_windows','replay_work_counts','remaining_hotspots'):
        csv_equal(root/f'{key}.csv',diagnostics[key])
    decision=read(root/'candidate_decision.json')
    require(decision['mechanisms_implemented']==['prepared_remainder_replay'],'more than one performance mechanism')
    require(exact(decision['measured_window_amdahl'])==exact(diagnostics['amdahl']),'Amdahl measured slice derivation')
    prereg=read(root/'raw_minimal/candidate_preregistration.json')
    require(decision['preregistration']==prereg and prereg['recorded_before_runtime_implementation'],'original candidate decision changed')
    f=prereg['selected_fraction'];s=prereg['preliminary_local_speedup_estimate_including_preparation']
    require(prereg['predicted_total_speedup']==1/((1-f)+f/s) and prereg['max_possible_speedup']==1/(1-f),'preregistered Amdahl arithmetic')
    require(decision['full_observed_speedups']=={p:data['full'][p]['solve_speedup'] for p in PLANTS},'candidate full observed timing')
    flowstar=flowstar_tables(root,repository)
    csv_equal(root/'flowstar_reused_widths.csv',flowstar['widths'])
    csv_equal(root/'flowstar_reused_summary.csv',flowstar['summary'])
    replays=replay_summary(root,recompute=recompute)
    require(read(root/'same_input_replay_equivalence.json')==replays,'saved replay count/coverage')
    endpoints=[]
    if recompute:
        for plant in PLANTS:endpoints.append(verify_endpoints(root/'raw_minimal/fresh_reference'/plant))
        endpoints.append(verify_endpoints(root/'raw_minimal/production/adaptive_van_der_pol_prepared_remainder_replay'))
    artifact_tests=verify_artifact_test_records(root)
    tests['additional_artifact_cases']=artifact_tests
    tests['unique_totals_including_artifact_cases']=dict(passed=tests['unique_totals']['passed']+artifact_tests['passed'],skipped=tests['unique_totals']['skipped'],failed=0,errors=0)
    return dict(passed=True,files=file_count,identity=identity,tests=tests,replay=replays,
                exact_saved_endpoint_recomputations=endpoints,full_math_recomputed=recompute,
                status=data['result']['status'],fresh_full_pairs=2,short_matched_pairs=21,long_ODE_solves_rerun=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--structural-only',action='store_true',help='skip evaluator and endpoint recomputation; intended for tamper tests')
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    result=verify(args.root,recompute=not args.structural_only)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
