"""Assemble completed measurements and derive the compact reviewable package."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from experiments.endpoint_roundoff_repair.frozen import ROOT
from experiments.repaired_solver_performance.compare import read,require
from experiments.repaired_solver_performance.diagnostics import flowstar_tables,profile_tables
from experiments.repaired_solver_performance.render import figures
from experiments.repaired_solver_performance.results import BASE,CANDIDATE,OLD_PACKAGE,PLANTS,REFERENCE,REPAIRED,RUNTIME,derive,write_csv
from experiments.repaired_solver_performance.verify import digest,replay_summary,write_manifest


def write_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def copy_new(source,target):
    """Raw measured bytes are copied once, or checked equal on assembly retry."""
    if source.is_dir():
        target.mkdir(parents=True,exist_ok=True)
        for item in sorted(source.iterdir()):copy_new(item,target/item.name)
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():require(digest(source)==digest(target),f'existing raw artifact changed: {target}')
        else:shutil.copy2(source,target)


def assemble(work,root,repository=ROOT):
    require((work/'production/COMPLETE.json').is_file(),'production sequence must complete before packaging')
    require((work/'matched_profiles/method.json').is_file(),'diagnostic sequence must complete before packaging')
    root.mkdir(parents=True,exist_ok=True);raw=root/'raw_minimal';raw.mkdir(exist_ok=True)
    for name in ('fresh_reference','production','matched_profiles','baseline_profile','replay_windows'):
        copy_new(work/name,raw/name)
    for name in ('candidate_preregistration.json','profile_parent.py','profile_current.py','capture_late.py','benchmark.log','matched_profiles.log'):
        copy_new(work/name,raw/name)
    for name in ('fresh_brusselator_endpoint_recomputation.json','fresh_van_der_pol_endpoint_recomputation.json','fresh_endpoint_recomputation.log','check_fresh_endpoints.py'):
        copy_new(work/name,raw/name)
    copy_new(work/'contention_observation.json',raw/'contention_observation.json')
    tests=root/'tests';tests.mkdir(exist_ok=True)
    copy_new(work/'full_matrix',tests/'full_matrix')
    copy_new(work/'startup',tests/'startup')
    for pattern in ('targeted_*.xml','targeted_*.log','targeted_*.exit','path_resolution.*','historical_fixture_reuse.json'):
        for path in work.glob(pattern):copy_new(path,tests/path.name)
    reused={}
    names=[f'{OLD_PACKAGE}/widths_full_prefix.csv',f'{OLD_PACKAGE}/raw_minimal/runtime_bridge.json']
    for run in ('vdp_full','brusselator_full','vdp_adaptive'):
        names.extend(f'{OLD_PACKAGE}/raw_minimal/{run}/{name}' for name in ('source.json','summary.json','bounds.csv','models.jsonl.gz','endpoint_audit.jsonl'))
    reused['endpoint_repaired']=dict(data_origin='REUSED_ENDPOINT_REPAIRED',scientific_sha='196a50e9131336d68df07ad0af353deca0092d19',
        package_commit=BASE,files={name:digest(repository/name) for name in names})
    names=[f'artifacts/runs/xiangru_adoption_20260907T032448Z/raw_minimal/native_{plant}/{name}' for plant in ('vdp','brusselator') for name in ('summary.json','models.jsonl.gz')]
    reused['flowstar']=dict(data_origin='REUSED_MATCHED_REFERENCE',scientific_sha='b85a3211748cb77b736fe4ad42ee02d8d2b81148',
        package_commit=BASE,files={name:digest(repository/name) for name in names})
    source=dict(schema='repaired_solver_performance_sources/1',base_commit=BASE,final_repaired_runtime=REPAIRED,
        full_reference_scientific=REFERENCE,production_candidate_scientific=CANDIDATE,tested_candidate_runtime=RUNTIME,
        branch='codex/repaired-solver-prepared-replay-performance-20260908T084224Z',
        candidate_runtime_files={str(p.relative_to(repository)):digest(p) for p in sorted((repository/'src/torch_tm_flowpipe').glob('*.py'))},
        frozen_execution_files={name:digest(repository/name) for name in ('experiments/endpoint_roundoff_repair/frozen.py',
            'experiments/run_vdp_dense_backend.py','experiments/run_brusselator_sr1000_parity.py',
            'artifacts/runs/xiangru_adoption_20260907T032448Z/MATCHED_CONTRACTS.json')},
        reused=reused,package_commit_is_scientific_source=False,
        note='Reference runtime is exactly final repaired0714. All new source runtime files at formalf627 equal testedae6. Later helper/report/package commits do not claim to be the source of earlier runs.')
    write_json(root/'SOURCE_MAP.json',source)
    context=read(work/'bootstrap.json')
    context.update(reference_source=str(work/'reference_source'),candidate_source=str(work/'candidate_source'),
        reference_scientific_sha=REFERENCE,candidate_scientific_sha=CANDIDATE,tested_runtime_sha=RUNTIME,
        torch='2.5.1+cu121',device='cpu',dtype='float64',interop_threads=1,formal_affinity=[3],diagnostic_affinity=[6],
        dependency_upgrades=False,old_worktree_files_modified=False,profile_times_used_as_speed_denominator=False,
        reference_order_note='The single fresh sequential reference also supplied missing complete checkpoints90/980, so it precedes matched short windows as authorized by goal5.1.')
    write_json(root/'RUN_CONTEXT.json',context)
    contracts={plant:read(raw/'fresh_reference'/plant/'execution_contract.json') for plant in PLANTS}
    contracts['van_der_pol_adaptive']=read(raw/'production/adaptive_van_der_pol_prepared_remainder_replay/execution_contract.json')
    write_json(root/'EXECUTION_CONTRACT.json',contracts)
    data=derive(root,repository);diagnostics=profile_tables(root);flowstar=flowstar_tables(root,repository)
    for name,key in [('timings_raw.csv','timings_raw'),('timing_summary.csv','timing_summary'),('full_width_equivalence.csv','widths')]:write_csv(root/name,data[key])
    for key in ('profile_windows','replay_work_counts','remaining_hotspots'):write_csv(root/f'{key}.csv',diagnostics[key])
    write_csv(root/'flowstar_reused_widths.csv',flowstar['widths']);write_csv(root/'flowstar_reused_summary.csv',flowstar['summary'])
    write_json(root/'full_run_summaries.json',data['full_summaries']);write_json(root/'RESULT.json',data['result'])
    write_json(raw/'archive_bridges.json',data['archive_bridges'])
    write_json(root/'same_input_replay_equivalence.json',replay_summary(root,recompute=False))
    decision=dict(mechanisms_implemented=['prepared_remainder_replay'],preregistration=read(raw/'candidate_preregistration.json'),
        measured_window_amdahl=diagnostics['amdahl'],full_observed_speedups={p:data['full'][p]['solve_speedup'] for p in PLANTS},
        interpretation='Amdahl estimates concern their exact windows. Full single-pair results and alternating repeats are reported separately. No second mechanism was added to chase1.5x.')
    write_json(root/'candidate_decision.json',decision)
    write_json(tests/'commands.json',dict(full_matrix=read(tests/'full_matrix/final_commands.json'),
        startup=read(tests/'startup/commands.json'),artifact_tests='Recorded in artifact_tests_command.json after execution',
        accounting='1094 passed /11 skipped in final full matrix. Targeted and independent clone repeats are not added again; six new distinct artifact tests are separate.'))
    figures(root,data,diagnostics,flowstar)
    write_manifest(root)
    return data['result']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(assemble(args.work_root,args.output),indent=2))
