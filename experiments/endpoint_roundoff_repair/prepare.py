"""Record the bounded source/call-site inventory for this one repair."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
PARENT='73c3b48a3dadd81cd03ecc827bde1228ef7f6ac8'


def call_sites():
    rows=[]
    for path in sorted((ROOT/'src').rglob('*.py')):
        def visit(node,context=()):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                context=(*context,node.name)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in {
                'substitute_const','substitute_const_with_roundoff','substitute_const_and_drop',
                'substitute_const_and_drop_with_roundoff','endpoint',
            }:
                name='.'.join(context)
                if name=='_final_range_boxes': purpose='仅诊断'
                elif path.name=='polynomial.py' or (path.name=='batched_dense_tm.py' and 'BatchedPolynomial.' in name):
                    purpose='点系数路径，其误差由同一调用者独立包住'
                elif 'dense_picard_validate_step' in name: purpose='验证器内部使用'
                elif path.name=='flowpipe.py': purpose='对外发布端点；normal左/右映射下一步输入'
                elif path.name=='brusselator_canonical_exchange.py': purpose='对外发布端点的只读证据导出'
                else: purpose='保证接口的内部调用'
                rows.append({'file':str(path.relative_to(ROOT)),'line':node.lineno,'caller':name,
                             'callee':node.func.attr,'purpose':purpose})
            for child in ast.iter_child_nodes(node): visit(child,context)
        visit(ast.parse(path.read_text()))
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scientific-sha',required=True)
    args=p.parse_args()
    git=lambda *a: subprocess.check_output(['git','-C',str(ROOT),*a],text=True).strip()
    assert git('rev-parse','HEAD')==args.scientific_sha and not git('status','--porcelain')
    old=json.loads((ROOT/'artifacts/runs/our_solver_replay_performance_20260907T074937Z/SOURCE_MAP.json').read_text())
    for rel,sha in old['reused_files'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha
    identities=[PARENT,old['base_sha'],old['previous_numerical_sha'],old['endpoint_driver_sha'],old['previous_audit_sha']]
    for sha in identities: assert git('cat-file','-t',sha)=='commit'
    native=Path('/srv/local/shengenli/flowstar')
    native_sha=old['native_flowstar_sha']
    assert subprocess.check_output(['git','-C',str(native),'cat-file','-t',native_sha],text=True).strip()=='commit'
    goal=Path('/srv/local/shengenli/codex/goal_vdp_terminal.md')
    value={'scientific_sha':args.scientific_sha,'parent_sha':PARENT,
           'branch':'codex/endpoint-roundoff-repair-and-revalidation-20260908',
           'source_root':str(ROOT),'source_clean_at_recording':True,
           'source_files':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted((ROOT/'src/torch_tm_flowpipe').glob('*.py'))},
           'old_numerical_sha':old['base_sha'],'old_optimized_sha':old['previous_numerical_sha'],
           'old_counterexample_driver_sha':old['endpoint_driver_sha'],'old_comparison_sha':old['previous_audit_sha'],
           'native_flowstar_sha':native_sha,'native_object_repository':str(native),
           'reused_files':old['reused_files'],'goal_sha256':hashlib.sha256(goal.read_bytes()).hexdigest(),
           'data_roles':{'old_cpu':'OLD_UNREPAIRED_ARCHIVE','new_cpu':'FRESH_ENDPOINT_REPAIRED','flowstar':'REUSED_MATCHED_REFERENCE'},
           'third_party_source_read':False,'scope':'Endpoint substitution and its direct carry/ledger consumers only'}
    with (args.output/'SOURCE_MAP.json').open('x') as f: json.dump(value,f,indent=2);f.write('\n')
    rows=call_sites()
    with (args.output/'endpoint_call_sites.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    raw=args.output/'raw_minimal';raw.mkdir(exist_ok=True)
    (raw/'GOAL_SPECIFICATION.md').write_bytes(goal.read_bytes())
    print(json.dumps({'scientific_sha':args.scientific_sha,'call_sites':len(rows),'reused_hashes_verified':len(old['reused_files'])}))


if __name__=='__main__': main()
