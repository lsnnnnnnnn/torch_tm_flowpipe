"""Read-only CPU comparison of archived P3 dense40 and metadata40 results.

Every-step hull/acceptance and both NN certificates are available. Complete
state/SR comparisons cover the five checkpoints actually saved by the dense
runner; this does not fabricate unsaved full states for the other steps.
"""
from pathlib import Path
import argparse, hashlib, json, time
import torch

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def equal(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(
            a.detach().cpu().contiguous().view(torch.uint8),b.detach().cpu().contiguous().view(torch.uint8))
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b

def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    result=dict(status='exception',mode='p3_dense_sparse_parity',script_sha256=sha(__file__))
    try:
        paths=[args.dense,args.sparse];meta=[read(p/'INPUT.json') for p in paths];done=[read(p/'RESULT.json') for p in paths]
        for p,m,d in zip(paths,meta,done):
            assert d['status']=='target_completed' and d['completed_step']==40 and d['input_sha256']==sha(p/'INPUT.json')
            assert m['mode']=='split_from_zero_strict_endpoint' and m['root_id']==1 and m['leaf_count']==2
            assert m['working_total_degree']==m['spatial_composition_order']==3 and m['validation_order']==4
            assert m['point_code_order']==2 and m['controller_policy']=='union_box'
        assert meta[0]['script_sha256']==done[0]['script_sha256']=='8775959155cebf4b7314ff828bb846f1754f9e53cd5bd41907ee7d7c8c6efc0d'
        assert meta[1]['script_sha256']==done[1]['script_sha256']==sha(Path(__file__).with_name('run_quad_metadata.py'))
        assert meta[1]['metadata_backend_sha256']==done[1]['metadata_backend_sha256']==sha(Path(__file__).with_name('metadata_backend.py'))
        cold=read(args.cold/'RESULT.json')
        assert cold['status']=='passed' and cold['mode']=='cold_boundary_replay' and cold['working_order']==3
        assert cold['script_sha256']==meta[1]['script_sha256'] and cold['source_input_sha256']==sha(args.sparse/'INPUT.json')
        assert cold['source_result_sha256']==sha(args.sparse/'RESULT.json')
        assert cold['all_endpoint_bytes_match'] and cold['all_controller_bytes_match'] and cold['all_step21_bytes_match']
        algorithm_env=[{k:v for k,v in m['environment'].items() if k.startswith('FLOWSTAR_')} for m in meta]
        assert algorithm_env[0]==algorithm_env[1],('FLOWSTAR policy environment differs',algorithm_env)
        same=['config_sha256','model_sha256','driver_sha256','boxes_sha256','engine','extensions','coverage','initial_affine_coverage',
              'settings','dynamics','variable_names','sr_queue','controller_policy','working_total_degree','point_code_order','validation_order']
        assert all(meta[0][k]==meta[1][k] for k in same)
        steps=[[json.loads(line) for line in (p/'steps.jsonl').read_text().splitlines()] for p in paths]
        assert [[row['step'] for row in group] for group in steps]==[list(range(1,41))]*2
        step_checks=[]
        for a,b in zip(*steps):
            checks={k:equal(torch.tensor(a[k],dtype=torch.float64),torch.tensor(b[k],dtype=torch.float64))
                    for k in ['tube','endpoint','union_tube','union_endpoint']}
            for key in ['accepted','root_covered','parent_state_sr_unchanged','trial_sr_length','held_controller_step','sr_reset_if_full','committed_sr_length']:
                checks[key]=a[key]==b[key]
            assert all(checks.values()),(a['step'],checks);step_checks.append(dict(step=a['step'],checks=checks))
        controller_checks=[]
        for step in [0,20]:
            values=[torch.load(p/f'controller_at_{step}.pt',map_location='cpu',weights_only=True) for p in paths]
            checks={key:equal(values[0][key],values[1][key]) for key in set(values[0])|set(values[1])}
            assert all(checks.values()),('controller',step,checks)
            for root,summary in zip(paths,done):
                held=[v for v in summary['controllers'] if v['completed_step']==step];assert len(held)==1
                assert held[0]['sha256']==sha(root/f'controller_at_{step}.pt')
            controller_checks.append(dict(step=step,checks=checks))
        checkpoints=['committed_20','after_strict_endpoint_20','after_controller_20','committed_21','committed_40'];state_checks=[]
        for name in checkpoints:
            side=[read(p/(name+'.json')) for p in paths]
            for root,m,s in zip(paths,meta,side):
                assert s['metadata']==m and s['state_sha256']==sha(root/(name+'.pt'))
            assert side[0]['state_fingerprint']==side[1]['state_fingerprint']
            for key in ['completed_step','next_step','phase','next_control_refresh_step','sr_history_origin_step','qlen','jlen']:
                assert side[0][key]==side[1][key],(name,key)
            values=[torch.load(p/(name+'.pt'),map_location='cpu',weights_only=True) for p in paths]
            checks={key:equal(values[0][key],values[1][key]) for key in set(values[0])|set(values[1])}
            assert all(checks.values()),(name,checks);state_checks.append(dict(checkpoint=name,checks=checks))
        endpoint=[torch.load(p/'endpoint_at_20.pt',map_location='cpu',weights_only=True) for p in paths]
        assert equal(*endpoint)
        for root in paths:assert sha(root/'endpoint_at_20.pt')==read(root/'endpoint_at_20.json')['error_sha256']
        result.update(status='passed',dense_input_path=str(args.dense/'INPUT.json'),sparse_input_path=str(args.sparse/'INPUT.json'),
            dense_input_sha256=sha(args.dense/'INPUT.json'),sparse_input_sha256=sha(args.sparse/'INPUT.json'),
            dense_result_sha256=sha(args.dense/'RESULT.json'),sparse_result_sha256=sha(args.sparse/'RESULT.json'),
            sparse_runner_sha256=meta[1]['script_sha256'],metadata_backend_sha256=meta[1]['metadata_backend_sha256'],
            cold_result_path=str(args.cold/'RESULT.json'),cold_result_sha256=sha(args.cold/'RESULT.json'),cold20_to21_all_bytes_equal=True,
            same_flowstar_environment=algorithm_env[0],
            same_contract_fields=same,step_checks=step_checks,controller_checks=controller_checks,state_checks=state_checks,
            endpoint_error_bytes_equal=True,full_state_checkpoint_count=5,
            limitation='Complete state/live SR checked at the five actually saved checkpoints only; all40 step hulls and acceptance checked.')
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result['process_s']=time.perf_counter()-started;(out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['step_checks','controller_checks','state_checks']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dense',type=Path,required=True)
    p.add_argument('--sparse',type=Path,required=True);p.add_argument('--cold',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
