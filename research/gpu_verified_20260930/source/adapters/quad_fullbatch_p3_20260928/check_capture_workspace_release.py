"""Original weighted-input checker with post-warmup unused-cache release.

Run its four independent owners unchanged, then compare each complete arm
with the archived original512/weighted128 result. No SR, advance or NN.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback

HERE=Path(__file__).parent
CHECK_SHA='2e81fdd59833fb395521836ccc831e4822f6d98842dc8bfab32bc50b6bc46aa0'
ADAPTER_SHA='259c479a113bb1c905da7c49ae888ba50002ec84c06acfbb253bd9b45edfd5ee'
REFERENCE_SHA='141c3f8dc285d7ecee34afc0195401c2b3cb702997af0dd6492f491a91d1aa08'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p,digest):
    assert sha(p)==digest
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
q=load('cache_release_frozen_weighted_check',HERE/'check_weighted_chunk128.py',CHECK_SHA)


def reference(a):
    assert sha(a.reference/'RESULT.json')==REFERENCE_SHA
    r=q.read(a.reference/'RESULT.json')
    assert r['status']=='passed' and r['device']=='cuda:0' and r['script_sha256']==CHECK_SHA
    assert r['adapter_sha256']==q.ADAPTER_SHA and r['source_payload_sha256']==q.PAYLOAD_SHA and r['source_audit_sha256']==q.AUDIT_SHA
    assert r['source_capture_passed'] is False and r['source_run_is_trajectory_qualification'] is False
    assert len(r['arms'])==4 and len(r['parity_cases'])==2
    files={'RESULT.json':REFERENCE_SHA}
    by_path={v['path']:v['result_sha256'] for case in r['parity_cases'] for v in case['arms']}
    for arm,(count,chunk) in zip(r['arms'],[(1024,512),(1024,128),(133,512),(133,128)]):
        name=f'eligible{count}_chunk{chunk}';p=a.reference/name
        assert arm==q.read(p/'RESULT.json') and sha(p/'RESULT.json')==by_path[name]
        files[name+'/RESULT.json']=by_path[name]
        assert (arm['eligible_count'],arm['input_rows'],arm['chunk_size'])==(count,1024,chunk)
        for filename,digest in arm['files_sha256'].items():
            assert sha(p/filename)==digest;files[name+'/'+filename]=digest
        assert set(arm['files_sha256'])=={'OUTPUT.pt','EVENTS.json','EAGER_EVENTS.json'}
    return r,files


def compare(a,result,old,torch):
    rows=[]
    for new,prior in zip(result['arms'],old['arms']):
        name=f"eligible{new['eligible_count']}_chunk{new['chunk_size']}";left=a.reference/name;right=a.output/name
        x,y=(torch.load(p/'OUTPUT.pt',map_location='cpu',weights_only=True) for p in [left,right])
        assert x.keys()==y.keys() and all(q.same(torch,x[k],y[k]) for k in x)
        for filename in ['EVENTS.json','EAGER_EVENTS.json']:assert q.exact(q.read(left/filename),q.read(right/filename))
        for field in ['statistics','eager_stats','map_calls','eligible_lanes','graph','counters']:
            assert q.exact(new[field],prior[field]),(name,field)
        for field in ['input_bytes_unchanged','owned_outputs_survive_replay','every_target_map_represented_in_events','ineligible_remainders_unchanged']:
            assert new[field] is prior[field] is True
        rows.append(dict(arm=name,all_three_output_tensors_bytes_equal=True,all_events_float_bytes_equal=True,
            statistics_equal=True,eager_statistics_equal=True,all_map_image_bad_bytes_equal=True,
            graph_counts_equal=True,input_bytes_unchanged=True,owned_outputs_survive_replay=True))
    return rows


def experiment(a):
    old,files=reference(a);adapter=load('postwarm_cache_release_candidate',HERE/'capture_workspace_release.py',ADAPTER_SHA)
    raw_load=q.load;bindings=[];contexts=[];result=None
    def entry_load(name,path):
        module=raw_load(name,path)
        if name=='weighted128_frozen_entry':
            raw_initialize=module.initialize
            def initialize(args):
                ctx=raw_initialize(args)
                from flowstar_gpu import graphing
                assert not bindings,'Install once, after original bootstrap and before any engine'
                binding=adapter.install(ctx.torch,graphing)
                bindings.append(binding);contexts.append(ctx)
                return ctx
            module.initialize=initialize
        return module
    q.load=entry_load
    try:
        result=q.experiment(a)
        assert len(bindings)==len(contexts)==1
        ctx=contexts[0];binding=bindings[0]
        comparisons=compare(a,result,old,ctx.torch)
        assert len(binding.releases)==sum(arm['graph']['captures'] for arm in result['arms'])>0
        for field in ['engine_python_sha256','extensions','reciprocal_extensions','reciprocal_candidate',
            'hybrid_adapter_sha256','metadata_backend_sha256','torch_version']:
            assert result[field]==old[field],field
        for name,digest in files.items():assert sha(a.reference/name)==digest
        result.update(weighted_adapter_sha256=result.pop('adapter_sha256'),adapter_sha256=ADAPTER_SHA,
            inherited_checker_sha256=CHECK_SHA,reference_result_sha256=REFERENCE_SHA,reference_files_sha256=files,
            runtime_policy=binding.policy,algorithm_identity=ctx.identity,postwarm_releases=list(binding.releases),
            reference_arm_comparisons=comparisons,
            scope='Post-warmup unused-cache release only. Original four fixed-input arms reused unchanged; complete output/events/stats match archived no-release arms. Allocator counters are observations, not value proofs. No SR/NN/advance/full-trajectory or performance qualification; source capture remains resource-terminated.')
        return result
    finally:
        q.load=raw_load
        for binding in bindings:binding.restore()
        if result is not None:result['cache_hook_restored']=True


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception')
    try:
        if a.check_local:
            import torch
            old,files=reference(a);p,audit,identity=q.source(a,torch)
            result=dict(status='passed_reference_and_payload_contract',reference_result_sha256=REFERENCE_SHA,
                reference_files_sha256=files,source_payload_sha256=q.PAYLOAD_SHA,source_audit_sha256=q.AUDIT_SHA,
                CUDA_run=False,numerical_map_run=False,source_capture_passed=False)
        else:result=experiment(a)
    except BaseException as e:result.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());raise
    finally:
        result.update(script_sha256=sha(__file__),process_s=time.perf_counter()-start)
        q.write(a.output/'RESULT.json',result);print(json.dumps({k:result[k] for k in ['status','process_s']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check-local',action='store_true')
    for name in ['capture','capture-audit','reference','output']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check',
        'host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend','metadata-check','hybrid-check']:
        p.add_argument('--'+name,type=Path)
    a=p.parse_args();a.mode='postwarm_cache_fixed_input_check';a.source40=None;main(a)
