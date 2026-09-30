"""Fixed real first-step inputs: original512 vs isolated weighted128 graphs.

The source capture hit its memory guard after saving an independently audited
payload. It is not a completed advance. This check has no plant/SR/NN runtime.
"""
from pathlib import Path
import argparse, dataclasses, gc, hashlib, importlib.util, json, struct, sys, time, traceback

HERE=Path(__file__).parent
ENTRY_SHA='22f4dda16dcfd30563e6652ec8af82034b3b138a7505dbf31714a8bd55820a72'
ADAPTER_SHA='ae343c546e2253896142f88c39e1df5e43561498aaf7540e722a1bb6d66d409b'
CAPTURE_SHA='91cb4b9fcf1a7531cf5a2d89eba4ff02018140b6f8b83e7ad52d5f84ea17d4b5'
PAYLOAD_SHA='895bad79fc31e5f6eefeb89f98e3401f0d7e2c2ec79a6ef4b4ddf678bed4fbe2'
AUDIT_SHA='5050e737a81fb4dcd023880df9cb6becf047a173536bfbb085a6f39fc48456f4'
AUDITOR_SHA='649acec83bca42979c92bf412a67617461f8e4eab9ac0e4d2a96806e1ba3846c'
TENSORS=('x','initial','original','current','eligible')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def code_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def same(torch,a,b):
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().cpu().contiguous().view(torch.uint8),b.detach().cpu().contiguous().view(torch.uint8))
def exact(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,float):return struct.pack('<d',a)==struct.pack('<d',b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b
def fingerprint(torch,x):
    x=x.detach().cpu().contiguous()
    return dict(shape=list(x.shape),dtype=str(x.dtype),bytes=x.numel()*x.element_size(),
        sha256=hashlib.sha256(memoryview(x.view(torch.uint8).numpy()).cast('B')).hexdigest())


def source(a,torch):
    assert sha(a.capture_audit/'AUDIT.json')==AUDIT_SHA
    audit=read(a.capture_audit/'AUDIT.json');inp=read(a.capture/'INPUT.json')
    assert audit['status']=='passed' and audit['classification']=='complete_payload_from_resource_terminated_capture'
    assert audit['source_capture_status']=='gpu_guard_11.5GiB' and audit['source_capture_passed'] is False
    assert audit['suitable_for_fixed_input_comparison'] and audit['suitable_for_trajectory_admission'] is False
    assert audit['source_result_missing'] and audit['source_sidecar_missing']
    assert not (a.capture/'RESULT.json').exists() and not (a.capture/'WEIGHTED_INPUT.json').exists()
    assert audit['audit_script_sha256']==AUDITOR_SHA and audit['source_capture_sha256']==CAPTURE_SHA
    assert audit['source_entry_sha256']==ENTRY_SHA and audit['source_pt_sha256']==PAYLOAD_SHA
    assert sha(a.capture/'WEIGHTED_INPUT.pt')==PAYLOAD_SHA and (a.capture/'WEIGHTED_INPUT.pt').stat().st_size==audit['source_pt_size_bytes']==12269070
    assert sha(a.capture/'INPUT.json')==audit['source_input_sha256']
    assert inp['source_identity']==audit['source_identity']
    assert inp['source_identity']['algorithm_identity']['script_sha256']==ENTRY_SHA
    p=torch.load(a.capture/'WEIGHTED_INPUT.pt',map_location='cpu',weights_only=True)
    assert p['n']==16 and p['batch']==1024 and p['working_order']==3 and p['point_code_order']==2
    assert p['validation_order']==p['support_family']==4 and p['h']==.005 and p['cutoff']==1e-6
    assert p['rounds']==2 and p['chunk_size']==512 and p['use_graph'] and p['trace_present'] is False
    assert p['attempted_step']==1 and p['held_controller_step']==0
    assert code_hash(p['code'])==p['code_sha256']==audit['code_sha256']
    tensors={k:p[k] for k in TENSORS};tensors.update(support_exponents=p['support_exponents'],initial_exponents=p['initial_exponents'])
    tensors.update({f'variable_exponents_{i}':v for i,v in enumerate(p['variable_exponents'])})
    # The independent archive audit includes every tensor, not just the five inputs.
    assert {k:fingerprint(torch,v) for k,v in tensors.items()}==audit['tensors']
    for name in ('x','initial','original','current'):
        assert p[name].dtype==torch.float64 and p[name].device.type=='cpu' and bool(torch.isfinite(p[name]).all())
    assert p['x'].shape==(1024,16,72) and p['initial'].shape==(1024,16,17)
    assert p['original'].shape==p['current'].shape==(1024,16,2)
    assert p['eligible'].dtype==torch.bool and p['eligible'].shape==(1024,) and int(p['eligible'].sum())==1024
    assert bool((p['original'][...,0]==-.1).all() and (p['original'][...,1]==.1).all())
    assert bool((p['current'][...,0]<=0).all() and (p['current'][...,1]>=0).all())
    assert bool((p['current'][...,0]>=-.1).all() and (p['current'][...,1]<=.1).all())
    assert len(p['variable_support_ids'])==len(p['variable_exponents'])==16
    for ids,exps in [(p['support_ids'],p['support_exponents']),(p['initial_ids'],p['initial_exponents']),*zip(p['variable_support_ids'],p['variable_exponents'])]:
        assert ids==sorted(set(ids)) and exps.dtype==torch.int64 and exps.shape==(len(ids),17)
        assert bool((exps>=0).all()) and int(exps.sum(-1).max())<=3
    assert all(set(ids)<=set(p['support_ids']) for ids in p['variable_support_ids'])
    zeros=[i for i,e in enumerate(p['support_exponents'].tolist()) if e[0]==0]
    zexp=[tuple(p['support_exponents'][i].tolist()) for i in zeros];expected=torch.zeros_like(p['x'][:,:,zeros])
    for j,e in enumerate(p['initial_exponents'].tolist()):
        assert e[0]==0;expected[:,:,zexp.index(tuple(e))]=p['initial'][:,:,j]
    assert same(torch,p['x'][:,:,zeros],expected)
    return p,audit,inp['source_identity']['algorithm_identity']


def arm(ctx,adapter,p,eligible_count,use_adapter,outdir):
    """One owner; no graph, input or output GPU reference escapes this function."""
    torch,d=ctx.torch,ctx.d
    from flowstar_gpu import support as sp,sparse_exec as se,weighted_validation as wv,monomials
    tables=d.build_tables(16,3).to('cuda:0');work=sp.SparseEngine(tables,d.poly.build_step_tables(tables,.005),'cuda:0')
    work.horner_edge_kernel=ctx.edge.horner_edge
    assert type(work) is sp.SparseEngine and isinstance(work.tables,monomials.MonomialTables)
    assert work.tables.k==3 and work.tables.T2==100947
    val=se.validation_engine_for_order(work,4);assert isinstance(val,ctx.hybrid.backend.MetadataEngine)
    names=[v['name'] for v in ctx.cfg['initial_set']];code2=d.compile_ode(ctx.cfg['dynamics_expressions'],names,order=2)
    code4=se.validation_code_for_order(code2,4,val);assert code_hash(dataclasses.asdict(code4))==p['code_sha256']
    sup=sp.make_support(16,4,False,tuple(p['support_ids']));isup=sp.make_support(16,4,False,tuple(p['initial_ids']))
    variables=tuple(sp.make_support(16,4,False,tuple(ids)) for ids in p['variable_support_ids'])
    for support,exps in [(sup,p['support_exponents']),(isup,p['initial_exponents']),*zip(variables,p['variable_exponents'])]:
        assert same(torch,torch.tensor(val.exponents(support).copy(),dtype=torch.int64),exps)
    inputs={k:p[k].to('cuda:0',copy=True) for k in TENSORS}
    if eligible_count==133:
        inputs['eligible'].zero_();inputs['eligible'][torch.where(p['eligible'])[0][::7][:133].to('cuda:0')]=True
    assert int(inputs['eligible'].sum())==eligible_count
    before={k:v.detach().cpu().clone() for k,v in inputs.items()};versions={k:v._version for k,v in inputs.items()}
    original_evaluate,original_refine=wv._evaluate_map,wv.refine_accepted
    binding=adapter.install(torch,wv,se) if use_adapter else None
    actual_evaluate=wv._evaluate_map;calls=[];previous=None;previous_cpu=None;ownership_replays=0;events=[]
    def tracked(*args,**kwargs):
        nonlocal previous,previous_cpu,ownership_replays
        returned=actual_evaluate(*args,**kwargs)
        if previous is not None:
            assert all(same(torch,x,y) for x,y in zip(previous,previous_cpu));ownership_replays+=1
        cache=val._weighted_graphs
        statics=next(iter(cache._segs.values()))[2]
        assert all(x.untyped_storage().data_ptr()!=y.untyped_storage().data_ptr() for x,y in zip(returned,statics))
        previous=returned;previous_cpu=tuple(v.detach().cpu().clone() for v in returned)
        calls.append(dict(rows=int(args[3].shape[0]),image=fingerprint(torch,returned[0]),bad=fingerprint(torch,returned[1])))
        return returned
    try:
        wv._evaluate_map=tracked
        torch.cuda.synchronize();start=time.perf_counter()
        output,statistics=wv.refine_accepted(code4,inputs['x'],sup,inputs['initial'],isup,variables,
            inputs['original'],inputs['current'],inputs['eligible'],val,1e-6,rounds=2,chunk_size=512,use_graph=True,trace=events.append)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        assert len(events)==sum(x['rows'] for x in calls) and events
        assert all(e['event']=='early_weighted_self_map' and e['attempt']==0 and 0<=e['refinement_round']<2
            and bool(before['eligible'][e['lane']]) and len(e['proposal_interval'])==len(e['input_remainder_vector'])==len(e['subset_by_component'])==16 for e in events)
        # With max_attempts=1 every returned map row is logged by validate_failed.
        cursor=0
        for call in calls:
            block=events[cursor:cursor+call['rows']];cursor+=call['rows']
            assert fingerprint(torch,torch.tensor([e['proposal_interval'] for e in block],dtype=torch.float64))==call['image']
            assert fingerprint(torch,torch.tensor([e['bad'] for e in block],dtype=torch.bool))==call['bad']
        assert all(same(torch,v,before[k]) and v._version==versions[k] for k,v in inputs.items())
        cache=val._weighted_graphs;graph_rows=[int(v[0][0].shape[0]) for v in cache._segs.values()]
        assert cache.enabled and cache.captures>=1 and cache.hits==len(calls) and graph_rows==[128 if use_adapter else 512]
        assert all(same(torch,x,y) for x,y in zip(previous,previous_cpu))
        counters=dict(binding.counters) if binding is not None else None
        if binding is not None:
            assert counters['refine_calls']==1 and counters['graph_map_calls']==len(calls)
            assert counters['target_rows']==len(events)
            assert counters['padded_rows']==sum(128-x['rows'] for x in calls)
            if eligible_count==133:assert counters['padded_rows']>0
        cpu_output=output.detach().cpu().clone();assert same(torch,output[~inputs['eligible']],inputs['current'][~inputs['eligible']])
        # Actual eager32 delegation with the adapter installed but outside refine.
        # Its full max4 failed-validation path remains the original function.
        wv._evaluate_map=actual_evaluate;eager_events=[]
        mask=torch.zeros_like(inputs['eligible']);mask[torch.where(inputs['eligible'])[0][:32]]=True
        graph_hits=cache.hits;counter_before=dict(binding.counters) if binding is not None else None
        ok,rem,eager_stats=wv.validate_failed(code4,inputs['x'],sup,inputs['initial'],isup,variables,
            inputs['original'],mask,val,1e-6,max_attempts=4,chunk_size=32,use_graph=False,trace=eager_events.append)
        assert cache.hits==graph_hits and (binding is None or binding.counters==counter_before)
        assert all(same(torch,v,before[k]) and v._version==versions[k] for k,v in inputs.items())
        # The graph outputs and returned refinement remain owned after eager work.
        assert same(torch,output,cpu_output) and all(same(torch,x,y) for x,y in zip(previous,previous_cpu))
        outdir.mkdir();torch.save(dict(remainder=cpu_output,eager_accepted=ok.cpu(),eager_remainder=rem.cpu()),outdir/'OUTPUT.pt')
        write(outdir/'EVENTS.json',events);write(outdir/'EAGER_EVENTS.json',eager_events)
        row=dict(eligible_count=eligible_count,input_rows=1024,eligible_lanes=torch.where(before['eligible'])[0].tolist(),chunk_size=128 if use_adapter else 512,
            graph=dict(enabled=cache.enabled,captures=cache.captures,hits=cache.hits,rows=graph_rows),
            map_calls=calls,event_count=len(events),statistics=statistics,counters=counters,
            input_bytes_unchanged=True,owned_outputs_survive_replay=True,ownership_replay_checks=ownership_replays,
            every_target_map_represented_in_events=True,ineligible_remainders_unchanged=True,eager_stats=eager_stats,
            elapsed_s_diagnostic_only=elapsed,files_sha256={n:sha(outdir/n) for n in ['OUTPUT.pt','EVENTS.json','EAGER_EVENTS.json']})
        write(outdir/'RESULT.json',row)
        return row
    finally:
        wv._evaluate_map=actual_evaluate
        if binding is not None:binding.restore()
        assert wv._evaluate_map is original_evaluate and wv.refine_accepted is original_refine


def experiment(a):
    assert sha(HERE/'run_fullbatch_p3.py')==ENTRY_SHA and sha(HERE/'weighted_chunk128.py')==ADAPTER_SHA
    w=load('weighted128_frozen_entry',HERE/'run_fullbatch_p3.py');ctx=w.initialize(a);torch=ctx.torch
    p,audit,identity=source(a,torch);assert ctx.identity==identity
    adapter=load('weighted128_candidate',HERE/'weighted_chunk128.py');rows=[];cases=[]
    for count in [1024,133]:
        arm_paths=[]
        for chunk in [512,128]:
            path=a.output/f'eligible{count}_chunk{chunk}';arm_paths.append(path)
            rows.append(arm(ctx,adapter,p,count,chunk==128,path))
            # Returned metadata is CPU/JSON only; release the previous owner/graph.
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        left,right=(torch.load(p/'OUTPUT.pt',map_location='cpu',weights_only=True) for p in arm_paths)
        assert left.keys()==right.keys() and all(same(torch,left[k],right[k]) for k in left)
        assert exact(read(arm_paths[0]/'EVENTS.json'),read(arm_paths[1]/'EVENTS.json'))
        assert exact(read(arm_paths[0]/'EAGER_EVENTS.json'),read(arm_paths[1]/'EAGER_EVENTS.json'))
        assert exact(rows[-2]['statistics'],rows[-1]['statistics']) and exact(rows[-2]['eager_stats'],rows[-1]['eager_stats'])
        cases.append(dict(eligible_count=count,input_rows=1024,output_bytes_equal=True,statistics_equal=True,
            input_bytes_unchanged=True,owned_outputs_survive_replay=True,all_target_map_bytes_equal=True,
            events_float_bytes_equal=True,eager_failed32_bytes_equal=True,
            arms=[dict(path=p.name,result_sha256=sha(p/'RESULT.json')) for p in arm_paths]))
    # Audit metadata is immutable provenance, not a replacement source success.
    assert sha(a.capture/'WEIGHTED_INPUT.pt')==PAYLOAD_SHA and sha(a.capture_audit/'AUDIT.json')==AUDIT_SHA
    assert sha(HERE/'weighted_chunk128.py')==ADAPTER_SHA
    return dict(status='passed',device='cuda:0',adapter_sha256=ADAPTER_SHA,source_entry_sha256=ENTRY_SHA,
        source_capture_sha256=CAPTURE_SHA,source_payload_sha256=PAYLOAD_SHA,source_audit_sha256=AUDIT_SHA,
        source_payload_complete=True,source_capture_status=audit['source_capture_status'],source_capture_passed=False,
        source_run_is_trajectory_qualification=False,source_result_missing=True,source_sidecar_missing=True,
        input_rows=1024,working_order=3,point_code_order=2,validation_order=4,original_chunk_size=512,candidate_chunk_size=128,
        parity_cases=cases,arms=rows,eager_failed32_unchanged=True,hooks_restored=True,
        engine_python_sha256=identity['engine']['python_sha256'],extensions=identity['extensions'],
        reciprocal_extensions=identity['reciprocal_extensions'],reciprocal_candidate=identity['reciprocal_candidate'],
        hybrid_adapter_sha256=identity['hybrid_adapter_sha256'],metadata_backend_sha256=identity['metadata_backend_sha256'],
        torch_version=torch.__version__,prefix_receipts=ctx.hybrid.receipts,
        max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),max_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        scope='Four independent fresh engine/cache owners; real full1024 input and same1024 input with133 strided eligible lanes. All16 map/image/bad/accepted trace, output remainder and all statistics bitwise compared. No SR/NN/advance/full-trajectory or performance qualification; source capture resource failure unchanged.')


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception')
    try:
        if a.check_local:
            import torch
            p,audit,identity=source(a,torch)
            assert exact(-0.0,-0.0) and not exact(-0.0,0.0) and not exact(1,True)
            result=dict(status='passed_actual_source_contract',source_payload_sha256=PAYLOAD_SHA,source_audit_sha256=AUDIT_SHA,
                input_rows=1024,eligible_count=int(p['eligible'].sum()),time_zero_initial_all_bytes_equal=True,
                source_capture_passed=False,source_run_is_trajectory_qualification=False,CUDA_run=False,numerical_map_run=False)
        else:result=experiment(a)
    except BaseException as e:result.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());raise
    finally:
        result.update(script_sha256=sha(__file__),process_s=time.perf_counter()-start)
        write(a.output/'RESULT.json',result);print(json.dumps({k:result[k] for k in ['status','process_s']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-local',action='store_true')
    for name in ['capture','capture-audit','output']:parser.add_argument('--'+name,type=Path,required=True)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check',
        'host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend','metadata-check','hybrid-check']:
        parser.add_argument('--'+name,type=Path)
    a=parser.parse_args();a.mode='weighted128_fixed_input_check';a.source40=None;main(a)
