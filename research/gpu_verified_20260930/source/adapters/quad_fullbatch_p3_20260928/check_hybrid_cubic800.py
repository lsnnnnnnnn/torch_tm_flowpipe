"""Actual saved cubic800 candidate differential: dense3/metadata4 vs dense4.

Only weighted maps and bounded validation; no SR, NN, advance, or graph claim.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, math, struct, sys, time, traceback

HERE=Path(__file__).parent
ADAPTER_SHA='7fd6080c4268478696dc92aac4e397027c7fe0830ba22b3297927e985f8d0dd8'
SOURCE_SHA='9994a46f0e3ef1b0d1e78c85760f8b50b0416b9cfb16bb7ae3d4cd5e5246f65b'
SOURCE_RESULT='5be1ffd7d822f50d0d79adfe5e9528687bcdff691110bd639ddfe4fa9492c177'
LONG_SHA='7b465c72046b4c23497ace95f4b3d74cba8fd556cbb48a5454ae0af1b3772c98'
FAILED=[497,609,625,737,753,849,865,881,961,977,993,1009]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def exact(a,b):
    """Includes float sign bits, container type, every field and list element."""
    if type(a) is not type(b):return False
    if isinstance(a,float):return struct.pack('<d',a)==struct.pack('<d',b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

def same(torch,a,b):
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().cpu().contiguous().view(torch.uint8),b.detach().cpu().contiguous().view(torch.uint8))


def source(a,torch):
    p=a.candidate_source;assert sha(p/'RESULT.json')==a.candidate_result_sha256==SOURCE_RESULT
    result=read(p/'RESULT.json');assert result['status']=='completed_local_candidate_validation' and result['script_sha256']==SOURCE_SHA
    assert result['fresh_working_family']==3 and result['point_code_order']==2 and result['validation_order']==4
    assert result['original_initial_and_cap_preserved'] and result['no_SR_restore'] and result['no_advance_or_NN']
    assert result['all_12_proposed_candidates_accepted'] is True and len(result['rows'])==2
    files={'RESULT.json':SOURCE_RESULT};rows=[]
    for group in result['rows']:
        assert group['old_r3_both_padding_all16_bytes'] is True
        z=group['zero_R'];assert sha(p/z['file'])==z['sha256'];files[z['file']]=z['sha256']
        assert [arm['label'] for arm in group['arms']]==['P2_zero_slots_r4','proposed_P3_r4']
        for arm in group['arms']:
            for field,digest_field in [('file','sha256'),('trace_file','trace_sha256')]:
                assert sha(p/arm[field])==arm[digest_field];files[arm[field]]=arm[digest_field]
            saved=torch.load(p/arm['file'],map_location='cpu',weights_only=True);events=read(p/arm['trace_file']);lanes=saved['global_lanes']
            assert lanes==arm['global_lanes'] and len(lanes) in [2,10]
            assert saved['h']==.005 and saved['cutoff']==1e-6 and saved['validation_order']==4
            assert saved['failed_remainders_are_invalid'] is True
            assert saved['working_candidate_degree']==(2 if arm['label'].startswith('P2') else 3)
            for name in ['point','initial','cap','same_R','same_R_image','emitted_remainder']:
                assert saved[name].dtype==torch.float64 and bool(torch.isfinite(saved[name]).all()),name
            assert saved['point'].shape==(len(lanes),16,len(saved['support_ids']))
            assert saved['initial'].shape==(len(lanes),16,len(saved['initial_ids']))
            assert saved['cap'].shape==saved['same_R'].shape==saved['same_R_image'].shape==saved['emitted_remainder'].shape==(len(lanes),16,2)
            assert bool((saved['cap'][...,0]==-.1).all() and (saved['cap'][...,1]==.1).all())
            assert bool((saved['same_R'][...,0]<=0).all() and (saved['same_R'][...,1]>=0).all())
            assert bool((saved['same_R'][...,0]>=-.1).all() and (saved['same_R'][...,1]<=.1).all())
            assert saved['accepted'].dtype==saved['same_R_bad'].dtype==torch.bool and saved['accepted'].shape==saved['same_R_bad'].shape==(len(lanes),)
            assert saved['accepted'].tolist()==arm['accepted'] and saved['same_R_bad'].tolist()==arm['same_R_bad']
            assert saved['exponents'].shape==(len(saved['support_ids']),17) and saved['initial_exponents'].shape==(len(saved['initial_ids']),17)
            assert saved['exponents'].dtype==saved['initial_exponents'].dtype==torch.int64 and int(saved['exponents'].sum(-1).max())<=3
            assert len(saved['variable_support_ids'])==16
            for ids in [saved['support_ids'],saved['initial_ids'],*saved['variable_support_ids']]:assert ids==sorted(set(ids))
            assert all(set(ids)<=set(saved['support_ids']) for ids in saved['variable_support_ids'])
            zero=[j for j,e in enumerate(saved['exponents'].tolist()) if e[0]==0]
            zexp=[tuple(saved['exponents'][j].tolist()) for j in zero];expected=torch.zeros_like(saved['point'][:,:,zero])
            for j,e in enumerate(saved['initial_exponents'].tolist()):expected[:,:,zexp.index(tuple(e))]=saved['initial'][:,:,j]
            assert same(torch,saved['point'][:,:,zero],expected)
            assert events and all(e['global_lane']==lanes[e['lane']] and 0<=e['attempt']<4 for e in events)
            rows.append((arm,saved,events))
    assert len(rows)==4 and sorted(lane for arm,saved,_ in rows if arm['label']=='proposed_P3_r4' for lane in saved['global_lanes'])==FAILED
    return result,files,rows


def guard_dense_builders(extra_modules):
    from flowstar_gpu import monomials
    raw=monomials.build_tables;calls=[];aliases=[]
    def tracked(n,k,*args,**kwargs):
        calls.append(dict(n=n,k=k))
        if k>=4:raise AssertionError('checker forbids every actual dense family4+ builder call')
        return raw(n,k,*args,**kwargs)
    modules=[m for name,m in list(sys.modules.items()) if name.startswith('flowstar_gpu')]+list(extra_modules)
    for module in modules:
        if module is None:continue
        for name,value in list(vars(module).items()):
            if value is raw:setattr(module,name,tracked);aliases.append(module.__name__+'.'+name)
    assert monomials.build_tables is tracked
    return calls,sorted(set(aliases))


def experiment(a):
    long=a.long_runner;assert sha(long)==LONG_SHA
    l=load('hybrid_cubic800_original_bootstrap',long);ctx=l.initialize(a);torch,c,d=ctx.torch,ctx.c,ctx.d
    prior,files,rows=source(a,torch);assert ctx.identity==prior['algorithm_identity']
    assert sha(a.hybrid)==ADAPTER_SHA
    from flowstar_gpu import monomials,support as sp,sparse_exec as se,weighted_validation as wv
    calls,checker_aliases=guard_dense_builders((ctx.c,ctx.d))
    adapter=load('hybrid_cubic800_adapter',a.hybrid);binding=adapter.install(ctx.identity,a.backend,extra_modules=(ctx.c,ctx.d))
    assert 'flowstar_gpu.monomials.build_tables' in binding.guarded_dense_aliases and 'flowstar_gpu.support.build_tables' in binding.guarded_dense_aliases
    tables=d.build_tables(16,3).to('cuda:0');eng=sp.SparseEngine(tables,d.poly.build_step_tables(tables,.005),'cuda:0')
    eng.horner_edge_kernel=ctx.edge.horner_edge
    assert type(eng) is sp.SparseEngine and not isinstance(eng,binding.backend.MetadataEngine)
    assert isinstance(eng.tables,monomials.MonomialTables) and eng.tables.k==3 and eng.tables.T2==100947 and eng.tables.exponents.is_cuda
    names=[v['name'] for v in ctx.cfg['initial_set']];code2=d.compile_ode(ctx.cfg['dynamics_expressions'],names,order=2)
    val=se.validation_engine_for_order(eng,4);code4=se.validation_code_for_order(code2,4,val)
    assert isinstance(val,binding.backend.MetadataEngine) and val.tables.k==4 and se.validation_engine_for_order(eng,4) is val
    assert se.validation_engine_for_order(eng,3) is eng and callable(val.tables.exponents)
    assert len(binding.receipts)==1 and binding.receipts[0]['full_degree_six_prefix']=={'full':100947,'spatial':74613}
    out=[]
    for arm,saved,expected_events in rows:
        sup=sp.make_support(16,4,False,tuple(saved['support_ids']));isup=sp.make_support(16,4,False,tuple(saved['initial_ids']))
        variables=tuple(sp.make_support(16,4,False,tuple(ids)) for ids in saved['variable_support_ids'])
        assert same(torch,saved['exponents'],torch.as_tensor(val.exponents(sup).copy(),dtype=torch.int64))
        assert same(torch,saved['initial_exponents'],torch.as_tensor(val.exponents(isup).copy(),dtype=torch.int64))
        point,initial,cap,R=(saved[k].cuda() for k in ['point','initial','cap','same_R'])
        plan=wv._plan(code4,sup,isup,variables,val,1e-6);assert plan['sup']==sup
        before=tuple(x.detach().clone() for x in (point,initial,cap,R))
        torch.cuda.synchronize();start=time.perf_counter()
        image,bad=wv._map(code4,point,point,R,plan,val,1e-6)
        assert same(torch,image,saved['same_R_image']) and same(torch,bad,saved['same_R_bad'])
        events=[];ok,rem,statistics=wv.validate_failed(code4,point,sup,initial,isup,variables,cap,
            torch.ones(len(point),dtype=torch.bool,device=point.device),val,1e-6,max_attempts=4,chunk_size=32,trace=events.append,use_graph=False)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        for event in events:event['global_lane']=saved['global_lanes'][event['lane']]
        assert same(torch,ok,saved['accepted']) and same(torch,rem,saved['emitted_remainder'])
        assert exact(events,expected_events) and exact(statistics,arm['statistics'])
        assert all(same(torch,x,y) for x,y in zip((point,initial,cap,R),before))
        out.append(dict(label=arm['label'],global_lanes=saved['global_lanes'],source_file=arm['file'],source_sha256=arm['sha256'],
            accepted=ok.cpu().tolist(),same_R_image_bytes_equal=True,same_R_bad_bytes_equal=True,accepted_bytes_equal=True,
            emitted_remainder_bytes_equal=True,events_float_bytes_equal=True,statistics_equal=True,input_all_bytes_unchanged=True,
            event_count=len(events),elapsed_s=elapsed))
    assert calls and all(x['k']<4 for x in calls)
    numerical_calls=list(calls);negative=[]
    for label,builder in [('monomials',monomials.build_tables),('support',sp.build_tables),('driver',d.build_tables)]:
        try:builder(16,4)
        except (AssertionError,RuntimeError):negative.append(label)
        else:raise AssertionError('dense4 guard failed')
    assert calls==numerical_calls,'Hybrid outer guards must reject before original builder dispatch'
    for name,digest in files.items():assert sha(a.candidate_source/name)==digest
    assert sha(a.hybrid)==ADAPTER_SHA
    return dict(status='passed',device=str(eng.tables.exponents.device),adapter_sha256=ADAPTER_SHA,backend_sha256=binding.backend_sha256,
        dependencies=binding.dependencies,source_result_sha256=SOURCE_RESULT,source_script_sha256=SOURCE_SHA,source_files_sha256=files,
        engine_python_sha256=ctx.identity['engine']['python_sha256'],extensions=ctx.identity['extensions'],
        reciprocal_extensions=ctx.identity['reciprocal_extensions'],reciprocal_candidate=ctx.identity['reciprocal_candidate'],torch_version=torch.__version__,
        prior_hybrid_CPU_result_sha256='4260c108b3f86e8bcece71b0a4e0e4e2a6033f46094d831697abbd4387be8522',
        working_order=3,point_code_order=2,validation_order=4,working3_dense=True,validation4_metadata=True,dense4_not_constructed=True,
        dense_builder_calls=numerical_calls,checker_guarded_dense_aliases=checker_aliases,hybrid_guarded_dense_aliases=binding.guarded_dense_aliases,
        blocked_dense4_negative=negative,prefix_receipts=binding.receipts,rows=out,
        max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),max_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        scope='Actual saved two groups x P2/P3 cubic800 candidates. All16 same-R weighted maps, max4 eager validation outputs/events compared by bytes. No graph replay, full advance/plant/SR/NN/trajectory or performance qualification.')


def local_check(a):
    import torch
    prior,files,rows=source(a,torch)
    assert exact([{'x':-0.0,'n':1,'b':True}],[{'x':-0.0,'n':1,'b':True}])
    assert not exact(-0.0,0.0) and not exact(1,True) and not exact(1.,math.nextafter(1.,2.)) and not exact({'x':1},{'x':1,'y':2})
    return dict(status='passed_actual_source_contract',source_result_sha256=SOURCE_RESULT,source_script_sha256=SOURCE_SHA,
        source_files_sha256=files,rows=[dict(label=arm['label'],global_lanes=saved['global_lanes'],accepted=arm['accepted'],event_count=len(events)) for arm,saved,events in rows],
        float_event_comparator_negative_checks=4,numerical_engine_run=False,CUDA_run=False,
        scope='CPU actual saved candidate/initial/support/cap/event/hash contract plus signed-zero/type/ULP/missing-key comparator negatives. No metadata factory, map, validate_failed or GPU.')


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception')
    try:result=local_check(a) if a.check_local else experiment(a)
    except BaseException as e:result.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());raise
    finally:
        result.update(script_sha256=sha(__file__),process_s=time.perf_counter()-start)
        write(a.output/'RESULT.json',result);print(json.dumps({k:result[k] for k in ['status','process_s']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check-local',action='store_true')
    for name in ['candidate-source','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--candidate-result-sha256',required=True)
    p.add_argument('--hybrid',type=Path,default=HERE/'hybrid_metadata.py')
    p.add_argument('--long-runner',type=Path,default=HERE.parent/'quad_fullbatch_sr_20260928/run_fullbatch_long.py')
    for name in ['backend','common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check','host-capacity-check','snapshot-check','source40','cold']:p.add_argument('--'+name,type=Path)
    for name in ['source40-result-sha256','cold-result-sha256']:p.add_argument('--'+name)
    main(p.parse_args())
