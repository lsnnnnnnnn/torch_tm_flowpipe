"""Two-stage CUDA gate for host factor storage, never a QUAD/NN trajectory.

small: frozen CPU fixture values, original CUDA SR and frozen B2 K20 parity.
capacity: own small gate then one B1024 q999->1000; real B2 history templates.
"""
from pathlib import Path
from types import ModuleType
import argparse,ast,copy,gc,hashlib,importlib.util,json,os,sys,time,traceback
HOST_SHA='1a7d9bbf0f16a4f3726b75a27ff938ff87b918141a5f0fe1504f26198dc4035b'
CPU_CHECK_SHA='514b39c74fa77ea54754dadb3d2394cf5f49b55d78d169b3517159fbedec6833'
CPU_GATE_SHA='c704ae33b03f8216c4e7df98d3abea3c166ebdfd1e57a1193374cd7259e0abc9'
OLD_SHA='17ac3a69f934fb6943df0f2dab8072d84cb327b2ce00bd58573bfdae64fcfe86'
COMMON_SHA='1c85e04988028f7ac95d945eeff9ef3e0936ee306c1f0a80650bb06d8073d985'
GIB=1024**3;BLOCK=16*1024**2

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def setup(args):
    global torch,h,old,sym,iv,rawprop,rawmake,c,values
    here=Path(__file__).parent
    assert sha(here/'host_ledger.py')==HOST_SHA and sha(here/'check_host_ledger.py')==CPU_CHECK_SHA
    assert sha(here/'periodic_sr.py')==OLD_SHA and sha(args.common)==COMMON_SHA
    assert sha(args.cpu_gate/'RESULT.json')==CPU_GATE_SHA
    cpu=read(args.cpu_gate/'RESULT.json');assert cpu['status']=='passed' and cpu['device']=='cpu'
    assert cpu['adapter_sha256']==HOST_SHA and cpu['script_sha256']==CPU_CHECK_SHA
    assert cpu['step_comparisons']==440 and cpu['negative_cases_rejected']==34
    assert cpu['original1000_reset_and_nextstep_cold_bytes_equal'] is True
    identity=read(args.identity);expected=identity['engine']['python_sha256']
    assert cpu['engine_python_sha256']==expected
    for rel,digest in expected.items():assert sha(args.engine_root/rel)==digest
    if args.mode=='capacity':
        gate=read(args.small_gate/'RESULT.json')
        assert gate['status']=='passed' and gate['mode']=='small' and gate['device']=='cuda'
        assert gate['script_sha256']==sha(__file__) and gate['host_sha256']==HOST_SHA
        assert gate['cpu_gate_sha256']==sha(args.cpu_gate/'RESULT.json') and gate['identity_sha256']==sha(args.identity)
        assert gate['small']['all_original_SR_and_image_bytes_equal'] and gate['small']['mixed_bad_lanes_preserve_raw']
        assert gate['small']['step_comparisons']==160 and gate['small']['rows']==[dict(batch=2,chunk=k,enabled=e,steps=40) for e,k in [(False,16),(True,1),(True,16)]]
        assert gate['small']['stream_cold_and_reset_passed'] and gate['dispatch']['matrix_calls']>0 and gate['dispatch']['history_sum_calls']>0
    runtime=load('host_cuda_common',args.common)
    assert runtime.E.resolve()==args.engine_root.resolve() and runtime.ORIGINAL['engine']['python_sha256']==expected
    runtime.bootstrap();torch=runtime.torch;torch.set_default_dtype(torch.float64)
    total=torch.cuda.get_device_properties(0).total_memory;assert total>=11*GIB
    torch.cuda.set_per_process_memory_fraction(11*GIB/total,0)
    from flowstar_gpu import symbolic_remainder as symbolic,interval,sparse_exec
    sym,iv=symbolic,interval;rawprop=sym.propagate;rawmake=sym.make_symbolic_remainder
    h=load('host_cuda_adapter',here/'host_ledger.py');old=load('host_cuda_old_k20',here/'periodic_sr.py')
    c=ModuleType('host_cuda_frozen_reference');c.se=sparse_exec;c.make_symbolic_remainder=rawmake
    c.fingerprint=lambda *a:None;c.save_state=lambda *a,**k:None
    tree=ast.parse((here/'check_host_ledger.py').read_text())
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='values'];assert len(nodes)==1
    env={'torch':torch,'iv':iv};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(here/'check_host_ledger.py'),'exec'),env)
    values=lambda k,b:tuple(x.to('cuda') for x in env['values'](k,b))
    extensions=runtime.ORIGINAL['extensions'];assert len(extensions)==8 and extensions==identity['extensions']
    for entry in extensions.values():assert sha(entry['path'])==entry['sha256']
    for name,module in [('flowstar_sr_interval_matmul',sym.sr_kernels),('flowstar_sr_history_sum',sym.sr_sum_kernels)]:
        assert module.available();assert sha(module._ext.__file__)==extensions[name]['sha256']
    if args.mode=='capacity':assert gate['extensions']==extensions and gate['torch_version']==torch.__version__
    return dict(engine_python_sha256=expected,extensions=extensions,torch_version=torch.__version__,common_sha256=COMMON_SHA,
        host_sha256=HOST_SHA,cpu_gate_sha256=sha(args.cpu_gate/'RESULT.json'),identity_sha256=sha(args.identity),
        small_gate_sha256=sha(args.small_gate/'RESULT.json') if args.small_gate else None,
        torch_allocation_cap_bytes=11*GIB,external_watch_limit_bytes=int(11.5*GIB))

def append(sr,J):sr.scalars_iv=iv.from_point(sr.scalars);sr.append_j(J)
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and h.digest(a)==h.digest(b)
def sr_equal(a,b):
    assert a.qlen==b.qlen and a.jlen==b.jlen
    for key in h.SR_FIELDS:
        x,y=getattr(a,key),getattr(b,key)
        if x is None or y is None:assert x is y;continue
        if key in ['phi_buf','phi_iv_buf','j_buf']:x,y=x[:a.qlen],y[:b.qlen]
        assert same(x,y),key

def finish(ledger,sr,J,status=None):
    append(sr,J);ledger.finish_step(sr,ledger.current_status if status is None else status,token=ledger.epoch+ledger.length)

def small(args):
    rows=[];steps=0
    for enabled,chunk in [(False,16),(True,1),(True,16)]:
        sr=rawmake(2,16,1000,'cuda');ledger=h.HostFactorLedger(2,lane_chunk=chunk,enabled=enabled)
        installed=old.install(c,enabled=enabled);ref=c.make_symbolic_remainder(2,16,1000,'cuda')
        try:
            for k in range(1,41):
                p,f,J=values(k,2);out=ledger.propagate(sr,p,f,status_before=ledger.current_status,raw_propagate=rawprop,sym=sym)
                expected=c.se.propagate(ref,p,strict=True,phi_i_iv=f);finish(ledger,sr,J);append(ref,J)
                assert same(out,expected);sr_equal(sr,ref);steps+=1
                if k==20:
                    path=args.output/f'stream_{enabled}_{chunk}_20';ledger.save_components(path,sr)
                    rs,rh=h.restore_components(path,rawmake,'cuda');sr_equal(sr,rs)
                    assert h.signature(rh.payload(rs))==h.signature(ledger.payload(sr))
                    p21,f21,J21=values(21,2);left=copy.deepcopy(sr);lh=copy.deepcopy(ledger)
                    a=lh.propagate(left,p21,f21,status_before=lh.current_status,raw_propagate=rawprop,sym=sym)
                    b=rh.propagate(rs,p21,f21,status_before=rh.current_status,raw_propagate=rawprop,sym=sym)
                    finish(lh,left,J21);finish(rh,rs,J21);assert same(a,b) and h.signature(lh.payload(left))==h.signature(rh.payload(rs))
                    del rs,rh,left,lh,a,b
            rows.append(dict(batch=2,chunk=chunk,enabled=enabled,steps=40))
        finally:installed.uninstall()
    # Mixed original inactive, NaN, reversed interval, and overflow lanes.
    sr=rawmake(5,16,1000,'cuda');base=rawmake(5,16,1000,'cuda');status=torch.tensor([0,1,0,0,0],dtype=torch.int8)
    ledger=h.HostFactorLedger(5,lane_chunk=2,status=status);installed=old.install(c,enabled=True);ref=c.make_symbolic_remainder(2,16,1000,'cuda')
    try:
        for k in range(1,41):
            p,f,J=values(k,5);pr,fr,Jr=values(k,2);f[1,0,0]=float('nan');p[1,0,0]=float('nan')
            if k==5:f[2,0,0]=float('nan')
            if k==7:f[3,0,0]=torch.tensor([1.,-1.],device='cuda')
            if k in [10,11]:p[4,0,0]=1e200;f[4,0,0]=1e200
            out=ledger.propagate(sr,p,f,status_before=status,raw_propagate=rawprop,sym=sym)
            raw=rawprop(base,p,strict=True,phi_i_iv=f);expected=c.se.propagate(ref,pr,strict=True,phi_i_iv=fr)
            after=status.clone()
            if k==5:after[2]=3
            if k==7:after[3]=3
            if k==20:after[4]=3
            finish(ledger,sr,J,after);append(base,J);append(ref,Jr);status=after
            assert same(out[0],expected[0]) and same(out[1:],raw[1:])
            assert same(sr.phi_iv_buf[:k,0],ref.phi_iv_buf[:k,0]) and same(sr.phi_iv_buf[:k,1:],base.phi_iv_buf[:k,1:])
            assert same(sr.phi_buf[:k],base.phi_buf[:k]) and same(sr.j_buf[:k],base.j_buf[:k]);steps+=1
            if k in [20,40]:
                assert ledger.replaced[k-1].tolist()==[True,False,False,False,False]
                assert ledger.fallback[k-1].tolist()==([False,False,False,False,True] if k==20 else [False]*5)
        path=args.output/'mixed40';ledger.save_components(path,sr);rs,rh=h.restore_components(path,rawmake,'cuda')
        assert h.signature(ledger.payload(sr))==h.signature(rh.payload(rs))
        bad=status.clone();bad[1]=0
        try:rh.propagate(rs,*values(41,5)[:2],status_before=bad,raw_propagate=rawprop,sym=sym)
        except AssertionError:pass
        else:raise AssertionError('inactive revival accepted')
    finally:installed.uninstall()
    # Transport/reset only. Real999 history is built separately by capacity().
    sr=rawmake(2,16,1000,'cuda');sr.reserve(1000);ledger=h.HostFactorLedger(2)
    p,f,J=values(1,2);sr.phi_buf.copy_(p);sr.phi_iv_buf=f.expand(1000,*f.shape).clone();sr.j_buf.copy_(J)
    sr.scalars_iv=iv.from_point(sr.scalars);sr.qlen=sr.jlen=ledger.length=1000
    ledger.factors.copy_(f.cpu());ledger.point_finite.fill_(True);ledger.factor_valid.fill_(True);ledger.status_before.zero_()
    ledger.eligible[19::20]=True;ledger.replaced[19::20]=True;ledger.last_rebuilt_lane.fill_(1000)
    assert ledger.reset_if_full(sr) and ledger.epoch==1000 and ledger.length==0
    path=args.output/'reset1000';ledger.save_components(path,sr);rs,rh=h.restore_components(path,rawmake,'cuda')
    assert h.signature(ledger.payload(sr))==h.signature(rh.payload(rs))
    return dict(rows=rows,step_comparisons=steps,all_original_SR_and_image_bytes_equal=True,
        mixed_bad_lanes_preserve_raw=True,stream_cold_and_reset_passed=True,inactive_revival_rejected=True,
        reset_scope='Synthetic transport/reset; no1000-step small-gate trajectory')


def resource_values(k):
    # Dyadic triangular point factors keep point products exact across BLAS
    # batch sizes; interval factors still have nonzero outward uncertainty.
    p=torch.eye(16,device='cuda').expand(2,16,16).clone()
    p[0,0,1]=(1 if k%2 else -1)/256;p[1,1,0]=(1 if k%3 else -2)/512
    f=iv.from_point(p);f[:,:2,:2,0]-=2.**-40;f[:,:2,:2,1]+=2.**-40
    J=torch.tensor([-2.**-28,2.**-27],device='cuda').expand(2,16,2).clone();J[1]*=2
    return p,f,J


def chunks(value,lane_axis,batch):
    if lane_axis==0:yield 0,1,value;return
    rows=max(1,BLOCK//(batch*value[0:1,0].numel()*value.element_size()))
    for q in range(0,len(value),rows):yield q,min(q+rows,len(value)),value[q:q+rows]

def repeat_lanes(value,axis,B=1024):
    repeats=[1]*value.ndim;repeats[axis]=B//2;return value.repeat(*repeats)

def copy_from_template(destination,reference,axis):
    for first,last,part in chunks(destination,axis,1024):
        source=reference if axis==0 else reference[first:last]
        expanded=repeat_lanes(source,axis);part.copy_(expanded);del expanded

def compare_template(views,reference):
    hashes={}
    for key,value in views.items():
        if not isinstance(value,torch.Tensor):continue
        ref=reference[key];axis=0 if key in ['current_status','last_rebuilt_lane','sr_scalars','sr_scalars_iv'] else 1
        hasher=hashlib.sha256()
        for first,last,part in chunks(value,axis,1024):
            actual=h.cpu_copy(part);expected=repeat_lanes(ref if axis==0 else ref[first:last],axis)
            assert same(actual,expected),key
            hasher.update(memoryview(actual.view(torch.uint8).numpy()).cast('B'));del actual,expected
        hashes[key]=dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hasher.hexdigest())
    return hashes


def capacity(args):
    marks={};start=time.perf_counter();sr=rawmake(2,16,1000,'cuda');sr.reserve(1000)
    ledger=h.HostFactorLedger(2,lane_chunk=16);installed=old.install(c,enabled=True);ref=c.make_symbolic_remainder(2,16,1000,'cuda');ref.reserve(1000)
    try:
        for k in range(1,1001):
            p,f,J=resource_values(k);out=ledger.propagate(sr,p,f,status_before=ledger.current_status,raw_propagate=rawprop,sym=sym)
            expected=c.se.propagate(ref,p,strict=True,phi_i_iv=f);finish(ledger,sr,J);append(ref,J)
            assert same(out,expected);sr_equal(sr,ref)
            if k==999:seed=ledger.payload(sr)
        reference=ledger.payload(sr);reference_image=h.cpu_copy(out)
    finally:installed.uninstall()
    marks['B2_actual1000_reference_s']=time.perf_counter()-start
    del sr,ledger,ref,p,f,J,out,expected;gc.collect();torch.cuda.empty_cache()
    t=time.perf_counter();sr=rawmake(1024,16,1000,'cuda');sr.reserve(1000)
    sr.phi_iv_buf=torch.empty((1000,1024,16,16,2),dtype=torch.float64,device='cuda')
    sr.scalars_iv=torch.empty((1024,16,2),dtype=torch.float64,device='cuda')
    ledger=h.HostFactorLedger(1024,lane_chunk=16);sr.qlen=sr.jlen=ledger.length=999
    ledger.epoch=seed['epoch']
    for key in ['factors','point_finite','factor_valid','status_before','eligible','replaced','fallback','current_status','last_rebuilt_lane']:
        target=getattr(ledger,key);target=target[:999] if key not in ['current_status','last_rebuilt_lane'] else target
        copy_from_template(target,seed[key],0 if key in ['current_status','last_rebuilt_lane'] else 1)
    for key in h.SR_FIELDS:
        target=getattr(sr,key);target=target[:999] if key in ['phi_buf','phi_iv_buf','j_buf'] else target
        copy_from_template(target,seed['sr_'+key],0 if key in ['scalars','scalars_iv'] else 1)
    del target
    views=ledger._snapshot_views(sr);h.validate_payload(views);before=compare_template(views,seed);del views,seed
    torch.cuda.synchronize();marks['B1024_seed_copy_hash_s']=time.perf_counter()-t
    p,f,J=resource_values(1000);p,f,J=[repeat_lanes(v,0) for v in [p,f,J]]
    t=time.perf_counter();out=ledger.propagate(sr,p,f,status_before=ledger.current_status,raw_propagate=rawprop,sym=sym)
    finish(ledger,sr,J);torch.cuda.synchronize();marks['B1024_single_step1000_s']=time.perf_counter()-t
    assert same(out,repeat_lanes(reference_image,0))
    assert ledger.replaced[999].all() and not ledger.fallback[999].any()
    after=compare_template(ledger._snapshot_views(sr),reference)
    t=time.perf_counter();checkpoint=args.output/'capacity1000_components';ledger.save_components(checkpoint,sr)
    manifest=read(checkpoint/'MANIFEST.json')
    assert all(v['raw_sha256']==after[k]['sha256'] for k,v in manifest['tensors'].items())
    marks['stream_save_s']=time.perf_counter()-t
    # Release both full-size owners before restore; never hold two SR/ledgers.
    del sr,ledger,out,p,f,J;gc.collect();torch.cuda.empty_cache()
    t=time.perf_counter();sr,ledger=h.restore_components(checkpoint,rawmake,'cuda')
    restored=compare_template(ledger._snapshot_views(sr),reference);assert restored==after
    marks['stream_restore_compare_s']=time.perf_counter()-t
    assert ledger.reset_if_full(sr) and ledger.epoch==1000 and ledger.length==sr.qlen==sr.jlen==0
    assert bool((sr.scalars==1).all() and (sr.scalars_iv==1).all()) and not ledger.last_rebuilt_lane.any()
    resetpath=args.output/'capacity_reset_components';ledger.save_components(resetpath,sr)
    return dict(batch=1024,n=16,capacity=1000,chunk=16,initial_q=999,full_batch_propagate_calls=1,
        real_template_batch=2,real_template_steps=1000,lane_template='lane % 2; real original propagate + qualified K20 history',
        full999_field_hashes=before,full1000_field_hashes=after,stream_restore_all_fields_equal=True,
        stream_manifest_sha256=sha(checkpoint/'MANIFEST.json'),reset_manifest_sha256=sha(resetpath/'MANIFEST.json'),
        original_reset_and_epoch1000=True,phase_times_s=marks,
        scope='Synthetic SR storage/arithmetic/resource fixture, not QUAD; no1000-step B1024 trajectory or NN executed.')


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result={'status':'exception'};calls={'matrix_calls':0,'history_sum_calls':0}
    try:
        identity=setup(args);mm=sym.sr_kernels.left_multiply_;ss=sym.sr_sum_kernels.sum_history;fallback=sym._matmul_iv;interval_mul=sym.iv.mul
        def matrix(left,prior):
            assert left.is_cuda and prior.is_cuda;calls['matrix_calls']+=1;return mm(left,prior)
        def history(m,j):
            assert m.is_cuda and j.is_cuda;calls['history_sum_calls']+=1;return ss(m,j)
        def forbidden(*a,**kw):raise AssertionError('CUDA fixture used Python/broadcast fallback')
        sym.sr_kernels.left_multiply_=matrix;sym.sr_sum_kernels.sum_history=history;sym._matmul_iv=forbidden;sym.iv.mul=forbidden
        try:detail=small(args) if args.mode=='small' else capacity(args)
        finally:sym.sr_kernels.left_multiply_=mm;sym.sr_sum_kernels.sum_history=ss;sym._matmul_iv=fallback;sym.iv.mul=interval_mul
        assert calls['matrix_calls']>0 and calls['history_sum_calls']>0
        result=dict(status='passed',device='cuda',mode=args.mode,**identity,**{args.mode:detail},dispatch=dict(calls,hooks_restored=True),
            no_original_extension_rebuild=True,owned_gpu_peak_source='External original11.5GiB watchdog process.json; not inferred from torch reserved')
    except BaseException as exc:result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:
        result.update(script_sha256=sha(__file__),process_s=time.perf_counter()-start)
        if 'torch' in globals() and torch.cuda.is_initialized():
            result.update(max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),max_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        write(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['small','capacity'],required=True)
    for key in ['engine-root','identity','cpu-gate','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--small-gate',type=Path);p.add_argument('--common',type=Path,default=Path(__file__).parent.parent/'quad_split_multistep_20260927/continuation.py')
    main(p.parse_args())
