"""B1024/n16/P2/q2 plant+stream-SR storage fixture, no ODE/NN/CUDA.

CPU ledgers reserve virtual capacity1000 but touch only2/3 factor rows. Original
SR uses factory capacity16 throughout; never reserve1000 or call whole payload.
"""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,os,resource,shutil,sys,time,traceback
HOST='1a7d9bbf0f16a4f3726b75a27ff938ff87b918141a5f0fe1504f26198dc4035b'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def rss():
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform=='darwin' else peak*1024)

def main(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result={'status':'exception'};marks=[]
    def guard(stage):
        peak=rss();marks.append(dict(stage=stage,peak_rss_bytes=peak));assert peak<1024**3,(stage,peak)
    try:
        here=Path(__file__).parent;host_path=here.parent/'quad_sr_reassociate_20260928/host_ledger.py'
        assert sha(host_path)==HOST
        snapshot_path=here/'snapshot.py';snapshot_sha=sha(snapshot_path)
        expected=json.loads(args.identity.read_text())['engine']['python_sha256']
        for rel,value in expected.items():assert sha(args.engine_root/rel)==value,rel
        sys.path.insert(0,str(args.engine_root/'src'))
        import torch
        from flowstar_gpu import interval as iv,support as sp,sparse_exec as se,symbolic_remainder as sym
        from flowstar_gpu.monomials import build_tables
        from flowstar_gpu.polynomial import build_step_tables
        torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.set_num_interop_threads(1)
        h=load('fullsnapshot_host',host_path);snap=load('fullsnapshot_candidate',snapshot_path)
        tables=build_tables(16,2);eng=sp.SparseEngine(tables,build_step_tables(tables,.005),'cpu')
        pre_sup=sp.make_support(16,2,False,tuple(range(18)));tmv_sup=sp.make_support(16,2,True,tuple(range(17)))
        pre=torch.zeros((1024,16,18));tmv=torch.zeros((1024,16,17))
        for axis in range(16):
            pre[:,axis,0]=.5+axis/64
            pre[:,axis,16-axis]=.125
            pre[:,axis,17]=(axis+1)/1024
            tmv[:,axis,16-axis]=1
        pre[:,0,0]+=torch.arange(1024)/2**20
        rem=torch.tensor([-2.**-30,2.**-29]).expand(1024,16,2).clone()
        status=torch.zeros(1024,dtype=torch.int8);status[-1]=1
        st=se.SparseState(pre=pre,pre_rem=rem.clone(),tmv=tmv,tmv_rem=rem.clone(),status=status,
                          pre_sup=pre_sup,tmv_sup=tmv_sup)
        cap=torch.tensor([-.1,.1]).expand(1024,16,2).clone()
        sr=sym.make_symbolic_remainder(1024,16,1000,'cpu')
        ledger=h.HostFactorLedger(1024,lane_chunk=16,status=status)
        assert len(sr.phi_buf)==16 and ledger.factors.shape==(1000,1024,16,16,2)
        guard('initial: factor capacity virtual; factory SR capacity16')
        def values(k):
            p=torch.eye(16).expand(1024,16,16).clone()
            p[:,0,1]=(k%3+1)/128;p[:,1,0]=-k/256
            f=iv.from_point(p);f[:,:2,:2,0]-=2.**-30;f[:,:2,:2,1]+=2.**-30
            J=torch.tensor([-2.**-25,2.**-24]).expand(1024,16,2).clone();J[:,1]*=k
            return p,f,J
        def append(st,sr,ledger,out,J,k):
            st.pre_rem=iv.add(out,J);st.tmv_rem=J.clone();sr.scalars_iv=iv.from_point(sr.scalars);sr.append_j(J)
            ledger.finish_step(sr,st.status,token=k)
        def fingerprints(st,sr,ledger,cap):
            assert ledger.length<=3 and sr.qlen<=3 and len(sr.phi_buf)==16
            # All views are explicitly live q<=3; no unused capacity is hashed.
            views=ledger._snapshot_views(sr)
            assert views['factors'].shape[0]<=3 and views['factors'].numel()*8<=12*1024**2
            return dict(plant={k:h.signature(getattr(st,k)) for k in snap.PLANT},
                        pre_ids=list(st.pre_sup.ids),tmv_ids=list(st.tmv_sup.ids),
                        pre_exponents=h.signature(snap.exponents(eng,st.pre_sup)),
                        tmv_exponents=h.signature(snap.exponents(eng,st.tmv_sup)),cap=h.signature(cap),
                        sr_and_ledger=h.signature(views))
        for k in [1,2]:
            p,f,J=values(k)
            out=ledger.propagate(sr,p,f,status_before=st.status,raw_propagate=sym.propagate,sym=sym)
            append(st,sr,ledger,out,J,k)
            assert len(sr.phi_buf)==16 and ledger.length==k
        guard('two actual full-B raw propagations finished')
        metadata=dict(host_adapter_sha256=HOST,fixture='CPU-only B1024/n16/P2 q2 storage',engine_python_sha256=expected)
        progress=dict(completed_step=2,phase='committed_before_endpoint_handoff')
        before=fingerprints(st,sr,ledger,cap)
        path=args.output/'committed2';manifest=snap.save(path,st,sr,ledger,eng,cap,metadata=metadata,progress=progress,host=h)
        assert before==fingerprints(st,sr,ledger,cap)
        guard('stream save q2')
        restored,rsr,rledger,rcap,rprogress=snap.restore(path,eng,metadata=metadata,host=h,raw_factory=sym.make_symbolic_remainder)
        assert progress==rprogress and before==fingerprints(restored,rsr,rledger,rcap)
        assert ledger.factors.data_ptr()!=rledger.factors.data_ptr() and sr.phi_buf.data_ptr()!=rsr.phi_buf.data_ptr()
        guard('cold restore q2; two owners')
        negative=[]
        def reject(label,fn):
            try:fn()
            except (AssertionError,KeyError):negative.append(label)
            else:raise AssertionError('invalid checkpoint accepted: '+label)
        reject('metadata mismatch',lambda:snap.restore(path,eng,metadata={**metadata,'fixture':'changed'},host=h,raw_factory=sym.make_symbolic_remainder))
        # Hardlink unchanged component files; atomically replace any changed
        # plant/manifest file so the immutable source checkpoint stays intact.
        def corrupted(label,plant_change=None,manifest_change=None):
            dest=args.output/('bad_'+label);shutil.copytree(path,dest,copy_function=os.link)
            m=json.loads((dest/'MANIFEST.json').read_text())
            if plant_change:
                v=torch.load(dest/'plant.pt',map_location='cpu',weights_only=True);plant_change(v)
                temporary=dest/'plant.new';torch.save(v,temporary);temporary.replace(dest/'plant.pt')
                m['files']['plant.pt']=sha(dest/'plant.pt');del v
            if manifest_change:manifest_change(m)
            temporary=dest/'MANIFEST.new';temporary.write_text(json.dumps(m)+'\n');temporary.replace(dest/'MANIFEST.json')
            reject(label,lambda:snap.restore(dest,eng,metadata=metadata,host=h,raw_factory=sym.make_symbolic_remainder))
        corrupted('unknown_phase',manifest_change=lambda m:m['progress'].update(phase='invented'))
        corrupted('boundary_phase_at_step2',manifest_change=lambda m:m['progress'].update(phase='after_controller'))
        corrupted('support_ids',plant_change=lambda v:v.update(pre_ids=(0,0,*v['pre_ids'][2:])))
        def wrong_exponents(v):v['pre_exponents'][1]=v['pre_exponents'][2]
        corrupted('support_exponents',plant_change=wrong_exponents)
        corrupted('cap',plant_change=lambda v:v['cap'].fill_(0.))
        corrupted('status_ledger_mismatch',plant_change=lambda v:v['status'].fill_(0))
        corrupted('completed_step_mismatch',manifest_change=lambda m:m['progress'].update(completed_step=3))
        wrong=copy.copy(st);wrong.pre_sup=tmv_sup
        reject('save_wrong_support_family',lambda:snap.save(args.output/'wrong_support',wrong,sr,ledger,eng,cap,metadata=metadata,progress=progress,host=h))
        assert before==fingerprints(st,sr,ledger,cap) and before==fingerprints(restored,rsr,rledger,rcap)
        guard('negative metadata/support/cap/status/phase cases')
        p,f,J=values(3)
        out=ledger.propagate(sr,p,f,status_before=st.status,raw_propagate=sym.propagate,sym=sym)
        reject('pending_without_fresh_J',lambda:snap.save(args.output/'pending',st,sr,ledger,eng,cap,metadata=metadata,progress={**progress,'completed_step':3},host=h))
        append(st,sr,ledger,out,J,3)
        coldout=rledger.propagate(rsr,p,f,status_before=restored.status,raw_propagate=sym.propagate,sym=sym)
        append(restored,rsr,rledger,coldout,J,3)
        assert h.signature(out)==h.signature(coldout)
        after=fingerprints(st,sr,ledger,cap);assert after==fingerprints(restored,rsr,rledger,rcap)
        assert sr.qlen==sr.jlen==ledger.length==3 and len(sr.phi_buf)==len(rsr.phi_buf)==16
        assert sha(snapshot_path)==snapshot_sha and sha(host_path)==HOST
        guard('third actual raw propagation: all fields byte equal')
        result=dict(status='passed',device='cpu',script_sha256=sha(__file__),serializer_sha256=snapshot_sha,
            host_adapter_sha256=HOST,engine_python_sha256=expected,batch=1024,n=16,working_order=2,
            original_SR_capacity=16,host_factor_capacity=1000,host_factor_live_rows=3,
            initial_completed_SR_history_steps=2,cold_next_step=3,full_plant_support_cap_status_and_SR_ledger_bytes_equal=True,
            next_rawprop_image_all_bytes_equal=True,checkpoint_manifest_sha256=sha(path/'MANIFEST.json'),
            negative_cases_rejected=negative,negative_count=len(negative),before_fingerprints=before,after_fingerprints=after,
            memory_guard_bytes=1024**3,rss_marks=marks,scope='Synthetic finite plant plus actual pinned SR propagation, CPU B1024 q2 save/restore/q3 only. Factor capacity1000 is virtual; only3 rows touched. No reserve1000 original SR, whole-ledger hash/payload, nonlinear ODE/NN, GPU, or real full40 qualification.')
    except BaseException as exc:result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:
        result['process_s']=time.perf_counter()-start;result['peak_rss_bytes']=rss()
        (args.output/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['engine_python_sha256','before_fingerprints','after_fingerprints']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['engine-root','identity','output']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
