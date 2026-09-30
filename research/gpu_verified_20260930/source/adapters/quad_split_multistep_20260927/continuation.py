"""Bounded conservative restart from an accepted local child flowpipe.

Old pre-678 history is deliberately replaced by the already enclosing local
child box. The stored p/R certifies step678; first new advance is step679.
This is not a coordinate transformation of the original root's historical SR.
"""
from pathlib import Path
from fractions import Fraction as F
from dataclasses import replace
import argparse, copy, hashlib, importlib.util, json, os, subprocess, sys, time

N=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z')
E=N/'engine_quad_normalization_center'
PREVIOUS=N/'runs/quad_split_probe_20260925'
ORIGINAL=json.loads((N/'runs/quad_normalization_center_20260924/combined/result.json').read_text())
os.environ.update(ORIGINAL['environment'])
sys.path.insert(0,str(E/'src'))
import torch, yaml
from flowstar_gpu import sparse_exec as se, support as sp, config, interval as iv
from flowstar_gpu import cuda_kernels as ck, tape_kernels as tk, sr_kernels, sr_sum_kernels, injective_index
from flowstar_gpu.monomials import build_tables
from flowstar_gpu.polynomial import build_step_tables
from flowstar_gpu.composition import build_schedule
from flowstar_gpu.ode_compiler import compile_ode
from flowstar_gpu.symbolic_remainder import make_symbolic_remainder

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')
def binary(name):
    info=ORIGINAL['extensions'][name];assert sha(info['path'])==info['sha256']
    spec=importlib.util.spec_from_file_location(name,info['path']);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def bootstrap():
    assert subprocess.check_output(['git','-C',str(E),'rev-parse','HEAD'],text=True).strip()=='a3fb2e94ba976aaf498c4a9cb3f98165cddcc272'
    assert not subprocess.check_output(['git','-C',str(E),'status','--porcelain'],text=True).strip()
    for name,expected in ORIGINAL['engine']['python_sha256'].items():
        assert sha(E/name)==expected,name
    for name,m,field,tried in [('flowstar_seg_kernels',ck,'_ext','_tried'),('flowstar_tape_kernels',tk,'_ext','_tried'),('flowstar_valid_kernels',tk,'_vext','_vtried'),('flowstar_sr_interval_matmul',sr_kernels,'_ext','_tried'),('flowstar_sr_history_sum',sr_sum_kernels,'_ext','_tried'),('flowstar_injective_index_v2',injective_index,'_ext','_tried')]:
        setattr(m,field,binary(name));setattr(m,tried,True)
    ck._ext=binary('flowstar_seg_private_output_v1')
    edge=binary('horner_edge_1312fa8b2aed')
    assert ck.available() and tk.available() and tk.valid_available()
    assert sr_kernels.available() and sr_sum_kernels.available() and injective_index.available()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from flowstar_gpu import determinism
    determinism.enable_determinism('cuda')
    torch.cuda.set_per_process_memory_fraction(11*2**30/torch.cuda.get_device_properties(0).total_memory)
    return edge

def setup(n,h,device,edge):
    tab=build_tables(n,2).to(device)
    eng=sp.SparseEngine(tab,build_step_tables(tab,h),device)
    if device=='cuda':eng.horner_edge_kernel=edge.horner_edge
    return eng,build_schedule(n,2,device)

def import_seed(paths,eng,sched):
    manifest=json.loads((PREVIOUS/'MANIFEST.json').read_text())['files']
    for path in paths:
        item=manifest[str(path.relative_to(PREVIOUS))]
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    seeds=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
    exps=[tuple(e) for e in eng.tables.exponents.cpu().tolist()];mapping={e:i for i,e in enumerate(exps)}
    old_exps=[[tuple(seed['exponents'][i]) for i in seed['support']] for seed in seeds]
    sup=sp.make_support(eng.tables.n,eng.tables.k,False,tuple(sorted(set(mapping[e] for es in old_exps for e in es)|{0})))
    dev=eng.device;B=len(seeds);n=eng.tables.n
    st=se.initial_sparse_state(torch.zeros(B,n,2,dtype=torch.float64,device=dev),eng,sched)
    pre=torch.zeros(B,n,sup.size,dtype=torch.float64,device=dev)
    for lane,(seed,es) in enumerate(zip(seeds,old_exps)):
        assert seed['accepted'] and max(map(sum,es))<=2
        for j,e in enumerate(es):pre[lane,:,sup.ids.index(mapping[e])]=seed['p'][0,:,j].to(dev)
        restored=pre[lane,:,[sup.ids.index(mapping[e]) for e in es]].cpu().contiguous()
        assert torch.equal(restored.view(torch.uint8),seed['p'][0].contiguous().view(torch.uint8))
        for row,initial in zip(seed['p'][0].tolist(),seed['initial'][0].tolist()):
            p0={e:F(c) for e,c in zip(es,row) if e[0]==0 and c}
            x0={tuple(seed['exponents'][i]):F(c) for i,c in zip(seed['initial_support'],initial) if c}
            assert p0==x0
    st.pre=pre;st.pre_sup=sup
    st.pre_rem=torch.cat([s['remainder'] for s in seeds]).to(dev)
    cap=torch.cat([s['cap'] for s in seeds]).to(dev)
    assert bool(iv.contains(cap,st.pre_rem).all())
    assert not bool(st.tmv_rem.any()) and bool((st.status==0).all())
    for state in range(n):
        assert torch.count_nonzero(st.tmv[:,state])==B
        pos=st.tmv_sup.ids.index(int(sched.var_image[state]))
        assert bool((st.tmv[:,state,pos]==1).all())
    return st,cap,seeds

def fingerprint(st,sr):
    result={}
    def take(name,x):
        result[name]=None if x is None else hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    for name in ['pre','pre_rem','tmv','tmv_rem','status']:take(name,getattr(st,name))
    for name in ['scalars','scalars_iv','phi_buf','phi_iv_buf','j_buf']:
        x=getattr(sr,name)
        if x is not None and name.startswith('phi'):x=x[:sr.qlen]
        if x is not None and name=='j_buf':x=x[:sr.jlen]
        take('sr_'+name,x)
    result.update(qlen=sr.qlen,jlen=sr.jlen,pre_support=st.pre_sup.ids,tmv_support=st.tmv_sup.ids)
    return result

def attempt(st,sr,code,eng,sched,settings,cap):
    # The owned trial prevents mutation of the committed parent's SR history.
    before=fingerprint(st,sr);trial_st=copy.deepcopy(st);trial_sr=copy.deepcopy(sr)
    torch.cuda.synchronize() if st.pre.is_cuda else None
    started=time.perf_counter()
    out,ok=se.advance_sparse(trial_st,code,eng,sched,settings,cap,trial_sr)
    torch.cuda.synchronize() if st.pre.is_cuda else None
    elapsed=time.perf_counter()-started
    assert fingerprint(st,sr)==before,'advance changed the committed state/SR'
    assert trial_sr.qlen==trial_sr.jlen==sr.jlen+1
    return out,ok,trial_sr,elapsed

def save_state(path,st,sr,eng,cap=None,step=None,phase='trial'):
    obj={k:getattr(st,k).detach().cpu() for k in ['pre','pre_rem','tmv','tmv_rem','status']}
    obj.update(pre_ids=st.pre_sup.ids,tmv_ids=st.tmv_sup.ids,
               pre_exponents=eng.tables.exponents[list(st.pre_sup.ids)].cpu(),
               tmv_exponents=eng.tables.exponents[eng.tables.spatial_index[list(st.tmv_sup.ids)]].cpu(),
               qlen=sr.qlen,jlen=sr.jlen,sr_max_size=sr.max_size,sr_capacity=sr.phi_buf.shape[0],
               n=eng.tables.n,k=eng.tables.k,h=eng.step.delta,step=step,phase=phase,
               cap=cap.cpu() if cap is not None else None)
    for k in ['scalars','scalars_iv','phi_buf','phi_iv_buf','j_buf']:
        x=getattr(sr,k)
        if x is not None:
            if k.startswith('phi'):x=x[:sr.qlen]
            if k=='j_buf':x=x[:sr.jlen]
            obj['sr_'+k]=x.detach().cpu()
    torch.save(obj,path)

def rational_poly(coeff,exps,domains,rem):
    lo,hi=map(lambda x:F(float(x)),rem)
    for c,e in zip(coeff,exps):
        a=b=F(float(c))
        for (l,u),power in zip(domains,e):
            if not power:continue
            low=F(0) if power%2==0 and l<=0<=u else min(l**power,u**power)
            high=max(l**power,u**power);v=[a*low,a*high,b*low,b*high];a,b=min(v),max(v)
        lo+=a;hi+=b
    return lo,hi

def composed_point(st,eng,lane,z,h):
    full=eng.tables.exponents.cpu().tolist();spatial=eng.tables.spatial_index.cpu().tolist()
    mid=[rational_poly(row,[full[spatial[i]][1:] for i in st.tmv_sup.ids],[(v,v) for v in z],rem)
         for row,rem in zip(st.tmv[lane].cpu(),st.tmv_rem[lane].cpu())]
    return [rational_poly(row,[full[i] for i in st.pre_sup.ids],[(h,h)]+mid,rem)
            for row,rem in zip(st.pre[lane].cpu(),st.pre_rem[lane].cpu())]

def qualification(out,edge):
    receipts=[]
    for device in ['cpu','cuda']:
        eng,sched=setup(2,.05,device,edge)
        paths=[PREVIOUS/'qualification_v2'/('check_'+device+'_child'+str(i)+'.pt') for i in range(2)]
        st,cap,seeds=import_seed(paths,eng,sched)
        sr=make_symbolic_remainder(2,2,1000,device);sr.reserve(8)
        settings=config.Settings(step=.05,order=2,cutoff=0.,remainder_estimation=.02,mode='strict',device=device)
        code=compile_ode(['z*z','0'],['x','z'],order=1)
        for step in [2,3]:
            new,ok,new_sr,elapsed=attempt(st,sr,code,eng,sched,settings,cap)
            assert bool(ok.all()),(device,step,ok)
            st,sr=se.prune_state(new,eng),new_sr
            for lane,seed in enumerate(seeds):
                for j in range(-8,9):
                    z=[F(0),F(j,8)]
                    ies=[seed['exponents'][i][1:] for i in seed['initial_support']]
                    physical=[rational_poly(row,ies,[(v,v) for v in z],[0.,0.])[0] for row in seed['initial'][0]]
                    exact=[physical[0]+step*F(.05)*physical[1]**2,physical[1]]
                    bound=composed_point(st,eng,lane,z,F(.05))
                    assert all(a<=v<=b for (a,b),v in zip(bound,exact)),(device,step,lane,j)
            receipts.append(dict(device=device,step=step,all_two_leaves_accepted=True,exact_point_checks=68,committed_sr_length=sr.jlen,elapsed_s=elapsed))
        _,ok,_,_=attempt(st,sr,code,eng,sched,settings,torch.zeros_like(cap))
        assert not bool(ok.all()),'zero-cap negative case unexpectedly passed'
    result=dict(status='passed',rows=receipts,script_sha256=sha(__file__),failed_trial_leaves_committed_state_sr_unchanged=True,
                scope='Imported certified forcing child flowpipes represent completed step1. Fresh SR continuation to steps2/3; CPU/CUDA exact Fraction composition checks, no repeated time and no dropped seed remainder.')
    write(out/'RESULT.json',result)

def experiment(out,edge):
    cfg_path=N/'runs/archcomp_failure_20260923/quad_author_resolved.yaml'
    assert sha(cfg_path)=='940cb0b6188c3b127eee28df5d34fbda6d072f9e8b09cee8f251bd6b098f9144'
    cfg=yaml.safe_load(cfg_path.read_text())
    old=json.loads((PREVIOUS/'probe/RESULT.json').read_text())
    qualification=json.loads((out.parent/'qualification_v3/RESULT.json').read_text())
    assert qualification['status']=='passed' and qualification['script_sha256']==sha(__file__)
    assert old['original_step']==678 and old['original_lane']==1
    groups={'axis5':[f'axis5_half{i}_rebuilt' for i in range(2)],'axis10':[f'axis10_half{i}_rebuilt' for i in range(2)],'joint10_5':[f'joint10_5_{i}{j}_rebuilt' for i in range(2) for j in range(2)]}
    records=[]
    for group,names in groups.items():
        eng,sched=setup(16,.005,'cuda',edge)
        paths=[PREVIOUS/'probe'/(name+'.pt') for name in names]
        st,cap,_=import_seed(paths,eng,sched)
        assert bool((cap[...,0]==-.1).all() and (cap[...,1]==.1).all())
        sr=make_symbolic_remainder(len(names),16,1000,'cuda');sr.reserve(8)
        settings=config.Settings(step=.005,order=2,cutoff=1e-6,remainder_estimation=.1,mode='strict',device='cuda')
        code=compile_ode(cfg['dynamics_expressions'],[x['name'] for x in cfg['initial_set']],order=1)
        rows=[];save_state(out/(group+'_imported678.pt'),st,sr,eng,cap,678,'imported_certified_flowpipe')
        for step in [679,680]:
            events=[];settings=replace(settings,refinement_callback=events.append)
            new,ok,new_sr,elapsed=attempt(st,sr,code,eng,sched,settings,cap)
            raw=dict(group=group,step=step,accepted=ok.cpu().tolist(),all_accepted=bool(ok.all()),advance_s=elapsed,trial_sr_length=new_sr.jlen,committed_parent_unchanged=True)
            save_state(out/(group+'_trial'+str(step)+'.pt'),new,new_sr,eng,cap,step,'trial_no_commit_if_any_leaf_failed')
            write(out/(group+'_trace'+str(step)+'.json'),events)
            rows.append(raw);print(json.dumps(raw),flush=True)
            if not bool(ok.all()):break
            st,sr=se.prune_state(new,eng),new_sr
            sr.reset_if_full()
            save_state(out/(group+'_committed'+str(step)+'.pt'),st,sr,eng,cap,step,'committed_before_controller_handoff')
        records.append(dict(group=group,seeds=[dict(name=n,path=str(p),sha256=sha(p)) for n,p in zip(names,paths)],steps=rows,complete_to_control_boundary=bool(rows[-1]['step']==680 and rows[-1]['all_accepted'])))
    result=dict(status='completed',groups=records,original_step=678,original_lane=1,config_sha256=sha(cfg_path),script_sha256=sha(__file__),environment={k:v for k,v in os.environ.items() if k.startswith('FLOWSTAR_')},extensions=ORIGINAL['extensions'],source_unchanged=not subprocess.check_output(['git','-C',str(E),'status','--porcelain'],text=True).strip(),
                scope='Conservative local-box restart. Imported actual accepted child p/R certifies step678; new SR starts empty, first new advance679. Stop at680 before endpoint handoff/controller refresh. No original historical SR retention, no full benchmark/root-prefix claim, no extra splitting or accepted-leaf-only success.')
    write(out/'RESULT.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    edge=bootstrap()
    (qualification if args.check_only else experiment)(args.output,edge)
