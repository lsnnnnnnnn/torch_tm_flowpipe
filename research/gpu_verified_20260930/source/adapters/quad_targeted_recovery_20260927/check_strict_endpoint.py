"""Small CPU/CUDA Fraction and nonempty-SR gates for strict_endpoint.py.

CUDA requires the existing pinned continuation bootstrap, never compilation.
Use --device cpu locally; no frozen source or archived evidence is changed.
"""
from pathlib import Path
from fractions import Fraction as F
from types import SimpleNamespace
import argparse, copy, hashlib, importlib.util, itertools, json, os, sys


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec); sys.modules[name] = obj; spec.loader.exec_module(obj)
    return obj


def main(a):
    identity = json.loads(a.identity.read_text()); identity = identity.get('engine', identity)
    for path, value in identity['python_sha256'].items(): assert sha(a.engine_root / path) == value, path
    os.environ.update(FLOWSTAR_COMPOSITION='horner', FLOWSTAR_GLUE='graph', FLOWSTAR_SUPPORT_POLICY='structural',
                      FLOWSTAR_INJECTIVE_MAPS='1', FLOWSTAR_INJECTIVE_GLUE='1',
                      FLOWSTAR_VALIDATION_POLICY='solution_plus_one', FLOWSTAR_CENTER_NORMALIZATION='1',
                      FLOWSTAR_WEIGHTED_VALIDATION='0', FLOWSTAR_RECENTER_VALIDATION='1',
                      FLOWSTAR_EARLY_WEIGHTED='1', FLOWSTAR_SELF_MAP_RETRIES='8')
    sys.path.insert(0, str(a.engine_root / 'src'))
    import torch
    from flowstar_gpu import support as sp, interval as iv, sparse_exec as se, config
    from flowstar_gpu.monomials import build_tables
    from flowstar_gpu.polynomial import build_step_tables
    from flowstar_gpu.composition import build_schedule
    from flowstar_gpu.ode_compiler import compile_ode
    from flowstar_gpu.symbolic_remainder import make_symbolic_remainder
    adapter = load('checked_strict_endpoint', Path(__file__).with_name('strict_endpoint.py'))
    edge = None
    if a.device == 'cuda':
        assert a.common and sha(a.common) == '1c85e04988028f7ac95d945eeff9ef3e0936ee306c1f0a80650bb06d8073d985'
        common = load('strict_endpoint_pinned_bootstrap', a.common)
        assert common.E.resolve() == a.engine_root.resolve()
        edge = common.bootstrap()
    else:
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    tab = build_tables(2, 3).to(a.device); eng = sp.SparseEngine(tab, build_step_tables(tab, .1), a.device)
    if edge is not None: eng.horner_edge_kernel = edge.horner_edge
    sched = build_schedule(2, 3, a.device)
    def same(x, y): return x.shape == y.shape and x.dtype == y.dtype and torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
    def tensor_id(x): return None if x is None else hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    def history(sr):
        return {k: tensor_id(getattr(sr, k)[:sr.qlen] if k.startswith('phi') and getattr(sr, k) is not None
                             else getattr(sr, k)[:sr.jlen] if k == 'j_buf' else getattr(sr, k))
                for k in ['scalars', 'scalars_iv', 'phi_buf', 'phi_iv_buf', 'j_buf']} | dict(qlen=sr.qlen,jlen=sr.jlen,max_size=sr.max_size)
    def point_poly(row, exps, z, h):
        return sum((F(float(v)) * h**e[0] * z[0]**e[1] * z[1]**e[2] for v, e in zip(row, exps)), F(0))
    def interval_poly(row, exps, boxes, rem):
        lo, hi = map(lambda x:F(float(x)), rem)
        for v, e in zip(row, exps):
            lower = upper = F(float(v))
            for (left, right), degree in zip(boxes, e):
                if not degree: continue
                low = F(0) if degree % 2 == 0 and left <= 0 <= right else min(left**degree, right**degree)
                high = max(left**degree, right**degree)
                terms = [lower*low,lower*high,upper*low,upper*high]; lower, upper = min(terms), max(terms)
            lo += lower; hi += upper
        return lo, hi
    def composed(st, z):
        full = tab.exponents.cpu().tolist(); spatial = tab.spatial_index.cpu().tolist(); rows = []
        for lane in range(2):
            intermediate = [interval_poly(row, [full[spatial[i]][1:] for i in st.tmv_sup.ids], [(v,v) for v in z], rem)
                            for row, rem in zip(st.tmv[lane].cpu(), st.tmv_rem[lane].cpu())]
            rows.append([interval_poly(row, [full[i] for i in st.pre_sup.ids], [(F(.1),F(.1))]+intermediate, rem)
                         for row, rem in zip(st.pre[lane].cpu(), st.pre_rem[lane].cpu())])
        return rows

    # Adversarial cubic time collisions plus spatial terms and subnormals.
    exps = [(0,0,0),(1,0,0),(2,0,0),(3,0,0),(0,1,0),(1,1,0),(2,0,1),(1,1,1)]
    lookup = {tuple(e):i for i,e in enumerate(tab.exponents.cpu().tolist())}
    support = sp.make_support(2,3,False,tuple(sorted(lookup[e] for e in exps)))
    boxes = torch.zeros(2,2,2,dtype=torch.float64,device=a.device)
    st = se.initial_sparse_state(boxes,eng,sched)
    st.pre = torch.zeros(2,2,support.size,dtype=torch.float64,device=a.device); st.pre_sup = support
    tiny = float.fromhex('0x0.0000000000001p-1022')
    coefficients = [[[.1,1e10,-1e11,.7,.3,-.2,.9,-.4],[0.,tiny,-tiny,tiny,0.,tiny,0.,0.]],
                    [[-.3,-1e10,1e11,-.9,-.7,.2,-.1,.6],[.2,.7,-.4,.3,-.2,.1,.4,-.5]]]
    for lane in range(2):
        for row in range(2):
            for exponent, value in zip(exps,coefficients[lane][row]): st.pre[lane,row,support.ids.index(lookup[exponent])] = value
    st.pre_rem = torch.tensor([[[-.004,.006],[-.0,0.]], [[-.003,.007],[-.002,.004]]],dtype=torch.float64,device=a.device)
    before = copy.deepcopy(st); untouched = [tensor_id(st.tmv),tensor_id(st.tmv_rem),tensor_id(st.status)]
    error = adapter.end_of_time_s(st,eng)
    assert untouched == [tensor_id(st.tmv),tensor_id(st.tmv_rem),tensor_id(st.status)]
    assert bool((error != 0).any())
    exact_error_checks = 0
    after_exps = [tuple(e) for e in tab.exponents[list(st.pre_sup.ids)].cpu().tolist()]
    before_exps = [tuple(e) for e in tab.exponents[list(before.pre_sup.ids)].cpu().tolist()]
    for lane in range(2):
        for row in range(2):
            exact = {}
            for coefficient, exponent in zip(before.pre[lane,row].cpu().tolist(), before_exps):
                e=(0,)+exponent[1:];exact[e]=exact.get(e,F(0))+F(coefficient)*F(.1)**exponent[0]
            for coefficient, exponent in zip(st.pre[lane,row].cpu().tolist(), after_exps): exact[exponent]-=F(coefficient)
            lo=hi=F(0)
            for e,v in exact.items():
                low,high = ((F(-1),F(1)) if any(p%2 for p in e[1:]) else (F(0),F(1))) if any(e[1:]) else (F(1),F(1))
                ends=[v*low,v*high];lo+=min(ends);hi+=max(ends)
            assert F(float(error[lane,row,0])) <= lo <= hi <= F(float(error[lane,row,1]))
            assert F(float(st.pre_rem[lane,row,0])) <= F(float(before.pre_rem[lane,row,0]))+lo
            assert F(float(st.pre_rem[lane,row,1])) >= F(float(before.pre_rem[lane,row,1]))+hi
            exact_error_checks += 1
    negative = []
    for name in ['nan', 'inverted_remainder', 'inactive_leaf', 'overflow']:
        invalid=copy.deepcopy(before)
        if name=='nan': invalid.pre[1,0,0]=float('nan')
        if name=='inverted_remainder': invalid.pre_rem[1,0]=torch.tensor([.2,-.2],device=a.device)
        if name=='inactive_leaf': invalid.status[1]=1
        if name=='overflow':
            invalid.pre.zero_();invalid.pre[0,0,0]=torch.finfo(torch.float64).max
            invalid.pre[0,0,support.ids.index(lookup[(1,0,0)])]=torch.finfo(torch.float64).max
        snapshot=[tensor_id(invalid.pre),tensor_id(invalid.pre_rem),invalid.pre_sup]
        try: adapter.end_of_time_s(invalid,eng)
        except ValueError: pass
        else: raise AssertionError('invalid boundary accepted: '+name)
        assert snapshot==[tensor_id(invalid.pre),tensor_id(invalid.pre_rem),invalid.pre_sup]
        negative.append(name)

    # Produce a real two-step state with nonzero ordinary remainder and live SR.
    boxes=torch.tensor([[[.125,.25],[-.75,.5]],[[-.5,-.25],[.25,.75]]],dtype=torch.float64,device=a.device)
    state=se.initial_sparse_state(boxes,eng,sched); initial=copy.deepcopy(state)
    sr=make_symbolic_remainder(2,2,1000,a.device);sr.reserve(8)
    cap=torch.tensor([-.05,.05],dtype=torch.float64,device=a.device).expand(2,2,2).clone()
    settings=config.Settings(step=.1,order=3,cutoff=0.,remainder_estimation=.05,mode='strict',device=a.device)
    code=compile_ode(['z*z','0'],['x','z'],order=2)
    for _ in range(2):
        state,ok=se.advance_sparse(state,code,eng,sched,settings,cap,sr);assert bool(ok.all())
        state=se.prune_state(state,eng)
    assert sr.qlen==sr.jlen==2 and bool(state.pre_rem.any())
    old_history=history(sr); old_j=sr.j_buf[:sr.jlen].clone(); state_before=copy.deepcopy(state)
    sr_before=copy.deepcopy(sr)
    charged=adapter.end_of_time_s(state,eng)
    assert history(sr)==old_history
    _,_,second_error=sp.evaluate_time_end_s_with_roundoff(state.pre,eng,state.pre_sup)
    expected_x0_rem=iv.add(state.pre_rem,second_error)
    captured=[]; appended=[]; original_compose=se.compose_s; original_append=sr.append_j
    def compose(*args,**kwargs):
        captured.append(args[1].clone());return original_compose(*args,**kwargs)
    def append(value):
        appended.append(value.clone());return original_append(value)
    se.compose_s=compose;sr.append_j=append
    try: advanced,ok=se.advance_sparse(state,code,eng,sched,settings,cap,sr)
    finally: se.compose_s=original_compose;sr.append_j=original_append
    assert bool(ok.all()) and sr.qlen==sr.jlen==3
    assert len(captured)==1 and same(captured[0],expected_x0_rem)
    assert len(appended)==1 and same(appended[0],sr.j_buf[2])
    assert same(old_j,sr.j_buf[:2])
    checked=0
    for z in itertools.product([F(-1),F(-1,2),F(0),F(1,2),F(1)],repeat=2):
        bounds=composed(advanced,z)
        for lane in range(2):
            ies=tab.exponents[list(initial.pre_sup.ids)].cpu().tolist()
            x0,z0=[point_poly(row,ies,z,F(0)) for row in initial.pre[lane].cpu()]
            target=[x0+3*F(.1)*z0*z0,z0]
            assert all(lo<=x<=hi for (lo,hi),x in zip(bounds[lane],target));checked+=2
    # Oracle sensitivity: a declared fresh marker must survive the nonempty SR
    # path. Its deliberately omitted twin must fail the same containment test.
    marker=torch.zeros(2,2,2,dtype=torch.float64,device=a.device)
    marker[:,0,0]=-1/64;marker[:,0,1]=1/32
    static_code=compile_ode(['0','0'],['x','z'],order=2)
    marker_outputs=[]
    for include in [True,False]:
        marked=copy.deepcopy(state_before); marked_sr=copy.deepcopy(sr_before)
        adapter.end_of_time_s(marked,eng)
        if include: marked.pre_rem=iv.add(marked.pre_rem,marker)
        new,ok=se.advance_sparse(marked,static_code,eng,sched,settings,cap,marked_sr)
        assert bool(ok.all()) and marked_sr.qlen==marked_sr.jlen==3
        marker_outputs.append(new)
    marker_checks=omitted_misses=0
    for z in itertools.product([F(-1),F(0),F(1)],repeat=2):
        good,bad=[composed(value,z) for value in marker_outputs]
        for lane in range(2):
            ies=tab.exponents[list(initial.pre_sup.ids)].cpu().tolist()
            x0,z0=[point_poly(row,ies,z,F(0)) for row in initial.pre[lane].cpu()]
            for extra in [F(-1,64),F(1,32)]:
                value=x0+2*F(.1)*z0*z0+extra
                assert good[lane][0][0]<=value<=good[lane][0][1]
                omitted_misses+=not(bad[lane][0][0]<=value<=bad[lane][0][1]);marker_checks+=1
    assert omitted_misses>0,'oracle did not detect deliberately omitted fresh uncertainty'
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='passed',device=a.device,torch_version=torch.__version__,engine_root=str(a.engine_root),
                engine_python_sha256=identity['python_sha256'],adapter_sha256=sha(Path(adapter.__file__)),
                check_script_sha256=sha(__file__),fraction_full_box_error_checks=exact_error_checks,
                negative_fail_closed=negative,nonempty_sr_boundary_unchanged=True,
                fresh_compose_received_exact_charged_remainder=True,normal_single_J_append=True,
                old_J_bytes_unchanged=True,ordinary_remainder_nonzero=True,analytic_composed_component_checks=checked,
                fresh_marker_component_checks=marker_checks,deliberately_omitted_marker_misses=omitted_misses,
                qlen_before=2,qlen_after=3,scope='Strict endpoint arithmetic and its small nonempty-SR carry path; no CROWN or general control-injection qualification')
    (a.output/'RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='engine_python_sha256'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--engine-root',type=Path,required=True)
    p.add_argument('--identity',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--common',type=Path);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
