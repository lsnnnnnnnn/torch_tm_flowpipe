"""Actual AST/fake-admission check for resource-only profile40; no CUDA imports."""
from pathlib import Path
from types import SimpleNamespace as NS
import argparse,ast,copy,hashlib,importlib.util,json

HERE=Path(__file__).parent
ROOT=HERE.parents[1]
BASE_SHA='b4d9927d89b61b2829de0733c00344b70282645c6254d6d2db61fde65927498d'
WATCH_SHA='4298689de9ccf8a97c0007334dcdbe6aedeb246b8acead81a4b1873941a56350'
PROFILE_SHA='0398dc1bb032fe85c7ef0a192e5667a12890934d85a71d5ab717adf360023166'
OLD_CHECK_SHA='266c7f98f11e472df05b05d24f7e6dafdd55d011f10328833658d30456eade77'
OLD_RESULT_SHA='4f69be5d9b05949dad859d8f613bb026733feb8f6fc6f33411e4960a92fa44c8'
GIB=2**30

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def definition(tree,name):return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)

def structure(tree,old):
    assert dump(definition(tree,'main'))==dump(definition(old,'main'))
    alias=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='Runtime' for t in n.targets))
    sentinel=object();env=dict(base=NS(Runtime=sentinel))
    exec(compile(ast.Module(body=[alias],type_ignores=[]),'<actual Runtime alias>','exec'),env)
    assert env['Runtime'] is sentinel
    comparisons=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='compare_reference40' for t in n.targets))
    assert ast.unparse(comparisons)=='compare_reference40 = base.compare_reference40'
    cli=copy.deepcopy(tree.body[-1]);prior=old.body[-1]
    path_loop=next(n for n in cli.body if isinstance(n,ast.For))
    assert [n.value for n in path_loop.iter.elts].count('resource-check')==1
    path_loop.iter.elts=[n for n in path_loop.iter.elts if n.value!='resource-check']
    extra=next(n for n in cli.body if isinstance(n,ast.Expr) and ast.unparse(n)=="p.add_argument('--resource-check-sha256', required=True)")
    cli.body.remove(extra);assert dump(cli)==dump(prior)
    assert not any(isinstance(n,(ast.Assign,ast.AugAssign)) and any(isinstance(t,ast.Name) and t.id=='__file__' for t in getattr(n,'targets',[])) for n in ast.walk(tree))
    return dict(main_AST_identical=True,Runtime_exact_same_object=True,comparison_exact_same_object=True,CLI_only_resource_gate_added=True)

def admission(tree,entry):
    constants={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:constants[n.targets[0].id]=eval(compile(ast.Expression(n.value),'<constant>','eval'),{'__builtins__':{},**constants})
            except (NameError,TypeError,AttributeError):pass
    required=['main_AST_identical','Runtime_exact_same_object','original_gates_before_allocation_increase',
              'previous_identity_unchanged','resource_identity_separate','invalid_admissions_do_not_raise_budget']
    assert (constants['ALLOCATOR_BYTES'],constants['GPU_GUARD_BYTES'],constants['RSS_GUARD_BYTES'])==(27*GIB//2,14*GIB,23*GIB//2)
    entry_sha=sha(entry);checker_sha=sha(__file__)
    def trial(total=16*GIB,fail_old=False,wrong_old=False,wrong_watch=False,wrong_receipt=False,bad_pass=False):
        events=[];args=NS(resource_check=Path('/fake/qualification'),resource_check_sha256='f'*64)
        original_numeric={'settings':{'h':.005,'P':3,'validation':4,'cap':[-.1,.1]},'B':1024,'SR_capacity':1000,'torch_allocation_cap_bytes':11*GIB,'external_watch_limit_bytes':23*GIB//2}
        previous=dict(script_sha256=BASE_SHA,torch_allocation_cap_bytes=11*GIB,external_watch_limit_bytes=23*GIB//2,
                      algorithm_identity=original_numeric,diagnostic_only=True,limitation='Original profile40 limitations')
        before=json.dumps(previous,sort_keys=True,allow_nan=False)
        cuda=NS(get_device_properties=lambda device:NS(total_memory=total),set_per_process_memory_fraction=lambda fraction,device:events.append(('set',fraction,device)))
        ctx=NS(identity=previous,torch=NS(cuda=cuda),untouched=object())
        fake_result=dict(status='passed',device='cpu',CUDA_tested=False,entry_sha256=entry_sha,base_sha256=BASE_SHA,
                         script_sha256=checker_sha,watchdog_sha256=WATCH_SHA,profile_helper_sha256=PROFILE_SHA,**dict.fromkeys(required,True))
        if bad_pass:fake_result['original_gates_before_allocation_increase']=False
        def digest(path):
            path=Path(path)
            if path==HERE/'nncs_watchdog_gpu14.py':return 'bad' if wrong_watch else WATCH_SHA
            if path==entry:return entry_sha
            if path==HERE/'check_phase_profile_gpu14_cpu.py':return checker_sha
            assert path==args.resource_check/'RESULT.json';return 'bad' if wrong_receipt else args.resource_check_sha256
        def read(path):assert path==args.resource_check/'RESULT.json';events.append(('resource_gate_read',));return fake_result
        error=ValueError('original own40 gate failed')
        def initialize(a):
            assert a is args;events.append(('old_initialize',))
            cuda.set_per_process_memory_fraction(11*GIB/total,0)
            events.append(('old_original_and_profile_and_own40_gates',))
            if fail_old:raise error
            if wrong_old:ctx.identity['script_sha256']='wrong'
            return ctx
        env=dict(constants,HERE=HERE,__file__=str(entry),sha=digest,read=read,hashlib=hashlib,json=json,base=NS(initialize=initialize))
        code=ast.Module(body=[definition(tree,'resource_gate'),definition(tree,'initialize')],type_ignores=[])
        exec(compile(code,'<actual resource gate and initialize AST>','exec'),env)
        try:answer=env['initialize'](args)
        except BaseException as exc:return events,exc,None,previous,before,original_numeric
        assert answer is ctx
        return events,None,ctx,previous,before,original_numeric
    events,error,ctx,previous,before,numeric=trial()
    assert error is None
    assert events==[('resource_gate_read',),('old_initialize',),('set',11/16,0),('old_original_and_profile_and_own40_gates',),('set',13.5/16,0)]
    assert json.dumps(previous,sort_keys=True,allow_nan=False)==before
    assert ctx.identity['algorithm_identity'] is numeric
    assert ctx.identity['base_profile_identity_sha256']==hashlib.sha256(before.encode()).hexdigest()
    assert ctx.identity['script_sha256']==entry_sha and ctx.identity['base_profile_entry_sha256']==BASE_SHA
    assert ctx.identity['torch_allocation_cap_bytes']==27*GIB//2 and ctx.identity['external_watch_limit_bytes']==14*GIB
    budget=ctx.identity['resource_budget'];assert budget['rss_guard_bytes']==23*GIB//2 and budget['allocation_set_after_original_bootstrap_and_admission'] is True
    negatives=[]
    for label,kw in [('failed_original_gate',dict(fail_old=True)),('incorrect_old_identity',dict(wrong_old=True)),('insufficient_total_memory',dict(total=14*GIB)),('wrong_watchdog',dict(wrong_watch=True)),('wrong_CPU_receipt_hash',dict(wrong_receipt=True)),('failed_CPU_admission',dict(bad_pass=True))]:
        ev,err,*_=trial(**kw);assert err is not None
        assert all(e[0]!='set' or e[1]==11*GIB/kw.get('total',16*GIB) for e in ev)
        negatives.append(label)
    return dict(original_gates_before_allocation_increase=True,previous_identity_unchanged=True,resource_identity_separate=True,
                invalid_admissions_do_not_raise_budget=True,negative_admissions=negatives,
                initialization_AST_executed=True,resource_gate_AST_executed=True)

def main(args):
    assert sha(args.entry)==args.expected_sha
    base=HERE/'run_fullbatch_p3_phase_profile.py';checker=HERE/'check_phase_profile_entry_cpu.py'
    for path,expected in [(base,BASE_SHA),(checker,OLD_CHECK_SHA),(HERE/'phase_profile.py',PROFILE_SHA),(HERE/'nncs_watchdog_gpu14.py',WATCH_SHA)]:assert sha(path)==expected
    old_result=ROOT/'results/quad_targeted_recovery_20260927/phase_profile_entry_cpu_v1_20260929/RESULT.json'
    assert sha(old_result)==OLD_RESULT_SHA
    prior=json.loads(old_result.read_text());assert prior['status']=='passed' and prior['entry_sha256']==BASE_SHA and prior['CUDA_tested'] is False
    tree,old=ast.parse(args.entry.read_text()),ast.parse(base.read_text())
    structural=structure(tree,old);initialized=admission(tree,args.entry)
    # Reuse the already-qualified checker for old initialization/runtime boundary;
    # this imports only its stdlib module and executes fake AST, never the solver.
    spec=importlib.util.spec_from_file_location('old_profile_entry_cpu_check',checker);oldcheck=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldcheck)
    args.output.mkdir(parents=True,exist_ok=False)
    original_initialization=oldcheck.initialization(old)
    original_boundary=oldcheck.boundary(old,args.output)
    result=dict(status='passed',device='cpu',entry_sha256=sha(args.entry),script_sha256=sha(__file__),base_sha256=BASE_SHA,
        watchdog_sha256=WATCH_SHA,profile_helper_sha256=PROFILE_SHA,old_entry_CPU_result_sha256=OLD_RESULT_SHA,
        **structural,**initialized,original_initialization_recheck=original_initialization,original_profile_boundary_recheck=original_boundary,
        CUDA_tested=False,scope='Actual resource wrapper/main AST and fake setter/file/old driver boundaries only; no CUDA allocations, numerical integration or new byte pairing.')
    (args.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',type=Path,default=HERE/'run_fullbatch_p3_phase_profile_gpu14.py');p.add_argument('--expected-sha',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
