"""AST and fake driver-boundary check; never imports or invokes CUDA."""
from pathlib import Path
from types import SimpleNamespace as NS
import argparse,ast,copy,hashlib,json,sys

HERE=Path(__file__).parent
BASE_SHA='660c75cf22346e71d5044a60b5bbd62320ed38e542727cbcb6805fad909095e0'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def definition(tree,name):return next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==name)
def dump(node):return ast.dump(node,include_attributes=False)

def main_ast(tree,base):
    candidate=copy.deepcopy(definition(tree,'main'));original=definition(base,'main')
    assert isinstance(candidate.body[0],ast.Assert)
    assert ast.unparse(candidate.body[0].test)=="args.mode == 'bounded40' and args.source40 is None"
    candidate.body.pop(0)
    outer=next(x for x in candidate.body if isinstance(x,ast.Try))
    # Explicit failure status is required before finally's success conversion.
    error_update=outer.handlers[0].body[0]
    assert isinstance(error_update,ast.Expr) and isinstance(error_update.value,ast.Call)
    status=[v for v in error_update.value.keywords if v.arg=='status']
    assert len(status)==1 and isinstance(status[0].value,ast.Constant) and status[0].value.value=='exception'
    error_update.value.keywords=[v for v in error_update.value.keywords if v.arg!='status']
    ctx_block=next(x for x in outer.finalbody if isinstance(x,ast.If) and ast.unparse(x.test)=="'ctx' in locals()")
    profile=[x for x in ctx_block.body if isinstance(x,ast.If) and ast.unparse(x.test)=="hasattr(ctx, 'phase_profile')"]
    assert len(profile)==1
    profile_text=ast.unparse(profile[0])
    assert profile_text.index('abort_step()') < profile_text.index('restore()') < profile_text.index('phase_profile_log.close()')
    assert ctx_block.body.index(profile[0]) < next(i for i,x in enumerate(ctx_block.body) if 'ctx.eager_segments.restore()'==ast.unparse(x))
    ctx_block.body.remove(profile[0])
    # The new diagnostic status gate must require the actual comparison receipt.
    success=next(x for x in outer.finalbody if isinstance(x,ast.If) and ast.unparse(x.test)=="result['status'] == 'bounded_prefix_completed'")
    assert any(isinstance(x,ast.Assert) and ast.unparse(x.test)=="'reference40_comparison' in result" for x in success.body)
    outer.finalbody.remove(success)
    diagnostic=next(x for x in outer.finalbody if isinstance(x,ast.Expr) and ast.unparse(x)=="result.update(diagnostic_only=True, no_speedup_claim=True)")
    outer.finalbody.remove(diagnostic)
    assert dump(candidate)==dump(original),'Numerical main differs outside reviewed diagnostic/failure deltas'
    return dict(main_AST_equal_after_explicit_diagnostic_deltas=True,failure_status_cannot_promote_success=True,
        original_numerical_driver_and_comparison_gates_preserved=True,profiler_restored_before_eager=True)


def boundary(tree,out):
    events=[];output=object();error=RuntimeError('raw failure')
    class Profile:
        active=None
        def begin_step(self,step,eng):
            assert self.active is None;self.active=step;events.append(('begin',step))
        def finish_step(self,step,*,original_sync_completed,advance_wall_s):
            assert self.active==step and original_sync_completed is True and ctx.synced
            assert advance_wall_s==.25
            self.active=None;events.append(('finish',step))
        def abort_step(self):self.active=None;events.append(('abort',))
    profile=Profile()
    class Base:
        def __init__(self,ctx_,args):self.ctx=ctx_;self.completed=0;events.append(('base_init',))
        def advance_sparse(self,*args):
            events.append(('raw_advance',));self.ctx.synced=False
            assert args==operands
            if self.ctx.raise_raw:raise error
            self.ctx.synced=True;events.append(('original_sync',));self.completed+=1
            self.observe(.25)
            return output
        def observe(self,duration):
            events.append(('observer',));assert profile.active is None
            if self.ctx.raise_observer:raise error
            return output
    fake=NS(graphing=object(),weighted_validation=object())
    saved={k:sys.modules.get(k) for k in ['flowstar_gpu']};sys.modules['flowstar_gpu']=fake
    marker=object()
    def install(*args,record):
        assert args==(marker,fake.graphing,marker,fake.weighted_validation,marker)
        assert events==[('base_init',)];return profile
    ctx=NS(torch=marker,c=NS(se=marker),host=marker,phase_profile_helper=NS(install=install),
        synced=False,raise_raw=False,raise_observer=False)
    env={'s':NS(Runtime=Base),'json':json}
    cls=definition(tree,'Runtime')
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'<actual Runtime AST>','exec'),env)
    args=NS(output=out)
    try:
        x=env['Runtime'](ctx,args);operands=tuple(object() for _ in range(7))
        assert x.advance_sparse(*operands) is output
        assert events==[('base_init',),('begin',1),('raw_advance',),('original_sync',),('finish',1),('observer',)]
        events.clear();ctx.raise_raw=True
        try:x.advance_sparse(*operands)
        except RuntimeError as caught:assert caught is error
        else:raise AssertionError('Raw failure not forwarded')
        assert events==[('begin',2),('raw_advance',),('abort',)] and profile.active is None
        events.clear();ctx.raise_raw=False;ctx.raise_observer=True
        try:x.advance_sparse(*operands)
        except RuntimeError as caught:assert caught is error
        else:raise AssertionError('Observer failure not forwarded')
        assert events==[('begin',2),('raw_advance',),('original_sync',),('finish',2),('observer',),('abort',)]
        assert profile.active is None
        ctx.phase_profile_log.close()
    finally:
        for name,value in saved.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value
    return dict(actual_Runtime_AST_executed_with_fake_driver=True,after_original_sync_before_observer=True,
        original_inputs_and_return_object_preserved=True,original_exceptions_forwarded_and_abort_called=True,
        install_after_base_Runtime_initialization=True)


def initialization(tree):
    constants={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant):
            for target in n.targets:
                if isinstance(target,ast.Name):constants[target.id]=n.value.value
    args=NS(profile_checker=Path('/fake/checker'),profile_check=Path('/fake/check'),profile_reference40=Path('/fake/reference'))
    here=Path('/fake/entry');source=here/'entry.py';candidate_sha='b'*64
    hashes={here/'phase_profile.py':constants['PROFILE_SHA'],args.profile_checker:constants['PROFILE_CHECK_SHA'],
        args.profile_check/'RESULT.json':constants['PROFILE_RESULT_SHA'],args.profile_reference40/'INPUT.json':constants['OWN40_INPUT_SHA'],
        args.profile_reference40/'RESULT.json':constants['OWN40_RESULT_SHA'],source:candidate_sha}
    identity=dict(script_sha256=constants['ENTRY_SHA'],policy='original660c',unchanged='algorithm settings')
    ctx=NS(identity=dict(identity));events=[];helper=object()
    def old_init(a):assert a is args;events.append('initialize');return ctx
    def gate(path,sid):
        assert path==args.profile_reference40 and sid==identity
        events.append('own40_gate_before_identity_change')
    def read(path):
        assert path==args.profile_check/'RESULT.json'
        return dict(status='passed',device='cpu',adapter_sha256=constants['PROFILE_SHA'],script_sha256=constants['PROFILE_CHECK_SHA'])
    def load(name,path,digest):assert path==here/'phase_profile.py' and digest==constants['PROFILE_SHA'];return helper
    env=dict(constants,HERE=here,__file__=str(source),sha=lambda p:hashes[Path(p)],read=read,load=load,s=NS(initialize=old_init,own40_gate=gate))
    exec(compile(ast.Module(body=[definition(tree,'initialize')],type_ignores=[]),'<actual initialize AST>','exec'),env)
    assert env['initialize'](args) is ctx and ctx.phase_profile_helper is helper
    assert events==['initialize','own40_gate_before_identity_change']
    assert ctx.identity['algorithm_identity']==identity and ctx.identity['script_sha256']==candidate_sha
    assert ctx.identity['policy']==constants['POLICY'] and ctx.identity['diagnostic_only'] is True
    return dict(actual_initialize_AST_executed_with_fake_files=True,own40_gate_sees_original_identity=True,
        candidate_identity_separate_from_algorithm_identity=True)


def main(args):
    assert sha(args.entry)==args.expected_sha
    base=HERE/'run_fullbatch_p3_selective_audit.py';assert sha(base)==BASE_SHA
    tree=ast.parse(args.entry.read_text());old=ast.parse(base.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    result=dict(status='passed',device='cpu',script_sha256=sha(__file__),entry_sha256=sha(args.entry),base_sha256=BASE_SHA,
        **main_ast(tree,old),**boundary(tree,args.output),**initialization(tree),CUDA_tested=False,
        scope='Actual entry AST and fake file/driver lifecycle only. No actual numerical/CUDA execution or byte pairing.')
    (args.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--entry',type=Path,default=HERE/'run_fullbatch_p3_phase_profile.py')
    p.add_argument('--expected-sha',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
