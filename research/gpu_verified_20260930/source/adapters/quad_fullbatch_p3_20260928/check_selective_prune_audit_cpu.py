"""CPU protocol check using the real frozen adapter and fake CUDA operations.

No tensor math, CUDA context, engine advance or real memory reduction is tested.
"""
from pathlib import Path
from types import SimpleNamespace as NS
import argparse,collections,hashlib,importlib.util,inspect,json,sys,time,weakref

HERE=Path(__file__).parent
DEFAULT_GRAPHING=Path('/private/tmp/quad_endpoint_a3fb_cpu_src/src/flowstar_gpu/graphing.py')
GRAPHING_SHA='62f1b07f581b4c996525e5fa1b17796eee6aba0d39bf67de55dfb3e1ae69f4d4'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def check(graphing_path):
    start=time.perf_counter();assert sha(graphing_path)==GRAPHING_SHA
    helper=load('selective_cpu_helper',HERE/'selective_prune_audit.py')
    assert sha(HERE/'working_graph_prune.py')==helper.WORKING_PRUNE_SHA
    adapter=load('selective_cpu_adapter',HERE/'working_graph_prune.py')
    graphing=load('selective_cpu_graphing',graphing_path)
    calls=[];capturing=[False]
    cuda=NS(is_current_stream_capturing=lambda:capturing[0],synchronize=lambda d:calls.append('sync'),
        empty_cache=lambda:calls.append('empty'),memory_allocated=lambda d:100,
        memory_reserved=lambda d:200,mem_get_info=lambda d:(800,1000))
    mock=NS(cuda=cuda);graphing.torch=mock
    class Cell:
        is_cuda=True
        def __init__(self,value):self.value=value
        def copy_(self,other):self.value=other.value
    class Graph:
        def replay(self):pass
    def entry():return ([Cell(0)],Graph(),Cell(2),())
    def cache(keys):
        c=graphing.GraphCache.__new__(graphing.GraphCache)
        c.device='cuda:0';c.enabled=True;c.captures=len(keys);c.hits=0;c._capture_stream=object()
        c._segs={k:entry() for k in keys};return c
    def engine(c):return NS(tables=NS(n=16,k=3),device='cuda:0',_graphs=c)
    def snapshot(binding,c):
        v=inspect.getclosurevars(binding.finish_step).nonlocals
        return dict(active=id(v['active']),target=id(v['target']),used_id=id(v['used']),initial_id=id(v['initial']),
            used=sorted(map(id,v['used'])),initial=sorted(map(id,v['initial'])),
            entries=sorted((id(k),id(e),id(e[1]),e[0][0].value,e[2].value) for k,e in c._segs.items()),
            captures=c.captures,hits=c.hits,stream=id(c._capture_stream),
            counters=dict(binding.counters),rows=json.dumps(binding.rows,sort_keys=True),calls=list(calls))
    def scalars(value):
        if isinstance(value,dict):return all(type(k) is str and scalars(v) for k,v in value.items())
        if isinstance(value,list):return all(scalars(v) for v in value)
        return type(value) in (str,int,bool)
    negative=[];positive=[]
    def rejected(name,fn):
        try:fn()
        except (AssertionError,KeyError):negative.append(name)
        else:raise AssertionError('Expected rejection: '+name)
    original=graphing.GraphCache.run
    hot=('compose','hot');stale=[('glue','emit'),(object(),'refine'),(123,)]
    for keys in [[hot],stale+[hot]]:
        calls.clear();c=cache(keys);eng=engine(c);binding=adapter.install(mock,graphing)
        try:
            binding.begin_step(eng);c.run(hot,None,[Cell(7)])
            weak=[weakref.ref(c._segs[k][1]) for k in keys if k!=hot]
            before=snapshot(binding,c);plan=helper.peek(binding,eng)
            assert helper.assert_same_plan(binding,eng,plan)==plan
            assert snapshot(binding,c)==before and scalars(plan)
            json.loads(json.dumps(plan,allow_nan=False))
            assert plan['current_key_ids']==sorted(map(id,keys)) and plan['used_key_ids']==[id(hot)]
            assert plan['stale_count']==len(keys)-1
            row=binding.finish_step(eng);assert helper.assert_finished(plan,row)
            assert calls==(['sync','empty'] if len(keys)>1 else []) and all(w() is None for w in weak)
            assert set(c._segs)=={hot}
            rejected('finish_count_mismatch_'+str(len(keys)),lambda:helper.assert_finished(plan,dict(row,evicted_entries=row['evicted_entries']+1)))
            rejected('finish_category_mismatch_'+str(len(keys)),lambda:helper.assert_finished(plan,dict(row,evicted_categories={'invalid':1})))
            positive.append(dict(stale_count=plan['stale_count'],categories=plan['stale_categories'],peek_closure_sets_counters_unchanged=True,
                plan_pure_JSON=True,actual_frozen_finish_matches=True,no_cache_or_entry_retained=True))
        finally:binding.restore()
        assert graphing.GraphCache.run is original
    cases=['unpaired','empty_used','missing_used','untracked_capture','capturing','different_owner','different_cache',
        'disabled_cache','plan_used_changed','plan_counter_changed','plan_stream_changed','source_digest','source_function','category_function']
    for kind in cases:
        calls.clear();c=cache([hot]+stale);eng=engine(c);binding=adapter.install(mock,graphing)
        try:
            if kind!='unpaired':binding.begin_step(eng)
            if kind not in ('unpaired','empty_used'):c.run(hot,None,[Cell(9)])
            cv=inspect.getclosurevars(binding.finish_step).nonlocals
            call=lambda:helper.peek(binding,eng)
            if kind=='missing_used':cv['used'].add(('missing',))
            elif kind=='untracked_capture':c._segs[('untracked',)]=entry()
            elif kind=='capturing':capturing[0]=True
            elif kind=='different_owner':call=lambda:helper.peek(binding,engine(c))
            elif kind=='different_cache':eng._graphs=cache([hot])
            elif kind=='disabled_cache':c.enabled=False
            elif kind.startswith('plan_'):
                plan=helper.peek(binding,eng)
                if kind=='plan_used_changed':c.run(stale[0],None,[Cell(10)])
                elif kind=='plan_counter_changed':c.hits+=1
                else:c._capture_stream=object()
                call=lambda:helper.assert_same_plan(binding,eng,plan)
            elif kind=='source_digest':binding.source_sha256='0'*64
            elif kind=='source_function':
                def replacement(eng):pass
                replacement.__qualname__='install.<locals>.finish_step';binding.finish_step=replacement
            elif kind=='category_function':
                # Mutate the closure only in this negative fixture; original source is untouched.
                index=binding.finish_step.__code__.co_freevars.index('category')
                binding.finish_step.__closure__[index].cell_contents=lambda key:'fake'
            rejected(kind,call)
        finally:
            capturing[0]=False;binding.restore()
        assert graphing.GraphCache.run is original
    assert sha(HERE/'working_graph_prune.py')==helper.WORKING_PRUNE_SHA and sha(graphing_path)==GRAPHING_SHA
    return dict(status='passed',device='cpu',script_sha256=sha(__file__),helper_sha256=sha(helper.__file__),
        working_prune_sha256=helper.WORKING_PRUNE_SHA,graphing_sha256=GRAPHING_SHA,
        positive_cases=positive,negative_cases=negative,negative_count=len(negative),
        actual_frozen_adapter_used=True,fake_CUDA_only=True,CUDA_run=False,numerical_solver_run=False,
        hooks_restored=True,elapsed_s=time.perf_counter()-start,
        scope='Read-only plan/identity/protocol gate only. Fake pre-populated GraphCache replays and CUDA memory/synchronization. No real GPU, Tensor or state/SR numerical qualification.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graphing-source',type=Path,default=DEFAULT_GRAPHING)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    result=check(args.graphing_source);(args.output/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result))
