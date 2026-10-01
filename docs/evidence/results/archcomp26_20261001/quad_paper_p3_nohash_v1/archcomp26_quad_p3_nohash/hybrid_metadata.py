"""No-digest dense working P3 plus on-demand validation P4 metadata.

No polynomial, interval, schedule, validation, or CUDA arithmetic is changed.
Install in a fresh process before constructing the working engine or any graph.
"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util, sys, time

BACKEND_SHA='3140024445ebac6845d037c6399e2ad8658ae63281eb066c1e87356c99bd990b'
DEPENDENCIES={
    'graded_lex.py':'5181161bf78c16273b6b5725e1e6a449d7d3617c59b05a9db7656fe3eb551cd3',
    'legacy_sparse_metadata.py':'67ebe12629e12617cfe56bbaafae5337ba41a9f54802d7cfa86bc2abafd822f2',
}
POLICY='dense_working3_on_demand_validation4_original_arithmetic'

def install(identity,backend_path,extra_modules=()):
    import numpy as np
    import torch
    from flowstar_gpu import support as sp, sparse_exec as se, monomials
    path=Path(backend_path);assert path.is_file()
    for name in DEPENDENCIES:assert path.with_name(name).is_file()
    assert se.VALIDATION_POLICY=='solution_plus_one' and se.COMPOSITION_MODE=='horner'
    assert se.SUPPORT_POLICY=='structural'
    original={name:getattr(sp,name) for name in ['make_support','_exps_for','spatial_to_full_ids','full_to_spatial_ids']}
    dense_builder=monomials.build_tables
    spec=importlib.util.spec_from_file_location('fullbatch_p3_metadata_backend',path)
    backend=importlib.util.module_from_spec(spec);sys.modules[spec.name]=backend;spec.loader.exec_module(backend)
    backend.install(identity)
    # Keep dense working-family factories exactly as before. In particular,
    # their full degree-six product basis has100947 rows, above the metadata
    # active-support guard100000. Only validation-family4 uses sparse factories.
    def choose(k):
        if k not in (3,4):raise ValueError('hybrid supports only working3/validation4')
        return k==4
    def make_support(n,k,spatial,ids):
        return (backend.make_support if choose(k) else original['make_support'])(n,k,spatial,ids)
    def exponents(sup):
        return (backend.exponents if choose(sup.k) else original['_exps_for'])(sup)
    def spatial_to_full(n,k,sup):
        return (backend.spatial_to_full if choose(k) else original['spatial_to_full_ids'])(n,k,sup)
    def full_to_spatial(n,k,sup):
        return (backend.full_to_spatial if choose(k) else original['full_to_spatial_ids'])(n,k,sup)
    sp.make_support,sp._exps_for=make_support,exponents
    sp.spatial_to_full_ids,sp.full_to_spatial_ids=spatial_to_full,full_to_spatial
    receipts=[]
    def validation_engine(eng,order):
        if isinstance(eng,backend.MetadataEngine):
            assert eng.tables.k==4 and order==4
            return eng
        assert eng.tables.k==3
        if order<=3:return eng
        assert order==4 and eng.step.lanes==0
        cache=se._eng_cache(eng,'_validation_engines');key=(order,eng.step.delta)
        if key not in cache:
            started=time.perf_counter()
            value=backend.MetadataEngine(eng.tables.n,4,eng.step.delta,eng.device)
            dense=dense_builder(eng.tables.n,3)  # Existing CPU cache; never family4.
            rows=dense.exponents.numpy();sp_rows=rows[dense.t_deg.numpy()==0,1:]
            counts={}
            for spatial,expected in [(False,rows),(True,sp_rows)]:
                family=value.tables.family(spatial)
                assert family.prefixes[6]==len(expected)
                for begin in range(0,len(expected),4096):
                    ids=tuple(range(begin,min(begin+4096,len(expected))))
                    actual=value.tables.exponents(ids,spatial)
                    assert np.array_equal(actual,expected[begin:begin+len(ids)])
                counts['spatial' if spatial else 'full']=len(expected)
            assert value._cutoff_zero.device==eng._cutoff_zero.device
            cache[key]=value
            receipts.append(dict(n=eng.tables.n,working=3,validation=4,full_degree_six_prefix=counts,
                prefix_checked=True,setup_s=time.perf_counter()-started))
        return cache[key]
    se.validation_engine_for_order=validation_engine
    # Catch accidental fallbacks/import aliases before they allocate dense4.
    aliases=[]
    def guarded_builder(n,k,*a,**kw):
        if k>=4:raise RuntimeError('dense validation tables are forbidden in hybrid P3')
        return dense_builder(n,k,*a,**kw)
    modules=[m for name,m in list(sys.modules.items()) if name.startswith('flowstar_gpu')]+list(extra_modules)
    for module in modules:
        if module is None:continue
        for name,value in list(vars(module).items()):
            if value is dense_builder:
                setattr(module,name,guarded_builder);aliases.append(module.__name__+'.'+name)
    assert monomials.build_tables is guarded_builder and sp.build_tables is guarded_builder
    return SimpleNamespace(backend=backend,validation_engine=validation_engine,receipts=receipts,
        guarded_dense_aliases=sorted(set(aliases)),policy=POLICY,
        source_path=str(Path(__file__)),backend_path=str(path),
        dependency_paths=[str(path.with_name(name)) for name in DEPENDENCIES])
