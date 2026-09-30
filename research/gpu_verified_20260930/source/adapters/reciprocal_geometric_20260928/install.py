"""Process-local five-route candidate over pinned a3fb; no engine/cache writes.

Install before constructing any engine, graph or tape. Source emission is CPU
only. CUDA compilation is opt-in and uses two new names in a separate directory.
"""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import hashlib, importlib.util, json, linecache, os, sys

ELEMENTARY_SHA='3e9d7766761c7fe5188da53269346b009072553c7e8fb8480dfa29c87ca4a9b1'
TAPE_SHA='d39264a7c9db965acc2cb31761b05d1295d66c7fd1080404bad192db532683fb'
NAMES=('rec_series_valid_g','rec_series_valid','rec_series_replay')

# Exactly the same device source is inserted into both independent extensions.
# Bad is monotone. Explicit finite/order checks prevent fmin/fmax hiding NaNs.
GEOMETRIC_DEVICE=r'''
__device__ __forceinline__ bool geometric_valid(IV a) {
  return isfinite(a.lo) && isfinite(a.hi) && a.lo <= a.hi;
}
__device__ IV geometric_tail(IV c, IV rec_c, IV u, int order, bool* bad) {
  bool local = !geometric_valid(c) || !geometric_valid(rec_c) || !geometric_valid(u)
               || (c.lo <= 0.0 && c.hi >= 0.0) || order < 1;
  if (local) { *bad = true; return {0.0, 0.0}; }
  checked_rec(c, &local);  // recover overflow bad even for sanitized cached rec(C)
  IV denominator = iv_add({1.0, 1.0}, u);
  IV numerator = iv_pow_int(iv_neg(u), order);
  IV quotient = checked_div(numerator, denominator, &local);
  IV tail = iv_mul(rec_c, quotient);
  local = local || !geometric_valid(denominator) || !geometric_valid(numerator)
                || !geometric_valid(quotient) || !geometric_valid(tail);
  if (local) { *bad = true; return {0.0, 0.0}; }
  return tail;
}
'''

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def replace_once(source,old,new):
    assert source.count(old)==1,old
    return source.replace(old,new)
def load_core():
    path=Path(__file__).with_name('core.py');name='reciprocal_geometric_candidate_core'
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def prepare(cache_root,*,allow_build=False,binary_paths=None):
    from flowstar_gpu import elementary as elem,tape_kernels as old
    assert sha(elem.__file__)==ELEMENTARY_SHA and sha(old.__file__)==TAPE_SHA
    source=Path(old.__file__).read_text()
    replay_old='''__device__ IV rec_taylor_remainder(IV c, IV tm_range, int order, bool* bad) {
  IV dom = iv_add(c, tm_range);
  IV q = checked_div(tm_range, dom, bad);
  q = iv_pow_int(q, order);
  q = checked_div(q, dom, bad);
  if (order % 2 == 1) q = iv_neg(q);
  return q;
}'''
    source=replace_once(source,replay_old,GEOMETRIC_DEVICE)
    source=replace_once(source,
        '  IV lag = rec_taylor_remainder(c_f, iv_mul(tm_range, c_f), k + 1, bad);\n  return iv_add(result, iv_mul(lag, const_part));',
        '  IV tail = geometric_tail(c_f, const_part, tm_range, k + 1, bad);\n  return iv_add(result, tail);')
    valid_old='''__device__ IV lag_rec(IV c, IV tm, int order, bool* bad) {
  IV dom = iv_add(c, tm);
  IV q = checked_div(tm, dom, bad);
  q = iv_pow_int(q, order);
  q = checked_div(q, dom, bad);
  if (order % 2 == 1) q = iv_neg(q);
  return q;
}'''
    source=replace_once(source,valid_old,GEOMETRIC_DEVICE+'''
__device__ IV lag_rec(IV c, IV u, int order, bool* bad) {
  IV rec_c = checked_rec(c, bad);
  return geometric_tail(c, rec_c, u, order, bad);
}''')
    source=replace_once(source,
        '    asmb.emit(V["R_MUL"], (T6, TMR, C0))\n    asmb.emit(V["LAG_REC"], (LAG, C0, T6, k + 1))\n    asmb.emit(V["R_MUL"], (T6, LAG, CMAIN))\n    asmb.emit(V["R_ADD"], (RESR, RESR, T6))',
        '    asmb.emit(V["LAG_REC"], (LAG, C0, TMR, k + 1))\n    asmb.emit(V["R_ADD"], (RESR, RESR, LAG))')
    # Names depend on the full candidate source before name substitution.
    revision=digest(source)[:12]
    names=dict(replay='flowstar_recip_geom_replay_'+revision,valid='flowstar_recip_geom_valid_'+revision)
    source=replace_once(source,'"flowstar_tape_kernels", _CPP_SRC, _CUDA_SRC,',repr(names['replay'])+', _CPP_SRC, _CUDA_SRC,')
    source=replace_once(source,'"flowstar_valid_kernels", _VALID_CPP, _VALID_CUDA,',repr(names['valid'])+', _VALID_CPP, _VALID_CUDA,')
    name='flowstar_gpu.reciprocal_geometric_candidate_tape';module=ModuleType(name)
    module.__package__='flowstar_gpu';module.__file__=str(Path(__file__).with_name('generated_tape_candidate.py'))
    linecache.cache[module.__file__]=(len(source),None,source.splitlines(True),module.__file__)
    sys.modules[name]=module;exec(compile(source,module.__file__,'exec'),module.__dict__)
    cache_root=Path(cache_root).resolve();binary_paths=binary_paths or {};loads={}
    def loader(extension_name,cpp,cuda,functions):
        if extension_name in binary_paths:
            path=Path(binary_paths[extension_name]).resolve()
            spec=importlib.util.spec_from_file_location(extension_name,path);extension=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(extension)
        else:
            if not allow_build:raise RuntimeError('candidate CUDA build not authorized; preflight new extensions explicitly')
            import torch
            from torch.utils.cpp_extension import load_inline
            if not torch.cuda.is_available():raise RuntimeError('candidate build requires CUDA')
            directory=cache_root/extension_name;directory.mkdir(parents=True,exist_ok=True)
            capability=torch.cuda.get_device_capability();arch=f'{capability[0]}{capability[1]}'
            extension=load_inline(name=extension_name,cpp_sources=[cpp],cuda_sources=[cuda],functions=functions,
                build_directory=str(directory),extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{arch},code=sm_{arch}']+
                    (['-ccbin','/usr/bin/g++-13'] if os.path.exists('/usr/bin/g++-13') else []),verbose=False)
            path=Path(extension.__file__).resolve()
        loads[extension_name]=dict(path=str(path),sha256=sha(path))
        return extension
    module.load_cuda_extension=loader
    metadata=dict(installer_sha256=sha(__file__),core_sha256=sha(Path(__file__).with_name('core.py')),
        original_elementary_sha256=ELEMENTARY_SHA,original_tape_sha256=TAPE_SHA,generated_python_sha256=digest(source),
        replay_cuda_sha256=digest(module._CUDA_SRC),valid_cuda_sha256=digest(module._VALID_CUDA),
        geometric_device_sha256=digest(GEOMETRIC_DEVICE),extension_names=names,cache_root=str(cache_root),
        covers=['generic','dense','python_replay','cuda_replay','cuda_valid'],fallback_disabled_routes=[],
        install_precondition='Fresh process: install before creating engines, cached graphs, valid/replay tapes.')
    return SimpleNamespace(core=load_core(),tape=module,original_tape=old,source=source,metadata=metadata,loads=loads)


def install(cache_root,*,allow_build=False,binary_paths=None):
    import flowstar_gpu
    from flowstar_gpu import elementary as elem
    handle=prepare(cache_root,allow_build=allow_build,binary_paths=binary_paths)
    previous={name:getattr(elem,name) for name in NAMES}
    assert all(fn.__module__==elem.__name__ for fn in previous.values()),'install once in a fresh process'
    for name in NAMES:setattr(elem,name,getattr(handle.core,name))
    sys.modules['flowstar_gpu.tape_kernels']=handle.tape;flowstar_gpu.tape_kernels=handle.tape
    def restore():
        for name,fn in previous.items():setattr(elem,name,fn)
        sys.modules['flowstar_gpu.tape_kernels']=handle.original_tape;flowstar_gpu.tape_kernels=handle.original_tape
    handle.restore=restore;handle.original_functions=previous;return handle


def emit(handle,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for filename,source in [('tape_candidate.py',handle.source),('replay.cu',handle.tape._CUDA_SRC),('valid.cu',handle.tape._VALID_CUDA)]:
        (output/filename).write_text(source)
    (output/'MANIFEST.json').write_text(json.dumps(handle.metadata,indent=2)+'\n')
