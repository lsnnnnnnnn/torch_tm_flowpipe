"""Observer-only endpoint scratch: unchanged dense3 range, 16 lane blocks.

Pure call helper, no driver/global patches. Every lane keeps all 1140 basis
slots and the original directed accumulation order. NN/safety are out of scope.
Call verify_sources once before using this helper in a fresh process.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,inspect

POLICY='observer_endpoint_only_original_dense3_range_full1024_lane64'
SOURCE_SHA256={
    'safety.py':'834285569d5ca84115392c891ecc53b7eed867c8e3af25fe4fc10d50263a22bf',
    'interval.py':'815a5e60d561fe508c64f2b34e6a5eec0a995a6165d4738437b18c770286dc5b',
    'cuda_kernels.py':'cdfbc847e3255dc3675af4f70e1a08d8647ebf3a165e3413742763916446cf9c',
    'support.py':'414c274973c585894e545d7d9e70f6a6f6365d7085ae18bd51d6ffbc4bfa4c50',
}

def verify_sources(raw):
    from flowstar_gpu import safety,interval,support,cuda_kernels
    assert raw is safety.rows_range_over_time_sparse
    for module in (safety,interval,support,cuda_kernels):
        path=Path(module.__file__)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==SOURCE_SHA256[path.name]
    assert inspect.getfile(raw)==safety.__file__
    return dict(SOURCE_SHA256)

def endpoint64(st,eng,piece,idx_hi=12,*,raw):
    import torch
    from flowstar_gpu import interval as iv
    assert idx_hi==12 and st.pre.ndim==3 and st.pre.shape[0]==1024 and st.pre.shape[1]>=12
    assert st.pre_rem.shape==(1024,st.pre.shape[1],2) and piece.shape==(1024,2)
    assert st.pre.dtype==st.pre_rem.dtype==piece.dtype==torch.float64
    assert st.pre.device==st.pre_rem.device==piece.device and st.pre.is_cuda
    assert (eng.tables.n,eng.tables.k,eng.tables.T)==(16,3,1140)
    assert (st.pre_sup.n,st.pre_sup.k,st.pre_sup.spatial)==(16,3,False)
    assert iv._kern(st.pre,piece),'Qualification uses the original directed CUDA kernels'
    versions=[x._version for x in (st.pre,st.pre_rem,piece)]
    endpoint=piece.new_empty((1024,12,2))
    for start in range(0,1024,64):
        stop=start+64
        view=SimpleNamespace(pre=st.pre[start:stop],pre_rem=st.pre_rem[start:stop],pre_sup=st.pre_sup)
        part=raw(view,eng,piece[start:stop],idx_hi)
        assert part.shape==(64,12,2) and part.dtype==endpoint.dtype and part.device==endpoint.device
        endpoint[start:stop].copy_(part)
        del part,view
    assert [x._version for x in (st.pre,st.pre_rem,piece)]==versions
    return endpoint
