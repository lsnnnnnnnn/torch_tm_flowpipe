"""One CUDA range operator via the existing NVRTC/driver, without new dependencies.

Only compiled code is cached. ResidentGroup is explicit caller-owned input, not
a hidden cross-call numerical cache. The CPU adapter includes all transfers.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
import hashlib
from pathlib import Path
import threading
import time

import torch


_MODULES = {}
_LOCK = threading.Lock()
FLAGS = ("--std=c++14", "--gpu-architecture=compute_70", "--fmad=false", "--ftz=false",
         "--prec-div=true", "--prec-sqrt=true")


def _check(code, operation):
    if code:
        raise RuntimeError(f"{operation} returned CUDA/NVRTC error {code}")


class KernelModule:
    def __init__(self, device):
        start = time.perf_counter()
        self.device = device
        torch.cuda.init()
        torch.empty(1,device=device)  # establish PyTorch's primary context
        libraries = sorted((Path(torch.__file__).resolve().parent.parent/"nvidia/cuda_nvrtc/lib").glob("libnvrtc.so*"))
        if not libraries:
            raise RuntimeError("installed PyTorch NVRTC library not found")
        self.nvrtc = C.CDLL(str(libraries[0]))
        self.driver = C.CDLL("libcuda.so.1")
        self.nvrtc.nvrtcCreateProgram.argtypes = [C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p]
        self.nvrtc.nvrtcCompileProgram.argtypes = [C.c_void_p,C.c_int,C.POINTER(C.c_char_p)]
        for name in ("nvrtcGetProgramLogSize","nvrtcGetPTXSize"):
            getattr(self.nvrtc,name).argtypes=[C.c_void_p,C.POINTER(C.c_size_t)]
        for name in ("nvrtcGetProgramLog","nvrtcGetPTX"):
            getattr(self.nvrtc,name).argtypes=[C.c_void_p,C.c_void_p]
        self.nvrtc.nvrtcDestroyProgram.argtypes=[C.POINTER(C.c_void_p)]
        self.driver.cuModuleLoadData.argtypes=[C.POINTER(C.c_void_p),C.c_void_p]
        self.driver.cuModuleGetFunction.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p]
        self.driver.cuLaunchKernel.argtypes=[C.c_void_p,*([C.c_uint]*7),C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p]
        self.source=Path(__file__).with_name("range_cuda_kernel.cu").read_bytes()
        program=C.c_void_p()
        _check(self.nvrtc.nvrtcCreateProgram(C.byref(program),self.source,b"range_cuda_kernel.cu",0,None,None),"nvrtcCreateProgram")
        try:
            options=(C.c_char_p*len(FLAGS))(*(f.encode() for f in FLAGS))
            code=self.nvrtc.nvrtcCompileProgram(program,len(FLAGS),options)
            size=C.c_size_t()
            _check(self.nvrtc.nvrtcGetProgramLogSize(program,C.byref(size)),"nvrtcGetProgramLogSize")
            log=C.create_string_buffer(size.value)
            _check(self.nvrtc.nvrtcGetProgramLog(program,log),"nvrtcGetProgramLog")
            self.log=log.value.decode()
            if code: raise RuntimeError(f"NVRTC compilation failed ({code}): {self.log}")
            _check(self.nvrtc.nvrtcGetPTXSize(program,C.byref(size)),"nvrtcGetPTXSize")
            ptx=C.create_string_buffer(size.value)
            _check(self.nvrtc.nvrtcGetPTX(program,ptx),"nvrtcGetPTX")
        finally:
            self.nvrtc.nvrtcDestroyProgram(C.byref(program))
        self.module=C.c_void_p()
        _check(self.driver.cuModuleLoadData(C.byref(self.module),ptx),"cuModuleLoadData")
        self.functions={}
        for name in ("validate_rows","range_powers","range_terms","range_sum","arithmetic_probe"):
            function=C.c_void_p()
            _check(self.driver.cuModuleGetFunction(C.byref(function),self.module,name.encode()),"cuModuleGetFunction")
            self.functions[name]=function
        major,minor=C.c_int(),C.c_int()
        _check(self.nvrtc.nvrtcVersion(C.byref(major),C.byref(minor)),"nvrtcVersion")
        self.build=dict(nvrtc_path=str(libraries[0]),nvrtc_version=[major.value,minor.value],flags=list(FLAGS),
            source_sha256=hashlib.sha256(self.source).hexdigest(),ptx_sha256=hashlib.sha256(ptx.raw).hexdigest(),
            compile_and_load_s=time.perf_counter()-start,compiler_log=self.log,device=str(device),
            device_name=torch.cuda.get_device_name(device),capability=list(torch.cuda.get_device_capability(device)))

    def launch(self,name,count,args):
        values=[C.c_void_p(x.data_ptr()) if isinstance(x,torch.Tensor) else C.c_int(x) for x in args]
        pointers=(C.c_void_p*len(values))(*(C.cast(C.byref(x),C.c_void_p) for x in values))
        # Always launch at least one block: even empty supports have a receipt.
        _check(self.driver.cuLaunchKernel(self.functions[name],max(1,(count+127)//128),1,1,128,1,1,0,
            C.c_void_p(torch.cuda.current_stream(self.device).cuda_stream),pointers,None),name)


def get_module(device="cuda:0"):
    device=torch.device(device)
    if device.index is None: device=torch.device("cuda",torch.cuda.current_device())
    with torch.cuda.device(device),_LOCK:
        if device not in _MODULES: _MODULES[device]=KernelModule(device)
        return _MODULES[device]


@dataclass(frozen=True)
class ResidentGroup:
    coefficients_lo: torch.Tensor
    coefficients_hi: torch.Tensor
    domain_lo: torch.Tensor
    domain_hi: torch.Tensor
    keys: torch.Tensor
    ops: torch.Tensor
    states: torch.Tensor
    mask: torch.Tensor
    power_keys: tuple
    point_coefficients: bool


def prepare_cuda_group(group, *, device="cuda:0"):
    get_module(device)
    keys=tuple(sorted({(v,p) for kind,v,_,values in group.plan.stages if kind=="power" for p in values}))
    mapping={k:i for i,k in enumerate(keys)}
    operations=[]
    for kind,v,indices,values in group.plan.stages:
        row=[-1]*len(group.plan.exponents)
        for i,value in zip(indices,values):
            row[i]=mapping[v,value] if kind=="power" else {-1.:-2,0.:-3,1.:-4}[value]
        operations.append(row)
    r=group.requests[0]
    # Per-call metadata copies are part of offload cost; caller can retain this
    # explicit object only in the separately labelled resident experiment.
    return ResidentGroup(*(x.to(device=device,copy=True) for x in
        (group.coefficients_lo,group.coefficients_hi,group.domain_lo,group.domain_hi)),
        torch.tensor(keys,dtype=torch.int32,device=device).reshape(-1,2),
        torch.tensor(operations,dtype=torch.int32,device=device).reshape(len(operations),len(r.exponents)),
        torch.tensor([v in r.state_variables for v in range(group.plan.n_vars)],dtype=torch.int32,device=device),
        torch.ones(len(group.requests),dtype=torch.uint8,device=device),keys,r.kind!="interval-coefficient")


def run_resident(group: ResidentGroup):
    """Submit four real kernels; caller synchronizes before using timings/results."""
    device=group.coefficients_lo.device
    with torch.cuda.device(device):
        module=get_module(device)
        B,O,T=group.coefficients_lo.shape
        V=group.domain_lo.shape[1]; P=len(group.power_keys); S=group.ops.shape[0]
        kwargs=dict(dtype=torch.float64,device=device)
        lo=torch.empty((B,O),**kwargs); hi=torch.empty_like(lo)
        tl=torch.empty((B,O,T),**kwargs); th=torch.empty_like(tl)
        pl=torch.empty((B,P),**kwargs); ph=torch.empty_like(pl)
        status=torch.empty(B,dtype=torch.int32,device=device)
        receipt=torch.zeros(4,dtype=torch.int64,device=device)
        module.launch("validate_rows",B,[group.coefficients_lo,group.coefficients_hi,group.domain_lo,group.domain_hi,
            group.states,group.mask,status,receipt,B,O,T,V,int(group.point_coefficients)])
        module.launch("range_powers",B*P,[group.domain_lo,group.domain_hi,group.keys,pl,ph,status,receipt,B,P,V])
        module.launch("range_terms",B*O*T,[group.coefficients_lo,group.coefficients_hi,pl,ph,group.ops,tl,th,status,receipt,B,O,T,P,S])
        module.launch("range_sum",B*O,[tl,th,lo,hi,status,receipt,B,O,T])
        return lo,hi,tl,th,status,pl,ph,receipt


def compute_cuda_group(group, *, diagnostics=False, timings=None):
    start=time.perf_counter()
    resident=prepare_cuda_group(group)
    torch.cuda.synchronize(resident.coefficients_lo.device)
    transferred=time.perf_counter()
    output=run_resident(resident)
    torch.cuda.synchronize(resident.coefficients_lo.device)
    computed=time.perf_counter()
    # Copy the status and execution receipt on every call, not just in audits.
    lo,hi,tl,th,status,pl,ph,receipt=output
    lo,hi,status,receipt=(x.cpu() for x in (lo,hi,status,receipt))
    if diagnostics: tl,th,pl,ph=(x.cpu() for x in (tl,th,pl,ph))
    else: tl=th=None
    assert receipt.tolist()==[1,1,1,1], "missing actual device kernel execution"
    active=status==0
    active &= (torch.isfinite(lo)&torch.isfinite(hi)&(lo<=hi)).all(dim=1)
    records=[[(v,p,float(pl[b,i]),float(ph[b,i]),False) for i,(v,p) in enumerate(resident.power_keys)]
             if diagnostics and active[b] else [] for b in range(len(status))]
    if timings is not None:
        for key,value in dict(h2d_and_structure_s=transferred-start,kernel_and_sync_s=computed-transferred,
                              d2h_and_checks_s=time.perf_counter()-computed,actual_kernel_invocations=4).items():
            timings[key]=timings.get(key,0)+value
    return lo,hi,tl,th,active,[False]*len(status),records


def primitive_probe(a,b,device="cuda:0"):
    module=get_module(device)
    a=torch.tensor(a,dtype=torch.float64,device=device)
    b=torch.tensor(b,dtype=torch.float64,device=device)
    output=torch.empty((a.numel(),4),dtype=torch.float64,device=device)
    module.launch("arithmetic_probe",a.numel(),[a,b,output,a.numel()])
    return output.cpu()
