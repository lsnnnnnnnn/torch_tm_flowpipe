"""Prototype: directed interval SR history matrix-vector sum in two stages.

Inputs Phi[Q,B,n,n,2] and J[Q,B,n,2] are contiguous CUDA float64, n<=32.
Stage one computes each interval dot over k into workspace[Q,B,n,2].
Stage two sums that workspace over Q. Workspace is Q*B*n*16 bytes rather
than Q*B*n*n*16 bytes; inputs are read only. There is no cubic broadcast.

Nonfinite/reversed input intervals produce NaNs, never hidden by fmin/fmax.
Finite-input overflow remains visibly nonfinite; callers MUST retain their
existing lane_nonfinite/rejection checks. Unsupported metadata must use the
caller's original path; this module does not change SR queues or scalars.
"""
from __future__ import annotations
import torch
from flowstar_gpu.cuda_kernels import load_cuda_extension

_CPP=r'''
#include <torch/extension.h>
void sr_j_sum(torch::Tensor phi, torch::Tensor j, torch::Tensor work, torch::Tensor out);
'''
_CUDA=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {
__global__ void sr_dot(const double* __restrict__ phi,
                       const double* __restrict__ j,
                       double* __restrict__ work,
                       long long count, int n) {
  const long long row=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=count)return;
  const long long qb=row/n;
  const int i=row%n;
  const double* a=phi+2*(qb*n*n+i*n);
  const double* b=j+2*qb*n;
  double lo=0.,hi=0.;bool bad=false;
  for(int k=0;k<n;++k){
    const double al=a[2*k], ah=a[2*k+1], bl=b[2*k], bh=b[2*k+1];
    if(!isfinite(al)||!isfinite(ah)||!isfinite(bl)||!isfinite(bh)||al>ah||bl>bh){bad=true;continue;}
    const double pl=fmin(fmin(__dmul_rd(al,bl),__dmul_rd(al,bh)),fmin(__dmul_rd(ah,bl),__dmul_rd(ah,bh)));
    const double ph=fmax(fmax(__dmul_ru(al,bl),__dmul_ru(al,bh)),fmax(__dmul_ru(ah,bl),__dmul_ru(ah,bh)));
    lo=__dadd_rd(lo,pl);hi=__dadd_ru(hi,ph);
  }
  work[2*row]=bad?NAN:lo;work[2*row+1]=bad?NAN:hi;
}
__global__ void sr_sum(const double* __restrict__ work,
                       double* __restrict__ out,
                       long long Q, long long count){
  const long long bi=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(bi>=count)return;
  double lo=0.,hi=0.;bool bad=false;
  for(long long q=0;q<Q;++q){
    const double al=work[2*(q*count+bi)], ah=work[2*(q*count+bi)+1];
    // Nonfinite source inputs were marked NaN in stage one. Infinities
    // produced by finite-input overflow remain observable to the lane gate.
    if(isnan(al)||isnan(ah)||al>ah){bad=true;continue;}
    lo=__dadd_rd(lo,al);hi=__dadd_ru(hi,ah);
  }
  out[2*bi]=bad?NAN:lo;out[2*bi+1]=bad?NAN:hi;
}
}
void sr_j_sum(torch::Tensor phi,torch::Tensor j,torch::Tensor work,torch::Tensor out){
  TORCH_CHECK(phi.is_cuda()&&j.is_cuda()&&work.is_cuda()&&out.is_cuda(),"CUDA required");
  TORCH_CHECK(phi.device()==j.device()&&phi.device()==work.device()&&phi.device()==out.device(),"same device required");
  TORCH_CHECK(phi.scalar_type()==torch::kFloat64&&j.scalar_type()==torch::kFloat64&&work.scalar_type()==torch::kFloat64&&out.scalar_type()==torch::kFloat64,"float64 required");
  TORCH_CHECK(phi.is_contiguous()&&j.is_contiguous()&&work.is_contiguous()&&out.is_contiguous(),"contiguous required");
  TORCH_CHECK(phi.dim()==5&&j.dim()==4&&work.dim()==4&&out.dim()==3,"bad rank");
  const long long Q=phi.size(0),B=phi.size(1),n=phi.size(2);
  TORCH_CHECK(n>=1&&n<=32&&B>=1,"unsupported size");
  TORCH_CHECK(phi.size(3)==n&&phi.size(4)==2&&j.size(0)==Q&&j.size(1)==B&&j.size(2)==n&&j.size(3)==2,"input shape mismatch");
  TORCH_CHECK(work.sizes()==j.sizes()&&out.size(0)==B&&out.size(1)==n&&out.size(2)==2,"output shape mismatch");
  const long long limit=2147483647LL*256;
  TORCH_CHECK(B<=limit/n&&Q<=limit/(B*n),"grid too large");
  const long long rows=Q*B*n,outputs=B*n;
  auto stream=at::cuda::getCurrentCUDAStream();
  if(Q>0){
    sr_dot<<<(unsigned int)((rows+255)/256),256,0,stream>>>(phi.data_ptr<double>(),j.data_ptr<double>(),work.data_ptr<double>(),rows,(int)n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  sr_sum<<<(unsigned int)((outputs+255)/256),256,0,stream>>>(work.data_ptr<double>(),out.data_ptr<double>(),Q,outputs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
_ext=None
_tried=False

def available()->bool:
 global _ext,_tried
 if not _tried:
  _tried=True
  _ext=load_cuda_extension('flowstar_sr_history_sum_prototype',_CPP,_CUDA,['sr_j_sum'])
 return _ext is not None

def supported(phi:torch.Tensor,j:torch.Tensor)->bool:
 if not (phi.is_cuda and j.is_cuda and phi.dtype==j.dtype==torch.float64 and phi.device==j.device and phi.is_contiguous() and j.is_contiguous() and phi.ndim==5 and j.ndim==4):return False
 q,b,n,m,e=phi.shape
 limit=2147483647*256
 return (b>=1 and 1<=n<=32 and m==n and e==2 and tuple(j.shape)==(q,b,n,2) and b*n<=limit and q<=limit//(b*n))

def sum_history(phi:torch.Tensor,j:torch.Tensor)->torch.Tensor:
 """Return [B,n,2], preserving both inputs; allocate one linear-Q workspace."""
 if not supported(phi,j):raise ValueError('unsupported SR J history shape, layout, dtype or device')
 q,b,n,_,_=phi.shape
 if q==0:return torch.zeros((b,n,2),dtype=phi.dtype,device=phi.device)
 if not available():raise RuntimeError('SR J history CUDA extension unavailable')
 with torch.cuda.device(phi.device):
  work=torch.empty_like(j)
  out=torch.empty((b,n,2),dtype=phi.dtype,device=phi.device)
  _ext.sr_j_sum(phi,j,work,out)
 return out
