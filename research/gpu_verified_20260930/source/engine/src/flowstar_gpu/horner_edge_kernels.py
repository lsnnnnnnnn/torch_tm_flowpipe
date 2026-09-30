"""Opt-in directed Horner edge kernel; loaded before graph capture.

Inputs: previous, child, g_rem, g_range, p_range, tail, dropped,
coefficient_error, addition_error_range. All inputs remain read-only.
Normal initialized torch allocation is intentional for this first experiment.
"""
import hashlib
import os
from pathlib import Path

import torch

CPP = r"""
#include <torch/extension.h>
torch::Tensor horner_edge(std::vector<torch::Tensor> x, int64_t variable);
"""

CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
struct Interval { double lo; double hi; };
__device__ __forceinline__ Interval read_iv(const double* x, int64_t i) {
  return {x[2*i], x[2*i+1]};
}
__device__ __forceinline__ Interval mul_iv(Interval a, Interval b) {
  return {
    fmin(fmin(__dmul_rd(a.lo, b.lo), __dmul_rd(a.lo, b.hi)),
         fmin(__dmul_rd(a.hi, b.lo), __dmul_rd(a.hi, b.hi))),
    fmax(fmax(__dmul_ru(a.lo, b.lo), __dmul_ru(a.lo, b.hi)),
         fmax(__dmul_ru(a.hi, b.lo), __dmul_ru(a.hi, b.hi)))
  };
}
__device__ __forceinline__ Interval add_iv(Interval a, Interval b) {
  return {__dadd_rd(a.lo, b.lo), __dadd_ru(a.hi, b.hi)};
}
__global__ void edge_kernel(
    const double* previous, const double* child,
    const double* g_rem, const double* g_range,
    const double* p_range, const double* tail, const double* dropped,
    const double* coefficient_error, const double* addition_error,
    double* out, int64_t count, int64_t rows, int64_t variables, int64_t v) {
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    const int64_t gi = (i / rows) * variables + v;
    const Interval pr = read_iv(child, i);
    const Interval gr = read_iv(g_rem, gi);
    Interval r = mul_iv(pr, gr);
    r = add_iv(r, mul_iv(read_iv(g_range, gi), pr));
    r = add_iv(r, mul_iv(read_iv(p_range, i), gr));
    r = add_iv(r, add_iv(read_iv(tail, i), read_iv(dropped, i)));
    r = add_iv(r, read_iv(coefficient_error, i));
    r = add_iv(r, read_iv(addition_error, i));
    r = add_iv(read_iv(previous, i), r);
    out[2*i] = r.lo;
    out[2*i+1] = r.hi;
  }
}
}

torch::Tensor horner_edge(std::vector<torch::Tensor> x, int64_t variable) {
  TORCH_CHECK(x.size() == 9, "nine interval inputs required");
  const auto& ref = x[0];
  for (int j = 0; j < 9; ++j) {
    TORCH_CHECK(x[j].is_cuda() && x[j].scalar_type() == at::kDouble,
                "CUDA float64 required");
    TORCH_CHECK(x[j].device() == ref.device() && x[j].is_contiguous(),
                "same device and contiguous layout required");
    TORCH_CHECK(x[j].dim() == 3 && x[j].size(2) == 2,
                "expected [batch, rows, 2]");
  }
  const int64_t batch = ref.size(0), rows = ref.size(1);
  const int64_t variables = x[2].size(1);
  TORCH_CHECK(variable >= 0 && variable < variables, "invalid variable");
  for (int j = 1; j < 9; ++j) {
    if (j == 2 || j == 3) {
      TORCH_CHECK(x[j].size(0) == batch && x[j].size(1) == variables,
                  "g tensors must agree in batch and variable count");
    } else {
      TORCH_CHECK(x[j].sizes() == ref.sizes(), "row tensor shape mismatch");
    }
  }
  c10::cuda::CUDAGuard guard(ref.device());
  auto out = torch::empty_like(ref);
  const int64_t count = ref.numel() / 2;
  if (count == 0) return out;
  const int blocks = int(std::min<int64_t>((count + 127) / 128, 65535));
  edge_kernel<<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      x[0].data_ptr<double>(), x[1].data_ptr<double>(),
      x[2].data_ptr<double>(), x[3].data_ptr<double>(),
      x[4].data_ptr<double>(), x[5].data_ptr<double>(),
      x[6].data_ptr<double>(), x[7].data_ptr<double>(),
      x[8].data_ptr<double>(), out.data_ptr<double>(), count, rows, variables, variable);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""


NAME = "horner_edge_1312fa8b2aed"
_ext = None


def load():
    """Preload in startup, fail visibly; keep the original seven caches separate."""
    global _ext
    if _ext is None:
        if torch.__version__.split("+")[0] != "2.5.1":
            raise RuntimeError("Horner edge requires qualified PyTorch 2.5.1")
        if not torch.cuda.is_available():
            raise RuntimeError("Horner edge requires CUDA")
        if hashlib.sha256((CPP + CUDA).encode()).hexdigest()[:12] != NAME.removeprefix("horner_edge_"):
            raise RuntimeError("Horner edge source identity changed")
        from torch.utils.cpp_extension import load_inline, get_default_build_root
        root = Path(os.environ.get("FLOWSTAR_HORNER_EDGE_CACHE",
                                   str(Path(get_default_build_root()) / "flowstar_horner_edge")))
        directory = root / NAME
        directory.mkdir(parents=True, exist_ok=True)
        _ext = load_inline(name=NAME, cpp_sources=CPP, cuda_sources=CUDA,
                           functions=["horner_edge"], build_directory=str(directory),
                           extra_cuda_cflags=["-O3", "--fmad=false", "--ftz=false"], verbose=False)
    return _ext
