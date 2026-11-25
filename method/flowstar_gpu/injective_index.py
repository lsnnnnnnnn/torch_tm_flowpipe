"""Static certified-injective index updates; no reductions or atomics.

Engine callers construct plans from host-built unique in-range maps via bind.
Plans and their read-only device indices live with the SparseEngine. No GPU
coefficient or index data controls Python execution or needs a host transfer.
"""
from dataclasses import dataclass
import os
import operator

import torch

from .cuda_kernels import load_cuda_extension

ENABLED = os.environ.get('FLOWSTAR_INJECTIVE_MAPS', '0') == '1'
_ext = None
_tried = False
_CPP = """
torch::Tensor injective_update_cuda(torch::Tensor dst, torch::Tensor src,
                                   torch::Tensor index, int64_t dim, bool add);
"""
_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__global__ void injective_update_kernel(double *dst, const double *src,
    const int64_t *index, int64_t count, int64_t width,
    int64_t target_width, int64_t inner, bool add) {
  const int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= count) return;
  const int64_t j = (q / inner) % width;
  const int64_t outer = q / (width * inner);
  const int64_t target = (outer * target_width + index[j]) * inner + q % inner;
  // Injectivity gives one writer per target. Native deterministic index_add_
  // seeds its reduction with +0 even for one term: preserve that seed before
  // adding to dst (notably, dst == src == -0 produces +0, not -0).
  dst[target] = add ? __dadd_rn(dst[target], __dadd_rn(0.0, src[q])) : src[q];
}

torch::Tensor injective_update_cuda(torch::Tensor dst, torch::Tensor src,
                                   torch::Tensor index, int64_t dim, bool add) {
  TORCH_CHECK(dst.is_cuda() && src.is_cuda() && index.is_cuda());
  TORCH_CHECK(dst.scalar_type() == at::kDouble && src.scalar_type() == at::kDouble);
  TORCH_CHECK(index.scalar_type() == at::kLong && index.dim() == 1);
  TORCH_CHECK(dst.is_contiguous() && src.is_contiguous() && index.is_contiguous());
  TORCH_CHECK(dst.device() == src.device() && dst.device() == index.device());
  TORCH_CHECK(dst.dim() == src.dim() && dim >= 0 && dim < dst.dim());
  TORCH_CHECK(src.size(dim) == index.numel());
  int64_t inner = 1;
  for (int64_t axis = 0; axis < src.dim(); ++axis) {
    if (axis != dim) TORCH_CHECK(dst.size(axis) == src.size(axis));
    if (axis > dim) inner *= src.size(axis);
  }
  const int64_t count = src.numel();
  if (!count) return dst;
  const c10::cuda::CUDAGuard guard(dst.device());
  injective_update_kernel<<<(count + 255) / 256, 256, 0,
      at::cuda::getCurrentCUDAStream(dst.get_device())>>>(dst.data_ptr<double>(),
      src.data_ptr<double>(), index.data_ptr<int64_t>(), count, src.size(dim),
      dst.size(dim), inner, add);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dst;
}
"""


def available():
    global _ext, _tried
    if not _tried:
        _tried = True
        _ext = load_cuda_extension('flowstar_injective_index_v2', _CPP, _CUDA,
                                   ['injective_update_cuda'])
    return _ext is not None


@dataclass(frozen=True)
class InjectiveMap:
    indices: torch.Tensor
    source_size: int
    target_size: int


def bind(host_indices, target_size, device):
    """Validate the immutable CPU map once, before graph capture/replay."""
    values = tuple(host_indices)
    if any(isinstance(v, bool) for v in values):
        raise ValueError('indices must be integers')
    try:
        values = tuple(operator.index(v) for v in values)
        target_size = operator.index(target_size)
    except TypeError as exc:
        raise ValueError('indices and target size must be integers') from exc
    if target_size < 0:
        raise ValueError('target size must be nonnegative')
    if len(set(values)) != len(values):
        raise ValueError('injective map contains repeated targets')
    if any(v < 0 or v >= target_size for v in values):
        raise ValueError('injective map target out of range')
    return InjectiveMap(torch.tensor(values, dtype=torch.int64, device=device), len(values), target_size)


def update_(dst, dim, src, plan, *, add=False):
    """Write through a certified static map; fallback preserves CPU behavior."""
    if not -dst.ndim <= dim < dst.ndim or src.ndim != dst.ndim:
        raise ValueError('invalid index axis or mismatched ranks')
    dim %= dst.ndim
    if dst.shape[dim] != plan.target_size or src.shape[dim] != plan.source_size:
        raise ValueError('tensor shape does not match the certified map')
    if src.numel() and dst.untyped_storage().data_ptr() == src.untyped_storage().data_ptr():
        raise ValueError('injective update requires disjoint input/output storage')
    if dst.is_cuda and dst.dtype == torch.float64 and dst.is_contiguous() and available():
        return _ext.injective_update_cuda(dst, src.contiguous(), plan.indices, dim, add)
    return dst.index_add_(dim, plan.indices, src) if add else dst.index_copy_(dim, plan.indices, src)
