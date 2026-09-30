"""Strict interval left-multiplication of SR history, without Q*B*n**3 tensors.

Dispatch/fallback and queue allocation remain with the SR caller.
A [B,n,n,2] and history [Q,B,n,n,2] must be disjoint contiguous CUDA float64
memory ranges, with 1 <= n <= 32. Each block snapshots one old history matrix in
shared memory before any write. Four directed endpoint products and ordered
directed sums enclose every exact interval dot; no Rump slack is assumed.

Nonfinite or reversed input intervals produce NaN output in affected dots,
so this path never hides invalid input via CUDA fmin/fmax's NaN behavior.
The caller MUST retain its existing lane_nonfinite checks; this is not a
replacement for branch rejection or an end-to-end arithmetic certificate.
"""
from __future__ import annotations

import torch
from .cuda_kernels import load_cuda_extension

_CPP = r'''
#include <torch/extension.h>
void sr_left_multiply_(torch::Tensor left, torch::Tensor history);
'''

_CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {
__global__ void sr_left_kernel(const double* __restrict__ left,
                               double* __restrict__ history,
                               long long B, int n) {
  const long long qb = (long long)blockIdx.x;
  const long long b = qb % B;
  const int size = n * n;
  const double* a = left + b * size * 2;
  double* dst = history + qb * size * 2;
  extern __shared__ double old[];
  for (int p = threadIdx.x; p < size * 2; p += blockDim.x)
    old[p] = dst[p];
  __syncthreads();

  const int ij = threadIdx.x;
  double lo = 0.0, hi = 0.0;
  bool bad = false;
  if (ij < size) {
    const int i = ij / n, j = ij % n;
    for (int k = 0; k < n; ++k) {
      const double al = a[2 * (i * n + k)];
      const double ah = a[2 * (i * n + k) + 1];
      const double bl = old[2 * (k * n + j)];
      const double bh = old[2 * (k * n + j) + 1];
      if (!isfinite(al) || !isfinite(ah) || !isfinite(bl) || !isfinite(bh)
          || al > ah || bl > bh) {
        bad = true;
        continue;
      }
      const double pl = fmin(fmin(__dmul_rd(al, bl), __dmul_rd(al, bh)),
                             fmin(__dmul_rd(ah, bl), __dmul_rd(ah, bh)));
      const double ph = fmax(fmax(__dmul_ru(al, bl), __dmul_ru(al, bh)),
                             fmax(__dmul_ru(ah, bl), __dmul_ru(ah, bh)));
      lo = __dadd_rd(lo, pl);
      hi = __dadd_ru(hi, ph);
    }
  }
  // Every old matrix was snapshotted; every dot finishes before global writes.
  __syncthreads();
  if (ij < size) {
    dst[2 * ij] = bad ? NAN : lo;
    dst[2 * ij + 1] = bad ? NAN : hi;
  }
}
} // namespace

void sr_left_multiply_(torch::Tensor left, torch::Tensor history) {
  TORCH_CHECK(left.is_cuda() && history.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(left.scalar_type() == torch::kFloat64 &&
              history.scalar_type() == torch::kFloat64, "float64 required");
  TORCH_CHECK(left.is_contiguous() && history.is_contiguous(), "contiguous required");
  TORCH_CHECK(left.device() == history.device(), "devices must agree");
  TORCH_CHECK(left.dim() == 4 && history.dim() == 5, "bad rank");
  const long long Q = history.size(0), B = history.size(1);
  const int n = history.size(2);
  TORCH_CHECK(n >= 1 && n <= 32 && B >= 1, "unsupported n or B");
  TORCH_CHECK(history.size(3) == n && history.size(4) == 2 &&
              left.size(0) == B && left.size(1) == n &&
              left.size(2) == n && left.size(3) == 2, "shape mismatch");
  TORCH_CHECK(Q <= 2147483647LL / B, "grid too large");
  if (Q == 0) return;
  const int threads = ((n * n + 31) / 32) * 32;
  const size_t shared = 2 * n * n * sizeof(double); // <= 16 KiB at n=32
  sr_left_kernel<<<(unsigned int)(Q * B), threads, shared,
                    at::cuda::getCurrentCUDAStream()>>>(
      left.data_ptr<double>(), history.data_ptr<double>(), B, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''

_ext = None
_tried = False


def available() -> bool:
    """Build only this small extension; failure leaves the caller's fallback."""
    global _ext, _tried
    if not _tried:
        _tried = True
        _ext = load_cuda_extension('flowstar_sr_interval_matmul', _CPP, _CUDA,
                                  ['sr_left_multiply_'])
    return _ext is not None


def supported(left: torch.Tensor, history: torch.Tensor) -> bool:
    """Metadata gate only; no allocation, host/device value copy, or JIT build."""
    if not (left.is_cuda and history.is_cuda and
            left.dtype == history.dtype == torch.float64 and
            left.device == history.device and
            left.is_contiguous() and history.is_contiguous() and
            left.ndim == 4 and history.ndim == 5):
        return False
    q, b, n, m, endpoints = history.shape
    left_start, history_start = left.data_ptr(), history.data_ptr()
    disjoint = (left_start + left.numel() * left.element_size() <= history_start or
                history_start + history.numel() * history.element_size() <= left_start)
    return (1 <= n <= 32 and b >= 1 and n == m and endpoints == 2 and
            tuple(left.shape) == (b, n, n, 2) and q <= 2147483647 // b and
            disjoint)


def left_multiply_(left: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    """Overwrite history with left @ history; preserve its data pointer.

    Unsupported metadata is rejected before a launch. The calling SR function
    chooses its original path when supported()/available() is false.
    """
    if not supported(left, history):
        raise ValueError('unsupported SR interval matrix layout, size, dtype, device, or alias')
    if history.shape[0] == 0:
        return history
    if not available():
        raise RuntimeError('SR interval CUDA extension unavailable')
    with torch.cuda.device(left.device):
        _ext.sr_left_multiply_(left, history)
    return history
