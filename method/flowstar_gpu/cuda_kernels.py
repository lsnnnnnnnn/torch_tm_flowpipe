"""Fused CUDA segment-product kernels with TRUE directed rounding (M10 perf).

The pair-product + per-slot segment-sum pattern is the hot core of every dense
or sparse polynomial multiply in this project (polynomial.mul_iv / mul_point,
range evaluation dots).  The eager-torch implementation costs ~10 elementwise
kernel passes per interval multiply (gathers, 4 products, amin/amax, nextafter
steps, three segment reductions, Rump bound); each pass re-reads the whole
pair tensor from HBM.  The kernels here fuse the entire pattern into ONE pass
that keeps per-pair intermediates in registers, and replace the emulated
rounding (RN + nextafter outward + a-priori Rump reduction bound) with the
hardware's ACTUAL directed-rounding intrinsics:

    __dmul_rd / __dmul_ru   round-toward--inf / +inf multiply (IEEE-correct)
    __dadd_rd / __dadd_ru   directed adds for the in-segment accumulation

Soundness model: every lower-endpoint operation rounds toward -inf and every
upper-endpoint one toward +inf, so each partial accumulator is a true directed
bound of the exact partial sum — the classical MPFR/Flow* argument, with NO
error-model slack at all.  Results are therefore both sound and TIGHTER than
the torch path (which pays <=1 ulp per elementwise op plus the Rump (2m-1)u
reduction inflation).  fp64 denormals are never flushed on NVIDIA hardware
(there is no fp64 FTZ mode), so the argument holds through underflow; positive
overflow under _rd yields DBL_MAX and under _ru +inf — both sound directed
bounds (callers preclude inf INPUTS, same contract as interval.mul).

Determinism: each output slot is accumulated SEQUENTIALLY in pair order by a
single thread (or, for the single-segment dot, by a fixed 64-chunk two-pass
schedule that depends only on the segment length) — bitwise reproducible by
construction, independent of launch configuration, and each batch row's result
is computed independently of R (the M3 "lane bitwise-equal to its B=1 run"
gate holds by construction).

Layout contract (all tensors contiguous, f64, indices int64, same device):

    seg_mul_iv (a [R, Sa, 2], b [R, Sb, 2], pair_i [P], pair_j [P],
                seg_offsets [M+1]) -> out [R, M, 2]
        out[r, m] = sum_{p in seg m} a[r, pair_i[p]] * b[r, pair_j[p]]
        (interval product per pair: min/max of the four directed endpoint
        candidates; directed in-order accumulation; empty segment -> [+0, +0])

    seg_mul_pt (a [R, Sa], b [R, Sb], pair_i, pair_j, seg_offsets)
        -> out [R, M]
        Plain RN fma accumulation in pair order — tier-P semantics (Flow*
        leaves this roundoff unbounded too; the validated pass absorbs it).
        FMA is licensed exactly because tier P has no per-op error contract.

    seg_dot_pt_iv (p [R, S], w [S, 2] or [R, S, 2], seg_offsets [M+1],
                   idx [Q]) -> out [R, M, 2]
        out[r, m] = sum_{q in seg m} p[r, idx[q]] * w[(r,) idx[q]]
        (two directed candidates per term since p is a point value).
        M == 1 (the intEvalNormal range-dot case) uses the fixed 64-chunk
        two-pass schedule so a single segment still parallelizes; M >= 2 runs
        one thread per (r, m).  The path is keyed ONLY on M, so results never
        depend on R.

`available()` reports whether the JIT extension compiled; every public kernel
raises RuntimeError when it did not (callers keep the eager-torch fallback).
Set FLOWSTAR_NO_CUDA_KERNEL=1 to force the fallback (CI on CPU boxes).

Reference implementations `*_ref` mirror the EXISTING torch emulation
semantics (RN + nextafter + Rump) — they are the fallback path and the
bench/test baseline; the kernels must produce contained-or-equal, tighter
intervals than them on identical inputs.
"""

from __future__ import annotations

import math
import os

import torch

# ---------------------------------------------------------------------------
# CUDA source
# ---------------------------------------------------------------------------

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

// Interval product of [al, ah] x [bl, bh] accumulated into (acc_lo, acc_hi):
// four candidates per endpoint, each computed with the matching directed
// multiply; fmin/fmax select the extremes (inputs are finite by contract, so
// no NaN can appear); directed adds keep the accumulators sound bounds.
__device__ __forceinline__ void iv_prod_acc(
    double al, double ah, double bl, double bh,
    double& acc_lo, double& acc_hi) {
  double lo = fmin(fmin(__dmul_rd(al, bl), __dmul_rd(al, bh)),
                   fmin(__dmul_rd(ah, bl), __dmul_rd(ah, bh)));
  double hi = fmax(fmax(__dmul_ru(al, bl), __dmul_ru(al, bh)),
                   fmax(__dmul_ru(ah, bl), __dmul_ru(ah, bh)));
  acc_lo = __dadd_rd(acc_lo, lo);
  acc_hi = __dadd_ru(acc_hi, hi);
}

// One thread per (r, m): sequential in-order walk of segment m's pair list.
__global__ void seg_mul_iv_kernel(
    const double* __restrict__ a,      // [R, Sa, 2]
    const double* __restrict__ b,      // [R, Sb, 2]
    const int64_t* __restrict__ pi,    // [P]
    const int64_t* __restrict__ pj,    // [P]
    const int64_t* __restrict__ off,   // [M+1]
    double* __restrict__ out,          // [R, M, 2]
    long long R, long long M, long long Sa, long long Sb) {
  long long total = R * M;
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       t < total; t += stride) {
    long long r = t / M, m = t % M;
    const double* ar = a + r * Sa * 2;
    const double* br = b + r * Sb * 2;
    double lo = 0.0, hi = 0.0;
    for (int64_t p = off[m]; p < off[m + 1]; ++p) {
      const double* av = ar + 2 * pi[p];
      const double* bv = br + 2 * pj[p];
      iv_prod_acc(av[0], av[1], bv[0], bv[1], lo, hi);
    }
    out[2 * t] = lo;
    out[2 * t + 1] = hi;
  }
}

// One thread per (r, m): RN fma accumulation in pair order (tier P).
__global__ void seg_mul_pt_kernel(
    const double* __restrict__ a,      // [R, Sa]
    const double* __restrict__ b,      // [R, Sb]
    const int64_t* __restrict__ pi,    // [P]
    const int64_t* __restrict__ pj,    // [P]
    const int64_t* __restrict__ off,   // [M+1]
    double* __restrict__ out,          // [R, M]
    long long R, long long M, long long Sa, long long Sb) {
  long long total = R * M;
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       t < total; t += stride) {
    long long r = t / M, m = t % M;
    const double* ar = a + r * Sa;
    const double* br = b + r * Sb;
    double acc = 0.0;
    for (int64_t p = off[m]; p < off[m + 1]; ++p) {
      acc = fma(ar[pi[p]], br[pj[p]], acc);
    }
    out[t] = acc;
  }
}

// Point x interval term accumulated into (acc_lo, acc_hi): two candidates.
__device__ __forceinline__ void pt_iv_acc(
    double pv, double wl, double wh, double& acc_lo, double& acc_hi) {
  acc_lo = __dadd_rd(acc_lo, fmin(__dmul_rd(pv, wl), __dmul_rd(pv, wh)));
  acc_hi = __dadd_ru(acc_hi, fmax(__dmul_ru(pv, wl), __dmul_ru(pv, wh)));
}

// M >= 2 path: one thread per (r, m), sequential over the segment.
__global__ void seg_dot_kernel(
    const double* __restrict__ p,      // [R, S]
    const double* __restrict__ w,      // [S, 2] or [R, S, 2]
    const int64_t* __restrict__ idx,   // [Q]
    const int64_t* __restrict__ off,   // [M+1]
    double* __restrict__ out,          // [R, M, 2]
    long long R, long long M, long long S, bool w_batched) {
  long long total = R * M;
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       t < total; t += stride) {
    long long r = t / M, m = t % M;
    const double* pr = p + r * S;
    const double* wr = w_batched ? (w + r * S * 2) : w;
    double lo = 0.0, hi = 0.0;
    for (int64_t q = off[m]; q < off[m + 1]; ++q) {
      int64_t id = idx[q];
      pt_iv_acc(pr[id], wr[2 * id], wr[2 * id + 1], lo, hi);
    }
    out[2 * t] = lo;
    out[2 * t + 1] = hi;
  }
}

// M == 1 two-pass path, pass 1: fixed 64-chunk partials.  Chunk c covers
// idx positions [c * clen, min((c+1) * clen, Q)) with clen = ceil(Q / 64) —
// a pure function of Q, so the schedule (and hence the bitwise result) never
// depends on R or the launch geometry.
__global__ void seg_dot1_partial_kernel(
    const double* __restrict__ p,      // [R, S]
    const double* __restrict__ w,      // [S, 2] or [R, S, 2]
    const int64_t* __restrict__ idx,   // [Q]
    double* __restrict__ work,         // [R, 64, 2]
    long long R, long long S, long long Q, bool w_batched) {
  const long long NC = 64;
  long long clen = (Q + NC - 1) / NC;
  long long total = R * NC;
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       t < total; t += stride) {
    long long r = t / NC, c = t % NC;
    const double* pr = p + r * S;
    const double* wr = w_batched ? (w + r * S * 2) : w;
    double lo = 0.0, hi = 0.0;
    long long q0 = c * clen;
    long long q1 = min(q0 + clen, Q);
    for (long long q = q0; q < q1; ++q) {
      int64_t id = idx[q];
      pt_iv_acc(pr[id], wr[2 * id], wr[2 * id + 1], lo, hi);
    }
    work[2 * t] = lo;
    work[2 * t + 1] = hi;
  }
}

// M == 1 pass 2: one thread per r folds the 64 chunk partials IN ORDER.
__global__ void seg_dot1_reduce_kernel(
    const double* __restrict__ work,   // [R, 64, 2]
    double* __restrict__ out,          // [R, 1, 2]
    long long R) {
  const long long NC = 64;
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       r < R; r += stride) {
    double lo = 0.0, hi = 0.0;
    const double* wr = work + r * NC * 2;
    for (long long c = 0; c < NC; ++c) {
      lo = __dadd_rd(lo, wr[2 * c]);
      hi = __dadd_ru(hi, wr[2 * c + 1]);
    }
    out[2 * r] = lo;
    out[2 * r + 1] = hi;
  }
}

// ---------------------------------------------------------------------------
// Elementwise interval ops (one thread per interval element, [..., 2] layout
// flattened to N elements).  Same soundness story as the segment kernels:
// every lower endpoint rounds toward -inf, every upper toward +inf — sound
// with zero emulation slack, deterministic (pure elementwise / fixed
// sequential reductions).  All are CUDA-graph capture-safe: no syncs, no
// device queries, outputs allocated by the caller-side torch::empty.
// ---------------------------------------------------------------------------

__global__ void iv_mul_kernel(
    const double* __restrict__ a,   // [N, 2]
    const double* __restrict__ b,   // [N, 2]
    double* __restrict__ out,       // [N, 2]
    long long N) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       t < N; t += stride) {
    double al = a[2 * t], ah = a[2 * t + 1];
    double bl = b[2 * t], bh = b[2 * t + 1];
    out[2 * t] = fmin(fmin(__dmul_rd(al, bl), __dmul_rd(al, bh)),
                      fmin(__dmul_rd(ah, bl), __dmul_rd(ah, bh)));
    out[2 * t + 1] = fmax(fmax(__dmul_ru(al, bl), __dmul_ru(al, bh)),
                          fmax(__dmul_ru(ah, bl), __dmul_ru(ah, bh)));
  }
}

// add (sub == false): [alo + blo]_rd, [ahi + bhi]_ru
// sub (sub == true) : [alo - bhi]_rd, [ahi - blo]_ru  (negation is exact, so
// __dadd with the negated operand IS the directed subtraction)
__global__ void iv_addsub_kernel(
    const double* __restrict__ a, const double* __restrict__ b,
    double* __restrict__ out, long long N, bool sub) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       t < N; t += stride) {
    if (sub) {
      out[2 * t] = __dadd_rd(a[2 * t], -b[2 * t + 1]);
      out[2 * t + 1] = __dadd_ru(a[2 * t + 1], -b[2 * t]);
    } else {
      out[2 * t] = __dadd_rd(a[2 * t], b[2 * t]);
      out[2 * t + 1] = __dadd_ru(a[2 * t + 1], b[2 * t + 1]);
    }
  }
}

__global__ void iv_neg_kernel(
    const double* __restrict__ a, double* __restrict__ out, long long N) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       t < N; t += stride) {
    out[2 * t] = -a[2 * t + 1];
    out[2 * t + 1] = -a[2 * t];
  }
}

__global__ void iv_mul_point_kernel(
    const double* __restrict__ a,   // [N, 2]
    const double* __restrict__ p,   // [N]
    double* __restrict__ out,       // [N, 2]
    long long N) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       t < N; t += stride) {
    double pv = p[t];
    out[2 * t] = fmin(__dmul_rd(a[2 * t], pv), __dmul_rd(a[2 * t + 1], pv));
    out[2 * t + 1] = fmax(__dmul_ru(a[2 * t], pv), __dmul_ru(a[2 * t + 1], pv));
  }
}

// Directed interval sum over the last stripped dim: x [N, m, 2] -> out [N, 2].
// One thread per output, SEQUENTIAL in index order — deterministic.
__global__ void iv_sum_kernel(
    const double* __restrict__ x, double* __restrict__ out,
    long long N, long long m) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       t < N; t += stride) {
    const double* row = x + t * m * 2;
    double lo = 0.0, hi = 0.0;
    for (long long j = 0; j < m; ++j) {
      lo = __dadd_rd(lo, row[2 * j]);
      hi = __dadd_ru(hi, row[2 * j + 1]);
    }
    out[2 * t] = lo;
    out[2 * t + 1] = hi;
  }
}

// Directed point-x-interval dot: p [N, m] x w ([m, 2] shared or [N, m, 2])
// -> out [N, 2].  One thread per output, sequential in index order.
__global__ void iv_dot_point_iv_kernel(
    const double* __restrict__ p, const double* __restrict__ w,
    double* __restrict__ out, long long N, long long m, bool w_shared) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       t < N; t += stride) {
    const double* pr = p + t * m;
    const double* wr = w_shared ? w : (w + t * m * 2);
    double lo = 0.0, hi = 0.0;
    for (long long j = 0; j < m; ++j) {
      pt_iv_acc(pr[j], wr[2 * j], wr[2 * j + 1], lo, hi);
    }
    out[2 * t] = lo;
    out[2 * t + 1] = hi;
  }
}

inline int n_blocks(long long total, int threads) {
  long long b = (total + threads - 1) / threads;
  return (int)std::min<long long>(b, 65535);
}

}  // namespace

torch::Tensor seg_mul_iv(torch::Tensor a, torch::Tensor b, torch::Tensor pair_i,
                         torch::Tensor pair_j, torch::Tensor seg_offsets) {
  const long long R = a.size(0), Sa = a.size(1), Sb = b.size(1);
  const long long M = seg_offsets.size(0) - 1;
  auto out = torch::empty({R, M, 2}, a.options());
  const int threads = 256;
  seg_mul_iv_kernel<<<n_blocks(R * M, threads), threads, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<double>(), b.data_ptr<double>(),
      pair_i.data_ptr<int64_t>(), pair_j.data_ptr<int64_t>(),
      seg_offsets.data_ptr<int64_t>(), out.data_ptr<double>(), R, M, Sa, Sb);
  return out;
}

torch::Tensor seg_mul_pt(torch::Tensor a, torch::Tensor b, torch::Tensor pair_i,
                         torch::Tensor pair_j, torch::Tensor seg_offsets) {
  const long long R = a.size(0), Sa = a.size(1), Sb = b.size(1);
  const long long M = seg_offsets.size(0) - 1;
  auto out = torch::empty({R, M}, a.options());
  const int threads = 256;
  seg_mul_pt_kernel<<<n_blocks(R * M, threads), threads, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<double>(), b.data_ptr<double>(),
      pair_i.data_ptr<int64_t>(), pair_j.data_ptr<int64_t>(),
      seg_offsets.data_ptr<int64_t>(), out.data_ptr<double>(), R, M, Sa, Sb);
  return out;
}

torch::Tensor iv_mul(torch::Tensor a, torch::Tensor b) {
  const long long N = a.numel() / 2;
  auto out = torch::empty_like(a);
  const int threads = 256;
  iv_mul_kernel<<<n_blocks(N, threads), threads, 0,
                  at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<double>(), b.data_ptr<double>(), out.data_ptr<double>(), N);
  return out;
}

torch::Tensor iv_addsub(torch::Tensor a, torch::Tensor b, bool sub) {
  const long long N = a.numel() / 2;
  auto out = torch::empty_like(a);
  const int threads = 256;
  iv_addsub_kernel<<<n_blocks(N, threads), threads, 0,
                     at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<double>(), b.data_ptr<double>(), out.data_ptr<double>(), N, sub);
  return out;
}

torch::Tensor iv_neg(torch::Tensor a) {
  const long long N = a.numel() / 2;
  auto out = torch::empty_like(a);
  const int threads = 256;
  iv_neg_kernel<<<n_blocks(N, threads), threads, 0,
                  at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<double>(), out.data_ptr<double>(), N);
  return out;
}

torch::Tensor iv_mul_point(torch::Tensor a, torch::Tensor p) {
  const long long N = a.numel() / 2;
  auto out = torch::empty_like(a);
  const int threads = 256;
  iv_mul_point_kernel<<<n_blocks(N, threads), threads, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<double>(), p.data_ptr<double>(), out.data_ptr<double>(), N);
  return out;
}

torch::Tensor iv_sum(torch::Tensor x) {
  const long long N = x.size(0), m = x.size(1);
  auto out = torch::empty({N, 2}, x.options());
  const int threads = 256;
  iv_sum_kernel<<<n_blocks(N, threads), threads, 0,
                  at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<double>(), out.data_ptr<double>(), N, m);
  return out;
}

torch::Tensor iv_dot_point_iv(torch::Tensor p, torch::Tensor w, bool w_shared) {
  const long long N = p.size(0), m = p.size(1);
  auto out = torch::empty({N, 2}, p.options());
  const int threads = 256;
  iv_dot_point_iv_kernel<<<n_blocks(N, threads), threads, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
      p.data_ptr<double>(), w.data_ptr<double>(), out.data_ptr<double>(), N, m,
      w_shared);
  return out;
}

torch::Tensor seg_dot_pt_iv(torch::Tensor p, torch::Tensor w,
                            torch::Tensor seg_offsets, torch::Tensor idx,
                            bool w_batched) {
  const long long R = p.size(0), S = p.size(1);
  const long long M = seg_offsets.size(0) - 1;
  const long long Q = idx.size(0);
  auto out = torch::empty({R, M, 2}, p.options());
  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (M == 1) {
    auto work = torch::empty({R, 64, 2}, p.options());
    seg_dot1_partial_kernel<<<n_blocks(R * 64, threads), threads, 0, stream>>>(
        p.data_ptr<double>(), w.data_ptr<double>(), idx.data_ptr<int64_t>(),
        work.data_ptr<double>(), R, S, Q, w_batched);
    seg_dot1_reduce_kernel<<<n_blocks(R, threads), threads, 0, stream>>>(
        work.data_ptr<double>(), out.data_ptr<double>(), R);
  } else {
    seg_dot_kernel<<<n_blocks(R * M, threads), threads, 0, stream>>>(
        p.data_ptr<double>(), w.data_ptr<double>(), idx.data_ptr<int64_t>(),
        seg_offsets.data_ptr<int64_t>(), out.data_ptr<double>(), R, M, S,
        w_batched);
  }
  return out;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor seg_mul_iv(torch::Tensor a, torch::Tensor b, torch::Tensor pair_i,
                         torch::Tensor pair_j, torch::Tensor seg_offsets);
torch::Tensor seg_mul_pt(torch::Tensor a, torch::Tensor b, torch::Tensor pair_i,
                         torch::Tensor pair_j, torch::Tensor seg_offsets);
torch::Tensor seg_dot_pt_iv(torch::Tensor p, torch::Tensor w,
                            torch::Tensor seg_offsets, torch::Tensor idx,
                            bool w_batched);
torch::Tensor iv_mul(torch::Tensor a, torch::Tensor b);
torch::Tensor iv_addsub(torch::Tensor a, torch::Tensor b, bool sub);
torch::Tensor iv_neg(torch::Tensor a);
torch::Tensor iv_mul_point(torch::Tensor a, torch::Tensor p);
torch::Tensor iv_sum(torch::Tensor x);
torch::Tensor iv_dot_point_iv(torch::Tensor p, torch::Tensor w, bool w_shared);
"""

# ---------------------------------------------------------------------------
# Lazy JIT build
# ---------------------------------------------------------------------------

_ext = None
_tried = False


def load_cuda_extension(name: str, cpp_src: str, cuda_src: str, functions: list[str]):
    """Shared JIT build plumbing for this project's CUDA extensions.

    Handles the machine-specific quirks ONCE for every kernel module
    (cuda_kernels, tape_kernels): the FLOWSTAR_NO_CUDA_KERNEL=1 opt-out, the
    venv-ninja PATH fix, the sm-arch gencode flag, and the g++-13 host-compiler
    pin (system GCC 15 is unparseable by nvcc 12.x's EDG frontend — libstdc++-15
    headers use builtins EDG lacks; -allow-unsupported-compiler does not help;
    fall back to the default host compiler when g++-13 is absent).

    Returns the loaded extension module, or None on ANY failure (no CUDA, nvcc
    missing, opt-out) — callers keep their torch fallback.
    """
    if os.environ.get("FLOWSTAR_NO_CUDA_KERNEL") == "1":
        return None
    if not torch.cuda.is_available():
        return None
    try:
        # torch's builder shells out to `ninja`; when the venv is used via
        # .venv/bin/python without activation, .venv/bin is not on PATH even
        # though ninja is installed there — prepend it so the build finds it.
        import sys

        venv_bin = os.path.dirname(sys.executable)
        if os.path.exists(os.path.join(venv_bin, "ninja")):
            os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

        from torch.utils.cpp_extension import load_inline

        cap = torch.cuda.get_device_capability()
        arch = f"{cap[0]}{cap[1]}"
        return load_inline(
            name=name,
            cpp_sources=[cpp_src],
            cuda_sources=[cuda_src],
            functions=functions,
            extra_cuda_cflags=[
                "-O3",
                f"-gencode=arch=compute_{arch},code=sm_{arch}",
            ]
            + (["-ccbin", "/usr/bin/g++-13"] if os.path.exists("/usr/bin/g++-13") else []),
            verbose=False,
        )
    except Exception:  # pragma: no cover  # build-env dependent, exercised on CPU CI
        return None


def _build() -> None:
    """Compile the extension once per process (no-op on later calls).

    Never raises: on any failure (no CUDA, nvcc missing, opt-out env var) the
    module simply reports available() == False and callers use the torch path.
    """
    global _ext, _tried
    if _tried:
        return
    _tried = True
    _ext = load_cuda_extension(
        "flowstar_seg_kernels",
        _CPP_SRC,
        _CUDA_SRC,
        [
            "seg_mul_iv",
            "seg_mul_pt",
            "seg_dot_pt_iv",
            "iv_mul",
            "iv_addsub",
            "iv_neg",
            "iv_mul_point",
            "iv_sum",
            "iv_dot_point_iv",
        ],
    )


def _guarded(dev, fn, *args):
    """Launch an extension binding with the CUDA device of its tensors made
    current. The kernels launch on the CURRENT device's stream; without the
    guard, tensors on cuda:N with current device cuda:0 produce an illegal
    memory access (M10 GOTCHA: found the first time a run targeted cuda:2).
    Overhead is ~2 us per call — negligible at our call rates."""
    with torch.cuda.device(dev):
        return fn(*args)


def available() -> bool:
    """True iff the CUDA extension compiled and kernels can be called."""
    _build()
    return _ext is not None


def _require() -> None:
    if not available():
        raise RuntimeError(
            "flowstar_gpu CUDA kernels unavailable "
            "(no CUDA / build failed / FLOWSTAR_NO_CUDA_KERNEL=1); "
            "use the *_ref torch fallback"
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_f64_cuda(name: str, t: torch.Tensor, dims: int) -> torch.Tensor:
    if t.dtype is not torch.float64:
        raise TypeError(f"{name}: rigorous kernels are float64-only, got {t.dtype}")
    if not t.is_cuda:
        raise TypeError(f"{name}: expected a CUDA tensor")
    if t.dim() != dims:
        raise ValueError(f"{name}: expected {dims}-D, got shape {tuple(t.shape)}")
    return t.contiguous()


def _check_idx(name: str, t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if t.dtype is not torch.int64:
        raise TypeError(f"{name}: indices must be int64, got {t.dtype}")
    if t.device != device:
        raise TypeError(f"{name}: index tensor on {t.device}, data on {device}")
    return t.contiguous()


def _check_pairs(
    pair_i: torch.Tensor, pair_j: torch.Tensor, seg_offsets: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pair_i = _check_idx("pair_i", pair_i, device)
    pair_j = _check_idx("pair_j", pair_j, device)
    seg_offsets = _check_idx("seg_offsets", seg_offsets, device)
    if pair_i.shape != pair_j.shape:
        raise ValueError(
            f"pair_i/pair_j length mismatch: {pair_i.shape[0]} vs {pair_j.shape[0]}"
        )
    return pair_i, pair_j, seg_offsets


# ---------------------------------------------------------------------------
# Public kernels
# ---------------------------------------------------------------------------


def seg_mul_iv(
    a: torch.Tensor,
    b: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    seg_offsets: torch.Tensor,
) -> torch.Tensor:
    """Interval pair-product segment sum (module docstring layout contract).

    a [R, Sa, 2], b [R, Sb, 2] finite interval coefficients; pair_i/pair_j [P]
    factor indices sorted by output slot; seg_offsets [M+1] segment bounds.
    Returns out [R, M, 2] with true directed rounding (sound AND tighter than
    the RN+nextafter+Rump emulation).
    """
    _require()
    a = _check_f64_cuda("a", a, 3)
    b = _check_f64_cuda("b", b, 3)
    pair_i, pair_j, seg_offsets = _check_pairs(pair_i, pair_j, seg_offsets, a.device)
    return _guarded(a.device, _ext.seg_mul_iv, a, b, pair_i, pair_j, seg_offsets)


def seg_mul_pt(
    a: torch.Tensor,
    b: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    seg_offsets: torch.Tensor,
) -> torch.Tensor:
    """Point pair-product segment sum: RN fma in pair order (tier P).

    a [R, Sa], b [R, Sb] -> out [R, M]. Deterministic (fixed order); FMA is
    within tier-P's contract (Flow* leaves this roundoff unbounded likewise).
    """
    _require()
    a = _check_f64_cuda("a", a, 2)
    b = _check_f64_cuda("b", b, 2)
    pair_i, pair_j, seg_offsets = _check_pairs(pair_i, pair_j, seg_offsets, a.device)
    return _guarded(a.device, _ext.seg_mul_pt, a, b, pair_i, pair_j, seg_offsets)


def seg_dot_pt_iv(
    p: torch.Tensor,
    w: torch.Tensor,
    seg_offsets: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """Point-times-interval segment dot (range evaluation core).

    p [R, S] point values; w [S, 2] (shared) or [R, S, 2] (per-row) interval
    weights; idx [Q] slot per term; seg_offsets [M+1]. Returns out [R, M, 2].
    M == 1 runs the fixed 64-chunk two-pass schedule (parallel over R x chunk,
    deterministic — chunking depends only on Q); M >= 2 one thread per (r, m).
    """
    _require()
    p = _check_f64_cuda("p", p, 2)
    if w.dim() == 2:
        w_batched = False
        w = _check_f64_cuda("w", w, 2)
        if w.shape[0] != p.shape[1]:
            raise ValueError(f"w rows {w.shape[0]} != p slots {p.shape[1]}")
    elif w.dim() == 3:
        w_batched = True
        w = _check_f64_cuda("w", w, 3)
        if w.shape[:2] != p.shape:
            raise ValueError(f"w leading shape {tuple(w.shape[:2])} != p {tuple(p.shape)}")
    else:
        raise ValueError(f"w must be [S, 2] or [R, S, 2], got {tuple(w.shape)}")
    if w.shape[-1] != 2:
        raise ValueError("w last dim must be 2 (interval endpoints)")
    idx = _check_idx("idx", idx, p.device)
    seg_offsets = _check_idx("seg_offsets", seg_offsets, p.device)
    return _guarded(p.device, _ext.seg_dot_pt_iv, p, w, seg_offsets, idx, w_batched)


# ---------------------------------------------------------------------------
# Elementwise interval ops (dispatch targets of interval.py's kernel path)
# ---------------------------------------------------------------------------
# Contract shared by all: f64 CUDA tensors, caller has already broadcast to a
# COMMON shape; wrappers only enforce dtype and make inputs contiguous.  The
# torch fallback (interval.py's original bodies) is the reference oracle.
# Every op is CUDA-graph capture-safe (pure kernel launches + torch.empty).


def _ew_check(name: str, t: torch.Tensor) -> torch.Tensor:
    if t.dtype is not torch.float64:
        raise TypeError(f"{name}: rigorous kernels are float64-only, got {t.dtype}")
    if not t.is_cuda:
        raise TypeError(f"{name}: expected a CUDA tensor")
    return t.contiguous()


def iv_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise interval product a [..., 2] * b [..., 2] (same shape).

    Four directed candidate products per element (__dmul_rd for the min,
    __dmul_ru for the max) — one kernel instead of the ~9-launch torch chain,
    sound with zero emulation slack. Finite-inputs contract as interval.mul.
    """
    _require()
    a, b = _ew_check("a", a), _ew_check("b", b)
    if a.shape != b.shape:
        raise ValueError(f"iv_mul: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    return _guarded(a.device, _ext.iv_mul, a, b)


def iv_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise directed interval add: [alo+blo]_rd, [ahi+bhi]_ru."""
    _require()
    a, b = _ew_check("a", a), _ew_check("b", b)
    if a.shape != b.shape:
        raise ValueError(f"iv_add: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    return _guarded(a.device, _ext.iv_addsub, a, b, False)


def iv_sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise directed interval sub: [alo-bhi]_rd, [ahi-blo]_ru."""
    _require()
    a, b = _ew_check("a", a), _ew_check("b", b)
    if a.shape != b.shape:
        raise ValueError(f"iv_sub: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    return _guarded(a.device, _ext.iv_addsub, a, b, True)


def iv_neg(a: torch.Tensor) -> torch.Tensor:
    """Elementwise interval negation (-hi, -lo) — exact, one kernel."""
    _require()
    return _guarded(a.device, _ext.iv_neg, _ew_check("a", a))


def iv_mul_point(a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Interval a [..., 2] times point p [...] (p pre-broadcast to a[..., 0])."""
    _require()
    a, p = _ew_check("a", a), _ew_check("p", p)
    if a.shape[:-1] != p.shape:
        raise ValueError(
            f"iv_mul_point: point shape {tuple(p.shape)} != {tuple(a.shape[:-1])}"
        )
    return _guarded(a.device, _ext.iv_mul_point, a, p)


def iv_sum(x: torch.Tensor) -> torch.Tensor:
    """Directed interval sum over the middle dim: x [N, m, 2] -> [N, 2].

    Sequential in-order accumulation per output (deterministic; row results
    independent of N).
    """
    _require()
    x = _ew_check("x", x)
    if x.dim() != 3 or x.shape[-1] != 2:
        raise ValueError(f"iv_sum: expected [N, m, 2], got {tuple(x.shape)}")
    return _guarded(x.device, _ext.iv_sum, x)


def iv_dot_point_iv(p: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Directed point-x-interval dot: p [N, m] x w [m, 2] (shared) or
    [N, m, 2] (per-row) -> [N, 2]. Sequential per output, deterministic."""
    _require()
    p, w = _ew_check("p", p), _ew_check("w", w)
    if p.dim() != 2:
        raise ValueError(f"iv_dot_point_iv: p must be [N, m], got {tuple(p.shape)}")
    if w.dim() == 2:
        if w.shape != (p.shape[1], 2):
            raise ValueError(f"iv_dot_point_iv: shared w must be [m, 2], got {tuple(w.shape)}")
        return _guarded(p.device, _ext.iv_dot_point_iv, p, w, True)
    if w.dim() == 3:
        if w.shape != (*p.shape, 2):
            raise ValueError(f"iv_dot_point_iv: w must be [N, m, 2], got {tuple(w.shape)}")
        return _guarded(p.device, _ext.iv_dot_point_iv, p, w, False)
    raise ValueError(f"iv_dot_point_iv: w must be [m, 2] or [N, m, 2], got {tuple(w.shape)}")


# ---------------------------------------------------------------------------
# Torch reference implementations (fallback path + test/bench baseline)
# ---------------------------------------------------------------------------
# These mirror the EXISTING dense-path emulation semantics — RN arithmetic,
# one nextafter step outward per elementwise op, Rump a-priori bound per
# reduction — deliberately self-contained (no imports from the modules being
# refactored around this file).  Consequently they are (slightly) WIDER than
# the fused kernels on identical inputs; the tests assert exactly that.

_U = 2.0**-53  # unit roundoff of f64
_ETA = 5e-324  # smallest positive subnormal


def _next_up(x: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(x, torch.full_like(x, math.inf))


def _next_down(x: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(x, torch.full_like(x, -math.inf))


def _rump_bound(abs_dot_rn: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """A-priori |RN dot - exact| bound, per-slot reduction lengths m [M]."""
    m_f = m.to(abs_dot_rn.dtype)
    factor = (2.0 * m_f - 1.0).clamp(min=0.0) * _U * (1.0 + 2.0 * m_f * _U)
    bound = _next_up(_next_up(abs_dot_rn * factor) + m_f * _ETA)
    return torch.where(m_f > 0, bound, torch.zeros_like(bound))


def _seg_sum(vals: torch.Tensor, seg_offsets: torch.Tensor) -> torch.Tensor:
    """RN segment sum [..., P] -> [..., M] via segment_reduce (any order)."""
    lengths = (seg_offsets[1:] - seg_offsets[:-1]).to(vals.device)  # [M]
    expanded = lengths.expand(*vals.shape[:-1], lengths.shape[0]).contiguous()
    return torch.segment_reduce(vals, "sum", lengths=expanded, axis=vals.dim() - 1)


def seg_mul_iv_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    seg_offsets: torch.Tensor,
) -> torch.Tensor:
    """Emulated-rounding reference for seg_mul_iv (wider; any device)."""
    pa = a[:, pair_i]  # [R, P, 2]
    pb = b[:, pair_j]  # [R, P, 2]
    cand = torch.stack(
        (
            pa[..., 0] * pb[..., 0],
            pa[..., 0] * pb[..., 1],
            pa[..., 1] * pb[..., 0],
            pa[..., 1] * pb[..., 1],
        ),
        dim=-1,
    )  # [R, P, 4]
    plo = _next_down(cand.amin(dim=-1))  # [R, P]
    phi = _next_up(cand.amax(dim=-1))
    lo_hat = _seg_sum(plo, seg_offsets)  # [R, M]
    hi_hat = _seg_sum(phi, seg_offsets)
    mag_hat = _seg_sum(torch.maximum(plo.abs(), phi.abs()), seg_offsets)
    lengths = (seg_offsets[1:] - seg_offsets[:-1]).to(a.device)
    err = _rump_bound(mag_hat, lengths)
    return torch.stack((_next_down(lo_hat - err), _next_up(hi_hat + err)), dim=-1)


def seg_mul_pt_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    seg_offsets: torch.Tensor,
) -> torch.Tensor:
    """RN (non-fma) reference for seg_mul_pt — tier-P has no error contract,
    so this NEED NOT be bitwise-equal to the kernel; it is the fallback path."""
    prods = a[:, pair_i] * b[:, pair_j]  # [R, P]
    return _seg_sum(prods, seg_offsets)


def seg_dot_pt_iv_ref(
    p: torch.Tensor,
    w: torch.Tensor,
    seg_offsets: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """Emulated-rounding reference for seg_dot_pt_iv (wider; any device)."""
    pv = p[:, idx]  # [R, Q]
    wv = w[idx] if w.dim() == 2 else w[:, idx]  # [Q, 2] or [R, Q, 2]
    if wv.dim() == 2:
        wv = wv.unsqueeze(0)  # [1, Q, 2] broadcast over R
    c1 = pv * wv[..., 0]  # [R, Q]
    c2 = pv * wv[..., 1]
    lo_hat = _seg_sum(torch.minimum(c1, c2), seg_offsets)  # [R, M]
    hi_hat = _seg_sum(torch.maximum(c1, c2), seg_offsets)
    mag_hat = _seg_sum(torch.maximum(c1.abs(), c2.abs()), seg_offsets)
    lengths = (seg_offsets[1:] - seg_offsets[:-1]).to(p.device)
    err = _rump_bound(mag_hat, lengths)
    return torch.stack((_next_down(lo_hat - err), _next_up(hi_hat + err)), dim=-1)
