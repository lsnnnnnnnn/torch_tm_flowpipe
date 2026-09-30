"""Single-kernel refinement-loop interpreter (M10 phase 3a).

The remainder refinement (flowpipe.refine_loop -> ode_compiler.exec_replay ->
elementary *_series_replay twins) is pure per-lane scalar interval arithmetic:
a tape of 10-300 instructions over [2]-sized intervals, iterated up to
MAX_REFINEMENT_STEPS times. Launched eagerly (or even as one CUDA graph per
iteration) that is thousands of ~2 us kernels; here the ENTIRE loop — all
iterations, all instructions, the containment/STOP_RATIO control flow — runs
as ONE kernel launch with ONE THREAD PER LANE.

Fidelity contract (the load-bearing design decision)
----------------------------------------------------
The device arithmetic REPLICATES the CURRENT eager-on-CUDA arithmetic MIX
op for op (doc-verification finding: an earlier version of this paragraph
claimed pure nextafter emulation — wrong): interval add/sub/mul use the
__dadd_rd/ru + __dmul_rd/ru directed intrinsics EXACTLY as interval.py's
kernel dispatch does on the eager path, while rec/pow_int/_div_int/
_mul_pos_int/width keep the nextafter emulation those eager helpers use,
with the same candidate/min/max shapes and the same op ORDER (Flow*'s
c1*R2 + c2*R1 + R1*R2 + c3 add order, the series twins' scalar-division
asymmetries, GOTCHAS #5 first-fail-dim update semantics), and
transcendental enclosures with transcendental.py's exact algorithms and
slack constants. Whatever the eager path computes, this kernel computes
bitwise: refinement CONTROL decisions (containment tests, width ratios vs
STOP_RATIO) are data-dependent, and only value-identical arithmetic makes
the kernel's control flow structurally identical to the audited eager
loop — directed rounding would produce ulp-level different
remainders whose knife-edge control could diverge. Tightness was never the
refinement bottleneck; launch count was. On CUDA the eager path's math library
calls (torch.sin etc.) lower to the same libdevice functions device code
calls, so kernel results are bitwise-equal to eager-on-CUDA in practice (the
test suite asserts value agreement and exact control agreement).

Transcendental soundness on device
----------------------------------
transcendental.py's slack constants were calibrated for BOTH torch-cpu and
torch-cuda (its module docstring); the CUDA C++ Programming Guide (13.0,
"Mathematical Functions" appendix, double-precision intrinsics table)
documents maximum ulp errors
        sin/cos: 2 ulp    exp: 1 ulp    log: 1 ulp    sqrt: correctly rounded
so transcendental.py's constants (documented bound + 2):
        ULP_SLACK_TRIG = 4, ULP_SLACK_EXP = 3, ULP_SLACK_LOG = 3, sqrt = 1
cover the device library with the same margin — the device port uses the SAME
constants (mirrored below as _DEV_ULP_*), the same directed-pi brackets, the
same TRIG_ENVELOPE = 1e12 precision cliff, and the same domain-violation
(bad-flag + sanitize-to-zero) policy as elementary._checked_*.

Layout: serialize_replay_tape(code, device) flattens a CompiledODE into int32
device arrays (one entry per instruction: op code, dst/a/b slot ids, var id,
pow exponent, cache_base, strict_slot) plus the elementary constant tables;
refine_tape_run(...) mirrors flowpipe.refine_loop's contract and returns
(cur [B, n, 2], bad [B], iters [B]).

Capacity: slot remainders live in per-thread local arrays — n_slots <= 512 and
n <= 64 (asserted at serialization; callers fall back to the eager loop).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .cuda_kernels import load_cuda_extension
from .ode_compiler import CompiledODE

# Instruction op codes (device switch labels). Order matters only for the
# device switch; keep in sync with _OPCODE below and the CUDA source.
_OPCODE = {
    "var": 0, "const": 1, "neg": 2, "add": 3, "sub": 4, "mul": 5, "pow": 6,
    "div": 7, "sin": 8, "cos": 9, "exp": 10, "log": 11, "sqrt": 12,
}

MAX_SLOTS = 512  # per-thread slot-remainder array size (doubles x2 = 8 KB local)
MAX_N = 64  # per-thread cur/new_rem array size

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

#define MAX_SLOTS 512
#define MAX_N 64

// Slack constants — MUST mirror transcendental.py (module docstring there and
// here). CUDA C++ Programming Guide 13.0 documented max ulp error (double):
// sin/cos 2, exp 1, log 1, sqrt correctly rounded (0).
#define ULP_TRIG 4
#define ULP_EXP 3
#define ULP_LOG 3
#define TRIG_ENVELOPE 1e12
// Directed float64 brackets of pi/2 (transcendental.py _HALF_PI_LO/_HI:
// exact halves of the pi brackets 0x1.921fb54442d18p+1 / ...19p+1).
#define HALF_PI_LO 0x1.921fb54442d18p+0
#define HALF_PI_HI 0x1.921fb54442d19p+0
// torch's clamp(min=5e-324) on the width-ratio denominator.
#define MIN_SUBNORMAL 4.9406564584124654e-324

struct IV { double lo, hi; };

__device__ __forceinline__ double nd(double x) { return nextafter(x, -CUDART_INF); }
__device__ __forceinline__ double nu(double x) { return nextafter(x,  CUDART_INF); }

// ---- interval primitives: EXACT ports of the CURRENT eager-on-CUDA mix ----
// interval.py's mul/add/sub/neg DISPATCH to the directed-rounding fused
// kernels (cuda_kernels.py) on CUDA, while rec / pow_int / _div_int /
// _mul_pos_int and iv.width remain nextafter-emulated — the device code below
// mirrors that exact mix so kernel values are bitwise-equal to the eager loop
// on CUDA (verified by the test battery). Both flavors are sound; matching
// the mix is what makes the CONTROL FLOW provably identical.
__device__ __forceinline__ IV iv_neg(IV a) { return {-a.hi, -a.lo}; }
__device__ __forceinline__ IV iv_add(IV a, IV b) {
  return {__dadd_rd(a.lo, b.lo), __dadd_ru(a.hi, b.hi)};
}
__device__ __forceinline__ IV iv_sub(IV a, IV b) {
  return {__dadd_rd(a.lo, -b.hi), __dadd_ru(a.hi, -b.lo)};
}
__device__ __forceinline__ IV iv_mul(IV a, IV b) {
  double lo = fmin(fmin(__dmul_rd(a.lo, b.lo), __dmul_rd(a.lo, b.hi)),
                   fmin(__dmul_rd(a.hi, b.lo), __dmul_rd(a.hi, b.hi)));
  double hi = fmax(fmax(__dmul_ru(a.lo, b.lo), __dmul_ru(a.lo, b.hi)),
                   fmax(__dmul_ru(a.hi, b.lo), __dmul_ru(a.hi, b.hi)));
  return {lo, hi};
}
// elementary._div_int: direct directed endpoint division by a nonzero int.
__device__ __forceinline__ IV div_int(IV a, int i) {
  double lo, hi;
  if (i > 0) { lo = a.lo / (double)i; hi = a.hi / (double)i; }
  else       { lo = a.hi / (double)i; hi = a.lo / (double)i; }
  return {nd(lo), nu(hi)};
}
// elementary._mul_pos_int: directed endpoint products with a positive int.
__device__ __forceinline__ IV mul_pos_int(IV a, int j) {
  return {nd(a.lo * (double)j), nu(a.hi * (double)j)};
}
// elementary._double: exact *2.
__device__ __forceinline__ IV iv_double(IV a) { return {a.lo * 2.0, a.hi * 2.0}; }

// interval.rec + elementary._checked_rec (NaN defense + sanitize-to-zero).
__device__ IV checked_rec(IV a, bool* bad) {
  bool b = (a.lo <= 0.0 && a.hi >= 0.0) || isnan(a.lo) || isnan(a.hi);
  double lo = nd(1.0 / a.hi), hi = nu(1.0 / a.lo);
  b = b || isinf(lo) || isinf(hi);  // 1/denormal overflow guard (interval.rec)
  if (b) { *bad = true; return {0.0, 0.0}; }
  return {lo, hi};
}
// interval.div (rec-then-mul, Flow* shape) + _checked_div sanitization.
__device__ IV checked_div(IV a, IV b, bool* bad) {
  bool local = isnan(a.lo) || isnan(a.hi);
  IV r = checked_rec(b, &local);
  if (local) { *bad = true; return {0.0, 0.0}; }
  return iv_mul(a, r);
}

// interval.pow_int: odd/even endpoint rule with nextafter chains.
__device__ double chain_up(double x, int p) {
  double acc = x;
  for (int i = 0; i < p - 1; ++i) acc = nu(acc * x);
  return acc;
}
__device__ double chain_down(double x, int p) {
  double acc = x;
  for (int i = 0; i < p - 1; ++i) {
    acc = nd(acc * x);
    if (acc < 0.0) acc = 0.0;  // torch clamp(min=0.0) semantics
  }
  return acc;
}
__device__ IV iv_pow_int(IV a, int p) {
  if (p % 2 == 1) {
    double lo = (a.lo >= 0.0) ? chain_down(a.lo, p) : -chain_up(-a.lo, p);
    double hi = (a.hi >= 0.0) ? chain_up(a.hi, p) : -chain_down(-a.hi, p);
    return {lo, hi};
  }
  double mn = fmin(fabs(a.lo), fabs(a.hi));
  double mx = fmax(fabs(a.lo), fabs(a.hi));
  bool crosses = (a.lo <= 0.0) && (a.hi >= 0.0);
  return {crosses ? 0.0 : chain_down(mn, p), chain_up(mx, p)};
}

// ---- transcendental enclosures: EXACT ports of transcendental.py ----------
__device__ __forceinline__ double steps_dn(double x, int s) {
  for (int i = 0; i < s; ++i) x = nd(x);
  return x;
}
__device__ __forceinline__ double steps_up(double x, int s) {
  for (int i = 0; i < s; ++i) x = nu(x);
  return x;
}

__device__ IV exp_iv_dev(IV a) {
  double lo = steps_dn(exp(a.lo), ULP_EXP);
  if (lo < 0.0) lo = 0.0;  // clamp(min=0): exp > 0 always
  return {lo, steps_up(exp(a.hi), ULP_EXP)};
}

// transcendental.log_iv raw semantics: bad lanes get [-inf, +inf] sentinel.
__device__ IV log_iv_raw(IV a, bool* bad) {
  if (a.lo <= 0.0) { *bad = true; return {-CUDART_INF, CUDART_INF}; }
  return {steps_dn(log(a.lo), ULP_LOG), steps_up(log(a.hi), ULP_LOG)};
}

// transcendental.sqrt_iv raw semantics (lo = 0 is fine, < 0 is bad).
__device__ IV sqrt_iv_raw(IV a, bool* bad) {
  if (a.lo < 0.0) { *bad = true; return {-CUDART_INF, CUDART_INF}; }
  double lo = nd(sqrt(a.lo));
  if (lo < 0.0) lo = 0.0;  // clamp(min=0)
  return {lo, nu(sqrt(a.hi))};
}
// elementary._checked_sqrt: raw + NaN defense + sanitize.
__device__ IV checked_sqrt(IV a, bool* bad) {
  bool local = isnan(a.lo) || isnan(a.hi);
  IV r = sqrt_iv_raw(a, &local);
  if (local) { *bad = true; return {0.0, 0.0}; }
  return r;
}

// transcendental._sin_cos: quadrant machinery, exact port (see its docstring
// for the soundness argument; is_cos selects the residue pair).
__device__ IV sincos_iv_dev(IV a, bool is_cos) {
  const int r_max = is_cos ? 0 : 1;
  const int r_min = is_cos ? 2 : 3;
  double m = fmax(fabs(a.lo), fabs(a.hi));
  if (!(m <= TRIG_ENVELOPE)) return {-1.0, 1.0};  // env lanes (incl NaN-safe)
  double al = a.lo, ah = a.hi;
  double q_lo = nd(fmin(al / HALF_PI_HI, al / HALF_PI_LO));
  double q_hi = nu(fmax(ah / HALF_PI_HI, ah / HALF_PI_LO));
  long long k_lo = (long long)floor(q_lo);
  long long k_hi = (long long)floor(q_hi);
  double v_lo = is_cos ? cos(al) : sin(al);
  double v_hi = is_cos ? cos(ah) : sin(ah);
  double lo_r = steps_dn(fmin(v_lo, v_hi), ULP_TRIG);
  if (lo_r < -1.0) lo_r = -1.0;
  double hi_r = steps_up(fmax(v_lo, v_hi), ULP_TRIG);
  if (hi_r > 1.0) hi_r = 1.0;
  // crossed(r): floor((k_hi-r)/4) > floor((k_lo-r)/4); >> 2 is floor-div by 4.
  if (((k_hi - (long long)r_min) >> 2) > ((k_lo - (long long)r_min) >> 2)) lo_r = -1.0;
  if (((k_hi - (long long)r_max) >> 2) > ((k_lo - (long long)r_max) >> 2)) hi_r = 1.0;
  return {lo_r, hi_r};
}

// ---- Lagrange tails (settings.h:322-452 via elementary.py, exact ports) ---
__device__ __forceinline__ IV tab_iv(const double* t, int i) {
  return {t[2 * i], t[2 * i + 1]};
}

__device__ IV exp_taylor_remainder(IV tm_range, int order, const double* frec) {
  IV prod = iv_pow_int(tm_range, order);
  IV j = exp_iv_dev(tm_range);
  return iv_mul(iv_mul(tab_iv(frec, order), prod), j);
}


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


__device__ IV trig_taylor_remainder(IV c, IV tm_range, int order,
                                    const double* frec, bool cos_cycle) {
  IV prod = iv_pow_int(tm_range, order);
  IV j = iv_add(tm_range, c);
  int k4 = cos_cycle ? (order + 1) % 4 : order % 4;
  IV jv;
  if (k4 == 0)      jv = sincos_iv_dev(j, false);
  else if (k4 == 1) jv = sincos_iv_dev(j, true);
  else if (k4 == 2) jv = iv_neg(sincos_iv_dev(j, false));
  else              jv = iv_neg(sincos_iv_dev(j, true));
  return iv_mul(iv_mul(tab_iv(frec, order), prod), jv);
}

__device__ IV log_taylor_remainder(IV tm_range, int order, bool* bad) {
  IV one = {1.0, 1.0};
  IV r = checked_rec(iv_add(tm_range, one), bad);
  r = iv_mul(r, tm_range);
  r = iv_pow_int(r, order);
  r = div_int(r, order);
  if ((order + 1) % 2 == 1) r = iv_neg(r);
  return r;
}

__device__ IV sqrt_taylor_remainder(IV tm_range, int order, const double* frec,
                                    const double* dfact, bool* bad) {
  IV one = {1.0, 1.0};
  IV r = checked_rec(iv_add(tm_range, one), bad);
  r = checked_sqrt(r, bad);
  r = iv_mul(r, tm_range);
  r = div_int(r, 2);
  r = iv_pow_int(r, order);
  IV factor = iv_mul(tab_iv(dfact, 2 * order - 3), tab_iv(frec, order));
  r = iv_mul(r, factor);
  if (order % 2 == 0) r = iv_neg(r);
  return r;
}

// ---- series replay twins (elementary.py *_series_replay, exact ports) -----
// `crow` = this lane's cache row [C, 2] flattened; `at`/`base` slot indices.
__device__ __forceinline__ IV cache_iv(const double* crow, int at) {
  return {crow[2 * at], crow[2 * at + 1]};
}

// Flow*'s exact MUL replay: c1*R2 + c2*R1 + R1*R2 + c3, in that add order.
__device__ IV replay_mul(IV r1, IV r2, const double* crow, int at) {
  IV out = iv_mul(cache_iv(crow, at), r2);
  out = iv_add(out, iv_mul(cache_iv(crow, at + 1), r1));
  out = iv_add(out, iv_mul(r1, r2));
  out = iv_add(out, cache_iv(crow, at + 2));
  return out;
}

__device__ IV exp_series_replay(IV rem, const double* crow, int base, int k,
                                const double* frec) {
  IV const_part = cache_iv(crow, base);  // exp(c)
  int cursor = base + 1;
  IV result = {0.0, 0.0};
  for (int i = k; i > 0; --i) {
    result = div_int(result, i);  // DIRECT scalar division (replay asymmetry)
    IV t = iv_mul(cache_iv(crow, cursor), rem);
    t = iv_add(t, iv_mul(cache_iv(crow, cursor + 1), result));
    t = iv_add(t, iv_mul(rem, result));
    t = iv_add(t, cache_iv(crow, cursor + 2));
    cursor += 3;
    result = t;
  }
  result = iv_mul(result, const_part);
  IV tm_range = iv_add(cache_iv(crow, cursor), rem);
  IV lag = exp_taylor_remainder(tm_range, k + 1, frec);
  return iv_add(result, iv_mul(const_part, lag));
}

__device__ IV rec_series_replay(IV rem, const double* crow, int base, int k,
                                bool* bad) {
  IV c_f = cache_iv(crow, base);
  IV const_part = cache_iv(crow, base + 1);  // 1/c
  int cursor = base + 2;
  IV r2 = iv_mul(rem, const_part);
  IV result = {0.0, 0.0};
  for (int i = 0; i < k; ++i) {
    result = iv_neg(result);
    IV t = iv_mul(cache_iv(crow, cursor), r2);
    t = iv_add(t, iv_mul(cache_iv(crow, cursor + 1), result));
    t = iv_add(t, iv_mul(r2, result));
    t = iv_add(t, cache_iv(crow, cursor + 2));
    cursor += 3;
    result = t;
  }
  result = iv_mul(result, const_part);
  IV tm_range = iv_add(cache_iv(crow, cursor), r2);
  IV tail = geometric_tail(c_f, const_part, tm_range, k + 1, bad);
  return iv_add(result, tail);
}

__device__ IV trig_series_replay(IV rem, const double* crow, int base, int k,
                                 const double* frec, bool cos_cycle) {
  IV const_part = cache_iv(crow, base);  // c
  int cursor = base + 1;
  IV pow_r = {0.0, 0.0};
  IV result = {0.0, 0.0};
  for (int i = 1; i <= k; ++i) {
    IV t = iv_mul(cache_iv(crow, cursor), rem);
    t = iv_add(t, iv_mul(cache_iv(crow, cursor + 1), pow_r));
    t = iv_add(t, iv_mul(rem, pow_r));
    t = iv_add(t, cache_iv(crow, cursor + 2));
    cursor += 3;
    pow_r = t;
    result = iv_add(result, iv_mul(pow_r, cache_iv(crow, cursor)));  // tmp slot
    cursor += 1;
  }
  IV tm_range = iv_add(cache_iv(crow, cursor), rem);
  IV lag = trig_taylor_remainder(const_part, tm_range, k + 1, frec, cos_cycle);
  return iv_add(result, lag);
}

__device__ IV log_series_replay(IV rem, const double* crow, int base, int k,
                                bool* bad) {
  IV c = cache_iv(crow, base);
  int cursor = base + 1;
  IV r2 = checked_div(rem, c, bad);  // interval division (rec-then-mul)
  IV result = div_int(r2, k);        // DIRECT scalar division (asymmetry)
  for (int i = k; i > 1; --i) {
    result = iv_neg(result);
    IV t = iv_mul(cache_iv(crow, cursor), r2);
    t = iv_add(t, iv_mul(cache_iv(crow, cursor + 1), result));
    t = iv_add(t, iv_mul(r2, result));
    t = iv_add(t, cache_iv(crow, cursor + 2));
    cursor += 3;
    result = t;
  }
  IV tm_range = iv_add(cache_iv(crow, cursor), r2);
  IV lag = log_taylor_remainder(tm_range, k + 1, bad);
  return iv_add(result, lag);
}

__device__ IV sqrt_series_replay(IV rem, const double* crow, int base, int k,
                                 const double* frec, const double* dfact,
                                 bool* bad) {
  IV c = cache_iv(crow, base);
  int cursor = base + 1;
  IV r_c = checked_div(rem, c, bad);
  IV r2 = div_int(r_c, 2);  // (remainder / C) / 2 (replay asymmetry)
  IV result = r2;
  int i = k, j = 2 * k - 3;
  while (i >= 2) {
    result = div_int(result, -i);
    result = mul_pos_int(result, j);
    IV t = iv_mul(cache_iv(crow, cursor), r2);
    t = iv_add(t, iv_mul(cache_iv(crow, cursor + 1), result));
    t = iv_add(t, iv_mul(r2, result));
    t = iv_add(t, cache_iv(crow, cursor + 2));
    cursor += 3;
    result = t;
    i -= 1; j -= 2;
  }
  IV const_part = checked_sqrt(c, bad);
  result = iv_mul(result, const_part);
  IV tm_range = iv_add(cache_iv(crow, cursor), iv_double(r2));
  IV lag = sqrt_taylor_remainder(tm_range, k + 1, frec, dfact, bad);
  return iv_add(result, iv_mul(lag, const_part));
}

// ---- the refinement mega-kernel -------------------------------------------
// One thread per lane b. Interprets the replay tape exactly like
// ode_compiler.exec_replay, then applies flowpipe.refine_loop's control flow
// (containment, GOTCHAS #5 first-fail-dim updates, STOP_RATIO, max_steps).
__global__ void refine_tape_kernel(
    const int* __restrict__ op, const int* __restrict__ dst,
    const int* __restrict__ ia, const int* __restrict__ ib,
    const int* __restrict__ var, const int* __restrict__ power,
    const int* __restrict__ cbase, const int* __restrict__ sslot,
    const int* __restrict__ out_slots,
    int n_instr, int n, int k,
    const double* __restrict__ cache, long long C,   // [B, C, 2]
    const double* __restrict__ tails, long long S,   // [B, S, 2]
    int strict,
    const double* __restrict__ frec,                 // [(rows), 2]
    const double* __restrict__ dfact,                // [(rows), 2]
    double dt_lo, double dt_hi,
    const double* __restrict__ int_diff,             // [B, n, 2]
    const double* __restrict__ accepted,             // [B, n, 2]
    const unsigned char* __restrict__ ok0,
    const unsigned char* __restrict__ bad0,
    double stop_ratio, int max_steps,
    double* __restrict__ cur_out,                    // [B, n, 2]
    unsigned char* __restrict__ bad_out,             // [B]
    int* __restrict__ iters_out,                     // [B]
    long long B) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       b < B; b += stride) {
    const double* crow = cache + b * C * 2;
    const double* trow = tails + b * S * 2;
    double slo[MAX_SLOTS], shi[MAX_SLOTS];
    IV cur[MAX_N], newr[MAX_N];
    for (int d = 0; d < n; ++d) {
      cur[d].lo = accepted[(b * n + d) * 2];
      cur[d].hi = accepted[(b * n + d) * 2 + 1];
    }
    bool refining = ok0[b] != 0;  // refine_loop: refining = ok.clone()
    bool lane_bad = bad0[b] != 0;
    int it_count = 0;

    for (int step = 0; step < max_steps && refining; ++step) {
      ++it_count;
      bool iter_bad = false;
      // ---- exec_replay over the tape (identical instruction semantics) ----
      for (int i = 0; i < n_instr; ++i) {
        IV r;
        switch (op[i]) {
          case 0: {  // var: candidate remainder (+ strict tail)
            r.lo = cur[var[i]].lo; r.hi = cur[var[i]].hi;
            if (strict) r = iv_add(r, {trow[2 * sslot[i]], trow[2 * sslot[i] + 1]});
            break;
          }
          case 1:  // const: the cached (interval-constant) remainder
            r = cache_iv(crow, cbase[i]);
            break;
          case 2:  // neg
            r = iv_neg({slo[ia[i]], shi[ia[i]]});
            break;
          case 3:  // add
            r = iv_add({slo[ia[i]], shi[ia[i]]}, {slo[ib[i]], shi[ib[i]]});
            break;
          case 4:  // sub
            r = iv_sub({slo[ia[i]], shi[ia[i]]}, {slo[ib[i]], shi[ib[i]]});
            break;
          case 5:  // mul
            r = replay_mul({slo[ia[i]], shi[ia[i]]}, {slo[ib[i]], shi[ib[i]]},
                           crow, cbase[i]);
            break;
          case 6: {  // pow: square-and-multiply with cursor discipline
            int degree = power[i];
            if (degree == 0) { r = {0.0, 0.0}; break; }
            IV result = {slo[ia[i]], shi[ia[i]]};
            IV temp = result;
            int cursor = cbase[i];
            int e = degree - 1;
            while (e > 0) {
              if (e & 1) { result = replay_mul(result, temp, crow, cursor); cursor += 3; }
              e >>= 1;
              if (e > 0) { temp = replay_mul(temp, temp, crow, cursor); cursor += 3; }
            }
            r = result;
            break;
          }
          case 7: {  // div: rec twin on denominator, then generic MUL replay
            IV rec_r = rec_series_replay({slo[ib[i]], shi[ib[i]]}, crow,
                                         cbase[i], k, &iter_bad);
            r = replay_mul({slo[ia[i]], shi[ia[i]]}, rec_r, crow,
                           cbase[i] + 3 * k + 3);
            break;
          }
          case 8:  // sin
            r = trig_series_replay({slo[ia[i]], shi[ia[i]]}, crow, cbase[i], k,
                                   frec, false);
            break;
          case 9:  // cos
            r = trig_series_replay({slo[ia[i]], shi[ia[i]]}, crow, cbase[i], k,
                                   frec, true);
            break;
          case 10:  // exp
            r = exp_series_replay({slo[ia[i]], shi[ia[i]]}, crow, cbase[i], k,
                                  frec);
            break;
          case 11:  // log
            r = log_series_replay({slo[ia[i]], shi[ia[i]]}, crow, cbase[i], k,
                                  &iter_bad);
            break;
          default:  // 12: sqrt
            r = sqrt_series_replay({slo[ia[i]], shi[ia[i]]}, crow, cbase[i], k,
                                   frec, dfact, &iter_bad);
            break;
        }
        slo[dst[i]] = r.lo; shi[dst[i]] = r.hi;
      }
      // ---- refine_loop post-replay: rem *= dt; += int_diff ----------------
      for (int d = 0; d < n; ++d) {
        IV r = {slo[out_slots[d]], shi[out_slots[d]]};
        r = iv_mul(r, {dt_lo, dt_hi});
        r = iv_add(r, {int_diff[(b * n + d) * 2], int_diff[(b * n + d) * 2 + 1]});
        if (!isfinite(r.lo) || !isfinite(r.hi)) iter_bad = true;
        newr[d] = r;
      }
      // bad = bad | (bad_replay & refining); refining &= ~bad_replay — a bad
      // replay iteration never updates cur (update mask uses the NEW refining).
      if (iter_bad) { lane_bad = true; refining = false; break; }
      // Containment + GOTCHAS #5 first-fail semantics + STOP_RATIO.
      int first_fail = n;
      for (int d = 0; d < n; ++d) {
        if (!(cur[d].lo <= newr[d].lo && newr[d].hi <= cur[d].hi)) {
          first_fail = d; break;
        }
      }
      bool fail_any = first_fail < n;
      bool improved_any = false;
      for (int d = 0; d < first_fail; ++d) {
        double wn = nu(newr[d].hi - newr[d].lo);   // iv.width (RN + next_up)
        double wo = nu(cur[d].hi - cur[d].lo);
        if (wo < MIN_SUBNORMAL) wo = MIN_SUBNORMAL;  // clamp(min=5e-324)
        if (wn / wo <= stop_ratio) improved_any = true;
        cur[d] = newr[d];
      }
      refining = !fail_any && improved_any;
    }

    for (int d = 0; d < n; ++d) {
      cur_out[(b * n + d) * 2] = cur[d].lo;
      cur_out[(b * n + d) * 2 + 1] = cur[d].hi;
    }
    bad_out[b] = lane_bad ? 1 : 0;
    iters_out[b] = it_count;
  }
}

// ---- transcendental probe (test-only surface) -----------------------------
// which: 0 sin, 1 cos, 2 exp, 3 log, 4 sqrt — raw transcendental.py
// semantics (log/sqrt return the [-inf, inf] sentinel + bad flag).
__global__ void transcendental_probe_kernel(
    const double* __restrict__ x, double* __restrict__ out,
    unsigned char* __restrict__ bad, long long N, int which) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       t < N; t += stride) {
    IV a = {x[2 * t], x[2 * t + 1]};
    IV r; bool b = false;
    if (which == 0)      r = sincos_iv_dev(a, false);
    else if (which == 1) r = sincos_iv_dev(a, true);
    else if (which == 2) r = exp_iv_dev(a);
    else if (which == 3) r = log_iv_raw(a, &b);
    else                 r = sqrt_iv_raw(a, &b);
    out[2 * t] = r.lo; out[2 * t + 1] = r.hi;
    bad[t] = b ? 1 : 0;
  }
}

inline int blocks_for(long long total, int threads) {
  long long nb = (total + threads - 1) / threads;
  return (int)std::min<long long>(nb, 65535);
}

}  // namespace

std::vector<torch::Tensor> refine_tape(
    torch::Tensor op, torch::Tensor dst, torch::Tensor ia, torch::Tensor ib,
    torch::Tensor var, torch::Tensor power, torch::Tensor cbase,
    torch::Tensor sslot, torch::Tensor out_slots, int64_t k,
    torch::Tensor cache, torch::Tensor tails, bool strict,
    torch::Tensor frec, torch::Tensor dfact,
    double dt_lo, double dt_hi,
    torch::Tensor int_diff, torch::Tensor accepted,
    torch::Tensor ok0, torch::Tensor bad0,
    double stop_ratio, int64_t max_steps) {
  const long long B = accepted.size(0);
  const int n = (int)accepted.size(1);
  auto cur = torch::empty_like(accepted);
  auto bad_out = torch::empty({B}, ok0.options());
  auto iters = torch::empty({B}, op.options());
  const int threads = 128;
  refine_tape_kernel<<<blocks_for(B, threads), threads, 0,
                       at::cuda::getCurrentCUDAStream()>>>(
      op.data_ptr<int>(), dst.data_ptr<int>(), ia.data_ptr<int>(),
      ib.data_ptr<int>(), var.data_ptr<int>(), power.data_ptr<int>(),
      cbase.data_ptr<int>(), sslot.data_ptr<int>(), out_slots.data_ptr<int>(),
      (int)op.size(0), n, (int)k,
      cache.data_ptr<double>(), cache.size(1),
      tails.data_ptr<double>(), tails.size(1),
      strict ? 1 : 0,
      frec.data_ptr<double>(), dfact.data_ptr<double>(),
      dt_lo, dt_hi,
      int_diff.data_ptr<double>(), accepted.data_ptr<double>(),
      ok0.data_ptr<unsigned char>(), bad0.data_ptr<unsigned char>(),
      stop_ratio, (int)max_steps,
      cur.data_ptr<double>(), bad_out.data_ptr<unsigned char>(),
      iters.data_ptr<int>(), B);
  return {cur, bad_out, iters};
}

std::vector<torch::Tensor> transcendental_probe(torch::Tensor x, int64_t which) {
  const long long N = x.size(0);
  auto out = torch::empty_like(x);
  auto bad = torch::empty({N}, x.options().dtype(torch::kUInt8));
  const int threads = 128;
  transcendental_probe_kernel<<<blocks_for(N, threads), threads, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<double>(), out.data_ptr<double>(),
      bad.data_ptr<unsigned char>(), N, (int)which);
  return {out, bad};
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> refine_tape(
    torch::Tensor op, torch::Tensor dst, torch::Tensor ia, torch::Tensor ib,
    torch::Tensor var, torch::Tensor power, torch::Tensor cbase,
    torch::Tensor sslot, torch::Tensor out_slots, int64_t k,
    torch::Tensor cache, torch::Tensor tails, bool strict,
    torch::Tensor frec, torch::Tensor dfact,
    double dt_lo, double dt_hi,
    torch::Tensor int_diff, torch::Tensor accepted,
    torch::Tensor ok0, torch::Tensor bad0,
    double stop_ratio, int64_t max_steps);
std::vector<torch::Tensor> transcendental_probe(torch::Tensor x, int64_t which);
"""

_ext = None
_tried = False


def _build() -> None:
    global _ext, _tried
    if _tried:
        return
    _tried = True
    _ext = load_cuda_extension(
        'flowstar_recip_geom_replay_1f9efda6325c', _CPP_SRC, _CUDA_SRC,
        ["refine_tape", "transcendental_probe"],
    )


def available() -> bool:
    """True iff the tape-interpreter extension compiled."""
    _build()
    return _ext is not None


# ---------------------------------------------------------------------------
# Tape serialization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayTape:
    """Flattened replay tape on one device (serialize_replay_tape).

    Instruction arrays (all [n_instr] int32 on `device`): op / dst / a / b /
    var / power / cache_base / strict_slot, plus out_slots [n] and the
    elementary constant tables frec / dfact [(rows), 2] f64 (interval
    enclosures from elementary.elem_tables; 1-row zeros for pure-poly tapes).
    `k` is the series order the cache layout was compiled for (0 when None).
    """

    op: torch.Tensor
    dst: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    var: torch.Tensor
    power: torch.Tensor
    cache_base: torch.Tensor
    strict_slot: torch.Tensor
    out_slots: torch.Tensor
    frec: torch.Tensor
    dfact: torch.Tensor
    n: int
    n_slots: int
    n_cache: int
    n_strict: int
    k: int
    device: str


def serialize_replay_tape(code: CompiledODE, device: str) -> ReplayTape:
    """Flatten a CompiledODE's replay semantics into device arrays.

    Raises ValueError when the tape exceeds the kernel's local-array capacity
    (n_slots > 512 or n > 64) — callers must fall back to the eager loop.
    """
    if code.n_slots > MAX_SLOTS:
        raise ValueError(
            f"tape has {code.n_slots} slots > kernel capacity {MAX_SLOTS}; "
            "use the eager refine_loop"
        )
    if code.n > MAX_N:
        raise ValueError(
            f"ODE has {code.n} components > kernel capacity {MAX_N}; "
            "use the eager refine_loop"
        )
    cols = {f: [] for f in ("op", "dst", "a", "b", "var", "power", "cache_base",
                            "strict_slot")}
    for ins in code.instrs:
        cols["op"].append(_OPCODE[ins.op])
        cols["dst"].append(ins.dst)
        cols["a"].append(ins.a)
        cols["b"].append(ins.b)
        cols["var"].append(ins.var)
        cols["power"].append(ins.power)
        cols["cache_base"].append(ins.cache_base)
        cols["strict_slot"].append(ins.strict_slot)
    as_i32 = lambda v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731

    k = code.order or 0
    if code.has_elem:
        from .elementary import elem_tables

        tabs = elem_tables(k, "cpu")
        frec = tabs.frec_iv.to(device=device, dtype=torch.float64)
        dfact = tabs.dfact_iv.to(device=device, dtype=torch.float64)
    else:
        frec = torch.zeros(1, 2, dtype=torch.float64, device=device)
        dfact = torch.zeros(1, 2, dtype=torch.float64, device=device)

    return ReplayTape(
        op=as_i32(cols["op"]),
        dst=as_i32(cols["dst"]),
        a=as_i32(cols["a"]),
        b=as_i32(cols["b"]),
        var=as_i32(cols["var"]),
        power=as_i32(cols["power"]),
        cache_base=as_i32(cols["cache_base"]),
        strict_slot=as_i32(cols["strict_slot"]),
        out_slots=as_i32(code.out_slots),
        frec=frec.contiguous(),
        dfact=dfact.contiguous(),
        n=code.n,
        n_slots=code.n_slots,
        n_cache=code.n_cache,
        n_strict=code.n_strict,
        k=k,
        device=device,
    )


# ---------------------------------------------------------------------------
# The one-launch refinement loop
# ---------------------------------------------------------------------------


def refine_tape_run(
    tape: ReplayTape,
    accepted: torch.Tensor,
    ok: torch.Tensor,
    bad: torch.Tensor,
    cache: torch.Tensor,
    tails: torch.Tensor,
    strict: bool,
    int_diff: torch.Tensor,
    dt: tuple[float, float] | torch.Tensor,
    stop_ratio: float,
    max_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the ENTIRE refinement loop as one kernel launch.

    Mirrors flowpipe.refine_loop(code, accepted, ok, bad, cache, tails,
    strict, int_diff, step, settings): accepted [B, n, 2] starting remainders,
    ok [B] bool lanes to refine, bad [B] bool pre-set failures, cache
    [B, C, 2] from exec_valid, tails [B, S, 2] strict tails, int_diff
    [B, n, 2], dt the step interval [0, delta] as (lo, hi) floats (pass a
    tensor only outside hot loops — it costs one host sync to read).

    Returns (cur [B, n, 2], bad [B], iters [B] int32) — cur/bad exactly as
    refine_loop returns them, iters the per-lane count of refinement
    iterations entered (diagnostic; eager equivalent: iterations during which
    the lane's `refining` flag was still True at loop top).
    """
    if not available():
        raise RuntimeError("tape kernels unavailable; use flowpipe.refine_loop")
    if isinstance(dt, torch.Tensor):
        dt_lo, dt_hi = (float(v) for v in dt.reshape(2).tolist())  # host sync
    else:
        dt_lo, dt_hi = dt
    accepted = accepted.contiguous()
    B = accepted.shape[0]
    if cache.shape[0] != B or int_diff.shape[0] != B or tails.shape[0] != B:
        raise ValueError("refine_tape_run: batch-dim mismatch across inputs")
    from .cuda_kernels import _guarded as _dg
    cur, bad_out, iters = _dg(accepted.device, _ext.refine_tape, 
        tape.op, tape.dst, tape.a, tape.b, tape.var, tape.power,
        tape.cache_base, tape.strict_slot, tape.out_slots, tape.k,
        cache.contiguous(), tails.contiguous(), bool(strict),
        tape.frec, tape.dfact, dt_lo, dt_hi,
        int_diff.contiguous(), accepted,
        ok.to(torch.uint8).contiguous(), bad.to(torch.uint8).contiguous(),
        float(stop_ratio), int(max_steps),
    )
    return cur, bad_out.to(torch.bool), iters


def transcendental_probe(x: torch.Tensor, which: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Test-only surface: evaluate ONE device transcendental enclosure.

    x [N, 2] CUDA f64 -> (enclosure [N, 2], bad [N] bool). `which` in
    {"sin", "cos", "exp", "log", "sqrt"} — raw transcendental.py semantics
    (log/sqrt: [-inf, inf] sentinel + bad on domain violations).
    """
    if not available():
        raise RuntimeError("tape kernels unavailable")
    idx = {"sin": 0, "cos": 1, "exp": 2, "log": 3, "sqrt": 4}[which]
    from .cuda_kernels import _guarded as _dg2
    out, bad = _dg2(x.device, _ext.transcendental_probe, x.contiguous(), idx)
    return out, bad.to(torch.bool)


# ===========================================================================
# Phase 3b: block-per-lane interpreters for the sparse valid & point passes
# ===========================================================================
# Design: serialize_valid_tape / serialize_point_tape LOWER a SpecTape into a
# flat stream of primitive micro-ops (encoded as int32/f64 field arrays plus
# shared index/pair/factor/constant pools); the device kernel is a switch
# interpreter, one CUDA BLOCK per lane, polynomial slot buffers in dynamic
# shared memory, scalar/remainder registers in shared, __syncthreads between
# micro-ops. All loop bounds, supports, cache cursors and liveness are
# resolved at serialization (pure Python, cached by the caller).
#
# ARITHMETIC PARITY: the same policy as the 3a refine kernel — replicate the
# CURRENT eager-on-CUDA mix bitwise: pair products and range dots with the
# phase-1/2 kernel semantics (directed products, IN-ORDER directed
# accumulation), elementwise interval ops directed (interval.py dispatch),
# rec / nextafter-emulated pieces exactly as elementary/support/rounding
# compute them, transcendentals via the shared device ports (bitwise vs
# transcendental.py on CUDA). Known non-bitwise site: NONE by construction —
# every reduction here is sequential in index order, matching the eager
# kernels' segment/sum semantics (the differential suite asserts bitwise).

_VOP = dict(
    VAR=1, CONST=2, NEG=3, ADDSUB=4, RANGE=5, MULP=6, CUT=7,
    R_MUL=8, R_ADD=9, R_SUB=10, R_NEG=11, R_COPY=12, R_MULT=13, R_MULI=14,
    R_ZERO=15, CACHE_WR=16, R_EXP=17, R_SIN=18, R_COS=19, R_RECCHK=20,
    R_LOGCHK=21, R_SQRTCHK=22, R_DBL=23,
    LAG_EXP=24, LAG_REC=25, LAG_SIN=26, LAG_COS=27, LAG_LOG=28, LAG_SQRT=29,
    BUF_ONE=30, SPLIT=32, BSC_REG=33, BSC_TAB=34, BSC_IMM=35,
    S0_ADD1=36, S0_ADDR=37, S0_SUBI=38, BUF_COPY=39, BUF_NEG=40, SET0R=41,
    OUT_IV=43, OUT_REM=44,
)

_POP = dict(
    VAR=1, CONST=2, NEG=3, ADDSUB=4, MULP=5, CUT=6, PREFIX=7, ONE=8,
    SPLITC=10, SC_REG=11, SC_IMM=12, DIV_IMM=13, A0_IMM=14, A0_REG=15,
    SET0=16, COPY=17, R_EXP=19, R_SIN=20, R_COS=21, R_LOG=22, R_SQRT=23,
    R_REC=24, R_DIVI=25, R_NEG=26, CUTBUF=27, OUT=28, DIVREG=29, R_MULI=30,
)

_VALID_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector>

namespace {

// ---- shared device library (mirrors the 3a refine kernel's primitives) ----
struct IV { double lo, hi; };
__device__ __forceinline__ double nd(double x) { return nextafter(x, -CUDART_INF); }
__device__ __forceinline__ double nu(double x) { return nextafter(x,  CUDART_INF); }
__device__ __forceinline__ IV iv_neg(IV a) { return {-a.hi, -a.lo}; }
__device__ __forceinline__ IV iv_add(IV a, IV b) {
  return {__dadd_rd(a.lo, b.lo), __dadd_ru(a.hi, b.hi)};
}
__device__ __forceinline__ IV iv_sub(IV a, IV b) {
  return {__dadd_rd(a.lo, -b.hi), __dadd_ru(a.hi, -b.lo)};
}
__device__ __forceinline__ IV iv_mul(IV a, IV b) {
  double lo = fmin(fmin(__dmul_rd(a.lo, b.lo), __dmul_rd(a.lo, b.hi)),
                   fmin(__dmul_rd(a.hi, b.lo), __dmul_rd(a.hi, b.hi)));
  double hi = fmax(fmax(__dmul_ru(a.lo, b.lo), __dmul_ru(a.lo, b.hi)),
                   fmax(__dmul_ru(a.hi, b.lo), __dmul_ru(a.hi, b.hi)));
  return {lo, hi};
}
__device__ __forceinline__ IV div_int(IV a, int i) {
  double lo, hi;
  if (i > 0) { lo = a.lo / (double)i; hi = a.hi / (double)i; }
  else       { lo = a.hi / (double)i; hi = a.lo / (double)i; }
  return {nd(lo), nu(hi)};
}
__device__ IV checked_rec(IV a, bool* bad) {
  bool b = (a.lo <= 0.0 && a.hi >= 0.0) || isnan(a.lo) || isnan(a.hi);
  double lo = nd(1.0 / a.hi), hi = nu(1.0 / a.lo);
  b = b || isinf(lo) || isinf(hi);
  if (b) { *bad = true; return {0.0, 0.0}; }
  return {lo, hi};
}
__device__ IV checked_div(IV a, IV b, bool* bad) {
  bool local = isnan(a.lo) || isnan(a.hi);
  IV r = checked_rec(b, &local);
  if (local) { *bad = true; return {0.0, 0.0}; }
  return iv_mul(a, r);
}
__device__ double chain_up(double x, int p) {
  double acc = x;
  for (int i = 0; i < p - 1; ++i) acc = nu(acc * x);
  return acc;
}
__device__ double chain_down(double x, int p) {
  double acc = x;
  for (int i = 0; i < p - 1; ++i) { acc = nd(acc * x); if (acc < 0.0) acc = 0.0; }
  return acc;
}
__device__ IV iv_pow_int(IV a, int p) {
  if (p % 2 == 1) {
    double lo = (a.lo >= 0.0) ? chain_down(a.lo, p) : -chain_up(-a.lo, p);
    double hi = (a.hi >= 0.0) ? chain_up(a.hi, p) : -chain_down(-a.hi, p);
    return {lo, hi};
  }
  double mn = fmin(fabs(a.lo), fabs(a.hi));
  double mx = fmax(fabs(a.lo), fabs(a.hi));
  bool crosses = (a.lo <= 0.0) && (a.hi >= 0.0);
  return {crosses ? 0.0 : chain_down(mn, p), chain_up(mx, p)};
}
#define ULP_TRIG 4
#define ULP_EXP 3
#define ULP_LOG 3
#define TRIG_ENVELOPE 1e12
#define HALF_PI_LO 0x1.921fb54442d18p+0
#define HALF_PI_HI 0x1.921fb54442d19p+0
__device__ __forceinline__ double steps_dn(double x, int s) {
  for (int i = 0; i < s; ++i) x = nd(x);
  return x;
}
__device__ __forceinline__ double steps_up(double x, int s) {
  for (int i = 0; i < s; ++i) x = nu(x);
  return x;
}
__device__ IV exp_iv_dev(IV a) {
  double lo = steps_dn(exp(a.lo), ULP_EXP);
  if (lo < 0.0) lo = 0.0;
  return {lo, steps_up(exp(a.hi), ULP_EXP)};
}
__device__ IV log_checked(IV a, bool* bad) {
  if (a.lo <= 0.0 || isnan(a.lo) || isnan(a.hi)) { *bad = true; return {0.0, 0.0}; }
  return {steps_dn(log(a.lo), ULP_LOG), steps_up(log(a.hi), ULP_LOG)};
}
__device__ IV sqrt_checked(IV a, bool* bad) {
  if (a.lo < 0.0 || isnan(a.lo) || isnan(a.hi)) { *bad = true; return {0.0, 0.0}; }
  double lo = nd(sqrt(a.lo));
  if (lo < 0.0) lo = 0.0;
  return {lo, nu(sqrt(a.hi))};
}
__device__ IV sincos_iv_dev(IV a, bool is_cos) {
  const int r_max = is_cos ? 0 : 1;
  const int r_min = is_cos ? 2 : 3;
  double m = fmax(fabs(a.lo), fabs(a.hi));
  if (!(m <= TRIG_ENVELOPE)) return {-1.0, 1.0};
  double q_lo = nd(fmin(a.lo / HALF_PI_HI, a.lo / HALF_PI_LO));
  double q_hi = nu(fmax(a.hi / HALF_PI_HI, a.hi / HALF_PI_LO));
  long long k_lo = (long long)floor(q_lo);
  long long k_hi = (long long)floor(q_hi);
  double v_lo = is_cos ? cos(a.lo) : sin(a.lo);
  double v_hi = is_cos ? cos(a.hi) : sin(a.hi);
  double lo_r = steps_dn(fmin(v_lo, v_hi), ULP_TRIG);
  if (lo_r < -1.0) lo_r = -1.0;
  double hi_r = steps_up(fmax(v_lo, v_hi), ULP_TRIG);
  if (hi_r > 1.0) hi_r = 1.0;
  if (((k_hi - (long long)r_min) >> 2) > ((k_lo - (long long)r_min) >> 2)) lo_r = -1.0;
  if (((k_hi - (long long)r_max) >> 2) > ((k_lo - (long long)r_max) >> 2)) hi_r = 1.0;
  return {lo_r, hi_r};
}
__device__ __forceinline__ IV tab_iv(const double* t, int i) {
  return {t[2 * i], t[2 * i + 1]};
}
__device__ IV lag_exp(IV tm, int order, const double* frec) {
  return iv_mul(iv_mul(tab_iv(frec, order), iv_pow_int(tm, order)), exp_iv_dev(tm));
}

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

__device__ IV lag_rec(IV c, IV u, int order, bool* bad) {
  IV rec_c = checked_rec(c, bad);
  return geometric_tail(c, rec_c, u, order, bad);
}
__device__ IV lag_trig(IV c, IV tm, int order, const double* frec, bool cosc) {
  IV prod = iv_pow_int(tm, order);
  IV j = iv_add(tm, c);
  int k4 = cosc ? (order + 1) % 4 : order % 4;
  IV jv;
  if (k4 == 0)      jv = sincos_iv_dev(j, false);
  else if (k4 == 1) jv = sincos_iv_dev(j, true);
  else if (k4 == 2) jv = iv_neg(sincos_iv_dev(j, false));
  else              jv = iv_neg(sincos_iv_dev(j, true));
  return iv_mul(iv_mul(tab_iv(frec, order), prod), jv);
}
__device__ IV lag_log(IV tm, int order, bool* bad) {
  IV one = {1.0, 1.0};
  IV r = checked_rec(iv_add(tm, one), bad);
  r = iv_mul(r, tm);
  r = iv_pow_int(r, order);
  r = div_int(r, order);
  if ((order + 1) % 2 == 1) r = iv_neg(r);
  return r;
}
__device__ IV lag_sqrt(IV tm, int order, const double* frec, const double* dfact,
                       bool* bad) {
  IV one = {1.0, 1.0};
  IV r = checked_rec(iv_add(tm, one), bad);
  r = sqrt_checked(r, bad);
  r = iv_mul(r, tm);
  r = div_int(r, 2);
  r = iv_pow_int(r, order);
  r = iv_mul(r, iv_mul(tab_iv(dfact, 2 * order - 3), tab_iv(frec, order)));
  if (order % 2 == 0) r = iv_neg(r);
  return r;
}

// ---- the valid-pass interpreter -------------------------------------------
// Shared layout: polys [n_bufs, stride, 2] | prod [s_prod, 2] |
// scalars [n_slots + NREG, 2] | bad flag (int).
__global__ void valid_tape_kernel(
    const int* __restrict__ opc, const int* __restrict__ io,   // [N], [N,12]
    const double* __restrict__ fo,                             // [N, 3]
    const int* __restrict__ idxp, const int* __restrict__ pia,
    const int* __restrict__ pjb, const int* __restrict__ soff,
    const double* __restrict__ facp, const double* __restrict__ tabp,
    const double* __restrict__ x,        // [B, n, Sx]
    const double* __restrict__ x_rem,    // [B, n, 2]
    double* __restrict__ cache,          // [B, C, 2]
    double* __restrict__ tails,          // [B, St, 2]
    double* __restrict__ out_c,          // [B, nc, Sfu, 2] (pre-zeroed)
    double* __restrict__ out_r,          // [B, nc, 2]
    unsigned char* __restrict__ bad,     // [B] in/out
    int N, long long B, int n, int Sx, int C, int St,
    int stride, int n_bufs, int s_prod, int n_scal, int nc, int Sfu,
    double cutoff) {
  extern __shared__ double sh[];
  double* polys = sh;                                  // n_bufs*stride*2
  double* prod = polys + (size_t)n_bufs * stride * 2;  // s_prod*2
  double* scal = prod + (size_t)s_prod * 2;            // n_scal*2
  __shared__ int lane_bad;
  const long long b = blockIdx.x;
  if (b >= B) return;
  const int tid = threadIdx.x, TB = blockDim.x;
  if (tid == 0) lane_bad = bad[b] ? 1 : 0;
  __syncthreads();

  #define POLY(bu, j) (polys + ((size_t)(bu) * stride + (j)) * 2)
  #define SC(r) (scal + (size_t)(r) * 2)
  #define LDIV(p) IV{(p)[0], (p)[1]}
  // Single-evaluation store: the macro argument is an EXPRESSION over the
// destination in every in-place op (res = res * c), so it must be evaluated
// exactly once BEFORE either endpoint store (double evaluation re-read the
// half-written destination — caught by the differential suite as a squared
// upper endpoint).
#define STIV(p, v) { IV _sv = (v); (p)[0] = _sv.lo; (p)[1] = _sv.hi; }

  for (int i = 0; i < N; ++i) {
    const int* a = io + (size_t)i * 12;
    const double* f = fo + (size_t)i * 3;
    switch (opc[i]) {
      case 1: {  // VAR dbuf, var, goff, Svar, keep, foff, slot, sslot
        const double* xr = x + ((b * n + a[1]) * (size_t)Sx);
        for (int j = tid; j < a[4]; j += TB) {
          double v = xr[idxp[a[2] + j]];
          POLY(a[0], j)[0] = v; POLY(a[0], j)[1] = v;
        }
        if (tid == 0) {
          IV tail = {0.0, 0.0};
          for (int j = a[4]; j < a[3]; ++j) {
            double v = xr[idxp[a[2] + j]];
            tail = iv_add(tail, iv_mul({v, v}, tab_iv(facp, a[5] + (j - a[4]))));
          }
          IV xrm = {x_rem[(b * n + a[1]) * 2], x_rem[(b * n + a[1]) * 2 + 1]};
          STIV(SC(a[6]), iv_add(xrm, tail));
          tails[(b * St + a[7]) * 2] = tail.lo;
          tails[(b * St + a[7]) * 2 + 1] = tail.hi;
        }
        break;
      }
      case 2:  // CONST dbuf, slot, cbase ; f: c, crl, crh
        if (tid == 0) {
          POLY(a[0], 0)[0] = f[0]; POLY(a[0], 0)[1] = f[0];
          SC(a[1])[0] = f[1]; SC(a[1])[1] = f[2];
          cache[(b * C + a[2]) * 2] = f[1];
          cache[(b * C + a[2]) * 2 + 1] = f[2];
        }
        break;
      case 3:  // NEG dbuf, abuf, S, slotd, slota
        for (int j = tid; j < a[2]; j += TB)
          STIV(POLY(a[0], j), iv_neg(LDIV(POLY(a[1], j))));
        if (tid == 0 && a[3] >= 0) STIV(SC(a[3]), iv_neg(LDIV(SC(a[4]))));
        break;
      case 4: {  // ADDSUB dbuf,abuf,bbuf,iaoff,iboff,Su,slotd,slota,slotb,sub
        for (int j = tid; j < a[5]; j += TB) {
          int sa = idxp[a[3] + j], sb = idxp[a[4] + j];
          IV av = sa >= 0 ? LDIV(POLY(a[1], sa)) : IV{0.0, 0.0};
          IV bv = sb >= 0 ? LDIV(POLY(a[2], sb)) : IV{0.0, 0.0};
          STIV(POLY(a[0], j), a[9] ? iv_sub(av, bv) : iv_add(av, bv));
        }
        if (tid == 0 && a[6] >= 0) {
          IV rr = a[9] ? iv_sub(LDIV(SC(a[7])), LDIV(SC(a[8])))
                       : iv_add(LDIV(SC(a[7])), LDIV(SC(a[8])));
          STIV(SC(a[6]), rr);
        }
        break;
      }
      case 5:  // RANGE rdst, abuf(-1 = prod scratch), from, S, foff
        if (tid == 0) {
          IV acc = {0.0, 0.0};
          for (int j = a[2]; j < a[3]; ++j) {
            IV cv = (a[1] < 0) ? IV{prod[j * 2], prod[j * 2 + 1]}
                               : LDIV(POLY(a[1], j));
            acc = iv_add(acc, iv_mul(cv, tab_iv(facp, a[4] + (j - a[2]))));
          }
          STIV(SC(a[0]), acc);
        }
        break;
      case 6:  // MULP abuf, bbuf, poff, soff, Sout -> prod
        for (int m = tid; m < a[4]; m += TB) {
          IV acc = {0.0, 0.0};
          for (int p = soff[a[3] + m]; p < soff[a[3] + m + 1]; ++p)
            acc = iv_add(acc, iv_mul(LDIV(POLY(a[0], pia[a[2] + p])),
                                     LDIV(POLY(a[1], pjb[a[2] + p]))));
          prod[m * 2] = acc.lo; prod[m * 2 + 1] = acc.hi;
        }
        break;
      case 7: {  // CUT dbuf, K, foff, rdst  (src = prod)
        // kept in parallel; dropped-range sequentially on thread 0.
        for (int j = tid; j < a[1]; j += TB) {
          IV c = {prod[j * 2], prod[j * 2 + 1]};
          double w = nu(c.hi - c.lo);
          double mid = (c.lo + c.hi) * 0.5;
          IV miv = {nd(mid), nu(mid)};
          bool wide = w >= 1e-12;
          bool small = (!wide) && (miv.lo >= -cutoff) && (miv.hi <= cutoff);
          IV kept = wide ? miv : c;
          if (small) kept = {0.0, 0.0};
          STIV(POLY(a[0], j), kept);
        }
        if (tid == 0) {
          IV acc = {0.0, 0.0};
          for (int j = 0; j < a[1]; ++j) {
            IV c = {prod[j * 2], prod[j * 2 + 1]};
            double w = nu(c.hi - c.lo);
            double mid = (c.lo + c.hi) * 0.5;
            IV miv = {nd(mid), nu(mid)};
            bool wide = w >= 1e-12;
            bool small = (!wide) && (miv.lo >= -cutoff) && (miv.hi <= cutoff);
            IV to_rem = wide ? iv_sub(c, miv) : IV{0.0, 0.0};
            if (small) to_rem = c;
            acc = iv_add(acc, iv_mul(to_rem, tab_iv(facp, a[2] + j)));
          }
          STIV(SC(a[3]), acc);
        }
        break;
      }
      case 8: if (tid == 0) STIV(SC(a[0]), iv_mul(LDIV(SC(a[1])), LDIV(SC(a[2])))); break;
      case 9: if (tid == 0) STIV(SC(a[0]), iv_add(LDIV(SC(a[1])), LDIV(SC(a[2])))); break;
      case 10: if (tid == 0) STIV(SC(a[0]), iv_sub(LDIV(SC(a[1])), LDIV(SC(a[2])))); break;
      case 11: if (tid == 0) STIV(SC(a[0]), iv_neg(LDIV(SC(a[1])))); break;
      case 12: if (tid == 0) STIV(SC(a[0]), LDIV(SC(a[1]))); break;
      case 13: if (tid == 0) STIV(SC(a[0]), iv_mul(LDIV(SC(a[1])), tab_iv(tabp, a[2]))); break;
      case 14: if (tid == 0) STIV(SC(a[0]), iv_mul(LDIV(SC(a[1])), IV{f[0], f[1]})); break;
      case 15: if (tid == 0) { SC(a[0])[0] = 0.0; SC(a[0])[1] = 0.0; } break;
      case 16: if (tid == 0) {
          cache[(b * C + a[0]) * 2] = SC(a[1])[0];
          cache[(b * C + a[0]) * 2 + 1] = SC(a[1])[1];
        } break;
      case 17: if (tid == 0) STIV(SC(a[0]), exp_iv_dev(LDIV(SC(a[1])))); break;
      case 18: if (tid == 0) STIV(SC(a[0]), sincos_iv_dev(LDIV(SC(a[1])), false)); break;
      case 19: if (tid == 0) STIV(SC(a[0]), sincos_iv_dev(LDIV(SC(a[1])), true)); break;
      case 20: if (tid == 0) {
          bool bb = false; IV r = checked_rec(LDIV(SC(a[1])), &bb);
          STIV(SC(a[0]), r); if (bb) lane_bad = 1;
        } break;
      case 21: if (tid == 0) {
          bool bb = false; IV r = log_checked(LDIV(SC(a[1])), &bb);
          STIV(SC(a[0]), r); if (bb) lane_bad = 1;
        } break;
      case 22: if (tid == 0) {
          bool bb = false; IV r = sqrt_checked(LDIV(SC(a[1])), &bb);
          STIV(SC(a[0]), r); if (bb) lane_bad = 1;
        } break;
      case 23: if (tid == 0) {
          SC(a[0])[0] = SC(a[1])[0] * 2.0; SC(a[0])[1] = SC(a[1])[1] * 2.0;
        } break;
      case 24: if (tid == 0) STIV(SC(a[0]), lag_exp(LDIV(SC(a[1])), a[2], tabp + 2 * a[3])); break;
      case 25: if (tid == 0) {
          bool bb = false;
          IV r = lag_rec(LDIV(SC(a[1])), LDIV(SC(a[2])), a[3], &bb);
          STIV(SC(a[0]), r); if (bb) lane_bad = 1;
        } break;
      case 26: if (tid == 0) STIV(SC(a[0]), lag_trig(LDIV(SC(a[1])), LDIV(SC(a[2])), a[3], tabp + 2 * a[4], false)); break;
      case 27: if (tid == 0) STIV(SC(a[0]), lag_trig(LDIV(SC(a[1])), LDIV(SC(a[2])), a[3], tabp + 2 * a[4], true)); break;
      case 28: if (tid == 0) {
          bool bb = false; IV r = lag_log(LDIV(SC(a[1])), a[2], &bb);
          STIV(SC(a[0]), r); if (bb) lane_bad = 1;
        } break;
      case 29: if (tid == 0) {
          bool bb = false;
          IV r = lag_sqrt(LDIV(SC(a[1])), a[2], tabp + 2 * a[3], tabp + 2 * a[4], &bb);
          STIV(SC(a[0]), r); if (bb) lane_bad = 1;
        } break;
      case 30: if (tid == 0) { POLY(a[0], 0)[0] = 1.0; POLY(a[0], 0)[1] = 1.0; } break;
      case 32: if (tid == 0) {
          STIV(SC(a[0]), LDIV(POLY(a[1], 0)));
          POLY(a[1], 0)[0] = 0.0; POLY(a[1], 0)[1] = 0.0;
        } break;
      case 33: {  // BSC_REG dbuf, abuf, ra, S — reg value fixed BEFORE loop
        IV rv = LDIV(SC(a[2]));
        __syncthreads();
        for (int j = tid; j < a[3]; j += TB)
          STIV(POLY(a[0], j), iv_mul(LDIV(POLY(a[1], j)), rv));
        break;
      }
      case 34: {  // BSC_TAB dbuf, abuf, toff, S
        IV rv = tab_iv(tabp, a[2]);
        for (int j = tid; j < a[3]; j += TB)
          STIV(POLY(a[0], j), iv_mul(LDIV(POLY(a[1], j)), rv));
        break;
      }
      case 35: {  // BSC_IMM dbuf, abuf, S ; f0, f1
        IV rv = {f[0], f[1]};
        for (int j = tid; j < a[2]; j += TB)
          STIV(POLY(a[0], j), iv_mul(LDIV(POLY(a[1], j)), rv));
        break;
      }
      case 36: if (tid == 0) STIV(POLY(a[0], 0), iv_add(LDIV(POLY(a[0], 0)), IV{1.0, 1.0})); break;
      case 37: if (tid == 0) STIV(POLY(a[0], 0), iv_add(LDIV(POLY(a[0], 0)), LDIV(SC(a[1])))); break;
      case 38: if (tid == 0) STIV(POLY(a[0], 0), iv_sub(LDIV(POLY(a[0], 0)), IV{f[0], f[0]})); break;
      case 39:  // BUF_COPY dbuf, abuf, S
        for (int j = tid; j < a[2]; j += TB) {
          POLY(a[0], j)[0] = POLY(a[1], j)[0];
          POLY(a[0], j)[1] = POLY(a[1], j)[1];
        }
        break;
      case 40:  // BUF_NEG dbuf, abuf, S
        for (int j = tid; j < a[2]; j += TB)
          STIV(POLY(a[0], j), iv_neg(LDIV(POLY(a[1], j))));
        break;
      case 41: if (tid == 0) STIV(POLY(a[0], 0), LDIV(SC(a[1]))); break;
      case 43: {  // OUT_IV comp, abuf, emboff, S
        double* oc = out_c + ((b * nc + a[0]) * (size_t)Sfu) * 2;
        for (int j = tid; j < a[3]; j += TB) {
          int t = idxp[a[2] + j];
          oc[t * 2] = POLY(a[1], j)[0];
          oc[t * 2 + 1] = POLY(a[1], j)[1];
        }
        break;
      }
      case 44: if (tid == 0) {
          out_r[(b * nc + a[0]) * 2] = SC(a[1])[0];
          out_r[(b * nc + a[0]) * 2 + 1] = SC(a[1])[1];
        } break;
      default: break;
    }
    __syncthreads();
  }
  if (tid == 0) bad[b] = lane_bad ? 1 : 0;
}

// ---- the point-pass interpreter -------------------------------------------
// Shared layout: polys [n_bufs, stride] | prod [s_prod] | scalars [NREG].
__global__ void point_tape_kernel(
    const int* __restrict__ opc, const int* __restrict__ io,
    const double* __restrict__ fo,
    const int* __restrict__ idxp, const int* __restrict__ pia,
    const int* __restrict__ pjb, const int* __restrict__ soff,
    const double* __restrict__ x,     // [B, n, Sx]
    double* __restrict__ out,         // [B, nc, Sfu] (pre-zeroed)
    int N, long long B, int n, int Sx,
    int stride, int n_bufs, int s_prod, int n_scal, int nc, int Sfu) {
  extern __shared__ double sh[];
  double* polys = sh;
  double* prod = polys + (size_t)n_bufs * stride;
  double* scal = prod + (size_t)s_prod;
  const long long b = blockIdx.x;
  if (b >= B) return;
  const int tid = threadIdx.x, TB = blockDim.x;
  #define PPOLY(bu, j) polys[(size_t)(bu) * stride + (j)]
  for (int i = 0; i < N; ++i) {
    const int* a = io + (size_t)i * 12;
    const double* f = fo + (size_t)i * 3;
    switch (opc[i]) {
      case 1: {  // VAR dbuf, var, goff, keep
        const double* xr = x + ((b * n + a[1]) * (size_t)Sx);
        for (int j = tid; j < a[3]; j += TB) PPOLY(a[0], j) = xr[idxp[a[2] + j]];
        break;
      }
      case 2: if (tid == 0) PPOLY(a[0], 0) = f[0]; break;
      case 3:
        for (int j = tid; j < a[2]; j += TB) PPOLY(a[0], j) = -PPOLY(a[1], j);
        break;
      case 4:  // ADDSUB dbuf,abuf,bbuf,iaoff,iboff,Su,sub
        for (int j = tid; j < a[5]; j += TB) {
          int sa = idxp[a[3] + j], sb = idxp[a[4] + j];
          double av = sa >= 0 ? PPOLY(a[1], sa) : 0.0;
          double bv = sb >= 0 ? PPOLY(a[2], sb) : 0.0;
          PPOLY(a[0], j) = a[6] ? (av - bv) : (av + bv);
        }
        break;
      case 5:  // MULP abuf, bbuf, poff, soff, Sout -> prod (RN fma in order)
        for (int m = tid; m < a[4]; m += TB) {
          double acc = 0.0;
          for (int p = soff[a[3] + m]; p < soff[a[3] + m + 1]; ++p)
            acc = fma(PPOLY(a[0], pia[a[2] + p]), PPOLY(a[1], pjb[a[2] + p]), acc);
          prod[m] = acc;
        }
        break;
      case 6:  // CUT dbuf, K ; f0 = cutoff (|c| <= eps -> 0)
        for (int j = tid; j < a[1]; j += TB) {
          double v = prod[j];
          PPOLY(a[0], j) = (fabs(v) <= f[0]) ? 0.0 : v;
        }
        break;
      case 7:  // PREFIX dbuf, K (no cutoff)
        for (int j = tid; j < a[1]; j += TB) PPOLY(a[0], j) = prod[j];
        break;
      case 8: if (tid == 0) PPOLY(a[0], 0) = 1.0; break;
      case 10: if (tid == 0) { scal[a[0]] = PPOLY(a[1], 0); PPOLY(a[1], 0) = 0.0; } break;
      case 11: {  // SC_REG dbuf, abuf, ra, S
        double rv = scal[a[2]];
        __syncthreads();
        for (int j = tid; j < a[3]; j += TB) PPOLY(a[0], j) = PPOLY(a[1], j) * rv;
        break;
      }
      case 12:
        for (int j = tid; j < a[2]; j += TB) PPOLY(a[0], j) = PPOLY(a[1], j) * f[0];
        break;
      case 13:
        for (int j = tid; j < a[2]; j += TB) PPOLY(a[0], j) = PPOLY(a[1], j) / f[0];
        break;
      case 14: if (tid == 0) PPOLY(a[0], 0) += f[0]; break;
      case 15: if (tid == 0) PPOLY(a[0], 0) += scal[a[1]]; break;
      case 16: if (tid == 0) PPOLY(a[0], 0) = scal[a[1]]; break;
      case 17:
        for (int j = tid; j < a[2]; j += TB) PPOLY(a[0], j) = PPOLY(a[1], j);
        break;
      case 19: if (tid == 0) scal[a[0]] = exp(scal[a[1]]); break;
      case 20: if (tid == 0) scal[a[0]] = sin(scal[a[1]]); break;
      case 21: if (tid == 0) scal[a[0]] = cos(scal[a[1]]); break;
      case 22: if (tid == 0) scal[a[0]] = log(scal[a[1]]); break;
      case 23: if (tid == 0) scal[a[0]] = sqrt(scal[a[1]]); break;
      case 24: if (tid == 0) scal[a[0]] = 1.0 / scal[a[1]]; break;
      case 25: if (tid == 0) scal[a[0]] = scal[a[1]] / f[0]; break;
      case 26: if (tid == 0) scal[a[0]] = -scal[a[1]]; break;
      case 29: {  // P_DIVREG dbuf, abuf, ra, S  (x / reg, RN)
        double rv = scal[a[2]];
        __syncthreads();
        for (int j = tid; j < a[3]; j += TB) PPOLY(a[0], j) = PPOLY(a[1], j) / rv;
        break;
      }
      case 30: if (tid == 0) scal[a[0]] = scal[a[1]] * f[0]; break;  // PR_MUL_IMM
      case 27:  // CUTBUF dbuf, abuf, S ; f0 = cutoff
        for (int j = tid; j < a[2]; j += TB) {
          double v = PPOLY(a[1], j);
          PPOLY(a[0], j) = (fabs(v) <= f[0]) ? 0.0 : v;
        }
        break;
      case 28: {  // OUT comp, abuf, emboff, S
        double* orow = out + (b * nc + a[0]) * (size_t)Sfu;
        for (int j = tid; j < a[3]; j += TB) orow[idxp[a[2] + j]] = PPOLY(a[1], j);
        break;
      }
      default: break;
    }
    __syncthreads();
  }
}

}  // namespace

std::vector<torch::Tensor> valid_tape(
    torch::Tensor opc, torch::Tensor io, torch::Tensor fo, torch::Tensor idxp,
    torch::Tensor pia, torch::Tensor pjb, torch::Tensor soff,
    torch::Tensor facp, torch::Tensor tabp,
    torch::Tensor x, torch::Tensor x_rem, torch::Tensor bad,
    int64_t C, int64_t St, int64_t stride, int64_t n_bufs, int64_t s_prod,
    int64_t n_scal, int64_t nc, int64_t Sfu, double cutoff) {
  const long long B = x.size(0);
  const int n = (int)x.size(1), Sx = (int)x.size(2);
  auto cache = torch::zeros({(long long)B, C, 2}, x.options());
  auto tails = torch::zeros({(long long)B, St, 2}, x.options());
  auto out_c = torch::zeros({(long long)B, nc, Sfu, 2}, x.options());
  auto out_r = torch::zeros({(long long)B, nc, 2}, x.options());
  size_t shbytes = ((size_t)n_bufs * stride * 2 + (size_t)s_prod * 2 +
                    (size_t)n_scal * 2) * sizeof(double);
  valid_tape_kernel<<<(int)B, 64, shbytes, at::cuda::getCurrentCUDAStream()>>>(
      opc.data_ptr<int>(), io.data_ptr<int>(), fo.data_ptr<double>(),
      idxp.data_ptr<int>(), pia.data_ptr<int>(), pjb.data_ptr<int>(),
      soff.data_ptr<int>(), facp.data_ptr<double>(), tabp.data_ptr<double>(),
      x.data_ptr<double>(), x_rem.data_ptr<double>(),
      cache.data_ptr<double>(), tails.data_ptr<double>(),
      out_c.data_ptr<double>(), out_r.data_ptr<double>(),
      bad.data_ptr<unsigned char>(),
      (int)opc.size(0), B, n, Sx, (int)C, (int)St, (int)stride, (int)n_bufs,
      (int)s_prod, (int)n_scal, (int)nc, (int)Sfu, cutoff);
  return {out_c, out_r, cache, tails};
}

torch::Tensor point_tape(
    torch::Tensor opc, torch::Tensor io, torch::Tensor fo, torch::Tensor idxp,
    torch::Tensor pia, torch::Tensor pjb, torch::Tensor soff, torch::Tensor x,
    int64_t stride, int64_t n_bufs, int64_t s_prod, int64_t n_scal,
    int64_t nc, int64_t Sfu) {
  const long long B = x.size(0);
  auto out = torch::zeros({(long long)B, nc, Sfu}, x.options());
  size_t shbytes = ((size_t)n_bufs * stride + (size_t)s_prod + (size_t)n_scal)
                   * sizeof(double);
  point_tape_kernel<<<(int)B, 64, shbytes, at::cuda::getCurrentCUDAStream()>>>(
      opc.data_ptr<int>(), io.data_ptr<int>(), fo.data_ptr<double>(),
      idxp.data_ptr<int>(), pia.data_ptr<int>(), pjb.data_ptr<int>(),
      soff.data_ptr<int>(), x.data_ptr<double>(), out.data_ptr<double>(),
      (int)opc.size(0), B, (int)x.size(1), (int)x.size(2), (int)stride,
      (int)n_bufs, (int)s_prod, (int)n_scal, (int)nc, (int)Sfu);
  return out;
}
"""

_VALID_CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> valid_tape(
    torch::Tensor opc, torch::Tensor io, torch::Tensor fo, torch::Tensor idxp,
    torch::Tensor pia, torch::Tensor pjb, torch::Tensor soff,
    torch::Tensor facp, torch::Tensor tabp,
    torch::Tensor x, torch::Tensor x_rem, torch::Tensor bad,
    int64_t C, int64_t St, int64_t stride, int64_t n_bufs, int64_t s_prod,
    int64_t n_scal, int64_t nc, int64_t Sfu, double cutoff);
torch::Tensor point_tape(
    torch::Tensor opc, torch::Tensor io, torch::Tensor fo, torch::Tensor idxp,
    torch::Tensor pia, torch::Tensor pjb, torch::Tensor soff, torch::Tensor x,
    int64_t stride, int64_t n_bufs, int64_t s_prod, int64_t n_scal,
    int64_t nc, int64_t Sfu);
"""

_vext = None
_vtried = False


def _vbuild() -> None:
    global _vext, _vtried
    if _vtried:
        return
    _vtried = True
    _vext = load_cuda_extension(
        'flowstar_recip_geom_valid_1f9efda6325c', _VALID_CPP, _VALID_CUDA,
        ["valid_tape", "point_tape"],
    )


def valid_available() -> bool:
    """True iff the valid/point-pass interpreter extension compiled."""
    _vbuild()
    return _vext is not None


# ---------------------------------------------------------------------------
# Phase 3b serializers: SpecTape -> micro-op streams
# ---------------------------------------------------------------------------

MAX_POLY_W = 192  # per-buffer slot capacity (shared-memory budget guard)
MAX_PROD_W = 1024
MAX_PAIRS = 65536
MAX_SHARED = 48 * 1024  # V100 default dynamic-shared ceiling (no opt-in attr)
NREG = 16  # scalar temp registers appended after the per-slot remainders


class _Pools:
    """Index/float pools shared by one serialized tape (deduped by key)."""

    def __init__(self):
        self.idx: list = []
        self.idx_n = 0
        self.pia: list = []
        self.pjb: list = []
        self.pairs_n = 0
        self.soff: list = []
        self.soff_n = 0
        self.fac: list = []
        self.fac_n = 0
        self.tab: list = []
        self.tab_n = 0
        self._memo: dict = {}

    def add_idx(self, key, arr) -> int:
        hit = self._memo.get(("i", key))
        if hit is not None:
            return hit
        import numpy as _np

        a = _np.ascontiguousarray(arr, dtype=_np.int32)
        off = self.idx_n
        self.idx.append(a)
        self.idx_n += a.shape[0]
        self._memo[("i", key)] = off
        return off

    def add_pairs(self, key, ia, jb, seg_off) -> tuple[int, int]:
        hit = self._memo.get(("p", key))
        if hit is not None:
            return hit
        import numpy as _np

        if ia.shape[0] > MAX_PAIRS:
            raise ValueError(f"pair table {ia.shape[0]} > {MAX_PAIRS}")
        po, so = self.pairs_n, self.soff_n
        self.pia.append(_np.ascontiguousarray(ia, dtype=_np.int32))
        self.pjb.append(_np.ascontiguousarray(jb, dtype=_np.int32))
        self.pairs_n += ia.shape[0]
        # seg_offsets are stored relative to THIS pair block's base.
        self.soff.append(_np.ascontiguousarray(seg_off, dtype=_np.int32))
        self.soff_n += seg_off.shape[0]
        self._memo[("p", key)] = (po, so)
        return po, so

    def add_fac(self, key, rows) -> int:
        hit = self._memo.get(("f", key))
        if hit is not None:
            return hit
        import numpy as _np

        r = _np.ascontiguousarray(rows, dtype=_np.float64).reshape(-1, 2)
        off = self.fac_n
        self.fac.append(r)
        self.fac_n += r.shape[0]
        self._memo[("f", key)] = off
        return off

    def add_tab(self, key, rows) -> int:
        hit = self._memo.get(("t", key))
        if hit is not None:
            return hit
        import numpy as _np

        r = _np.ascontiguousarray(rows, dtype=_np.float64).reshape(-1, 2)
        off = self.tab_n
        self.tab.append(r)
        self.tab_n += r.shape[0]
        self._memo[("t", key)] = off
        return off


class _Asm:
    def __init__(self):
        self.opc: list = []
        self.ints: list = []
        self.flts: list = []

    def emit(self, opc: int, ints=(), flts=()) -> None:
        self.opc.append(opc)
        row = list(ints) + [0] * (12 - len(ints))
        self.ints.append(row)
        frow = list(flts) + [0.0] * (3 - len(flts))
        self.flts.append(frow)


class _BufAlloc:
    def __init__(self):
        self.free: list = []
        self.next = 0
        self.peak = 0

    def get(self) -> int:
        if self.free:
            return self.free.pop()
        b = self.next
        self.next += 1
        self.peak = self.next
        return b

    def rel(self, b: int) -> None:
        self.free.append(b)


@dataclass
class ValidTape:
    """Serialized valid-pass micro-op stream + pools (device tensors)."""

    opc: torch.Tensor
    io: torch.Tensor
    fo: torch.Tensor
    idxp: torch.Tensor
    pia: torch.Tensor
    pjb: torch.Tensor
    soff: torch.Tensor
    facp: torch.Tensor
    tabp: torch.Tensor
    n_cache: int
    n_strict: int
    stride: int
    n_bufs: int
    s_prod: int
    n_scal: int
    nc: int
    Sfu: int
    cutoff: float
    n_slots: int
    device: str


@dataclass
class PointTape:
    opc: torch.Tensor
    io: torch.Tensor
    fo: torch.Tensor
    idxp: torch.Tensor
    pia: torch.Tensor
    pjb: torch.Tensor
    soff: torch.Tensor
    stride: int
    n_bufs: int
    s_prod: int
    n_scal: int
    nc: int
    Sfu: int
    device: str


def _inv_maps(su_size: int, ea, eb):
    import numpy as np

    inva = np.full(su_size, -1, dtype=np.int32)
    invb = np.full(su_size, -1, dtype=np.int32)
    ea_np = ea.cpu().numpy()
    eb_np = eb.cpu().numpy()
    inva[ea_np] = np.arange(len(ea_np), dtype=np.int32)
    invb[eb_np] = np.arange(len(eb_np), dtype=np.int32)
    return inva, invb


def _last_reads(spec) -> dict:
    last: dict[int, int] = {}
    for pos, si in enumerate(spec.instrs):
        for s in (si.a, si.b):
            if s >= 0:
                last[s] = pos
    for s in spec.out_slots:
        last[s] = len(spec.instrs)  # kept alive until the output stage
    return last


def _finalize_valid(asmb, pools, code, spec, alloc, s_prod, device, cutoff, n) -> ValidTape:
    import numpy as np

    def cat_i(parts):
        return (np.concatenate(parts) if parts else np.zeros(1, dtype=np.int32))

    def cat_f(parts):
        return (np.concatenate(parts).reshape(-1) if parts
                else np.zeros(2, dtype=np.float64))

    stride = max(1, _finalize_valid._stride)
    if stride > MAX_POLY_W:
        raise ValueError(f"poly width {stride} > {MAX_POLY_W}: fall back")
    if s_prod > MAX_PROD_W:
        raise ValueError(f"product width {s_prod} > {MAX_PROD_W}: fall back")
    n_scal = code.n_slots + NREG
    shared = (alloc.peak * stride * 2 + s_prod * 2 + n_scal * 2) * 8
    if shared > MAX_SHARED:
        raise ValueError(f"shared budget {shared} > {MAX_SHARED}: fall back")
    dev = device
    return ValidTape(
        opc=torch.tensor(asmb.opc, dtype=torch.int32, device=dev),
        io=torch.tensor(np.array(asmb.ints, dtype=np.int32), device=dev),
        fo=torch.tensor(np.array(asmb.flts, dtype=np.float64), device=dev),
        idxp=torch.from_numpy(cat_i(pools.idx)).to(dev),
        pia=torch.from_numpy(cat_i(pools.pia)).to(dev),
        pjb=torch.from_numpy(cat_i(pools.pjb)).to(dev),
        soff=torch.from_numpy(cat_i(pools.soff)).to(dev),
        facp=torch.from_numpy(cat_f(pools.fac)).to(dev),
        tabp=torch.from_numpy(cat_f(pools.tab)).to(dev),
        n_cache=code.n_cache,
        n_strict=max(code.n_strict, 1),
        stride=stride,
        n_bufs=max(alloc.peak, 1),
        s_prod=max(s_prod, 1),
        n_scal=n_scal,
        nc=len(spec.out_slots),
        Sfu=spec.sup_out_union.size,
        cutoff=cutoff,
        n_slots=code.n_slots,
        device=dev,
    )


def serialize_valid_tape(spec, code, eng, cutoff: float, device: str) -> ValidTape:
    """Lower a SpecTape's VALID pass to the micro-op interpreter encoding.

    Raises ValueError on any capacity-guard violation (caller falls back to
    the eager/graphed exec_valid_s). Cache layout, op order and rounding mix
    replicate exec_valid_s exactly (module comment: parity policy).
    """
    from .elementary import elem_tables

    V = _VOP
    asmb, pools, alloc = _Asm(), _Pools(), _BufAlloc()
    k = spec.k
    tabs = elem_tables(max(k, 1), "cpu")
    frec_off = pools.add_tab("frec", tabs.frec_iv.numpy())
    dfact_off = pools.add_tab("dfact", tabs.dfact_iv.numpy())
    irec_off = pools.add_tab("irec", tabs.int_rec_iv.numpy())
    stride_seen = [1]

    def fac_off(sup, frm=0):
        rows = eng.factor(sup).cpu().numpy()
        return pools.add_fac((id(eng.factor(sup)), frm), rows[frm:])

    slotmap: dict[int, tuple] = {}  # tape slot -> (buf, S, sup)
    last = _last_reads(spec)
    n = len(spec.out_slots)

    def track(S):
        stride_seen[0] = max(stride_seen[0], S)

    def newbuf(S):
        track(S)
        return alloc.get()

    def release_operands(pos, si):
        for s in (si.a, si.b):
            if s >= 0 and last.get(s) == pos and s in slotmap:
                alloc.rel(slotmap[s][0])

    # scalar temp register ids (after the n_slots remainder registers)
    T = [code.n_slots + i for i in range(NREG)]
    s_prod_max = [1]

    def emit_mul_valid(abuf, Sa, sup_a, a_rem, fbuf, Sf, b_rem, pb, keep,
                      cache_at, out_rem_reg, range_b_reg):
        """tm_mul_valid_s lowering. Returns (out_buf, keep)."""
        rp1, t1, t2, t3 = T[10], T[11], T[12], T[13]
        asmb.emit(V["RANGE"], (rp1, abuf, 0, Sa, fac_off(sup_a)))
        S2 = pb.sup_out.size
        s_prod_max[0] = max(s_prod_max[0], S2)
        ia = pb.pair_ia.cpu().numpy()
        jb = pb.pair_jb.cpu().numpy()
        so = pb.seg_offsets.cpu().numpy()
        po, soo = pools.add_pairs(id(pb), ia, jb, so)
        asmb.emit(V["MULP"], (abuf, fbuf, po, soo, S2))
        asmb.emit(V["R_MUL"], (t1, a_rem, b_rem))
        asmb.emit(V["R_MUL"], (t2, range_b_reg, a_rem))
        asmb.emit(V["R_ADD"], (t1, t1, t2))
        asmb.emit(V["R_MUL"], (t2, rp1, b_rem))
        asmb.emit(V["R_ADD"], (t1, t1, t2))
        if keep < S2:
            asmb.emit(V["RANGE"], (t2, -1, keep, S2, fac_off(pb.sup_out, keep)))
        else:
            asmb.emit(V["R_ZERO"], (t2,))
        out = newbuf(keep)
        asmb.emit(V["CUT"], (out, keep, fac_off(pb.sup_out), t3))
        asmb.emit(V["R_ADD"], (t2, t2, t3))
        asmb.emit(V["R_ADD"], (out_rem_reg, t1, t2))
        asmb.emit(V["CACHE_WR"], (cache_at, rp1))
        asmb.emit(V["CACHE_WR"], (cache_at + 1, range_b_reg))
        asmb.emit(V["CACHE_WR"], (cache_at + 2, t2))
        return out, keep

    for pos, si in enumerate(spec.instrs):
        if si.op == "var":
            sv = si.sup_in
            buf = newbuf(si.keep_len)
            goff = pools.add_idx(("g", pos), si.gather.cpu().numpy())
            foff = fac_off(sv, si.keep_len) if si.keep_len < sv.size else 0
            asmb.emit(V["VAR"], (buf, si.var, goff, sv.size, si.keep_len,
                                 foff, si.dst, si.strict_slot))
            slotmap[si.dst] = (buf, si.keep_len, si.sup_out)
        elif si.op == "const":
            buf = newbuf(1)
            asmb.emit(V["CONST"], (buf, si.dst, si.cache_base),
                      (si.const, si.crem_lo, si.crem_hi))
            slotmap[si.dst] = (buf, 1, si.sup_out)
        elif si.op == "neg":
            ab, Sa, sup = slotmap[si.a]
            buf = newbuf(Sa)
            asmb.emit(V["NEG"], (buf, ab, Sa, si.dst, si.a))
            slotmap[si.dst] = (buf, Sa, sup)
            release_operands(pos, si)
        elif si.op in ("add", "sub"):
            su, ea, eb = si.emb
            ab, Sa, _ = slotmap[si.a]
            bb, Sb, _ = slotmap[si.b]
            inva, invb = _inv_maps(su.size, ea, eb)
            iao = pools.add_idx(("ia", pos), inva)
            ibo = pools.add_idx(("ib", pos), invb)
            buf = newbuf(su.size)
            asmb.emit(V["ADDSUB"], (buf, ab, bb, iao, ibo, su.size,
                                    si.dst, si.a, si.b, 1 if si.op == "sub" else 0))
            slotmap[si.dst] = (buf, su.size, su)
            release_operands(pos, si)
        elif si.op == "mul":
            ab, Sa, sup_a = slotmap[si.a]
            bb, Sb, sup_b = slotmap[si.b]
            rb = T[14]
            asmb.emit(V["RANGE"], (rb, bb, 0, Sb, fac_off(sup_b)))
            out, keep = emit_mul_valid(ab, Sa, sup_a, si.a, bb, Sb, si.b,
                                       si.pair, si.keep_len, si.cache_base,
                                       si.dst, rb)
            slotmap[si.dst] = (out, keep, si.sup_out)
            release_operands(pos, si)
        elif si.op == "pow":
            ab, Sa, sup_a = slotmap[si.a]
            if si.power == 0:
                buf = newbuf(1)
                asmb.emit(V["BUF_ONE"], (buf,))
                asmb.emit(V["R_ZERO"], (si.dst,))
                slotmap[si.dst] = (buf, 1, si.sup_out)
            else:
                res = newbuf(Sa)
                tmp = newbuf(Sa)
                asmb.emit(V["BUF_COPY"], (res, ab, Sa))
                asmb.emit(V["BUF_COPY"], (tmp, ab, Sa))
                resr, tmpr = T[8], T[9]
                asmb.emit(V["R_COPY"], (resr, si.a))
                asmb.emit(V["R_COPY"], (tmpr, si.a))
                cursor = si.cache_base
                rS, tS = Sa, Sa
                rsup, tsup = sup_a, sup_a
                rb = T[14]
                for entry in si.chain:
                    if entry[0] == "r":
                        _t, pb, keep, sup_res, sup_tmp = entry
                        asmb.emit(V["RANGE"], (rb, tmp, 0, tS, fac_off(sup_tmp)))
                        nres, keep = emit_mul_valid(res, rS, sup_res, resr,
                                                    tmp, tS, tmpr, pb, keep,
                                                    cursor, resr, rb)
                        alloc.rel(res)
                        res, rS = nres, keep
                        rsup, _ = _spec_kept(pb, spec.k)
                    else:
                        _t, pb, keep, sup_tmp = entry
                        asmb.emit(V["RANGE"], (rb, tmp, 0, tS, fac_off(sup_tmp)))
                        ntmp, keep = emit_mul_valid(tmp, tS, sup_tmp, tmpr,
                                                    tmp, tS, tmpr, pb, keep,
                                                    cursor, tmpr, rb)
                        alloc.rel(tmp)
                        tmp, tS = ntmp, keep
                        tsup, _ = _spec_kept(pb, spec.k)
                    cursor += 3
                alloc.rel(tmp)
                asmb.emit(V["R_COPY"], (si.dst, resr))
                slotmap[si.dst] = (res, rS, si.sup_out)
            release_operands(pos, si)
        elif si.op in ("exp", "sin", "cos", "log", "sqrt", "div"):
            _emit_series_valid(asmb, pools, alloc, V, si, spec, code, eng,
                               slotmap, T, emit_mul_valid, fac_off, newbuf,
                               frec_off, dfact_off, irec_off, tabs, s_prod_max)
            release_operands(pos, si)
        else:  # pragma: no cover  # unreachable: specialize emits only known ops
            raise AssertionError(si.op)

    for i, (slot, emb) in enumerate(zip(spec.out_slots, spec.out_embeds)):
        buf, S, _ = slotmap[slot]
        eo = pools.add_idx(("oe", i), emb.cpu().numpy())
        asmb.emit(V["OUT_IV"], (i, buf, eo, S))
        asmb.emit(V["OUT_REM"], (i, slot))

    _finalize_valid._stride = stride_seen[0]
    return _finalize_valid(asmb, pools, code, spec, alloc, s_prod_max[0],
                           eng.device, cutoff, n)


def _spec_kept(pb, k):
    from . import support as sp

    keep = pb.sup_out.keep_len(k)
    return sp.make_support(pb.sup_out.n, pb.sup_out.k, pb.sup_out.spatial,
                           pb.sup_out.ids[:keep]), keep


def _emit_series_valid(asmb, pools, alloc, V, si, spec, code, eng, slotmap, T,
                       emit_mul_valid, fac_off, newbuf, frec_off, dfact_off,
                       irec_off, tabs, s_prod_max):
    """Lower one elementary/div instruction, mirroring elementary.*_valid_g
    op-for-op (same cache cursor discipline, same rounding mix)."""
    import math as _m
    from fractions import Fraction as _F

    k = spec.k
    arg = si.b if si.op == "div" else si.a
    ab, Sa, _sup_arg = slotmap[arg]
    Sf = si.sup_f.size
    base = si.cache_base
    C0, CMAIN, FRANGE, RESR, TMR, LAG, T6, AUX = (T[0], T[1], T[2], T[3], T[4],
                                                  T[5], T[6], T[7])
    fb = newbuf(Sf)
    asmb.emit(V["BUF_COPY"], (fb, ab, Sf))
    asmb.emit(V["SPLIT"], (C0, fb))

    def mul_step(res, curS, sup_cur, rem_reg, pb, keep, cursor, f_rem_reg):
        nres, keep = emit_mul_valid(res, curS, sup_cur, rem_reg, fb2[0],
                                    Sf, f_rem_reg, pb, keep, cursor, rem_reg,
                                    FRANGE)
        alloc.rel(res)
        return nres, keep

    fb2 = [fb]  # the F buffer the chain multiplies against (maybe rescaled)

    if si.op == "exp":
        asmb.emit(V["R_EXP"], (CMAIN, C0))
        asmb.emit(V["CACHE_WR"], (base, CMAIN))
        asmb.emit(V["RANGE"], (FRANGE, fb, 0, Sf, fac_off(si.sup_f)))
        res = newbuf(1)
        asmb.emit(V["BUF_ONE"], (res,))
        asmb.emit(V["R_ZERO"], (RESR,))
        curS, cursor = 1, base + 1
        for idx, (pb, keep, sup_cur) in enumerate(si.chain):
            i = k - idx
            asmb.emit(V["BSC_TAB"], (res, res, irec_off + i, curS))
            asmb.emit(V["R_MULT"], (RESR, RESR, irec_off + i))
            res, curS = mul_step(res, curS, sup_cur, RESR, pb, keep, cursor, arg)
            cursor += 3
            asmb.emit(V["S0_ADD1"], (res,))
        asmb.emit(V["BSC_REG"], (res, res, CMAIN, curS))
        asmb.emit(V["R_MUL"], (RESR, RESR, CMAIN))
        asmb.emit(V["CACHE_WR"], (cursor, FRANGE))
        asmb.emit(V["R_ADD"], (TMR, FRANGE, arg))
        asmb.emit(V["LAG_EXP"], (LAG, TMR, k + 1, frec_off))
        asmb.emit(V["R_MUL"], (T6, CMAIN, LAG))
        asmb.emit(V["R_ADD"], (RESR, RESR, T6))
        asmb.emit(V["R_COPY"], (si.dst, RESR))
        alloc.rel(fb)
        slotmap[si.dst] = (res, curS, si.sup_out)
        return

    if si.op in ("sin", "cos"):
        cosc_cycle = si.op == "cos"
        asmb.emit(V["CACHE_WR"], (base, C0))
        # Wheel registers must survive ALL k iterations — keep them clear of
        # every scratch this loop writes (T[11] is emit_mul_valid-internal and
        # dead between calls, so the powr*tmp accumap scratch uses it).
        SINC, COSC, MSIN, MCOS, TMP, POWR = T[1], T[6], T[7], T[8], T[15], T[9]
        SCR = T[11]
        asmb.emit(V["R_SIN"], (SINC, C0))
        asmb.emit(V["R_COS"], (COSC, C0))
        asmb.emit(V["R_NEG"], (MSIN, SINC))
        asmb.emit(V["R_NEG"], (MCOS, COSC))
        wheel = ({1: MSIN, 2: MCOS, 3: SINC, 0: COSC} if cosc_cycle
                 else {1: COSC, 2: MSIN, 3: MCOS, 0: SINC})
        d0 = COSC if cosc_cycle else SINC
        asmb.emit(V["RANGE"], (FRANGE, fb, 0, Sf, fac_off(si.sup_f)))
        res = newbuf(1)
        asmb.emit(V["SET0R"], (res, d0))
        asmb.emit(V["R_ZERO"], (RESR,))
        pow_b = newbuf(1)
        asmb.emit(V["BUF_ONE"], (pow_b,))
        asmb.emit(V["R_ZERO"], (POWR,))
        curS, powS, cursor = 1, 1, base + 1
        for i in range(1, k + 1):
            pb, keep, sup_cur = si.chain[i - 1]
            pow_b, powS = mul_step(pow_b, powS, sup_cur, POWR, pb, keep, cursor, arg)
            cursor += 3
            asmb.emit(V["R_MULT"], (TMP, wheel[i % 4], frec_off + i))
            asmb.emit(V["CACHE_WR"], (cursor, TMP))
            cursor += 1
            term = newbuf(powS)
            asmb.emit(V["BSC_REG"], (term, pow_b, TMP, powS))
            su, er, ep = si.acc[i - 1]
            inva, invb = _inv_maps(su.size, er, ep)
            iao = pools.add_idx(("ta", id(si), i), inva)
            ibo = pools.add_idx(("tb", id(si), i), invb)
            nres = newbuf(su.size)
            asmb.emit(V["ADDSUB"], (nres, res, term, iao, ibo, su.size,
                                    -1, 0, 0, 0))
            alloc.rel(res)
            alloc.rel(term)
            res, curS = nres, su.size
            asmb.emit(V["R_MUL"], (SCR, POWR, TMP))
            asmb.emit(V["R_ADD"], (RESR, RESR, SCR))
        asmb.emit(V["CACHE_WR"], (cursor, FRANGE))
        asmb.emit(V["R_ADD"], (TMR, FRANGE, arg))
        asmb.emit(V["LAG_COS" if cosc_cycle else "LAG_SIN"],
                  (LAG, C0, TMR, k + 1, frec_off))
        asmb.emit(V["R_ADD"], (RESR, RESR, LAG))
        asmb.emit(V["R_COPY"], (si.dst, RESR))
        alloc.rel(fb)
        alloc.rel(pow_b)
        slotmap[si.dst] = (res, curS, si.sup_out)
        return

    if si.op == "log":
        asmb.emit(V["CACHE_WR"], (base, C0))
        asmb.emit(V["R_LOGCHK"], (CMAIN, C0))       # log_c (bad on c <= 0)
        asmb.emit(V["R_RECCHK"], (AUX, C0))         # rec_c (bad on 0 in c)
        f2 = newbuf(Sf)
        asmb.emit(V["BSC_REG"], (f2, fb, AUX, Sf))  # tmf_c = tm_f * rec_c
        alloc.rel(fb)
        fb2[0] = f2
        T2REM = T[8]
        asmb.emit(V["R_MUL"], (T2REM, arg, AUX))    # tmf_c_rem
        asmb.emit(V["RANGE"], (FRANGE, f2, 0, Sf, fac_off(si.sup_f)))
        res = newbuf(Sf)
        asmb.emit(V["BUF_COPY"], (res, f2, Sf))     # seed_f()
        asmb.emit(V["BSC_TAB"], (res, res, irec_off + k, Sf))
        asmb.emit(V["R_MULT"], (RESR, T2REM, irec_off + k))
        curS, cursor = Sf, base + 1
        for idx, (pb, keep, sup_cur) in enumerate(si.chain):
            i = k - 1 - idx
            asmb.emit(V["S0_SUBI"], (res,), (1.0 / i,))
            nb = newbuf(curS)
            asmb.emit(V["BUF_NEG"], (nb, res, curS))
            alloc.rel(res)
            res = nb
            asmb.emit(V["R_NEG"], (RESR, RESR))
            res, curS = mul_step(res, curS, sup_cur, RESR, pb, keep, cursor, T2REM)
            cursor += 3
        asmb.emit(V["S0_ADDR"], (res, CMAIN))
        asmb.emit(V["CACHE_WR"], (cursor, FRANGE))
        asmb.emit(V["R_ADD"], (TMR, FRANGE, T2REM))
        asmb.emit(V["LAG_LOG"], (LAG, TMR, k + 1))
        asmb.emit(V["R_ADD"], (RESR, RESR, LAG))
        asmb.emit(V["R_COPY"], (si.dst, RESR))
        alloc.rel(f2)
        slotmap[si.dst] = (res, curS, si.sup_out)
        return

    if si.op == "sqrt":
        asmb.emit(V["CACHE_WR"], (base, C0))
        asmb.emit(V["R_SQRTCHK"], (CMAIN, C0))      # sqrt_c
        TWOC, T2REM = T[9], T[8]
        asmb.emit(V["R_DBL"], (TWOC, C0))
        asmb.emit(V["R_RECCHK"], (AUX, TWOC))       # rec_2c
        f2 = newbuf(Sf)
        asmb.emit(V["BSC_REG"], (f2, fb, AUX, Sf))  # tmf_2c
        alloc.rel(fb)
        fb2[0] = f2
        asmb.emit(V["R_MUL"], (T2REM, arg, AUX))    # tmf_2c_rem
        asmb.emit(V["RANGE"], (FRANGE, f2, 0, Sf, fac_off(si.sup_f)))
        res = newbuf(Sf)
        asmb.emit(V["BUF_COPY"], (res, f2, Sf))
        asmb.emit(V["R_COPY"], (RESR, T2REM))
        curS, cursor = Sf, base + 1
        i, j = k, 2 * k - 3
        for pb, keep, sup_cur in si.chain:
            q = j / -i
            if _F(q) == _F(j, -i):
                pair = (q, q)
            else:
                pair = (_m.nextafter(q, -_m.inf), _m.nextafter(q, _m.inf))
            asmb.emit(V["BSC_IMM"], (res, res, curS), pair)
            asmb.emit(V["R_MULI"], (RESR, RESR), pair)
            asmb.emit(V["S0_ADD1"], (res,))
            res, curS = mul_step(res, curS, sup_cur, RESR, pb, keep, cursor, T2REM)
            cursor += 3
            i, j = i - 1, j - 2
        asmb.emit(V["S0_ADD1"], (res,))
        asmb.emit(V["BSC_REG"], (res, res, CMAIN, curS))
        asmb.emit(V["R_MUL"], (RESR, RESR, CMAIN))
        asmb.emit(V["R_DBL"], (T6, FRANGE))
        asmb.emit(V["CACHE_WR"], (cursor, T6))
        asmb.emit(V["R_DBL"], (TMR, T2REM))
        asmb.emit(V["R_ADD"], (TMR, T6, TMR))
        asmb.emit(V["LAG_SQRT"], (LAG, TMR, k + 1, frec_off, dfact_off))
        asmb.emit(V["R_MUL"], (T6, LAG, CMAIN))
        asmb.emit(V["R_ADD"], (RESR, RESR, T6))
        asmb.emit(V["R_COPY"], (si.dst, RESR))
        alloc.rel(f2)
        slotmap[si.dst] = (res, curS, si.sup_out)
        return

    # div: rec series on the denominator, then the pair2 multiply.
    asmb.emit(V["CACHE_WR"], (base, C0))
    asmb.emit(V["R_RECCHK"], (CMAIN, C0))           # rec_c
    asmb.emit(V["CACHE_WR"], (base + 1, CMAIN))
    T2REM = T[8]
    f2 = newbuf(Sf)
    asmb.emit(V["BSC_REG"], (f2, fb, CMAIN, Sf))    # tmf_c
    alloc.rel(fb)
    fb2[0] = f2
    asmb.emit(V["R_MUL"], (T2REM, arg, CMAIN))      # tmf_c_rem
    asmb.emit(V["RANGE"], (FRANGE, f2, 0, Sf, fac_off(si.sup_f)))
    res = newbuf(1)
    asmb.emit(V["BUF_ONE"], (res,))
    asmb.emit(V["R_ZERO"], (RESR,))
    curS, cursor = 1, base + 2
    for pb, keep, sup_cur in si.chain:
        nb = newbuf(curS)
        asmb.emit(V["BUF_NEG"], (nb, res, curS))
        alloc.rel(res)
        res = nb
        asmb.emit(V["R_NEG"], (RESR, RESR))
        res, curS = mul_step(res, curS, sup_cur, RESR, pb, keep, cursor, T2REM)
        cursor += 3
        asmb.emit(V["S0_ADD1"], (res,))
    asmb.emit(V["BSC_REG"], (res, res, CMAIN, curS))
    asmb.emit(V["R_MUL"], (RESR, RESR, CMAIN))
    asmb.emit(V["CACHE_WR"], (cursor, FRANGE))
    asmb.emit(V["R_ADD"], (TMR, FRANGE, T2REM))
    asmb.emit(V["LAG_REC"], (LAG, C0, TMR, k + 1))
    asmb.emit(V["R_ADD"], (RESR, RESR, LAG))
    alloc.rel(f2)
    # outer multiply: numerator x rec_result at cache_base + 3k + 3.
    pb2, keep2, sup_a2, sup_rec = si.pair2
    nb_buf, Sn, _sup_n = slotmap[si.a]
    RB = T[14]
    asmb.emit(V["RANGE"], (RB, res, 0, curS, fac_off(sup_rec)))
    fb2[0] = res  # emit_mul_valid's F operand becomes the rec result
    nres, keep2b = emit_mul_valid(nb_buf, Sn, sup_a2, si.a, res, curS, RESR,
                                  pb2, keep2, base + 3 * k + 3, si.dst, RB)
    alloc.rel(res)
    slotmap[si.dst] = (nres, keep2b, si.sup_out)


def serialize_point_tape(spec, eng, cutoff: float, device: str) -> PointTape:
    """Lower a SpecTape's POINT pass (exec_point_s contract)."""
    import math as _m
    import numpy as np

    P = _POP
    asmb, pools, alloc = _Asm(), _Pools(), _BufAlloc()
    k = spec.k
    stride_seen = [1]
    s_prod_max = [1]
    slotmap: dict[int, tuple] = {}
    last = _last_reads(spec)
    C0, CM, T2, T3 = 0, 1, 2, 3

    def newbuf(S):
        stride_seen[0] = max(stride_seen[0], S)
        return alloc.get()

    def release_operands(pos, si):
        for s in (si.a, si.b):
            if s >= 0 and last.get(s) == pos and s in slotmap:
                alloc.rel(slotmap[s][0])

    def emit_pairs(pb):
        s_prod_max[0] = max(s_prod_max[0], pb.sup_out.size)
        return pools.add_pairs(
            id(pb), pb.pair_ia.cpu().numpy(), pb.pair_jb.cpu().numpy(),
            pb.seg_offsets.cpu().numpy())

    def ctx_mul(a_buf, Sa, pb, keep, cut=True):
        po, so = emit_pairs(pb)
        asmb.emit(P["MULP"], (a_buf, fbuf[0], po, so, pb.sup_out.size))
        out = newbuf(keep)
        if cut:
            asmb.emit(P["CUT"], (out, keep), (cutoff,))
        else:
            asmb.emit(P["PREFIX"], (out, keep))
        return out, keep

    fbuf = [0]

    for pos, si in enumerate(spec.instrs):
        if si.op == "var":
            buf = newbuf(si.keep_len)
            goff = pools.add_idx(("g", pos), si.gather.cpu().numpy()[: si.keep_len])
            asmb.emit(P["VAR"], (buf, si.var, goff, si.keep_len))
            slotmap[si.dst] = (buf, si.keep_len)
        elif si.op == "const":
            buf = newbuf(1)
            asmb.emit(P["CONST"], (buf,), (si.const,))
            slotmap[si.dst] = (buf, 1)
        elif si.op == "neg":
            ab, Sa = slotmap[si.a]
            buf = newbuf(Sa)
            asmb.emit(P["NEG"], (buf, ab, Sa))
            slotmap[si.dst] = (buf, Sa)
            release_operands(pos, si)
        elif si.op in ("add", "sub"):
            su, ea, eb = si.emb
            ab, Sa = slotmap[si.a]
            bb, Sb = slotmap[si.b]
            inva, invb = _inv_maps(su.size, ea, eb)
            iao = pools.add_idx(("pa", pos), inva)
            ibo = pools.add_idx(("pb", pos), invb)
            buf = newbuf(su.size)
            asmb.emit(P["ADDSUB"], (buf, ab, bb, iao, ibo, su.size,
                                    1 if si.op == "sub" else 0))
            slotmap[si.dst] = (buf, su.size)
            release_operands(pos, si)
        elif si.op == "mul":
            ab, Sa = slotmap[si.a]
            bb, Sb = slotmap[si.b]
            po, so = emit_pairs(si.pair)
            asmb.emit(P["MULP"], (ab, bb, po, so, si.pair.sup_out.size))
            buf = newbuf(si.keep_len)
            asmb.emit(P["CUT"], (buf, si.keep_len), (cutoff,))
            slotmap[si.dst] = (buf, si.keep_len)
            release_operands(pos, si)
        elif si.op == "pow":
            ab, Sa = slotmap[si.a]
            if si.power == 0:
                buf = newbuf(1)
                asmb.emit(P["ONE"], (buf,))
                slotmap[si.dst] = (buf, 1)
            else:
                res = newbuf(Sa)
                tmp = newbuf(Sa)
                asmb.emit(P["COPY"], (res, ab, Sa))
                asmb.emit(P["COPY"], (tmp, ab, Sa))
                rS, tS = Sa, Sa
                for entry in si.chain:
                    if entry[0] == "r":
                        _t, pb, keep, *_ = entry
                        po, so = emit_pairs(pb)
                        asmb.emit(P["MULP"], (res, tmp, po, so, pb.sup_out.size))
                        alloc.rel(res)
                        res = newbuf(keep)
                        asmb.emit(P["CUT"], (res, keep), (cutoff,))
                        rS = keep
                    else:
                        _t, pb, keep, *_ = entry
                        po, so = emit_pairs(pb)
                        asmb.emit(P["MULP"], (tmp, tmp, po, so, pb.sup_out.size))
                        alloc.rel(tmp)
                        tmp = newbuf(keep)
                        asmb.emit(P["CUT"], (tmp, keep), (cutoff,))
                        tS = keep
                alloc.rel(tmp)
                slotmap[si.dst] = (res, rS)
            release_operands(pos, si)
        elif si.op in ("exp", "sin", "cos", "log", "sqrt", "div"):
            arg = si.b if si.op == "div" else si.a
            ab, Sa = slotmap[arg]
            Sf = si.sup_f.size
            fb = newbuf(Sf)
            asmb.emit(P["COPY"], (fb, ab, Sf))
            asmb.emit(P["SPLITC"], (C0, fb))
            fbuf[0] = fb
            if si.op == "exp":
                res = newbuf(1)
                asmb.emit(P["ONE"], (res,))
                curS = 1
                for idx, (pb, keep, _s) in enumerate(si.chain):
                    i = k - idx
                    nb = newbuf(curS)
                    asmb.emit(P["DIV_IMM"], (nb, res, curS), (float(i),))
                    alloc.rel(res)
                    res, curS = ctx_mul(nb, curS, pb, keep)
                    alloc.rel(nb)
                    asmb.emit(P["A0_IMM"], (res,), (1.0,))
                asmb.emit(P["R_EXP"], (CM, C0))
                out = newbuf(curS)
                asmb.emit(P["SC_REG"], (out, res, CM, curS))
                alloc.rel(res)
            elif si.op in ("sin", "cos"):
                cosc = si.op == "cos"
                asmb.emit(P["R_SIN"], (T2, C0))
                asmb.emit(P["R_COS"], (T3, C0))
                res = newbuf(1)
                asmb.emit(P["CONST"], (res,), (0.0,))
                asmb.emit(P["SET0"], (res, T3 if cosc else T2))
                pw = newbuf(1)
                asmb.emit(P["ONE"], (pw,))
                curS, pwS = 1, 1
                for i in range(1, k + 1):
                    pb, keep, _s = si.chain[i - 1]
                    npw, pwS = ctx_mul(pw, pwS, pb, keep, cut=False)
                    alloc.rel(pw)
                    pw = npw
                    r4 = i % 4
                    wheel = ({1: (T2, True), 2: (T3, True), 3: (T2, False),
                              0: (T3, False)} if cosc else
                             {1: (T3, False), 2: (T2, True), 3: (T3, True),
                              0: (T2, False)})
                    src, negd = wheel[r4]
                    if negd:
                        asmb.emit(P["R_NEG"], (4, src))
                        src = 4
                    asmb.emit(P["R_DIVI"], (5, src), (float(_m.factorial(i)),))
                    term = newbuf(pwS)
                    asmb.emit(P["SC_REG"], (term, pw, 5, pwS))
                    su, er, ep = si.acc[i - 1]
                    inva, invb = _inv_maps(su.size, er, ep)
                    iao = pools.add_idx(("sa", id(si), i), inva)
                    ibo = pools.add_idx(("sb", id(si), i), invb)
                    nres = newbuf(su.size)
                    asmb.emit(P["ADDSUB"], (nres, res, term, iao, ibo, su.size, 0))
                    alloc.rel(res)
                    alloc.rel(term)
                    res, curS = nres, su.size
                alloc.rel(pw)
                out = newbuf(curS)
                asmb.emit(P["CUTBUF"], (out, res, curS), (cutoff,))
                alloc.rel(res)
            elif si.op == "log":
                f2 = newbuf(Sf)
                asmb.emit(P["DIVREG"], (f2, fb, C0, Sf))
                fbuf[0] = f2
                res = newbuf(Sf)
                asmb.emit(P["DIV_IMM"], (res, f2, Sf), (float(k),))
                curS = Sf
                for idx, (pb, keep, _s) in enumerate(si.chain):
                    i = k - 1 - idx
                    asmb.emit(P["A0_IMM"], (res,), (-(1.0 / i),))
                    nb = newbuf(curS)
                    asmb.emit(P["NEG"], (nb, res, curS))
                    alloc.rel(res)
                    res, curS = ctx_mul(nb, curS, pb, keep)
                    alloc.rel(nb)
                asmb.emit(P["R_LOG"], (CM, C0))
                asmb.emit(P["A0_REG"], (res, CM))
                out, curS = res, curS
                alloc.rel(f2)
            elif si.op == "sqrt":
                asmb.emit(P["R_MULI"], (4, C0), (2.0,))
                f2 = newbuf(Sf)
                asmb.emit(P["DIVREG"], (f2, fb, 4, Sf))
                fbuf[0] = f2
                res = newbuf(Sf)
                asmb.emit(P["COPY"], (res, f2, Sf))
                curS = Sf
                i, j = k, 2 * k - 3
                for pb, keep, _s in si.chain:
                    nb = newbuf(curS)
                    asmb.emit(P["SC_IMM"], (nb, res, curS), (j / -i,))
                    alloc.rel(res)
                    asmb.emit(P["A0_IMM"], (nb,), (1.0,))
                    res, curS = ctx_mul(nb, curS, pb, keep)
                    alloc.rel(nb)
                    i, j = i - 1, j - 2
                asmb.emit(P["A0_IMM"], (res,), (1.0,))
                asmb.emit(P["R_SQRT"], (CM, C0))
                out = newbuf(curS)
                asmb.emit(P["SC_REG"], (out, res, CM, curS))
                alloc.rel(res)
                alloc.rel(f2)
            else:  # div: rec series on b, then pair2 mul + cutoff
                asmb.emit(P["R_REC"], (CM, C0))
                f2 = newbuf(Sf)
                asmb.emit(P["SC_REG"], (f2, fb, CM, Sf))
                fbuf[0] = f2
                res = newbuf(1)
                asmb.emit(P["ONE"], (res,))
                curS = 1
                for pb, keep, _s in si.chain:
                    nb = newbuf(curS)
                    asmb.emit(P["NEG"], (nb, res, curS))
                    alloc.rel(res)
                    res, curS = ctx_mul(nb, curS, pb, keep)
                    alloc.rel(nb)
                    asmb.emit(P["A0_IMM"], (res,), (1.0,))
                rec_b = newbuf(curS)
                asmb.emit(P["SC_REG"], (rec_b, res, CM, curS))
                alloc.rel(res)
                alloc.rel(f2)
                pb2, keep2, _sa, _sr = si.pair2
                nb_buf, Sn = slotmap[si.a]
                po, so = emit_pairs(pb2)
                asmb.emit(P["MULP"], (nb_buf, rec_b, po, so, pb2.sup_out.size))
                out = newbuf(keep2)
                asmb.emit(P["CUT"], (out, keep2), (cutoff,))
                alloc.rel(rec_b)
                curS = keep2
            alloc.rel(fb)
            slotmap[si.dst] = (out, curS)
            release_operands(pos, si)
        else:  # pragma: no cover  # unreachable: specialize emits only known ops
            raise AssertionError(si.op)

    for i, (slot, emb) in enumerate(zip(spec.out_slots, spec.out_embeds)):
        buf, S = slotmap[slot]
        eo = pools.add_idx(("oe", i), emb.cpu().numpy())
        asmb.emit(P["OUT"], (i, buf, eo, S))

    stride = max(1, stride_seen[0])
    if stride > MAX_POLY_W:
        raise ValueError(f"point poly width {stride} > {MAX_POLY_W}")
    if s_prod_max[0] > MAX_PROD_W:
        raise ValueError(f"point product width {s_prod_max[0]} > {MAX_PROD_W}")
    n_scal = NREG
    shared = (alloc.peak * stride + s_prod_max[0] + n_scal) * 8
    if shared > MAX_SHARED:
        raise ValueError(f"point shared budget {shared} > {MAX_SHARED}")

    def cat_i(parts):
        return (np.concatenate(parts) if parts else np.zeros(1, dtype=np.int32))

    return PointTape(
        opc=torch.tensor(asmb.opc, dtype=torch.int32, device=device),
        io=torch.tensor(np.array(asmb.ints, dtype=np.int32), device=device),
        fo=torch.tensor(np.array(asmb.flts, dtype=np.float64), device=device),
        idxp=torch.from_numpy(cat_i(pools.idx)).to(device),
        pia=torch.from_numpy(cat_i(pools.pia)).to(device),
        pjb=torch.from_numpy(cat_i(pools.pjb)).to(device),
        soff=torch.from_numpy(cat_i(pools.soff)).to(device),
        stride=stride,
        n_bufs=max(alloc.peak, 1),
        s_prod=max(s_prod_max[0], 1),
        n_scal=n_scal,
        nc=len(spec.out_slots),
        Sfu=spec.sup_out_union.size,
        device=device,
    )


def exec_valid_tape(vt: ValidTape, x: torch.Tensor, x_rem: torch.Tensor,
                    bad: torch.Tensor):
    """One-launch sparse validated pass (exec_valid_s contract).

    x [B, n, Sx], x_rem [B, n, 2], bad [B] bool (updated). Returns
    (coeffs_iv [B, n, Sfu, 2], rem [B, n, 2], cache [B, C, 2],
    strict_tails [B, St, 2]).
    """
    if not valid_available():
        raise RuntimeError("valid tape kernels unavailable")
    bad_u8 = bad.to(torch.uint8).contiguous()
    from .cuda_kernels import _guarded as _dg3
    out_c, out_r, cache, tails = _dg3(x.device, _vext.valid_tape, 
        vt.opc, vt.io, vt.fo, vt.idxp, vt.pia, vt.pjb, vt.soff, vt.facp,
        vt.tabp, x.contiguous(), x_rem.contiguous(), bad_u8,
        vt.n_cache, vt.n_strict, vt.stride, vt.n_bufs, vt.s_prod, vt.n_scal,
        vt.nc, vt.Sfu, vt.cutoff)
    bad |= bad_u8.to(torch.bool)
    return out_c, out_r, cache, tails


def exec_point_tape(pt: PointTape, x: torch.Tensor) -> torch.Tensor:
    """One-launch sparse point pass (exec_point_s contract): [B,n,Sx] -> [B,n,Sfu]."""
    if not valid_available():
        raise RuntimeError("valid tape kernels unavailable")
    from .cuda_kernels import _guarded as _dg4
    return _dg4(x.device, _vext.point_tape, 
        pt.opc, pt.io, pt.fo, pt.idxp, pt.pia, pt.pjb, pt.soff, x.contiguous(),
        pt.stride, pt.n_bufs, pt.s_prod, pt.n_scal, pt.nc, pt.Sfu)
