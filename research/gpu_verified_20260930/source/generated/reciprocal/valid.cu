
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
