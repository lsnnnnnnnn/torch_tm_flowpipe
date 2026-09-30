
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
