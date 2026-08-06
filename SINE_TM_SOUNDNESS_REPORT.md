# Soundness report for `sin_tm`

Status: **PASS for the declared float64 domain and orders 0–3.**

`sin_tm` is a generic batched Taylor-model primitive.  It is not specialized
to TORA.  The formal production path does not call `torch.sin` or `torch.cos`:
the sine and cosine of the expansion center are enclosed by a 32-term rational
Maclaurin recurrence, with every multiplication, addition, and division
expanded by `torch.nextafter`.

For a model written as `c + delta`, the retained polynomial is the centered
Taylor expansion through the requested order.  The following contributions
are added to the interval remainder:

1. uncertainty in the outward interval coefficients for `sin(c)` and
   `cos(c)`;
2. floating-point error in retained multiplication and addition routes;
3. every product whose complete total degree exceeds the fixed basis order;
4. polynomial/remainder and remainder/remainder products; and
5. the analytic Lagrange tail `[-r^(q+1)/(q+1)!, +r^(q+1)/(q+1)!]`, using
   `|f^(q+1)| <= 1` and an outward bound `r` on `delta`.

The formal contract requires CPU or CUDA float64, `|c| <= 8`, and
`range(delta) <= 4` in magnitude.  A wider composition domain raises an error
that instructs the caller to split or fail closed.  Non-finite input,
non-float64 arithmetic, and unsupported order also fail closed.

The test suite covers point, symmetric and asymmetric intervals; orders 0–3;
nonzero input remainder; fixed-support composition overflow; extrema crossing;
wide-domain rejection; CPU/CUDA parity; actual TORA `x3` domains; and a
100-decimal-place independent `mpmath` sanity oracle.  The oracle is only a
regression check—the formal justification is the analytic tail and outward
interval arithmetic above.

Machine-readable cases are under
`outputs/tora_q3_native_matched_20260806/sine_tm/`.  The formal test command and
its raw log are recorded with the final test evidence.
