# Huan `flowstar-gpu` proof-contract audit (2026-08-26)

## Scope and provenance

This is a plant-only, read-only audit of clean upstream engine commit
`d5f0b68fcd36ba5f582733624f074728fe9720d8`.  The authoritative paper source is
`docs/paper/main.tex` in that tree.  Controller, CROWN, auto_LiRPA, ONNX, TMNP,
and NDB paths were not started.  The machine-readable contract is
`outputs/huan_repro_audit/proof_to_code_map.csv`.

The map distinguishes a current clean-source reproduction from the historical
`CROWN-Reach-GPU` records.  Every historical result record names a `-dirty`
engine revision, while no corresponding dirty patch is archived; those results
cannot establish exact provenance for this clean-source audit.

## Contract result

Fourteen claims were mapped using the required schema.  D1 and D2 pass on CPU
and CUDA, including 7 interval edge cases and 987 independently checked
reduction schedules on each device.  The CUDA run reports
`cuda_kernel_available=true`, so it did not silently exercise only the fallback
path.

That local success does **not** close the paper-level proof contract:

- `FP_NO_FTZ_STARTUP` is `CONTRADICTED`.  The paper says the engine asserts the
  no-FTZ/gradual-underflow condition at startup.  Production
  `src/flowstar_gpu/determinism.py::enable_determinism` configures deterministic
  algorithms but contains no such assertion.  The behavior is checked only by
  an out-of-band test and by this audit.
- `STRICT_VERSUS_PARITY` is `CONTRADICTED`.  Strict composition inflates its
  final coefficient GEMM, but `symbolic_remainder.py::propagate` updates the
  live Phi queue with `torch.einsum` and has neither a strict-mode input nor a
  roundoff charge.  Retained monomial-image point-product coefficients likewise
  lack a visible complete charge before the final GEMM.
- `FP_OVERFLOW_DIVZERO_FAIL_CLOSED` is only `PARTIALLY_MAPPED`.  Division has an
  explicit bad mask, but generic arithmetic overflow may yield an extended
  interval and `assert_valid` permits infinity.  Such an interval is not a
  finite certificate.
- `TRANSCENDENTAL_ASSUMPTIONS` is `ASSUMPTION_ONLY`: the ulp budgets are tested
  against mpmath, but no production startup calibration binds them to the
  deployed library version.
- Consequently `POLYNOMIAL_ONLY_UNCONDITIONAL` is `CONTRADICTED`: even a
  polynomial-only run has not established the paper's unconditional claim
  until no-FTZ is asserted and strict point-coefficient roundoff is completely
  accounted for.

## D1: elementwise interval operations

The audit calls the shipped `interval` and `transcendental` functions and uses
exact binary rationals (`fractions.Fraction`) or 200-digit mpmath values as the
oracle.  It covers addition, subtraction/cancellation, multiplication with
signed zero and subnormals, division by a zero-containing interval, square root
from the smallest subnormal to the largest finite float, and an overflow
candidate.  CPU and CUDA each enclose all 7 cases.  The overflow case is
classified as `NONFINITE_EXTENDED_ENCLOSURE_NOT_A_FINITE_CERTIFICATE`, not as a
successful finite proof.

The smallest-subnormal identities were observed on both devices, but this is
environment evidence, not the missing production startup assertion.

## D2: any-order reductions

The exact oracle is computed before rounding with `Fraction`.  The rounded
result is produced under sequential, pairwise, chunk-3, chunk-17, permuted,
FMA-when-available, and `torch.dot` schedules.  Cases include lengths
1, 2, 3, 31, 32, 33, 63, 64, 65, 257, and 4097, severe cancellation,
underflowing products, a near-overflow sum, mixed magnitudes/signs, and 128
seeded adversarial searches.  The enclosing radius is always obtained from the
shipped `rounding.dot_error_bound`; it is not reimplemented by the oracle.

Each finite-intermediate case satisfies the computed bound, 987/987 on CPU and
987/987 on CUDA, and each records the runtime hypothesis `m*u <= 1/4`.  A
non-finite schedule is explicitly outside the theorem's hypotheses rather than
being counted as a pass.

## Gate implication

D1 and D2 are green only as local microkernels.  The proof map contains
contradictions and partial mappings, and D3--D6 must still be adjudicated.
Therefore this document does not authorize the frozen VDP or throughput phases.
The final gate decision and deliverable ledger are recorded separately after
all remaining microreproduction work.
