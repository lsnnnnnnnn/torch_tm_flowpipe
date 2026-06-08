# Batched Dense Taylor Model Prototype Notes

## Why This Exists

This prototype explores a dense tensor representation for small total-degree Taylor models. The goal is to test whether batched PyTorch kernels can remove Python object overhead in coefficient arithmetic before making any claims about reachability or Flow* parity.

It is intentionally experimental. The production sparse `Polynomial` and `TaylorModel` classes remain the main API.

## Representation

`BatchedMonomialBasis` stores all exponent tuples with total degree at most `order`, ordered by ascending degree and lexicographic exponent tuple. It precomputes index tensors for valid truncated multiplication and records over-order products separately so their interval range can be moved into Taylor-model remainders.

`BatchedPolynomial` stores coefficients as a tensor with shape `[batch, out_dim, n_terms]`. The basis owns exponent metadata and multiplication scatter plans, while coefficient tensors stay in PyTorch for add, affine map, evaluation, range bounding, and truncated multiplication.

`BatchedTaylorModel` stores a dense polynomial plus interval remainders `rem_lo/rem_hi` with shape `[batch, out_dim]`, together with fixed domain boxes `domain_lo/domain_hi` with shape `[batch, dim]`.

## Operations

Implemented operations include basis construction, monomial evaluation, interval monomial bounds, polynomial add/sub/scale/affine map, truncated multiplication with `scatter_add_`, range bounds, tensor evaluation, Taylor-model add/sub/scale/affine map, conservative Taylor-model multiplication, component/concat helpers, and a fixed Van der Pol-like step:

```text
dx/dt = y
dy/dt = y - x - x^2*y
```

The demo uses fixed step size and fixed order only. It has no adaptive rejection and no Flow* comparison path.

## Conservative Remainder Limitations

Taylor-model multiplication uses the conservative decomposition:

```text
(p + I) * (q + J) = trunc(p*q) + dropped_range + B(p)*J + B(q)*I + I*J
```

`dropped_range` is bounded by interval evaluation of the dense over-order monomial products. This is conservative for the small sampled tests, but it is not tight. Duplicate dropped monomial products are bounded term-by-term rather than being merged into a sparse dropped polynomial first, which can widen remainders.

The current range bound is also interval arithmetic over monomials. It is intentionally simple and can be much wider than split evaluation or normalized Flow*-style evaluation.

## Parity Status

The test suite checks small cases against the scalar sparse `Polynomial` path for basis ordering, coefficient add/sub/scale/affine map, evaluation, and truncated multiplication. Taylor-model tests check that dropped terms are represented in the remainder and that sampled fixed-step Van der Pol trajectories are contained by dense model ranges.

This is correctness parity for small algebraic cases only. It is not a full reachability parity result.

## Speed Status

The demo writes `dense_tm_microbench_summary.csv` and `dense_tm_demo_report.md` under `outputs/batched_dense_tm_prototype/`. It compares the dense tensor path with a scalar Python loop up to a scalar cap, then continues dense-only batches. CPU and CUDA are reported separately when requested and available.

The timing numbers are microbenchmarks for this prototype operation only. They should not be presented as end-to-end reachability speedups.

## Not A Flow* Replacement

This module does not add a Flow* mechanism, does not modify the C++ Flow* probe, and does not replace the production sparse Taylor-model implementation. It is a first dense representation experiment that can be removed or redesigned without changing the main API.

## Next Steps

1. Merge duplicate dropped monomial products before interval bounding to reduce remainder width.
2. Add split interval or Bernstein-style range bounds for dense dropped terms.
3. Profile the multiplication plan and range bound separately on larger batches and CUDA.
4. Decide whether to continue the GPU path only after small-case containment remains stable and microbenchmarks show a clear dense advantage.
