# GPU Strategy Reality Check

This project should stop treating the Flow* C++ probe as an implementation
route. The probe is a diagnostic oracle only: it is useful for checking
correctness, tightness, segment traces, and where the PyTorch Taylor-model
prototype diverges from Flow*, but it is not a second implementation path to
grow inside this repository.

## Core Claim

Python/Torch is not expected to beat C++ Flow* for a single small adaptive
flowpipe. A one-off Van der Pol step, or a short adaptive flowpipe with a small
number of Taylor-model terms, is exactly the workload where C++ scalar data
structures and mature Flow* algorithms should win.

The speed question is therefore not:

- Can PyTorch match the original Flow* horizon-10 Van der Pol benchmark?

The speed question is:

- Can batched PyTorch Taylor-model operations beat CPU baselines at meaningful
  batch sizes?

## Plausible Speed Path

The only plausible speed path is batched/GPU Taylor-model arithmetic integrated
with CROWN/auto_LiRPA-style workloads. In that setting, many related boxes,
controllers, branches, or verification subproblems could share the same dense
monomial basis and run coefficient operations as tensor kernels.

That path requires the plant Taylor-model representation to be tensorizable:

- coefficients stored in batched tensors, not Python dictionaries of scalar
  tensors;
- a canonical monomial basis per `(dimension, order)`;
- precomputed multiplication/truncation scatter plans;
- batched interval domains and remainders;
- no Python loop over terms or samples in the hot path.

The current `Polynomial` and `TaylorModel` classes are sparse Python object
representations. They are useful for clarity and diagnostics, but they do not
by themselves justify a GPU speed claim.

## Decision Rule

If plant Taylor-model arithmetic cannot be batched and GPU accelerated, this
project should not continue as a CPU Flow* reimplementation. The plant should
remain Flow* C++ for reachability/tightness, and Python should focus on
controller/CROWN orchestration, experiment automation, diagnostics, and data
exchange.

The benchmark in `experiments/batched_tm_gpu_microbench.py` is the decision
instrument for this question. It does not claim that PyTorch is faster unless
the emitted data show a clear speedup at realistic batch sizes.

