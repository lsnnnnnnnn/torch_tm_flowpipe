"""Experimental local allocation for completely overwritten CUDA outputs.

No global deterministic setting is changed. The public cuda_kernels wrappers
keep their dtype/device/shape checks and device/stream guard. Only the eight
audited wrapper functions below use raw allocation; iv_neg, tape zero buffers,
metadata, injective destinations and public output ownership are unchanged.

This uses a PyTorch detail API and is deliberately pinned to 2.5.1. An
unsupported version or failed build reports unavailable so callers can retain
the original extension. This module never replaces the original by itself.
"""
from __future__ import annotations

import hashlib
import re

import torch

from . import cuda_kernels as base


NAME = "flowstar_seg_private_output_v1"
SUPPORTED = torch.__version__.split("+")[0] == "2.5.1"
AUDITED_CUDA_SHA256 = "27b3c4081ab83aebdef98b72f55f694975affb3918643a0ec26ca02a0f53d81d"
FUNCTIONS = ["seg_mul_iv", "seg_mul_pt", "seg_dot_pt_iv", "iv_mul",
             "iv_addsub", "iv_neg", "iv_mul_point", "iv_sum", "iv_dot_point_iv"]
_ext = None
_tried = False

# Replacements are exact, function-local and checked for their expected count.
# The mathematical CUDA kernels and their launch order remain byte-identical.
_ALLOCATIONS = {
    "seg_mul_iv": [("torch::empty({R, M, 2}, a.options())", "private_output_empty({R, M, 2}, a)")],
    "seg_mul_pt": [("torch::empty({R, M}, a.options())", "private_output_empty({R, M}, a)")],
    "iv_mul": [("torch::empty_like(a)", "private_output_empty(a.sizes(), a, a.numel() % 2 == 0)")],
    "iv_addsub": [("torch::empty_like(a)", "private_output_empty(a.sizes(), a, a.numel() % 2 == 0)")],
    "iv_mul_point": [("torch::empty_like(a)", "private_output_empty(a.sizes(), a, a.numel() % 2 == 0)")],
    "iv_sum": [("torch::empty({N, 2}, x.options())", "private_output_empty({N, 2}, x)")],
    "iv_dot_point_iv": [("torch::empty({N, 2}, p.options())", "private_output_empty({N, 2}, p)")],
    "seg_dot_pt_iv": [
        ("torch::empty({R, M, 2}, p.options())", "private_output_empty({R, M, 2}, p)"),
        ("torch::empty({R, 64, 2}, p.options())", "private_output_empty({R, 64, 2}, p)"),
    ],
}

_HELPER = r"""
// Private to these wrappers: each selected kernel writes every logical output
// scalar on the current stream before the result can be consumed. The 64-chunk
// workspace is fully written (including empty chunks) before its reduction.
inline torch::Tensor private_output_empty(at::IntArrayRef sizes,
                                         const torch::Tensor& reference,
                                         bool full_coverage = true) {
  // Preserve the old initialized allocation for unsupported shapes/layouts.
  // In particular, an odd number of interval scalars leaves one scalar outside
  // the legacy kernel's N=numel/2 loop, so it MUST NOT use raw allocation.
  if (!full_coverage || !reference.is_cuda() ||
      reference.scalar_type() != at::kDouble || !reference.is_contiguous()) {
    return torch::empty(sizes, reference.options());
  }
  auto result = torch::Tensor(at::detail::empty_cuda(sizes, reference.options()));
  PRIVATE_TEST_POISON
  return result;
}
"""


def sources(*, test_poison: str | None = None):
    """Generate the auditable source; optional poison is for separate test builds."""
    if hashlib.sha256(base._CUDA_SRC.encode()).hexdigest() != AUDITED_CUDA_SHA256:
        raise RuntimeError("CUDA source changed: private-output coverage must be re-audited")
    if test_poison not in (None, "nan", "finite"):
        raise ValueError("unknown private-output test poison")
    poison = {None: "", "nan": "result.fill_(std::numeric_limits<double>::quiet_NaN());",
              "finite": "result.fill_(0x1.23456789abcdep+30);"}[test_poison]
    helper = _HELPER.replace("PRIVATE_TEST_POISON", poison)
    cuda = base._CUDA_SRC
    for function, replacements in _ALLOCATIONS.items():
        pattern = r"(torch::Tensor " + function + r"\([\s\S]*?\n\})"
        matches = list(re.finditer(pattern, cuda))
        if len(matches) != 1:
            raise RuntimeError(f"unexpected wrapper source for {function}")
        match = matches[0]
        body = match.group(0)
        for old, new in replacements:
            if body.count(old) != 1:
                raise RuntimeError(f"allocation audit changed for {function}: {old}")
            body = body.replace(old, new)
        cuda = cuda[:match.start()] + body + cuda[match.end():]
    cuda = cuda.replace("#include <ATen/cuda/CUDAContext.h>",
                        "#include <ATen/cuda/CUDAContext.h>\n#include <ATen/cuda/EmptyTensor.h>\n#include <limits>", 1)
    cuda = cuda.replace("namespace {", "namespace {\n" + helper, 1)
    return base._CPP_SRC, cuda


def available():
    global _ext, _tried
    if not _tried:
        _tried = True
        if SUPPORTED and hashlib.sha256(base._CUDA_SRC.encode()).hexdigest() == AUDITED_CUDA_SHA256:
            cpp, cuda = sources()
            _ext = base.load_cuda_extension(NAME, cpp, cuda, FUNCTIONS)
    return _ext is not None
