"""Bitwise-determinism setup (plan D9, GOTCHAS E4).

Reproducibility policy: the SAME device must produce bitwise-identical results
across runs (asserted by tests/unit/test_determinism.py); CPU vs CUDA are NOT
required to match bitwise (different reduction orders/FMA), but both must pass
the soundness and parity gates independently — the Rump reduction bounds make
any reduction order sound, so determinism here is purely about reproducibility
and debuggability, never about soundness.

Import-order caveat: CUBLAS reads its workspace config when the first CUBLAS
kernel launches, so `enable_determinism()` must run before ANY matmul — the
public entry points (reach, benchmarks) call it first thing.
"""

from __future__ import annotations

import os
from typing import Any

import torch

_NO_FTZ_CHECKED: set[str] = set()


def _probe_gradual_underflow(device: torch.device, check_custom: bool) -> dict[str, Any]:
    """Exercise representative binary64 subnormal operations on one path."""
    zero = torch.zeros((), dtype=torch.float64, device=device)
    one = torch.ones((), dtype=torch.float64, device=device)
    eta = torch.nextafter(zero, one)
    smallest_normal = torch.tensor(
        torch.finfo(torch.float64).tiny, dtype=torch.float64, device=device
    )
    results: dict[str, Any] = {
        "nextafter_eta": eta.item() == 5e-324,
        "mul_identity": (eta * one).item() == 5e-324,
        "normal_to_subnormal": (smallest_normal * (2.0**-52)).item() == 5e-324,
        "subnormal_add": (eta + eta).item() == 1e-323,
        "custom_checked": False,
        "custom_mul_identity": True,
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if check_custom and device.type == "cuda":
        from . import cuda_kernels as ck

        if ck.available():
            interval = torch.stack((eta, eta)).reshape(1, 2)
            point_one = torch.ones(1, dtype=torch.float64, device=device)
            custom = ck.iv_mul_point(interval, point_one)
            torch.cuda.synchronize(device)
            results["custom_checked"] = True
            results["custom_mul_identity"] = bool(
                (custom[0, 0] == eta) & (custom[0, 1] == eta)
            )
    return results


def assert_gradual_underflow(device: str | torch.device | None = None) -> None:
    """Fail closed unless every active float64 path preserves subnormals.

    CPU is always checked because host fallbacks remain part of the engine.
    Selecting CUDA additionally checks torch CUDA arithmetic and, when the
    shipped custom extension is active, a directed-rounding custom kernel.
    The cache is per concrete device and can be cleared by tests to exercise
    simulated FTZ rejection.
    """
    requested = torch.device("cpu" if device is None else device)
    devices = [torch.device("cpu")]
    if requested.type == "cuda":
        devices.append(requested)
    for active in devices:
        key = str(active)
        if key in _NO_FTZ_CHECKED:
            continue
        observations = _probe_gradual_underflow(active, check_custom=True)
        failed = [
            name for name, value in observations.items()
            if name != "custom_checked" and value is not True
        ]
        if failed:
            raise FloatingPointError(
                f"gradual-underflow/no-FTZ startup check failed on {active}: "
                f"{', '.join(failed)}"
            )
        _NO_FTZ_CHECKED.add(key)


def enable_determinism(device: str | torch.device | None = None) -> None:
    """Force deterministic kernels. Safe to call more than once.

    - CUBLAS_WORKSPACE_CONFIG=:4096:8 makes CUBLAS GEMMs deterministic
      (required by torch.use_deterministic_algorithms on CUDA >= 10.2).
    - use_deterministic_algorithms(True) makes torch error out on any
      nondeterministic kernel instead of silently using it — that hard error is
      exactly what we want during development (design rule: no duplicate-index
      scatter_add in polynomial products; static sorted index maps instead).
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    assert_gradual_underflow(device)
