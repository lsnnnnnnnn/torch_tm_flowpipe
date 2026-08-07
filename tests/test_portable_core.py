from __future__ import annotations

from decimal import Decimal
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts.check_readme_surface import check_readme
from scripts.run_tora_q3_secret_scan import scan_repository
from torch_tm_flowpipe.batched_dense_tm import BatchedMonomialBasis
from torch_tm_flowpipe.interval import Interval


ROOT = Path(__file__).resolve().parents[1]


def _decimal(value: torch.Tensor) -> Decimal:
    return Decimal.from_float(float(value))


def _assert_contains_exact_decimal(interval: Interval, value: Decimal) -> None:
    assert _decimal(interval.lo) <= value <= _decimal(interval.hi)


@pytest.mark.unit
def test_interval_add_sub_mul_div_are_outward_and_use_nextafter() -> None:
    left = Interval(0.1, 0.3)
    right = Interval(-0.2, 0.4)
    left_lo, left_hi = _decimal(left.lo), _decimal(left.hi)
    right_lo, right_hi = _decimal(right.lo), _decimal(right.hi)

    added = left + right
    assert _decimal(added.lo) <= left_lo + right_lo
    assert _decimal(added.hi) >= left_hi + right_hi
    assert added.lo == torch.nextafter(
        left.lo + right.lo, torch.full_like(left.lo, -torch.inf)
    )
    assert added.hi == torch.nextafter(
        left.hi + right.hi, torch.full_like(left.hi, torch.inf)
    )

    subtracted = left - right
    _assert_contains_exact_decimal(subtracted, left_lo - right_hi)
    _assert_contains_exact_decimal(subtracted, left_hi - right_lo)

    multiplied = left * right
    exact_products = (
        left_lo * right_lo,
        left_lo * right_hi,
        left_hi * right_lo,
        left_hi * right_hi,
    )
    _assert_contains_exact_decimal(multiplied, min(exact_products))
    _assert_contains_exact_decimal(multiplied, max(exact_products))

    divisor = Interval(0.7, 1.1)
    divided = left / divisor
    exact_quotients = (
        left_lo / _decimal(divisor.lo),
        left_lo / _decimal(divisor.hi),
        left_hi / _decimal(divisor.lo),
        left_hi / _decimal(divisor.hi),
    )
    _assert_contains_exact_decimal(divided, min(exact_quotients))
    _assert_contains_exact_decimal(divided, max(exact_quotients))
    with pytest.raises(ZeroDivisionError):
        left / Interval(-1.0, 1.0)


def _route_fingerprint(tensors: tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.unit
def test_dense_q3_route_tables_have_frozen_fingerprints() -> None:
    basis = BatchedMonomialBasis.build(6, 3, "cpu")
    assert _route_fingerprint(basis.multiplication_plan_for_degree(None)) == (
        "0d7c9fa6e3c06039aa9807cd6b8cdf06b7f729f5578e546f84584a3082ee747c"
    )
    assert _route_fingerprint(basis.integration_plan(0)) == (
        "46aaf7613901c572c723cce8f366891500c9f4c67f51708f9651a3ad6fe9edac"
    )
    kept_left, _, _, dropped_left, _, _, unique_dropped = (
        basis.multiplication_plan_for_degree(None)
    )
    assert kept_left.numel() == 455
    assert dropped_left.numel() == 6601
    assert unique_dropped.shape == (840, 6)


@pytest.mark.unit
def test_readme_links_and_command_paths_exist() -> None:
    result = check_readme(ROOT)
    assert result["missing_relative_links"] == []
    assert result["missing_relative_command_paths"] == []


@pytest.mark.integration
def test_readme_portable_example_executes() -> None:
    completed = subprocess.run(
        [sys.executable, "examples/tora_q3_one_step.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert completed.stdout.strip() == "status=validated accepted_leaves=1/1"


@pytest.mark.protocol
def test_public_artifact_scan_covers_current_tree_and_clean_history() -> None:
    scan = scan_repository(ROOT)
    assert scan["reachable_commits"]
    assert scan["working_records"]
    assert scan["history_records"]
    assert scan["unallowlisted_matches"] == []
    current_sensitive = [
        record
        for record in scan["sensitive_suffix_candidates"]
        if record["scope"] in {"working_tracked_tree", "working_untracked_tree"}
    ]
    assert current_sensitive == []
