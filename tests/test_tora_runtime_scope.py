from __future__ import annotations

import pytest

from experiments.benchmark_tora_q3_backend import summarize_timing_samples


@pytest.mark.unit
def test_runtime_scope_summary_uses_median_iqr_min_and_max() -> None:
    summary = summarize_timing_samples([1.0, 2.0, 3.0, 4.0])
    assert summary["repeats"] == 4
    assert summary["median_seconds"] == 2.5
    assert summary["iqr_seconds"] == 1.5
    assert summary["min_seconds"] == 1.0
    assert summary["max_seconds"] == 4.0


@pytest.mark.unit
def test_runtime_scope_summary_rejects_invalid_samples() -> None:
    with pytest.raises(ValueError):
        summarize_timing_samples([])
    with pytest.raises(ValueError):
        summarize_timing_samples([1.0, -1.0])
