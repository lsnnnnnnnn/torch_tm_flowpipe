from __future__ import annotations

import csv

from experiments.run_c4_performance_gate import WORKER, _csv_scientific_sha, _iqr


def test_performance_worker_binds_production_scope_and_authoritative_vdp_policy() -> None:
    assert "DENSE_OBSERVER_NONE" in WORKER
    assert 'trigger="proactive_depth1_on_named_contexts"' in WORKER
    assert 'named_contexts=("polynomial_truncation",)' in WORKER
    assert "wall_s = time.perf_counter() - started" in WORKER
    assert WORKER.index("wall_s = time.perf_counter() - started") < WORKER.index(
        "snapshot_started = time.perf_counter()"
    )


def test_vdp_historical_column_projection_ignores_only_new_columns(tmp_path) -> None:
    baseline = tmp_path / "baseline.csv"
    current = tmp_path / "current.csv"
    with baseline.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("state", "queue", "stage_runtime_s"))
        writer.writeheader()
        writer.writerow({"state": "abc", "queue": "def", "stage_runtime_s": "10"})
    with current.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("state", "queue", "stage_runtime_s", "new_observer_counter"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "state": "abc",
                "queue": "def",
                "stage_runtime_s": "1",
                "new_observer_counter": "99",
            }
        )
    baseline_sha, fields = _csv_scientific_sha(baseline)
    current_sha, projected = _csv_scientific_sha(current, fields=fields)
    assert fields == projected == ("state", "queue")
    assert baseline_sha == current_sha


def test_inclusive_iqr_is_defined_for_required_three_and_five_repeats() -> None:
    assert _iqr([1.0, 2.0, 3.0]) == 1.0
    assert _iqr([1.0, 2.0, 3.0, 4.0, 5.0]) == 2.0

