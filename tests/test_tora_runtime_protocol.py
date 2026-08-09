from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tora_q3_stage_parity_fused_20260809"


def load(relative: str) -> dict[str, object]:
    return json.loads((OUTPUT / relative).read_text(encoding="utf-8"))


@pytest.mark.regression
@pytest.mark.protocol
def test_fused_timing_excludes_warmup_and_has_five_stable_repeats() -> None:
    runtime = load("fused_kernel/common_control_runtime.json")
    assert runtime["excluded_warmup"]["completed_segments"] == 200
    assert runtime["runtime"]["repeat_count"] == 5
    assert runtime["completed_segments_each"] == [200] * 5
    assert runtime["checksum_stable"] is True
    with (OUTPUT / "fused_kernel/t20_runtime_repeats.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        repeats = list(csv.DictReader(handle))
    assert len(repeats) == 5
    assert [int(row["repeat"]) for row in repeats] == [1, 2, 3, 4, 5]
    assert all(row["status"] == "PASS" for row in repeats)
    assert len({float(row["checksum"]) for row in repeats}) == 1


@pytest.mark.regression
@pytest.mark.protocol
def test_common_control_and_native_runtime_scopes_are_not_mixed() -> None:
    common = load("common_control/summary.json")
    native = load("native_full_loop/hierarchical_gates.json")
    assert common["workload"]["controller_time_included"] is False
    assert common["workload"]["independent_native_closed_loop"] is False
    assert native["common_control_substitution_forbidden"] is True
    assert all(
        values == {"T5": None, "T10": None, "T20": None}
        for values in native["torch_target_width_availability"].values()
    )


@pytest.mark.regression
@pytest.mark.protocol
def test_formal_gpu_runtime_records_device_sync_telemetry_and_memory() -> None:
    fused = load("fused_kernel/summary.json")
    resources = load("fused_kernel/resource_recheck.json")
    assert fused["device"] == "Tesla V100-SXM2-16GB"
    assert fused["dtype"] == "float64"
    assert fused["gates"]["P2_program_sync"]["observed"] == 1
    assert fused["gates"]["P3_aten_to"]["observed"] == 25
    assert fused["profiler"]["fused_segmented"]["cuda_launch_api_count"] == 7941
    assert resources["maximum_process_rss_bytes"] == 6925746176
    assert resources["peak_cuda_memory_bytes"] == 1031874048
