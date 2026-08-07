from __future__ import annotations

from scripts.attribute_tora_q3_chrome_trace import (
    aggregate_attribution,
    attribute_points,
    sanitize_python_frame,
)


def test_source_attribution_chooses_innermost_containing_interval() -> None:
    targets = [
        {"operator": "aten::item", "start": 5.0, "thread": 7, "input_shapes": "[[]]"},
        {
            "operator": "aten::_local_scalar_dense",
            "start": 5.1,
            "thread": 7,
            "input_shapes": "[[]]",
        },
        {"operator": "aten::to", "start": 8.0, "thread": 7, "input_shapes": "[[48,5]]"},
    ]
    intervals = [
        (0.0, 10.0, 7, "outer.py:1:outer"),
        (4.0, 6.0, 7, "inner.py:2:inner"),
    ]
    labels = attribute_points(targets, intervals, missing="missing")
    assert labels == ["inner.py:2:inner", "inner.py:2:inner", "outer.py:1:outer"]

    sync, transfers, summary = aggregate_attribution(
        targets, labels, ["sin_tm", "sin_tm", "affine_composition"]
    )
    assert sync[0]["source_callsite"] == "inner.py:2:inner"
    assert sync[0]["host_scalar_sync_estimate"] == 1
    assert transfers[0]["aten_to_count"] == 1
    assert summary["host_scalar_sync_estimate"] == 1


def test_python_frame_sanitization_never_retains_private_root() -> None:
    assert sanitize_python_frame(
        "torch_tm_flowpipe/batched_dense_tm.py(134): __post_init__"
    ) == "src/torch_tm_flowpipe/batched_dense_tm.py:134:__post_init__"
    assert sanitize_python_frame("unrelated/package.py(1): call") is None
