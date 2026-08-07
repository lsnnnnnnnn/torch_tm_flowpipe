from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from torch_tm_flowpipe.tora_controller import (
    EXPECTED_ORIGINAL_CONTROLLER_SHA256,
    NativeToraController,
    ToraAutoLirpaControllerBounder,
    _normalized_inputs,
    file_sha256,
    resolve_external_controller_path,
)
from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_initial_model,
    tora_q3_boundary_from_model,
)


@pytest.mark.unit
def test_controller_sha_contract_and_input_normalization_shapes() -> None:
    zeros = torch.zeros(48, dtype=torch.float64)
    boundary = tora_q3_boundary_from_model(
        build_tora_q3_initial_model(zeros, zeros)
    )
    lower, upper, weight, center = _normalized_inputs(boundary)
    assert EXPECTED_ORIGINAL_CONTROLLER_SHA256 == "52a50c6bc6b1b45b89319edc809cd4d3baca06c32f9fd2ddceb2e95007414418"
    assert lower.shape == upper.shape == (48, 10)
    assert weight.shape == (48, 4, 6)
    assert center.shape == (48, 4)
    assert torch.equal(lower[:, 0], torch.zeros(48))
    assert torch.equal(upper[:, 0], torch.zeros(48))
    assert torch.equal(lower[:, 1:6], -torch.ones((48, 5), dtype=torch.float64))
    assert torch.equal(upper[:, 1:6], torch.ones((48, 5), dtype=torch.float64))
    assert torch.equal(lower[:, 6:], torch.zeros((48, 4), dtype=torch.float64))
    assert torch.equal(upper[:, 6:], torch.zeros((48, 4), dtype=torch.float64))


@pytest.mark.unit
def test_external_controller_absence_skips_but_supplied_asset_fails_closed(
    tmp_path: Path,
) -> None:
    assert resolve_external_controller_path(environ={}) is None
    with pytest.raises(FileNotFoundError, match="is not a file"):
        resolve_external_controller_path(str(tmp_path / "missing.onnx"))
    wrong = tmp_path / "wrong.onnx"
    wrong.write_bytes(b"not the frozen controller")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_external_controller_path(str(wrong))


@pytest.mark.external_integration
@pytest.mark.protocol
def test_external_controller_sha_and_nominal_onnx_reference() -> None:
    path = resolve_external_controller_path()
    if path is None:
        pytest.skip("TORA_CONTROLLER_PATH is required")
    try:
        import onnx
        from onnx.reference import ReferenceEvaluator
    except ImportError:
        pytest.skip("onnx is unavailable in this external environment")
    torch.set_default_dtype(torch.float64)
    states = torch.tensor(
        [[0.65, -0.65, -0.35, 0.55], [0.6, -0.7, -0.4, 0.5]],
        dtype=torch.float64,
    )
    candidate = NativeToraController(path)(states).detach().numpy()
    reference = np.asarray(
        ReferenceEvaluator(onnx.load(path)).run(
            None, {"input": states.float().numpy().reshape(-1, 1, 1, 4)}
        )[0]
    ).reshape(-1, 1)
    assert np.max(np.abs(candidate - reference)) <= 1e-6


@pytest.mark.external_integration
@pytest.mark.cuda
@pytest.mark.slow
def test_external_initial_b48_controller_bound_matches_observation() -> None:
    model_value = os.environ.get("TORA_CONTROLLER_PATH")
    trace_value = os.environ.get("TORA_CONTROLLER_TRACE_PATH")
    if not model_value or not trace_value:
        pytest.skip("TORA_CONTROLLER_PATH and TORA_CONTROLLER_TRACE_PATH are required")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    try:
        import onnx  # noqa: F401
        import auto_LiRPA  # noqa: F401
    except ImportError:
        pytest.skip("onnx and auto_LiRPA are required for controller bounds")
    torch.set_default_dtype(torch.float64)
    zeros = torch.zeros(48, dtype=torch.float64, device="cuda")
    boundary = tora_q3_boundary_from_model(
        build_tora_q3_initial_model(zeros, zeros, device="cuda")
    )
    bounder = ToraAutoLirpaControllerBounder(Path(model_value), boundary)
    result = bounder.bound(boundary)
    observation = json.loads(Path(trace_value).read_text(encoding="utf-8"))["rows"][0]
    for actual, expected in (
        (result.output_lower_before_outward, observation["controller_output_interval_before_outward_composition"]["lower"]),
        (result.output_upper_before_outward, observation["controller_output_interval_before_outward_composition"]["upper"]),
        (result.output_lower_after_outward, observation["controller_output_interval_after_outward_composition"]["lower"]),
        (result.output_upper_after_outward, observation["controller_output_interval_after_outward_composition"]["upper"]),
    ):
        assert np.max(np.abs(actual.numpy() - np.asarray(expected))) <= 1e-12
    assert result.maximum_slope_gap <= 1e-12
