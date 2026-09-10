import pytest
import torch

from experiments.live_gpu_packets.long_horizon import run_long_horizon
from experiments.live_gpu_packets.verify_long_horizon import verify_long_horizon


pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module", autouse=True)
def fixed_threads():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def test_original_b1_streaming_state_chain_and_checkpoint_are_independently_verified(tmp_path):
    output = tmp_path / "van-der-pol-two-step"
    result = run_long_horizon("van_der_pol", "Gp", 2, output)
    receipt = verify_long_horizon(output, require_achieved=True)
    assert result["scope"] == "ORIGINAL_UNPARTITIONED_B1"
    assert result["previous_answers_loaded"] is False
    assert receipt["steps"] == 2 and receipt["packets"] > 0
    assert receipt["queue_owner_schema"] == "c3_cross_step_sr_v1"
    assert receipt["packet_kernel_invocations"] == 4 * receipt["packets"]


def test_brusselator_stream_uses_accepted_boundary_owner_schema(tmp_path):
    output = tmp_path / "brusselator-two-step"
    result = run_long_horizon("brusselator", "Gp", 2, output)
    receipt = verify_long_horizon(output, require_achieved=True)
    assert result["scope"] == "ORIGINAL_UNPARTITIONED_B1"
    assert result["previous_answers_loaded"] is False
    assert receipt["steps"] == 2 and receipt["packets"] > 0
    assert receipt["queue_owner_schema"] == "accepted_boundary_sr_v1"
    assert receipt["packet_kernel_invocations"] == 4 * receipt["packets"]
