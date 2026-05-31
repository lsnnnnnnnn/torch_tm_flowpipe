import pytest
import torch

from torch_tm_flowpipe import Interval, flowpipe_step
from torch_tm_flowpipe.ode_examples import scalar_quadratic_ode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_interval_smoke():
    device = torch.device("cuda")
    seg = flowpipe_step(
        scalar_quadratic_ode,
        [Interval(torch.tensor(0.0, device=device), torch.tensor(0.1, device=device))],
        h=0.01,
        order=4,
    )
    assert seg.status == "validated"
    assert seg.final_tm.range_box()[0].lo.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_tensor_interval_mul():
    device = torch.device("cuda")
    iv = Interval(torch.tensor(1.0, device=device), torch.tensor(2.0, device=device)) * Interval(
        torch.tensor(3.0, device=device), torch.tensor(4.0, device=device)
    )
    assert iv.contains(torch.tensor(8.0, device=device))
