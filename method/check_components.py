"""Exercise the production residual evaluator and strict history on CPU."""
import torch
from flowstar_gpu import interval
from flowstar_gpu.symbolic_remainder import make_symbolic_remainder, propagate

from controllers.partition_batch import partition_chunks
from controllers.residual import DelayedReLUResidual
from flowstar_gpu.sparse_exec import _sr_linear_prepare


def check_controller():
    # Positive and negative copies of the same activation reach output 0.
    # The exact two-output network is zero on the whole box.
    layers = []
    for inputs, outputs in ((4, 25), (25, 25), (25, 2)):
        layers.append((torch.zeros(inputs, outputs, dtype=torch.float64),
                       torch.zeros(outputs, dtype=torch.float64)))
    layers[0][0][0, 0] = 1
    layers[1][0][0, :2] = 1
    layers[2][0][:2, 0] = torch.tensor([1., -1.], dtype=torch.float64)
    hull = torch.tensor([[[-1., 1.]] * 4], dtype=torch.float64)
    slope = torch.zeros(1, 2, 4, dtype=torch.float64)
    result = DelayedReLUResidual(torch, interval, layers).evaluate(hull, slope)
    residual = result['intersected_residual_interval']
    assert bool(((residual[..., 0] <= 0) & (residual[..., 1] >= 0)).all())
    assert bool(interval.contains(result['original_residual_interval'], residual).all())
    assert sum(chunk.shape[0] for chunk in partition_chunks(torch, interval, hull)) == 256
    print('Residual contains the exact zero output and lies inside the forward bound.')
    print('Partition covers the input box with 256 subboxes.')


def check_history():
    # A previous error of 3 is transported by a factor enclosing 1/3.
    queue = make_symbolic_remainder(batch=1, n=1, max_size=4, device='cpu')
    one = torch.ones(1, 1, 1, dtype=torch.float64)
    propagate(queue, one, strict=True, phi_i_iv=interval.from_point(one))
    queue.append_j(torch.tensor([[[3., 3.]]], dtype=torch.float64))
    inverse = torch.tensor([[1/3]], dtype=torch.float64)
    inverse_bounds = torch.tensor([[[0.33, 0.34]]], dtype=torch.float64)
    index = torch.tensor([0])
    _, _, point, bounds = _sr_linear_prepare(one, inverse, inverse_bounds, index, index, True)
    transported = propagate(queue, point, strict=True, phi_i_iv=bounds)
    lo, hi = transported[0, 0].tolist()
    assert 0.98 < lo <= 1.0 <= hi < 1.03, (lo, hi)
    print(f'History encloses the exact result 1 in [{lo:.6f}, {hi:.6f}].')


if __name__ == '__main__':
    check_controller()
    check_history()
