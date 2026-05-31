import torch

from torch_tm_flowpipe import Interval


def test_interval_add_mul_contains():
    a = Interval(1.0, 2.0)
    b = Interval(-3.0, 4.0)
    c = a + b
    assert c.contains(-2.0)
    assert c.contains(6.0)
    d = a * b
    assert d.contains(-6.0)
    assert d.contains(8.0)


def test_interval_pow_even_crossing_zero():
    a = Interval(-2.0, 3.0).pow_int(2)
    assert a.contains(0.0)
    assert a.contains(9.0)
    assert float(a.lo) <= 0.0
    assert float(a.hi) >= 9.0


def test_interval_outward_width_positive():
    w = Interval(0.0, 1.0).width()
    assert torch.isfinite(w)
    assert float(w) >= 1.0
