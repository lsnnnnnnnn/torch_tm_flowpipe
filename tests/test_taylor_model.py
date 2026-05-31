from torch_tm_flowpipe import Interval, TaylorModel, TMVector


def test_taylor_model_multiplication_range_contains_samples():
    domain = [Interval(0.0, 1.0)]
    x = TaylorModel.variable(0, domain, order=4)
    y = x * x + Interval(-0.01, 0.01)
    box = y.range_box()
    assert box.contains(-0.01)
    assert box.contains(1.01)


def test_tmvector_identity_and_drop_time():
    x0 = TMVector.identity([Interval(0.0, 0.1)], order=3)
    extended = x0.extend_domain(Interval(0.0, 0.01))
    final = extended.substitute_const(1, 0.01).drop_variable(1)
    assert final.n_vars == 1
    assert final.active_variables() == {0}
