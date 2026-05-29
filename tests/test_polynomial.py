from torch_tm_flowpipe import Interval, Polynomial


def test_polynomial_interval_evaluation():
    x = Polynomial.variable(0, 2)
    y = Polynomial.variable(1, 2)
    p = x * x + 2.0 * y + 1.0
    iv = p.evaluate_interval([Interval(0.0, 1.0), Interval(-1.0, 2.0)])
    assert iv.contains(-1.0)
    assert iv.contains(6.0)


def test_substitute_and_drop_variable():
    x = Polynomial.variable(0, 2)
    tau = Polynomial.variable(1, 2)
    p = x + 2.0 * tau + tau * tau
    q = p.substitute_const(1, 3.0).drop_variable(1)
    iv = q.evaluate_interval([Interval(1.0, 1.0)])
    assert iv.contains(16.0)
    assert q.n_vars == 1


def test_extend_vars():
    x = Polynomial.variable(0, 1)
    q = x.extend_vars(2)
    assert q.n_vars == 3
    assert q.active_variables() == {0}
