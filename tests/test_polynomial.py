from torch_tm_flowpipe import Interval, Polynomial, evaluate_interval_normal


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

def test_polynomial_cutoff_conservatively_contains_original_range():
    x = Polynomial.variable(0, 2)
    y = Polynomial.variable(1, 2)
    p = 1.0 + 1e-12 * x + 2.0 * y + 5e-11 * x * y
    domain = [Interval(-2.0, 1.0), Interval(0.0, 3.0)]

    kept, removed_range = p.cutoff(1e-10, domain)
    original_range = p.evaluate_interval(domain)
    cutoff_range = kept.evaluate_interval(domain) + removed_range

    assert cutoff_range.contains_interval(original_range, tol=1e-15)
    assert kept.active_variables() == {1}


def test_evaluate_interval_normal_uses_time_step_and_normal_state_domain():
    tau = Polynomial.variable(0, 3)
    x = Polynomial.variable(1, 3)
    y = Polynomial.variable(2, 3)
    poly = 2.0 * tau * tau * x * x + 3.0 * tau * y + x * y
    domain = [Interval(0.0, 0.1), Interval(-50.0, 50.0), Interval(10.0, 12.0)]
    normal = evaluate_interval_normal(
        poly,
        domain,
        step_exp_table={0: Interval(1.0), 1: Interval(0.0, 0.1), 2: Interval(0.0, 0.01)},
        state_var_indices=[1, 2],
        time_var_index=0,
    )
    generic = poly.evaluate_interval(domain)

    assert normal.width().item() < generic.width().item()
    for tau_v in [0.0, 0.025, 0.05, 0.1]:
        for x_v in [-1.0, -0.25, 0.0, 0.75, 1.0]:
            for y_v in [-1.0, 0.0, 1.0]:
                value = poly.evaluate_point([tau_v, x_v, y_v])
                assert normal.contains(value, tol=1e-12)


def test_evaluate_interval_normal_without_time_var_contains_samples():
    x = Polynomial.variable(0, 2)
    y = Polynomial.variable(1, 2)
    poly = x * x + 0.5 * x * y - 2.0 * y
    normal = evaluate_interval_normal(poly, [Interval(-9.0, 9.0), Interval(-3.0, 3.0)], time_var_index=None)

    for x_v in [-1.0, -0.5, 0.0, 1.0]:
        for y_v in [-1.0, 0.25, 1.0]:
            assert normal.contains(poly.evaluate_point([x_v, y_v]), tol=1e-12)
