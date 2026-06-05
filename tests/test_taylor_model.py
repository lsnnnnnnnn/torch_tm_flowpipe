from torch_tm_flowpipe import Interval, TaylorModel, TMVector, taylor_model_mul_breakdown


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


def test_taylor_model_multiplication_breakdown_matches_mul_widths():
    domain = [Interval(-1.0, 1.0)]
    x = TaylorModel.variable(0, domain, order=1).with_remainder(Interval(-0.1, 0.1))
    breakdown = taylor_model_mul_breakdown(x, x, order=1)
    product = x * x

    assert breakdown["dropped_trunc_width"] > 0.0
    assert breakdown["total_remainder_width"] == product.remainder.width().item()
    assert breakdown["output_total_range_width"] == product.range_box().width().item()
    assert breakdown["finite"] is True


def test_split_interval_evaluation_is_no_wider_for_dropped_terms():
    domain = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    x = TaylorModel.variable(0, domain, order=1, truncation_range_split=2)
    y = TaylorModel.variable(1, domain, order=1, truncation_range_split=2)
    dropped_poly = (x.polynomial * y.polynomial) * (x.polynomial + y.polynomial)

    plain = dropped_poly.evaluate_interval(domain)
    split = dropped_poly.evaluate_interval_split(domain, 2)

    assert split.width().item() <= plain.width().item() + 1e-12


def test_taylor_model_mul_breakdown_records_split():
    domain = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    x = TaylorModel.variable(0, domain, order=1, truncation_range_split=2)
    y = TaylorModel.variable(1, domain, order=1, truncation_range_split=2)

    breakdown = taylor_model_mul_breakdown(x * y, x + y, order=1)

    assert breakdown["truncation_range_split"] == 2
