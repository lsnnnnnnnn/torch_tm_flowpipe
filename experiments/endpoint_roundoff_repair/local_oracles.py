"""Independent rational checks for the repaired production substitution APIs."""
from fractions import Fraction as F
from itertools import product
import argparse
import json
from pathlib import Path
import subprocess

import torch

from torch_tm_flowpipe import Interval, Polynomial, TaylorModel, TMVector
from torch_tm_flowpipe.batched_dense_tm import sparse_tmvector_to_dense, dense_to_sparse_tmvector


def exact_coefficients(poly, variable, h):
    result = {}
    for exp, coefficient in poly.terms.items():
        reduced = exp[:variable] + exp[variable + 1:]
        result[reduced] = result.get(reduced, F(0)) + F(float(coefficient)) * F(h) ** exp[variable]
    return result


def mul(a, b):
    values = [x*y for x in a for y in b]
    return min(values), max(values)


def exact_error_range(exact, point, domain):
    """Rational natural range of the exact coefficient-error polynomial."""
    lo, hi = F(0), F(0)
    for exp in set(exact) | set(point.terms):
        coefficient = exact.get(exp, F(0)) - F(float(point.terms.get(exp, 0.)))
        term = coefficient, coefficient
        for power, variable in zip(exp, domain):
            lower, upper = F(float(variable.lo)), F(float(variable.hi))
            values = [lower**power, upper**power]
            if power and power % 2 == 0 and lower <= 0 <= upper:
                values.append(F(0))
            term = mul(term, (min(values), max(values)))
        lo, hi = lo+term[0], hi+term[1]
    return lo, hi


def assert_model_contains(source, output, h):
    exact = exact_coefficients(source.polynomial, source.n_vars-1, h)
    lo, hi = exact_error_range(exact, output.polynomial, output.domain)
    lo += F(float(source.remainder.lo))
    hi += F(float(source.remainder.hi))
    assert F(float(output.remainder.lo)) <= lo, (output.remainder, lo)
    assert hi <= F(float(output.remainder.hi)), (output.remainder, hi)
    return [str(lo), str(hi)]


def case_model(spatial, order, h, remainder, *, component=0):
    domain = [Interval(*b) for b in [(-2., .5), (.25, 1.5), (-.75, 2.)][:spatial]]
    domain.append(Interval(min(-.125, h), max(.125, h)))
    z = (0,)*spatial
    terms = {z+(0,): -1., z+(1,): 100.}
    # All time powers up to the actual order, including merges and cancellation.
    for k in range(2, order+1):
        terms[z+(k,)] = (-1.)**k * (1.25 + component) * 10.**k
    for index in range(spatial):
        beta = tuple(int(i == index) for i in range(spatial))
        terms[beta+(0,)] = -(index+1.)
        terms[beta+(1,)] = 100.*(index+1.)
        terms[beta+(order-1,)] = (-1.)**index * 2.**40
    if component:
        terms = {e: -0.375*c for e,c in terms.items()}
    return TaylorModel(Polynomial(terms, spatial+1), Interval(*remainder), domain, order=order)


CASES = [
    (spatial, order, h, cutoff, remainder)
    for spatial, order, h in [
        (1, 4, 0.), (2, 4, .5), (3, 4, .01),
        (1, 6, .02), (2, 6, -.02), (3, 6, -.125),
    ]
    for cutoff, remainder in [(None, (0., 0.)), (1e-10, (-1e-5, 2e-5))]
]


def check_case(case):
    spatial, order, h, cutoff, remainder = case
    source = TMVector(case_model(spatial, order, h, remainder, component=i) for i in range(2))
    dense = sparse_tmvector_to_dense(source, order=order)
    dense_before = dense.clone()
    dpoly, dlo, dhi, qlo, qhi = dense.poly.substitute_const_and_drop_with_roundoff(
        spatial, h, dense.domain_lo, dense.domain_hi)
    sparse = source.substitute_const(spatial, h).drop_variable(spatial).apply_cutoff(cutoff)
    internal = dense.endpoint(spatial, h)
    ledger_lo, ledger_hi = internal.ledger.total(internal.rem_lo)
    assert torch.equal(ledger_lo, internal.rem_lo) and torch.equal(ledger_hi, internal.rem_hi)
    internal = internal.apply_cutoff(cutoff)
    cutoff_ledger_lo, cutoff_ledger_hi = internal.ledger.total(internal.rem_lo)
    assert torch.equal(cutoff_ledger_lo, internal.rem_lo) and torch.equal(cutoff_ledger_hi, internal.rem_hi)
    converted = dense_to_sparse_tmvector(internal)
    # The conversion must carry the full mathematical payload even when the
    # ledger is consolidated into initial_remainder on the return trip.
    roundtrip = dense_to_sparse_tmvector(sparse_tmvector_to_dense(converted, order=order))
    rows = []
    for i, model in enumerate(source):
        poly, error, coefficients = model.polynomial.substitute_const_with_roundoff(spatial, h, model.domain)
        exact = exact_coefficients(model.polynomial, spatial, h)
        for beta, q in exact.items():
            bounds = coefficients[beta+(0,)]
            assert F(float(bounds.lo)) <= q <= F(float(bounds.hi))
            term = dpoly.basis.term_index(beta)
            assert F(float(qlo[0, i, term])) <= q <= F(float(qhi[0, i, term]))
        exact_error = exact_error_range(exact, poly.drop_variable(spatial), model.domain[:-1])
        assert F(float(error.lo)) <= exact_error[0] <= exact_error[1] <= F(float(error.hi))
        paths = {label: assert_model_contains(model, result[i], h) for label, result in (
            ('sparse', sparse), ('dense', converted), ('dense_sparse_dense', roundtrip))}
        rows.append({'component':i, 'coefficient_count':len(exact), 'paths_exact_error_plus_R':paths,
                     'sparse_error_hex':[float(error.lo).hex(), float(error.hi).hex()],
                     'dense_error_hex':[float(dlo[0,i]).hex(),float(dhi[0,i]).hex()]})
    again = dense.endpoint(spatial, h)
    assert torch.equal(dense.poly.coeffs, dense_before.poly.coeffs)
    assert torch.equal(dense.rem_lo, dense_before.rem_lo) and torch.equal(dense.rem_hi, dense_before.rem_hi)
    assert torch.equal(again.rem_lo, dense.endpoint(spatial, h).rem_lo)
    return {'spatial':spatial, 'order':order, 'h_hex':h.hex(), 'h_exact':str(F(h)),
            'cutoff':cutoff, 'remainder':remainder, 'rows':rows, 'passed':True}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    git=lambda *a: subprocess.check_output(['git','-C',str(root),*a],text=True).strip()
    assert not git('status','--porcelain'), 'formal oracle requires a clean committed tree'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from experiments.our_solver_performance.reference_endpoint import ENTRIES, endpoint_witness
    witnesses=[]
    for entry in ENTRIES:
        _, endpoint, exact=endpoint_witness(entry)
        contains=[F(float(b.lo)) <= exact <= F(float(b.hi)) for b in endpoint.range_box()]
        assert all(contains)
        witnesses.append({'entry':entry,'exact':str(exact),'contains':contains,
                          'full_bounds_hex':[[float(b.lo).hex(),float(b.hi).hex()] for b in endpoint.range_box()]})
    data={'source_sha':git('rev-parse','HEAD'),'source_clean':True,
          'oracle':'independent Fraction coefficient evaluation and exact error-polynomial natural range',
          'original_witnesses':witnesses,'general_cases':[check_case(c) for c in CASES]}
    with args.output.open('x') as f:
        json.dump(data,f,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps({'source_sha':data['source_sha'],'original_witnesses_passed':2,'general_cases_passed':len(CASES)}))


if __name__ == '__main__':
    main()
