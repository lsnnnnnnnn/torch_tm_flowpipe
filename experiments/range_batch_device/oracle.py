"""Independent exact rational oracle. Never imports the implementation's powers."""
from fractions import Fraction as Q
import math


def multiply(a, b):
    v = [x*y for x in a for y in b]
    return min(v), max(v)


def power(a, b, n):
    a, b = Q(a), Q(b)
    if n == 0:
        return Q(1), Q(1)
    if n % 2:
        return a**n, b**n
    return (Q(0) if a <= 0 <= b else min(abs(a), abs(b))**n, max(abs(a), abs(b))**n)


def enclose(lo, hi, expected):
    assert math.isfinite(float(lo)) and math.isfinite(float(hi))
    assert Q(float(lo)) <= expected[0] <= expected[1] <= Q(float(hi)), (float(lo).hex(), float(hi).hex(), expected)


def check(request, result, *, require_terms=True, require_powers=True):
    assert result.ok, result
    domains = [(Q(float(a)), Q(float(b))) for a,b in zip(request.domain_lo, request.domain_hi)]
    powers = {(v,p): power(*domains[v], p) for e in request.exponents for v,p in enumerate(e) if p}
    if request.kind == "normal":
        for v in request.state_variables:
            domains[v] = (Q(-1), Q(1))
    for v,p,a,b,*_ in result.powers:
        enclose(a, b, powers[v,p])
    required = {(v,p) for v,p in powers if request.kind != "normal" or v not in request.state_variables}
    if require_powers and request.step_powers is None:
        assert {(v,p) for v,p,*_ in result.powers} == required
    terms_checked = 0
    for output in range(request.coefficients_lo.shape[0]):
        total = [Q(0), Q(0)]
        for i,e in enumerate(request.exponents):
            expected = Q(float(request.coefficients_lo[output,i])), Q(float(request.coefficients_hi[output,i]))
            for v,p in enumerate(e):
                if p and (request.kind != "normal" or v not in request.state_variables):
                    factor = powers[v,p]
                    if request.step_powers is not None and v == request.time_variable and p in request.step_powers:
                        factor = tuple(Q(float(x)) for x in (request.step_powers[p].lo, request.step_powers[p].hi))
                        assert factor[0] <= powers[v,p][0] <= powers[v,p][1] <= factor[1]
                    expected = multiply(expected, factor)
            if request.kind == "normal":
                factor = (Q(-1) if any(e[v]%2 for v in request.state_variables)
                    else Q(0) if any(e[v] for v in request.state_variables) else Q(1), Q(1))
                expected = multiply(expected, factor)
            if require_terms:
                enclose(result.terms_lo[output,i], result.terms_hi[output,i], expected)
            total[0] += expected[0]; total[1] += expected[1]
            terms_checked += 1
        enclose(result.lo[output], result.hi[output], total)
    return {"powers": len(required), "terms": terms_checked, "outputs": len(result.lo)}
