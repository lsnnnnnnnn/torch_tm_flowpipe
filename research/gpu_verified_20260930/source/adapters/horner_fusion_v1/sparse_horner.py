"""Experimental strict spatial Horner composition; no engine files modified.

The tree follows native Flow* Polynomial::toHornerForm: remove one factor
of the first variable with positive degree, recurse, multiply, then add.
Every retained point arithmetic error is outward enclosed and charged once.
"""
import torch

from . import interval as iv
from . import support as sp


def _tree(support, eng):
    cache = getattr(eng, "_horner_tree_cache", None)
    if cache is None:
        cache = {}
        eng._horner_tree_cache = cache
    if support in cache:
        return cache[support]
    rows = [(tuple(map(int, e)), j) for j, e in enumerate(sp._exps_for(support))]

    def build(terms):
        constant, groups = None, {}
        for e, j in terms:
            if not any(e):
                constant = j
                continue
            v = next(i for i, p in enumerate(e) if p)
            reduced = list(e)
            reduced[v] -= 1
            groups.setdefault(v, []).append((tuple(reduced), j))
        return constant, tuple((v, build(group)) for v, group in sorted(groups.items()))

    tree = build(rows)
    cache[support] = tree
    return tree


def compose_horner(a_coeffs, a_rem, g_coeffs, g_rem, sup_a, sup_g,
                   order, cutoff_threshold, strict, sched, eng):
    """Drop-in compose_s experiment, restricted to the strict spatial path."""
    if not strict or not sup_a.spatial or not sup_g.spatial:
        raise ValueError('Horner prototype requires strict spatial operands')
    batch, n, _ = a_coeffs.shape
    edge = getattr(eng, "horner_edge_kernel", None) if a_coeffs.is_cuda else None
    if edge is not None and not iv._kern(a_rem, g_rem):
        edge = None
    zero_sup = sp.make_support(sup_a.n, sup_a.k, True, (0,))
    zero_rem = torch.zeros(batch, n, 2, dtype=a_coeffs.dtype, device=a_coeffs.device)
    g_range = sp.range_spatial_s(g_coeffs, eng, sup_g)

    def evaluate(node):
        constant, children = node
        if constant is None:
            coeff = torch.zeros(batch, n, 1, dtype=a_coeffs.dtype, device=a_coeffs.device)
        else:
            coeff = a_coeffs[..., constant:constant + 1]
        remainder, support = zero_rem, zero_sup
        for variable, child in children:
            pc, pr, ps = evaluate(child)
            gc = g_coeffs[:, variable:variable + 1, :].expand(-1, n, -1)
            gr = g_rem[:, variable:variable + 1, :]
            p_range = sp.range_spatial_s(pc, eng, ps)
            pair = eng.pair(ps, sup_g)
            product, coefficient_error = sp.mul_point_s_with_roundoff(pc, gc, pair, eng)
            kept, tail, kept_sup = sp.ctrunc_spatial_s(product, eng, pair.sup_out, order)
            kept, dropped = sp.cutoff_spatial_s(kept, eng, kept_sup, cutoff_threshold)
            point_sum, union = sp.add_aligned(coeff, support, kept, kept_sup, eng, interval=False)
            exact_sum, _ = sp.add_aligned(iv.from_point(coeff), support,
                                          iv.from_point(kept), kept_sup, eng, interval=True)
            add_error = iv.sub(exact_sum, iv.from_point(point_sum))
            addition_error = sp.range_spatial_iv_s(add_error, eng, union)
            operands = [remainder, pr, g_rem, g_range, p_range, tail,
                        dropped, coefficient_error, addition_error]
            compatible = edge is not None and all(
                t.is_contiguous() and t.dtype == a_coeffs.dtype
                and t.device == a_coeffs.device and t.ndim == 3
                and t.shape == (batch, n, 2) for t in operands
            )
            if compatible:
                remainder = edge(operands, variable)
            else:
                rem = iv.mul(pr, gr)
                rem = iv.add(rem, iv.mul(g_range[:, variable:variable + 1, :], pr))
                rem = iv.add(rem, iv.mul(p_range, gr))
                rem = iv.add(rem, iv.add(tail, dropped))
                rem = iv.add(rem, coefficient_error)
                rem = iv.add(rem, addition_error)
                remainder = iv.add(remainder, rem)
            coeff, support = point_sum, union
        return coeff, remainder, support

    coeff, rem, support = evaluate(_tree(sup_a, eng))
    return coeff, iv.add(rem, a_rem), support
