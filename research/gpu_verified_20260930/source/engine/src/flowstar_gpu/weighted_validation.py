"""Optional strict time-weighted Picard validation for previously failed lanes.

For an unchanged polynomial p with p(0)=x0, validate e(t) in (t/h)R by
symbolically factoring t/h from the integrated defect. No interval division
by t is used. Every candidate is re-evaluated; only a full inclusion recovers
a lane. The returned uniform remainder includes zero. These caches must not
be used by the ordinary constant-remainder refinement loop.
"""
import torch
from . import elementary as elem, interval as iv, support as sp
from .rounding import next_up


def _plan(code, sup, initial_sup, var_sups, eng, cutoff):
    from . import sparse_exec as se
    cache = se._eng_cache(eng, "_weighted_plans")
    key = (se._code_serial(code), sup, initial_sup, var_sups, cutoff)
    if key in cache:
        return cache[key]
    if eng.step.lanes != 0 or not eng.step.delta > 0:
        raise ValueError("weighted validation currently requires a positive common step")
    n, k = eng.tables.n, eng.tables.k
    exponents = [tuple(v) for v in eng.tables.exponents.cpu().tolist()]
    ids = {v: i for i, v in enumerate(exponents)}
    time_id = ids[(1,) + (0,) * n]
    full_sup = sp.make_support(n, k, False, tuple(sorted(set(sup.ids) | {time_id})))
    weighted_vars = tuple(sp.make_support(n, k, False, tuple(sorted(set(v.ids) | {time_id}))) for v in var_sups)
    spec = se.specialize(code, weighted_vars, code.order, full_sup, eng)
    device = eng.tables.exponents.device
    index = lambda v: torch.tensor(v, dtype=torch.long, device=device)
    input_positions = index([full_sup.ids.index(i) for i in sup.ids])
    zero_ids = [i for i in sup.ids if exponents[i][0] == 0]
    if any(exponents[i][0] != 0 or i not in zero_ids for i in initial_sup.ids):
        raise ValueError("initial polynomial must have only supported time-zero terms")
    zero_positions = index([sup.ids.index(i) for i in zero_ids])
    initial_positions = index([zero_ids.index(i) for i in initial_sup.ids])
    shifted_ids, shifted_positions = [], []
    for j, i in enumerate(sup.ids):
        powers = list(exponents[i])
        if powers[0]:
            powers[0] -= 1
            shifted_ids.append(ids[tuple(powers)])
            shifted_positions.append(j)
    field_ids = spec.sup_out_union.ids
    residual_sup = sp.make_support(n, k, False, tuple(sorted(set(field_ids) | set(shifted_ids))))
    h = torch.tensor(eng.step.delta, dtype=torch.float64, device=device)
    h_iv = iv.from_point(h)
    divisors = torch.tensor([exponents[i][0] + 1 for i in field_ids], dtype=torch.float64, device=device)
    weights, bad_div = iv.div(h_iv, iv.from_point(divisors))
    if bool(bad_div.any()):
        raise ValueError("invalid positive integration divisor")
    weights = torch.where((divisors == 1)[:, None], h_iv, weights)
    cache[key] = dict(spec=spec, sup=full_sup, input_positions=input_positions,
                      time_position=full_sup.ids.index(time_id), zero_positions=zero_positions,
                      initial_positions=initial_positions, shifted_positions=index(shifted_positions),
                      shifted_targets=index([residual_sup.ids.index(i) for i in shifted_ids]),
                      field_targets=index([residual_sup.ids.index(i) for i in field_ids]),
                      residual_size=residual_sup.size, factors=eng.factor(residual_sup),
                      weights=weights, h_iv=h_iv,
                      tabs=elem.elem_tables(code.order, str(device)) if code.has_elem else None)
    return cache[key]


def _map(code, coefficients, point_coefficients, candidate, plan, eng, cutoff):
    from . import sparse_exec as se
    x_iv = iv.from_point(coefficients)
    slope, bad_div = iv.div(candidate, plan["h_iv"])
    j = plan["time_position"]
    x_iv[:, :, j] = iv.add(x_iv[:, :, j], slope)
    bad = torch.zeros(candidate.shape[0], dtype=torch.bool, device=candidate.device)
    bad |= bad_div.any()  # The positive common h is shared by every lane.
    field, tail, ranges, strict_tails = se.exec_valid_s(
        plan["spec"], code, x_iv, torch.zeros_like(candidate), eng, cutoff,
        plan["tabs"], bad_out=bad, interval_coefficients=True,
    )
    polynomial = torch.zeros((coefficients.shape[0], coefficients.shape[1], plan["residual_size"], 2),
                             dtype=coefficients.dtype, device=coefficients.device)
    polynomial[:, :, plan["field_targets"]] = iv.mul(field, plan["weights"])
    subtract = iv.mul(iv.from_point(point_coefficients[:, :, plan["shifted_positions"]]), plan["h_iv"])
    targets = plan["shifted_targets"]
    polynomial[:, :, targets] = iv.sub(polynomial[:, :, targets], subtract)
    residual = iv.sum(iv.mul(polynomial, plan["factors"]), dim=-1)
    image = iv.add(residual, iv.mul(tail, plan["h_iv"]))
    for value in (field, tail, ranges, strict_tails, image):
        bad = bad | ~torch.isfinite(value).flatten(1).all(1)
    return image, bad


def _evaluate_map(code, coefficients, point, candidate, plan, eng, cutoff, use_graph):
    if not use_graph or not candidate.is_cuda:
        return _map(code, coefficients, point, candidate, plan, eng, cutoff)
    lanes = candidate.shape[0]
    if 0 < lanes < 512:
        # Rows are independent. Reuse the full-chunk graph for a short tail,
        # then discard padded outputs; no padded row enters acceptance.
        def pad(value):
            return torch.cat((value, value[-1:].expand(512 - lanes, *value.shape[1:])), dim=0)
        coefficients, point, candidate = (pad(value) for value in (coefficients, point, candidate))
    from .graphing import GraphCache
    from .sparse_exec import _code_serial
    cache = getattr(eng, "_weighted_graphs", None)
    if cache is None:
        cache = eng._weighted_graphs = GraphCache(str(candidate.device))
    key = (_code_serial(code), id(plan), tuple(coefficients.shape), tuple(point.shape),
           tuple(candidate.shape), candidate.dtype, cutoff)
    # Private graph pools are bounded; old public outputs already own copies.
    if key not in cache._segs and len(cache._segs) >= 1:
        cache._segs.pop(next(iter(cache._segs)))
    def evaluate(a, b, c):
        return _map(code, a, b, c, plan, eng, cutoff)
    output = cache.run(key, evaluate, [coefficients, point, candidate], keepalive=(code, plan))
    return tuple(value[:lanes].clone() for value in output)


def validate_failed(code, x, sup, initial, initial_sup, var_sups, original,
                    eligible, eng, cutoff, *, max_attempts=4, chunk_size=32, trace=None, use_graph=False):
    """Return recovered mask and owned uniform remainders; inputs are read-only."""
    if max_attempts < 0 or chunk_size < 1:
        raise ValueError("invalid weighted validation limits")
    recovered = torch.zeros_like(eligible)
    emitted = torch.zeros_like(original)
    valid = eligible & torch.isfinite(x).flatten(1).all(1)
    valid &= torch.isfinite(initial).flatten(1).all(1) & torch.isfinite(original).flatten(1).all(1)
    valid &= ((original[..., 0] <= 0) & (original[..., 1] >= 0)).all(1)
    statistics = dict(eligible=int(eligible.sum()), attempted=0, recovered=0,
                      evaluations=0, initial_mismatch=0, max_attempts=max_attempts)
    lanes = torch.where(valid)[0]
    if lanes.numel() == 0 or max_attempts == 0:
        return recovered, emitted, statistics
    plan = _plan(code, sup, initial_sup, var_sups, eng, cutoff)
    for offset in range(0, lanes.numel(), chunk_size):
        selected = lanes[offset:offset + chunk_size]
        point = x.index_select(0, selected)
        expected = torch.zeros_like(point[:, :, plan["zero_positions"]])
        expected[:, :, plan["initial_positions"]] = initial.index_select(0, selected)
        same_initial = (point[:, :, plan["zero_positions"]] == expected).flatten(1).all(1)
        statistics["initial_mismatch"] += int((~same_initial).sum())
        selected = selected[same_initial]
        point = point[same_initial]
        if selected.numel() == 0:
            continue
        statistics["attempted"] += selected.numel()
        coefficients = torch.zeros((len(selected), x.shape[1], plan["sup"].size), dtype=x.dtype, device=x.device)
        coefficients[:, :, plan["input_positions"]] = point
        cap = original.index_select(0, selected)
        candidate = cap.clone()
        active = torch.ones(len(selected), dtype=torch.bool, device=x.device)
        for attempt in range(max_attempts):
            image, bad = _evaluate_map(code, coefficients, point, candidate, plan, eng, cutoff, use_graph)
            statistics["evaluations"] += int(active.sum())
            dimensions = iv.contains(candidate, image)
            passed = active & dimensions.all(1) & ~bad
            if trace is not None:
                for local in torch.where(active)[0].tolist():
                    trace(dict(event="weighted_self_map", lane=int(selected[local]), attempt=attempt,
                               accepted=bool(passed[local]), bad=bool(bad[local]),
                               input_remainder_vector=candidate[local].detach().cpu().tolist(),
                               proposal_interval=image[local].detach().cpu().tolist(),
                               subset_by_component=dimensions[local].detach().cpu().tolist()))
            # e(t) is scaled by t/h, so its ordinary tube bound must include 0.
            zero = torch.zeros_like(image[..., 0])
            uniform = torch.stack((torch.minimum(image[..., 0], zero), torch.maximum(image[..., 1], zero)), -1)
            recovered[selected[passed]] = True
            emitted[selected[passed]] = uniform[passed]
            active &= ~passed & ~bad
            if not bool(active.any()):
                break
            radius = next_up(torch.maximum(image[..., 0].abs(), image[..., 1].abs()) * 1.01)
            proposal = torch.stack((torch.maximum(cap[..., 0], -radius), torch.minimum(cap[..., 1], radius)), -1)
            # A failed image is only a proposal; the next full map must pass.
            candidate = torch.where(active[:, None, None], proposal, candidate)
    statistics["recovered"] = int(recovered.sum())
    return recovered, emitted, statistics


def validate_recentered(code, x, sup, initial, initial_sup, var_sups, original,
                        eligible, eng, cutoff, *, max_attempts=4, chunk_size=32, trace=None):
    """Validate a new approximation p+c*t/h; return its coefficients AND bound.

    The configured cap is unchanged but is relative to the new polynomial.
    This is not a proof for the old p. Time-zero coefficients stay identical;
    every new point coefficient (including RN construction error) is validated
    afresh. Unsupported layouts and failed lanes return the original polynomial.
    """
    if max_attempts < 0 or chunk_size < 1:
        raise ValueError("invalid recentered validation limits")
    recovered = torch.zeros_like(eligible)
    emitted = torch.zeros_like(original)
    output = x.clone()
    statistics = dict(eligible=int(eligible.sum()), attempted=0, recovered=0,
                      evaluations=0, initial_mismatch=0, unsupported_layout=False)
    valid = eligible & torch.isfinite(x).flatten(1).all(1)
    valid &= torch.isfinite(initial).flatten(1).all(1) & torch.isfinite(original).flatten(1).all(1)
    valid &= ((original[..., 0] <= 0) & (original[..., 1] >= 0)).all(1)
    lanes = torch.where(valid)[0]
    if lanes.numel() == 0 or max_attempts == 0:
        return recovered, output, emitted, statistics
    plan = _plan(code, sup, initial_sup, var_sups, eng, cutoff)
    if plan["sup"] != sup:
        statistics["unsupported_layout"] = True
        return recovered, output, emitted, statistics
    for offset in range(0, lanes.numel(), chunk_size):
        selected = lanes[offset:offset + chunk_size]
        point = x.index_select(0, selected)
        cap = original.index_select(0, selected)
        image, bad = _map(code, point, point, torch.zeros_like(cap), plan, eng, cutoff)
        statistics["evaluations"] += len(selected)
        # This is only a point approximation proposal. Its rounded coefficients
        # receive a complete new interval proof below, with no reused map cache.
        point[:, :, plan["time_position"]] += image.mean(-1) / eng.step.delta
        local_eligible = ~bad & torch.isfinite(point).flatten(1).all(1)
        def local_trace(event):
            trace(dict(event, event="recentered_self_map", lane=int(selected[event["lane"]])))
        ok, rem, info = validate_failed(
            code, point, sup, initial.index_select(0, selected), initial_sup,
            var_sups, cap, local_eligible, eng, cutoff, max_attempts=max_attempts,
            chunk_size=chunk_size, trace=local_trace if trace is not None else None,
        )
        for name in ("attempted", "evaluations", "initial_mismatch"):
            statistics[name] += info[name]
        recovered[selected] = ok
        output[selected[ok]] = point[ok]
        emitted[selected[ok]] = rem[ok]
    statistics["recovered"] = int(recovered.sum())
    return recovered, output, emitted, statistics



def refine_accepted(code, x, sup, initial, initial_sup, var_sups, original,
                    current, eligible, eng, cutoff, *, rounds=2, chunk_size=512,
                    use_graph=True, trace=None):
    """Intersect only freshly validated bounds for the same p/initial/domain.

    A proposal may be wider than current but never exceeds the configured
    original cap. A failed proposal leaves the existing valid bound intact.
    """
    if rounds < 0 or chunk_size < 1:
        raise ValueError("invalid early weighted refinement limits")
    from .rounding import next_down
    valid = eligible & torch.isfinite(current).flatten(1).all(1)
    valid &= iv.contains(original, current).all(1)
    valid &= ((current[..., 0] <= 0) & (current[..., 1] >= 0)).all(1)
    result = current.clone()
    proposal = torch.stack((
        torch.maximum(original[..., 0], next_down(current[..., 0] * 1.01 - 1e-12)),
        torch.minimum(original[..., 1], next_up(current[..., 1] * 1.01 + 1e-12)),
    ), -1)
    records = []
    for iteration in range(rounds):
        if not bool(valid.any()):
            break
        def record(event):
            trace(dict(event, event="early_weighted_self_map", refinement_round=iteration))
        accepted, new, info = validate_failed(
            code, x, sup, initial, initial_sup, var_sups, proposal, valid, eng, cutoff,
            max_attempts=1, chunk_size=chunk_size, use_graph=use_graph,
            trace=record if trace is not None else None,
        )
        intersection = torch.stack((torch.maximum(result[..., 0], new[..., 0]),
                                    torch.minimum(result[..., 1], new[..., 1])), -1)
        if bool((accepted & (intersection[..., 0] > intersection[..., 1]).any(1)).any()):
            raise FloatingPointError("two validated same-polynomial bounds are disjoint")
        result = torch.where(accepted[:, None, None], intersection, result)
        proposal = torch.where(accepted[:, None, None], new, proposal)
        valid = accepted
        records.append(info)
    return result, dict(eligible=int(eligible.sum()), rounds=records)
