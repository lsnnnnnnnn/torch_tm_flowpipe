"""B=1 accepted two-round refinement in one graph; all other paths stay original.

Only two successful, non-disjoint proofs commit. A failed precondition, map or
intersection delegates to the original refinement, including its diagnostics
and exceptions. This does not replace failed-lane or recentered validation.
"""

import inspect
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


def _two_rounds(torch, wv, next_down, code, x, initial, original, current,
                eligible, plan, eng, cutoff):
    """Pure tensor body; each map is the unchanged production interval map."""
    proposal = torch.stack((
        torch.maximum(original[..., 0], next_down(current[..., 0] * 1.01 - 1e-12)),
        torch.minimum(original[..., 1], wv.next_up(current[..., 1] * 1.01 + 1e-12)),
    ), -1)
    valid = eligible & torch.isfinite(current).flatten(1).all(1)
    valid &= wv.iv.contains(original, current).all(1)
    valid &= ((current[..., 0] <= 0) & (current[..., 1] >= 0)).all(1)
    valid &= torch.isfinite(x).flatten(1).all(1) & torch.isfinite(initial).flatten(1).all(1)
    valid &= torch.isfinite(proposal).flatten(1).all(1)
    valid &= ((proposal[..., 0] <= 0) & (proposal[..., 1] >= 0)).all(1)
    expected = torch.zeros_like(x[:, :, plan["zero_positions"]])
    expected[:, :, plan["initial_positions"]] = initial
    valid &= (x[:, :, plan["zero_positions"]] == expected).flatten(1).all(1)
    coefficients = torch.zeros((1, x.shape[1], plan["sup"].size), dtype=x.dtype, device=x.device)
    coefficients[:, :, plan["input_positions"]] = x
    result = current.clone()
    flags = [valid.all()]
    for _ in range(2):
        image, bad = wv._map(code, coefficients, x, proposal, plan, eng, cutoff)
        passed = valid & wv.iv.contains(proposal, image).all(1) & ~bad
        zero = torch.zeros_like(image[..., 0])
        uniform = torch.stack((torch.minimum(image[..., 0], zero),
                               torch.maximum(image[..., 1], zero)), -1)
        intersection = torch.stack((torch.maximum(result[..., 0], uniform[..., 0]),
                                    torch.minimum(result[..., 1], uniform[..., 1])), -1)
        disjoint = passed & (intersection[..., 0] > intersection[..., 1]).any(1)
        commit = passed & ~disjoint
        result = torch.where(commit[:, None, None], intersection, result)
        # An inactive second evaluation repeats the original proposal; it never
        # supplies an accepted output or receives a failed image as a new bound.
        proposal = torch.where(commit[:, None, None], uniform, proposal)
        valid = commit
        flags.extend((passed.all(), ~disjoint.any()))
    return result, torch.stack(flags)


def _finish(result, flags, fallback):
    """One packed host read; the fast statistics are fixed only on full success."""
    if all(flags.detach().cpu().tolist()):
        record = dict(eligible=1, attempted=1, recovered=1, evaluations=1,
                      initial_mismatch=0, max_attempts=1)
        return result.clone(), dict(eligible=1, rounds=[record.copy(), record.copy()]), True
    value, info = fallback()
    return value, info, False


def install(torch, wv, se, *, check_first=1):
    if check_first not in (0, 1):
        raise ValueError("check_first must be zero or one")
    if torch.__version__.split("+")[0] != "2.5.1":
        raise RuntimeError("production fused-single adapter requires PyTorch 2.5.1")
    for module, name in ((wv, "flowstar_gpu.weighted_validation"),
                         (se, "flowstar_gpu.sparse_exec")):
        if (not isinstance(module, ModuleType) or module.__name__ != name
                or sys.modules.get(name) is not module
                or not Path(module.__file__).is_file()):
            raise RuntimeError(f"expected preloaded source module {name}")
    raw_refine, raw_evaluate = wv.refine_accepted, wv._evaluate_map
    raw_map, raw_plan = wv._map, wv._plan
    for function, name in ((raw_refine, "refine_accepted"), (raw_evaluate, "_evaluate_map"),
                           (raw_map, "_map"), (raw_plan, "_plan")):
        if inspect.getfile(function) != str(Path(wv.__file__)) or function.__name__ != name:
            raise RuntimeError("fused-single requires pristine weighted functions")
    from flowstar_gpu.graphing import GraphCache, WARMUP
    from flowstar_gpu.rounding import next_down
    owner, active, checked = None, False, False
    counters = dict(refine_calls=0, fast_calls=0, fallback_calls=0,
                    graph_map_calls=0, replay_map_calls=0, warmup_map_calls=0,
                    graph_captures=0, candidate_exceptions=0, reference_checks=0,
                    reference_map_evaluations=0)

    def refine(code, x, sup, initial, initial_sup, var_sups, original, current,
               eligible, eng, cutoff, *, rounds=2, chunk_size=512, use_graph=True, trace=None):
        nonlocal owner, active, checked
        counters["refine_calls"] += 1

        def fallback():
            counters["fallback_calls"] += 1
            return raw_refine(code, x, sup, initial, initial_sup, var_sups, original,
                              current, eligible, eng, cutoff, rounds=rounds,
                              chunk_size=chunk_size, use_graph=use_graph, trace=trace)

        values = (x, initial, original, current, eligible)
        if (trace is not None or not use_graph or rounds != 2 or chunk_size != 512
                or x.shape[0] != 1 or not x.is_cuda
                or any(not value.is_contiguous() for value in values)
                or any(value.dtype != torch.float64 for value in values[:-1])
                or eligible.dtype != torch.bool):
            return fallback()
        if active:
            raise RuntimeError("fused-single refinement is not reentrant")
        if owner is None:
            if getattr(eng, "_weighted_fused_single_graphs", None) is not None:
                raise RuntimeError("fused-single refuses a previously captured graph")
            owner = eng
        if eng is not owner:
            raise RuntimeError("fused-single is scoped to one candidate engine")
        active = True
        try:
            try:
                plan = wv._plan(code, sup, initial_sup, var_sups, eng, cutoff)
                cache = getattr(eng, "_weighted_fused_single_graphs", None)
                if cache is None:
                    cache = eng._weighted_fused_single_graphs = GraphCache(str(x.device))
                if not cache.enabled:
                    raise RuntimeError("preloaded CUDA graph kernels unavailable")
                key = (se._code_serial(code), id(plan), tuple(value.shape for value in values),
                       x.dtype, cutoff)
                if key not in cache._segs and cache._segs:
                    cache._segs.pop(next(iter(cache._segs)))

                def body(a, b, c, d, e):
                    return _two_rounds(torch, wv, next_down, code, a, b, c, d, e,
                                       plan, eng, cutoff)

                captures = cache.captures
                value, flags = cache.run(key, body, list(values), keepalive=(code, plan))
                new_captures = cache.captures - captures
                counters["graph_captures"] += new_captures
                counters["replay_map_calls"] += 2
                counters["warmup_map_calls"] += 2 * WARMUP * new_captures
                counters["graph_map_calls"] += 2 + 2 * WARMUP * new_captures
            except Exception as error:
                counters["candidate_exceptions"] += 1
                counters["last_candidate_error"] = f"{type(error).__name__}: {error}"
                return fallback()
            value, info, committed = _finish(value, flags, fallback)
            if committed:
                if check_first and not checked:
                    # A new-input diagnostic, not a rerun of an old experiment.
                    reference, expected = raw_refine(
                        code, x, sup, initial, initial_sup, var_sups, original, current,
                        eligible, eng, cutoff, rounds=rounds, chunk_size=chunk_size,
                        use_graph=use_graph, trace=trace)
                    counters["reference_checks"] += 1
                    counters["reference_map_evaluations"] += sum(r["evaluations"] for r in expected["rounds"])
                    if (not torch.equal(value.contiguous().view(torch.uint8),
                                        reference.contiguous().view(torch.uint8)) or info != expected):
                        raise RuntimeError("fused-single first successful result/statistics differ from original refinement")
                    checked = True
                counters["fast_calls"] += 1
            return value, info
        finally:
            active = False

    def restore():
        if (active or wv.refine_accepted is not refine or wv._evaluate_map is not raw_evaluate
                or wv._map is not raw_map or wv._plan is not raw_plan):
            raise RuntimeError("fused-single binding no longer owned; refusing restore")
        wv.refine_accepted = raw_refine

    receipt = dict(policy="single_lane_two_round_weighted_graph_success_only",
                   source_path=str(Path(__file__)), weighted_path=str(Path(wv.__file__)),
                   sparse_exec_path=str(Path(se.__file__)), B=1, rounds=2,
                   check_first=check_first, failed_validation_unchanged=True,
                   recentered_validation_unchanged=True, trace_uses_original=True,
                   fallback="original refine_accepted, including default 512-row map padding",
                   graph_map_calls_definition="two maps per successful replay plus its eager warmup maps; capture records are not executions; failed captures and reference/fallback maps excluded",
                   timing_qualification="includes graph warmup/capture and optional one-time original refinement byte/statistics comparison",
                   qualification="new implementation candidate; CPU control semantics only until a separate GPU qualification",
                   no_jit=True, no_digest_operations=True)
    wv.refine_accepted = refine
    return SimpleNamespace(receipt=receipt, counters=counters, restore=restore)
