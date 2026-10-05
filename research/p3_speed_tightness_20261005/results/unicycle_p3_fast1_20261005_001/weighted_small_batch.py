"""Keep original two-round accepted refinement, with small fixed graph scratch.

Install on pristine weighted_validation before constructing the candidate's
engine. Only CUDA graph calls inside refine_accepted change their batch shape.
No numerical formula, failed-lane validation, SR or observer is replaced.
"""

import inspect
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


ALLOWED_ROWS = (1, 16, 32)


def pad_rows(torch, value, rows):
    lanes = value.shape[0]
    if rows not in ALLOWED_ROWS or not 0 < lanes <= rows:
        raise ValueError("small weighted scratch requires 0 < batch <= rows in (1, 16, 32)")
    return (torch.cat((value, value[-1:].expand(rows - lanes, *value.shape[1:])), dim=0)
            if lanes < rows else value)


def install(torch, wv, se, *, rows=32):
    if rows not in ALLOWED_ROWS:
        raise ValueError("small weighted scratch rows must be 1, 16 or 32")
    for module, name in ((wv, "flowstar_gpu.weighted_validation"),
                         (se, "flowstar_gpu.sparse_exec")):
        if (not isinstance(module, ModuleType) or module.__name__ != name
                or sys.modules.get(name) is not module
                or not Path(module.__file__).is_file()):
            raise RuntimeError(f"expected preloaded source module {name}")
    raw_evaluate, raw_refine = wv._evaluate_map, wv.refine_accepted
    if (inspect.getfile(raw_evaluate) != str(Path(wv.__file__))
            or inspect.getfile(raw_refine) != str(Path(wv.__file__))
            or raw_evaluate.__name__ != "_evaluate_map"
            or raw_refine.__name__ != "refine_accepted"):
        raise RuntimeError("small weighted scratch requires the pristine weighted functions")
    active, owner = False, None
    counters = dict(refine_calls=0, graph_map_calls=0, target_rows=0,
                    padded_rows=0, chunk_size=rows)

    def evaluate(code, coefficients, point, candidate, plan, eng, cutoff, use_graph):
        if not active or not use_graph or not candidate.is_cuda:
            return raw_evaluate(code, coefficients, point, candidate, plan, eng, cutoff, use_graph)
        lanes = candidate.shape[0]
        if not 0 < lanes <= rows or coefficients.shape[0] != lanes or point.shape[0] != lanes:
            raise ValueError("small weighted graph input batches differ or exceed selected rows")
        coefficients, point, candidate = (pad_rows(torch, value, rows) for value in
                                          (coefficients, point, candidate))
        from flowstar_gpu.graphing import GraphCache
        cache = getattr(eng, "_weighted_graphs", None)
        if cache is None:
            cache = eng._weighted_graphs = GraphCache(str(candidate.device))
        key = (se._code_serial(code), id(plan), tuple(coefficients.shape), tuple(point.shape),
               tuple(candidate.shape), candidate.dtype, cutoff)
        if key not in cache._segs and len(cache._segs) >= 1:
            cache._segs.pop(next(iter(cache._segs)))
        def fn(a, b, c):
            return wv._map(code, a, b, c, plan, eng, cutoff)
        output = cache.run(key, fn, [coefficients, point, candidate], keepalive=(code, plan))
        counters["graph_map_calls"] += 1
        counters["target_rows"] += lanes
        counters["padded_rows"] += rows - lanes
        return tuple(value[:lanes].clone() for value in output)

    def refine(code, x, sup, initial, initial_sup, var_sups, original, current, eligible,
               eng, cutoff, *, rounds=2, chunk_size=512, use_graph=True, trace=None):
        nonlocal active, owner
        if not use_graph or not x.is_cuda:
            return raw_refine(code, x, sup, initial, initial_sup, var_sups, original, current,
                              eligible, eng, cutoff, rounds=rounds, chunk_size=chunk_size,
                              use_graph=use_graph, trace=trace)
        if active or rounds != 2 or chunk_size != 512 or not 0 < x.shape[0] <= rows:
            raise RuntimeError("small weighted scratch requires a nonnested default two-round batch <= rows")
        if owner is None:
            if getattr(eng, "_weighted_graphs", None) is not None:
                raise RuntimeError("small weighted scratch refuses a previously captured weighted graph")
            owner = eng
        if owner is not eng:
            raise RuntimeError("small weighted scratch is scoped to one fresh candidate engine")
        active = True
        try:
            result = raw_refine(code, x, sup, initial, initial_sup, var_sups, original, current,
                                eligible, eng, cutoff, rounds=rounds, chunk_size=rows,
                                use_graph=use_graph, trace=trace)
            counters["refine_calls"] += 1
            return result
        finally:
            active = False

    def restore():
        if active or wv._evaluate_map is not evaluate or wv.refine_accepted is not refine:
            raise RuntimeError("small weighted binding no longer owned; refusing restore")
        wv._evaluate_map, wv.refine_accepted = raw_evaluate, raw_refine

    receipt = dict(policy=f"original_early_weighted_two_rounds_fixed{rows}_graph_scratch", source_path=str(Path(__file__)),
                   weighted_path=str(Path(wv.__file__)), sparse_exec_path=str(Path(se.__file__)),
                   rounds=2, maximum_input_rows=rows, fixed_graph_rows=rows,
                   scope="accepted CUDA graph refinement only; original formulas and owned output clones",
                   qualification="new implementation candidate; saved-output equivalence pending",
                   no_jit=True, no_digest_operations=True)
    wv._evaluate_map, wv.refine_accepted = evaluate, refine
    return SimpleNamespace(receipt=receipt, counters=counters, restore=restore)
