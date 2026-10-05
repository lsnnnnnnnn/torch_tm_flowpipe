"""Fixed-batch, success-only fusion of the original two weighted maps.

NAV uses B=25/rows=32; QUAD uses B=1024/rows=256. Each chunk replays one
two-round graph. Public outputs and flags own their storage before graph reuse.
Only an all-lanes, both-rounds success commits; otherwise the original control
flow, using the already-qualified fixed scratch size, supplies the result.
"""

import inspect
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


CONFIGURATIONS = ((25, 32), (1024, 256), (640, 512))


def _admit_plan(wv, se, plan, batch):
    """Keep the qualified QUAD metadata factory; never unwrap or replace it."""
    path = Path(wv.__file__)
    if inspect.getfile(plan) == str(path) and plan.__name__ == "_plan":
        return dict(kind="original_weighted_plan", source_path=str(path))
    backend = sys.modules.get("fullbatch_p3_metadata_backend")
    if (batch != 1024 or not isinstance(backend, ModuleType)
            or backend.__name__ != "fullbatch_p3_metadata_backend"
            or not Path(backend.__file__).is_file()
            or Path(backend.__file__).name != "metadata_backend.py"
            or getattr(backend, "weighted_plan", None) is not plan
            or plan.__globals__ is not backend.__dict__
            or plan.__name__ != "weighted_plan"
            or inspect.getfile(plan) != str(Path(backend.__file__))
            or getattr(backend, "wv", None) is not wv
            or getattr(backend, "se", None) is not se
            or getattr(backend, "_installed", None) is not True):
        raise RuntimeError("weighted plan is neither pristine nor the installed QUAD metadata backend")
    original = getattr(backend, "_original_weighted", None)
    if (not inspect.isfunction(original) or original.__name__ != "_plan"
            or inspect.getfile(original) != str(path)
            or original.__globals__ is not wv.__dict__):
        raise RuntimeError("QUAD metadata backend did not retain the original weighted plan")
    return dict(kind="existing_qualified_quad_metadata_plan", source_path=str(Path(backend.__file__)),
                registered_module=backend.__name__, original_plan_path=str(path),
                installed_function_retained=True, metadata_plan_replaced=False)


def _pad(torch, value, rows):
    lanes = value.shape[0]
    if not 0 < lanes <= rows:
        raise ValueError("invalid fixed weighted chunk")
    return (torch.cat((value, value[-1:].expand(rows - lanes, *value.shape[1:])), 0)
            if lanes < rows else value)


def _two_rounds(torch, wv, next_down, code, x, initial, original, current,
                eligible, plan, eng, cutoff):
    """Same lane arithmetic as the B=1 adapter, without any host decisions."""
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
    coefficients = torch.zeros((x.shape[0], x.shape[1], plan["sup"].size),
                               dtype=x.dtype, device=x.device)
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
        proposal = torch.where(commit[:, None, None], uniform, proposal)
        valid = commit
        flags.extend((passed.all(), ~disjoint.any()))
    return result, torch.stack(flags)


def _run_chunks(torch, values, rows, replay):
    """Copy graph-static outputs BEFORE the next replay of the same key."""
    result = torch.empty_like(values[3])
    flags = []
    for offset in range(0, values[0].shape[0], rows):
        inputs = [_pad(torch, value[offset:offset + rows], rows) for value in values]
        value, chunk_flags = replay(inputs)
        lanes = result[offset:offset + rows].shape[0]
        result[offset:offset + rows].copy_(value[:lanes])
        flags.append(chunk_flags.clone())
    return result, torch.cat(flags)


def _finish(result, flags, fallback):
    if all(flags.detach().cpu().tolist()):
        lanes = result.shape[0]
        record = dict(eligible=lanes, attempted=lanes, recovered=lanes,
                      evaluations=lanes, initial_mismatch=0, max_attempts=1)
        return result, dict(eligible=lanes, rounds=[record.copy(), record.copy()]), True
    value, info = fallback()
    return value, info, False


def install(torch, wv, se, *, batch, rows, check_first=1):
    if (batch, rows) not in CONFIGURATIONS or check_first not in (0, 1):
        raise ValueError("requires (batch, rows)=(25,32), (1024,256), or (640,512), check_first=0/1")
    if torch.__version__.split("+")[0] != "2.5.1":
        raise RuntimeError("production fused-batch adapter requires PyTorch 2.5.1")
    for module, name in ((wv, "flowstar_gpu.weighted_validation"),
                         (se, "flowstar_gpu.sparse_exec")):
        if (not isinstance(module, ModuleType) or module.__name__ != name
                or sys.modules.get(name) is not module
                or not Path(module.__file__).is_file()):
            raise RuntimeError(f"expected preloaded source module {name}")
    raw_refine, raw_evaluate = wv.refine_accepted, wv._evaluate_map
    raw_map, raw_plan = wv._map, wv._plan
    for function, name in ((raw_refine, "refine_accepted"), (raw_evaluate, "_evaluate_map"),
                           (raw_map, "_map")):
        if inspect.getfile(function) != str(Path(wv.__file__)) or function.__name__ != name:
            raise RuntimeError(f"fused-batch requires pristine weighted numeric function {name}")
    plan_admission = _admit_plan(wv, se, raw_plan, batch)
    from flowstar_gpu.graphing import GraphCache, WARMUP
    from flowstar_gpu.rounding import next_down
    owner, active, original_mode, checked = None, False, None, False
    counters = dict(refine_calls=0, fast_calls=0, fallback_calls=0, chunk_size=rows,
                    graph_replays=0, replay_map_calls=0, warmup_map_calls=0,
                    graph_map_calls=0, graph_captures=0, candidate_exceptions=0,
                    reference_checks=0, reference_map_evaluations=0,
                    reference_graph_map_calls=0, fallback_graph_map_calls=0,
                    original_graph_pool_releases=0, replay_target_rows=0, replay_padded_rows=0)

    def evaluate(code, coefficients, point, candidate, plan, eng, cutoff, use_graph):
        # Failed-lane/recentered validation outside this wrapper remains original.
        if original_mode is None or not use_graph or not candidate.is_cuda:
            return raw_evaluate(code, coefficients, point, candidate, plan, eng, cutoff, use_graph)
        lanes = candidate.shape[0]
        if not 0 < lanes <= rows or coefficients.shape[0] != lanes or point.shape[0] != lanes:
            raise ValueError("original fixed-scratch input batches differ")
        inputs = [_pad(torch, value, rows) for value in (coefficients, point, candidate)]
        cache = getattr(eng, "_weighted_fused_batch_original_graphs", None)
        if cache is None:
            cache = eng._weighted_fused_batch_original_graphs = GraphCache(str(candidate.device))
        key = (se._code_serial(code), id(plan), tuple(value.shape for value in inputs),
               candidate.dtype, cutoff)
        if key not in cache._segs and cache._segs:
            cache._segs.pop(next(iter(cache._segs)))
        def body(a, b, c):
            return raw_map(code, a, b, c, plan, eng, cutoff)
        output = cache.run(key, body, inputs, keepalive=(code, plan))
        counters[original_mode + "_graph_map_calls"] += 1
        return tuple(value[:lanes].clone() for value in output)

    def release_original_graph(eng):
        cache = getattr(eng, "_weighted_fused_batch_original_graphs", None)
        if cache is not None:
            # Original refinement returns owned tensors. Finish their copies
            # before releasing this short-lived, adapter-owned reference pool.
            torch.cuda.synchronize(cache.device)
            del eng._weighted_fused_batch_original_graphs
            counters["original_graph_pool_releases"] += 1

    def refine(code, x, sup, initial, initial_sup, var_sups, original, current, eligible,
               eng, cutoff, *, rounds=2, chunk_size=512, use_graph=True, trace=None):
        nonlocal owner, active, original_mode, checked
        if active:
            raise RuntimeError("fused-batch refinement is not reentrant")
        counters["refine_calls"] += 1
        values = (x, initial, original, current, eligible)
        fixed_scope = use_graph and x.is_cuda and x.shape[0] == batch and chunk_size == 512

        def original_call(mode):
            nonlocal original_mode
            original_mode = mode if fixed_scope else None
            try:
                return raw_refine(code, x, sup, initial, initial_sup, var_sups, original,
                                  current, eligible, eng, cutoff, rounds=rounds,
                                  chunk_size=rows if fixed_scope else chunk_size,
                                  use_graph=use_graph, trace=trace)
            finally:
                original_mode = None
                if fixed_scope:
                    release_original_graph(eng)

        def fallback():
            counters["fallback_calls"] += 1
            return original_call("fallback")

        if not fixed_scope:
            return fallback()
        if owner is None:
            if any(getattr(eng, name, None) is not None for name in
                   ("_weighted_graphs", "_weighted_fused_batch_original_graphs")):
                raise RuntimeError("fused-batch requires a fresh weighted graph owner")
            owner = eng
        if owner is not eng:
            raise RuntimeError("fused-batch is scoped to one candidate engine")
        active = True
        try:
            if (trace is not None or rounds != 2
                    or any(value.shape[0] != batch or value.device != x.device for value in values)
                    or any(not value.is_contiguous() for value in values)
                    or any(value.dtype != torch.float64 for value in values[:-1])
                    or eligible.dtype != torch.bool):
                return fallback()
            try:
                plan = raw_plan(code, sup, initial_sup, var_sups, eng, cutoff)
                # Preserve the frozen engine's SR-growth reclamation hook,
                # which clears _weighted_graphs before transient SR buffers.
                cache = getattr(eng, "_weighted_graphs", None)
                if cache is None:
                    cache = eng._weighted_graphs = GraphCache(str(x.device))
                if not cache.enabled:
                    raise RuntimeError("preloaded CUDA graph kernels unavailable")
                key = ("fused_two_rounds", se._code_serial(code), id(plan),
                       tuple((rows, *value.shape[1:]) for value in values), x.dtype, cutoff)
                if key not in cache._segs and cache._segs:
                    cache._segs.pop(next(iter(cache._segs)))
                def body(a, b, c, d, e):
                    return _two_rounds(torch, wv, next_down, code, a, b, c, d, e, plan, eng, cutoff)
                def replay(inputs):
                    captures = cache.captures
                    output = cache.run(key, body, inputs, keepalive=(code, plan))
                    new_captures = cache.captures - captures
                    counters["graph_captures"] += new_captures
                    counters["graph_replays"] += 1
                    counters["replay_map_calls"] += 2
                    counters["warmup_map_calls"] += 2 * WARMUP * new_captures
                    counters["graph_map_calls"] += 2 + 2 * WARMUP * new_captures
                    return output
                value, flags = _run_chunks(torch, values, rows, replay)
                counters["replay_target_rows"] += 2 * batch
                counters["replay_padded_rows"] += 2 * (((batch + rows - 1) // rows) * rows - batch)
            except Exception as error:
                counters["candidate_exceptions"] += 1
                counters["last_candidate_error"] = f"{type(error).__name__}: {error}"
                return fallback()
            value, info, committed = _finish(value, flags, fallback)
            if committed:
                if check_first and not checked:
                    reference, expected = original_call("reference")
                    counters["reference_checks"] += 1
                    counters["reference_map_evaluations"] += sum(r["evaluations"] for r in expected["rounds"])
                    if (not torch.equal(value.contiguous().view(torch.uint8),
                                        reference.contiguous().view(torch.uint8)) or info != expected):
                        raise RuntimeError("fused-batch first result/statistics differ from original refinement")
                    checked = True
                counters["fast_calls"] += 1
            return value, info
        finally:
            active = False

    def restore():
        if (active or original_mode is not None or wv.refine_accepted is not refine
                or wv._evaluate_map is not evaluate or wv._map is not raw_map or wv._plan is not raw_plan):
            raise RuntimeError("fused-batch binding no longer owned; refusing restore")
        wv.refine_accepted, wv._evaluate_map = raw_refine, raw_evaluate

    receipt = dict(policy=f"batch{batch}_two_round_weighted_graph_fixed{rows}_success_only",
                   implementation_revision="v3_add_nav640_fixed512", plan_admission=plan_admission,
                   source_path=str(Path(__file__)), weighted_path=str(Path(wv.__file__)),
                   sparse_exec_path=str(Path(se.__file__)), B=batch, fixed_graph_rows=rows,
                   rounds=2, check_first=check_first, failed_validation_unchanged=True,
                   recentered_validation_unchanged=True, trace_uses_original=True,
                   fallback=f"original refine_accepted with fixed{rows} scratch policy",
                   commit="all lanes pass both unchanged interval maps; one packed host flag read",
                   fused_graph_owner="eng._weighted_graphs; existing SR-growth reclamation retained",
                   original_graph_owner="short-lived eng._weighted_fused_batch_original_graphs; released after each reference/fallback call",
                   graph_map_calls_definition="2 per fused replay plus 2*WARMUP per new capture; reference/fallback separate",
                   timing_qualification="includes capture and one live-input original byte/statistics check",
                   qualification="new candidate; CPU semantics only until a separate GPU qualification",
                   no_jit=True, no_digest_operations=True)
    wv.refine_accepted, wv._evaluate_map = refine, evaluate
    return SimpleNamespace(receipt=receipt, policy=receipt["policy"], source_path=receipt["source_path"],
                           weighted_path=receipt["weighted_path"], sparse_exec_path=receipt["sparse_exec_path"],
                           counters=counters, restore=restore)
