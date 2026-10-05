"""New small-input tensor/cache/tail gate, callable on CPU or preloaded CUDA.

No controller, ODE integration, old checker, extension loader, or content digest
is used. GPU use requires the caller's existing no-JIT prepared runtime.
"""


def check(torch, se, binding, *, device="cpu", metadata_backend=None):
    from flowstar_gpu import monomials, polynomial, support as sp, ode_compiler, graphing
    before = dict(binding.counters)
    rows = []
    graph_captures = 0
    for order, batch, intervals, indirect in ((3, 1, False, False), (4, 3, False, False),
                                             (4, 3, True, False), (4, 3, True, True)):
        if order == 4 and metadata_backend is not None:
            eng = metadata_backend.MetadataEngine(2, 4, 0.02, device)
        else:
            tables = monomials.build_tables(2, order).to(device)
            eng = sp.SparseEngine(tables, polynomial.build_step_tables(tables, 0.02), device)
        sup = sp.make_support(2, order, False, (0, 1, 2, 3))
        rhs = (["sin(x1*x2)+cos(x1*x2)", "cos(x1+x2)+sin(x1+x2)"] if indirect else
               ["sin(x1)+cos(x1)+sin(x1)", "cos(x2)+sin(x2)+cos(x2)"])
        code = ode_compiler.compile_ode(rhs, ["x1", "x2"], order=order)
        spec = se.specialize(code, (sup, sup), order, sup, eng)
        tabs = se.elem.elem_tables(order, str(torch.device(device)))
        cache = graphing.GraphCache(device) if torch.device(device).type == "cuda" else None
        if cache is not None and not cache.enabled:
            raise RuntimeError("GPU trig-power gate requires preloaded graph kernels")
        case_before = dict(binding.counters)
        saved_graph_outputs = None
        for variant in (0, 1):
            # A strided input and changing centers expose value-identity and
            # call-lifetime mistakes; no old saved numerical input is replayed.
            storage = torch.zeros((batch, 2, 8), dtype=torch.float64, device=device)
            point = storage[:, :, ::2]
            point[:, 0, 0] = -0.31 + variant * 0.037 + torch.arange(batch, device=device) * 0.009
            point[:, 1, 0] = 0.23 - variant * 0.021
            point[:, 0, 1], point[:, 1, 2] = 0.016, -0.014
            point[:, :, 3] = 0.002 + variant * 0.001
            x = torch.stack((point - 1e-8, point + 2e-8), -1) if intervals else point
            rem = torch.empty((batch, 2, 2), dtype=torch.float64, device=device)
            rem[:, :, 0], rem[:, :, 1] = -2e-7, 3e-7
            snapshots, versions = [v.clone() for v in (x, rem)], [v._version for v in (x, rem)]

            def execute(fn, a, b):
                bad = torch.zeros(a.shape[0], dtype=torch.bool, device=a.device)
                output = fn(spec, code, a, b, eng, 1e-6, tabs, bad_out=bad,
                            interval_coefficients=intervals)
                return tuple(output) + (bad,)

            expected = execute(binding.original_exec, x, rem)
            eager = execute(binding.candidate_exec, x, rem)
            outputs = [("eager", eager)]
            if cache is not None:
                actual = cache.run(("new_trig_power_gate", order, batch, intervals, indirect),
                                   lambda a, b: execute(binding.candidate_exec, a, b),
                                   [x, rem], keepalive=(spec, code, eng, tabs))
                outputs.append(("graph", actual))
                if saved_graph_outputs is not None:
                    if not all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                               for a, b in zip(*saved_graph_outputs)):
                        raise RuntimeError("a later graph replay changed previously owned output copies")
                owned = tuple(v.clone() for v in actual)
                saved_graph_outputs = (owned, tuple(v.clone() for v in owned))
            for dispatch, actual in outputs:
                for index, (old, new) in enumerate(zip(expected, actual)):
                    if (old.shape != new.shape or old.dtype != new.dtype
                            or not torch.equal(old.contiguous().view(torch.uint8),
                                               new.contiguous().view(torch.uint8))):
                        raise RuntimeError(f"trig power {dispatch} differs at output {index}")
            for value, snapshot, version in zip((x, rem), snapshots, versions):
                if (value._version != version or not torch.equal(value.contiguous().view(torch.uint8),
                                                                 snapshot.contiguous().view(torch.uint8))):
                    raise RuntimeError("trig candidate changed caller input")
            rows.append(dict(order=order, batch=batch, interval_coefficients=intervals,
                             indirect_input=indirect, variant=variant,
                             compared_fields=["coefficients", "remainder", "range_cache", "strict_tails", "bad_mask"],
                             eager_byte_equal=True, graph_byte_equal=cache is not None,
                             input_bytes_unchanged=True))
        delta = {name: binding.counters[name] - case_before[name] for name in case_before}
        if indirect:
            if delta["reused_powers"] != 0 or delta["ineligible_calls"] <= 0:
                raise RuntimeError("non-direct input reused a power chain")
        elif delta["reused_powers"] <= 0 or delta["saved_power_chains"] <= 0:
            raise RuntimeError("direct sin/cos pairs did not consume the new reuse path")
        if cache is not None:
            torch.cuda.synchronize(device)
            graph_captures += cache.captures
            cache._segs.clear()
    delta = {name: binding.counters[name] - before[name] for name in before}
    if delta["failed_execs"] or not delta["reused_powers"]:
        raise RuntimeError("trig power gate did not qualify reuse")
    return dict(status="PASSED_NEW_TRIG_POWER_GATE", torch_version=torch.__version__, device=device,
                cases=rows, counter_delta=delta, cuda_graph_captures=graph_captures,
                scope="new small direct/indirect point/interval inputs, changed-value replay; no trajectory qualification",
                no_jit=True, digest_operations=0, old_checker_runs=0)
