"""Isolated real-input 128-row CUDA-graph diagnostics, not a solver benchmark.

Reuse frozen admissions and numerical graphs. Warm outside the profiler, then
measure one cache hit per owner; output/input bytes and capture count must agree.
Explicit completion synchronization at both profile boundaries is diagnostic
only. Never use these durations as whole-trajectory or pure speedup evidence.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, collections, dataclasses, gc, hashlib, importlib.util, json, math, sys, time, traceback, weakref

HERE = Path(__file__).parent
PINS = {
    'check_valid_chunk128.py': '30a49eea8ff44be2c8bf7d3f41c60f4ffbfc09a5cecd84108b0e77d4c999fd28',
    'check_weighted_chunk128.py': '2e81fdd59833fb395521836ccc831e4822f6d98842dc8bfab32bc50b6bc46aa0',
    'valid_chunk128.py': '6b97a9ed19b8433b79be15f97381241bc9a79c071b90fa4a81f12c4d216dcc40',
    'weighted_chunk128.py': 'ae343c546e2253896142f88c39e1df5e43561498aaf7540e722a1bb6d66d409b',
    'nncs_watchdog_gpu14.py': '4298689de9ccf8a97c0007334dcdbe6aedeb246b8acead81a4b1873941a56350',
}
ORDINARY_RESULT_SHA = '053947f483678ed087b849f18c90f34cc827215c0dea884541ae072d978307cf'
ALLOCATOR_BYTES, GPU_GUARD_BYTES, RSS_GUARD_BYTES = 27*2**30//2, 14*2**30, 23*2**30//2


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load(filename):
    path = HERE/filename
    assert sha(path) == PINS[filename], filename
    spec = importlib.util.spec_from_file_location('kernel_profile_'+path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def kernel_summary(trace):
    """CUPTI is admitted by real positive-duration CUDA kernel events only."""
    grouped = {}; devices = set(); streams = set(); memcpy_count = 0
    for event in trace['traceEvents']:
        if event.get('cat') == 'gpu_memcpy':
            memcpy_count += 1
        if event.get('cat') != 'kernel' or event.get('ph') != 'X':
            continue
        duration = event.get('dur')
        assert type(duration) in (float, int) and math.isfinite(duration) and duration >= 0
        if duration == 0:
            continue
        device = event.get('args', {}).get('device')
        assert type(device) is int and device >= 0, 'Kernel event has no CUDA device'
        devices.add(device); streams.add(str(event.get('args', {}).get('stream')))
        name = event['name']; row = grouped.setdefault(name, dict(calls=0, total_us=0., max_us=0.))
        row['calls'] += 1; row['total_us'] += duration; row['max_us'] = max(row['max_us'], duration)
    rows = [dict(name=name, **row) for name, row in grouped.items()]
    rows.sort(key=lambda x: -x['total_us'])
    return dict(status='passed_positive_cuda_kernel_trace' if rows else 'unsupported_no_positive_cuda_kernel_events',
                kernels=rows, kernel_calls=sum(x['calls'] for x in rows),
                kernel_duration_sum_us=sum(x['total_us'] for x in rows), devices=sorted(devices),
                streams=sorted(streams), gpu_memcpy_events=memcpy_count,
                scope='Single profiled graph-cache hit. Duration sum counts kernels only, excludes copies and gaps; not uninstrumented performance.')


def single_replay(torch, checker, call, inputs, cache, key, output_count, name, output):
    assert cache.enabled and key in cache._segs and cache.captures == 1
    assert cache._segs[key][0][0].shape[0] == 128
    before = {k: v.detach().cpu().clone().contiguous() for k, v in inputs.items()}
    versions = {k: v._version for k, v in inputs.items()}
    captures, hits = cache.captures, cache.hits
    baseline = tuple(t.detach().to('cpu', copy=True).contiguous() for t in call())
    assert len(baseline) == output_count and cache.captures == captures and cache.hits == hits+1
    assert all(v._version == versions[k] and checker.same(torch, v, before[k]) for k, v in inputs.items())
    activities = torch.profiler.supported_activities()
    precheck = dict(kineto_available=torch.profiler.kineto_available(), supported_activities=sorted(str(x) for x in activities))
    if not precheck['kineto_available'] or torch.profiler.ProfilerActivity.CUDA not in activities:
        return dict(name=name, status='unsupported_profiler_cuda_unavailable', precheck=precheck,
                    numerical_profile_comparison_performed=False, CUDA_kernel_trace=False)
    # These are additional diagnostic window boundaries, not solver timing.
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False, profile_memory=False, with_stack=False) as profile:
        with torch.profiler.record_function('isolated_'+name+'_one_cache_hit'):
            actual = call()
        torch.cuda.synchronize()
    after = tuple(t.detach().to('cpu', copy=True).contiguous() for t in actual)
    assert len(after) == output_count and all(checker.same(torch, x, y) for x, y in zip(baseline, after))
    assert all(v._version == versions[k] and checker.same(torch, v, before[k]) for k, v in inputs.items())
    assert cache.captures == captures and cache.hits == hits+2
    trace_path = output/(name+'_trace.json')
    profile.export_chrome_trace(str(trace_path))
    summary = kernel_summary(read(trace_path))
    cpu = [dict(name=row.key, calls=row.count, self_cpu_us=row.self_cpu_time_total,
                cpu_total_us=row.cpu_time_total) for row in profile.key_averages()]
    cpu.sort(key=lambda x: -x['self_cpu_us'])
    write(output/(name+'_CPU_EVENTS.json'), cpu)
    write(output/(name+'_KERNELS.json'), summary)
    return dict(name=name, status=summary['status'], precheck=precheck, CUDA_kernel_trace=bool(summary['kernel_calls']),
                profiled_cache_hits=1, baseline_cache_hits=1, new_captures=0, explicit_profile_boundary_syncs=2,
                all_output_bytes_equal=True, all_input_bytes_and_versions_unchanged=True,
                inputs=checker.fingerprints(torch, before, 'inputs'), outputs=checker.fingerprints(torch, after, 'outputs'),
                captures=cache.captures, hits_before=hits, hits_after=cache.hits,
                kernel_calls=summary['kernel_calls'], kernel_duration_sum_us=summary['kernel_duration_sum_us'],
                trace_sha256=sha(trace_path), kernels_sha256=sha(output/(name+'_KERNELS.json')),
                CPU_events_sha256=sha(output/(name+'_CPU_EVENTS.json')),
                scope='Profiled once after warmup; profiler may add its own synchronization/overhead. Original arithmetic and graph body unchanged.')


def ordinary_owner(ctx, checker, ps, output, reference):
    # Exact reconstruction from the frozen check_valid_chunk128.arm prefix.
    torch, d, se = ctx.torch, ctx.d, ctx.c.se
    from flowstar_gpu import support as sp, elementary as elem
    tables = d.build_tables(16, 3).to('cuda:0')
    work = sp.SparseEngine(tables, d.poly.build_step_tables(tables, .005), 'cuda:0')
    work.horner_edge_kernel = ctx.edge.horner_edge
    eng = se.validation_engine_for_order(work, 4)
    names = [v['name'] for v in ctx.cfg['initial_set']]
    code2 = d.compile_ode(ctx.cfg['dynamics_expressions'], names, order=2)
    code = se.validation_code_for_order(code2, 4, eng); tabs = elem.elem_tables(4, 'cuda:0')
    assert checker.source.canonical_sha(dataclasses.asdict(code)) == ps[0]['code_sha256'] == ps[1]['code_sha256']
    def support(v):
        s = sp.make_support(v['n'], v['k'], v['spatial'], tuple(v['ids']))
        assert checker.exact(torch, checker.source.support_summary(s, eng), v)
        return s
    p = ps[0]; v = p['specialization']
    spec = se.specialize(code, tuple(support(s) for s in v['variable_supports']), v['k'], support(v['x_support']), eng,
                        **{k: v[k] for k in ['validated_full_leaves', 'validated_direct_leaves', 'validated_linear_leaves']})
    assert checker.exact(torch, checker.source.spec_summary(spec, eng), p['actual_spec_layout'])
    for key, value in p['elementary_tables'].items():
        assert checker.exact(torch, getattr(tabs, key), value)
    raw = se._graphed_valid; adapter = load('valid_chunk128.py'); binding = adapter.install(torch, se)
    try:
        ins = {name: p[name].to('cuda:0', copy=True) for name in ['x', 'x_rem', 'bad']}
        versions = {k: v._version for k, v in ins.items()}
        value = se._graphed_valid(spec, code, ins['x'], ins['x_rem'], ins['bad'], eng, 1e-6, tabs)
        cpu = tuple(t.detach().to('cpu', copy=True).contiguous() for t in value)
        assert checker.fingerprints(torch, cpu, 'outputs') == reference['arms'][1]['cases'][0]['outputs']
        assert all(v._version == versions[k] and checker.same(torch, v, p[k]) for k, v in ins.items())
        cache = eng._graphs; key = ('vd', id(spec))
        assert eng._vtape[id(spec)] is None and cache.captures == 1 and cache.hits == 8
        del ins, value, cpu
        # Real step2 first128 lanes; same already-warmed spec/cache, new input.
        ins = {name: ps[1][name][:128].to('cuda:0', copy=True) for name in ['x', 'x_rem', 'bad']}
        def call():
            return raw(spec, code, ins['x'], ins['x_rem'], ins['bad'], eng, 1e-6, tabs)
        result = single_replay(torch, checker, call, ins, cache, key, 5, 'ordinary_step2_lanes0_127', output)
        result.update(source_attempted_step=2, source_lanes=list(range(128)),
                      full1024_step1_five_outputs_equal_frozen_CUDA_receipt=True,
                      whole_advance_run=False, interpreter_capacity_fallback_is_graph=True)
        return result, (weakref.ref(work), weakref.ref(eng), weakref.ref(cache))
    finally:
        binding.restore(); assert se._graphed_valid is raw


def weighted_owner(ctx, checker, ordinary_checker, p, output, reference_dir):
    # Exact support/code reconstruction from check_weighted_chunk128.arm.
    torch, d = ctx.torch, ctx.d
    from flowstar_gpu import support as sp, sparse_exec as se, weighted_validation as wv, monomials
    tables = d.build_tables(16, 3).to('cuda:0'); work = sp.SparseEngine(tables, d.poly.build_step_tables(tables, .005), 'cuda:0')
    work.horner_edge_kernel = ctx.edge.horner_edge
    assert type(work) is sp.SparseEngine and isinstance(work.tables, monomials.MonomialTables)
    assert work.tables.k == 3 and work.tables.T2 == 100947
    val = se.validation_engine_for_order(work, 4); assert isinstance(val, ctx.hybrid.backend.MetadataEngine)
    names = [v['name'] for v in ctx.cfg['initial_set']]; code2 = d.compile_ode(ctx.cfg['dynamics_expressions'], names, order=2)
    code4 = se.validation_code_for_order(code2, 4, val); assert checker.code_hash(dataclasses.asdict(code4)) == p['code_sha256']
    sup = sp.make_support(16, 4, False, tuple(p['support_ids'])); isup = sp.make_support(16, 4, False, tuple(p['initial_ids']))
    variables = tuple(sp.make_support(16, 4, False, tuple(ids)) for ids in p['variable_support_ids'])
    for support, exps in [(sup, p['support_exponents']), (isup, p['initial_exponents']), *zip(variables, p['variable_exponents'])]:
        assert checker.same(torch, torch.tensor(val.exponents(support).copy(), dtype=torch.int64), exps)
    inputs = {k: p[k].to('cuda:0', copy=True) for k in checker.TENSORS}
    versions = {k: v._version for k, v in inputs.items()}
    actual = wv._evaluate_map; saved = []
    def observe(*args):
        # Observe the first actual round0/first128 map; do not invent a proposal.
        result = actual(*args)
        if not saved:
            code, coefficients, point, candidate, plan, engine, cutoff, use_graph = args
            assert engine is val and use_graph and coefficients.shape[0] == point.shape[0] == candidate.shape[0] == 128
            saved.append((code, coefficients.detach().clone(), point.detach().clone(), candidate.detach().clone(), plan, engine, cutoff))
        return result
    wv._evaluate_map = observe
    try:
        full, statistics = wv.refine_accepted(code4, inputs['x'], sup, inputs['initial'], isup, variables,
            inputs['original'], inputs['current'], inputs['eligible'], val, 1e-6,
            rounds=2, chunk_size=512, use_graph=True, trace=None)
    finally:
        wv._evaluate_map = actual
    assert all(v._version == versions[k] and checker.same(torch, v, p[k]) for k, v in inputs.items())
    receipt = read(reference_dir/'RESULT.json'); path = reference_dir/'eligible1024_chunk128/OUTPUT.pt'
    assert sha(path) == receipt['arms'][1]['files_sha256']['OUTPUT.pt']
    expected = torch.load(path, map_location='cpu', weights_only=True)
    assert checker.same(torch, full, expected['remainder']) and checker.exact(statistics, receipt['arms'][1]['statistics'])
    assert len(saved) == 1 and ctx.weighted128.counters['refine_calls'] == 1
    assert ctx.weighted128.counters['graph_map_calls'] == 16 and ctx.weighted128.counters['padded_rows'] == 0
    code, coefficients, point, candidate, plan, engine, cutoff = saved.pop()
    cache = val._weighted_graphs; assert cache.captures == 1 and cache.hits == 16 and len(cache._segs) == 1
    key = (se._code_serial(code), id(plan), tuple(coefficients.shape), tuple(point.shape), tuple(candidate.shape), candidate.dtype, cutoff)
    assert key in cache._segs
    def body(a, b, c):
        return wv._map(code, a, b, c, plan, engine, cutoff)
    def call():
        return cache.run(key, body, [coefficients, point, candidate], keepalive=(code, plan))
    result = single_replay(torch, ordinary_checker, call, dict(coefficients=coefficients, point=point, candidate=candidate),
                           cache, key, 2, 'weighted_round0_lanes0_127', output)
    result.update(source_attempted_step=1, source_refinement_round=0, source_lanes=list(range(128)),
                  full1024_two_round_remainder_and_statistics_equal_frozen_CUDA_receipt=True,
                  full_refinement_map_calls=16, full_refinement_rounds=2, whole_advance_run=False)
    return result, (weakref.ref(work), weakref.ref(val), weakref.ref(cache))


def check_local(args, vc, wc, ps, wp):
    good = dict(traceEvents=[dict(cat='kernel', ph='X', name='actual_test_kernel', dur=5., args=dict(device=0, stream=7)),
                             dict(cat='kernel', ph='X', name='actual_test_kernel', dur=3., args=dict(device=0, stream=7)),
                             dict(cat='cpu_op', ph='X', name='CPU_only', dur=100.)])
    parsed = kernel_summary(good); assert parsed['kernel_calls'] == 2 and parsed['kernel_duration_sum_us'] == 8.
    for events in [[], [good['traceEvents'][2]], [dict(good['traceEvents'][0], dur=0.)]]:
        assert kernel_summary(dict(traceEvents=events))['status'] == 'unsupported_no_positive_cuda_kernel_events'
    for event in [dict(good['traceEvents'][0], dur=-1.), dict(good['traceEvents'][0], dur=float('nan')),
                  dict(good['traceEvents'][0], args={})]:
        try:
            kernel_summary(dict(traceEvents=[event]))
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid CUDA timing accepted')
    return dict(status='passed_CPU_payload_admission_and_trace_parser', actual_source_payloads_checked=True,
                ordinary_steps=[p['attempted_step'] for p in ps], weighted_step=wp['attempted_step'],
                positive_kernel_parser_checked=True, CPU_only_and_zero_duration_rejected=True,
                invalid_duration_and_missing_device_rejected=True, CUDA_run=False, CUDA_kernel_trace=False,
                numerical_graph_run=False, scope='CPU archive admission and synthetic trace parser only; not native/CUDA/numerical qualification.')


def main(args):
    args.output.mkdir(parents=True, exist_ok=False); start = time.perf_counter(); result = dict(status='failed'); ctx = None
    try:
        for name, digest in PINS.items():
            assert sha(HERE/name) == digest, name
        vc = load('check_valid_chunk128.py'); wc = load('check_weighted_chunk128.py')
        import torch
        torch.set_num_threads(1)
        ps, ordinary_identity, ordinary_files = vc.payloads(args, torch)
        wa = SimpleNamespace(capture=args.weighted_capture, capture_audit=args.weighted_capture_audit)
        wp, weighted_audit, weighted_identity = wc.source(wa, torch)
        assert sha(args.ordinary_check/'RESULT.json') == ORDINARY_RESULT_SHA
        ordinary_reference = read(args.ordinary_check/'RESULT.json')
        assert ordinary_reference['status'] == 'passed' and ordinary_reference['device'] == 'cuda:0'
        assert ordinary_reference['script_sha256'] == PINS['check_valid_chunk128.py']
        assert ordinary_reference['all_five_outputs_all_three_cases_bytes_equal'] is True
        if args.check_local:
            result = check_local(args, vc, wc, ps, wp)
            return 0
        ctx = vc.w.initialize(args)  # All frozen source/extension/reciprocal/metadata/weighted/cache admissions.
        assert ctx.torch is torch and ctx.identity == ordinary_identity == ordinary_reference['source_identity']
        assert ctx.identity['base_P3_identity'] == weighted_identity
        assert torch.__version__ == ordinary_reference['torch_version']
        total = torch.cuda.get_device_properties(0).total_memory; assert total >= 15*2**30
        torch.cuda.set_per_process_memory_fraction(ALLOCATOR_BYTES/total, 0)
        write(args.output/'INPUT.json', dict(script_sha256=sha(__file__), source_identity=ctx.identity,
            ordinary_files_sha256=ordinary_files, weighted_payload_sha256=wc.PAYLOAD_SHA, ordinary_reference_sha256=ORDINARY_RESULT_SHA,
            resource_budget=dict(allocator_bytes=ALLOCATOR_BYTES, GPU_guard_bytes=GPU_GUARD_BYTES, RSS_guard_bytes=RSS_GUARD_BYTES),
            scope='Two sequential128 owners. Resource budget changed only after frozen admissions; no NN/SR/plant advance or source-run qualification.'))
        results = []; released = []
        for kind in ['ordinary', 'weighted']:
            if kind == 'ordinary':
                row, refs = ordinary_owner(ctx, vc, ps, args.output, ordinary_reference)
            else:
                row, refs = weighted_owner(ctx, wc, vc, wp, args.output, args.weighted_check)
            results.append(row); gc.collect(); torch.cuda.empty_cache()
            assert all(ref() is None for ref in refs), 'Owner/graph escaped its diagnostic arm'
            released.append(kind); write(args.output/(kind+'_RESULT.json'), row)
        result = dict(status='passed_isolated_kernel_profiles' if all(row['CUDA_kernel_trace'] for row in results) else 'unsupported_cuda_kernel_trace',
            arms=results, owner_release_order=released, both_owners_released=True,
            source_capture_passed=False, source_run_is_trajectory_qualification=False,
            fullbatch_qualification=False, end_to_end_strict_certificate=False, no_speedup_claim=True,
            explicit_added_profile_boundary_syncs=sum(row.get('explicit_profile_boundary_syncs', 0) for row in results),
            max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), max_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            scope='One profiled128 graph hit per actual fixed input after original warmup. Full1024 source inputs admitted, not a whole trajectory; no NN/SR/advance. Instrumented kernel ranking only.')
        return 0 if result['status'] == 'passed_isolated_kernel_profiles' else 2
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        if ctx is not None:
            ctx.postwarm.restore(); ctx.postwarm_log.close(); ctx.weighted128.restore()
            result['runtime_hooks_restored'] = True
        result.update(script_sha256=sha(__file__), source_pins=PINS, process_s=time.perf_counter()-start)
        write(args.output/'RESULT.json', result)
        print(json.dumps({k: result[k] for k in ['status', 'process_s']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--check-local', action='store_true')
    for name in ['capture', 'capture-audit', 'weighted-capture', 'weighted-capture-audit', 'ordinary-check', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ['common', 'candidate-build', 'candidate-check', 'endpoint-check', 'injection-check', 'boundary-check',
                 'host-small-check', 'host-capacity-check', 'snapshot-check', 'p3-snapshot-check', 'hybrid-cpu-check',
                 'metadata-backend', 'metadata-check', 'hybrid-check', 'weighted-check', 'weighted-checker',
                 'cache-release-check', 'cache-release-checker']:
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args(); args.mode = 'kernel_profile_fixed_inputs'; args.source40 = None
    raise SystemExit(main(args))
