"""Baseline/candidate fixed-input parity via frozen ordinary/weighted arms.

No new numerical body, profiler, NN, SR or trajectory. Candidate is installed
only after baseline owners are discarded. Counters measure Python construction
and warmup, not the kernels executed by graph replay.
"""
from pathlib import Path
from types import SimpleNamespace, ModuleType
import argparse, gc, hashlib, importlib.util, json, sys, time, traceback, weakref

HERE = Path(__file__).parent
FIXED_SHA = '3f056cc5dd3e3bdd6d1cb594402b7c3659271e97dc9600ab741ab2092c5d1b87'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path, expected):
    assert sha(path) == expected, str(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixed = load('trig_reuse_frozen_fixed_inputs', HERE/'kernel_profile_fixed_inputs.py', FIXED_SHA)


def owned_arm(ctx, invoke, all_refs):
    """Observe engine/graph ownership only; invoke the frozen arm unchanged."""
    from flowstar_gpu import graphing
    torch, se = ctx.torch, ctx.c.se
    raw_factory, raw_init = se.validation_engine_for_order, graphing.GraphCache.__init__
    refs = []
    def factory(work, order):
        eng = raw_factory(work, order)
        refs.extend([weakref.ref(work), weakref.ref(eng)])
        return eng
    def graph_init(cache, *args, **kwargs):
        raw_init(cache, *args, **kwargs); refs.append(weakref.ref(cache))
    se.validation_engine_for_order, graphing.GraphCache.__init__ = factory, graph_init
    try:
        result = invoke()
    finally:
        se.validation_engine_for_order, graphing.GraphCache.__init__ = raw_factory, raw_init
        all_refs.extend(refs)
    def CPU_only(v):
        if isinstance(v, torch.Tensor):
            assert v.device.type == 'cpu'
        elif isinstance(v, dict):
            for value in v.values(): CPU_only(value)
        elif isinstance(v, (tuple, list)):
            for value in v: CPU_only(value)
    CPU_only(result); gc.collect(); torch.cuda.empty_cache()
    assert refs and all(ref() is None for ref in refs), 'Frozen arm retained an engine/graph owner'
    return result, dict(observed_owner_references=len(refs), all_engine_and_graph_owners_released=True)


def check_owner_CPU(torch):
    """Exercise actual observer restoration/leak rejection, without CUDA."""
    class Owner:
        pass
    class Cache:
        def __init__(self):
            pass
    fake = ModuleType('flowstar_gpu'); fake.graphing = SimpleNamespace(GraphCache=Cache)
    previous = sys.modules.get('flowstar_gpu'); sys.modules['flowstar_gpu'] = fake
    factory = lambda work, order: Owner()
    ctx = SimpleNamespace(torch=SimpleNamespace(Tensor=torch.Tensor, cuda=SimpleNamespace(empty_cache=lambda: None)),
                          c=SimpleNamespace(se=SimpleNamespace(validation_engine_for_order=factory)))
    original_init = Cache.__init__; held = []; refs = []
    def invoke(leak=False, fail=False):
        work = Owner(); eng = ctx.c.se.validation_engine_for_order(work, 4); graph = Cache()
        if leak: held.extend([work, eng, graph])
        if fail: raise ValueError('actual test arm failed')
        return dict(CPU_result=True)
    try:
        result, receipt = owned_arm(ctx, invoke, refs)
        assert result['CPU_result'] and receipt['all_engine_and_graph_owners_released']
        try: owned_arm(ctx, lambda: invoke(leak=True), refs)
        except AssertionError: pass
        else: raise AssertionError('Leaked owner accepted')
        assert ctx.c.se.validation_engine_for_order is factory and Cache.__init__ is original_init
        held.clear(); gc.collect(); assert all(ref() is None for ref in refs)
        try: owned_arm(ctx, lambda: invoke(fail=True), refs)
        except ValueError as exc: assert str(exc) == 'actual test arm failed'
        else: raise AssertionError('Original arm failure swallowed')
        assert ctx.c.se.validation_engine_for_order is factory and Cache.__init__ is original_init
        gc.collect(); assert all(ref() is None for ref in refs)
        return dict(actual_owner_observer_exercised=True, leaked_owner_rejected=True,
                    observer_hooks_restored_on_original_failure=True, actual_CUDA_owner_tested=False)
    finally:
        if previous is None: sys.modules.pop('flowstar_gpu')
        else: sys.modules['flowstar_gpu'] = previous


def main(args):
    args.output.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    result = dict(status='failed'); ctx = None; candidate = None; refs = []; ctx_weighted_active = False; exit_code = 1
    try:
        for name, digest in fixed.PINS.items(): assert sha(HERE/name) == digest
        vc = fixed.load('check_valid_chunk128.py'); wc = fixed.load('check_weighted_chunk128.py')
        import torch
        torch.set_num_threads(1)
        ps, ordinary_identity, files = vc.payloads(args, torch)
        wa = SimpleNamespace(capture=args.weighted_capture, capture_audit=args.weighted_capture_audit)
        wp, audit, weighted_identity = wc.source(wa, torch)
        assert sha(args.ordinary_check/'RESULT.json') == fixed.ORDINARY_RESULT_SHA
        ordinary_reference = fixed.read(args.ordinary_check/'RESULT.json')
        if args.check_local:
            result = fixed.check_local(args, vc, wc, ps, wp)
            result.update(status='passed_CPU_harness_source_admission_only', candidate_tested=False,
                          owner_observer_CPU_check=check_owner_CPU(torch), harness_CUDA_executed=False,
                          scope='Actual archived CPU payload admission and reused parser checks only. Candidate and owner lifecycle await actual CUDA execution.')
            exit_code = 0
        else:
            assert args.candidate_sha256 and sha(args.candidate) == args.candidate_sha256
            module = load('trig_direct_reuse_candidate', args.candidate, args.candidate_sha256)
            ctx = vc.w.initialize(args); ctx_weighted_active = True
            assert ctx.torch is torch and ctx.identity == ordinary_identity == ordinary_reference['source_identity']
            assert ctx.identity['base_P3_identity'] == weighted_identity
            assert torch.__version__ == ordinary_reference['torch_version']
            # wc.arm owns its weighted128 binding; avoid wrapping it twice.
            ctx.weighted128.restore(); ctx_weighted_active = False
            total = torch.cuda.get_device_properties(0).total_memory; assert total >= 15*2**30
            torch.cuda.set_per_process_memory_fraction(fixed.ALLOCATOR_BYTES/total, 0)
            fixed.write(args.output/'INPUT.json', dict(base_identity=ctx.identity, candidate_path=str(args.candidate),
                candidate_sha256=args.candidate_sha256, frozen_harness_sha256=FIXED_SHA,
                ordinary_source_files_sha256=files, weighted_payload_sha256=wc.PAYLOAD_SHA,
                resource_budget=dict(allocator_bytes=fixed.ALLOCATOR_BYTES, GPU_guard_bytes=fixed.GPU_GUARD_BYTES,
                                     RSS_guard_bytes=fixed.RSS_GUARD_BYTES), source_capture_passed=False,
                scope='Fixed-input parity only, no new trajectory/performance qualification.'))
            valid_adapter = fixed.load('valid_chunk128.py'); weighted_adapter = fixed.load('weighted_chunk128.py')
            data = {}; receipts = {}
            for label in ['baseline', 'candidate']:
                if label == 'candidate':
                    assert all(ref() is None for ref in refs)
                    candidate = module.install(torch, ctx.c.se)
                before = dict(candidate.counters) if candidate is not None else None
                (ordinary, vr), vo = owned_arm(ctx, lambda: vc.arm(ctx, valid_adapter, ps, 128), refs)
                ordinary_plan = dict(last_plan=dict(candidate.last_plan),
                    python_counter_delta={k:v-before[k] for k,v in candidate.counters.items()}) if candidate is not None else None
                before = dict(candidate.counters) if candidate is not None else None
                directory = args.output/(label+'_weighted1024')
                wr, wo = owned_arm(ctx, lambda: wc.arm(ctx, weighted_adapter, wp, 1024, True, directory), refs)
                weighted_plan = dict(last_plan=dict(candidate.last_plan),
                    python_counter_delta={k:v-before[k] for k,v in candidate.counters.items()}) if candidate is not None else None
                data[label] = ordinary
                receipts[label] = dict(ordinary=vr, weighted=wr, owner_release=[vo, wo],
                    candidate_observations=dict(ordinary=ordinary_plan, weighted=weighted_plan))
                fixed.write(args.output/(label+'_RECEIPT.json'), receipts[label])
            for original, proposed in zip(data['baseline'], data['candidate']):
                assert len(original) == len(proposed) == 5
                assert all(vc.same(torch, a, b) for a, b in zip(original, proposed))
            assert len(data['baseline']) == len(data['candidate']) == 3
            assert receipts['baseline']['ordinary'] == receipts['candidate']['ordinary']
            left, right = (args.output/(label+'_weighted1024') for label in ['baseline', 'candidate'])
            a, b = (torch.load(path/'OUTPUT.pt', map_location='cpu', weights_only=True) for path in [left, right])
            assert a.keys() == b.keys() and all(vc.same(torch, a[key], b[key]) for key in a)
            for name in ['EVENTS.json', 'EAGER_EVENTS.json']:
                assert wc.exact(fixed.read(left/name), fixed.read(right/name))
            wa, wb = (receipts[label]['weighted'] for label in ['baseline', 'candidate'])
            for key in ['map_calls', 'statistics', 'eager_stats', 'graph', 'counters', 'eligible_lanes']:
                assert wc.exact(wa[key], wb[key]), key
            for label in receipts:
                wr = receipts[label]['weighted']
                assert wr['input_bytes_unchanged'] and wr['owned_outputs_survive_replay']
                assert wr['every_target_map_represented_in_events'] and wr['ineligible_remainders_unchanged']
                for row in receipts[label]['ordinary']['cases']:
                    assert row['input_bytes_unchanged'] and row['marked_bad_preserved'] and row['owned_previous_outputs_survive_next_replay']
            for name, digest in files.items(): assert sha(args.capture/name) == digest
            assert sha(args.weighted_capture/'WEIGHTED_INPUT.pt') == wc.PAYLOAD_SHA
            assert sha(args.candidate) == args.candidate_sha256
            result = dict(status='passed_fixed_real_input_byte_parity', candidate_sha256=args.candidate_sha256,
                candidate_policy=candidate.policy, candidate_python_counters=dict(candidate.counters),
                candidate_plan_observations=receipts['candidate']['candidate_observations'],
                counter_scope='Python eager warmup/capture construction and eager32 only; not graph replay execution counts.',
                ordinary_three_cases_all_five_outputs_bytes_equal=True, ordinary_bad_mask_and_replay_alias_gates=True,
                weighted_full1024_original_two_rounds_output_and_statistics_equal=True,
                weighted_all_map_image_bad_and_EVENTS_float_bytes_equal=True, eager32_all_outputs_events_statistics_equal=True,
                all_input_bytes_versions_unchanged=True, both_arms_all_owners_released=True,
                source_capture_passed=False, source_run_is_trajectory_qualification=False,
                fullbatch_qualification=False, end_to_end_strict_certificate=False, no_speedup_claim=True,
                receipts={label:sha(args.output/(label+'_RECEIPT.json')) for label in receipts},
                scope='Fresh baseline and candidate owners. Existing frozen checkers execute all numerical bodies. No profiler/NN/SR/advance/full40 or long-run qualification.')
            exit_code = 0
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        # Return a failing code after the exception scope releases traceback
        # frames, so discarded arm frames cannot retain graph owners at restore.
        exit_code = 1
    finally:
        gc.collect()
        if candidate is not None:
            if all(ref() is None for ref in refs):
                candidate.restore(); result['candidate_restored_after_owner_release'] = True
            else:
                result.update(status='failed_owner_release', candidate_restored_after_owner_release=False); exit_code = 1
        if ctx is not None:
            ctx.postwarm.restore(); ctx.postwarm_log.close()
            if ctx_weighted_active: ctx.weighted128.restore()
            result['original_runtime_hooks_restored'] = True
        result.update(script_sha256=sha(__file__), frozen_harness_sha256=FIXED_SHA, process_s=time.perf_counter()-started)
        fixed.write(args.output/'RESULT.json', result)
        print(json.dumps({k:result[k] for k in ['status','process_s']}), flush=True)
    return exit_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--check-local', action='store_true')
    parser.add_argument('--candidate', type=Path, default=HERE/'trig_direct_reuse.py'); parser.add_argument('--candidate-sha256')
    for name in ['capture','capture-audit','weighted-capture','weighted-capture-audit','ordinary-check','output']:
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check',
                 'host-small-check','host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check',
                 'metadata-backend','metadata-check','hybrid-check','weighted-check','weighted-checker',
                 'cache-release-check','cache-release-checker']:
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args(); args.mode = 'trig_direct_reuse_fixed_input_parity'; args.source40 = None
    raise SystemExit(main(args))
