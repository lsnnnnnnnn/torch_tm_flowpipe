"""Candidate-owned cold20->21 and long1000; frozen numerical mains/Runtime.

Checkpoint source identity and execution identity are distinct. Only the cold
identity/gate/comparison boundaries differ from the original P3 cold sequence.
One process, one run; candidate hooks live until process exit.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback

HERE = Path(__file__).parent
SHORT_WRAPPER_SHA = '1bbf4e7a8a81b2efb453da6cf9e3f6f8b3bc56a08eda1958088998361d2c4ee9'
CANDIDATE_SHA = '304ebc9c783fde32c8c1b4f6117ba83548198e9d38820398928e25533a1f0f07'
PROFILE_SHA = '780240cbaf08f02fc7f0a6967bfa7f00218395e5419a630ab075e78121775e7e'
SELECTIVE_SHA = '660c75cf22346e71d5044a60b5bbd62320ed38e542727cbcb6805fad909095e0'
LONG_SHA = '9d39ffdc78e37b108fa6518a39ed38ee58ad722ced84fffcc086f719aa6a38e9'
PROFILE_COMMAND_SHA = '3705e932fbe1c1c54a47f10694fd0dcf5281f682ee4ba657c3295b6f51137e01'
LONG_COMMAND_SHA = '959a4254cabcf5bc1cb2dd35841210ab2b3576881fdd64a76da511efaa4b11c9'
SOURCE_FILES = {
    'RESULT.json': '49dcd81db47e65a9d18c7e66917c86363148fced1e4238a912e251687815e523',
    'INPUT.json': '38e00dcfc2fb830c9152cff52f1a6424b57e91eaf249bada6116104ccc5717ca',
    'TRIG_REUSE_RUN.json': 'f94e7db1392327cc6ed35bc9fcf333e495d76f1c4114278aa7cb4cead8272337',
    'REFERENCE40_COMPARISON.json': 'beca30ebcb4a6fa47d740394af1a349756af4244d3ab19c13126e698a8c0fba3',
    'phase_profile.jsonl': '0dfbdddffa68e7ae7e4e376e517adcbf7cf49c39a1354d6414e8258a7f9f3751',
}
STATES = ('committed_20', 'after_endpoint_20', 'after_controller_20', 'committed_21')
ARTIFACTS = ('endpoint_20', 'controller_20', 'transfer_20', 'observer_21')
_started = False


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load(name, filename, digest):
    path = HERE/filename
    assert sha(path) == digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def arguments(a):
    cold_mode = a.mode == 'cold20_to21'
    base_name = 'run_fullbatch_p3_phase_profile_gpu14.py' if cold_mode else 'run_fullbatch_p3_gpu14.py'
    base_sha = PROFILE_SHA if cold_mode else LONG_SHA
    assert sha(a.base_command) == (PROFILE_COMMAND_SHA if cold_mode else LONG_COMMAND_SHA)
    command = read(a.base_command)
    argv = command['argv']; positions = [i for i, v in enumerate(argv) if Path(v).name == base_name]
    assert len(positions) == 1
    tail = argv[positions[0]+1:]; assert len(tail) % 2 == 0
    fields = {}
    for flag, value in zip(tail[::2], tail[1::2]):
        assert flag.startswith('--') and not value.startswith('--')
        key = flag[2:].replace('-', '_'); assert key not in fields
        fields[key] = value if key == 'mode' or key.endswith('_sha256') else Path(value)
    assert not a.output.exists() and a.output.resolve() != fields['output'].resolve()
    fields.update(output=a.output, mode=a.mode)
    if cold_mode:
        fields['source40'] = a.candidate40
    assert command['source_sha256'][base_name] == base_sha and sha(HERE/base_name) == base_sha
    assert sha(HERE/'trig_direct_reuse.py') == CANDIDATE_SHA
    return SimpleNamespace(**fields), command


def source40_fields(inp, result, receipt, comparison, short_identity, profile_identity=None):
    identity = inp['source_identity']
    assert result['source_identity'] == identity
    assert result['script_sha256'] == PROFILE_SHA
    assert identity['script_sha256'] == receipt['wrapper_sha256'] == SHORT_WRAPPER_SHA
    assert receipt['candidate_sha256'] == identity['trig_reuse_adapter_sha256'] == CANDIDATE_SHA
    assert receipt['base_entry_sha256'] == PROFILE_SHA
    assert receipt['status'] == 'candidate40_original_byte_and_phase_gates_passed'
    assert result['status'] == 'diagnostic40_bytes_equal'
    assert inp['mode'] == result['mode'] == 'bounded40'
    assert inp['original_target_steps'] == result['original_target_steps'] == 1000
    assert inp['budget_steps'] == result['budget_steps'] == result['completed_step'] == 40
    assert result['accepted_lane_steps'] == 40960 and result['all_steps_all_lanes_accepted'] is True
    assert result['controller_calls'] == 2 and result['controller_refresh_steps'] == [0, 20]
    assert result['pruned_checkpoint20_support_cache_verified'] is True
    assert identity['algorithm_identity']['baseline_algorithm_identity_sha256'] == canonical(short_identity)
    if profile_identity is not None:
        assert identity['base_profile_gpu14_identity_sha256'] == canonical(profile_identity)
    assert (identity['batch_size'], identity['working_total_degree'], identity['point_code_order'], identity['validation_order']) == (1024, 3, 2, 4)
    assert identity['early_weighted_chunk_size'] == identity['early_weighted_graph_padding'] == 128
    assert result['fullbatch_qualification'] is False and result['end_to_end_strict_certificate'] is False
    counters = receipt['candidate_python_counters']
    assert counters['exec_calls'] == counters['completed_execs'] > 0 and counters['failed_execs'] == 0
    assert counters['reuse_hits'] > 0
    plan = receipt['candidate_last_plan']
    assert (plan['trig_calls'], plan['eligible_direct_calls'], plan['unique_direct_plans'], plan['potential_reuse_hits'], plan['order']) == (45, 45, 6, 39, 4)
    assert comparison['status'] == 'passed' and comparison['source_identity'] == identity
    assert comparison['observer_steps'] == list(range(1, 41)) and comparison['observer_all_tensor_bytes_equal'] is True
    assert comparison['eviction_sequence_equal'] is True and comparison['full_hash_steps'] == [2,3,4,9,10,20,28,30,33]
    assert comparison['noop_steps'] == 31
    assert set(result['snapshots']) == set(comparison['snapshots']) == set(STATES+('committed_40',))
    for row in comparison['snapshots'].values():
        assert all(row[k] is True for k in ['plant_all_fields','SR_and_host_all_components','progress','all_component_file_hashes_verified'])
        assert row['component_count'] == 14
    return identity


def source40_gate(source, short_identity, selective, profile_identity=None):
    source = Path(source)
    for filename, digest in SOURCE_FILES.items():
        assert sha(source/filename) == digest, filename
    inp, result, receipt, comparison = [read(source/name) for name in ['INPUT.json','RESULT.json','TRIG_REUSE_RUN.json','REFERENCE40_COMPARISON.json']]
    identity = source40_fields(inp, result, receipt, comparison, short_identity, profile_identity)
    assert result['input_sha256'] == SOURCE_FILES['INPUT.json']
    selective.reference_pair_gate(source, identity, result)
    for name, digest in receipt['files_sha256'].items():
        assert name in SOURCE_FILES and sha(source/name) == digest
    fixed = identity['trig_reuse_fixed_input_qualification']
    short = load('trig_continue_frozen_short_wrapper', 'run_fullbatch_p3_trig_reuse40.py', SHORT_WRAPPER_SHA)
    short.parity_gate(Path(fixed['path']), fixed['result_sha256'])
    assert sha(Path(fixed['path'])/'INPUT.json') == receipt['fixed_parity_input_sha256']
    selective.w.counters_gate(result['valid128_counters'], 40)
    selective.b.counters_gate(result['working_graph_counters'], 40)
    selective.eager_counters(result['working_eager_counters'], 40)
    selective.log_gate(source, list(range(1, 41)), result)
    selective.eager_log_gate(source, list(range(1, 41)), result)
    for name, digest in result['snapshots'].items():
        assert sha(source/name/'MANIFEST.json') == digest
    assert result['observer_endpoint64_calls'] == 40
    for key in ['observer_driver_function_unchanged','phase_profile_restored','working_eager_restored','horner_cleanup_restored','working_graph_hook_restored','valid_adapter_restored','cache_release_adapter_restored','weighted_adapter_restored']:
        assert result[key] is True
    pc = result['phase_profile_counters']
    assert pc['begun_steps'] == pc['finished_steps'] == 40 and pc['aborted_steps'] == 0 and pc['restored'] is True
    assert [row['step'] for row in result['phase_profile_rows']] == list(range(1, 41))
    return identity, dict(source40_path=str(source), source_input_sha256=SOURCE_FILES['INPUT.json'],
        source_result_sha256=SOURCE_FILES['RESULT.json'], source_trig_receipt_sha256=SOURCE_FILES['TRIG_REUSE_RUN.json'],
        source_identity_sha256=canonical(identity))


def compare_snapshot(runtime, left, right, source_identity):
    a, sa = runtime.ctx.trig_p3.snapshot_summary(left)
    b, sb = runtime.ctx.trig_p3.snapshot_summary(right)
    assert a['metadata'] == source_identity and b['metadata'] == runtime.ctx.identity
    assert a['progress'] == b['progress']
    ta = runtime.t.load(Path(left)/'plant.pt', map_location='cpu', weights_only=True)
    tb = runtime.t.load(Path(right)/'plant.pt', map_location='cpu', weights_only=True)
    assert runtime.signature(ta) == runtime.signature(tb)
    assert sa['metadata'] == sb['metadata'] and set(sa['tensors']) == set(sb['tensors'])
    assert len(sa['tensors']) == 14
    for name in sa['tensors']:
        for key in ['shape', 'dtype', 'raw_sha256']:
            assert sa['tensors'][name][key] == sb['tensors'][name][key], (name, key)
    return dict(plant_all_fields=True, SR_and_host_all_components=True, progress=True)


def compare_artifact(runtime, left, right, source_identity):
    values = []
    for path, identity in [(Path(left), source_identity), (Path(right), runtime.ctx.identity)]:
        side = read(path.with_suffix('.json'))
        assert side['source_identity'] == identity and side['pt_sha256'] == sha(path)
        data = runtime.t.load(path, map_location='cpu', weights_only=True)
        value = runtime.signature(data); assert value == side['signature']; values.append(value)
    assert values[0] == values[1]
    return True


# The body below is copied from frozen run_fullbatch_p3.cold. A checked diff
# lists only gate/source-metadata/separate-identity comparison changes.
def cold(runtime,source):
    x=runtime;ctx=x.ctx;d=x.d;source=Path(source)
    binding=ctx.trig_source_binding;source_identity=ctx.trig_source_identity
    assert source.resolve()==Path(binding['source40_path']).resolve()
    # Original factory sequence except initial state/SR: restore exactly one SR.
    cfg=ctx.cfg;names=[e['name'] for e in cfg['initial_set']]
    settings=d.Settings(step=.005,order=3,cutoff=1e-6,remainder_estimation=d.parse_rem_est(cfg['remainder_estimation'],16),mode='strict',device='cuda:0')
    model=d.build_crown(cfg,'cuda:0',relax='same-slope',input_layout='native')
    code=d.compile_ode(cfg['dynamics_expressions'],names,order=2)
    tables=d.build_tables(16,3).to('cuda:0');step=d.poly.build_step_tables(tables,.005);sched=d.build_schedule(16,3,'cuda:0');eng=d.SparseEngine(tables,step,'cuda:0')
    st,sr,ledger,cap,progress=ctx.snapshot.restore(source/'committed_20',eng,metadata=source_identity,host=x.h,raw_factory=x.raw['make_symbolic_remainder'])
    assert progress['completed_step']==20 and progress['phase']=='committed_before_endpoint_handoff' and progress['held_controller_step']==0
    assert sr.max_size==len(sr.phi_buf)==1000 and ledger.length==20 and ledger.epoch==0
    expected_cap=x.raw['build_rem_est'](settings,16,1024,'cuda:0')
    assert x.signature(cap)==x.signature(expected_cap);del expected_cap
    x.st,x.sr,x.ledger,x.cap,x.completed=st,sr,ledger,cap,20
    x.held=progress['held_controller_step'];x.held_signature=progress['held_controller_signature']
    x.broken=x.t.tensor(progress['broken'],dtype=x.t.bool,device='cpu');x.last_accepted=x.t.tensor(progress['last_accepted_step'],dtype=x.t.int64,device='cpu')
    assert not bool(x.broken.any()) and bool((x.last_accepted==20).all());x.attach_reset()
    ctx.c.se._eng_cache(eng,'_acc_sups')[('state',)]=(st.pre_sup,st.tmv_sup)
    d.end_of_time_s(st,eng)
    hull=d.hull_ranges_s(st,eng,12)
    T,L,U=d.crown_bounds(model,cfg,hull[...,0].contiguous(),hull[...,1].contiguous(),input_layout='native')
    T,L,U=d.apply_crown_transport(T,L,U,'native-f64');d.inject_controls_s(st,T,L,U,[13,14,15],12)
    st,ok=d.advance_sparse(st,code,eng,sched,settings,cap,sr);st=d.prune_state(st,eng)
    assert bool(ok.all()) and not bool(x.broken.any());assert sr.reset_if_full() is False
    x.save('committed_21')
    checks={name:compare_snapshot(x,source/name,x.args.output/name,source_identity) for name in ['committed_20','after_endpoint_20','after_controller_20','committed_21']}
    for name in ['endpoint_20','controller_20','transfer_20','observer_21']:checks[name]=compare_artifact(x,source/(name+'.pt'),x.args.output/(name+'.pt'),source_identity)
    assert x.controller_calls==1 and x.controller_steps==[20]
    return dict(status='passed',mode='cold20_to21',checks=checks,**binding)


def cold_gate(a, source_identity, links, short_identity, selective):
    root = a.candidate_cold
    assert len(a.candidate_cold_result_sha256) == len(a.candidate_cold_receipt_sha256) == 64
    assert sha(root/'RESULT.json') == a.candidate_cold_result_sha256
    assert sha(root/'TRIG_CONTINUE_RUN.json') == a.candidate_cold_receipt_sha256
    inp, result, receipt = [read(root/name) for name in ['INPUT.json','RESULT.json','TRIG_CONTINUE_RUN.json']]
    identity = inp['source_identity']
    assert result['source_identity'] == identity
    assert identity['script_sha256'] == receipt['wrapper_sha256'] == sha(__file__)
    assert identity['execution_mode'] == receipt['mode'] == 'cold20_to21'
    assert identity['baseline_selective_identity_sha256'] == canonical(short_identity)
    assert identity['algorithm_identity'] == source_identity['algorithm_identity']
    assert identity['checkpoint_source_identity_sha256'] == canonical(source_identity)
    assert receipt['status'] == 'candidate_cold_passed' and receipt['candidate_sha256'] == CANDIDATE_SHA
    assert receipt['base_main_sha256'] == result['script_sha256'] == SELECTIVE_SHA
    assert result['input_sha256'] == sha(root/'INPUT.json')
    assert inp['mode'] == result['mode'] == 'cold20_to21' and result['status'] == 'passed'
    assert Path(inp['source40']).resolve() == a.candidate40.resolve()
    assert result['original_target_steps'] == inp['original_target_steps'] == 1000
    assert result['budget_steps'] == inp['budget_steps'] == 40
    for key, value in links.items():
        assert result[key] == value
    assert result['completed_step'] == 21 and result['accepted_lane_steps'] == 1024
    assert result['all_steps_all_lanes_accepted'] is True
    assert result['controller_calls'] == 1 and result['controller_refresh_steps'] == [20]
    assert result['pruned_checkpoint20_support_cache_verified'] is True
    assert set(result['snapshots']) == set(STATES) and set(result['checks']) == set(STATES+ARTIFACTS)
    for name in STATES:
        assert result['checks'][name] == dict(plant_all_fields=True, SR_and_host_all_components=True, progress=True)
        assert sha(root/name/'MANIFEST.json') == result['snapshots'][name]
        assert read(root/name/'MANIFEST.json')['metadata'] == identity
    for name in ARTIFACTS:
        assert result['checks'][name] is True
        sides = []
        for directory, expected_identity in [(a.candidate40, source_identity), (root, identity)]:
            side = read(directory/(name+'.json'))
            assert side['source_identity'] == expected_identity and side['step'] == (21 if name == 'observer_21' else 20)
            assert sha(directory/(name+'.pt')) == side['pt_sha256']; sides.append(side)
        assert sides[0]['signature'] == sides[1]['signature']
    assert result['observer_endpoint64_calls'] == 1 and result['observer_driver_function_unchanged'] is True
    owner, = result['hybrid_validation_owners']
    assert (owner['n'], owner['working'], owner['validation']) == (16, 3, 4) and owner['prefix_checked'] is True
    assert owner['full_degree_six_prefix'] == dict(full=100947, spatial=74613)
    for key in ['working_eager_restored','horner_cleanup_restored','working_graph_hook_restored','valid_adapter_restored','cache_release_adapter_restored','weighted_adapter_restored']:
        assert result[key] is True
    selective.eager_counters(result['working_eager_counters'], 1)
    selective.w.counters_gate(result['valid128_counters'], 1)
    selective.b.counters_gate(result['working_graph_counters'], 1)
    weighted = result['weighted128_counters']
    assert weighted['chunk_size'] == 128 and 0 <= weighted['refine_calls'] <= 1 and weighted['graph_map_calls'] >= 0
    assert result['postwarm_releases']
    for row in result['postwarm_releases']:
        assert row['allocated_after'] <= row['allocated_before'] and row['reserved_after'] <= row['reserved_before']
    selective.log_gate(root, [21], result); selective.eager_log_gate(root, [21], result)
    for name, digest in receipt['files_sha256'].items():
        assert sha(root/name) == digest
    counters = receipt['candidate_python_counters']
    assert counters['exec_calls'] == counters['completed_execs'] > 0 and counters['failed_execs'] == 0 and counters['reuse_hits'] > 0
    return dict(path=str(root), result_sha256=a.candidate_cold_result_sha256,
        input_sha256=sha(root/'INPUT.json'), receipt_sha256=a.candidate_cold_receipt_sha256,
        identity_sha256=canonical(identity), complete_four_snapshots_and_four_artifacts=True)


def run(a):
    global _started
    assert not _started, 'One-shot process only'; _started = True
    args, command = arguments(a)
    assert all(os.environ.get(k) == v for k, v in command['environment'].items())
    candidate = load('trig_continue_frozen_candidate', 'trig_direct_reuse.py', CANDIDATE_SHA)
    cold_mode = a.mode == 'cold20_to21'
    if cold_mode:
        profile = load('trig_continue_frozen_profile', 'run_fullbatch_p3_phase_profile_gpu14.py', PROFILE_SHA)
        base = profile.base.s; raw_initialize = base.initialize
        assert sha(base.__file__) == SELECTIVE_SHA
    else:
        base = load('trig_continue_frozen_long', 'run_fullbatch_p3_gpu14.py', LONG_SHA)
        raw_initialize = base.initialize
    holder = {}
    receipt = dict(status='starting', mode=a.mode, wrapper_sha256=sha(__file__), candidate_sha256=CANDIDATE_SHA,
        base_main_sha256=SELECTIVE_SHA if cold_mode else LONG_SHA, base_command_sha256=sha(a.base_command),
        candidate40_path=str(a.candidate40), candidate40_result_sha256=SOURCE_FILES['RESULT.json'],
        candidate40_receipt_sha256=SOURCE_FILES['TRIG_REUSE_RUN.json'],
        numerical_main_Runtime_unchanged=True, cold_sequence_only_identity_boundaries_changed=cold_mode,
        hook_lifetime='One-shot process exit; candidate never restored while graph owners may remain.',
        candidate_hook_restored=False, fullbatch_qualification=False, end_to_end_strict_certificate=False,
        result_script_sha256_semantics='Frozen reused main; INPUT.source_identity names the actual wrapper and candidate.')

    def initialize(original_args):
        if cold_mode:
            # profile.initialize calls this same selective module. Restore its
            # original initializer only during that original admission chain.
            base.initialize = raw_initialize
            try:
                ctx = profile.initialize(original_args)
            finally:
                base.initialize = initialize
            previous = dict(ctx.identity); short_identity = previous['algorithm_identity']
            selective = base
            source_identity, links = source40_gate(a.candidate40, short_identity, selective, previous)
            own_cold = None
        else:
            ctx = raw_initialize(original_args)  # Original historical40/cold first.
            previous = dict(ctx.identity); short_identity = previous['qualified_short_identity']
            selective = base.w
            source_identity, links = source40_gate(a.candidate40, short_identity, selective)
            own_cold = cold_gate(a, source_identity, links, short_identity, selective)
        assert not ctx.hybrid.receipts and not ctx.postwarm.releases
        assert ctx.valid128.counters['calls'] == 0 and ctx.weighted128.counters['refine_calls'] == 0
        binding = candidate.install(ctx.torch, ctx.c.se)
        assert binding.policy == source_identity['trig_reuse_policy']
        ctx.trig_source_identity = source_identity; ctx.trig_source_binding = links; ctx.trig_p3 = selective.P3
        ctx.identity = dict(previous, script_sha256=sha(__file__), execution_mode=a.mode,
            policy='candidate_trig_direct_reuse_'+a.mode, algorithm_identity=dict(source_identity['algorithm_identity']),
            trig_reuse_adapter_sha256=CANDIDATE_SHA, trig_reuse_policy=binding.policy,
            baseline_initializer_identity_sha256=canonical(previous), baseline_selective_identity_sha256=canonical(short_identity),
            checkpoint_source_identity_sha256=canonical(source_identity),
            candidate_qualification=dict(source40=links, cold=own_cold),
            baseline_qualification=dict(source40=str(args.source40), cold=str(args.cold)) if not cold_mode else None,
            top_level_INPUT_source40_cold_scope='Historical base prerequisites; candidate paths are in source_identity.candidate_qualification.' if not cold_mode else 'Candidate checkpoint source; snapshot load verifies its own source identity.',
            phase_profile_active=False, fullbatch_qualification=False, end_to_end_strict_certificate=False,
            limitation='Candidate-owned identity and checkpoint provenance. All original numerical admissions and Runtime retained. Cold compares candidate source/current identities separately. Long retains original50 NN/1000 SR and original per-lane failure semantics. No end-to-end strict NNCS certificate.')
        holder.update(ctx=ctx, binding=binding)
        receipt.update(installed_after_original_and_candidate_gates_before_Runtime=True,
            baseline_initializer_identity_sha256=canonical(previous), baseline_selective_identity_sha256=canonical(short_identity),
            execution_identity_sha256=canonical(ctx.identity), candidate_qualification=ctx.identity['candidate_qualification'])
        write(args.output/'TRIG_CONTINUE_RUN.json', receipt)
        return ctx

    base.initialize = initialize
    if cold_mode:
        base.cold = cold
    started = time.perf_counter()
    try:
        base.main(args)  # Frozen main, complete numerical Runtime and cleanup.
        result = read(args.output/'RESULT.json')
        assert result['script_sha256'] == receipt['base_main_sha256']
        assert result['source_identity'] == read(args.output/'INPUT.json')['source_identity'] == holder['ctx'].identity
        if cold_mode:
            assert result['status'] == 'passed' and result['completed_step'] == 21 and result['accepted_lane_steps'] == 1024
            assert result['controller_calls'] == 1 and result['controller_refresh_steps'] == [20]
            assert set(result['checks']) == set(STATES+ARTIFACTS)
            receipt['status'] = 'candidate_cold_passed'
        else:
            receipt['status'] = 'candidate_long_terminal'
            receipt['numerical_status'] = result['status']
            receipt['overall_horizon_completed'] = result['overall_horizon_completed']
    except BaseException as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        receipt['wrapper_process_s'] = time.perf_counter()-started
        if 'binding' in holder:
            receipt['candidate_python_counters'] = dict(holder['binding'].counters)
            receipt['candidate_last_plan'] = dict(holder['binding'].last_plan)
            receipt['counters_scope'] = 'Python warmup/capture construction only; graph replay does not increment counters.'
        for name in ['RESULT.json','INPUT.json','working_prune.jsonl','working_eager.jsonl']:
            if (args.output/name).exists():
                receipt.setdefault('files_sha256', {})[name] = sha(args.output/name)
        if args.output.exists():
            write(args.output/'TRIG_CONTINUE_RUN.json', receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['cold20_to21','long1000'], required=True)
    p.add_argument('--base-command', type=Path, required=True)
    p.add_argument('--candidate40', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--candidate-cold', type=Path)
    p.add_argument('--candidate-cold-result-sha256')
    p.add_argument('--candidate-cold-receipt-sha256')
    a = p.parse_args()
    assert (a.candidate_cold is not None) == (a.mode == 'long1000')
    if a.mode == 'long1000':
        assert a.candidate_cold_result_sha256 and a.candidate_cold_receipt_sha256
    run(a)
