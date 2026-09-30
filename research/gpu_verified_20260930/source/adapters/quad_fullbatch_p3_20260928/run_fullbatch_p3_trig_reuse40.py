"""One-shot candidate40 wrapper; frozen phase-GPU14 main/Runtime stay intact.

Use a fresh process and the original watchdog/environment. No CUDA is admitted
without the pinned actual fixed-input parity receipt. Candidate hooks are not
restored while graph owners may remain alive; this process exits afterwards.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback

HERE = Path(__file__).parent
BASE = 'run_fullbatch_p3_phase_profile_gpu14.py'
BASE_SHA = '780240cbaf08f02fc7f0a6967bfa7f00218395e5419a630ab075e78121775e7e'
COMMAND_SHA = '3705e932fbe1c1c54a47f10694fd0dcf5281f682ee4ba657c3295b6f51137e01'
CANDIDATE_SHA = '304ebc9c783fde32c8c1b4f6117ba83548198e9d38820398928e25533a1f0f07'
CHECKER_SHA = '9a9429011384b3138f2c80c619152efb099221b9de41c61bdb659cb1dc65e808'
POLICY = 'candidate_trig_direct_reuse_full1024_P3_original_phase_gpu14_bounded40'
PASS_FIELDS = ('ordinary_three_cases_all_five_outputs_bytes_equal',
    'ordinary_bad_mask_and_replay_alias_gates',
    'weighted_full1024_original_two_rounds_output_and_statistics_equal',
    'weighted_all_map_image_bad_and_EVENTS_float_bytes_equal',
    'eager32_all_outputs_events_statistics_equal', 'all_input_bytes_versions_unchanged',
    'both_arms_all_owners_released', 'candidate_restored_after_owner_release',
    'original_runtime_hooks_restored')
_started = False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def arguments(command, output):
    """Read only the pinned original script's option/value pairs; never exec argv."""
    assert sha(command) == COMMAND_SHA
    doc = read(command)
    argv = doc['argv']
    positions = [i for i, value in enumerate(argv) if Path(value).name == BASE]
    assert len(positions) == 1
    tail = argv[positions[0]+1:]
    assert len(tail) % 2 == 0
    fields = {}
    for flag, value in zip(tail[::2], tail[1::2]):
        assert flag.startswith('--') and not value.startswith('--')
        key = flag[2:].replace('-', '_')
        assert key not in fields
        fields[key] = value if key in ('mode', 'resource_check_sha256') else Path(value)
    assert fields['mode'] == 'bounded40' and output.resolve() != fields['output'].resolve()
    assert not output.exists(), 'Candidate output must be a new directory'
    fields.update(output=output, source40=None)
    assert doc['source_sha256'][BASE] == BASE_SHA
    assert sha(HERE/BASE) == BASE_SHA and sha(HERE/'trig_direct_reuse.py') == CANDIDATE_SHA
    return SimpleNamespace(**fields), doc


def parity_fields(result):
    assert result['status'] == 'passed_fixed_real_input_byte_parity'
    assert result['candidate_sha256'] == CANDIDATE_SHA and result['script_sha256'] == CHECKER_SHA
    assert all(result[name] is True for name in PASS_FIELDS)
    assert result['fullbatch_qualification'] is False and result['end_to_end_strict_certificate'] is False
    assert result['source_capture_passed'] is False and result['source_run_is_trajectory_qualification'] is False
    for label, interval in [('ordinary', False), ('weighted', True)]:
        observed = result['candidate_plan_observations'][label]
        plan = observed['last_plan']
        assert plan['order'] == 4 and plan['interval_coefficients'] is interval
        assert plan['trig_calls'] == plan['eligible_direct_calls'] == 45
        assert 0 < plan['potential_reuse_hits'] <= 39
        delta = observed['python_counter_delta']
        assert delta['exec_calls'] == delta['completed_execs'] > 0
        assert delta['failed_execs'] == 0 and delta['reuse_hits'] > 0


def parity_gate(directory, digest):
    assert len(digest) == 64 and sha(directory/'RESULT.json') == digest
    result = read(directory/'RESULT.json')
    parity_fields(result)
    assert sha(HERE/'check_trig_direct_reuse.py') == CHECKER_SHA
    inp = read(directory/'INPUT.json')
    assert inp['candidate_sha256'] == CANDIDATE_SHA
    assert set(result['receipts']) == {'baseline', 'candidate'}
    for label, expected in result['receipts'].items():
        assert sha(directory/(label+'_RECEIPT.json')) == expected
    return result, inp


def run(a):
    global _started
    assert not _started, 'One-shot process only'
    _started = True
    args, command = arguments(a.base_command, a.output)
    assert all(os.environ.get(k) == v for k, v in command['environment'].items())
    fixed, fixed_input = parity_gate(a.trig_check, a.trig_check_sha256)
    base = load('trig40_frozen_phase_gpu14', BASE, BASE_SHA)
    candidate_module = load('trig40_frozen_candidate', 'trig_direct_reuse.py', CANDIDATE_SHA)
    raw_initialize = base.initialize
    holder = {}
    receipt = dict(status='starting', wrapper_sha256=sha(__file__), candidate_sha256=CANDIDATE_SHA,
        base_entry_sha256=BASE_SHA, base_command_sha256=COMMAND_SHA,
        fixed_parity_result_sha256=a.trig_check_sha256,
        fixed_parity_input_sha256=sha(a.trig_check/'INPUT.json'),
        base_main_and_Runtime_unchanged=True, candidate_hook_restored=False,
        hook_lifetime='One-shot process exit; never restore candidate while graph owners may remain.',
        result_script_sha256_semantics='RESULT.script_sha256 identifies unchanged base main; INPUT.source_identity identifies this wrapper/candidate.',
        fullbatch_qualification=False, end_to_end_strict_certificate=False, fullT5_qualified=False)

    def initialize(original_args):
        ctx = raw_initialize(original_args)  # All original source/numeric/resource gates first.
        previous = dict(ctx.identity)
        assert previous['script_sha256'] == BASE_SHA
        assert fixed_input['base_identity'] == previous['base_postwarm_identity']
        assert not ctx.hybrid.receipts, 'Install before any validation owner/spec/graph'
        assert ctx.valid128.counters['calls'] == 0 and ctx.weighted128.counters['refine_calls'] == 0
        assert not ctx.postwarm.releases and not hasattr(ctx, 'phase_profile')
        binding = candidate_module.install(ctx.torch, ctx.c.se)
        holder.update(ctx=ctx, binding=binding)
        receipt['installed_after_original_gates_before_Runtime'] = True
        receipt['baseline_profile_identity_sha256'] = canonical(previous)
        ctx.identity = dict(previous, script_sha256=sha(__file__), policy=POLICY,
            base_profile_gpu14_entry_sha256=BASE_SHA,
            base_profile_gpu14_identity_sha256=canonical(previous),
            algorithm_identity=dict(policy=binding.policy, candidate_sha256=CANDIDATE_SHA,
                baseline_algorithm_identity_sha256=canonical(previous['algorithm_identity'])),
            trig_reuse_adapter_sha256=CANDIDATE_SHA, trig_reuse_policy=binding.policy,
            trig_reuse_fixed_input_qualification=dict(path=str(a.trig_check), result_sha256=a.trig_check_sha256,
                                                     checker_sha256=CHECKER_SHA),
            fullbatch_qualification=False, end_to_end_strict_certificate=False,
            limitation='Candidate direct-variable trig reuse; original full1024/NN/P3/point2/validation4/SR1000 and all40 observer/five-snapshot/nine full-byte prune gates retained. Fixed-input parity alone is not trajectory qualification. This bounded40 diagnostic is not fullT5 or an end-to-end strict NNCS certificate.')
        write(args.output/'TRIG_REUSE_RUN.json', receipt)
        return ctx

    base.initialize = initialize
    started = time.perf_counter()
    try:
        base.main(args)  # Original main, Runtime, all40 byte pairing and phase gates.
        result = read(args.output/'RESULT.json')
        assert result['script_sha256'] == BASE_SHA
        assert result['status'] == 'diagnostic40_bytes_equal'
        assert result['completed_step'] == 40 and result['accepted_lane_steps'] == 40960
        assert result['controller_calls'] == 2 and result['controller_refresh_steps'] == [0, 20]
        assert result['source_identity'] == read(args.output/'INPUT.json')['source_identity'] == holder['ctx'].identity
        receipt['status'] = 'candidate40_original_byte_and_phase_gates_passed'
    except BaseException as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        receipt['wrapper_process_s'] = time.perf_counter()-started
        if 'binding' in holder:
            binding = holder['binding']
            receipt.update(candidate_python_counters=dict(binding.counters), candidate_last_plan=dict(binding.last_plan),
                counters_scope='Python eager warmup/capture construction; graph replay does not increment these counters.')
        for name in ['RESULT.json', 'INPUT.json', 'REFERENCE40_COMPARISON.json', 'phase_profile.jsonl']:
            path = args.output/name
            if path.exists():
                receipt.setdefault('files_sha256', {})[name] = sha(path)
        if args.output.exists():
            write(args.output/'TRIG_REUSE_RUN.json', receipt)
        # No graph invocation or candidate.restore follows this point.


def check_local(a):
    args, command = arguments(a.base_command, a.output)
    assert args.mode == 'bounded40' and args.source40 is None
    assert args.resource_check_sha256 == 'b2ec10c2741299d17638b6a1f94ca2a4ad7f038743103ac30a91f5da59c34942'
    # Synthetic protocol fixture only; this is never a numerical admission.
    fixture = dict(status='passed_fixed_real_input_byte_parity', candidate_sha256=CANDIDATE_SHA,
        script_sha256=CHECKER_SHA, **{k:True for k in PASS_FIELDS},
        fullbatch_qualification=False, end_to_end_strict_certificate=False,
        source_capture_passed=False, source_run_is_trajectory_qualification=False,
        candidate_plan_observations={label:dict(last_plan=dict(order=4,interval_coefficients=mode,
            trig_calls=45,eligible_direct_calls=45,potential_reuse_hits=39),
            python_counter_delta=dict(exec_calls=3,completed_execs=3,failed_execs=0,reuse_hits=117))
            for label,mode in [('ordinary',False),('weighted',True)]})
    parity_fields(fixture)
    rejected = []
    for key, value in [('status','passed_CPU_harness_source_admission_only'), ('candidate_sha256','0'*64),
                       ('script_sha256','0'*64), *[(name,False) for name in PASS_FIELDS]]:
        bad = copy.deepcopy(fixture); bad[key] = value
        try:
            parity_fields(bad)
        except AssertionError:
            rejected.append(key)
        else:
            raise AssertionError('Bad admission accepted: '+key)
    assert not any(name == 'flowstar_gpu' or name.startswith('flowstar_gpu.') for name in sys.modules)
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(status='passed_CPU_parameter_and_rejection_checks_only', wrapper_sha256=sha(__file__),
        base_entry_sha256=BASE_SHA, candidate_sha256=CANDIDATE_SHA, base_command_sha256=COMMAND_SHA,
        copied_argument_count=len(vars(args)), original_resource_budget=command['resource_budget'],
        rejected_fields=rejected, synthetic_schema_fixture_not_numerical_admission=True,
        actual_fixed_CUDA_receipt_read=False, base_initialize_executed=False, CUDA_run=False,
        scope='Pinned argument mapping and fail-closed schema only; actual fixed gate/result still required.')
    write(args.output/'RESULT.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-command', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trig-check', type=Path)
    p.add_argument('--trig-check-sha256')
    p.add_argument('--check-local', action='store_true')
    a = p.parse_args()
    if a.check_local:
        check_local(a)
    else:
        assert a.trig_check is not None and a.trig_check_sha256 is not None
        run(a)
