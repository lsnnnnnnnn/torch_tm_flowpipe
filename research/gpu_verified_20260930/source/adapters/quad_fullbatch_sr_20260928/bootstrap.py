"""Pinned source/library bootstrap for the original QUAD B1024/P2 driver.

This module creates no engine, schedule, tape, graph, NN model or flowpipe.
Only initialize() loads the pinned CUDA libraries. The caller owns its new
full-batch boundary/SR qualification and hooks, official driver.main call,
forty-step capture, and independent cold replay. No B2 experiment is reused.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import os
import sys

HERE = Path(__file__).parent
N = Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z')
COMMON_SHA = '1c85e04988028f7ac95d945eeff9ef3e0936ee306c1f0a80650bb06d8073d985'
REFERENCE_SHA = '4fad5bf5a59bd71b9779f8145c85a79d08a9b0840f22d6856b87c3c3fa32d4ea'
RECIPROCAL_SHA = 'eb5147e4074269f7b568838e0575849a97436295b45cf4c3082be1315d5cf32e'
ENDPOINT_SHA = 'f4310a1b0484fd344ae894641302e146fcd232a8c7b296546a0fcdc94b0cd326'
ENDPOINT_CHECK_SHA = '7eb2c1a217238eadcea04d8a373eb00c4eb48b219de9fac2bf3d8cb5db69d188'
INJECTION_SHA = 'a79b19ad1bd97ae87b40f81239352a8b9859621757bd044df6fd615630c64310'
INJECTION_CHECK_SHA = 'ff35186cf01de2f63fbd5f9d344d743d86021dd9f135918daa0381f5738c5e4a'
CONFIG_SHA = '940cb0b6188c3b127eee28df5d34fbda6d072f9e8b09cee8f251bd6b098f9144'
MODEL_SHA = 'fabd84e411f4b0ebe0d6b996be6e4bd2adc48cf1118ab99c82180563b95532dd'
DRIVER_SHA = '1bc3aeea0e9216fe8432ae78f18c9592d19658c23cdde02b90a54d0b14c08a31'
BOXES_SHA = '8dd18155d7d63b311fece5732251758a4984c0abb428377c135e68c244cedd30'
SETTINGS = dict(step=.005, order=2, cutoff=1e-6, remainder_estimation=.1,
                mode='strict', device='cuda')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def boundary_gates(args, original):
    """Existing small mathematical gates, not a full-B1024 admission."""
    endpoint_path = HERE.parent / 'quad_targeted_recovery_20260927/strict_endpoint.py'
    injection_path = HERE.parent / 'quad_control_transfer_20260927/strict_injection.py'
    assert sha(endpoint_path) == ENDPOINT_SHA
    assert sha(endpoint_path.with_name('check_strict_endpoint.py')) == ENDPOINT_CHECK_SHA
    assert sha(injection_path) == INJECTION_SHA
    assert sha(injection_path.with_name('check_injection.py')) == INJECTION_CHECK_SHA
    endpoint = read(args.endpoint_check / 'RESULT.json')
    injection = read(args.injection_check / 'RESULT.json')
    for receipt in (endpoint, injection):
        assert receipt['status'] == 'passed' and receipt['device'] == 'cuda'
        assert receipt['engine_python_sha256'] == original['engine']['python_sha256']
        assert receipt['torch_version'] == original['torch_version']
    assert endpoint['adapter_sha256'] == ENDPOINT_SHA
    assert endpoint['check_script_sha256'] == ENDPOINT_CHECK_SHA
    assert all(endpoint[k] is True for k in ('nonempty_sr_boundary_unchanged',
        'fresh_compose_received_exact_charged_remainder', 'normal_single_J_append',
        'old_J_bytes_unchanged', 'ordinary_remainder_nonzero'))
    assert endpoint['deliberately_omitted_marker_misses'] > 0
    assert injection['adapter_sha256'] == INJECTION_SHA
    assert injection['script_sha256'] == INJECTION_CHECK_SHA
    assert injection['driver_sha256'] == DRIVER_SHA
    assert injection['extensions'] == original['extensions']
    assert injection['exact_coefficient_checks'] == 72
    assert injection['exact_point_component_checks'] == 162
    assert injection['analytic_composed_checks'] == 60
    assert len(injection['negative_fail_before_mutation']) == 8
    assert injection['deliberately_omitted_control_remainder_misses'] > 0
    assert all(injection[k] is True for k in ('original_same_device_point_bytes_equal',
        'sr_boundary_unchanged', 'old_J_bytes_unchanged',
        'fresh_compose_received_exact_remainder', 'exactly_one_normal_J_append',
        'instrumented_bare_full_state_and_live_SR_bytes_equal'))
    assert {row['order'] for row in injection['actual_boundaries']} == {3, 4}
    for row in injection['actual_boundaries']:
        assert row['same_device_original_point_bytes_equal'] and row['archived_point_bytes_equal']
        for key in ('config_sha256', 'model_sha256', 'boxes_sha256'):
            assert row['source_identity'][key] == original[key]
    return endpoint_path, injection_path, endpoint, injection


def initialize(args):
    """args: common, candidate_build/check, endpoint_check, injection_check Paths.

    Call once in a fresh process. Return ctx; original driver.build_crown stays
    untouched and will create its model only when the caller runs driver.main.
    No endpoint, injection or SR helper is installed by this bootstrap.
    """
    assert not any(k == 'flowstar_gpu' or k.startswith('flowstar_gpu.') for k in sys.modules), \
        'Fresh process required: reciprocal install must precede engines, graphs and tapes'
    assert sha(args.common) == COMMON_SHA
    reference = N / 'runs/quad_normalization_center_20260924/combined/result.json'
    assert sha(reference) == REFERENCE_SHA
    original = read(reference)
    for key, expected in [('config', CONFIG_SHA), ('model', MODEL_SHA), ('driver_path', DRIVER_SHA)]:
        assert sha(original[key]) == expected
        assert original['driver_sha256' if key == 'driver_path' else key + '_sha256'] == expected
    boxes_path = N / 'runs/archcomp_failure_20260923/huan_box_parity_preload/boxes.json'
    assert sha(boxes_path) == original['boxes_sha256'] == BOXES_SHA
    ep_path, inj_path, ep_gate, inj_gate = boundary_gates(args, original)
    reciprocal_path = HERE.parent / 'quad_reciprocal_20260928/run_quad_reciprocal.py'
    assert sha(reciprocal_path) == RECIPROCAL_SHA
    reciprocal = load('fullbatch_reciprocal_admission', reciprocal_path)
    # This pure file gate checks source and both actual qualified new SOs.
    directory, build, gate = reciprocal.candidate_gate(args)
    assert build['common_sha256'] == COMMON_SHA
    assert build['baseline_extensions'] == original['extensions']
    assert build['torch_version'] == gate['torch_version'] == original['torch_version']

    c = load('fullbatch_frozen_common', args.common)
    assert c.N == N and c.ORIGINAL == original
    assert c.E.resolve() == Path(build['engine_root']).resolve()
    edge = c.bootstrap()  # Source and prebuilt libraries only; no c.setup/attempt.
    torch = c.torch
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.__version__ == original['torch_version']
    assert c.se.VALIDATION_POLICY == 'solution_plus_one'
    assert all(os.environ.get(k) == v for k, v in original['environment'].items())
    assert all(os.environ.get(k) == v for k, v in inj_gate['environment'].items())
    installer = load('fullbatch_reciprocal_installer', directory / 'install.py')
    candidate = installer.install(Path(gate['candidate']['cache_root']), allow_build=False,
        binary_paths={name: item['path'] for name, item in gate['extensions'].items()})
    assert candidate.metadata == gate['candidate']
    assert candidate.tape.available() and candidate.tape.valid_available()
    assert candidate.loads == gate['extensions']

    cfg_path = Path(original['config'])
    cfg = c.yaml.safe_load(cfg_path.read_text())
    cfg['_config_dir'] = cfg_path.parent
    assert (cfg_path.parent.parent / cfg['model_dir']).resolve() == Path(original['model'])
    assert cfg['num_vars'] == len(cfg['initial_set']) == 16
    assert cfg['num_nn_input'] == 12 and cfg['num_nn_output'] == 3
    assert cfg['steps'] == 50 and cfg['step_size'] == .1 and cfg['ode_step_size'] == .005
    assert cfg['ode_order'] == 2 and float(cfg['cut_off_threshold']) == 1e-6
    assert cfg['remainder_estimation'] == [-.1, .1] and cfg['sr_queue'] == 1000
    assert cfg['bound_opts'] == {'activation_bound_option': 'same-slope'}
    assert cfg['input_shape'] == [-1, 12]
    boxes = torch.tensor(read(boxes_path), dtype=torch.float64, device='cpu')
    assert boxes.shape == (1024, 16, 2) and bool(torch.isfinite(boxes).all())
    assert bool((boxes[..., 0] <= boxes[..., 1]).all())
    d = load('fullbatch_original_driver', Path(original['driver_path']))
    assert d.sparse_exec_module is c.se
    assert d.SparseEngine is c.sp.SparseEngine and d.build_tables is c.build_tables
    assert d.SR_QUEUE == cfg['sr_queue']
    # Returning the archived boxes does not overwrite original make_cells here.
    # The caller must install an audited full1024 clone hook, not slice/reorder.
    identity = dict(script_sha256=sha(__file__), common_script_sha256=COMMON_SHA,
        reference_path=str(reference), reference_sha256=REFERENCE_SHA,
        config_path=str(cfg_path), config_sha256=CONFIG_SHA,
        model_path=original['model'], model_sha256=MODEL_SHA,
        driver_path=original['driver_path'], driver_sha256=DRIVER_SHA,
        boxes_path=str(boxes_path), boxes_sha256=BOXES_SHA,
        engine=original['engine'], extensions=original['extensions'],
        environment={k: v for k, v in os.environ.items() if k.startswith('FLOWSTAR_') or k in
            ['CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'TORCH_EXTENSIONS_DIR']},
        reciprocal_admission_sha256=RECIPROCAL_SHA, reciprocal_policy=reciprocal.POLICY,
        reciprocal_candidate=candidate.metadata, reciprocal_extensions=dict(candidate.loads),
        reciprocal_build_path=str(args.candidate_build),
        reciprocal_build_result_sha256=sha(args.candidate_build / 'RESULT.json'),
        reciprocal_check_path=str(args.candidate_check),
        reciprocal_check_result_sha256=sha(args.candidate_check / 'RESULT.json'),
        endpoint_adapter_sha256=ENDPOINT_SHA, endpoint_check_script_sha256=ENDPOINT_CHECK_SHA,
        endpoint_check_result_sha256=sha(args.endpoint_check / 'RESULT.json'),
        injection_adapter_sha256=INJECTION_SHA, injection_check_script_sha256=INJECTION_CHECK_SHA,
        injection_check_result_sha256=sha(args.injection_check / 'RESULT.json'),
        batch_size=1024, settings=SETTINGS, working_total_degree=2,
        point_code_order=1, validation_order=3, metadata_policy='original_dense_tables',
        controller_policy='original_full1024_per_lane_direct_pre_plus_R_box',
        controller_calls_per_refresh=1, controller_transport='native-f64',
        fullbatch_qualification=False, end_to_end_strict_certificate=False,
        limitation='Bootstrap only. Fullbatch boundary/SR and own40/cold gates are required separately; CROWN certificates remain conditional.')
    return SimpleNamespace(c=c, d=d, torch=torch, edge=edge, cfg=cfg, boxes=boxes,
        identity=identity, candidate=candidate, endpoint_path=ep_path, injection_path=inj_path,
        original_build_crown=d.build_crown, original_make_cells=d.make_cells)


def driver_argv(ctx):
    """Exact full-batch plant/NN options; no order/grid/horizon override."""
    return [ctx.identity['driver_path'], ctx.identity['config_path'], '--device', 'cuda:0',
            '--strict', '--engine', 'sparse', '--crown-domain', 'box',
            '--crown-relax', 'same-slope', '--crown-transport', 'native-f64',
            '--crown-input-layout', 'native', '--nn-mode', 'crown']
