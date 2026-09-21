"""Qualification and bounded strict-plant NNCS screens; never an NNCS certificate."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

BASELINE_SHA = '280abb400610f56210a7a5be61d5f98be3e27251'
XIANGRU_SHA = '1c16d4ef2cb91cc94b1c784f7383e1eba135d8d3'
CONTROLLER = {
    'backend': 'ONNX -> onnx2pytorch -> auto_LiRPA CROWN, in process',
    'arithmetic': 'native float64 round-to-nearest, box, same-slope',
    'rpc_float32': False,
    'strict_qualified': False,
    'known_gap': 'CROWN RN affine/bias computations and einsum coefficient injection have no proven outward roundoff accounting; --strict selects plant Settings only. No end-to-end strict NNCS certificate.',
}

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_driver(engine):
    sys.path.insert(0, str(engine / 'src'))
    spec = importlib.util.spec_from_file_location('nncs_baseline_driver', engine / 'integrations/crown_reach/gpu_driver.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def prepare(args):
    import yaml
    import onnx
    X = args.source
    assert subprocess.check_output(['git', '-C', str(X), 'rev-parse', 'HEAD'], text=True).strip() == XIANGRU_SHA
    driver = load_driver(args.engine)
    out = args.configs
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    manifests = {p.stem: p for p in (X / 'experiments/reachability/benchmarks').glob('*.json')}
    ids = ['acc', 'cartpole', 'double_pendulum_less_robust', 'double_pendulum_more_robust', 'nav_robust', 'nav_standard', 'single_pendulum', 'tora_homogeneous', 'tora_relu_tanh', 'tora_sigmoid', 'airplane', 'attitude_control', 'quad', 'unicycle']
    for name in ids:
        source = manifests.get(name, X / 'src/configs' / (name + '.yaml'))
        row = {'id': name, 'contract_source': str(source), 'source_sha256': sha(source), 'controller': CONTROLLER, 'run_status': 'not_run', 'blockers': []}
        if source.suffix == '.json':
            manifest = json.loads(source.read_text())
            if 'crownreach' in manifest['backends']:
                cfg = copy.deepcopy(manifest['backends']['crownreach']['config'])
                cfg['model_dir'] = cfg['model_dir'].replace('${ARCH_COMP_ROOT}', str(X / 'ARCH-COMP2024'))
            elif name == 'tora_homogeneous':
                yaml_source = X / 'experiments/flowstar_gpu_baseline/configs/tora_homogeneous_gf1_full.yaml'
                cfg = yaml.safe_load(yaml_source.read_text())
                cfg['model_dir'] = str((yaml_source.parent.parent / cfg['model_dir']).resolve())
                row['yaml_source'] = str(yaml_source)
                row['yaml_sha256'] = sha(yaml_source)
            else:
                row['blockers'].append('ACC exact derived 6D-to-5D input adapter is absent from this baseline generic driver')
                model = X / 'ARCH-COMP2024/benchmarks/ACC/controller_5_20.onnx'
                row.update(model=str(model), model_exists=model.exists(), model_sha256=sha(model), candidate_config=None)
                graph = onnx.load(str(model))
                onnx.checker.check_model(graph)
                row['onnx_checker'] = 'passed'
                row['onnx_ops'] = sorted({node.op_type for node in graph.graph.node})
                rows.append(row)
                continue
        else:
            cfg = yaml.safe_load(source.read_text())
            cfg['model_dir'] = str((source.parent.parent / cfg['model_dir']).resolve())
        model = Path(cfg['model_dir'])
        row.update(model=str(model), model_exists=model.exists(), model_sha256=sha(model) if model.exists() else None)
        if model.exists():
            graph = onnx.load(str(model))
            try:
                onnx.checker.check_model(graph)
                row['onnx_checker'] = 'passed'
            except Exception as exc:
                row['onnx_checker'] = repr(exc)
            row['onnx_ops'] = sorted({node.op_type for node in graph.graph.node})
            row['onnx_inputs'] = [x.name for x in graph.graph.input]
        if not model.exists(): row['blockers'].append('model_missing')
        h, period = float(cfg['ode_step_size']), float(cfg['step_size'])
        if h > period or not math.isclose(period / h, round(period / h), rel_tol=0, abs_tol=1e-10):
            row['blockers'].append('ode_step_does_not_divide_controller_period')
        declared = [x['name'] for x in cfg['initial_set']]
        grid = cfg.get('split_vars', [])
        B = math.prod(int(x.get('splits') or 1) for x in cfg['initial_set'] if x['name'] in grid)
        assert B == driver.make_cells(cfg).shape[0]
        row.update(B=B, order=cfg['ode_order'], h=h, controller_period=period, full_horizon=cfg['steps']*period,
                   full_controller_updates=cfg['steps'], cutoff=cfg['cut_off_threshold'], remainder=cfg['remainder_estimation'],
                   queue=driver.SR_QUEUE, queue_source='baseline driver hardcoded, not inferred from unspecified manifest',
                   input_shape=cfg['input_shape'], output_scale=cfg['output_scale'], output_offset=cfg['output_offset'],
                   split_vars=grid, split_vars_are_prefix=(grid == declared[:len(grid)]),
                   controller_bounds_options_requested=cfg.get('bound_opts'),
                   controller_bounds_options_actual={'activation_bound_option': 'same-slope'})
        if cfg.get('bound_opts', {}).get('conv_mode') is not None:
            row['blockers'].append('driver ignores requested bound_opts.conv_mode; controller configuration parity unqualified')
        try:
            for rhs in cfg['dynamics_expressions']: driver.parse(rhs, declared)
            row['plant_expression_parser'] = 'passed'
        except Exception as exc:
            row['plant_expression_parser'] = repr(exc)
            row['blockers'].append('plant_expression_parser_error')
        target = out / (name + '_full.yaml')
        target.write_text(yaml.safe_dump(cfg, sort_keys=False))
        row['candidate_config'] = str(target)
        row['candidate_config_sha256'] = sha(target)
        prefix = copy.deepcopy(cfg)
        prefix['steps'] = 1
        prefix_target = out / (name + '_period1.yaml')
        prefix_target.write_text(yaml.safe_dump(prefix, sort_keys=False))
        row['period1_config'] = str(prefix_target)
        row['period1_config_sha256'] = sha(prefix_target)
        row['prefix_scope'] = 'one complete controller period, all original partitions, not the full-horizon property'
        rows.append(row)
    packages = {}
    for name in ('torch', 'yaml', 'onnx', 'onnx2pytorch', 'auto_LiRPA'):
        mod = importlib.import_module(name)
        packages[name] = {'version': getattr(mod, '__version__', None), 'file': mod.__file__}
    import torch
    write(args.output / 'qualification.json', {'baseline_archive_source_sha': BASELINE_SHA, 'xiangru_sha': XIANGRU_SHA,
          'baseline_driver_sha256': sha(args.engine / 'integrations/crown_reach/gpu_driver.py'),
          'baseline_python_sha256': {str(p.relative_to(args.engine)): sha(p) for p in sorted((args.engine/'src/flowstar_gpu').glob('*.py'))},
          'controller': CONTROLLER, 'python': sys.executable, 'packages': packages, 'affinity': sorted(os.sched_getaffinity(0)),
          'torch_threads': torch.get_num_threads(), 'rows': rows})
    print(json.dumps([{'id': r['id'], 'B': r.get('B'), 'blockers': r['blockers']} for r in rows], indent=2))

def child(args):
    import yaml
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
        torch.cuda.set_per_process_memory_fraction(11*2**30/torch.cuda.get_device_properties(args.device).total_memory)
    def synchronize():
        if args.device.startswith('cuda'): torch.cuda.synchronize()
    started = time.perf_counter()
    d = load_driver(args.engine)
    cfg = yaml.safe_load(args.config.read_text())
    result = {'device': args.device, 'cuda_allocator_cap_bytes': 11*2**30 if args.device.startswith('cuda') else None, 'benchmark': args.benchmark, 'plant_mode': 'strict', 'controller': CONTROLLER,
              'script_sha256': sha(__file__), 'config': str(args.config), 'config_sha256': sha(args.config), 'scope': 'one controller period; original B retained',
              'full_horizon_certificate': False, 'timing': {'plant_advance_s': 0.0, 'controller_bound_s': 0.0, 'controller_setup_s': 0.0, 'observer_s': 0.0, 'trace_io_s': 0.0},
              'accepted_steps_all_lanes': 0, 'attempted_steps': 0, 'all_steps_accepted': True}
    trace = (args.output / (args.benchmark + '_trace.jsonl')).open('w')
    def wrap_time(name, field):
        original = getattr(d, name)
        def wrapped(*a, **kw):
            synchronize()
            t = time.perf_counter()
            try: return original(*a, **kw)
            finally:
                synchronize()
                result['timing'][field] += time.perf_counter() - t
        setattr(d, name, wrapped)
    wrap_time('build_crown', 'controller_setup_s')
    wrap_time('crown_bounds', 'controller_bound_s')
    advance = d.advance_sparse
    def observed(*a, **kw):
        synchronize()
        t = time.perf_counter()
        state, ok = advance(*a, **kw)
        synchronize()
        result['timing']['plant_advance_s'] += time.perf_counter() - t
        result['attempted_steps'] += 1
        result['all_steps_accepted'] &= bool(ok.all())
        result['accepted_steps_all_lanes'] += int(ok.all())
        t = time.perf_counter()
        eng, settings = a[2], a[4]
        h = settings.step
        tube = d.hull_ranges_s(state, eng, cfg['num_vars'])
        domain = torch.tensor([[h, h]] * state.pre.shape[0], dtype=torch.float64, device=args.device)
        direct_endpoint = d.rows_range_over_time_sparse(state, eng, domain, cfg['num_vars'])
        original_pre = state.pre.clone()
        endpoint_state = copy.copy(state)
        d.end_of_time_s(endpoint_state, eng)
        endpoint = d.hull_ranges_s(endpoint_state, eng, cfg['num_vars'])
        assert torch.allclose(state.pre, original_pre, rtol=0, atol=0, equal_nan=True), 'observer mutated solver state'
        record = {'ode_step': result['attempted_steps'], 'state_order': [x['name'] for x in cfg['initial_set']],
                  'accepted': ok.tolist(), 'status': state.status.tolist(), 'endpoint_definition': 'production end_of_time_s on shallow state copy, then hull_ranges_s', 'endpoint': endpoint.tolist(), 'tube': tube.tolist(), 'direct_interval_time_restriction_endpoint': direct_endpoint.tolist()}
        result['timing']['observer_s'] += time.perf_counter() - t
        t = time.perf_counter()
        def finite_json(value):
            if isinstance(value, list): return [finite_json(x) for x in value]
            if isinstance(value, float) and not math.isfinite(value): return None
            return value
        record = {k: finite_json(v) for k,v in record.items()}
        trace.write(json.dumps(record, allow_nan=False) + '\n'); trace.flush()
        result['timing']['trace_io_s'] += time.perf_counter() - t
        return state, ok
    d.advance_sparse = observed
    old_argv = sys.argv
    sys.argv = [str(args.engine / 'integrations/crown_reach/gpu_driver.py'), str(args.config), '--device', args.device, '--engine', 'sparse', '--strict', '--crown-domain', 'box', '--crown-relax', 'same-slope', '--print-final-hull', '--metrics-json', str(args.output / (args.benchmark + '_metrics.json'))]
    result['driver_argv'] = list(sys.argv)
    try:
        result['driver_returncode'] = d.main()
        result['expected_steps'] = round(cfg['step_size']/cfg['ode_step_size']) * cfg['steps']
        assert result['attempted_steps'] == result['expected_steps'] or not result['all_steps_accepted']
        result['status'] = 'plant_prefix_accepted_controller_unqualified' if result['all_steps_accepted'] else 'plant_prefix_rejected'
    except Exception as exc:
        result.update(status='error', error=repr(exc))
        traceback.print_exc()
    finally:
        sys.argv = old_argv
        trace.close()
        result['timing']['child_wall_s'] = time.perf_counter() - started
        result['timing']['unattributed_setup_bookkeeping_s'] = result['timing']['child_wall_s'] - sum(v for k,v in result['timing'].items() if k != 'child_wall_s')
        write(args.output / (args.benchmark + '_result.json'), result)
    return 0 if result['status'] != 'error' else 1

def run(args):
    assert args.benchmark in ('single_pendulum', 'tora_homogeneous')
    config = args.configs / (args.benchmark + '_period1.yaml')
    argv = ['taskset', '-c', '9', sys.executable, str(Path(__file__).resolve()), 'child', '--engine', str(args.engine), '--output', str(args.output), '--benchmark', args.benchmark, '--config', str(config)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    t = time.perf_counter()
    entry = {'argv': argv, 'timeout_s': args.timeout, 'environment': {k:env[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE')}}
    with (args.output/(args.benchmark+'_stdout.log')).open('w') as log:
        try:
            proc = subprocess.run(argv, cwd=args.output, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            entry.update(status='completed' if proc.returncode == 0 else 'error', returncode=proc.returncode)
        except subprocess.TimeoutExpired:
            entry.update(status='timeout', returncode=None)
    entry['process_wall_s'] = time.perf_counter()-t
    write(args.output/(args.benchmark+'_process.json'), entry)
    print(json.dumps(entry, indent=2))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare','run','child'])
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--source', type=Path)
    ap.add_argument('--configs', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--benchmark')
    ap.add_argument('--config', type=Path)
    ap.add_argument('--timeout', type=float, default=90)
    ap.add_argument('--device', default='cpu')
    args=ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    return {'prepare':prepare,'run':run,'child':child}[args.mode](args)

if __name__=='__main__': sys.exit(main())
