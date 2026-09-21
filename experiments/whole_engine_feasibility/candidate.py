"""Whole sparse strict engine on frozen plant tasks; no CPU range-service calls.

Run via PYTHONPATH=<isolated engine>/src:src:. . Observation is a separate
operation on complete saved factored models, never part of a timing-only run.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import torch

from .baseline import frozen_case


def source_identity(path):
    root = Path(path).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    return {'root': str(root), 'head': git('rev-parse', 'HEAD'),
            'status': git('status', '--porcelain'), 'diff': git('diff', '--stat'),
            'python_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((root / 'src/flowstar_gpu').glob('*.py'))}}


def save_factored(state, tab, *, step, h, lane_ids, accepted):
    """Lossless CPU snapshot: no composing, range evaluation or coefficient cutoff."""
    spatial = tab.exponents[tab.spatial_index].detach().cpu().tolist()
    full = tab.exponents.detach().cpu().tolist()
    return {'step': step, 'h_hex': h.hex(), 'lane_ids': lane_ids,
            'accepted': accepted,
            'pre_exponents': [full[i] for i in state.pre_sup.ids],
            'tmv_exponents': [spatial[i] for i in state.tmv_sup.ids],
            'pre': state.pre.detach().cpu().tolist(),
            'pre_rem': state.pre_rem.detach().cpu().tolist(),
            'tmv': state.tmv.detach().cpu().tolist(),
            'tmv_rem': state.tmv_rem.detach().cpu().tolist()}


def advance_transaction(state, sr, code, eng, sched, settings, rem):
    """Commit the whole batch only if every lane validates; otherwise retain inputs.

    The engine freezes failed flowpipe lanes but mutates shared SR buffers.
    A speculative SR copy gives an actual rollback boundary. Copy cost is part
    of solve time. There is no retry, lane revival or hidden step-size change.
    """
    from flowstar_gpu.sparse_exec import advance_sparse, prune_state
    pending_sr = copy.deepcopy(sr)
    pending, ok = advance_sparse(state, code, eng, sched, settings, rem, pending_sr)
    accepted = ok.detach().cpu().tolist()
    if not all(accepted):
        return state, sr, accepted, pending.status.detach().cpu().tolist(), False
    pending = prune_state(pending, eng)
    reset = pending_sr.reset_if_full()
    return pending, pending_sr, accepted, pending.status.detach().cpu().tolist(), reset


def run(args):
    # Select before importing the engine; importing a differently configured
    # engine in the same process is rejected below rather than misreported.
    for option, variable in [('composition', 'FLOWSTAR_COMPOSITION'), ('glue', 'FLOWSTAR_GLUE')]:
        value = getattr(args, option, None)
        if value is not None:
            os.environ[variable] = value
    if args.engine_root is not None:
        engine_src = args.engine_root.resolve() / 'src'
        if not (engine_src / 'flowstar_gpu' / '__init__.py').is_file():
            raise FileNotFoundError(f'flowstar_gpu source is missing from {engine_src}')
        sys.path.insert(0, str(engine_src))
    import flowstar_gpu
    if args.engine_root is not None:
        actual = Path(flowstar_gpu.__file__).resolve()
        if not actual.is_relative_to(engine_src):
            raise RuntimeError(f'wrong flowstar_gpu import: {actual}')
    from flowstar_gpu.config import Settings
    from flowstar_gpu.determinism import enable_determinism
    from flowstar_gpu.monomials import build_tables
    from flowstar_gpu.polynomial import build_step_tables
    from flowstar_gpu.composition import build_schedule
    from flowstar_gpu.ode_compiler import compile_ode
    from flowstar_gpu.flowpipe import build_rem_est
    from flowstar_gpu.support import SparseEngine
    from flowstar_gpu.sparse_exec import initial_sparse_state
    from flowstar_gpu.symbolic_remainder import make_symbolic_remainder
    from flowstar_gpu import cuda_kernels, tape_kernels, sparse_exec, glue
    try:
        from flowstar_gpu import sr_kernels
    except ImportError:
        sr_kernels = None  # Older explicitly selected baseline checkouts.
    algorithms = {'composition': getattr(sparse_exec, 'COMPOSITION_MODE', 'monomial'),
                  'glue': glue.GLUE_MODE,
                  'rhs_form': getattr(args, 'rhs_form', 'original'),
                  'sr_interval_update': 'cuda_directed_if_supported' if sr_kernels else 'broadcast_interval'}
    for option in ('composition', 'glue'):
        requested = getattr(args, option, None)
        if requested is not None and algorithms[option] != requested:
            raise RuntimeError(f'engine already imported with a different {option}: {algorithms[option]}')

    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    case = frozen_case(args.plant, args.batch)
    rhs = list(case['rhs_expression_strings'])
    if algorithms['rhs_form'] == 'regrouped':
        if args.plant != 'van_der_pol':
            raise ValueError('regrouped RHS currently qualified only for van_der_pol')
        rhs = ['y', '(1-x*x)*y-x']
    algorithms['rhs_executed'] = rhs
    algorithms['rhs_contract'] = case['rhs_expression_strings']
    # This compatibility normalization accepts the explicit helper contract.
    h = float(case['h'])
    sync = (lambda: torch.cuda.synchronize()) if args.device == 'cuda' else (lambda: None)
    cold = time.perf_counter()
    enable_determinism(args.device)
    extensions = {'interval': False, 'tape': False, 'validation': False}
    if args.device == 'cuda':
        # Validation has a separate lazy extension; compile it before the solve timer.
        extensions = {'interval': cuda_kernels.available(), 'tape': tape_kernels.available(),
                      'validation': tape_kernels.valid_available()}
        if sr_kernels is not None:
            extensions['sr_interval'] = sr_kernels.available()
        if not all(extensions.values()):
            raise RuntimeError(f'candidate CUDA extensions unavailable: {extensions}')
    sync()
    cold_s = time.perf_counter() - cold
    started = time.perf_counter()
    settings = Settings(step=h, order=case['order'], cutoff=case['cutoff'],
                        remainder_estimation=case['remainder_estimation'],
                        sr_queue=case['sr_capacity'], mode='strict', device=args.device,
                        max_refinement_steps=case['refinement_limit'], stop_ratio=case['stop_ratio'])
    tab = build_tables(2, settings.order).to(args.device)
    table_step = build_step_tables(tab, h)
    sched = build_schedule(2, settings.order, args.device)
    code = compile_ode(rhs, ['x', 'y'], order=settings.order - 1)
    boxes = torch.tensor(case['boxes'], dtype=torch.float64, device=args.device)
    eng = SparseEngine(tab, table_step, args.device)
    state = initial_sparse_state(boxes, eng, sched)
    sr = make_symbolic_remainder(args.batch, 2, settings.sr_queue, args.device)
    rem = build_rem_est(settings, 2, args.batch)
    sync()
    initial_s = time.perf_counter() - started
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    excluded_s = 0.0
    counts = [0] * args.batch
    statuses = [0] * args.batch
    reset_steps, attempts = [], []
    failure = None
    raw = gzip.open(args.output / 'factored.jsonl.gz', 'wt') if args.record else None
    try:
        for index in range(1, args.steps + 1):
            step_start = time.perf_counter()
            state, sr, accepted, statuses, reset = advance_transaction(
                state, sr, code, eng, sched, settings, rem)
            sync()
            dt = time.perf_counter() - step_start
            if not all(accepted):
                failure = {'step': index, 'engine_accepted': accepted,
                           'status': statuses, 'batch_committed': False}
                break
            counts = [index] * args.batch
            if reset:
                reset_steps.append(index)
            if raw is not None:
                export_start = time.perf_counter()
                record = save_factored(state, tab, step=index, h=h,
                                       lane_ids=case['lane_ids'], accepted=accepted)
                raw.write(json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n')
                raw.flush()
                excluded_s += time.perf_counter() - export_start
            if args.record:
                attempts.append({'step': index, 'solve_step_s': dt, 'queue_len': sr.queue_len,
                                 'reset': reset})
            if index % args.progress_every == 0 or index == args.steps:
                # Progress I/O is outside the numerical timer in recorded runs.
                log_start = time.perf_counter()
                print(json.dumps({'plant': args.plant, 'batch': args.batch, 'step': index,
                                  'solve_s': time.perf_counter() - started - excluded_s}), flush=True)
                excluded_s += time.perf_counter() - log_start
            if args.device == 'cuda' and torch.cuda.memory_allocated() > args.max_gpu_gib * 2**30:
                failure = {'step': index, 'reason': 'gpu_memory_budget'}
                break
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > args.max_cpu_gib * 2**20:
                failure = {'step': index, 'reason': 'cpu_memory_budget'}
                break
            if time.perf_counter() - started > args.timeout_s:
                failure = {'step': index, 'reason': 'wall_time_budget'}
                break
        sync()
        wall_s = time.perf_counter() - started - excluded_s
    finally:
        if raw is not None:
            raw.close()
    engine_root = Path(flowstar_gpu.__file__).resolve().parents[2]
    result = {'schema': 'whole-engine-feasibility/1', 'plant': args.plant,
              'batch': args.batch, 'requested_steps': args.steps, 'accepted_steps': counts,
              'accepted_lane_steps': sum(counts), 'completed': min(counts) == args.steps,
              'failure': failure, 'engine_status': statuses, 'settings': asdict(settings),
              'case': case, 'algorithms': algorithms, 'route': 'Huan_sparse_strict', 'device': args.device,
              'solve_wall_s': wall_s, 'wall_s': wall_s, 'cold_startup_s': cold_s,
              'initial_state_and_plan_s': initial_s, 'recording_s': excluded_s,
              'timing_only': not args.record, 'sr_reset_steps': reset_steps,
              'sr_commit': 'speculative full SR tensor copy; atomic batch commit; stop at first failed batch',
              'timer': 'tables/code/initialization + every full advance + SR copies/commit + synchronization + necessary status return; excludes extension startup and optional model recording',
              'observer': 'none during solve; complete factored states saved when --record',
              'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
              'cuda_build': torch.version.cuda, 'extensions': extensions,
              'engine_source': source_identity(engine_root),
              'adapter_source': source_identity(Path(__file__).resolve().parents[2]),
              'affinity': sorted(os.sched_getaffinity(0)), 'torch_threads': torch.get_num_threads(),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'gpu_name': torch.cuda.get_device_name() if args.device == 'cuda' else None,
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0,
              'limits': {'timeout_s': args.timeout_s, 'max_gpu_gib': args.max_gpu_gib,
                         'max_cpu_gib': args.max_cpu_gib},
              'attempts': attempts}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['plant', 'batch', 'completed', 'accepted_steps', 'failure', 'solve_wall_s', 'cold_startup_s']}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plant', choices=['van_der_pol', 'brusselator'], required=True)
    parser.add_argument('--engine-root', type=Path,
                        help='explicit external engine checkout; recorded and checked after import')
    parser.add_argument('--batch', type=int, choices=[1, 2, 32], required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--composition', choices=['monomial', 'horner'])
    parser.add_argument('--glue', choices=['eager', 'compile', 'graph'])
    parser.add_argument('--rhs-form', choices=['original', 'regrouped'], default='original')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--progress-every', type=int, default=100)
    parser.add_argument('--timeout-s', type=float, default=3600)
    parser.add_argument('--max-gpu-gib', type=float, default=12)
    parser.add_argument('--max-cpu-gib', type=float, default=16)
    args = parser.parse_args()
    if args.steps < 1 or args.progress_every < 1:
        parser.error('steps/progress-every must be positive')
    run(args)


if __name__ == '__main__':
    main()
