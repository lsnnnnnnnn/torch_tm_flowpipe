"""Finite sequential executions for the preregistered complete-engine decision."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .baseline import ROOT, PYTHON, gr_command, run_flowstar


def gpu_snapshot():
    return {name: subprocess.check_output(command, text=True).strip() for name, command in (
        ('gpus', ['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader']),
        ('processes', ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader']))}


def environment(run_root, gpu, engine_root=None):
    compiler = Path('/srv/local/shengenli/.huan-audit-gxx13/bin')
    engine_root = Path(engine_root).resolve() if engine_root else run_root / 'engine'
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                PYTHONPATH=f'{engine_root}/src:{ROOT}/src:{ROOT}',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                PYTHONDONTWRITEBYTECODE='1', CUDA_HOME='/usr/local/cuda-12.6',
                TORCH_EXTENSIONS_DIR=str(run_root / 'cache/py311_torch251_cu126_gcc13'),
                CXX=str(compiler / 'x86_64-conda-linux-gnu-g++'),
                CC=str(compiler / 'x86_64-conda-linux-gnu-gcc'),
                TORCH_CUDA_ARCH_LIST='7.0', MAX_JOBS='2')


def candidate_command(plant, batch, steps, output, record):
    result = ['taskset', '-c', '2', str(PYTHON), '-m',
              'experiments.whole_engine_feasibility.candidate',
              '--plant', plant, '--batch', str(batch), '--steps', str(steps),
              '--output', str(output), '--progress-every', '20', '--timeout-s', '3600']
    return result + (['--record'] if record else [])


def execute(command, env, output, *, formal=False):
    before = gpu_snapshot()
    if formal:
        selected = next(row for row in before['gpus'].splitlines()
                        if row.split(',')[0].strip() == env['CUDA_VISIBLE_DEVICES'])
        memory = int(selected.split(',')[-2].strip().split()[0])
        if memory > 10:
            raise RuntimeError(f'formal GPU is occupied; no run started: {selected}')
    receipt = {'command': command, 'before': before, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'gpu': env['CUDA_VISIBLE_DEVICES'], 'formal': formal}
    print(json.dumps({'event': 'start', 'output': str(output), 'command': command}), flush=True)
    log = output.with_suffix('.log')
    start = time.perf_counter()
    with log.open('w') as stream:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=3660)
            receipt['returncode'] = result.returncode
        except subprocess.TimeoutExpired:
            receipt.update(returncode=None, timeout=True)
    receipt['process_wall_s'] = time.perf_counter() - start
    receipt['after'] = gpu_snapshot()
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'event': 'finish', 'output': str(output), 'returncode': receipt['returncode'],
                      'process_wall_s': receipt['process_wall_s']}), flush=True)
    if receipt['returncode'] != 0:
        raise RuntimeError(f'run failed; see {log}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['continuity', 'horizon', 'timing', 'flowstar'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--engine-root', type=Path,
                        help='external engine checkout; defaults to the original experiment engine')
    args = parser.parse_args()
    root = args.run_root.resolve()
    env = environment(root, args.gpu, args.engine_root)
    plants = ['van_der_pol', 'brusselator']
    if args.phase in ('continuity', 'horizon'):
        batch, steps = (2, 120) if args.phase == 'continuity' else (1, 1000)
        for plant in plants:
            output = root / 'runs' / f'{args.phase}_{plant}'
            execute(candidate_command(plant, batch, steps, output, True), env, output)
    elif args.phase == 'timing':
        for plant in plants:
            for pair in range(1, 4):
                order = ['Gr', 'candidate'] if pair % 2 else ['candidate', 'Gr']
                for route in order:
                    output = root / 'runs' / f'timing_{plant}_pair{pair}_{route}'
                    if route == 'Gr':
                        command, gr_env = gr_command(plant, 32, 20, output)
                        gr_env.update({k: env[k] for k in ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
                                      'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']})
                    else:
                        command, gr_env = candidate_command(plant, 32, 20, output, False), env
                    execute(command, gr_env, output, formal=True)
    else:
        binary = None
        for plant in plants:
            output = root / 'runs' / f'flowstar_b32x20_{plant}'
            result = run_flowstar(plant, output, binary=binary)
            binary = result['binary']
            print(json.dumps({'event': 'flowstar_complete', 'plant': plant,
                              'summary': result['summary']}), flush=True)


if __name__ == '__main__':
    main()
