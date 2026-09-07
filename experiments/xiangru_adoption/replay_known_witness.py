"""Replay the preserved operands through the candidate's actual SR function."""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--witness', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    import flowstar_gpu
    from flowstar_gpu import flowpipe, sparse_exec, symbolic_remainder as sr, cuda_kernels as ck
    from flowstar_gpu.config import Settings
    from flowstar_gpu.determinism import enable_determinism
    enable_determinism()
    torch.set_num_threads(1)
    root = args.candidate.resolve()
    sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()
    assert not dirty, dirty
    assert Path(sr.__file__).resolve().is_relative_to(root / 'src')
    assert flowpipe.propagate is sr.propagate and sparse_exec.propagate is sr.propagate
    assert ck.available(), 'Must run current compiled CUDA kernels, not silent fallback'
    witness_bytes = args.witness.read_bytes()
    witness = json.loads(witness_bytes)
    matrices, columns = witness['matrices'], witness['historical_j_columns']
    phis, js, exact = [], [], [Fraction(), Fraction()]
    for k, matrix in enumerate(matrices):
        a = [[Fraction(v) for v in row] for row in matrix]
        for q in range(1, len(phis)):
            b = phis[q]
            phis[q] = [[sum((a[i][j]*b[j][l] for j in range(2)), Fraction())
                         for l in range(2)] for i in range(2)]
        phis.append(a)
        exact = [sum((phis[q][i][j]*js[q-1][j]
                      for q in range(1,len(phis)) for j in range(2)), Fraction()) for i in range(2)]
        if k < len(columns): js.append([Fraction(v) for v in columns[k]])
    assert [str(v) for v in exact] == witness['exact_result_fraction']
    assert str(exact[1]) == '-4503599627370497/38685626227668133590597632'
    results = []
    for device in ('cpu', 'cuda'):
        for mode in ('parity', 'strict'):
            settings = Settings(step=0.02, order=6, sr_queue=1000, mode=mode, device=device)
            state = sr.make_symbolic_remainder(1, 2, settings.sr_queue, device)
            inputs = [torch.tensor([m], dtype=torch.float64, device=device) for m in matrices]
            j_inputs = [torch.tensor([[[v, v] for v in col]], dtype=torch.float64, device=device) for col in columns]
            if device == 'cuda': torch.cuda.synchronize()
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device == 'cuda': activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities) as prof:
                for k, phi in enumerate(inputs):
                    actual = sr.propagate(state, phi)
                    if k < len(j_inputs): state.append_j(j_inputs[k])
                if device == 'cuda': torch.cuda.synchronize()
            bounds = actual[0].cpu().tolist()
            contained = [Fraction(lo) <= v <= Fraction(hi) for (lo, hi), v in zip(bounds, exact)]
            kernels = sorted({event.name for event in prof.events()
                              if event.device_type == torch.autograd.DeviceType.CUDA})
            if device == 'cuda': assert kernels, 'No actual CUDA work recorded'
            results.append({'device': device, 'requested_mode': mode, 'parsed_mode': settings.mode,
                            'mode_dispatch': 'Both production advance functions call this same mode-independent SR function.',
                            'bounds': bounds, 'bounds_hex': [[v.hex() for v in pair] for pair in bounds],
                            'contains_exact': contained, 'all_contained': all(contained),
                            'cuda_kernel_names': kernels, 'qlen': state.qlen, 'jlen': state.jlen})
    result = {'schema': 'xiangru_known_witness_replay/1', 'source_sha': sha,
              'candidate_root': str(root), 'source_clean': True,
              'python_executable': sys.executable, 'python_version': sys.version,
              'torch_version': torch.__version__, 'torch_file': torch.__file__,
              'imported_package': flowstar_gpu.__file__, 'source_function': inspect.getsourcefile(sr.propagate),
              'source_line': inspect.getsourcelines(sr.propagate)[1],
              'callers': {'flowpipe.advance_uses_same_function': True, 'sparse_exec.advance_sparse_uses_same_function': True},
              'extension_path': ck._ext.__file__, 'cpu_threads': torch.get_num_threads(),
              'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'original_witness_sha256': hashlib.sha256(witness_bytes).hexdigest(),
              'original_replay_script': 'scripts/huan_strict_roundoff_audit.py at our frozen evidence base',
              'matrices': matrices, 'historical_j_columns': columns,
              'exact_result_fraction': [str(v) for v in exact], 'runs': results,
              'classification': 'CONFIRMED_UNDERENCLOSURE' if any(not r['all_contained'] for r in results) else 'WITNESS_CONTAINED',
              'scope': 'Local history transform inclusion test with original operands; does not assert these operands occur on a particular frozen trajectory.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'classification':result['classification'], 'runs':[
        {k:r[k] for k in ['device','parsed_mode','bounds','contains_exact','cuda_kernel_names']} for r in results]},indent=2))

if __name__ == '__main__': main()
