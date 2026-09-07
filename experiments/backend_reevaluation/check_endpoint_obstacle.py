"""Bounded check of a missing prerequisite encountered in advance's caller.

No ODE, history update, composition, preconditioning or integration is run.
Observe the actual accepted-endpoint substitution and its remainder ownership.
"""
import argparse
from fractions import Fraction as F
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys


class EndpointObserved(Exception):
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--implementation", choices=["A", "B"], required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); args.source = args.source.resolve()
    expected_sha = {"A": "1c16d4ef2cb91cc94b1c784f7383e1eba135d8d3",
                    "B": "743f6205e6408072193ad76e940e7f15030e8d3c"}[args.implementation]
    git = lambda *a: subprocess.check_output(["git", "-C", str(args.source), *a], text=True).strip()
    assert git("rev-parse", "HEAD") == expected_sha and not git("status", "--porcelain")
    sys.path.insert(0, str(args.source / "src"))
    import torch
    import flowstar_gpu
    from flowstar_gpu import flowpipe as fp, sparse_exec as se, support as sp, cuda_kernels as ck
    from flowstar_gpu.config import Settings
    from flowstar_gpu.determinism import enable_determinism
    assert Path(flowstar_gpu.__file__).resolve().is_relative_to(args.source / "src")
    enable_determinism(); torch.set_num_threads(1)
    assert ck.available()
    rows = []
    for device in ["cpu", "cuda"]:
        for backend in ["dense", "sparse"]:
            settings = Settings(step=.01, order=4, sr_queue=100, mode="strict", device=device)
            tab = fp.build_tables(2, settings.order).to(device)
            step = fp.build_step_tables(tab, settings.step)
            sched = fp.build_schedule(2, settings.order, device)
            eng = sp.SparseEngine(tab, step, device)
            # Both components are -1 + 100*t with zero remainder. Spatial
            # coordinates remain independent in [-1,1], with zero coefficients.
            box = torch.zeros(1, 2, 2, dtype=torch.float64, device=device)
            dense = fp.initial_flowpipe(box, tab)
            pre = torch.zeros_like(dense.pre_coeffs)
            exponents = tab.exponents.cpu().tolist()
            time_id = exponents.index([1, 0, 0])
            pre[..., 0] = -1.; pre[..., time_id] = 100.
            dense.pre_coeffs = pre; dense.pre_rem.zero_()
            if backend == "dense":
                state, fn = dense, fp.advance
                call = lambda: fp.advance(state, None, tab, step, sched, settings, fp.build_rem_est(settings, 2, 1))
                marker = "c0 = x0_sp_full"
            else:
                state = se.initial_sparse_state(box, eng, sched)
                support = sp.make_support(2, settings.order, False, (0, time_id))
                state.pre, state.pre_sup = pre[..., list(support.ids)], support
                state.pre_rem.zero_()
                fn = se.advance_sparse
                call = lambda: fn(state, None, eng, sched, settings, fp.build_rem_est(settings, 2, 1))
                marker = "c0 = x0f"
            lines, first = inspect.getsourcelines(fn)
            stop = next(first+i for i, line in enumerate(lines) if marker in line)
            observed = {}
            def trace(frame, event, arg):
                if frame.f_code is fn.__code__ and event == "line" and frame.f_lineno == stop:
                    observed.update(frame.f_locals)
                    raise EndpointObserved()
                return trace
            sys.settrace(trace)
            try: call()
            except EndpointObserved: pass
            finally: sys.settrace(None)
            assert observed, "Real endpoint stage not reached"
            c = observed["x0_sp_full" if backend == "dense" else "x0f"][0]
            remainder = observed.get("x0_rem", state.pre_rem)[0]
            assert torch.count_nonzero(c[..., 1:]) == 0
            target = F(-1.)+F(100.)*F(settings.step)
            bounds = [[F(float(c[i, 0]))+F(float(remainder[i, 0])),
                       F(float(c[i, 0]))+F(float(remainder[i, 1]))] for i in range(2)]
            rows.append(dict(implementation=args.implementation, backend=backend, device=device,
                             requested_mode="strict", actual_mode=settings.mode,
                             source_file=str(Path(inspect.getsourcefile(fn)).relative_to(args.source)),
                             observed_line=stop, function=fn.__name__,
                             pre_terms=[{"degrees": [0, 0, 0], "coefficient": -1.},
                                        {"degrees": [1, 0, 0], "coefficient": 100.}],
                             pre_remainders=[[0., 0.], [0., 0.]], h=settings.step,
                             endpoint_coefficients=c[:, 0].cpu().tolist(),
                             endpoint_remainders=remainder.cpu().tolist(),
                             exact_image=[[str(target), str(target)]]*2,
                             returned_image=[[str(v) for v in pair] for pair in bounds],
                             contains=all(lo <= target <= hi for lo, hi in bounds),
                             mapping="Exact binary64 polynomial -1+100*t evaluated at binary64(0.01); singleton endpoint image in each component.",
                             call_chain=[fn.__name__, "evaluate_time_end_s" if backend == "sparse" else "evaluate_time_end"],
                             no_history_no_scaling=True))
    result = dict(schema="backend_reevaluation_endpoint_obstacle/1", source_sha=git("rev-parse", "HEAD"),
                  source_clean=not git("status", "--porcelain"), source_path=str(args.source),
                  driver_sha=subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True).strip(),
                  imported_package=flowstar_gpu.__file__, python=sys.executable, torch_version=torch.__version__,
                  segment_extension=ck._ext.__file__, rows=rows,
                  scope="Local endpoint prerequisite only. No third dynamical system or trajectory run; no numerical repair.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps([{k: r[k] for k in ["implementation", "device", "backend", "exact_image", "returned_image", "contains"]} for r in rows], indent=2))


if __name__ == "__main__":
    main()
