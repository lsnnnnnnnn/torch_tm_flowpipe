"""Two frozen local maps, through the pinned implementations' production stages.

Input adapters replace only stages *before* the map under test. A trace or an
observer stops immediately after that map. This is a local inclusion experiment,
not a claim that the operands occur in either benchmark's trajectory.
"""
import argparse
from fractions import Fraction as F
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

SHAS = {
    "A": "1c16d4ef2cb91cc94b1c784f7383e1eba135d8d3",
    "B": "743f6205e6408072193ad76e940e7f15030e8d3c",
    "C": "4939fb288c941a67f55cc191f4d75f8594692f47",
}


class StageDone(Exception):
    pass


def bounds_of(interval):
    return [float(interval.lo), float(interval.hi)]


def exact_history(matrices, columns, scales):
    phis, js, outputs = [], [], []
    for k, matrix in enumerate(matrices):
        a = [[F(v) / F(scales[k][j]) for j, v in enumerate(row)] for row in matrix]
        for q in range(1, len(phis)):
            b = phis[q]
            phis[q] = [[sum((a[i][j]*b[j][l] for j in range(2)), F())
                        for l in range(2)] for i in range(2)]
        phis.append(a)
        outputs.append([sum((phis[q][i][j]*js[q-1][j]
                             for q in range(1, len(phis)) for j in range(2)), F())
                        for i in range(2)])
        if k < len(columns):
            js.append(list(map(F, columns[k])))
    return outputs


def finish_history(row, matrices, columns, scales, outputs):
    exact = exact_history(matrices, columns, scales)
    row.update(matrices=matrices, columns=columns, scales=scales, stages=outputs,
               exact_stages=[[str(v) for v in stage] for stage in exact],
               exact_image=[[str(v), str(v)] for v in exact[-1]],
               returned_image=[[str(F(v)) for v in pair] for pair in outputs[-1]],
               contains=all(F(lo) <= v <= F(hi)
                            for got, want in zip(outputs, exact)
                            for (lo, hi), v in zip(got, want)))
    return row


def finish_normalization(row, coefficients, remainders, scales, **extra):
    target = F(1.1) + F(2.0**-52)
    images = []
    for c, rem, s in zip(coefficients, remainders, scales):
        radius = sum(map(lambda v: abs(F(v)), c[1:]), F())
        images.append([F(s)*(F(c[0])-radius+F(rem[0])),
                       F(s)*(F(c[0])+radius+F(rem[1]))])
    row.update(input_coefficients=[[0., 1.1, 0.], [0., 0., 1.1]],
               input_remainders=[[-2.0**-52, 2.0**-52]]*2,
               domain=[[-1., 1.], [-1., 1.]],
               retained_coefficients=coefficients, scaled_remainders=remainders,
               scales=scales, exact_image=[[str(-target), str(target)]]*2,
               returned_image=[[str(v) for v in pair] for pair in images],
               upper_margins=[str(hi-target) for lo, hi in images],
               contains=all(lo <= -target and hi >= target for lo, hi in images), **extra)
    return row


def candidate_rows(args, witness):
    import torch
    from flowstar_gpu import flowpipe as fp, sparse_exec as se, support as sp
    from flowstar_gpu import interval as iv, polynomial as poly, symbolic_remainder as sr
    from flowstar_gpu import cuda_kernels as ck
    from flowstar_gpu.config import Settings
    from flowstar_gpu.determinism import enable_determinism
    enable_determinism()
    torch.set_num_threads(1)
    rows = []
    for device in args.devices:
        if device == "cuda":
            assert ck.available(), "Explicit CUDA path requires its own compiled extension"
        t = lambda v: torch.tensor(v, dtype=torch.float64, device=device)
        for backend in ["dense", "sparse"]:
            for mode in ["parity", "strict"]:
                settings = Settings(step=.01, order=4, sr_queue=100, mode=mode, device=device)
                tab = fp.build_tables(2, settings.order).to(device)
                step = fp.build_step_tables(tab, settings.step)
                sched = fp.build_schedule(2, settings.order, device)
                eng = sp.SparseEngine(tab, step, device)
                box = t([[[-1., 1.], [-1., 1.]]])
                state = fp.initial_flowpipe(box, tab) if backend == "dense" else se.initial_sparse_state(box, eng, sched)
                module, fn = (fp, fp.advance) if backend == "dense" else (se, se.advance_sparse)
                call = (lambda queue: fp.advance(state, None, tab, step, sched, settings,
                                                  fp.build_rem_est(settings, 2, 1), queue)) if backend == "dense" else (
                        lambda queue: se.advance_sparse(state, None, eng, sched, settings,
                                                        fp.build_rem_est(settings, 2, 1), queue))
                base = dict(implementation=args.implementation, device=device, backend=backend,
                            requested_mode=mode, actual_mode=settings.mode,
                            source_file=str(Path(inspect.getsourcefile(fn)).relative_to(args.source)),
                            function=fn.__name__, call_chain=[fn.__name__],
                            batch=1, dtype="float64", local_adapter=True)
                # Exercise actual new-Phi formation, queue multiplication and image
                # accumulation. Endpoint evaluation is the input seam, not under test.
                for variant, scales in [("original", [[1., 1.]]*3),
                                        ("nonunit_scale", [[1., 1.], [1.1, 1.1], [1.1, 1.1]])]:
                    queue = sr.make_symbolic_remainder(1, 2, 100, device)
                    outputs, formed = [], []
                    for k, matrix in enumerate(witness["matrices"]):
                        a = t([matrix]); inv = 1. / t([scales[k]])
                        queue.scalars = inv
                        if hasattr(queue, "scalars_iv"):
                            queue.scalars_iv = iv.rec(iv.from_point(t([scales[k]])))[0]
                        c = torch.zeros(1, 2, tab.Ts, dtype=torch.float64, device=device)
                        c[..., sched.var_image] = a
                        def observe(q, phi, **kwargs):
                            out = sr.propagate(q, phi, **kwargs)
                            outputs.append(out[0].cpu().tolist())
                            formed.append({"point": phi[0].cpu().tolist(),
                                           "interval": kwargs["phi_i_iv"][0].cpu().tolist()
                                           if kwargs.get("phi_i_iv") is not None else None})
                            raise StageDone()
                        if backend == "dense":
                            patches = [patch.object(poly, "evaluate_time_end", return_value=c)]
                            if hasattr(poly, "evaluate_time_end_with_roundoff"):
                                patches.append(patch.object(poly, "evaluate_time_end_with_roundoff",
                                                            return_value=(c, torch.zeros(1, 2, 2, dtype=torch.float64, device=device))))
                        else:
                            support = sp.make_support(2, settings.order, True, tuple(range(tab.Ts)))
                            full_ids = tuple(int(v) for v in tab.spatial_index.cpu().tolist())
                            full = sp.make_support(2, settings.order, False, full_ids)
                            patches = [patch.object(sp, "evaluate_time_end_s", return_value=(c, full))]
                            if hasattr(sp, "evaluate_time_end_s_with_roundoff"):
                                patches.append(patch.object(sp, "evaluate_time_end_s_with_roundoff",
                                                            return_value=(c, full, torch.zeros(1, 2, 2, dtype=torch.float64, device=device))))
                        from contextlib import ExitStack
                        with ExitStack() as stack:
                            for p in patches: stack.enter_context(p)
                            stack.enter_context(patch.object(module, "propagate", side_effect=observe))
                            try: call(queue)
                            except StageDone: pass
                            else: raise AssertionError("History observer did not execute")
                        if k < len(witness["historical_j_columns"]):
                            queue.append_j(t([[[v, v] for v in witness["historical_j_columns"][k]]]))
                    row = dict(base, witness="history", variant=variant,
                               call_chain=[fn.__name__, "symbolic_remainder.propagate"], formed_matrices=formed,
                               adapter="Supply exact endpoint coefficients; observe real Phi formation and propagate; append original J columns.")
                    rows.append(finish_history(row, witness["matrices"], witness["historical_j_columns"], scales, outputs))

                # Supply the centered map at the compose output seam; execute all
                # normalization and SR cutoff logic, then stop before Picard.
                queue = sr.make_symbolic_remainder(1, 2, 100, device)
                c = torch.zeros(1, 2, tab.Ts, dtype=torch.float64, device=device)
                c[:, torch.arange(2, device=device), sched.var_image] = 1.1
                rem = t([[[-2.0**-52, 2.0**-52]]*2])
                if backend == "dense":
                    compose_patch = patch.object(fp, "compose", return_value=(c, rem))
                    marker = "new_x0 = torch.zeros("
                else:
                    support = sp.make_support(2, settings.order, True, tuple(range(tab.Ts)))
                    compose_patch = patch.object(se, "compose_s", return_value=(c, rem, support))
                    marker = "x, sup_x = new_x0, sup_x0n"
                lines, first = inspect.getsourcelines(fn)
                stop = next(first+i for i, line in enumerate(lines) if marker in line)
                captured = {}
                def trace(frame, event, arg):
                    if frame.f_code is fn.__code__ and event == "line" and frame.f_lineno == stop:
                        captured.update(frame.f_locals)
                        raise StageDone()
                    return trace
                with compose_patch:
                    sys.settrace(trace)
                    try: call(queue)
                    except StageDone: pass
                    finally: sys.settrace(None)
                assert captured, "Normalization observer did not execute"
                out_c = captured["new_tmv_coeffs" if backend == "dense" else "new_tmv"][0]
                if backend == "sparse":
                    out_ids = list(captured["sup_nt"].ids)
                    coefficients = [[float(out_c[i, out_ids.index(j)]) if j in out_ids else 0.
                                     for j in [0, *sched.var_image.cpu().tolist()]] for i in range(2)]
                else:
                    coefficients = out_c[:, [0, *sched.var_image.cpu().tolist()]].cpu().tolist()
                row = dict(base, witness="normalization", variant="centered_affine",
                           adapter="Compose-output input seam; observe after strict normalization and SR cutoff, before next Picard stage.",
                           observed_line=stop)
                rows.append(finish_normalization(row, coefficients, captured["new_tmv_rem"][0].cpu().tolist(),
                                                 captured["S"][0].cpu().tolist(),
                                                 inverse=queue.scalars[0].cpu().tolist(),
                                                 inverse_interval=queue.scalars_iv[0].cpu().tolist()
                                                 if getattr(queue, "scalars_iv", None) is not None else None,
                                                 current_j=queue.j_buf[0, 0].cpu().tolist()))
                print(json.dumps({"implementation": args.implementation, "device": device,
                                  "backend": backend, "mode": mode,
                                  "normalization_contains": rows[-1]["contains"]}), flush=True)
    return rows, {"segment_extension": ck._ext.__file__ if ck._ext else None}


def our_rows(args, witness):
    from torch_tm_flowpipe import Interval, Polynomial, TaylorModel, TMVector
    from torch_tm_flowpipe import accepted_boundary_sr as ab, symbolic_remainder as sr
    from torch_tm_flowpipe.flowpipe import _tmvector_range_box_normal, _interval_magnitude
    base = dict(implementation="C", device="cpu", backend="accepted_boundary_sr",
                requested_mode="our_cpu_reference", actual_mode="our_cpu_reference", batch=1,
                dtype="float64", local_adapter=True)
    rows = []
    for variant, scales in [("original", [[1., 1.]]*3),
                            ("nonunit_scale", [[1., 1.], [1.1, 1.1], [1.1, 1.1]])]:
        queue = sr.FlowstarSymbolicRemainderQueue.empty_accepted_boundary_sr(2, 100)
        outputs = []
        for k, matrix in enumerate(witness["matrices"]):
            point, enclosed, image, stats = sr.accepted_boundary_sr_queue_propagate(
                queue, tuple(map(tuple, matrix)), expected_boundary_index=k, reference=Interval.zero())
            outputs.append(list(map(bounds_of, image)))
            if k < len(witness["historical_j_columns"]):
                queue, _ = sr.accepted_boundary_sr_queue_commit(
                    queue, point, enclosed, tuple(Interval.point(v) for v in witness["historical_j_columns"][k]),
                    scales=scales[k+1], accepted_boundary_index=k+1, reference=Interval.zero())
        row = dict(base, witness="history", variant=variant, source_file="src/torch_tm_flowpipe/symbolic_remainder.py",
                   function="accepted_boundary_sr_queue_propagate",
                   call_chain=["prepare_accepted_boundary_sr", "accepted_boundary_sr_queue_propagate",
                               "_tensorized_interval_matrix_update_and_image"],
                   adapter="Identity-owned queue convention instead of dead first Phi; same exact J/Phi map.")
        rows.append(finish_history(row, witness["matrices"], witness["historical_j_columns"], scales, outputs))
    domain = [Interval(-1., 1.), Interval(-1., 1.)]
    models = TMVector([TaylorModel(Polynomial({(1, 0) if i == 0 else (0, 1): 1.1}, 2),
                                  Interval(-2.0**-52, 2.0**-52), domain, order=4) for i in range(2)])
    scales = [float(_interval_magnitude(v)) for v in _tmvector_range_box_normal(models, None)]
    normalized, owners = ab._scale_and_cutoff_right_map(models, scales, 1e-10)
    cs = [[float(m.polynomial.terms.get(e, 0.)) for e in [(0, 0), (1, 0), (0, 1)]] for m in normalized]
    row = dict(base, witness="normalization", variant="centered_affine",
               source_file="src/torch_tm_flowpipe/accepted_boundary_sr.py", function="_scale_and_cutoff_right_map",
               call_chain=["_flowstar_normalized_insertion_transition", "commit_accepted_boundary_sr", "_scale_and_cutoff_right_map"],
               adapter="Two CPU Taylor models; same normal-evaluation scale selection and production normalization function.")
    rows.append(finish_normalization(row, cs, list(map(lambda m: bounds_of(m.remainder), normalized)), scales,
                                     current_owner_additions=list(map(bounds_of, owners))))
    return rows, {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--implementation", choices=SHAS, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--witness", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
    args = p.parse_args(); args.source = args.source.resolve()
    git = lambda *a: subprocess.check_output(["git", "-C", str(args.source), *a], text=True).strip()
    assert git("rev-parse", "HEAD") == SHAS[args.implementation]
    assert not git("status", "--porcelain"), "Source must be clean"
    sys.path.insert(0, str(args.source / "src"))
    import torch
    import importlib
    package = importlib.import_module("torch_tm_flowpipe" if args.implementation == "C" else "flowstar_gpu")
    assert Path(package.__file__).resolve().is_relative_to(args.source / "src")
    torch.set_num_threads(1)
    witness = json.loads(args.witness.read_text())
    assert str(exact_history(witness["matrices"], witness["historical_j_columns"], [[1., 1.]]*3)[-1][1]) == witness["exact_result_fraction"][1]
    rows, metadata = our_rows(args, witness) if args.implementation == "C" else candidate_rows(args, witness)
    result = dict(schema="backend_reevaluation_crosscheck/1", source_sha=git("rev-parse", "HEAD"),
                  source_clean=not git("status", "--porcelain"), source_path=str(args.source),
                  driver_sha=subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True).strip(),
                  imported_package=package.__file__, python=sys.executable, torch_version=torch.__version__,
                  frozen_witness_sha256=hashlib.sha256(args.witness.read_bytes()).hexdigest(),
                  rows=rows, **metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"rows": len(rows), "failures": sum(not r["contains"] for r in rows)}))


if __name__ == "__main__":
    main()
