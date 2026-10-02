#!/usr/bin/env python3
"""Isolated old-author QUAD P3 weighted256 full-horizon candidate; no digests or JIT."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
ENGINE = N / "engine_quad_normalization_center"
DRIVER = Path("/srv/local/shengenli/xiangru_adoption_20260907T032448Z/xiangru_upstream/src/flowstar_gpu/integrations/crown_reach.py")
MODEL = Path("/srv/local/shengenli/CROWN-Reach-GPU/ARCH-COMP2024/benchmarks/QUAD/quad_controller_3_64_torch.onnx")
CACHE = N / "cache_private_entry/py311_torch251_cu126_gcc13"
RECIP_CACHE = N / "cache_reciprocal_geometric_20260928"
RECIP_SOURCE = N / "runs/quad_control_transfer_20260927/reciprocal_five_cuda_v2/sources/tape_candidate.py"
RECIP_CORE = N / "runs/reciprocal_geometric_20260928/core.py"
ENDPOINT = N / "runs/quad_targeted_recovery_20260927/strict_endpoint.py"
INJECTION = N / "runs/quad_control_transfer_20260927/strict_injection.py"
HOST = N / "runs/quad_sr_reassociate_20260928/host_ledger.py"
TRIG = N / "runs/quad_fullbatch_p3_20260928/trig_direct_reuse.py"
NOHASH = N / "runs/archcomp26_20261001/p3_quad_old_stage_a_40_nohash_observer_v1/archcomp26_quad_p3_nohash"

EXTENSIONS = {
    "flowstar_seg_kernels": CACHE / "flowstar_seg_kernels/flowstar_seg_kernels.so",
    "flowstar_sr_interval_matmul": CACHE / "flowstar_sr_interval_matmul/flowstar_sr_interval_matmul.so",
    "flowstar_sr_history_sum": CACHE / "flowstar_sr_history_sum/flowstar_sr_history_sum.so",
    "flowstar_injective_index_v2": CACHE / "flowstar_injective_index_v2/flowstar_injective_index_v2.so",
    "flowstar_seg_private_output_v1": CACHE / "flowstar_seg_private_output_v1/flowstar_seg_private_output_v1.so",
    "horner_edge_1312fa8b2aed": N / "cache_horner_entry/horner_edge_1312fa8b2aed/horner_edge_1312fa8b2aed.so",
    "flowstar_recip_geom_replay_1f9efda6325c": RECIP_CACHE / "flowstar_recip_geom_replay_1f9efda6325c/flowstar_recip_geom_replay_1f9efda6325c.so",
    "flowstar_recip_geom_valid_1f9efda6325c": RECIP_CACHE / "flowstar_recip_geom_valid_1f9efda6325c/flowstar_recip_geom_valid_1f9efda6325c.so",
}
ENV = {
    "FLOWSTAR_WEIGHTED_VALIDATION": "0", "FLOWSTAR_RECENTER_VALIDATION": "1",
    "FLOWSTAR_SELF_MAP_RETRIES": "8", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "FLOWSTAR_GLUE": "graph", "FLOWSTAR_INJECTIVE_MAPS": "1",
    "FLOWSTAR_HORNER_EDGE_CACHE": str(N / "cache_horner_entry"),
    "FLOWSTAR_CENTER_NORMALIZATION": "1", "FLOWSTAR_VALIDATION_POLICY": "solution_plus_one",
    "FLOWSTAR_COMPOSITION": "horner", "FLOWSTAR_EARLY_WEIGHTED": "1",
    "FLOWSTAR_INJECTIVE_GLUE": "1", "FLOWSTAR_SUPPORT_POLICY": "structural",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "TORCH_EXTENSIONS_DIR": str(CACHE), "PYTHONDONTWRITEBYTECODE": "1",
}


def prohibited(*_args, **_kwargs):
    raise RuntimeError("content digest or JIT/build disabled in this run")


def python_script(function=None, **kwargs):
    """Keep auto_LiRPA decorated functions in Python without invoking TorchScript."""
    if function is None and not kwargs:
        return lambda value: value
    if callable(function) and not kwargs:
        return function
    raise RuntimeError("unsupported TorchScript decorator form in no-JIT arm")


def guard():
    for name in hashlib.algorithms_guaranteed:
        if hasattr(hashlib, name):
            setattr(hashlib, name, prohibited)
    hashlib.new = prohibited
    if hasattr(hashlib, "file_digest"):
        hashlib.file_digest = prohibited
    import torch.utils.cpp_extension as extension
    extension.load = extension.load_inline = prohibited


def load(name, path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_config(path, output, mode):
    import yaml

    # Use the archived old-author full contract; compare bytes directly.
    if path.read_bytes() != (Path(__file__).parent / "quad_author_resolved.yaml").read_bytes():
        raise ValueError("source must equal the saved old-author full YAML")
    cfg = yaml.safe_load(path.read_text())
    if not (cfg["num_vars"] == 16 and cfg["num_nn_input"] == 12
            and cfg["num_nn_output"] == 3 and cfg["ode_order"] == 2
            and cfg["steps"] == 50 and cfg["step_size"] == 0.1
            and cfg["ode_step_size"] == 0.005
            and [e["splits"] for e in cfg["initial_set"][:6]] == [8, 8, 8, 2, 1, 1]
            and cfg["model_dir"] == str(MODEL) and MODEL.is_file()
            and cfg["constraints_target"] == ["-x3 + 0.94", "x3 - 1.06"]):
        raise ValueError("not the saved old-author QUAD full contract")
    expr = ["".join(value.split()) for value in cfg["dynamics_expressions"]]
    if (expr[1] != "cos(x8)*sin(x9)*x4+(sin(x7)*sin(x8)*sin(x9)-cos(x7)*cos(x9))*x5+(cos(x7)*sin(x8)*sin(x9)+sin(x7)*cos(x9))*x6"
            or expr[3] != "x12*x5*x11*x6-9.81*sin(x8)"
            or expr[4] != "x10*x6-x11*x6-9.81*sin(x8)"):
        raise ValueError("old-author x2/x4/x5 equations differ")
    cfg = copy.deepcopy(cfg)
    cfg["ode_order"] = 3  # P3 working engine; source old-author ODE point order stays P2.
    cfg["sr_queue"] = 1000
    generated = output / "config.yaml"
    generated.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg, generated


def prepare(batch):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("P3 candidate requires reserved physical GPU3")
    for key, value in ENV.items():
        os.environ[key] = value
    sys.path.insert(0, str(ENGINE / "src"))
    guard()
    import torch
    if torch.__version__ != "2.5.1+cu121" or not torch.cuda.is_available():
        raise RuntimeError("saved PyTorch/CUDA runtime unavailable")
    torch.compile = prohibited
    torch.jit.script = python_script
    torch.jit.trace = prohibited
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cap = min(27 * 2**29, int(torch.cuda.get_device_properties(0).total_memory * 0.9))
    torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(0).total_memory)

    import flowstar_gpu
    from flowstar_gpu import cuda_kernels as ck, elementary as elem
    ck.load_cuda_extension = prohibited
    ck._ext = load("flowstar_seg_kernels", EXTENSIONS["flowstar_seg_kernels"])
    ck._tried = True
    core = load("quad_paper_p3_reciprocal_core", RECIP_CORE)
    for name in ("rec_series_valid_g", "rec_series_valid", "rec_series_replay"):
        setattr(elem, name, getattr(core, name))
    tape = load("flowstar_gpu.tape_kernels", RECIP_SOURCE)
    tape.load_cuda_extension = prohibited
    tape._ext = load("flowstar_recip_geom_replay_1f9efda6325c", EXTENSIONS["flowstar_recip_geom_replay_1f9efda6325c"])
    tape._vext = load("flowstar_recip_geom_valid_1f9efda6325c", EXTENSIONS["flowstar_recip_geom_valid_1f9efda6325c"])
    tape._tried = tape._vtried = True
    flowstar_gpu.tape_kernels = tape
    for module_name, binary_name in (
        ("sr_kernels", "flowstar_sr_interval_matmul"),
        ("sr_sum_kernels", "flowstar_sr_history_sum"),
        ("injective_index", "flowstar_injective_index_v2"),
        ("private_output_kernels", "flowstar_seg_private_output_v1"),
        ("horner_edge_kernels", "horner_edge_1312fa8b2aed"),
    ):
        module = __import__("flowstar_gpu." + module_name, fromlist=[module_name])
        module._ext = load(binary_name, EXTENSIONS[binary_name])
        if hasattr(module, "_tried"):
            module._tried = True
    if not (ck.available() and tape.available() and tape.valid_available()):
        raise RuntimeError("saved numerical CUDA extensions unavailable")

    driver = load("archcomp26_quad_paper_p3_driver", DRIVER)
    from flowstar_gpu import sparse_exec as se, weighted_validation as wv, graphing
    hybrid = load("archcomp26_quad_p3_hybrid_nohash", NOHASH / "hybrid_metadata.py")
    hybrid_receipt = hybrid.install(None, NOHASH / "metadata_backend.py", extra_modules=(driver,))
    weighted = load("archcomp26_quad_p3_weighted256_nohash", Path(__file__).parent / "weighted_chunk256.py")
    weighted_receipt = weighted.install(torch, wv, se) if batch == 1024 else None
    valid = load("archcomp26_quad_p3_valid_nohash", NOHASH / "valid_chunk128.py")
    valid_receipt = valid.install(torch, se) if batch == 1024 else None
    postwarm_receipt = prune_receipt = horner_receipt = eager_receipt = None
    if batch == 1024:
        postwarm = load("archcomp26_quad_p3_postwarm_nohash", NOHASH / "capture_workspace_release.py")
        postwarm_receipt = postwarm.install(torch, graphing)
        prune = load("archcomp26_quad_p3_prune_nohash", NOHASH / "working_graph_prune.py")
        prune_receipt = prune.install(torch, graphing)
        from flowstar_gpu import sparse_horner
        horner = load("archcomp26_quad_p3_horner_nohash", NOHASH / "horner_closure_cleanup.py")
        horner_receipt = horner.install(sparse_horner)
        eager = load("archcomp26_quad_p3_eager_nohash", NOHASH / "working_eager_segments.py")
        eager_receipt = eager.install(torch, graphing)
    endpoint = load("archcomp26_quad_p3_endpoint", ENDPOINT)
    injection = load("archcomp26_quad_p3_injection", INJECTION)
    host = load("archcomp26_quad_p3_host", HOST)
    trig = load("archcomp26_quad_p3_trig", TRIG)
    trig_receipt = trig._bind(se)  # install() computes a prohibited digest.
    if not hasattr(se, "COMPOSE_PARENT_ASSEMBLY"):
        se.COMPOSE_PARENT_ASSEMBLY = None  # Shared driver metrics only.
    return (torch, driver, se, host, endpoint, injection, trig_receipt, hybrid_receipt,
            weighted_receipt, valid_receipt, postwarm_receipt, prune_receipt,
            horner_receipt, eager_receipt, cap)


def execute(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {"status": "exception", "mode": args.mode, "completed_substeps": 0}
    start_record = {
        "schema": "quad-old-stage-a-p3-weighted256-full1000-nohash-v1", "started_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode, "source_config": str(args.source_config),
        "engine": str(ENGINE), "driver": str(DRIVER), "model": str(MODEL),
        "method": "working-P3/point-P2/validation-P4 strict, geometric reciprocal, strict endpoint/injection, host-K20 full SR, trig reuse",
        "qualification": "old-author QUAD equations, new 1000-step P3 weighted256 candidate; no end-to-end proof",
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "source_modules": [str(RECIP_CORE), str(RECIP_SOURCE), str(ENDPOINT), str(INJECTION), str(HOST), str(TRIG)],
        "nohash_adapters": [str(p) for p in sorted(NOHASH.glob("*.py"))] +
                           [str(Path(__file__).parent / "weighted_chunk256.py")],
        "preloaded_binaries": {name: {"path": str(path), "bytes": path.stat().st_size,
                                      "mtime_ns": path.stat().st_mtime_ns}
                               for name, path in EXTENSIONS.items()},
        "controller_note": "old Stage A source YAML model path; controller identity not reverified in this ablation",
        "observer_arm": args.mode,
        "observer_artifacts": "one per-step NPY each for 1024x12x4 physical bounds, accepted mask, and status; no content digest",
        "comparison_scope": "same old-author equations and P3 engine, 1000-step candidate; archived 128-row full run is a different dated implementation",
    }
    (output / "START.json").write_text(json.dumps(start_record, indent=2) + "\n")
    observations = []
    try:
        import numpy as np

        cfg, config_path = source_config(args.source_config, output, args.mode)
        batch = 1024
        (torch, driver, se, host, endpoint, injection, trig, hybrid, weighted, valid,
         postwarm, prune, horner, eager, cap_bytes) = prepare(batch)
        cells = driver.make_cells(cfg)
        if cells.shape != (batch, 16, 2) or not bool(torch.isfinite(cells).all()):
            raise ValueError("old-author initial partition mismatch")
        start_record.update(generated_config=str(config_path), expected_boxes=batch,
                            allocator_cap_bytes=cap_bytes,
                            hybrid_metadata_policy=hybrid.policy,
                            weighted_policy=weighted.policy if weighted else None,
                            valid_policy=valid.policy if valid else None,
                            postwarm_policy=postwarm.policy if postwarm else None,
                            working_prune_policy=prune.policy if prune else None,
                            horner_cleanup_policy=horner.policy if horner else None,
                            working_eager_policy=eager.policy if eager else None,
                            trig_policy=trig.policy)
        original_engine = driver.SparseEngine
        original_initial = driver.initial_sparse_state
        original_sr = driver.make_symbolic_remainder
        original_advance = driver.advance_sparse
        original_prune = driver.prune_state
        original_crown = driver.crown_bounds
        original_endpoint = driver.end_of_time_s
        original_inject = driver.inject_controls_s
        from flowstar_gpu import symbolic_remainder as sym
        if se.propagate is not sym.propagate:
            raise RuntimeError("symbolic propagation alias changed")
        raw_propagate = sym.propagate
        holder = {"engine": None, "state": None, "sr": None, "ledger": None,
                  "pending_status": None, "propagate_calls": 0, "nn_calls": 0,
                  "control_refresh_steps": []}

        def sparse_engine(*a, **kw):
            if holder["engine"] is not None:
                raise RuntimeError("more than one working P3 engine")
            eng = original_engine(*a, **kw)
            if eng.tables.n != 16 or eng.tables.k != 3:
                raise ValueError("wrong working order")
            holder["engine"] = eng
            from flowstar_gpu import horner_edge_kernels
            if not callable(getattr(horner_edge_kernels._ext, "horner_edge", None)):
                raise RuntimeError("saved Horner edge binary lacks required export")
            eng.horner_edge_kernel = horner_edge_kernels._ext.horner_edge
            return eng

        def initial_state(*a, **kw):
            state = original_initial(*a, **kw)
            holder["state"] = state
            return state

        def symbolic_remainder(B, n, max_size, device):
            if (B, n, max_size) != (batch, 16, 1000) or holder["state"] is None:
                raise ValueError("wrong symbolic remainder layout")
            sr = original_sr(B, n, max_size, device)
            torch.cuda.empty_cache()
            sr.reserve(1000)
            ledger = host.HostFactorLedger(B, n, max_size, lane_chunk=16, status=holder["state"].status)
            if inspect.getmodule(raw_propagate) is not sym:
                raise RuntimeError("saved symbolic propagation module changed")
            ledger._raw = raw_propagate  # Bypass only the original source-file digest guard.
            holder["sr"], holder["ledger"] = sr, ledger
            raw_reset = sr.reset

            def reset():
                ledger._ready(sr)
                if ledger.length != 1000:
                    raise RuntimeError("premature SR reset")
                raw_reset()
                if sr.qlen or sr.jlen:
                    raise RuntimeError("SR reset did not clear history")
                ledger.epoch += ledger.length
                ledger.length = 0
                ledger.last_rebuilt_lane.zero_()
                ledger._ready(sr)

            sr.reset = reset
            return sr

        def propagate(sr, phi_i, strict=False, phi_i_iv=None):
            if not strict or phi_i_iv is None or sr is not holder["sr"] or holder["pending_status"] is None:
                raise ValueError("unexpected SR propagation path")
            holder["propagate_calls"] += 1
            if holder["propagate_calls"] != 1:
                raise RuntimeError("multiple SR propagations in one substep")
            return holder["ledger"].propagate(sr, phi_i, phi_i_iv,
                                               status_before=holder["pending_status"],
                                               raw_propagate=raw_propagate, sym=sym)

        def crown_bounds(model, config, lower, upper, **kw):
            if lower.shape != (batch, 12) or kw != {"input_layout": "native"}:
                raise ValueError("controller transport/layout mismatch")
            T, L, U = original_crown(model, config, lower, upper, **kw)
            if T.shape != (batch, 3, 12) or L.shape != (batch, 3) or U.shape != (batch, 3):
                raise ValueError("controller affine certificate shape mismatch")
            if not bool((torch.isfinite(T).all() and torch.isfinite(L).all()
                         and torch.isfinite(U).all() and (L <= U).all())):
                raise ValueError("CROWN controller certificate invalid")
            holder["nn_calls"] += 1
            holder["control_refresh_steps"].append(len(observations))
            return T, L, U

        def strict_inject(st, T, L, U, u_ids, nn_in):
            if st is not holder["state"]:
                raise RuntimeError("unexpected state at control injection")
            injection.inject_controls_s(st, T, L, U, u_ids, nn_in, eng=holder["engine"])

        def strict_endpoint(st, eng):
            if st is not holder["state"] or eng is not holder["engine"]:
                raise RuntimeError("unexpected endpoint state")
            return endpoint.end_of_time_s(st, eng)

        observer_region_wall_s = 0.0
        with (output / "observations.jsonl").open("x") as log:
            def advance(st, code, eng, sched, settings, cap, sr):
                nonlocal observer_region_wall_s
                if (st is not holder["state"] or eng is not holder["engine"]
                        or sr is not holder["sr"] or settings.order != 3
                        or code.order != 2 or settings.step != 0.005):
                    raise RuntimeError("unexpected P3 numerical route")
                holder["pending_status"] = st.status.clone()
                holder["propagate_calls"] = 0
                if eager:
                    eager.begin_step(eng)
                    prune.begin_step(eng)
                try:
                    state, accepted = original_advance(st, code, eng, sched, settings, cap, sr)
                    if holder["propagate_calls"] != 1:
                        raise RuntimeError("SR propagation missing")
                    holder["ledger"].finish_step(sr, state.status,
                                                 token=holder["ledger"].epoch + holder["ledger"].length)
                finally:
                    holder["pending_status"] = None
                holder["state"] = state
                good = accepted.detach().cpu().tolist()
                union = None
                if args.mode == "observer_on":
                    observer_started = time.perf_counter()
                    active = accepted.bool()
                    tube = driver.hull_ranges_s(state, eng, 12)
                    t_end = torch.full((batch, 2), settings.step, dtype=torch.float64, device=state.pre.device)
                    end = driver.rows_range_over_time_sparse(state, eng, t_end, 12)
                    step_number = len(observations) + 1
                    np.save(output / f"observer_{step_number:04d}_bounds.npy",
                            torch.cat((tube, end), dim=-1).detach().cpu().numpy(), allow_pickle=False)
                    np.save(output / f"observer_{step_number:04d}_accepted.npy",
                            accepted.detach().cpu().numpy(), allow_pickle=False)
                    np.save(output / f"observer_{step_number:04d}_status.npy",
                            state.status.detach().cpu().numpy(), allow_pickle=False)
                    if bool(active.any()):
                        selected = torch.cat((tube[active], end[active]), dim=-1)
                        if not bool(torch.isfinite(selected).all() and
                                    (selected[..., 0] <= selected[..., 1]).all() and
                                    (selected[..., 2] <= selected[..., 3]).all()):
                            raise FloatingPointError("accepted physical tube/endpoint invalid")
                        union = torch.stack((selected[..., 0].min(0).values,
                                             selected[..., 1].max(0).values,
                                             selected[..., 2].min(0).values,
                                             selected[..., 3].max(0).values), -1).cpu().tolist()
                    observer_region_wall_s += time.perf_counter() - observer_started
                row = {"substep": len(observations) + 1, "accepted_count": sum(good),
                       "status_counts": {str(v): state.status.detach().cpu().tolist().count(v)
                                         for v in sorted(set(state.status.detach().cpu().tolist()))},
                       "rejected_lanes": [i for i, value in enumerate(good) if not value],
                       "sr_length": holder["ledger"].length,
                       "sr_epoch": holder["ledger"].epoch,
                       "tube_endpoint_union_12x4": union}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                observations.append(row)
                if row["substep"] % 20 == 0 or row["rejected_lanes"]:
                    log.flush()
                    (output / "progress.json").write_text(json.dumps({k: row[k] for k in
                        ("substep", "accepted_count", "status_counts", "rejected_lanes", "sr_length", "sr_epoch")}) + "\n")
                if row["rejected_lanes"]:
                    (output / "FIRST_REJECT.json").write_text(json.dumps(row, allow_nan=False) + "\n")
                    raise RuntimeError(f"first numerical refusal at substep {row['substep']}")
                return state, accepted

            def prune_state(st, eng):
                state = original_prune(st, eng)
                if prune:
                    prune.finish_step(eng)
                    eager.finish_step(eng)
                holder["state"] = state
                return state

            driver.SparseEngine = sparse_engine
            driver.initial_sparse_state = initial_state
            driver.make_symbolic_remainder = symbolic_remainder
            driver.advance_sparse = advance
            driver.prune_state = prune_state
            driver.crown_bounds = crown_bounds
            driver.inject_controls_s = strict_inject
            driver.end_of_time_s = strict_endpoint
            se.propagate = propagate
            argv = [str(DRIVER), str(config_path), "--device", "cuda:0", "--strict",
                    "--engine", "sparse", "--crown-domain", "box", "--crown-relax", "same-slope",
                    "--crown-transport", "native-f64", "--crown-input-layout", "native",
                    "--nn-mode", "crown", "--order", "3", "--print-final-hull",
                    "--metrics-json", str(output / "metrics.json")]
            start_record["driver_argv"] = argv
            (output / "START.json").write_text(json.dumps(start_record, indent=2) + "\n")
            previous = sys.argv
            sys.argv = argv
            try:
                driver_return = driver.main()
            finally:
                sys.argv = previous

        expected_steps = cfg["steps"] * 20
        all_accepted = len(observations) == expected_steps and all(
            row["accepted_count"] == batch for row in observations)
        # Compare direct numerical arrays across fresh arms after driver.main's
        # internal solver clock has stopped. No digest or lossy summary is used.
        import numpy as np
        final_state, final_engine = holder["state"], holder["engine"]
        final_t = torch.full((batch, 2), cfg["ode_step_size"],
                             dtype=torch.float64, device=final_state.pre.device)
        final_tube = driver.hull_ranges_s(final_state, final_engine, 12)
        final_endpoint = driver.rows_range_over_time_sparse(
            final_state, final_engine, final_t, 12)
        for name, value in (("final_tube_12x2", final_tube),
                            ("final_endpoint_12x2", final_endpoint),
                            ("final_status", final_state.status)):
            np.save(output / (name + ".npy"), value.detach().cpu().numpy(), allow_pickle=False)
        metrics_path = output / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
        final_hull = metrics.get("final_hull") if metrics else None
        target = (final_hull is not None and "x3" in final_hull and
                  final_hull["x3"][0] >= 0.94 and final_hull["x3"][1] <= 1.06)
        result.update(status="completed" if driver_return == 0 and all_accepted else "incomplete",
                      driver_return=driver_return, expected_substeps=expected_steps,
                      completed_substeps=len(observations), expected_lane_substeps=batch * expected_steps,
                      accepted_lane_substeps=sum(row["accepted_count"] for row in observations),
                      all_lanes_accepted=all_accepted, nn_calls=holder["nn_calls"],
                      control_refresh_steps=holder["control_refresh_steps"],
                      sr_length=holder["ledger"].length if holder["ledger"] else None,
                      sr_epoch=holder["ledger"].epoch if holder["ledger"] else None,
                      metadata_validation_receipts=hybrid.receipts,
                      weighted_counters=weighted.counters if weighted else None,
                      valid_counters=valid.counters if valid else None,
                      postwarm_releases=len(postwarm.releases) if postwarm else None,
                      working_prune_counters=prune.counters if prune else None,
                      working_eager_counters=eager.counters if eager else None,
                      trig_counters=trig.counters, trig_plan=trig.last_plan,
                      metrics_broken=metrics.get("broken") if metrics else None,
                      observer_mode=args.mode,
                      observer_region_wall_s=observer_region_wall_s,
                      direct_comparison_arrays=["final_tube_12x2.npy", "final_endpoint_12x2.npy",
                                                "final_status.npy"],
                      final_hull=final_hull,
                      endpoint_target_inside=None,
                      driver_elapsed_s=metrics.get("elapsed_s") if metrics else None,
                      wall_s=time.perf_counter() - started,
                      end_to_end_strict_certificate=False)
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                      completed_substeps=len(observations),
                      accepted_lane_substeps=sum(row["accepted_count"] for row in observations),
                      wall_s=time.perf_counter() - started)
        raise
    finally:
        if "torch" in locals() and torch.cuda.is_initialized():
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["peak_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("observer_on", "observer_off"), required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
