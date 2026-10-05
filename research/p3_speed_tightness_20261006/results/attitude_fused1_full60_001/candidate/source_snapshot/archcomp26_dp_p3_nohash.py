#!/usr/bin/env python3
"""Isolated DP less P3 diagnostic using existing CUDA libraries, with no digests or builds.

This ports the 2026-09-30 QUAD numerical ingredients to the 7-variable DP
contract. It does not inherit the QUAD-only 1024-lane qualification receipts.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
ENGINE = N / "engine_quad_normalization_center"
DRIVER = Path("/srv/local/shengenli/xiangru_adoption_20260907T032448Z/xiangru_upstream/src/flowstar_gpu/integrations/crown_reach.py")
CONFIG = N / "runs/archcomp_review_20260923/contracts/double_pendulum_less_robust.yaml"
CACHE = N / "cache_private_entry/py311_torch251_cu126_gcc13"
RECIP_CACHE = N / "cache_reciprocal_geometric_20260928"
RECIP_SOURCE = N / "runs/quad_control_transfer_20260927/reciprocal_five_cuda_v2/sources/tape_candidate.py"
RECIP_CORE = N / "runs/reciprocal_geometric_20260928/core.py"
ENDPOINT = N / "runs/quad_targeted_recovery_20260927/strict_endpoint.py"
TRIG = N / "runs/quad_fullbatch_p3_20260928/trig_direct_reuse.py"
MODEL = Path("/srv/local/shengenli/CROWN-Reach/ARCH-COMP2024/benchmarks/Double_Pendulum/controller_double_pendulum_less_robust.onnx")
HORNER = N / "cache_horner_entry/horner_edge_1312fa8b2aed/horner_edge_1312fa8b2aed.so"

EXTENSIONS = {
    "flowstar_seg_kernels": CACHE / "flowstar_seg_kernels/flowstar_seg_kernels.so",
    "flowstar_sr_interval_matmul": CACHE / "flowstar_sr_interval_matmul/flowstar_sr_interval_matmul.so",
    "flowstar_sr_history_sum": CACHE / "flowstar_sr_history_sum/flowstar_sr_history_sum.so",
    "flowstar_injective_index_v2": CACHE / "flowstar_injective_index_v2/flowstar_injective_index_v2.so",
    "flowstar_seg_private_output_v1": CACHE / "flowstar_seg_private_output_v1/flowstar_seg_private_output_v1.so",
    "horner_edge_1312fa8b2aed": HORNER,
    "flowstar_recip_geom_replay_1f9efda6325c": RECIP_CACHE / "flowstar_recip_geom_replay_1f9efda6325c/flowstar_recip_geom_replay_1f9efda6325c.so",
    "flowstar_recip_geom_valid_1f9efda6325c": RECIP_CACHE / "flowstar_recip_geom_valid_1f9efda6325c/flowstar_recip_geom_valid_1f9efda6325c.so",
}

ENV = {
    "FLOWSTAR_WEIGHTED_VALIDATION": "0",
    "FLOWSTAR_RECENTER_VALIDATION": "1",
    "FLOWSTAR_SELF_MAP_RETRIES": "8",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "FLOWSTAR_GLUE": "graph",
    "FLOWSTAR_INJECTIVE_MAPS": "1",
    "FLOWSTAR_CENTER_NORMALIZATION": "1",
    "FLOWSTAR_VALIDATION_POLICY": "solution_plus_one",
    "FLOWSTAR_COMPOSITION": "horner",
    "FLOWSTAR_INJECTIVE_GLUE": "1",
    "FLOWSTAR_SUPPORT_POLICY": "structural",
    "FLOWSTAR_EARLY_WEIGHTED": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TORCH_EXTENSIONS_DIR": str(CACHE),
    "FLOWSTAR_HORNER_EDGE_CACHE": str(N / "cache_horner_entry"),
}


def prohibit(*_args, **_kwargs):
    raise RuntimeError("content digests and CUDA extension builds are disabled for this run")


def apply_guards():
    # Fail before any accidental content digest or extension build is attempted.
    hashlib.sha256 = prohibit
    original_new = hashlib.new

    def guarded_new(name, *args, **kwargs):
        if name.lower().replace("-", "") == "sha256":
            return prohibit()
        return original_new(name, *args, **kwargs)

    hashlib.new = guarded_new
    import torch.utils.cpp_extension as extension
    extension.load_inline = prohibit


def load_python(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_binary(name, path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load prebuilt CUDA extension: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def strict_injection(st, T, L, U, u_ids, nn_in, eng):
    """Dimension-generic form of the frozen strict QUAD injection arithmetic."""
    import torch
    from flowstar_gpu import interval as iv, support as sp

    n, m, B = eng.tables.n, T.shape[1], st.pre.shape[0]
    if ((n, nn_in, m, tuple(u_ids)) not in ((7, 4, 2, (5, 6)), (10, 6, 3, (7, 8, 9)),
                                          (16, 12, 3, (13, 14, 15)))
            or st.pre.ndim != 3 or st.pre.shape[1] != n
            or st.pre.shape[2] != st.pre_sup.size or st.pre_sup.n != n
            or st.pre_sup.k != eng.tables.k or st.pre_sup.spatial
            or not st.pre_sup.ids or st.pre_sup.ids[0] != 0
            or max(st.pre_sup.degs) > eng.tables.k):
        raise ValueError("strict injection requires an approved time-free FULL layout")
    device = st.pre.device
    if (T.shape != (B, m, nn_in) or L.shape != (B, m) or U.shape != (B, m)
            or st.pre_rem.shape != (B, n, 2) or st.status.shape != (B,)
            or st.status.dtype != torch.int8 or st.status.device != device
            or device != eng._cutoff_zero.device
            or any(v.dtype != torch.float64 or v.device != device
                   for v in (st.pre, st.pre_rem, T, L, U))
            or not bool(((st.status >= 0) & (st.status <= 3)).all())):
        raise ValueError("DP injection requires matching binary64 rows and certificates")
    finite = lambda x: bool(torch.isfinite(x).all())
    interval = lambda x: finite(x) and bool((x[..., 0] <= x[..., 1]).all())
    checks = {
        "pre_finite": finite(st.pre),
        "T_finite": finite(T),
        "L_finite": finite(L),
        "U_finite": finite(U),
        "pre_rem_finite": finite(st.pre_rem),
        "pre_rem_ordered": bool((st.pre_rem[..., 0] <= st.pre_rem[..., 1]).all()),
        "L_U_ordered": bool((L <= U).all()),
    }
    if not all(checks.values()):
        checks["bad_pre_rem_rows"] = torch.nonzero(
            ~(torch.isfinite(st.pre_rem).all(-1) & (st.pre_rem[..., 0] <= st.pre_rem[..., 1]))
        ).cpu().tolist()[:12]
        checks["bad_control_rows"] = torch.nonzero(
            ~(torch.isfinite(L) & torch.isfinite(U) & (L <= U))
        ).cpu().tolist()[:12]
        checks["L"] = L.detach().cpu().tolist()
        checks["U"] = U.detach().cpu().tolist()
        checks["U_minus_L"] = (U - L).detach().cpu().tolist()
        checks["T"] = T.detach().cpu().tolist()
        raise ValueError("DP injection invalid input: " + json.dumps(checks))
    exps = eng.exponents(st.pre_sup) if hasattr(eng, "exponents") else sp._exps_for(st.pre_sup)
    if exps.shape != (st.pre_sup.size, n + 1) or bool((exps[:, 0] != 0).any()):
        raise ValueError("DP controller sampling requires a time-free FULL support")
    owners = (st.pre.untyped_storage().data_ptr(), st.pre_rem.untyped_storage().data_ptr())
    if owners[0] == owners[1] or any(
        x.untyped_storage().data_ptr() in owners for x in (st.tmv, st.tmv_rem, T, L, U)
    ):
        raise ValueError("DP control rows must own storage separate from plant and certificate")

    # Preserve the original RN polynomial and charge every interval error.
    center = (U + L) * 0.5
    point = torch.einsum("bmi,bit->bmt", T, st.pre[:, :nn_in])
    point[..., 0] += center
    if not finite(center) or not finite(point):
        raise FloatingPointError("control polynomial overflow")
    input_rem = iv.dot_point_iv(T, st.pre_rem[:, :nn_in].unsqueeze(1), dim=-1)
    coeff = iv.dot_point_iv(
        T.unsqueeze(2), iv.from_point(st.pre[:, :nn_in].transpose(-1, -2)).unsqueeze(1), dim=-1
    )
    coeff[..., 0, :] = iv.add(coeff[..., 0, :], iv.from_point(center))
    coefficient_error = iv.sub(coeff, iv.from_point(point))
    error_range = sp.range_normal_iv_s(coefficient_error, eng, st.pre_sup)
    bias_residual = iv.sub(torch.stack((L, U), -1), iv.from_point(center))
    remainder = iv.add(iv.add(input_rem, bias_residual), error_range)
    if not all(interval(x) for x in (input_rem, coeff, coefficient_error, error_range, bias_residual, remainder)):
        raise FloatingPointError("strict DP injection produced invalid bounds")
    st.pre[:, u_ids] = point.clone()
    st.pre_rem[:, u_ids] = remainder.clone()


def configure_environment():
    for key, value in ENV.items():
        os.environ[key] = value
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("this diagnostic is reserved for physical GPU 3")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def prepare_runtime(controller_residual=True, envelope="interval"):
    configure_environment()
    sys.path.insert(0, str(ENGINE / "src"))
    apply_guards()
    import torch
    if torch.__version__ != "2.5.1+cu121" or not torch.cuda.is_available():
        raise RuntimeError("the saved PyTorch/CUDA environment is unavailable")
    import flowstar_gpu
    from flowstar_gpu import cuda_kernels as ck, elementary as elem
    ck.load_cuda_extension = prohibit
    ck._ext = load_binary("flowstar_seg_kernels", EXTENSIONS["flowstar_seg_kernels"])
    ck._tried = True

    # The candidate changes Python valid/replay and both independent CUDA tapes.
    core = load_python("dp_reciprocal_core", RECIP_CORE)
    for name in ("rec_series_valid_g", "rec_series_valid", "rec_series_replay"):
        setattr(elem, name, getattr(core, name))
    tape = load_python("flowstar_gpu.tape_kernels", RECIP_SOURCE)
    tape._ext = load_binary("flowstar_recip_geom_replay_1f9efda6325c", EXTENSIONS["flowstar_recip_geom_replay_1f9efda6325c"])
    tape._tried = True
    tape._vext = load_binary("flowstar_recip_geom_valid_1f9efda6325c", EXTENSIONS["flowstar_recip_geom_valid_1f9efda6325c"])
    tape._vtried = True
    flowstar_gpu.tape_kernels = tape

    for module_name, extension_name in (
        ("sr_kernels", "flowstar_sr_interval_matmul"),
        ("sr_sum_kernels", "flowstar_sr_history_sum"),
        ("injective_index", "flowstar_injective_index_v2"),
        ("private_output_kernels", "flowstar_seg_private_output_v1"),
        ("horner_edge_kernels", "horner_edge_1312fa8b2aed"),
    ):
        module = __import__("flowstar_gpu." + module_name, fromlist=[module_name])
        module._ext = load_binary(extension_name, EXTENSIONS[extension_name])
        if hasattr(module, "_tried"):
            module._tried = True

    driver = load_python("archcomp26_dp_p3_driver", DRIVER)
    from archcomp26_dp_interval_controller import load_controller, residual_interval
    from archcomp26_dp_affine_controller import affine_residual_interval
    if envelope not in ("interval", "affine"):
        raise ValueError("unknown DP controller envelope")
    layers = load_controller(MODEL, torch, "cuda:0")
    endpoint = load_python("archcomp26_dp_strict_endpoint", ENDPOINT)
    trig = load_python("archcomp26_dp_trig_reuse", TRIG)
    from flowstar_gpu import sparse_exec as se
    trig_handle = trig._bind(se)  # install() performs a prohibited digest check
    if not hasattr(se, "COMPOSE_PARENT_ASSEMBLY"):
        # The shared driver reads this field only while writing metrics.  This
        # older P3 core does not expose it; null records that fact faithfully.
        se.COMPOSE_PARENT_ASSEMBLY = None
    driver.end_of_time_s = endpoint.end_of_time_s
    owner = {}
    controller_audit = []
    base_engine = driver.SparseEngine

    def capture_engine(*args, **kwargs):
        if owner:
            raise RuntimeError("one run must create exactly one sparse engine")
        eng = base_engine(*args, **kwargs)
        owner["engine"] = eng
        return eng

    original_crown_bounds = driver.crown_bounds

    def capture_crown_bounds(model, config, lower, upper, input_layout="native"):
        T, L, U = original_crown_bounds(model, config, lower, upper, input_layout=input_layout)
        hull = torch.stack((lower, upper), dim=-1)
        if envelope == "affine":
            network, residual, layer_diagnostics = affine_residual_interval(hull, T, layers)
            linear = None
        else:
            network, linear, residual = residual_interval(hull, T, layers)
            layer_diagnostics = None
        bad = ~(torch.isfinite(L) & torch.isfinite(U) & (L <= U))
        gap = U - L
        width = residual[..., 1] - residual[..., 0]
        controller_audit.append({
            "period": len(controller_audit),
            "envelope": envelope,
            "crown_bias_inversion_count": int(bad.sum().item()),
            "crown_bias_min_gap": float(gap.min().item()),
            "directed_residual_width_max": float(width.max().item()),
            "directed_residual_width_median": float(width.median().item()),
            "directed_network_first_lane": network[0].detach().cpu().tolist(),
            "directed_linear_first_lane": linear[0].detach().cpu().tolist() if linear is not None else None,
            "directed_residual_first_lane": residual[0].detach().cpu().tolist(),
            "affine_layer_diagnostics": layer_diagnostics,
        })
        return T, residual[..., 0], residual[..., 1]

    def inject(st, T, L, U, u_ids, nn_in):
        with torch.no_grad():
            return strict_injection(st, T, L, U, u_ids, nn_in, owner["engine"])

    driver.SparseEngine = capture_engine
    if controller_residual:
        driver.crown_bounds = capture_crown_bounds
    driver.inject_controls_s = inject
    if not all((ck.available(), tape.available(), tape.valid_available())):
        raise RuntimeError("preloaded CUDA libraries did not become available")
    return torch, driver, trig_handle, controller_audit


def contract_config(path, mode):
    import yaml
    config = yaml.safe_load(CONFIG.read_text())
    assert config["num_vars"] == 7 and config["num_nn_input"] == 4 and config["num_nn_output"] == 2
    assert config["model_dir"] == str(MODEL)
    assert config["steps"] == 20 and config["step_size"] == 0.05 and config["ode_step_size"] == 0.01
    assert [e["splits"] for e in config["initial_set"][:4]] == [5, 5, 3, 3]
    assert config["constraints_safe"] == ["-th1 - 1.7", "th1 - 2", "-th2 - 1.7", "th2 - 2", "-u1 - 1.7", "u1 - 2", "-u2 - 1.7", "u2 - 2"]
    if mode == "smoke":
        for entry in config["initial_set"][:4]:
            lo, hi = entry["interval"]
            entry["interval"] = [lo, lo + (hi - lo) / entry["splits"]]
            entry["splits"] = 0
        config["split_vars"] = []
    if mode in ("smoke", "batch-smoke"):
        config["steps"] = 1
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config


def self_check():
    """Regress the generalized injection against the frozen QUAD arithmetic."""
    sys.path.insert(0, str(ENGINE / "src"))
    apply_guards()
    import torch
    from types import SimpleNamespace
    from flowstar_gpu import monomials, nn_coupling, polynomial, support

    torch.set_default_dtype(torch.float64)
    original = load_python(
        "archcomp26_original_strict_injection",
        N / "runs/quad_control_transfer_20260927/strict_injection.py",
    )

    def state(n, order):
        tables = monomials.build_tables(n, order)
        step = polynomial.build_step_tables(tables, 0.01)
        engine = support.SparseEngine(tables, step, "cpu")
        sup = support.make_support(n, order, False, tuple(nn_coupling.onehot_ids(tables)))
        pre = torch.zeros((1, n, sup.size), dtype=torch.float64)
        for i in range(n):
            pre[0, i, 0] = 0.2 + i / 40
            pre[0, i, i + 1] = 0.01
        rem = torch.tensor([[[-0.0001, 0.0002]]] * n, dtype=torch.float64).transpose(0, 1)
        s = SimpleNamespace(pre=pre, pre_rem=rem, pre_sup=sup,
                            tmv=pre.clone(), tmv_rem=rem.clone(),
                            status=torch.zeros(1, dtype=torch.int8))
        return engine, s

    engine, a = state(16, 2)
    b = copy.copy(a)
    b.pre, b.pre_rem, b.tmv, b.tmv_rem = (
        a.pre.clone(), a.pre_rem.clone(), a.tmv.clone(), a.tmv_rem.clone()
    )
    T = torch.arange(36, dtype=torch.float64).reshape(1, 3, 12) / 100
    L = torch.tensor([[-0.3, -0.2, -0.1]], dtype=torch.float64)
    U = L + 0.03
    original.inject_controls_s(a, T, L, U, [13, 14, 15], 12, eng=engine)
    strict_injection(b, T, L, U, [13, 14, 15], 12, engine)
    assert torch.equal(a.pre, b.pre) and torch.equal(a.pre_rem, b.pre_rem)

    dp_engine, dp = state(7, 3)
    t = T[:, :2, :4].contiguous()
    strict_injection(dp, t, L[:, :2], U[:, :2], [5, 6], 4, dp_engine)
    assert bool(torch.isfinite(dp.pre).all())
    assert bool(torch.isfinite(dp.pre_rem).all())
    assert bool((dp.pre_rem[..., 0] <= dp.pre_rem[..., 1]).all())
    attitude_engine, attitude = state(10, 3)
    strict_injection(attitude, T[:, :, :6].contiguous(), L, U, [7, 8, 9], 6,
                     attitude_engine)
    assert bool(torch.isfinite(attitude.pre).all())
    assert bool(torch.isfinite(attitude.pre_rem).all())
    assert bool((attitude.pre_rem[..., 0] <= attitude.pre_rem[..., 1]).all())
    driver = load_python("archcomp26_transport_reference", DRIVER)
    native = driver.apply_crown_transport(t, L[:, :2], U[:, :2], "native-f64")
    rpc = driver.apply_crown_transport(t, L[:, :2], U[:, :2], "rpc-float32")
    assert all(torch.equal(x, y) for x, y in zip(native, (t, L[:, :2], U[:, :2])))
    assert all(torch.equal(x, y.to(torch.float32).to(torch.float64)) for x, y in zip(rpc, native))
    print(json.dumps({"quad_injection_bitwise_equal": True, "dp_rows": [5, 6],
                      "dp_finite_ordered": True, "attitude_rows": [7, 8, 9],
                      "attitude_finite_ordered": True, "native_f64_preserved": True,
                      "rpc_float32_projection_checked": True}))


def execute(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "config.yaml"
    config = contract_config(config_path, args.mode)
    record = {
        "schema": "archcomp26-dp-less-p3-diagnostic-v1",
        "mode": args.mode,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_engine": str(ENGINE),
        "shared_driver": str(DRIVER),
        "reciprocal_python": [str(RECIP_CORE), str(RECIP_SOURCE)],
        "prebuilt_libraries": {k: str(v) for k, v in EXTENSIONS.items()},
        "strict_endpoint": str(ENDPOINT),
        "trig_reuse": str(TRIG),
        "config": str(config_path),
        "controller": config["model_dir"],
        "physical_gpu": 3,
        "method": "work-P3/point-P2/validation-P4, geometric reciprocal, strict endpoint/injection, full SR, trig reuse, independent directed controller residual",
        "controller_transport": "native-f64",
        "controller_envelope": args.controller_envelope,
        "qualification": "new DP port diagnostic; independent local controller envelope, complete end-to-end floating-point proof pending",
        "controller_model_evidence": "server 2024 model directly byte-compared equal to fixed official 2026 less ONNX by contract audit; no content digest",
        "driver_core_metrics_compatibility": "missing COMPOSE_PARENT_ASSEMBLY reporting field is recorded as null; no numerical dispatch changed",
        "environment": {**ENV, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")},
    }
    (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
    started = time.perf_counter()
    result = {"status": "exception", "completed_substeps": 0}
    try:
        torch, driver, trig, controller_audit = prepare_runtime(envelope=args.controller_envelope)
        steps = []
        original_advance = driver.advance_sparse
        metrics = output / "metrics.json"
        observations = output / "observations.jsonl"
        with observations.open("x") as log:
            def observed_advance(st, code, eng, sched, settings, cap, sr):
                state, accepted = original_advance(st, code, eng, sched, settings, cap, sr)
                tube = driver.hull_ranges_s(state, eng, 4)
                t_end = torch.full((state.pre.shape[0], 2), settings.step, dtype=torch.float64, device=state.pre.device)
                endpoint = driver.rows_range_over_time_sparse(state, eng, t_end, 4)
                # Failed lanes retain the last accepted state.  Their frozen
                # tube cannot establish safety for the rejected substep.
                inside = accepted & ((tube[..., 0] >= -1.7) & (tube[..., 1] <= 2.0)).all(dim=1)
                row = {
                    "substep": len(steps) + 1,
                    "accepted": accepted.detach().cpu().tolist(),
                    "tube_endpoint_valid": accepted.detach().cpu().tolist(),
                    "status": state.status.detach().cpu().tolist(),
                    "whole_tube_inside_safe_box": inside.detach().cpu().tolist(),
                    "tube": tube.detach().cpu().tolist(),
                    "endpoint": endpoint.detach().cpu().tolist(),
                }
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                steps.append((sum(row["accepted"]), len(row["accepted"]), sum(row["whole_tube_inside_safe_box"])))
                return state, accepted

            driver.advance_sparse = observed_advance
            argv = [str(DRIVER), str(config_path), "--device", "cuda:0", "--engine", "sparse",
                    "--strict", "--order", "3", "--crown-domain", "box",
                    "--crown-relax", "same-slope", "--crown-transport", "native-f64",
                    "--crown-input-layout", "native", "--nn-mode", "crown",
                    "--print-final-hull", "--metrics-json", str(metrics)]
            record["driver_argv"] = argv
            (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
            previous = sys.argv
            sys.argv = argv
            try:
                code = driver.main()
            finally:
                sys.argv = previous
        expected = config["steps"] * 5
        all_accepted = len(steps) == expected and all(a == b for a, b, _ in steps)
        all_inside = len(steps) == expected and all(s == b for _, b, s in steps)
        result = {
            "status": "completed" if code == 0 and all_accepted else "incomplete",
            "driver_return": code,
            "expected_substeps": expected,
            "completed_substeps": len(steps),
            "accepted_lane_substeps": sum(a for a, _, _ in steps),
            "all_lanes_accepted": all_accepted,
            "whole_tube_inside_safe_box": all_inside,
            "per_substep_accepted": [a for a, _, _ in steps],
            "per_substep_inside_safe_box": [s for _, _, s in steps],
            "trig_counters": dict(trig.counters),
            "trig_plan": dict(trig.last_plan),
            "controller_audit": controller_audit,
            "wall_s": time.perf_counter() - started,
            "timing_scope": "driver plus full per-lane tube/endpoint diagnostic observation and JSONL writes",
            "end_to_end_strict_certificate": False,
        }
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                      completed_substeps=len(steps) if "steps" in locals() else 0,
                      accepted_lane_substeps=sum(a for a, _, _ in steps) if "steps" in locals() else 0,
                      controller_audit=controller_audit if "controller_audit" in locals() else [],
                      wall_s=time.perf_counter() - started)
        raise
    finally:
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "batch-smoke", "full"), required=True)
    parser.add_argument("--controller-envelope", choices=("interval", "affine"), default="interval")
    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
