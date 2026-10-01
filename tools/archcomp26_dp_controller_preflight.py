#!/usr/bin/env python3
"""Inspect first-period DP CROWN certificates only; no plant or CUDA builds."""

import argparse
import json
from pathlib import Path

from archcomp26_dp_affine_controller import affine_residual_interval
from archcomp26_dp_interval_controller import load_controller, residual_interval, sample_containment
from archcomp26_dp_p3_nohash import CONFIG, DRIVER, prepare_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch, driver, _, _ = prepare_runtime(controller_residual=False)
    import yaml
    from flowstar_gpu import polynomial, support

    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    driver.enable_determinism()
    controller = Path(yaml.safe_load(CONFIG.read_text())["model_dir"])
    layers = load_controller(controller, torch, "cuda:0")

    cases = (("one_box", args.smoke_config), ("all_225_boxes", CONFIG))
    reports = {}
    first_box = None
    for label, path in cases:
        config = yaml.safe_load(path.read_text())
        config["_config_dir"] = path.parent
        cells = driver.make_cells(config)
        assert cells.shape[1:] == (7, 2)
        if label == "one_box":
            assert cells.shape[0] == 1
            first_box = cells[0].clone()
        else:
            assert cells.shape[0] == 225
            assert torch.equal(cells[0], first_box)
        tables = driver.build_tables(7, 3).to("cuda:0")
        step = polynomial.build_step_tables(tables, 0.01)
        schedule = driver.build_schedule(7, 3, "cuda:0")
        engine = support.SparseEngine(tables, step, "cuda:0")
        state = driver.initial_sparse_state(cells.to("cuda:0"), engine, schedule)
        boxes = driver.hull_ranges_s(state, engine, 4)
        model = driver.build_crown(config, "cuda:0", relax="same-slope", input_layout="native")
        T, L, U = driver.crown_bounds(model, config, boxes[..., 0].contiguous(), boxes[..., 1].contiguous(), input_layout="native")
        T, L, U = driver.apply_crown_transport(T, L, U, "native-f64")
        network, linear, residual = residual_interval(boxes, T, layers)
        affine_network, affine_residual, layer_diagnostics = affine_residual_interval(boxes, T, layers)
        width = residual[..., 1] - residual[..., 0]
        affine_width = affine_residual[..., 1] - affine_residual[..., 0]
        lanes = ([0] if label == "one_box" else
                 sorted(set((0, 1, 10, 224, int(affine_width.amax(dim=1).argmax().item())))))
        point_check = sample_containment(
            boxes, T, residual, layers, lanes,
        )
        affine_point_check = sample_containment(
            boxes, T, affine_residual, layers, lanes,
        )
        bad = ~(torch.isfinite(L) & torch.isfinite(U) & (L <= U))
        reports[label] = {
            "num_boxes": int(cells.shape[0]),
            "raw_first_box": cells[0, :4].tolist(),
            "crown_input_hull_first_box": boxes[0].detach().cpu().tolist(),
            "T_first_box": T[0].detach().cpu().tolist(),
            "L_first_box": L[0].detach().cpu().tolist(),
            "U_first_box": U[0].detach().cpu().tolist(),
            "U_minus_L_first_box": (U[0] - L[0]).detach().cpu().tolist(),
            "T_all_finite": bool(torch.isfinite(T).all()),
            "L_all_finite": bool(torch.isfinite(L).all()),
            "U_all_finite": bool(torch.isfinite(U).all()),
            "bad_certificate_count": int(bad.sum().item()),
            "bad_certificate_first_12": torch.nonzero(bad).cpu().tolist()[:12],
            "directed_network_first_box": network[0].detach().cpu().tolist(),
            "directed_Tx_first_box": linear[0].detach().cpu().tolist(),
            "directed_residual_first_box": residual[0].detach().cpu().tolist(),
            "directed_residual_width_first_box": width[0].detach().cpu().tolist(),
            "directed_residual_width_max": float(width.max().item()),
            "directed_residual_width_median": float(width.median().item()),
            "point_sample_check": point_check,
            "affine_network_first_box": affine_network[0].detach().cpu().tolist(),
            "affine_residual_first_box": affine_residual[0].detach().cpu().tolist(),
            "affine_residual_width_first_box": affine_width[0].detach().cpu().tolist(),
            "affine_residual_width_max": float(affine_width.max().item()),
            "affine_residual_width_median": float(affine_width.median().item()),
            "point_sample_lanes": lanes,
            "affine_residual_inside_simple_interval_count": int(
                ((residual[..., 0] <= affine_residual[..., 0])
                 & (affine_residual[..., 1] <= residual[..., 1])).sum().item()
            ),
            "affine_point_sample_check": affine_point_check,
            "affine_layer_diagnostics": layer_diagnostics,
        }
    result = {
        "scope": "CROWN T plus independent directed interval and affine-ReLU residuals on actual initial sparse-state hull; native-f64 and native input layout, no plant integration",
        "controller": str(config["model_dir"]),
        "shared_driver": str(DRIVER),
        "results": reports,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
