#!/usr/bin/env python3
"""CPU-only P3 six-control layout check for the saved Airplane full box."""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path

import archcomp26_dp_p3_nohash as helper


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
SOURCE = N / "runs/archcomp26_20261001/airplane_continuous_order3_fullbox_20261002/PREFLIGHT.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(SOURCE.read_text())
    if (source["full_initial_box_count"] != 1 or source["ode_order"] != 3
            or source["ode_substeps"] != 200
            or source["controller_input"]["shape"] != ["N", 12]
            or source["controller_output"]["shape"] != ["N", 6]
            or source["physical_initial_box"] !=
            ([[0, 0]] * 3 + [[0.0, 1.0]] * 6 + [[0, 0]] * 3)
            or source["gpu_initialized"] is not False):
        raise ValueError("saved Airplane full-box source preflight differs")
    captured = io.StringIO()
    with redirect_stdout(captured):
        helper.self_check()
    injection = json.loads(captured.getvalue().strip().splitlines()[-1])
    if (injection.get("airplane_rows") != [13, 14, 15, 16, 17, 18]
            or injection.get("airplane_finite_ordered") is not True):
        raise RuntimeError("P3 Airplane six-output CPU injection check failed")
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only preflight initialized CUDA")
    if not (7**20 < 2**63 and 8**20 < 2**63 and 9**20 >= 2**63):
        raise RuntimeError("P3/P4 radix-size assumptions changed")
    report = {
        "schema": "archcomp26-airplane-p3-six-output-cpu-preflight-nohash-v1",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "source_preflight": str(SOURCE),
        "full_initial_box_count": 1, "physical_initial_box": source["physical_initial_box"],
        "numeric_profile": "Taylor order 3; 20 periods x 0.1 s; smoke 1 period x 10 substeps",
        "validation_policy": "solution_order (strict validation at order 3)",
        "order4_radix_limit": "9^20 >= 2^63; saved order4 validation table unsupported",
        "p3_engine_variables": 19, "nn_input_dimensions": 12,
        "nn_output_dimensions": 6, "control_rows": injection["airplane_rows"],
        "cpu_strict_injection_finite_ordered": injection["airplane_finite_ordered"],
        "gpu_initialized": torch.cuda.is_initialized(),
        "numerical_run_started": False, "content_digest_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Airplane P3 full-box six-output CPU preflight passed")


if __name__ == "__main__":
    main()
