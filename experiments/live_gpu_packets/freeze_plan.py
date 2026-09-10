"""Freeze the complete finite experiment plan before any official execution."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import torch

from experiments.endpoint_roundoff_repair.frozen import setup
from experiments.range_batch_device.common import save, sha
from experiments.live_range_solver.runner import PARTITION, ROOT
from torch_tm_flowpipe.range_packets import DEFAULT_PACKET_LIMITS

from .campaign import FORMAL_ORDERS, PARENT_ROOT, diagnostic_cases, formal_cases, scientific_identity
from .compare_horizons import CPU_OBJECTS, FLOWSTAR_OBJECTS


GOAL = Path("/srv/local/shengenli/codex/goal_vdp_terminal.md")


def freeze(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise SystemExit("the scientific tree must be committed and clean before plan freeze")
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader"], text=True).strip()
    plan = dict(schema="live-gpu-packet-global-plan-v1", frozen_before_official_execution=True,
        frozen_utc=datetime.now(timezone.utc).isoformat(), source_sha=head,
        parent_delivery="e4d920aa710bd407666c60285632a4c294edb9da",
        goal_path=str(GOAL), goal_sha256=sha(GOAL), scientific_sources=scientific_identity(),
        partition_path=str(PARTITION.relative_to(ROOT)), partition_sha256=sha(PARTITION),
        parent_artifact=str(PARENT_ROOT.relative_to(ROOT)),
        parent_files={name: sha(PARENT_ROOT / name) for name in
            ("PERFORMANCE_RESULT.json", "paired_speedups.csv", "time_partition.csv",
             "actual_grouping.csv", "coverage/coverage_summary.json")},
        diagnostic_cases=diagnostic_cases(), formal_cases=formal_cases(),
        formal_orders=[list(row) for row in FORMAL_ORDERS], formal_blocks=5,
        no_post_hoc_formal_samples=True,
        full_horizon_cases=[dict(plant="van_der_pol", route="Gp", batch=1,
                                 original_unpartitioned=True, steps=1000, target="T10"),
                            dict(plant="brusselator", route="Gp", batch=1,
                                 original_unpartitioned=True, steps=1000, target="T20")],
        configurations={plant: setup(plant)[0].as_dict()
                        for plant in ("van_der_pol", "brusselator")},
        packet_limits=DEFAULT_PACKET_LIMITS.__dict__, packet_mode_default=False,
        scheduling=dict(max_wait_s=.020, max_group=32, oldest_expired_first=True,
                        no_additional_wait_search=True),
        performance_gate=dict(G0_over_Gp_median_min=1.15,
            Gp_faster_than_G0_pairs_min=4, pairs_per_plant=5,
            compare_to_faster_strict_cpu=True,
            stable_cpu_regression_definition=(
                "median(min(S,Q)/Gp)<1 and Gp faster than min(S,Q) in at most 2/5 pairs"),
            stable_cpu_regression_forbidden=True,
            scratch_no_sustained_growth_definition=(
                "per formal Gp run, final quarter of packet sequence has no capacity allocation"),
            scratch_no_sustained_growth_required=True),
        execution_order=["affected_tests", "reused_parent_opportunity", "diagnostic_matrix",
                         "correctness_gate", "formal_five_blocks", "full_horizons",
                         "current_observer_width_comparison", "package_acceptance"],
        environment=dict(python=subprocess.check_output([
            "/srv/local/shengenli/miniforge3/envs/py11/bin/python", "-c",
            "import sys;print(sys.version.replace('\\n',' '))"], text=True).strip(),
            torch=torch.__version__, imported_torch=str(Path(torch.__file__).resolve()),
            gpu=gpu, cuda_visible_devices="0", cpu_affinity=[2],
            torch_intra_op_threads=1, torch_inter_op_threads=1,
            omp_threads=1, mkl_threads=1, openblas_threads=1),
        reused_complete_object_sources={str(path.relative_to(ROOT)): sha(path) for path in
                                        (*CPU_OBJECTS.values(), *FLOWSTAR_OBJECTS.values())},
        claims=dict(full_gpu_engine=False, whole_solver_formal_proof=False,
                    saved_answers_used_to_advance=False))
    save(output / "PLAN_FROZEN.json", plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output), indent=2))


if __name__ == "__main__":
    main()
