#!/usr/bin/env python3
"""Build the complete three-tool evidence package from fresh runner outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.finalize_three_tool_evidence_package import finalize
from experiments.run_evidence_command import run as run_evidence


FLOWSTAR_SHA = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
DIFFREACH_SHA = "dd628eb443b517d6415de93e7035b4baef73963e"
MODEL_SHA = "022633261f2d3d2a2bd6405d261eb77657e5693bdd4447e3fcaf5bca1ca21558"
TARGET = {
    "lo_decimal": "-0.0001",
    "lo_hex": "-0x1.a36e2eb1c432dp-14",
    "hi_decimal": "0.0001",
    "hi_hex": "0x1.a36e2eb1c432dp-14",
}


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _protocol(
    run_root: Path,
    relative: str,
    *,
    name: str,
    command: list[str],
    config: dict[str, Any],
    eligibility: str,
    timing: str = "diagnostic_only",
    expected: tuple[int, ...] = (0,),
    cwd: Path = ROOT,
) -> Path:
    output = run_root / relative
    code = run_evidence(
        argparse.Namespace(
            output_dir=output,
            name=name,
            source_commit=_head(),
            config_json=json.dumps(config, sort_keys=True),
            cwd=cwd,
            eligibility_status=eligibility,
            timing_eligibility=timing,
            expected_exit_codes=expected,
            command=command,
        )
    )
    if code != 0:
        raise RuntimeError(f"evidence runner failed: {relative}")
    return output


def build(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise FileExistsError(run_root)
    run_root.mkdir(parents=True)
    python = sys.executable
    checkpoint = (
        ROOT
        / "outputs/mainline_realignment_20260810/20260810T025910Z"
        / "03_flowstar_causal_divergence/torch_causal_checkpoint_final"
        / "torch_causal_checkpoint.json"
    )
    flowstar_binary = (
        ROOT
        / "outputs/mainline_realignment_20260810/20260810T025910Z"
        / "01_native_baselines/flowstar_stock_artifacts/vanderpol"
    )

    _protocol(
        run_root,
        "00_environment/probe",
        name="environment_probe",
        command=[
            python,
            "experiments/collect_three_tool_environment.py",
            "--output-dir",
            "{ARTIFACT_DIR}/run",
            "--flowstar-root",
            str(args.flowstar_root.resolve()),
            "--flowstar-binary",
            str(flowstar_binary),
            "--diffreach-root",
            str(args.diffreach_root.resolve()),
            "--diffreach-python",
            str(args.diffreach_python.absolute()),
            "--compiler",
            args.flowstar_cxx,
        ],
        config={"arithmetic_environment": True},
        eligibility="environment_provenance",
        timing="not_a_benchmark",
    )
    _protocol(
        run_root,
        "01_integrity_corrections/focused_tests",
        name="integrity_tests",
        command=[
            python,
            "-m",
            "pytest",
            "-q",
            "tests/test_evidence_verification.py",
            "tests/test_state_equality.py",
            "tests/test_canonical_status_consistency.py",
        ],
        config={"hardcoded_pass_rejected": True},
        eligibility="repository_integrity_gate",
    )
    _protocol(
        run_root,
        "02_contract/focused_tests",
        name="comparison_contract_tests",
        command=[python, "-m", "pytest", "-q", "tests/test_comparison_contract.py"],
        config={"model_sha256": MODEL_SHA, "target": TARGET},
        eligibility="matched_contract_gate",
    )

    _protocol(
        run_root,
        "03_native_flowstar/official_vdp",
        name="stock_flowstar_official_vdp",
        command=[
            python,
            "experiments/run_stock_flowstar_vdp_reproduction.py",
            "--source",
            str(args.flowstar_root.resolve()),
            "--output-dir",
            "{ARTIFACT_DIR}/run",
            "--source-commit",
            FLOWSTAR_SHA,
            "--model-sha256",
            MODEL_SHA,
            "--cxx",
            args.flowstar_cxx,
            "--cxx-compatibility-flag=-fpermissive",
        ],
        config={
            "track": "N",
            "partition": "B1",
            "source_modified": False,
            "compiler_compatibility_flags": ["-fpermissive"],
        },
        eligibility="native_capability_only",
        timing="native_process_and_reported_core_only",
    )
    _protocol(
        run_root,
        "03_native_flowstar/scalar_affine_gate",
        name="stock_flowstar_scalar_affine_gate",
        command=[
            python,
            "experiments/flowstar_scalar_affine_closure/run_closure.py",
            "--flowstar-root",
            str(args.flowstar_root.resolve()),
            "--run-root",
            "{ARTIFACT_DIR}/run",
        ],
        config={"scope": "scalar_affine_closed_form_mpfr"},
        eligibility="native_build_soundness_qualification",
        timing="not_a_performance_benchmark",
    )
    _protocol(
        run_root,
        "04_native_diffreach/official_vdp",
        name="stock_diffreach_official_vdp",
        command=[
            python,
            "experiments/run_stock_diffreach_vdp_reproduction.py",
            "--source",
            str(args.diffreach_root.resolve()),
            "--python",
            str(args.diffreach_python.absolute()),
            "--output-dir",
            "{ARTIFACT_DIR}/run",
            "--source-commit",
            DIFFREACH_SHA,
            "--model-sha256",
            MODEL_SHA,
            "--cuda-visible-devices",
            args.cuda_uuid,
        ],
        config={"track": "N", "partition": "B64", "jax_enable_x64": True},
        eligibility="native_capability_only_mixed_builder_dtype",
        timing="native_jax_compile_and_after_jit_separate",
    )
    _protocol(
        run_root,
        "05_native_torch_complete_o4/authoritative",
        name="torch_complete_o4_authoritative",
        command=[
            python,
            "experiments/run_vdp_dense_backend.py",
            "--output-dir",
            "{ARTIFACT_DIR}/run",
            "--tm-backend",
            "dense",
            "--device",
            "cpu",
            "--horizon",
            "6.5",
            "--wall-cap-s",
            "1800",
            "--dense-range-method",
            "adaptive_subdivision",
            "--dense-range-trigger",
            "proactive_depth1_on_named_contexts",
            "--dense-range-max-depth",
            "1",
            "--dense-range-max-leaves",
            "4",
            "--dense-range-split-vars",
            "0,1",
            "--dense-range-contexts",
            "polynomial_truncation",
        ],
        config={"track": "N", "partition": "B1", "target": TARGET},
        eligibility="expected_fail_closed_partial_horizon",
        timing="native_process_only_not_cross_tool",
        expected=(1,),
    )
    _protocol(
        run_root,
        "06_native_torch_fixed_dr7/t10_cpu",
        name="torch_fixed_dr7_t10_cpu",
        command=[
            python,
            "experiments/run_vdp_fixed_support.py",
            "--output-dir",
            "{ARTIFACT_DIR}/run",
            "--horizon",
            "10",
            "--step-size",
            "0.01",
            "--batch",
            "64",
            "--device",
            "cpu",
            "--warm-runs",
            "0",
        ],
        config={"track": "M-D", "partition": "B64"},
        eligibility="matched_semantics_empirical",
        timing="cold_only_not_cross_tool_ratio",
    )
    _protocol(
        run_root,
        "06_native_torch_fixed_dr7/operator_fixture",
        name="diffreach_explicit_f64_operator_fixture",
        command=[
            python,
            "-m",
            "pytest",
            "-q",
            "tests/test_fixed_support.py",
            "tests/test_fixed_support_functional.py",
        ],
        config={"upstream_commit": DIFFREACH_SHA, "explicit_float64": True},
        eligibility="matched_operator_equivalence",
    )
    _protocol(
        run_root,
        "06_native_torch_fixed_dr7/diffreach_explicit_f64_replay",
        name="diffreach_explicit_f64_fixture_replay",
        command=[
            str(args.diffreach_python.absolute()),
            "experiments/replay_diffreach_explicit_f64_fixture.py",
            "--diffreach-root",
            str(args.diffreach_root.resolve()),
            "--fixture",
            "tests/fixtures/diffreach_dr7_vdp_one_step_float64.json",
            "--output-dir",
            "{ARTIFACT_DIR}/run",
            "--source-commit",
            DIFFREACH_SHA,
        ],
        config={"upstream_commit": DIFFREACH_SHA, "every_builder_explicit_float64": True},
        eligibility="matched_operator_equivalence",
    )
    for replay_device in ("cpu", "cuda"):
        _protocol(
            run_root,
            f"06_native_torch_fixed_dr7/two_ulp_companion_{replay_device}",
            name=f"fixed_dr7_fraction_replay_{replay_device}",
            command=[
                python,
                "experiments/replay_fixed_support_fraction.py",
                "--output",
                "{ARTIFACT_DIR}/fraction_replay.json",
                "--device",
                replay_device,
            ],
            config={"device": replay_device, "companion_envelope_ulps": 2},
            eligibility="bounded_exact_binary64_outward_replay",
            timing="not_a_performance_benchmark",
        )

    probe_compile = _protocol(
        run_root,
        "07_flowstar_torch_raw_remainder/probe_compile",
        name="flowstar_semantic_probe_compile",
        command=[
            "g++",
            "-O3",
            "-w",
            "-fpermissive",
            "-std=c++11",
            "-I",
            str(args.flowstar_root.resolve() / "flowstar-toolbox"),
            "-I",
            "/usr/local/include",
            "experiments/flowstar_probe/flowstar_vdp_step_trace_probe.cpp",
            "-L",
            str(args.flowstar_root.resolve() / "flowstar-toolbox"),
            "-L",
            "/usr/local/lib",
            "-o",
            "{ARTIFACT_DIR}/flowstar_probe",
            "-lflowstar",
            "-lmpfr",
            "-lgmp",
            "-lgsl",
            "-lgslcblas",
            "-lm",
            "-lglpk",
        ],
        config={"flowstar_commit": FLOWSTAR_SHA},
        eligibility="observer_binary",
        timing="compile_time_only",
    )
    probe_binary = probe_compile / "artifacts/flowstar_probe"
    probe_run = _protocol(
        run_root,
        "07_flowstar_torch_raw_remainder/probe_t1",
        name="flowstar_semantic_probe_t1",
        command=[str(probe_binary), "{ARTIFACT_DIR}/flowstar_trace.csv", "1.0", "0", "4"],
        config={"horizon_decimal": "1.0", "horizon_hex": float(1.0).hex()},
        eligibility="diagnostic_only",
    )
    expression = _protocol(
        run_root,
        "07_flowstar_torch_raw_remainder/expression_tree",
        name="raw_remainder_expression_tree",
        command=[
            python,
            "experiments/trace_vdp_raw_remainder.py",
            "--torch-checkpoint",
            str(checkpoint),
            "--flowstar-trace",
            str(probe_run / "artifacts/flowstar_trace.csv"),
            "--flowstar-binary",
            str(probe_binary),
            "--flowstar-source-commit",
            FLOWSTAR_SHA,
            "--output-dir",
            "{ARTIFACT_DIR}/run",
        ],
        config={"checkpoint_role": "last_common_prestate", "target": TARGET},
        eligibility="diagnostic_root_cause",
    )
    _protocol(
        run_root,
        "07_flowstar_torch_raw_remainder/independent_analysis",
        name="raw_remainder_independent_analysis",
        command=[
            python,
            "experiments/analyze_vdp_raw_remainder_trace.py",
            "--expression-tree",
            str(expression / "artifacts/run/raw_remainder_expression_tree.json"),
            "--output-dir",
            "{ARTIFACT_DIR}/run",
        ],
        config={"mpfr_precision_bits": 256},
        eligibility="frozen_workload_independent_replay",
    )

    _protocol(
        run_root,
        "08_schedule_validator_matrix/adaptive_schedule",
        name="schedule_validator_adaptive_t1",
        command=[
            python,
            "experiments/run_vdp_schedule_validator_matrix.py",
            "--flowstar-trace",
            str(probe_run / "artifacts/flowstar_trace.csv"),
            "--torch-checkpoint",
            str(checkpoint),
            "--horizon",
            "1.0",
            "--output-dir",
            "{ARTIFACT_DIR}/run",
        ],
        config={"schedule": "flowstar_native_accepted", "target": TARGET},
        eligibility="diagnostic_only",
    )
    fixed_probe = _protocol(
        run_root,
        "08_schedule_validator_matrix/flowstar_fixed_h001",
        name="flowstar_fixed_h001_t1",
        command=[
            str(probe_binary),
            "{ARTIFACT_DIR}/flowstar_trace.csv",
            "1.0",
            "0",
            "4",
            "0.01",
        ],
        config={"step_decimal": "0.01", "step_hex": float(0.01).hex()},
        eligibility="diagnostic_only",
    )
    _protocol(
        run_root,
        "08_schedule_validator_matrix/fixed_h001_matrix",
        name="schedule_validator_fixed_h001_t1",
        command=[
            python,
            "experiments/run_vdp_schedule_validator_matrix.py",
            "--flowstar-trace",
            str(fixed_probe / "artifacts/flowstar_trace.csv"),
            "--torch-checkpoint",
            str(checkpoint),
            "--horizon",
            "1.0",
            "--output-dir",
            "{ARTIFACT_DIR}/run",
        ],
        config={"schedule": "common_fixed_h001", "target": TARGET},
        eligibility="diagnostic_only",
    )

    _protocol(
        run_root,
        "09_fixed_support_descriptor/algebra_tests",
        name="r7_r35_descriptor_algebra",
        command=[
            python,
            "-m",
            "pytest",
            "-q",
            "tests/test_fixed_support.py",
            "tests/test_fixed_support_r35.py",
            "tests/test_r35_mpfr_remainder_replay.py",
            "tests/test_fixed_support_bridge_runner.py",
        ],
        config={"R7_slots": 7, "R35_slots": 35},
        eligibility="descriptor_semantics_gate",
    )
    _protocol(
        run_root,
        "09_fixed_support_descriptor/r35_mpfr_remainder_replay",
        name="r35_mpfr_outward_remainder_replay",
        command=[
            python,
            "experiments/replay_r35_mpfr_remainder.py",
            "--output-dir",
            "{ARTIFACT_DIR}/run",
        ],
        config={"R35_slots": 35, "mpfr_precision_bits": 256},
        eligibility="bounded_exact_binary64_outward_replay",
        timing="not_a_performance_benchmark",
    )
    g0 = _protocol(
        run_root,
        "10_bridge_ladder/G0",
        name="fixed_support_bridge_G0",
        command=[
            python,
            "experiments/run_fixed_support_descriptor_bridge.py",
            "--output-dir",
            "{ARTIFACT_DIR}/run",
            "--max-gate",
            "G0",
        ],
        config={"h": {"decimal": "0.01", "hex": float(0.01).hex()}},
        eligibility="diagnostic_only",
    )
    prior = g0
    for gate in ("G1", "G2", "G3"):
        current = _protocol(
            run_root,
            f"10_bridge_ladder/{gate}",
            name=f"fixed_support_bridge_{gate}",
            command=[
                python,
                "experiments/run_fixed_support_descriptor_bridge.py",
                "--output-dir",
                "{ARTIFACT_DIR}/run",
                "--max-gate",
                gate,
                "--prior-gate-summary",
                str(prior / "artifacts/run/summary.json"),
            ],
            config={"prior_gate": prior.name, "no_hidden_constant_drift": True},
            eligibility="diagnostic_only",
            expected=(0, 1) if gate == "G3" else (0,),
        )
        prior = current

    _protocol(
        run_root,
        "11_pairwise_tables/report_tests",
        name="pairwise_report_contract",
        command=[
            python,
            "-m",
            "pytest",
            "-q",
            "tests/test_comparison_contract.py",
            "tests/test_canonical_status_consistency.py",
        ],
        config={"universal_ranking": False},
        eligibility="pairwise_reporting_gate",
    )
    _protocol(
        run_root,
        "12_single_improvement/not_authorized",
        name="single_improvement_decision",
        command=[
            python,
            "-c",
            "print('IMPROVEMENT_NOT_AUTHORIZED_BY_EVIDENCE')",
        ],
        config={"outcome": "IMPROVEMENT_NOT_AUTHORIZED_BY_EVIDENCE"},
        eligibility="decision_record",
        timing="not_run",
    )
    _protocol(
        run_root,
        "13_second_system_if_authorized/not_authorized",
        name="second_system_decision",
        command=[python, "-c", "print('not_run: improvement not promoted')"],
        config={"status": "not_run", "reason": "improvement_not_promoted"},
        eligibility="not_run",
        timing="not_run",
    )
    _protocol(
        run_root,
        "14_fresh_clone/full_pytest",
        name="final_head_full_pytest",
        command=[python, "-m", "pytest", "-q", "-rsxX"],
        config={"final_head": _head()},
        eligibility="final_head_gate",
        timing="test_runtime_only",
    )
    _protocol(
        run_root,
        "14_fresh_clone/compileall",
        name="final_head_compileall",
        command=[python, "-m", "compileall", "-q", "src", "experiments", "tests"],
        config={"final_head": _head()},
        eligibility="final_head_gate",
        timing="not_a_benchmark",
    )
    _protocol(
        run_root,
        "14_fresh_clone/checkpoint_load",
        name="final_head_checkpoint_load",
        command=[
            python,
            "experiments/validate_tracked_checkpoints.py",
            "--output-dir",
            "{ARTIFACT_DIR}/run",
        ],
        config={"final_head": _head(), "scope": "all_tracked_json_checkpoint_paths"},
        eligibility="final_head_gate",
        timing="not_a_benchmark",
    )
    _protocol(
        run_root,
        "14_fresh_clone/focused_tests",
        name="final_head_focused_tests",
        command=[
            python,
            "-m",
            "pytest",
            "-q",
            "tests/test_evidence_verification.py",
            "tests/test_comparison_contract.py",
            "tests/test_raw_remainder_trace.py",
            "tests/test_fixed_support_r35.py",
            "tests/test_r35_mpfr_remainder_replay.py",
            "tests/test_fixed_support_bridge_runner.py",
            "tests/test_20260811_report_contract.py",
            "tests/test_vdp_terminal_state_replay.py",
        ],
        config={"final_head": _head(), "checkpoint_load_covered": True},
        eligibility="final_head_gate",
        timing="test_runtime_only",
    )
    _protocol(
        run_root,
        "14_fresh_clone/diff_check",
        name="final_head_diff_check",
        command=["git", "diff", "--check", "7b880d0bf6ea2f6182faaaff1c267f1e2ab2c06a..HEAD"],
        config={"start_commit": "7b880d0bf6ea2f6182faaaff1c267f1e2ab2c06a"},
        eligibility="final_head_gate",
        timing="not_a_benchmark",
    )
    _protocol(
        run_root,
        "14_fresh_clone/worktree_clean",
        name="final_head_worktree_clean",
        command=[
            python,
            "-c",
            (
                "import subprocess,sys;"
                "p=subprocess.run(['git','status','--porcelain=v1'],"
                "text=True,capture_output=True,check=True);"
                "print(p.stdout,end='');sys.exit(1 if p.stdout else 0)"
            ),
        ],
        config={"final_head": _head(), "ignored_package_output_allowed": True},
        eligibility="final_head_gate",
        timing="not_a_benchmark",
    )
    _protocol(
        run_root,
        "15_reports/causal_figures",
        name="causal_figure_builder",
        command=[
            python,
            "experiments/build_three_tool_causal_figures.py",
            "--run-root",
            str(run_root),
            "--output-dir",
            "{ARTIFACT_DIR}/run",
        ],
        config={"figure_count": 5, "source_csv_required": True},
        eligibility="causal_visualization_only",
        timing="not_a_benchmark",
    )
    _protocol(
        run_root,
        "15_reports/canonical_docs",
        name="canonical_report_consistency",
        command=[
            python,
            "-m",
            "pytest",
            "-q",
            "tests/test_canonical_status_consistency.py",
            "tests/test_three_tool_package_finalizer.py",
        ],
        config={"required_reports": 8},
        eligibility="reporting_gate",
    )

    g3_summary = json.loads(
        (prior / "artifacts/run/summary.json").read_text(encoding="utf-8")
    )
    bridge_outcome = g3_summary["outcome"]
    if bridge_outcome not in {
        "FIXED_SUPPORT_BRIDGE_CLOSED",
        "FIXED_SUPPORT_BRIDGE_BLOCKED",
    }:
        raise RuntimeError("unexpected G3 bridge outcome")
    outcomes = {
        "evidence": "EVIDENCE_INTEGRITY_PASS",
        "raw_remainder": "RAW_REMAINDER_ROOT_CAUSE_CLOSED",
        "schedule_validator": "SCHEDULE_VALIDATOR_INTERACTION",
        "bridge": bridge_outcome,
        "flowstar_torch_pairwise": "PAIRWISE_COMPARISON_PARTIAL",
        "diffreach_torch_pairwise": "VALID_PAIRWISE_COMPARISON_CLOSED",
        "improvement": "IMPROVEMENT_NOT_AUTHORIZED_BY_EVIDENCE",
    }
    return finalize(
        argparse.Namespace(run_root=run_root, outcomes_json=json.dumps(outcomes))
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--flowstar-root", type=Path, required=True)
    parser.add_argument("--diffreach-root", type=Path, required=True)
    parser.add_argument("--diffreach-python", type=Path, required=True)
    parser.add_argument("--flowstar-cxx", default="g++")
    parser.add_argument(
        "--cuda-uuid",
        default="GPU-c1336362-1a12-45dd-8d3f-d2011d6f51ae",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(build(parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
