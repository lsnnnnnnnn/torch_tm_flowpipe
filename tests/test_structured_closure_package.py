import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


RUN_RELATIVE = Path(
    "outputs/structured_remainder_compiled_fixed_support_20260810/"
    "20260810T070908Z"
)
PRIOR_RELATIVE = Path("outputs/mainline_realignment_20260810/20260810T025910Z")
DERIVED_MACHINE_FILES = (
    "claim_registry.csv",
    "failure_attribution.json",
    "fixed_support_compiled_results.csv",
    "fixed_support_equivalence.json",
    "fixed_support_object_baseline.csv",
    "fixed_support_outward_results.csv",
    "fixed_support_profile.json",
    "memory.csv",
    "native_baselines.json",
    "second_system_results.csv",
    "soundness_matrix.csv",
    "structured_horizon_ladder.csv",
    "structured_semantics.json",
    "structured_terminal_ab.json",
    "timing.csv",
)
REQUIRED_FIGURES = (
    "complete_o4_margin_at_terminal.png",
    "fixed_support_host_sync_kernel_counts.png",
    "fixed_support_runtime_breakdown.png",
    "fixed_support_runtime_vs_batch.png",
    "fixed_support_soundness_scope.png",
    "flowstar_style_t_x_overlay.png",
    "flowstar_style_t_y_overlay.png",
    "phase_tube_overlay.png",
    "structured_width_decomposition_vs_time.png",
    "validated_horizon_by_in_framework_lane.png",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_structured_closure_package_checksums_claims_and_public_paths():
    repository_root = Path(__file__).resolve().parents[1]
    run_root = repository_root / RUN_RELATIVE
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "structured_remainder_compiled_manifest_v1"
    assert manifest["outcomes"] == json.loads(
        (run_root / "failure_attribution.json").read_text(encoding="utf-8")
    )
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=repository_root, check=True,
            capture_output=True, text=True,
        ).stdout.splitlines()
    )
    required_tracked = {
        (RUN_RELATIVE / row["path"]).as_posix() for row in manifest["files"]
    }
    required_tracked.update({
        (RUN_RELATIVE / "manifest.json").as_posix(),
        (RUN_RELATIVE / "SHA256SUMS").as_posix(),
    })
    assert required_tracked <= tracked
    for row in manifest["files"]:
        path = run_root / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert _sha(path) == row["sha256"]

    for line in (run_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        assert _sha(repository_root / relative) == expected

    with (run_root / "claim_registry.csv").open(encoding="utf-8", newline="") as handle:
        claims = {row["claim"]: row for row in csv.DictReader(handle)}
    assert claims["fixed_compiled_cpu_t10_completed_arithmetic_changed"]["formal_claim_eligible"] == "False"
    assert claims["fixed_compiled_v100_t10_completed_arithmetic_changed"]["performance_measurement_eligible"] == "True"
    assert claims["structured_s1_terminal_candidate"]["requested_horizon_completed"] == "False"

    ladder = list(csv.DictReader((run_root / "structured_horizon_ladder.csv").open(encoding="utf-8", newline="")))
    assert {row["status"] for row in ladder} == {"not_run_after_stop"}
    assert {row["fresh_request_started"] for row in ladder} == {"False"}
    assert {row["paired_baseline_started"] for row in ladder} == {"False"}

    for path in [*run_root.glob("*"), *(run_root / "raw_public").rglob("*")]:
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".jsonl", ".log", ".md", ".txt", ".yaml", ".yml"}:
            continue
        assert "/srv/local/shengenli" not in path.read_text(encoding="utf-8", errors="replace")


def test_structured_closure_derivatives_rebuild_from_committed_public_evidence(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    canonical = repository_root / RUN_RELATIVE
    rebuilt = tmp_path / RUN_RELATIVE.name
    subprocess.run(
        [
            sys.executable,
            str(repository_root / "experiments/build_structured_remainder_compiled_package.py"),
            "--run-root",
            str(rebuilt),
            "--prior-root",
            str(repository_root / PRIOR_RELATIVE),
            "--evidence-root",
            str(canonical / "raw_public"),
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    compared = [*DERIVED_MACHINE_FILES]
    compared.extend(f"figures/{name}" for name in REQUIRED_FIGURES)
    assert {name: (rebuilt / name).read_bytes() for name in compared} == {
        name: (canonical / name).read_bytes() for name in compared
    }
    assert _tree_bytes(rebuilt / "raw_public") == _tree_bytes(canonical / "raw_public")
