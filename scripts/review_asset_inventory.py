"""One-time grouped disposition inventory for the review branch."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = "8ca0bb034f597db3e16bce83ad974422a9b74bba"
SELECTED = {
    "artifacts/runs/endpoint_roundoff_repair_20260908",
    "artifacts/runs/live_gpu_packets_20260910T023603Z",
    "artifacts/runs/xiangru_adoption_20260907T032448Z",
    "artifacts/runs/vdp_c3_cross_step_causal_closure_20260827",
    "artifacts/runs/brusselator_sr1000_c4_closure_20260828",
    "artifacts/runs/brusselator_live_range_c5_20260828",
    "artifacts/runs/repaired_solver_performance_20260908T034636Z",
    "artifacts/runs/boundary_execution_20260908T172756Z",
    "artifacts/runs/range_batch_device_20260909T030609Z",
    "artifacts/runs/live_range_solver_20260909T053007Z",
    "artifacts/runs/resident_tm_block_20260914T032650Z",
}
CODE_ROOTS = ("src/", "experiments/", "tests/", "scripts/")


def group(path):
    parts = path.split("/")
    if parts[0] in ("outputs", "evidence", "audits") and len(parts) > 1:
        return "/".join(parts[:2])
    if parts[0] == "artifacts" and len(parts) > 2:
        return "/".join(parts[:3])
    if parts[0] == "src" and len(parts) > 1:
        return "/".join(parts[:2])
    if parts[0] == "experiments" and len(parts) > 1:
        return "/".join(parts[:2])
    if parts[0] == "tests":
        return "tests"
    if parts[0] in ("docs", "benchmarks", "scripts") and len(parts) > 1:
        return "/".join(parts[:2])
    return parts[0]


def main():
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().rstrip("\0").split("\0")
    groups = defaultdict(list)
    for relative in tracked:
        groups[group(relative)].append(relative)
    text_sources = {}
    for relative in tracked:
        if relative.startswith(CODE_ROOTS) and Path(relative).suffix in (".py", ".sh", ".json", ".yaml"):
            try:
                text_sources[relative] = (ROOT / relative).read_text(encoding="utf-8")
            except (UnicodeError, OSError):
                pass
    doc_sources = {}
    for relative in tracked:
        if relative.startswith(("docs/", "README.md", "handoff.md", "experiments/README.md")) and Path(relative).suffix == ".md":
            try:
                doc_sources[relative] = (ROOT / relative).read_text(encoding="utf-8")
            except (UnicodeError, OSError):
                pass
    rows = []
    for name, files in sorted(groups.items()):
        size = sum((ROOT / path).stat().st_size for path in files)
        extensions = Counter(Path(path).suffix or "[none]" for path in files)
        # Also find paths assembled as ROOT / "outputs" / "run_name".
        token = name.split("/")[-1]
        code_refs = [path for path, content in text_sources.items()
                     if path not in files and (name in content or token in content)]
        doc_refs = [path for path, content in doc_sources.items()
                    if path not in files and (name in content or token in content)]
        if name.startswith("src/"):
            decision, reason = "retain_current", "sole numerical package"
        elif name == "tests":
            decision, reason = "retain_support", "active and versioned regression collection"
        elif name in SELECTED:
            decision, reason = "retain_selected", "raw main result or directly used dependency"
        elif name.startswith(("outputs/", "evidence/", "audits/")):
            decision = "retain_support" if code_refs else "archive_candidate"
            reason = ("literal code/test/script reference found" if code_refs else
                      "no literal direct reference; check dynamic loaders and old docs before removal")
        elif name.startswith("experiments/"):
            decision = "retain_current" if name == "experiments/review_suite" else "retain_support"
            reason = "current CLI" if name == "experiments/review_suite" else "older runner/helper may be imported dynamically"
        else:
            decision, reason = "retain_pending", "review documentation/config or incidental asset"
        rows.append(dict(group_path=name, file_count=len(files), bytes=size,
                         file_types=";".join(f"{key}:{value}" for key, value in extensions.most_common(5)),
                         experiment=name.split("/")[-1], code_test_reference_count=len(code_refs),
                         direct_references=";".join(code_refs[:8]), documentation_reference_count=len(doc_refs),
                         decision=decision, reason=reason, recovery_commit=PARENT))
    out = ROOT / "docs/maintenance/asset_disposition.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    candidates = [r for r in rows if r["decision"] == "archive_candidate"]
    print(f"{len(tracked)} tracked files in {len(rows)} grouped assets; {len(candidates)} archive candidates")
    for r in candidates:
        print(f"{r['group_path']}\t{r['file_count']}\t{r['bytes']}")
    return rows


if __name__ == "__main__":
    main()
