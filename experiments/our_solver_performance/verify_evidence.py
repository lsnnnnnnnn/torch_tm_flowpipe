"""Recompute this goal's reference-defect stop branch from submitted evidence."""
import argparse
import csv
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from experiments.xiangru_adoption.common import measure
from experiments.xiangru_adoption.verify_evidence import require, checksum
from experiments.xiangru_adoption.build_tables import read_csv, write_csv
from experiments.our_solver_performance.reference_endpoint import BASE, PREVIOUS, STOP, SOURCE_FILES, ENTRIES

REPO = Path(__file__).resolve().parents[2]
FROZEN = "artifacts/runs/xiangru_adoption_20260907T032448Z"
DRIVER = "f37fd212e530bffc9518262970df8d5892f07d8a"
NOT_RUN = "NOT_RUN_REFERENCE_DEFECT_STOP"
PLANTS = (("brusselator", "brusselator", 20), ("van_der_pol", "vdp", 10))
METRICS = ("endpoint_x", "endpoint_y", "tube_x", "tube_y")


def read(path):
    return json.loads(path.read_text())


def blob(sha, relative):
    return subprocess.check_output(["git", "-C", str(REPO), "show", f"{sha}:{relative}"])


def derived_tables(root):
    timings, widths = [], []
    for plant, short, horizon in PLANTS:
        for mode, directory in (("reference", "our_" + short + "_full"), ("native_flowstar", "native_" + short)):
            relative = f"{FROZEN}/raw_minimal/{directory}/summary.json"
            raw = read(REPO / relative)
            timings.append(dict(plant=plant, horizon=horizon, mode=mode, status="REUSED",
                                accepted_steps=raw["accepted_steps"], solve_seconds=raw["solve_seconds"],
                                export_seconds=raw["export_seconds"], speedup="", source=relative))
        timings.append(dict(plant=plant, horizon=horizon, mode="optimized", status=NOT_RUN,
                            accepted_steps="", solve_seconds="", export_seconds="", speedup="", source=""))
        path = f"{FROZEN}/raw_minimal/our_{short}_full/bounds.csv"
        prior = list(csv.DictReader(blob(BASE, path).decode().splitlines()))
        current = read_csv(REPO / path)
        require(len(prior) == len(current) == 1000, "frozen complete step coverage")
        for old, now in zip(prior, current, strict=True):
            require(old == now, "frozen full-run bounds changed")
            for metric in METRICS:
                widths.append(dict(plant=plant, step=now["step"], t_end_exact=now["t_end_exact"],
                                   metric=metric, archived_lower=now[f"published_{metric}_lo"],
                                   archived_upper=now[f"published_{metric}_hi"], archive_equal_to_base=True,
                                   optimized_lower="", optimized_upper="", fresh_pair_equal="", status=NOT_RUN))
    return {"full_timings.csv": timings, "full_width_equivalence.csv": widths}


def test_summary(root):
    commands = read(root / "tests/commands.json")
    rows = []
    for command in commands["complete_matrix"]:
        path = root / command["xml"]
        document = ET.parse(path).getroot()
        cases = document.findall(".//testcase")
        failed = [c for c in cases if c.find("failure") is not None or c.find("error") is not None]
        skipped = [c for c in cases if c.find("skipped") is not None]
        exit_code = int((root / command["exit"]).read_text())
        rows.append(dict(group=command["group"], passed=len(cases)-len(failed)-len(skipped),
                         skipped=len(skipped), failed=len(failed), exit_code=exit_code,
                         failures=[c.attrib["name"] for c in failed],
                         source_sha=command["source_sha"], log=command["log"], xml=command["xml"],
                         seconds=sum(float(s.get("time", 0)) for s in document.findall(".//testsuite"))))
    totals = {k: sum(row[k] for row in rows) for k in ("passed", "skipped", "failed")}
    return dict(groups=rows, distinct_project_totals=totals, targeted_tests_already_in_root=True,
                historical_fixture_available=True, all_project_green=False,
                current_numerical_regressions_all_pass=False,
                failures_are_reference_endpoint_defect=True)


def verify(root, check_hashes=True, check_tests=True):
    if check_hashes:
        require((root / "SHA256SUMS").read_text() == checksum(root), "outer checksums")
    source = read(root / "SOURCE_MAP.json")
    require(source["base_sha"] == BASE and source["previous_numerical_sha"] == PREVIOUS
            and source["endpoint_driver_sha"] == DRIVER and source["implementation_patch_sha"] is None,
            "source identity")
    for relative, digest in source["reused_files"].items():
        payload = (REPO / relative).read_bytes()
        require(payload == blob(BASE, relative), "reused evidence differs from base: " + relative)
        require(hashlib.sha256(payload).hexdigest() == digest, "reused digest")
    require(subprocess.call(["git", "-C", str(REPO), "diff", "--quiet", BASE, "--", "src"]) == 0,
            "solver source changed after stop")
    raw = read(root / "raw_minimal/endpoint_reference.json")
    require(raw["source_sha"] == DRIVER and raw["numerical_base_sha"] == BASE and raw["source_clean"], "witness source")
    require(raw["imported_package"] == raw["source_root"] + "/src/torch_tm_flowpipe/__init__.py", "import identity")
    require((raw["device"], raw["dtype"], raw["batch"], raw["threads"], raw["interop_threads"], raw["cpu_affinity"])
            == ("cpu", "float64", 1, 1, 1, [5]), "witness runtime contract")
    for relative in SOURCE_FILES:
        payload = blob(BASE, relative)
        require(payload == blob(PREVIOUS, relative) == blob(DRIVER, relative), "reference numerical continuity")
        require(raw["source_files"][relative] == hashlib.sha256(payload).hexdigest(), "witness source digest")
    time = float.fromhex(raw["time_hex"])
    exact = -1 + 100 * F(time)
    require(raw["time_hex"] == (0.01).hex() and raw["time_fraction"] == str(F(time))
            and exact == F(3, 144115188075855872) and raw["exact_value"] == str(exact), "binary64 exact input")
    require(raw["order"] == 4 and raw["cutoff_hex"] == (1e-10).hex(), "local contract")
    require([r["entry"] for r in raw["rows"]] == list(ENTRIES), "actual entry coverage")
    domain = [[(-1.).hex(), (1.).hex()]] * 2
    terms = [{"degrees": [0, 0, 0], "coefficient": [(-1.).hex()]*2},
             {"degrees": [0, 0, 1], "coefficient": [(100.).hex()]*2}]
    for row in raw["rows"]:
        expected_input = dict(domain=domain+[[0.0.hex(), time.hex()]], variables=["ux", "uy", "tau"],
                              components=[dict(remainder=[0.0.hex()]*2, terms=terms)]*2)
        require(row["input_model"] == expected_input, "input model changed")
        # Independently reconstruct the stored binary64 cancellation and the
        # existing cutoff/range outward padding; this is NOT a sound repair.
        require(-1. + 100. * time == 0., "recorded point cancellation")
        radius = math.ulp(0.) if row["entry"] == "published_endpoint" else 0.
        rem = [(-radius if radius else 0.).hex(), radius.hex()]
        expected_output = dict(domain=domain, variables=["ux", "uy"],
                               components=[dict(remainder=rem, terms=[])]*2)
        require(row["output_model"] == expected_output, "substitution output changed")
        represented = [[v.hex() for v in pair] for pair in measure(row["output_model"])]
        require(row["represented_range_hex"] == represented, "complete represented range")
        full = [[math.nextafter(-radius, -math.inf).hex(), math.nextafter(radius, math.inf).hex()]]*2
        require(row["full_range_hex"] == full, "interval endpoint changed")
        contained = [F(float.fromhex(lo)) <= exact <= F(float.fromhex(hi)) for lo, hi in full]
        require(row["contains"] == contained == [False, False], "containment and stop classification")
    contract = read(root / "EXECUTION_CONTRACT.json")
    matched = read(REPO / FROZEN / "MATCHED_CONTRACTS.json")
    require(contract["matched_contracts"] == matched, "inherited execution parameters")
    require(contract["numerical_changes"] is False and contract["fresh_solver_runs"] == 0, "stop scope")
    decision = read(root / "optimization_decision.json")
    require(decision == dict(status=STOP, implementation_authorized=False, mechanism_implemented=False,
                             replay_decision=NOT_RUN, fresh_profile=NOT_RUN, f=None, s=None, predicted_speedup=None,
                             reason="Goal section 2.2: exact underenclosure at our reference endpoint entry."),
            "optimization/replay decision changed")
    expected_result = dict(status=STOP, implementation_patch_sha=None, numerical_source_unchanged=True,
                           endpoint_exact_containment=False, optimized_mode_added=False,
                           fresh_full_runs=0, speedup_established=False, fresh_width_equivalence=NOT_RUN,
                           historical_fixture_available=True)
    require(read(root / "RESULT.json") == expected_result, "final state changed")
    with tempfile.TemporaryDirectory() as tmp:
        for name, rows in derived_tables(root).items():
            path = Path(tmp) / name
            write_csv(path, rows)
            require(path.read_bytes() == (root / name).read_bytes(), "recomputed table differs: " + name)
    fixture = read(root / "raw_minimal/historical_fixture_recovery.json")
    require(fixture["historical_fixture_available"] and fixture["digest"] == fixture["expected"]
            == "3da0feed583dde3055fbbe039de32edec9516a2028562585eab448567a1c3f02", "historical fixture digest")
    fixture_manifest = read(root / "raw_minimal/historical_fixture_manifest.json")
    require(hashlib.sha256(json.dumps(fixture_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            == fixture["digest"] and len(fixture_manifest) == fixture["file_count"] == 25,
            "historical manifest content")
    if check_tests:
        summary = test_summary(root)
        require(read(root / "test_results.json") == summary, "test accounting")
        require(len(summary["groups"]) == 8, "complete isolated matrix")
        for row in summary["groups"]:
            if row["group"] == "root":
                require(row["exit_code"] == 1 and set(row["failures"]) == {
                    f"test_binary64_endpoint_full_model_contains_exact_value[{e}]" for e in ENTRIES},
                    "unexpected root failure or hidden endpoint regression")
            else:
                require(row["exit_code"] == 0 and row["failed"] == 0, "new isolated-group failure")
    return dict(status="VERIFIED", decision=STOP, endpoint_entries=2, failing_components=4,
                archived_bound_rows=8000, fresh_solver_runs=0,
                verification_scope="Submitted evidence and exact local witness; no long experiments rerun.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
