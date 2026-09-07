"""Small extension of the previous verifier for this early-stop decision.

Recompute exact maps and every new table. This schema cannot award adoption:
no repaired candidate, full-horizon run, or performance experiment exists.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import tempfile

from experiments.xiangru_adoption.verify_evidence import require, checksum
from experiments.xiangru_adoption.build_tables import read_csv, write_csv
from experiments.backend_reevaluation.crosscheck import SHAS

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "artifacts/runs/xiangru_adoption_20260907T032448Z"
STOP = "REPAIR_REUSE_NOT_LOCAL__NO_ADOPTION"
REASON = "Third independent endpoint-substitution underenclosure; goal sections 4/8 stop candidate repairs before worktree creation."
METRICS = ["endpoint_x", "endpoint_y", "tube_x", "tube_y"]


def read(path):
    return json.loads(path.read_text())


def check_maps(root):
    witness = read(FROZEN / "known_witness_replay.json")
    source = read(root / "SOURCE_MAP.json")
    rows = []
    for impl in "ABC":
        data = read(root / "raw_minimal" / (impl + ".json"))
        require(data["source_sha"] == SHAS[impl] and data["source_clean"], "source identity/clean status")
        require(source["implementations"][impl]["sha"] == SHAS[impl] and source["implementations"][impl]["clean"], "source manifest identity")
        require(data["source_path"] == source["implementations"][impl]["path"], "source path")
        package = "torch_tm_flowpipe" if impl == "C" else "flowstar_gpu"
        require(data["imported_package"] == data["source_path"] + "/src/" + package + "/__init__.py", "actual import")
        require(data["source_files"] == source["implementations"][impl]["files"], "numerical source hashes")
        require(data["driver_sha"] == source["drivers"][impl], "driver identity")
        require(data["frozen_witness_sha256"] == hashlib.sha256((FROZEN / "known_witness_replay.json").read_bytes()).hexdigest(), "frozen witness digest")
        if impl != "C":
            require(data["cuda_kernel_names"], "actual CUDA execution")
            require(data["segment_extension"].startswith(source["extension_caches"][impl] + "/"), "isolated extension source")
        actual_coverage = set()
        for r in data["rows"]:
            mode = r["actual_mode"]
            require(r["requested_mode"] == mode and r["implementation"] == impl, "mode or implementation changed")
            require(r["batch"] == 1 and r["dtype"] == "float64" and r["local_adapter"], "map preconditions")
            key = (r["device"], r["backend"], mode, r["witness"], r["variant"])
            require(key not in actual_coverage, "duplicate coverage")
            actual_coverage.add(key)
            if r["witness"] == "history":
                require(r["matrices"] == witness["matrices"] and r["columns"] == witness["historical_j_columns"], "original history input changed")
                scales = [[1., 1.]]*3 if r["variant"] == "original" else [[1., 1.], [1.1, 1.1], [1.1, 1.1]]
                require(r["scales"] == scales, "history scale input changed")
                matrices = [[[F(v)/F(scales[k][j]) for j, v in enumerate(row)] for row in matrix]
                            for k, matrix in enumerate(r["matrices"])]
                mv = lambda m, v: [sum((x*y for x, y in zip(row, v)), F()) for row in m]
                j0, j1 = [list(map(F, col)) for col in r["columns"]]
                first = mv(matrices[1], j0)
                final = [a+b for a, b in zip(mv(matrices[2], first), mv(matrices[2], j1))]
                exact_stages = [[F(), F()], first, final]
                require(r["exact_stages"] == [[str(v) for v in stage] for stage in exact_stages], "exact history result")
                contained = all(F(lo) <= v <= F(hi) for got, want in zip(r["stages"], exact_stages, strict=True)
                                for (lo, hi), v in zip(got, want, strict=True))
                exact = [[v, v] for v in final]
                returned = [list(map(F, pair)) for pair in r["stages"][-1]]
                if impl == "B" and mode == "strict":
                    for matrix, formed in zip(matrices, r["formed_matrices"], strict=True):
                        require(formed["interval"] is not None, "new Phi interval missing")
                        require(all(F(lo) <= v <= F(hi) for row, intervals in zip(matrix, formed["interval"])
                                    for v, (lo, hi) in zip(row, intervals)), "new Phi loses reciprocal/product error")
            else:
                require(r["witness"] == "normalization" and r["variant"] == "centered_affine", "unknown map")
                require(r["input_coefficients"] == [[0., 1.1, 0.], [0., 0., 1.1]] and
                        r["input_remainders"] == [[-2.0**-52, 2.0**-52]]*2 and
                        r["domain"] == [[-1., 1.]]*2, "centered normalization input changed")
                target = F(1.1) + F(2.0**-52)
                exact = [[-target, target]]*2
                returned = []
                for c, rem, s in zip(r["retained_coefficients"], r["scaled_remainders"], r["scales"], strict=True):
                    require(c[0] == 0 and s > 0, "center/positive-scale prerequisite")
                    radius = sum((abs(F(v)) for v in c[1:]), F())
                    returned.append([F(s)*(-radius+F(rem[0])), F(s)*(radius+F(rem[1]))])
                contained = all(lo <= -target and hi >= target for lo, hi in returned)
                require(r["upper_margins"] == [str(hi-target) for lo, hi in returned], "normalization margin")
            require(r["exact_image"] == [[str(v) for v in pair] for pair in exact], "exact image tampered")
            require(r["returned_image"] == [[str(v) for v in pair] for pair in returned], "returned result cannot be reconstructed")
            require(r["contains"] == contained, "inclusion result tampered")
            rows.append({k: r[k] for k in ["implementation", "source_file", "function", "device", "backend", "actual_mode", "witness", "variant"]}
                        | {"exact_preimage": json.dumps({"matrices": r["matrices"], "columns": r["columns"], "scales": r["scales"]})
                           if r["witness"] == "history" else "P_i=binary64(1.1)*u_i; I_i=[-2^-52,2^-52]; u_i in [-1,1]",
                           "exact_image": json.dumps(r["exact_image"]), "returned_image": json.dumps(r["returned_image"]),
                           "contains": contained, "preconditions": r["adapter"], "call_chain": " -> ".join(r["call_chain"])})
        variants = [("history", "original"), ("history", "nonunit_scale"), ("normalization", "centered_affine")]
        expected = {(d, b, m, w, v) for d in ["cpu", "cuda"] for b in ["dense", "sparse"]
                    for m in ["parity", "strict"] for w, v in variants} if impl != "C" else {
                        ("cpu", "accepted_boundary_sr", "our_cpu_reference", w, v) for w, v in variants}
        require(actual_coverage == expected, "mode/backend/device coverage")
    endpoint = []
    for impl in "AB":
        data = read(root / "raw_minimal" / ("endpoint_" + impl + ".json"))
        require(data["source_sha"] == SHAS[impl] and data["source_clean"], "endpoint source identity")
        require(data["driver_sha"] == source["drivers"]["endpoint_"+impl], "endpoint driver identity")
        require({(r["device"], r["backend"]) for r in data["rows"]} == {(d,b) for d in ["cpu","cuda"] for b in ["dense","sparse"]}, "endpoint route coverage")
        for r in data["rows"]:
            require(r["requested_mode"] == r["actual_mode"] == "strict", "endpoint mode")
            require(r["pre_terms"] == [{"degrees": [0,0,0], "coefficient": -1.}, {"degrees": [1,0,0], "coefficient": 100.}]
                    and r["pre_remainders"] == [[0.,0.]]*2 and r["h"] == .01, "endpoint input tampered")
            value = F(-1.) + F(100.)*F(r["h"])
            require(str(value) == "3/144115188075855872", "endpoint exact oracle")
            returned = [[F(c)+F(lo), F(c)+F(hi)] for c, (lo,hi) in zip(r["endpoint_coefficients"], r["endpoint_remainders"], strict=True)]
            require(r["exact_image"] == [[str(value), str(value)]]*2 and
                    r["returned_image"] == [[str(v) for v in pair] for pair in returned], "endpoint output tampered")
            contained = all(lo <= value <= hi for lo,hi in returned)
            require(r["contains"] == contained and r["no_history_no_scaling"], "endpoint classification or independence")
            endpoint.append(dict(implementation=impl, device=r["device"], backend=r["backend"], contains=contained))
    return rows, endpoint


def derived(root):
    rows, endpoint = check_maps(root)
    checks = {impl: all(r["contains"] for r in rows if r["implementation"] == impl and r["actual_mode"] != "parity") for impl in "ABC"}
    third = any(not r["contains"] for r in endpoint if r["implementation"] == "A")
    decision = "OUR_REFERENCE_DEFECT_FOUND__ADOPTION_STOP" if not checks["C"] else STOP if third else "KNOWN_DEFECTS_NOT_CLOSED__NO_ADOPTION"
    tables = {"WITNESS_CROSSCHECK.csv": rows}
    for name in ["widths_full_prefix.csv", "width_summary.csv", "width_at_checkpoints.csv"]:
        tables[name] = [dict(plant=p, metric=m, mode="repaired_strict", status="NOT_STARTED", reason=REASON,
                            candidate_width="", ratio_to_ours="", ratio_to_flowstar="")
                        for p in ["brusselator", "van_der_pol"] for m in METRICS]
    tables["horizon_matrix.csv"] = []
    tables["timings_raw.csv"] = []
    for p, short, horizon in [("brusselator", "brusselator", 20.), ("van_der_pol", "vdp", 10.)]:
        for mode, folder in [("our_cpu_reference", "our_"+short+"_full"), ("native_flowstar", "native_"+short)]:
            relative = "raw_minimal/" + folder + "/summary.json"
            raw = read(FROZEN / relative)
            tables["horizon_matrix.csv"].append(dict(plant=p, mode=mode, requested_horizon=horizon,
                accepted_horizon=raw["accepted_horizon"], accepted_steps=raw["accepted_steps"], status="REUSED_FROZEN", source=relative, reason="Frozen numerical observation; no new correctness guarantee."))
            tables["timings_raw.csv"].append(dict(plant=p, mode=mode, B=1, device="cpu", status="REUSED_FROZEN",
                solve_seconds=raw["solve_seconds"], export_seconds=raw["export_seconds"], repetitions=1,
                source=relative, reason="Previous single measurement, not a fresh paired performance result."))
        tables["horizon_matrix.csv"].append(dict(plant=p, mode="repaired_strict", requested_horizon=horizon,
            accepted_horizon="", accepted_steps="", status="NOT_STARTED", source="", reason=REASON))
        for b in [1, 8, 32]:
            tables["timings_raw.csv"].append(dict(plant=p, mode="repaired_strict", B=b, device="cuda", status="NOT_STARTED",
                solve_seconds="", export_seconds="", repetitions=0, source="", reason=REASON))
    tables["timing_summary.csv"] = [dict(plant=p, status="NOT_STARTED", flowstar_time_ratio="", our_throughput_gain="", reason=REASON)
                                    for p in ["brusselator", "van_der_pol"]]
    result = dict(status=decision, upstream_sha=SHAS["A"], candidate_patch_sha=None,
                  optional_backend_added=False, candidate_numerical_edits=False,
                  known_maps_contained_in_strict=checks, third_independent_obstacle=third,
                  repaired_candidate_horizons="NOT_STARTED", width_gate="NOT_ESTABLISHED", speed_gate="NOT_ESTABLISHED",
                  scope="Local maps, not observed long-trajectory counterexamples; old strict source is not globally certified.")
    return tables, result


def verify(root, check_hashes=True):
    if check_hashes:
        require((root / "SHA256SUMS").read_text() == checksum(root), "outer checksums")
    source = read(root / "SOURCE_MAP.json")
    for relative, digest in source["frozen_reused_files"].items():
        require(hashlib.sha256((FROZEN / relative).read_bytes()).hexdigest() == digest, "frozen evidence changed")
    require((root / "MATCHED_CONTRACTS.json").read_bytes() == (FROZEN / "MATCHED_CONTRACTS.json").read_bytes(), "matched contracts changed")
    tables, result = derived(root)
    with tempfile.TemporaryDirectory() as td:
        for name, rows in tables.items():
            path = Path(td) / name; write_csv(path, rows)
            require(path.read_bytes() == (root / name).read_bytes(), "recomputed table differs: " + name)
    require(read(root / "RESULT.json") == result, "decision disagrees with raw data")
    scope = read(root / "REPAIR_SCOPE.json")
    require(scope["candidate_worktree_created"] is False and not scope["candidate_numerical_edits"]
            and not scope["our_numerical_edits"], "numerical mutation after stop")
    require(scope["allowed_chains"] == ["history_matrix_and_error_propagation", "normalization_scaling_and_reconstruction"], "repair scope changed")
    context = read(root / "RUN_CONTEXT.json")
    require(context["status"] == result["status"] and context["candidate_patch_sha"] is None
            and context["upstream_sha"] == SHAS["A"] and context["driver_shas"] == source["drivers"], "run context contradicts evidence")
    require(read(root / "LONG_PREFIX_STATE_CHECKS.json")["status"] == "NOT_STARTED", "unrun prefix relabelled")
    require(read(root / "LOCAL_CONTAINMENT_RESULT.json")["complete_error_ownership_closed"] is False, "local tests promoted to global closure")
    return dict(status="VERIFIED", decision=result["status"], exact_map_rows=len(tables["WITNESS_CROSSCHECK.csv"]),
                independent_endpoint_rows=8, recomputed_tables=len(tables),
                scope="Static exact evidence verification; source reruns require local/private third-party source access.")


def main():
    p = argparse.ArgumentParser(); p.add_argument("root", type=Path)
    p.add_argument("--write-tables", action="store_true"); p.add_argument("--write-hashes", action="store_true")
    args = p.parse_args()
    if args.write_tables:
        tables, result = derived(args.root)
        for name, rows in tables.items(): write_csv(args.root / name, rows)
        (args.root / "RESULT.json").write_text(json.dumps(result, indent=2)+"\n")
    if args.write_hashes:
        (args.root / "SHA256SUMS").write_text(checksum(args.root))
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
