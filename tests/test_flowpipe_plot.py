import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import torch

from torch_tm_flowpipe.flowpipe_plot import (
    SCHEMA,
    export_geometry,
    main,
    render_matplotlib,
    validate_geometry,
    write_matlab,
)


def observer(path: Path, step: int, *, accepted=(True, True), state_count=3) -> None:
    bounds = torch.zeros((len(accepted), state_count, 4), dtype=torch.float64)
    for lane in range(len(accepted)):
        for state in range(state_count):
            center = 10 * step + 2 * lane + state
            bounds[lane, state] = torch.tensor(
                [center - 1, center + 1, center - .25, center + .25],
                dtype=torch.float64,
            )
    torch.save(
        {
            "bounds": bounds,
            "accepted": torch.tensor(accepted, dtype=torch.bool),
            "status": torch.zeros(len(accepted), dtype=torch.int8),
        },
        path / f"observer_{step}.pt",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sidecar(path: Path, step: int, identity=None) -> None:
    pt = path / f"observer_{step}.pt"
    value = {"step": step, "pt_sha256": sha256(pt)}
    if identity is not None:
        value["source_identity"] = identity
    (path / f"observer_{step}.json").write_text(json.dumps(value), encoding="utf-8")


def run_evidence(path: Path, identity: dict, *, budget=2, status="completed") -> None:
    input_value = {
        "budget_steps": budget,
        "source_identity": identity,
    }
    input_path = path / "INPUT.json"
    input_path.write_text(json.dumps(input_value), encoding="utf-8")
    (path / "RESULT.json").write_text(json.dumps({
        "input_sha256": sha256(input_path),
        "status": status,
        "completed_step": budget,
        "budget_steps": budget,
        "source_identity": identity,
        "fullbatch_qualification": False,
    }), encoding="utf-8")


def plot_identity(
    benchmark: str,
    instance_id: str,
    coordinate_names: list[str],
    step_size: float,
    expected_steps: int,
) -> dict:
    return {
        "benchmark": benchmark,
        "instance_id": instance_id,
        "coordinate_names": coordinate_names,
        "step_size": step_size,
        "expected_steps": expected_steps,
    }


def native_ranges(
    path: Path,
    *,
    state_count: int = 12,
    records=((0, 1), (0, 3), (1, 1), (1, 3)),
) -> None:
    record = struct.Struct(f"<QQd{4 * state_count}d")
    with path.open("wb") as handle:
        for lane, step in records:
            values = []
            for state in range(state_count):
                center = 10 * step + 2 * lane + state
                values.extend([center - 1, center + 1, center - .25, center + .25])
            handle.write(record.pack(lane, step, .5, *values))


def legacy_spec(*, timed_state_region=False):
    region = {
        "label": "Safe region",
        "role": "safe",
        "bounds": {"x1": [-2.0, 2.0], "x2": [-3.0, 3.0]},
        "time": {"kind": "endpoint", "at": .5} if timed_state_region else {"kind": "all"},
    }
    return {
        "schema": "torch-tm-flowpipe-plot-spec-v1",
        "benchmark": "demo",
        "contract_status": "legacy test contract",
        "identity_binding": "informational_legacy_unbound",
        "coordinate_names": ["x1", "x2", "x3"],
        "units": {"t": "s", "x1": "m"},
        "initial_set": {
            "label": "Initial set",
            "bounds": {"x1": [-1.0, 1.0], "x2": [-2.0, 2.0]},
        },
        "regions": [region],
    }


def official_spec():
    return {
        "schema": "torch-tm-flowpipe-plot-spec-v2",
        "benchmark": "demo",
        "instance_id": "demo-continuous",
        "contract_status": "official content frozen; execution contract unresolved",
        "identity_binding": "official_contract_sources_hash_declared",
        "run_binding": "series_source_identity_plot_contract_required",
        "warning": "Instance-bound only; do not claim a matched official solver run.",
        "model_domain": "continuous_time",
        "source_refs": [
            {"path": "benchmarks/demo/Specifications.txt", "sha256": "a" * 64},
        ],
        "horizon": {"kind": "continuous_time", "start": 0.0, "end": 1.0},
        "property_quantifier": "endpoint",
        "coordinate_names": ["x1", "x2", "x3"],
        "units": {"t": "s", "x1": "m"},
        "initial_set": {
            "label": "Official initial set",
            "bounds": {
                "x1": [-1.0, 1.0],
                "x2": [-2.0, 2.0],
                "x3": [0.0, 0.0],
            },
        },
        "regions": [{
            "label": "Official endpoint target",
            "role": "target",
            "bounds": {"x1": [-0.5, 0.5]},
            "time": {"kind": "endpoint", "at": 1.0},
        }],
    }


def source_assets(value) -> set[tuple[str, str]]:
    assets: set[tuple[str, str]] = set()
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            assets.add((value["path"], value["sha256"]))
        for child in value.values():
            assets.update(source_assets(child))
    elif isinstance(value, list):
        for child in value:
            assets.update(source_assets(child))
    return assets


class FlowpipePlotTests(unittest.TestCase):
    def test_all_materialized_archcomp_v2_specs_are_audited_and_valid(self):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (repo_root / "benchmarks/archcomp26/manifest.json").read_text(encoding="utf-8")
        )
        audits = json.loads(
            (repo_root / "benchmarks/archcomp26/evidence/contract_audits_20261001.json")
            .read_text(encoding="utf-8")
        )
        status = json.loads(
            (repo_root / "benchmarks/plot_specs/archcomp26_status.json")
            .read_text(encoding="utf-8")
        )
        manifest_by_id = {row["id"]: row for row in manifest["instances"]}
        report_asset = (
            "reference/ARCH_COMP26_AINNCS.pdf",
            manifest["official_sources"]["report"]["pdf_sha256"],
        )
        materialized = [row for row in status["instances"] if row["plot_spec"]]
        self.assertEqual(len(materialized), 8)
        for row in materialized:
            spec = json.loads((repo_root / row["plot_spec"]).read_text(encoding="utf-8"))
            manifest_row = manifest_by_id[row["instance_id"]]
            self.assertEqual(spec["instance_id"], manifest_row["id"])
            self.assertEqual(spec["benchmark"], manifest_row["benchmark"])
            audited_assets = source_assets(
                audits["audits"][manifest_row["contract_audit"]]["source_files"]
            ) | {report_asset}
            self.assertTrue(
                {(item["path"], item["sha256"]) for item in spec["source_refs"]}
                <= audited_assets
            )

            coordinates = spec["coordinate_names"]
            horizon = float(spec["horizon"]["end"])
            projection_coordinate = next(iter(spec["regions"][0]["bounds"]))
            with tempfile.TemporaryDirectory() as scratch:
                root = Path(scratch)
                observer(root, 1, state_count=len(coordinates))
                sidecar(root, 1, plot_identity(
                    spec["benchmark"], spec["instance_id"],
                    coordinates, horizon, 1,
                ))
                geometry = export_geometry(
                    [("instance-bound test series", root)],
                    benchmark=spec["benchmark"],
                    instance_id=spec["instance_id"],
                    coordinate_names=coordinates,
                    projection_text=f"t,{projection_coordinate}",
                    view="tube",
                    step_size=horizon,
                    expected_steps=1,
                    spec=spec,
                )
                self.assertEqual(
                    geometry["spec_binding"]["status"],
                    "official_content_series_plot_contract_binding_verified",
                )

    def test_v2_spec_requires_series_instance_binding_without_claiming_full_contract(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            observer(root, 1)
            arguments = dict(
                series=[("candidate", root)],
                benchmark="demo",
                instance_id="demo-continuous",
                coordinate_names=["x1", "x2", "x3"],
                projection_text="t,x1",
                view="tube",
                step_size=1.0,
                expected_steps=1,
                spec=official_spec(),
            )
            with self.assertRaisesRegex(ValueError, "verified across all observer sidecars"):
                export_geometry(**arguments)
            sidecar(root, 1, plot_identity(
                "demo", "demo-continuous", ["x1", "x2", "x3"], 1.0, 1
            ))
            geometry = export_geometry(**arguments)
            self.assertEqual(geometry["instance_id"], "demo-continuous")
            self.assertEqual(
                geometry["spec_binding"]["status"],
                "official_content_series_plot_contract_binding_verified",
            )
            self.assertEqual(geometry["spec_binding"]["source_ref_count"], 1)
            script = root / "official.m"
            write_matlab(geometry, script)
            script_text = script.read_text(encoding="utf-8")
            self.assertIn("official source hashes declared", script_text)
            self.assertIn("series benchmark, instance, coordinates", script_text)
            self.assertIn("do not claim a matched official solver run", script_text)
            with self.assertRaisesRegex(ValueError, "explicit --instance-id"):
                export_geometry(**{**arguments, "instance_id": None})
            with self.assertRaisesRegex(ValueError, "instance_id does not match"):
                export_geometry(**{**arguments, "instance_id": "other"})

            bad = official_spec()
            bad["source_refs"][0]["sha256"] = "A" * 64
            with self.assertRaisesRegex(ValueError, "lowercase hexadecimal"):
                export_geometry(**{**arguments, "spec": bad})
            bad = official_spec()
            bad["source_refs"][0]["path"] = "/absolute/source"
            with self.assertRaisesRegex(ValueError, "safe relative path"):
                export_geometry(**{**arguments, "spec": bad})
            bad = official_spec()
            bad["horizon"]["end"] = 2.0
            with self.assertRaisesRegex(ValueError, "numerical horizon"):
                export_geometry(**{**arguments, "spec": bad})
            bad = official_spec()
            del bad["initial_set"]["bounds"]["x3"]
            with self.assertRaisesRegex(ValueError, "bound every coordinate"):
                export_geometry(**{**arguments, "spec": bad})
            bad = official_spec()
            bad["regions"][0]["bounds"] = {}
            with self.assertRaisesRegex(ValueError, "property-region bounds"):
                export_geometry(**{**arguments, "spec": bad})
            bad = official_spec()
            bad["regions"][0]["time"] = {"kind": "all"}
            with self.assertRaisesRegex(ValueError, "horizon endpoint"):
                export_geometry(**{**arguments, "spec": bad})
            bad = official_spec()
            bad["model_domain"] = "discrete_time"
            with self.assertRaisesRegex(ValueError, "only continuous_time"):
                export_geometry(**{**arguments, "spec": bad})

            bad_identity = plot_identity(
                "demo", "other", ["x1", "x2", "x3"], 1.0, 1
            )
            sidecar(root, 1, bad_identity)
            with self.assertRaisesRegex(ValueError, "source_identity instance_id"):
                export_geometry(**arguments)
            sidecar(root, 1, plot_identity(
                "demo", "demo-continuous", ["x1", "x2", "x3"], 1.0, 1
            ))

            tampered = json.loads(json.dumps(geometry))
            tampered["instance_id"] = "other"
            with self.assertRaisesRegex(ValueError, "instance_id does not match"):
                validate_geometry(tampered)

    def test_time_tube_uses_tube_columns_and_separates_projection_gaps(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            observer(root, 1)
            observer(root, 3)
            geometry = export_geometry(
                [("candidate", root)],
                benchmark="demo",
                coordinate_names=["x1", "x2", "x3"],
                projection_text="t,x2",
                view="tube",
                step_size=.5,
                expected_steps=3,
            )
            series = geometry["series"][0]
            self.assertEqual(geometry["schema"], SCHEMA)
            self.assertEqual(geometry["interpolation"], "none")
            self.assertEqual(series["projection_unobserved_step_count"], 1)
            self.assertEqual(series["projection_unobserved_step_ranges"], [[2, 2]])
            self.assertEqual(series["display_selection"], {
                "policy": "all-observed", "requested_steps": None
            })
            self.assertEqual(series["frames"][0]["boxes"], [[0.0, .5, 10.0, 14.0]])
            self.assertEqual(series["frames"][1]["boxes"], [[1.0, 1.5, 30.0, 34.0]])

    def test_state_endpoint_keeps_lane_ids_and_one_box_per_lane(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            observer(root, 2)
            geometry = export_geometry(
                [("candidate", root)],
                benchmark="demo",
                coordinate_names=["x1", "x2", "x3"],
                projection_text="x1,x3",
                view="endpoint",
                step_size=.5,
                expected_steps=2,
            )
            frame = geometry["series"][0]["frames"][0]
            self.assertEqual(frame["lane_ids"], [0, 1])
            self.assertEqual(frame["boxes"], [
                [19.75, 20.25, 21.75, 22.25],
                [21.75, 22.25, 23.75, 24.25],
            ])

    def test_time_endpoint_is_at_exact_step_end(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            observer(root, 2)
            geometry = export_geometry(
                [("candidate", root)],
                benchmark="demo",
                coordinate_names=["x1", "x2", "x3"],
                projection_text="t,x1",
                view="endpoint",
                step_size=.5,
                expected_steps=2,
            )
            self.assertEqual(geometry["series"][0]["frames"][0]["boxes"], [
                [1.0, 1.0, 19.75, 22.25]
            ])

    def test_partial_policy_checks_omitted_observers_and_marks_lane_ids(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            observer(root, 1, accepted=(True, False))
            observer(root, 2)
            arguments = dict(
                series=[("candidate", root)],
                benchmark="demo",
                coordinate_names=["x1", "x2", "x3"],
                projection_text="x1,x2",
                view="endpoint",
                step_size=.5,
                expected_steps=2,
                display_steps={2},
            )
            with self.assertRaisesRegex(ValueError, "only 1/2 lanes accepted"):
                export_geometry(**arguments)
            geometry = export_geometry(**arguments, partial_policy="mark")
            series = geometry["series"][0]
            self.assertEqual(series["projection_partial_step_ranges"], [[1, 1]])
            self.assertEqual(series["displayed_partial_step_ranges"], [])
            geometry = export_geometry(
                **{**arguments, "display_steps": {1}}, partial_policy="mark"
            )
            frame = geometry["series"][0]["frames"][0]
            self.assertFalse(frame["complete"])
            self.assertEqual(frame["lane_ids"], [0])

    def test_sparse_8_of_1000_coverage_is_not_a_solver_failure_claim(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            saved = [20, 100, 580, 589, 590, 799, 800, 1000]
            for step in saved:
                observer(root, step)
            (root / "RESULT.json").write_text(json.dumps({
                "status": "completed", "completed_step": 1000,
                "overall_horizon_completed": True,
            }), encoding="utf-8")
            (root / "steps.jsonl").write_text(
                "".join(json.dumps({"step": step}) + "\n" for step in range(1, 1001)),
                encoding="utf-8",
            )
            geometry = export_geometry(
                [("P3", root)],
                benchmark="demo",
                coordinate_names=["x1", "x2", "x3"],
                projection_text="t,x3",
                view="endpoint",
                step_size=.005,
                expected_steps=1000,
                display_steps={100, 1000},
            )
            series = geometry["series"][0]
            self.assertEqual(series["projection_unobserved_step_count"], 992)
            self.assertEqual(series["display_omitted_observed_step_count"], 6)
            self.assertEqual(series["displayed_steps"], [100, 1000])
            self.assertIn("does not infer solver failure", series["projection_unobserved_reason"])
            self.assertEqual(series["solver_run_evidence"]["steps.jsonl"]["row_count"], 1000)

    def test_all_sidecars_are_hash_bound_even_when_not_displayed(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            observer(root, 1)
            observer(root, 2)
            identity = {"model_sha256": "a" * 64}
            sidecar(root, 1, identity)
            sidecar(root, 2, identity)
            arguments = dict(
                series=[("candidate", root)],
                benchmark="demo",
                coordinate_names=["x1", "x2", "x3"],
                projection_text="t,x1",
                view="tube",
                step_size=.5,
                expected_steps=2,
                display_steps={2},
            )
            geometry = export_geometry(**arguments)
            self.assertEqual(
                geometry["series"][0]["source_identity"]["status"],
                "verified_equal_across_all_observer_sidecars",
            )
            files = geometry["series"][0]["source_identity"]["files"]
            self.assertEqual([item["step"] for item in files], [1, 2])
            self.assertTrue(all(len(item["source_sha256"]) == 64 for item in files))
            self.assertTrue(all(item["sidecar"]["pt_sha256_verified"] for item in files))
            empty = json.loads((root / "observer_1.json").read_text(encoding="utf-8"))
            empty["source_identity"] = {}
            (root / "observer_1.json").write_text(json.dumps(empty), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                export_geometry(**arguments)
            sidecar(root, 1, identity)
            bad = json.loads((root / "observer_1.json").read_text(encoding="utf-8"))
            bad["pt_sha256"] = "0" * 64
            (root / "observer_1.json").write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pt_sha256 does not match"):
                export_geometry(**arguments)

    def test_native_variable_state_records_decode_without_step_contiguity(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "ranges.bin"
            native_ranges(path, state_count=4)
            geometry = export_geometry(
                [("native B2", path)],
                benchmark="demo",
                coordinate_names=["x1", "x2", "x3", "x4"],
                projection_text="t,x2",
                view="endpoint",
                step_size=.5,
                expected_steps=3,
                expected_lanes=2,
            )
            series = geometry["series"][0]
            self.assertEqual(series["record_size"], 152)
            self.assertEqual(series["projection_unobserved_step_ranges"], [[2, 2]])
            self.assertEqual(series["frames"][0]["boxes"], [[.5, .5, 10.75, 13.25]])
            self.assertEqual(series["frames"][0]["source_record_index_min"], 0)
            self.assertEqual(series["frames"][0]["source_record_index_max"], 2)
            self.assertIsNone(series["frames"][0]["accepted_lanes"])

    def test_native_partial_and_truncation_fail_closed(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "ranges.bin"
            native_ranges(path, state_count=4, records=((0, 1),))
            arguments = dict(
                series=[("native", path)],
                benchmark="demo",
                coordinate_names=["x1", "x2", "x3", "x4"],
                projection_text="t,x2",
                view="tube",
                step_size=.5,
                expected_steps=1,
                expected_lanes=2,
            )
            with self.assertRaisesRegex(ValueError, "1/2 recorded lanes"):
                export_geometry(**arguments)
            geometry = export_geometry(**arguments, partial_policy="mark")
            self.assertEqual(geometry["series"][0]["projection_partial_step_ranges"], [[1, 1]])
            path.write_bytes(b"truncated")
            with self.assertRaisesRegex(ValueError, "positive multiple of 152"):
                export_geometry(**arguments)

    def test_observer_lane_count_must_not_shrink_to_survivors(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            observer(root, 1, accepted=(True, True))
            observer(root, 2, accepted=(True,))
            with self.assertRaisesRegex(ValueError, "lane count changes"):
                export_geometry(
                    [("candidate", root)],
                    benchmark="demo",
                    coordinate_names=["x1", "x2", "x3"],
                    projection_text="t,x1",
                    view="tube",
                    step_size=.5,
                    expected_steps=2,
                )

    def test_adjacent_contract_and_identity_are_checked(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            observer(root, 1)
            observer(root, 2)
            identity = {
                "model_sha256": "a" * 64,
                "settings": {"step": .5},
                "batch_size": 2,
                "execution_budget_steps": 2,
            }
            for step in (1, 2):
                sidecar(root, step, identity)
            run_evidence(root, identity)
            geometry = export_geometry(
                [("candidate", root)],
                benchmark="demo",
                coordinate_names=["x1", "x2", "x3"],
                projection_text="t,x1",
                view="tube",
                step_size=.5,
                expected_steps=2,
            )
            evidence = geometry["series"][0]["solver_run_evidence"]
            self.assertEqual(
                evidence["binding"]["series"],
                "verified_equal_to_observer_sidecars_direct",
            )
            self.assertEqual(
                evidence["plot_contract_check"]["status"],
                "matched_identity_bound_declarations",
            )
            with self.assertRaisesRegex(ValueError, "--step-size"):
                export_geometry(
                    [("candidate", root)],
                    benchmark="demo",
                    coordinate_names=["x1", "x2", "x3"],
                    projection_text="t,x1",
                    view="tube",
                    step_size=.25,
                    expected_steps=2,
                )

    def test_result_and_each_contract_field_keep_separate_bindings(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            for step in (1, 2):
                observer(root, step)
            identity = {
                "model_sha256": "a" * 64,
                "settings": {"step": .5},
                "execution_budget_steps": 2,
            }
            for step in (1, 2):
                sidecar(root, step, identity)
            (root / "INPUT.json").write_text(
                json.dumps({"batch_size": 2}), encoding="utf-8"
            )
            (root / "RESULT.json").write_text(json.dumps({
                "source_identity": identity,
                "budget_steps": 2,
                "status": "completed",
            }), encoding="utf-8")
            geometry = export_geometry(
                [("candidate", root)],
                benchmark="demo",
                coordinate_names=["x1", "x2", "x3"],
                projection_text="t,x1",
                view="tube",
                step_size=.5,
                expected_steps=2,
            )
            series = geometry["series"][0]
            contract = series["solver_run_evidence"]["plot_contract_check"]
            self.assertEqual(
                contract["status"],
                "identity_bound_run_with_unverified_or_unbound_plot_fields",
            )
            self.assertEqual(
                contract["declaration_binding"]["lane_count"],
                "adjacent_declaration_unbound_to_observer_series",
            )
            self.assertEqual(
                series["lane_universe_binding"],
                "matched_adjacent_declaration_unbound_to_observers",
            )

            input_value = {
                "budget_steps": 2,
                "source_identity": {**identity, "batch_size": 2},
            }
            for step in (1, 2):
                sidecar(root, step, input_value["source_identity"])
            (root / "INPUT.json").write_text(json.dumps(input_value), encoding="utf-8")
            (root / "RESULT.json").write_text(
                json.dumps({"status": "completed", "budget_steps": 2}),
                encoding="utf-8",
            )
            geometry = export_geometry(
                [("candidate", root)],
                benchmark="demo",
                coordinate_names=["x1", "x2", "x3"],
                projection_text="t,x1",
                view="tube",
                step_size=.5,
                expected_steps=2,
            )
            binding = geometry["series"][0]["solver_run_evidence"]["binding"]
            self.assertEqual(
                binding["result_series"],
                "RESULT_lacks_identity_or_hashed_INPUT_binding",
            )
            (root / "RESULT.json").write_text(
                json.dumps({"status": "completed", "budget_steps": 2,
                            "process_s": True}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "process_s must be"):
                export_geometry(
                    [("candidate", root)], benchmark="demo",
                    coordinate_names=["x1", "x2", "x3"], projection_text="t,x1",
                    view="tube", step_size=.5, expected_steps=2,
                )

    def test_steps_timing_has_its_own_hash_binding_and_strict_order(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            for step in (1, 2):
                observer(root, step)
            identity = {
                "model_sha256": "a" * 64,
                "settings": {"step": .5},
                "batch_size": 2,
                "execution_budget_steps": 2,
            }
            for step in (1, 2):
                sidecar(root, step, identity)
            run_evidence(root, identity)
            steps_path = root / "steps.jsonl"
            steps_path.write_text(
                '{"step":1,"advance_s":0.1}\n{"step":2,"advance_s":0.2}\n',
                encoding="utf-8",
            )
            result = json.loads((root / "RESULT.json").read_text(encoding="utf-8"))
            result["steps_sha256"] = sha256(steps_path)
            (root / "RESULT.json").write_text(json.dumps(result), encoding="utf-8")
            geometry = export_geometry(
                [("candidate", root)], benchmark="demo",
                coordinate_names=["x1", "x2", "x3"], projection_text="t,x1",
                view="tube", step_size=.5, expected_steps=2,
            )
            timing = geometry["series"][0]["solver_run_evidence"]["steps.jsonl"]
            self.assertEqual(timing["binding"],
                             "verified_to_series_via_identity_bound_RESULT_hash")
            self.assertAlmostEqual(timing["advance_s_sum"], .3)

            steps_path.write_text('{"step":1}\n{"step":1}\n', encoding="utf-8")
            result["steps_sha256"] = sha256(steps_path)
            (root / "RESULT.json").write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique and strictly increasing"):
                export_geometry(
                    [("candidate", root)], benchmark="demo",
                    coordinate_names=["x1", "x2", "x3"], projection_text="t,x1",
                    view="tube", step_size=.5, expected_steps=2,
                )

            steps_path.write_text('{"step":1}\n{"step":3}\n', encoding="utf-8")
            result["steps_sha256"] = sha256(steps_path)
            (root / "RESULT.json").write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "beyond --expected-steps"):
                export_geometry(
                    [("candidate", root)], benchmark="demo",
                    coordinate_names=["x1", "x2", "x3"], projection_text="t,x1",
                    view="tube", step_size=.5, expected_steps=2,
                )

            steps_path.write_text('{"step":1,"advance_s":true}\n', encoding="utf-8")
            result["steps_sha256"] = sha256(steps_path)
            (root / "RESULT.json").write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid advance_s"):
                export_geometry(
                    [("candidate", root)], benchmark="demo",
                    coordinate_names=["x1", "x2", "x3"], projection_text="t,x1",
                    view="tube", step_size=.5, expected_steps=2,
                )
            bad_identity = {**identity, "model_sha256": "b" * 64}
            run_evidence(root, bad_identity)
            with self.assertRaisesRegex(ValueError, "does not match observer sidecars"):
                export_geometry(
                    [("candidate", root)],
                    benchmark="demo",
                    coordinate_names=["x1", "x2", "x3"],
                    projection_text="t,x1",
                    view="tube",
                    step_size=.5,
                    expected_steps=2,
                )

    def test_state_state_render_guard_uses_actual_selected_boxes(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            observer(root, 1)
            with patch("torch_tm_flowpipe.flowpipe_plot.MAX_STATE_STATE_BOXES", 1):
                with self.assertRaisesRegex(ValueError, "exceed 1 boxes"):
                    export_geometry(
                        [("candidate", root)],
                        benchmark="demo",
                        coordinate_names=["x1", "x2", "x3"],
                        projection_text="x1,x2",
                        view="endpoint",
                        step_size=.5,
                        expected_steps=1000,
                    )

    def test_matlab_uses_patches_units_and_rejects_timed_state_region(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            observer(root, 1)
            geometry = export_geometry(
                [("candidate", root)],
                benchmark="demo",
                coordinate_names=["x1", "x2", "x3"],
                projection_text="x1,x2",
                view="endpoint",
                step_size=.5,
                expected_steps=1,
                spec=legacy_spec(),
            )
            script = root / "plot.m"
            write_matlab(geometry, script)
            text = script.read_text(encoding="utf-8")
            self.assertIn("h_initial = patch", text)
            self.assertIn("h_region_1 = patch", text)
            self.assertIn("xlabel('x1 [m]')", text)
            self.assertIn("Legacy informational plot spec; not identity-bound.", text)
            with self.assertRaisesRegex(ValueError, "time-scoped regions"):
                export_geometry(
                    [("candidate", root)],
                    benchmark="demo",
                    coordinate_names=["x1", "x2", "x3"],
                    projection_text="x1,x2",
                    view="endpoint",
                    step_size=.5,
                    expected_steps=1,
                    spec=legacy_spec(timed_state_region=True),
                )

    def test_footer_discloses_hidden_partial_unprojected_region_and_empty_legend(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            observer(root, 1, accepted=(True, False))
            observer(root, 2)
            spec = legacy_spec()
            spec["regions"][0]["bounds"] = {"x3": [-3.0, 3.0]}
            spec["regions"][0]["time"] = {"kind": "endpoint", "at": 1.0}
            geometry = export_geometry(
                [("candidate", root)],
                benchmark="demo",
                coordinate_names=["x1", "x2", "x3"],
                projection_text="t,x1",
                view="endpoint",
                step_size=.5,
                expected_steps=2,
                display_steps={2},
                partial_policy="mark",
                spec=spec,
            )
            script = root / "footer.m"
            write_matlab(geometry, script)
            text = script.read_text(encoding="utf-8")
            self.assertIn("projection partial 1; displayed partial 0", text)
            self.assertIn("Plot-spec regions not drawn in this projection", text)

            empty = Path(scratch) / "empty"
            empty.mkdir()
            observer(empty, 1, accepted=(False, False))
            empty_geometry = export_geometry(
                [("empty", empty)],
                benchmark="demo",
                coordinate_names=["x1", "x2", "x3"],
                projection_text="t,x1",
                view="endpoint",
                step_size=.5,
                expected_steps=1,
                partial_policy="mark",
            )
            empty_script = empty / "plot.m"
            write_matlab(empty_geometry, empty_script)
            self.assertNotIn("legend([", empty_script.read_text(encoding="utf-8"))

    def test_native_scope_and_acceptance_unknown_are_visible(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            ranges = root / "ranges.bin"
            native_ranges(ranges, state_count=4)
            input_value = {
                "requested_periods": 3,
                "scientific": {"h": .5, "period": .5, "leaf_count": 2},
            }
            input_path = root / "INPUT.json"
            input_path.write_text(json.dumps(input_value), encoding="utf-8")
            (root / "RESULT.json").write_text(json.dumps({
                "input_sha256": sha256(input_path),
                "status": "target_completed",
                "full_T5_full1024_completed": False,
                "full_T5_root1_B2_completed": True,
                "fullbatch_qualification": False,
                "returncode": 0,
            }), encoding="utf-8")
            geometry = export_geometry(
                [("native B2", ranges)],
                benchmark="demo",
                coordinate_names=["x1", "x2", "x3", "x4"],
                projection_text="t,x2",
                view="endpoint",
                step_size=.5,
                expected_steps=3,
                expected_lanes=2,
            )
            result = geometry["series"][0]["solver_run_evidence"]["RESULT.json"]
            self.assertFalse(result["full_T5_full1024_completed"])
            script = root / "native.m"
            write_matlab(geometry, script)
            text = script.read_text(encoding="utf-8")
            self.assertIn("record-complete; acceptance unknown", text)
            self.assertIn("full_T5_full1024_completed=false", text)

    def test_geometry_redraw_and_validator_fail_closed(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            observer(root, 1)
            geometry = export_geometry(
                [("candidate", root)],
                benchmark="demo",
                coordinate_names=["x1", "x2", "x3"],
                projection_text="t,x1",
                view="endpoint",
                step_size=.5,
                expected_steps=1,
            )
            geometry_path = root / "saved.json"
            geometry_path.write_text(json.dumps(geometry), encoding="utf-8")
            output = root / "redrawn"
            self.assertEqual(main([
                "--geometry", str(geometry_path), "--output", str(output), "--no-render"
            ]), 0)
            self.assertTrue(output.with_suffix(".m").is_file())
            png, pdf = render_matplotlib(geometry, root / "rendered")
            self.assertTrue(png.is_file())
            self.assertTrue(pdf.is_file())
            tampered = json.loads(json.dumps(geometry))
            tampered["series"][0]["frames"][0]["complete"] = "false"
            with self.assertRaisesRegex(ValueError, "complete must be boolean"):
                validate_geometry(tampered)
            tampered = json.loads(json.dumps(geometry))
            tampered["series"][0]["frames"][0]["time_end"] = .25
            with self.assertRaisesRegex(ValueError, "frame time interval"):
                validate_geometry(tampered)
            tampered = json.loads(json.dumps(geometry))
            tampered["series"][0]["frames"][0]["boxes"][0][0] = .25
            with self.assertRaisesRegex(ValueError, "box x bounds"):
                validate_geometry(tampered)
            tampered = json.loads(json.dumps(geometry))
            tampered["spec"] = None
            with self.assertRaisesRegex(ValueError, "geometry spec must be"):
                validate_geometry(tampered)
            tampered = json.loads(json.dumps(geometry))
            tampered["series"][0]["lane_universe_binding"] = "bad\nvalue"
            with self.assertRaisesRegex(ValueError, "MATLAB text"):
                write_matlab(tampered, root / "bad.m")

    def test_cli_requires_horizon_and_applies_display_selection(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            observer(root, 1)
            observer(root, 2)
            output = root / "cli"
            base = [
                "--series", f"candidate={root}",
                "--benchmark", "demo",
                "--state-count", "3",
                "--projection", "t,x1",
                "--output", str(output),
                "--no-render",
            ]
            with self.assertRaisesRegex(ValueError, "expected-steps is required"):
                main(base)
            with self.assertRaisesRegex(ValueError, "step-size is required"):
                main(base + ["--expected-steps", "2"])
            self.assertEqual(main(base + [
                "--expected-steps", "2", "--step-size", ".5",
                "--display-steps", "2"
            ]), 0)
            receipt = json.loads(output.with_suffix(".render.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema"], "torch-tm-flowpipe-render-receipt-v1")
            self.assertIsNotNone(receipt["timings_seconds"]["matlab_script_generation"])
            self.assertIsNone(receipt["timings_seconds"]["matplotlib_png_pdf_render"])
            geometry = json.loads(output.with_suffix(".geometry.json").read_text())
            self.assertNotIn("instance_id", geometry)
            self.assertEqual(geometry["series"][0]["displayed_steps"], [2])
            self.assertEqual(geometry["series"][0]["display_omitted_observed_step_count"], 1)

            for name, content in (("null.json", "null"), ("list.json", "[]")):
                spec_path = root / name
                spec_path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "explicit --spec JSON"):
                    main(base + [
                        "--expected-steps", "2", "--step-size", ".5",
                        "--spec", str(spec_path),
                    ])

            with self.assertRaisesRegex(ValueError, "saved geometry"):
                main([
                    "--geometry", str(output.with_suffix(".geometry.json")),
                    "--instance-id", "wrong-instance",
                    "--output", str(root / "redraw"),
                    "--no-render",
                ])

            for step in (1, 2):
                sidecar(root, step, plot_identity(
                    "demo", "demo-continuous", ["x1", "x2", "x3"], .5, 2
                ))
            v2_spec_path = root / "v2.json"
            v2_spec_path.write_text(json.dumps(official_spec()), encoding="utf-8")
            v2_output = root / "v2-cli"
            v2_args = [
                "--series", f"candidate={root}",
                "--benchmark", "demo",
                "--state-count", "3",
                "--projection", "t,x1",
                "--output", str(v2_output),
                "--no-render",
                "--expected-steps", "2",
                "--step-size", ".5",
                "--spec", str(v2_spec_path),
            ]
            with self.assertRaisesRegex(ValueError, "explicit --instance-id"):
                main(v2_args)
            self.assertEqual(
                main(v2_args + ["--instance-id", "demo-continuous"]), 0
            )
            v2_receipt = json.loads(
                v2_output.with_suffix(".render.json").read_text(encoding="utf-8")
            )
            self.assertEqual(v2_receipt["instance_id"], "demo-continuous")

    def test_matlab_text_fields_reject_control_characters(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            observer(root, 1)
            arguments = dict(
                series=[("candidate", root)], benchmark="demo",
                coordinate_names=["x1", "x2", "x3"], projection_text="t,x1",
                view="tube", step_size=.5, expected_steps=1,
            )
            with self.assertRaisesRegex(ValueError, "benchmark"):
                export_geometry(**{**arguments, "benchmark": "bad\nname"})
            with self.assertRaisesRegex(ValueError, "series label"):
                export_geometry(**{**arguments, "series": [("bad\nlabel", root)]})


if __name__ == "__main__":
    unittest.main()
