"""Small synthetic saved-range fixture for the no-digest overlay entry."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from torch_tm_flowpipe.flowpipe_plot_nohash import main


def build_fixture(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    spec = root / "synthetic_plot_spec.json"
    spec.write_text(json.dumps({
        "schema": "torch-tm-flowpipe-plot-spec-v1",
        "benchmark": "Synthetic overlay fixture",
        "instance_id": "synthetic-overlay",
        "contract_status": "synthetic parser and renderer fixture; no solver run",
        "identity_binding": "informational_legacy_unbound",
        "warning": "These intervals are synthetic and do not describe reachable states.",
        "horizon": {"kind": "continuous_time", "start": 0, "end": 0.25},
        "property_quantifier": "conjunction",
        "coordinate_names": ["x1", "x2"],
        "units": {"t": "s"},
        "initial_set": {"label": "Initial fixture box", "bounds": {
            "x1": [0, 1], "x2": [0, 1],
        }},
        "regions": [
            {"label": "Safe fixture band", "role": "safe",
             "bounds": {"x1": [-1, 2], "x2": [-1, 2]},
             "time": {"kind": "all"}},
            {"label": "Endpoint fixture target", "role": "target",
             "bounds": {"x1": [0.4, 0.8]},
             "time": {"kind": "endpoint", "at": 0.25}},
        ],
    }, indent=2), encoding="utf-8")
    record = struct.Struct("<QQd8d")
    source_specs = (
        ("A saved prefix", [(1, (0, .4, .1, .3, 0, .2, .05, .15))]),
        ("B saved horizon", [
            (1, (0, .5, .2, .4, 0, .3, .1, .2)),
            (2, (.2, .7, .4, .6, .1, .5, .2, .4)),
        ]),
    )
    inputs = []
    for index, (label, rows) in enumerate(source_specs, 1):
        directory = root / f"source_{index}"
        directory.mkdir(exist_ok=True)
        ranges = directory / "ranges.bin"
        with ranges.open("wb") as handle:
            for step, bounds in rows:
                handle.write(record.pack(0, step, .125, *bounds))
        prefix = directory / "t_x1_tube"
        assert main([
            "--ranges", str(ranges), "--output", str(prefix),
            "--label", label, "--benchmark", "Synthetic overlay fixture",
            "--instance-id", "synthetic-overlay",
            "--coordinate-names", "x1,x2", "--projection", "t,x1",
            "--view", "tube", "--step-size", "0.125",
            "--expected-steps", "2", "--expected-lanes", "1",
            "--spec", str(spec),
        ]) == 0
        inputs.append(prefix.with_suffix(".geometry.json"))
    overlay = root / "overlay_t_x1_tube"
    assert main([
        "--geometry", str(inputs[0]), "--geometry", str(inputs[1]),
        "--output", str(overlay),
    ]) == 0
    return inputs[0], inputs[1], overlay


class NoHashOverlayTest(unittest.TestCase):
    def test_overlay_and_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second, overlay = build_fixture(Path(temp))
            combined = json.loads(overlay.with_suffix(".geometry.json").read_text())
            receipt = json.loads(overlay.with_suffix(".render.json").read_text())
            self.assertEqual([item["label"] for item in combined["series"]],
                             ["A saved prefix", "B saved horizon"])
            self.assertEqual(combined["series"][0]["projection_unobserved_step_ranges"],
                             [[2, 2]])
            self.assertEqual(combined["series"][1]["projection_unobserved_step_ranges"],
                             [])
            self.assertEqual(receipt["mode"], "geometry-overlay")
            self.assertEqual(len(receipt["input_geometries"]), 2)
            self.assertTrue(all(value >= 0 for value in receipt["timings_seconds"].values()))
            self.assertFalse(overlay.with_suffix(".m").exists())
            self.assertNotIn("matlab", receipt["artifacts"])
            for suffix in (".geometry.json", ".png", ".pdf", ".render.json"):
                self.assertTrue(overlay.with_suffix(suffix).is_file())

            duplicate = Path(temp) / "duplicate.geometry.json"
            duplicate.write_text(first.read_text(), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate series label"):
                main(["--geometry", str(first), "--geometry", str(duplicate),
                      "--output", str(Path(temp) / "rejected_duplicate")])
            incompatible = Path(temp) / "incompatible.geometry.json"
            altered = json.loads(second.read_text())
            altered["spec"]["regions"][0]["label"] = "Different Safe fixture"
            incompatible.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "geometry spec differs"):
                main(["--geometry", str(first), "--geometry", str(incompatible),
                      "--output", str(Path(temp) / "rejected_contract")])
            with self.assertRaisesRegex(ValueError, "must not replace an input"):
                main(["--geometry", str(first), "--geometry", str(second),
                      "--output", str(first.with_suffix(""))])


if __name__ == "__main__":
    unittest.main()
