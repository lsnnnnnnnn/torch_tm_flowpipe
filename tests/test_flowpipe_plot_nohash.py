"""No-digest plotting checks; all native fixtures are saved data, never solver runs."""
import hashlib
from contextlib import ExitStack
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from torch_tm_flowpipe import flowpipe_plot as plot
from torch_tm_flowpipe.flowpipe_plot_nohash import main


ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_B2 = Path(
    "/Users/shengenli/Documents/ChatGPT/verification/results/"
    "quad_native_matched_20260928/evidence_long/native/order3_1000/ranges.bin"
)


def forbid_digest(*_args, **_kwargs):
    raise AssertionError("content digests must not be computed by the no-hash path")


class NoHashPlotTest(unittest.TestCase):
    def _run_guarded(self, argv, entry=main):
        with ExitStack() as guard:
            for name in (*hashlib.algorithms_guaranteed, "new", "file_digest"):
                guard.enter_context(patch.object(hashlib, name, forbid_digest))
            guard.enter_context(patch.object(plot, "_sha256", forbid_digest))
            self.assertEqual(entry(argv), 0)

    def test_other_cli_saved_redraws_are_python_only(self):
        from torch_tm_flowpipe import tm_octagon_nohash as octagon

        examples = [
            (plot.main, ROOT / "docs/evidence/results/flowpipe_plot_nohash_overlay_20261003_001/overlay_t_x1_tube.geometry.json"),
            (lambda _argv: octagon.main(), ROOT / "docs/evidence/results/archcomp26_20261001/tm_octagon_stream_harmonic_smoke_20261002_001/harmonic_stream.geometry.json"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            for index, (entry, source) in enumerate(examples):
                before = source.read_bytes()
                output = Path(temp) / f"redraw_{index}"
                argv = ["--geometry", str(source), "--output", str(output)]
                with patch.object(sys, "argv", ["redraw", *argv]):
                    self._run_guarded(argv, entry)
                self.assertEqual(source.read_bytes(), before)
                self.assertFalse(output.with_suffix(".m").exists())
                receipt = output.with_suffix(".render.json").read_text()
                self.assertNotIn('"matlab"', receipt)
                self.assertNotIn('"sha256"', receipt)
                for suffix in (".png", ".pdf", ".render.json"):
                    self.assertGreater(output.with_suffix(suffix).stat().st_size, 0)

    def test_dp_native_and_geometry_redraw_without_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            ranges = work / "ranges.bin"
            record = struct.Struct("<QQd16d")
            with ranges.open("wb") as handle:
                for step in (1, 2):
                    for lane in (0, 1):
                        bounds = []
                        for state in range(4):
                            lo = 1.0 + .01 * state + .001 * lane
                            bounds.extend((lo, lo + .1, lo + .02, lo + .08))
                        handle.write(record.pack(lane, step, .01, *bounds))
            run_config = work / "START.json"
            run_config.write_text(json.dumps({"method": "synthetic fixture"}), encoding="utf-8")
            (work / "RESULT.json").write_text(
                json.dumps({"status": "completed", "exit_code": 0}), encoding="utf-8"
            )
            output = work / "dp_t_theta1_tube"
            argv = [
                "--ranges", str(ranges), "--output", str(output),
                "--label", "synthetic native fixture", "--benchmark", "Double Pendulum",
                "--instance-id", "double-pendulum-less-robust",
                "--coordinate-names", "theta1,theta2,theta1_dot,theta2_dot",
                "--projection", "t,theta1", "--view", "tube",
                "--step-size", "0.01", "--expected-steps", "100",
                "--expected-lanes", "2", "--run-config", str(run_config),
                "--spec", str(ROOT / "benchmarks/plot_specs/double_pendulum_less_robust_2026_nohash.json"),
            ]
            self._run_guarded(argv)
            with self.assertRaisesRegex(ValueError, "horizon"):
                self._run_guarded([
                    "99" if value == "100" else value for value in argv
                ])
            geometry = json.loads(output.with_suffix(".geometry.json").read_text())
            receipt = json.loads(output.with_suffix(".render.json").read_text())
            self.assertEqual(plot._initial_box(geometry), [0.0, 0.0, 1.0, 1.3])
            self.assertEqual(plot._region_boxes(geometry)[0]["box"], [0.0, 1.0, -1.7, 2.0])
            self.assertEqual(geometry["spec"]["regions"][0]["time"], {"kind": "all"})
            self.assertEqual(geometry["series"][0]["projection_unobserved_step_count"], 98)
            self.assertIsNone(geometry["series"][0]["frames"][0]["accepted_lanes"])
            self.assertEqual(geometry["series"][0]["solver_run_evidence"]["run_configuration"]["content"],
                             {"method": "synthetic fixture"})
            self.assertEqual(geometry["series"][0]["solver_run_evidence"]["RESULT.json"]["status"],
                             "completed")
            self.assertIn("unbound", geometry["series"][0]["solver_run_evidence"]["RESULT_binding"])
            self.assertNotIn("sha256", json.dumps(receipt).lower())
            self.assertFalse(output.with_suffix(".m").exists())
            self.assertNotIn("matlab", receipt["artifacts"])
            for suffix in (".geometry.json", ".png", ".pdf", ".render.json"):
                self.assertTrue(output.with_suffix(suffix).is_file())
            self._run_guarded([
                "--geometry", str(output.with_suffix(".geometry.json")),
                "--output", str(work / "redraw"),
            ])
            self.assertFalse((work / "redraw.m").exists())

    @unittest.skipUnless(ARCHIVED_B2.is_file(), "archived QUAD B2 ranges.bin is unavailable")
    def test_archived_quad_b2_full_render_without_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "archived_quad_root1_b2_t_x3_tube"
            self._run_guarded([
                "--ranges", str(ARCHIVED_B2), "--output", str(output),
                "--label", "archived QUAD native root1 B2 (not 2026)",
                "--benchmark", "QUAD",
                "--coordinate-names", ",".join(f"x{i}" for i in range(1, 13)),
                "--projection", "t,x3", "--view", "tube",
                "--step-size", "0.005", "--expected-steps", "1000",
                "--expected-lanes", "2",
                "--spec", str(ROOT / "benchmarks/plot_specs/quad_legacy_author_contract.json"),
            ])
            geometry = json.loads(output.with_suffix(".geometry.json").read_text())
            self.assertEqual(len(geometry["series"][0]["frames"]), 1000)
            self.assertEqual(geometry["series"][0]["label"],
                             "archived QUAD native root1 B2 (not 2026)")
            self.assertEqual(geometry["spec"]["regions"][0]["time"],
                             {"kind": "endpoint", "at": 5.0})
            self.assertEqual(geometry["series"][0]["frames"][0]["accepted_lanes"], None)
            self.assertFalse(output.with_suffix(".m").exists())
            for suffix in (".png", ".pdf", ".render.json"):
                self.assertTrue(output.with_suffix(suffix).is_file())


if __name__ == "__main__":
    unittest.main()
