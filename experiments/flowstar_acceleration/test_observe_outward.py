"""Complete partitioned observations preserve horizon and lane accounting."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from .observe_outward import run


class CompleteObservation(unittest.TestCase):
    def fixture(self, root, *, written_steps=20):
        lanes = list(range(32))
        summary = dict(case={"lane_ids": lanes}, accepted_steps=[20]*32,
                       requested_steps=20, completed=True, failure=None,
                       settings={"step": .01}, plant="synthetic_observer_fixture")
        (root/"summary.json").write_text(json.dumps(summary))
        with gzip.open(root/"factored.jsonl.gz", "wt") as f:
            for step in range(1, written_steps+1):
                record = dict(step=step, lane_ids=lanes, accepted=[True]*32, h_hex=.01.hex(),
                    pre_exponents=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
                    tmv_exponents=[[0,1,0],[0,0,1]],
                    pre=[[[1., .01, .1, 0.],[2., -.02, 0., .2]]]*32,
                    pre_rem=[[[0.,0.],[0.,0.]]]*32,
                    tmv=[[[1.,0.],[0.,1.]]]*32, tmv_rem=[[[0.,0.],[0.,0.]]]*32)
                f.write(json.dumps(record)+"\n")
        return root/"factored.jsonl.gz"

    def test_complete_partitioned_twenty_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = run(self.fixture(root), root/"observed", "outward_binary64", "outward_binary64")
            self.assertTrue(report["complete_observation"])
            self.assertEqual(report["observed_steps"], [20]*32)
            self.assertEqual(report["bound_rows"], 2560)

    def test_missing_last_step_is_failed_not_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "incomplete observation"):
                run(self.fixture(root, written_steps=19), root/"observed", "outward_binary64", "outward_binary64")
            report = json.loads((root/"observed/summary.json").read_text())
            self.assertFalse(report["complete_observation"])
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["observed_steps"], [19]*32)
