"""The supported route must reject numerical failure and an unverified engine."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.review_suite import cli


class WholeEngineEntryTests(unittest.TestCase):
    def test_incomplete_successful_process_keeps_evidence_and_fails(self):
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch) / "run"

            def rejected_process(command, **kwargs):
                output.mkdir()
                (output / "summary.json").write_text(json.dumps({
                    "completed": False, "failure": {"step": 2}, "accepted_steps": [1]}))
                return 0

            with patch.object(cli, "checked_engine_root", return_value=Path(scratch)), \
                    patch.object(cli, "execute", side_effect=rejected_process):
                with self.assertRaisesRegex(RuntimeError, "did not complete"):
                    cli.whole_engine_run("van_der_pol", "cpu", scratch, output, steps=2)
            self.assertEqual(json.loads((output / "summary.json").read_text())["accepted_steps"], [1])

    def test_reject_unverified_engine_and_dirty_accepted_engine(self):
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "src/flowstar_gpu"
            source.mkdir(parents=True)
            (source / "__init__.py").touch()
            for head, status in (("cff8758", ""), (cli.WHOLE_ENGINE_REVISION, " M changed.py")):
                with self.subTest(head=head, status=status), \
                        patch.object(cli.subprocess, "check_output", side_effect=[head, status]):
                    with self.assertRaisesRegex(ValueError, "requires clean"):
                        cli.checked_engine_root(scratch)

    def test_smoke_failure_does_not_delete_partial_evidence(self):
        with tempfile.TemporaryDirectory() as scratch:
            evidence = Path(scratch) / "accepted-prefix.json"
            evidence.write_text('{"accepted_steps": [1]}')
            with patch.object(cli.tempfile, "mkdtemp", return_value=scratch), \
                    patch.object(cli, "whole_engine_run", side_effect=RuntimeError("rejected")):
                with self.assertRaisesRegex(RuntimeError, "rejected"):
                    cli.whole_engine_smoke("cpu", scratch)
            self.assertTrue(evidence.is_file())

    def test_adaptive_is_not_silently_run_as_fixed(self):
        with tempfile.TemporaryDirectory() as scratch, \
                patch.object(cli, "profile_check"), \
                patch.object(cli.subprocess, "check_output", return_value=""), \
                patch.object(cli, "whole_engine_run") as run:
            with self.assertRaisesRegex(ValueError, "only the two fixed"):
                cli.run_experiment("vdp-adaptive", "whole-engine", Path(scratch) / "run",
                                   engine_root=scratch, device="cpu")
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
