from __future__ import annotations

import argparse
import json
import sys

from experiments.run_evidence_command import run


def test_evidence_runner_emits_complete_portable_protocol(tmp_path) -> None:
    output = tmp_path / "runner"
    args = argparse.Namespace(
        output_dir=output,
        name="unit",
        source_commit="a" * 40,
        config_json='{"decimal":"0.01","hex":"0x1.47ae147ae147bp-7"}',
        cwd=tmp_path,
        eligibility_status="diagnostic_only",
        timing_eligibility="not_for_cross_tool_ratio",
        expected_exit_codes=(0,),
        command=[
            sys.executable,
            "-c",
            "from pathlib import Path; Path(r'{ARTIFACT_DIR}/value.txt').write_text('ok\\n')",
        ],
    )
    assert run(args) == 0
    required = {
        "config.json",
        "summary.json",
        "stdout.log",
        "stderr.log",
        "command.txt",
        "exit_code.txt",
        "timing.json",
        "artifact_index.json",
    }
    assert required <= {path.name for path in output.iterdir()}
    index = json.loads((output / "artifact_index.json").read_text())
    paths = {row["path"] for row in index["files"]}
    assert "artifacts/value.txt" in paths
    assert all(not path.startswith("/") for path in paths)
    assert json.loads((output / "summary.json").read_text())["status"] == "pass"
