import copy
import hashlib
import json
from pathlib import Path
from statistics import median

import pytest

from torch_tm_flowpipe.archcomp26_status_report import (
    _verify_bound_file,
    collect_status,
    render_markdown,
    summarize_matrix,
)


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_plot_receipt_file_binding_is_relative_and_exact(tmp_path):
    source = tmp_path / "generator.py"
    source.write_text("print('bound')\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    record = {"path": "generator.py", "sha256": digest}
    assert _verify_bound_file(
        tmp_path,
        record,
        path_key="path",
        sha_key="sha256",
        label="generator",
    ) == source.resolve()

    for bad_path in (str(source.resolve()), "../generator.py"):
        with pytest.raises(ValueError, match="repository-relative|escapes"):
            _verify_bound_file(
                tmp_path,
                {**record, "path": bad_path},
                path_key="path",
                sha_key="sha256",
                label="generator",
            )
    with pytest.raises(ValueError, match="SHA-256"):
        _verify_bound_file(
            tmp_path,
            {**record, "sha256": "0" * 64},
            path_key="path",
            sha_key="sha256",
            label="generator",
        )


def test_current_status_report_is_deterministic_and_fail_closed():
    status = collect_status()
    assert len(status["rows"]) == 16
    assert sum(status["run_counts"].values()) == 64
    assert status["run_counts"] == {"not_started": 64}
    assert status["support_counts"] == {"unassessed": 64}
    assert all(
        row["instance"]["contract"]["status"] == "unresolved"
        for row in status["rows"]
    )
    assert len(status["audits"]) == 11

    huan_rows = load(
        "research/gpu_verified_20260930/report/evidence/huan_parity_campaign.json"
    )["rows"]
    assert status["huan"]["medians"]["process_wall_s"] == median(
        row["process_wall_s"] for row in huan_rows
    )
    assert status["p3"]["fullbatch_qualification"] is False
    assert status["p3"]["end_to_end_strict_certificate"] is False
    assert status["plot"]["experiments_started"] is False
    assert status["plot_contracts"]["counts"] == {
        "instances": 16,
        "materialized_v2_content_contract": 8,
        "materialized_v3_content_contract": 1,
        "axis_aligned_content_ready_not_materialized": 0,
        "blocked_fail_closed": 7,
    }

    report = render_markdown(status)
    assert report == render_markdown(collect_status())
    assert report.count("`not_started`") == 64
    assert "not the final experiment report" in report
    assert "Experiments paused: **yes**" in report
    assert "not a same-contract speedup ratio" in report
    assert "materialized v3=1" in report
    assert "fail-closed blocked=7" in report
    assert "Plot content status does not resolve" in report
    assert "/Users/" not in report
    assert "/srv/" not in report
    assert report == (ROOT / "docs/ARCHCOMP26_EXECUTION_STATUS.md").read_text(
        encoding="utf-8"
    )


def test_status_report_rejects_malformed_matrix():
    manifest = load("benchmarks/archcomp26/manifest.json")
    matrix = load("benchmarks/archcomp26/execution_matrix.json")

    missing_method = copy.deepcopy(matrix)
    missing_method["methods"].pop()
    with pytest.raises(ValueError, match="matrix methods"):
        summarize_matrix(manifest, missing_method)

    partial_override = copy.deepcopy(matrix)
    partial_override["cells"]["quad-reach"]["pytorch_gpu"] = {
        "run": {"status": "completed"}
    }
    with pytest.raises(ValueError, match="incomplete override"):
        summarize_matrix(manifest, partial_override)
