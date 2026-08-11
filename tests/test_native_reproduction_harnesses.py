from __future__ import annotations

import sys

from experiments.collect_three_tool_environment import _invoked_and_resolved
from experiments.run_stock_diffreach_vdp_reproduction import _python_paths
from experiments.run_stock_flowstar_vdp_reproduction import (
    _plot_horizon,
    _plot_segment_count,
    _reported_core_seconds,
    parse_args,
)


def test_flowstar_plot_horizon_uses_coordinate_rows_only(tmp_path) -> None:
    plot = tmp_path / "x.plt"
    plot.write_text(
        "set terminal postscript\n"
        "0.0 1.0\n0.5 1.2\n\n"
        "0.5 1.1\n1.0 1.4\n",
        encoding="utf-8",
    )
    assert _plot_horizon(plot) == 1.0


def test_flowstar_plot_segment_count_uses_numeric_blocks(tmp_path) -> None:
    plot = tmp_path / "x.plt"
    plot.write_text(
        "set terminal postscript\nplot '-'\n"
        "0 1\n1 1\n1 2\n0 2\n0 1\n\n\n"
        "1 2\n2 2\n2 3\n1 3\n1 2\n\n\ne\n",
        encoding="utf-8",
    )
    assert _plot_segment_count(plot) == 2


def test_flowstar_compiler_compatibility_flag_is_explicit() -> None:
    args = parse_args(
        [
            "--source",
            "/tmp/source",
            "--output-dir",
            "/tmp/output",
            "--source-commit",
            "abc",
            "--model-sha256",
            "def",
            "--cxx",
            "g++",
            "--cxx-compatibility-flag=-fpermissive",
        ]
    )
    assert args.cxx == "g++"
    assert args.cxx_compatibility_flag == ["-fpermissive"]


def test_flowstar_reported_core_time_parses_stock_label() -> None:
    assert _reported_core_seconds("time cost: 0.504992\n") == 0.504992


def test_environment_probe_preserves_interpreter_invocation_symlink(tmp_path) -> None:
    invoked = tmp_path / "pinned-python"
    invoked.symlink_to(sys.executable)
    kept, resolved = _invoked_and_resolved(invoked)
    assert kept == invoked.absolute()
    assert kept != resolved
    assert resolved == invoked.resolve()


def test_diffreach_runner_preserves_interpreter_invocation_symlink(tmp_path) -> None:
    invoked = tmp_path / "diffreach-python"
    invoked.symlink_to(sys.executable)
    kept, resolved = _python_paths(invoked)
    assert kept == invoked.absolute()
    assert kept != resolved
    assert resolved == invoked.resolve()
