from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "three_tool_reaudit"))

from flowstar_vdp_reproduction import parse_plot_blocks


def test_parse_plot_blocks_keeps_segment_and_horizon_semantics(tmp_path: Path) -> None:
    plot = tmp_path / "result.plt"
    plot.write_text(
        "set terminal postscript\nplot '-'\n"
        "0 1\n0.1 2\n\n"
        "0.1 1.5\n0.2 2.5\n\ne\n",
        encoding="utf-8",
    )
    blocks = parse_plot_blocks(plot)
    assert len(blocks) == 2
    assert blocks[-1] == {
        "t_lower": 0.1,
        "t_upper": 0.2,
        "value_lower": 1.5,
        "value_upper": 2.5,
    }
