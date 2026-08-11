from __future__ import annotations

from experiments.run_stock_flowstar_vdp_reproduction import _plot_horizon


def test_flowstar_plot_horizon_uses_coordinate_rows_only(tmp_path) -> None:
    plot = tmp_path / "x.plt"
    plot.write_text(
        "set terminal postscript\n"
        "0.0 1.0\n0.5 1.2\n\n"
        "0.5 1.1\n1.0 1.4\n",
        encoding="utf-8",
    )
    assert _plot_horizon(plot) == 1.0
