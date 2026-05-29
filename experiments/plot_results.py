from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(paths):
    rows = []
    for p in paths:
        with Path(p).open() as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", help="CSV files produced by the experiment scripts")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    rows = read_rows(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [f"{r['system']}\nh={r['h']} o={r['order']}" for r in rows]
    final_widths = [float(r["final_width"]) for r in rows]
    runtimes = [float(r["runtime_s"]) for r in rows]

    plt.figure(figsize=(max(8, len(rows) * 0.7), 4.8))
    plt.bar(range(len(rows)), final_widths)
    plt.xticks(range(len(rows)), labels, rotation=75, ha="right")
    plt.ylabel("final width")
    plt.title("Taylor-model flowpipe final widths")
    plt.tight_layout()
    plt.savefig(out_dir / "final_widths.png", dpi=180)
    plt.close()

    plt.figure(figsize=(max(8, len(rows) * 0.7), 4.8))
    plt.bar(range(len(rows)), runtimes)
    plt.xticks(range(len(rows)), labels, rotation=75, ha="right")
    plt.ylabel("runtime (s)")
    plt.title("Taylor-model flowpipe runtime")
    plt.tight_layout()
    plt.savefig(out_dir / "runtime.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
