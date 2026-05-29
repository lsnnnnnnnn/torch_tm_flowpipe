from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(rel: str, argv: list[str] | None = None) -> None:
    path = ROOT / rel
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(path)] + (argv or [])
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = old_argv


def main() -> None:
    print("[runtime] examples/scalar_quadratic.py")
    run_script("examples/scalar_quadratic.py")
    print("[runtime] examples/van_der_pol_short.py")
    run_script("examples/van_der_pol_short.py")
    print("[runtime] examples/affine_controlled.py")
    run_script("examples/affine_controlled.py")

    print("[runtime] experiments/scalar_quadratic_grid.py")
    run_script("experiments/scalar_quadratic_grid.py", ["--csv", "outputs/scalar_quadratic_grid.csv"])
    print("[runtime] experiments/harmonic_oscillator.py")
    run_script("experiments/harmonic_oscillator.py", ["--csv", "outputs/harmonic_oscillator.csv"])
    print("[runtime] experiments/van_der_pol_sampling.py")
    run_script("experiments/van_der_pol_sampling.py", ["--csv", "outputs/van_der_pol_sampling.csv"])


if __name__ == "__main__":
    main()
