"""Run one generated Flow* toolbox C++ case and leave stdout/stderr/artifacts."""
from __future__ import annotations

import argparse

from run_flowstar import run_flowstar_toolbox


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and execute one generated Flow* C++ case.")
    parser.add_argument("cpp", help="generated .cpp file")
    parser.add_argument("--flowstar-root", default=None, help="path to chenxin415/flowstar root or set FLOWSTAR_ROOT")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--no-build-flowstar-lib", action="store_true")
    parser.add_argument("--compiler", default=None)
    args = parser.parse_args()
    result = run_flowstar_toolbox(
        args.cpp,
        flowstar_root=args.flowstar_root,
        output_dir=args.output_dir,
        timeout_s=args.timeout_s,
        build_lib=not args.no_build_flowstar_lib,
        compiler=args.compiler,
        run_executable=True,
    )
    print(result)
    if result.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
