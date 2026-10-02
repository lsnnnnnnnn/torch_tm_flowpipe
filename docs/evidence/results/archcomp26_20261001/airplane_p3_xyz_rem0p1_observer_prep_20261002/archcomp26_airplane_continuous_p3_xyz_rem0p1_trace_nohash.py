#!/usr/bin/env python3
"""One Airplane P3 first-period trace with only x/y/z Picard guesses widened."""

from pathlib import Path

import archcomp26_airplane_continuous_p3_nohash as airplane
import archcomp26_airplane_continuous_p3_trace_nohash as trace


if __name__ == "__main__":
    airplane.AUTHOR = Path(__file__).resolve().parent / "run.py"
    raise SystemExit(trace.main())
