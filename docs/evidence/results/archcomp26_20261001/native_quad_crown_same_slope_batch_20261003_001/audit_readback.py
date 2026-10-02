#!/usr/bin/env python3
"""Local readback of the copied controller-only raw arrays; no digest checks."""

import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "native_quad_allbox_firststep_recenter_gate_20261003_003/run/rpc.json"


def main():
    old = json.loads(OLD.read_text())["coefficients"]
    new = json.loads((HERE / "RAW_COEFFICIENTS.json").read_text())
    pairs = (("lA", "T", 12), ("lbias", "u_min", 1),
             ("ubias", "u_max", 1), ("uA", "lA", 12))
    counts = {}
    for left, right, width in pairs:
        a = new[left]
        b = new[right] if left == "uA" else old[right]
        assert len(a) == len(b) == 1024
        n = 0
        for lane in range(1024):
            assert len(a[lane]) == len(b[lane]) == 3
            for output in range(3):
                x = a[lane][output]
                y = b[lane][output]
                if width == 1:
                    x, y = [x], [y]
                assert len(x) == len(y) == width
                for v, w in zip(x, y):
                    assert math.isfinite(v) and math.isfinite(w)
                    n += v != w
        counts[f"{left}_vs_{right}_mismatches"] = n
    result = {
        "scope": "local readback of copied new four-array receipt versus saved first-call RPC",
        "arrays": {"lA": [1024, 3, 12], "uA": [1024, 3, 12],
                   "lbias": [1024, 3], "ubias": [1024, 3]},
        "all_values_finite": True,
        "exact_numeric_mismatch_counts": counts,
        "no_digest_verification": True,
        "limits": "Equality of recorded numbers is not a proof of CROWN numerical soundness or native float32 transfer.",
    }
    (HERE / "AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
