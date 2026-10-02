"""Generate a one-box/one-small-step diagnostic from the frozen paper QUAD source.

This generator does not edit its input.  The output is an isolated diagnostic,
never a benchmark result or a replacement for the full native source.
"""

from pathlib import Path
import sys


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected exactly one source anchor: {old[:72]!r}")
    return source.replace(old, new, 1)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: generator frozen.cpp isolated.cpp")
    frozen, isolated = map(Path, sys.argv[1:])
    source = frozen.read_text()
    source = replace_once(
        source,
        '#include "../../flowstar/flowstar-toolbox/Continuous.h"',
        '#include "Continuous.h"\n#include "quad_gate_observer.h"',
    )
    source = replace_once(source, "int steps = 50;", "int steps = 1;")
    source = replace_once(
        source,
        "    vector<Symbolic_Remainder> symbolic_remainders;",
        "    initial_sets.resize(1);\n    initial_boxes.resize(1);\n"
        "    vector<Symbolic_Remainder> symbolic_remainders;",
    )
    source = replace_once(
        source,
        '        cl.CallMethod("CROWN_reach", params, output_coefficients);',
        '        cl.CallMethod("CROWN_reach", params, output_coefficients);\n'
        '        quad_gate::save_rpc(params, output_coefficients);',
    )
    source = replace_once(
        source,
        "initial_sets[sub_iter], 0.1, setting, safeSet, symbolic_remainders[sub_iter]);",
        "initial_sets[sub_iter], 0.005, setting, safeSet, symbolic_remainders[sub_iter]);",
    )
    source = replace_once(
        source,
        "        pool.join();\n        arch_ranges::record(results);",
        "        pool.join();\n        quad_gate::save_state(results, setting);\n"
        "        arch_ranges::record(results);",
    )
    source = replace_once(
        source,
        '        cout << "Flow* finished." << endl;',
        '        cout << "Flow* finished. GATE_PREFIX_ONLY T=0.005" << endl;',
    )
    isolated.parent.mkdir(parents=True, exist_ok=True)
    isolated.write_text(source)


if __name__ == "__main__":
    main()
