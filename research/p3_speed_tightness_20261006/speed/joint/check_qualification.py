#!/usr/bin/env python3
"""New CPU control-record checks only; no Torch, numerical run or old checker."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from run_quad_joint_candidate import qualify


def main():
    checked = []
    numerical = dict(failed_execs=0, exec_calls=6, completed_execs=6,
                     reused_powers=72, reused_power_chains=18)
    trig = SimpleNamespace(last_plan=dict(opposite_op_pairs=3))

    def control(refreshes):
        return SimpleNamespace(records=[{} for _ in range(refreshes)], counters=dict(
            base_nn_calls=refreshes, extra_nn_calls=refreshes, total_nn_calls=2*refreshes,
            injected_refreshes=refreshes, total_control_rows=refreshes*1024*3))

    flags = dict(gate_passed=True, trig_restored=True, control_restored=True)
    for mode, refreshes in (("batch2", 2), ("full50", 50)):
        receipt = qualify(mode, trig, control(refreshes), numerical, **flags)
        assert receipt["expected_control"]["total_nn_calls"] == 2*refreshes
        assert receipt["saved_outputs_equal_claim"] is False
        checked.append(mode+" complete control coverage accepted")
    for label in ("missing extra NN calls", "failed trig execution", "missing control restore", "missing GPU gate"):
        c, n, f = control(50), copy.deepcopy(numerical), dict(flags)
        if label == "missing extra NN calls":
            c.counters["extra_nn_calls"] = 0
            c.counters["total_nn_calls"] = 50
        elif label == "failed trig execution":
            n["failed_execs"] = 1
        elif label == "missing control restore":
            f["control_restored"] = False
        else:
            f["gate_passed"] = False
        try:
            qualify("full50", trig, c, n, **f)
        except RuntimeError:
            checked.append(label+" rejected")
        else:
            raise AssertionError(label)
    receipt = dict(status="PASS_RECORD_ONLY_CHECKS", cases=checked, cases_passed=len(checked),
                   uses_synthetic_records=True, torch_imported=False, numerical_runs=0,
                   gpu_runs=0, old_checker_runs=0, digest_operations=0,
                   qualifies_joint_numerics=False)
    Path(__file__).with_name("QUALIFICATION_CPU_CHECK.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
