"""Three bounded candidate observations after the formal timing matrix."""
import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import time

import torch
import torch_tm_flowpipe as core
from torch_tm_flowpipe.prepared_remainder_replay import prepared_remainder_replay
from torch_tm_flowpipe.packed_boundary_range import packed_boundary_execution, make_plan
from experiments.endpoint_roundoff_repair.frozen import step
from experiments.boundary_execution.profile import Attribution, TensorCounts, state_identity, CATEGORIES
from experiments.boundary_execution.analyze import SCIENTIFIC_SHA
from experiments.boundary_execution.verify import source_files, sha, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--formal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert (args.formal/'SCHEDULE_COMPLETED.json').is_file(), 'observe only after formal timing finishes'
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    expected = source_files(SCIENTIFIC_SHA)
    assert all(sha(ROOT/p) == digest for p, digest in expected.items())
    import os
    assert sorted(os.sched_getaffinity(0)) == [2]
    cases = ['brusselator_before0995', 'brusselator_before1001', 'van_der_pol_before0100']
    for case in cases:
        loaded = core.load_terminal_checkpoint(args.inputs/case)
        current, state = loaded.current, loaded.normal_state
        before = state_identity(current, state)
        plant = loaded.contract['plant']
        rows, operations = [], []
        after_states = []
        for counts in (False, True):
            attribution = Attribution()
            make_plan.cache_clear()
            with prepared_remainder_replay(True), packed_boundary_execution(True), attribution, (TensorCounts(attribution) if counts else nullcontext()):
                start = time.perf_counter()
                result = step(plant, current, state, state.step_index + 1)
                elapsed = time.perf_counter() - start
            assert result.status == 'validated' and state_identity(current, state) == before
            after = state_identity(result.reset_tm, result.flowstar_normal_state)
            after_states.append(after)
            for category in [*CATEGORIES, 'outside_boundary']:
                seconds = sum(v for (c, _), v in attribution.seconds.items() if c == category)
                if category == 'outside_boundary':
                    seconds = elapsed - sum(attribution.seconds.values())
                rows.append(dict(pass_kind='counts' if counts else 'wall', category=category,
                    plant=plant, step=state.step_index + 1, history_length=before['history_length'],
                    seconds=seconds, step_seconds=elapsed,
                    calls=sum(v for (c, _), v in attribution.calls.items() if c == category),
                    interval_constructions=attribution.intervals[category], tensor_outputs=attribution.tensors[category],
                    explicit_copy_events=attribution.copies[category]))
            if counts:
                operations = [dict(category=c, operation=op, calls=count)
                              for (c, op), count in sorted(attribution.tensor_ops.items())]
        assert after_states[0] == after_states[1]
        output = args.output/case; output.mkdir()
        for name, values in [('breakdown.csv', rows), ('tensor_operations.csv', operations)]:
            with (output/name).open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
        (output/'source.json').write_text(json.dumps(dict(scientific_sha=SCIENTIFIC_SHA,
            scientific_sources=expected, harness_sha256=sha(__file__), prepared_remainder_replay=True,
            packed_boundary_execution=True, affinity=[2], threads=1, torch_version=torch.__version__,
            before=before, after=after_states[0], actual_bindings=attribution.bindings), indent=2)+'\n')
    (args.output/'COMPLETE.json').write_text(json.dumps({'cases': cases, 'scientific_sha': SCIENTIFIC_SHA}, indent=2)+'\n')


if __name__ == '__main__':
    main()
