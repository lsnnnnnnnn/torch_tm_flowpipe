"""Full boundary objects, real next-state consumers, rollback and resume."""
import argparse
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from fractions import Fraction
import functools
import gzip
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import torch_tm_flowpipe as core
import torch_tm_flowpipe.accepted_boundary_sr as sr
from torch_tm_flowpipe.prepared_remainder_replay import prepared_remainder_replay
from torch_tm_flowpipe.packed_boundary_range import packed_boundary_execution
from experiments.endpoint_roundoff_repair.frozen import setup, step


TIMING_FIELDS = frozenset({'host_to_device_s', 'dense_kernel_s', 'device_to_host_s'})


def canonical(value):
    """All numerical payloads and decisions; omit only three elapsed counters."""
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
        return {'tensor': str(value.dtype), 'shape': list(value.shape),
                'values': [v.hex() for v in values] if value.is_floating_point() else values}
    if isinstance(value, float):
        return {'float_hex': value.hex()}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if is_dataclass(value):
        return {'type': type(value).__name__, 'fields': {f.name: canonical(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, dict):
        return {'items': [[canonical(k), canonical(v)] for k, v in value.items() if k not in TIMING_FIELDS]}
    if isinstance(value, (tuple, list)):
        return [canonical(x) for x in value]
    raise TypeError(f'unrecorded boundary payload type: {type(value)}')


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value), separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@contextmanager
def capture_boundary_objects():
    records, originals = [], []
    for name in ('prepare_accepted_boundary_sr', 'commit_accepted_boundary_sr'):
        original = getattr(sr, name)

        def wrap(fn, label):
            @functools.wraps(fn)
            def observed(*a, **kw):
                result = fn(*a, **kw)
                records.append({'stage': label, 'result': canonical(result)})
                return result
            return observed

        observed = wrap(original, name)
        for module_name, module in list(sys.modules.items()):
            if module is None or not module_name.startswith('torch_tm_flowpipe'):
                continue
            for alias, value in list(vars(module).items()):
                if value is original:
                    originals.append((module, alias, value))
                    setattr(module, alias, observed)
    try:
        yield records
    finally:
        for module, alias, value in reversed(originals):
            setattr(module, alias, value)


def verify_sequence(checkpoint, *, steps, output=None):
    loaded = core.load_terminal_checkpoint(checkpoint)
    plant = loaded.contract['plant']
    config = setup(plant)[0]
    lanes = [(loaded.current, loaded.normal_state), (loaded.current, loaded.normal_state)]
    original_state = digest(lanes[0])
    rows, saved_results = [], [[], []]
    for offset in range(steps):
        records = []
        for enabled in (False, True):
            current, state = lanes[int(enabled)]
            before = digest((current, state))
            with prepared_remainder_replay(True), packed_boundary_execution(enabled), capture_boundary_objects() as stages:
                segment = step(plant, current, state, state.step_index + 1)
                assert segment.status == 'validated', segment.message
                assert digest((current, state)) == before, 'accepted input mutated'
                record = {'step': state.step_index + 1, 'history_before': len(state.symbolic_queue.J) if state.symbolic_queue else 0,
                          'segment': canonical(segment), 'stages': stages}
                assert [s['stage'] for s in stages] == ['prepare_accepted_boundary_sr', 'commit_accepted_boundary_sr']
                after = digest(segment)
                for _ in range(2):
                    segment.endpoint_raw_tm.range_box()
                    segment.tm.range_box()
                    segment.flowstar_normal_state.endpoint_tm().range_box()
                assert digest(segment) == after, 'measurement appended error or changed ownership'
            records.append(record)
            saved_results[int(enabled)].append(segment)
            lanes[int(enabled)] = (segment.reset_tm, segment.flowstar_normal_state)
        assert records[0] == records[1], f'full boundary payload differs at {plant} step {records[0]["step"]}'
        rows.append(records[0])
    assert digest((loaded.current, loaded.normal_state)) == original_state
    assert steps >= 2
    # Resume both modes from the first actual returned state. No box reset.
    if output:
        for enabled in (False, True):
            first = saved_results[int(enabled)][0]
            index = first.flowstar_normal_state.step_index
            resumed_path = output / ('resume_candidate' if enabled else 'resume_baseline')
            core.save_terminal_checkpoint(resumed_path, current=first.reset_tm, normal_state=first.flowstar_normal_state,
                scheduler={'accepted_steps': index, 'time_exact': str(index * Fraction(first.h))},
                contract={'plant': plant, 'config': config.as_dict()},
                provenance={'purpose': 'boundary execution uninterrupted/resume comparison',
                            'prepared_remainder_replay': True, 'packed_boundary_execution': enabled})
            restored = core.load_terminal_checkpoint(resumed_path, expected_order=config.order, expected_dtype='float64')
            with prepared_remainder_replay(True), packed_boundary_execution(enabled):
                resumed = step(plant, restored.current, restored.normal_state, index + 1)
            assert canonical(resumed) == canonical(saved_results[int(enabled)][1]), 'resume changed actual next-step input'
    # Force the existing endpoint failure path after validation with a real
    # nonempty history. Neither lane may replace the accepted current/queue.
    for enabled in (False, True):
        first = saved_results[int(enabled)][0]
        current, state = first.reset_tm, first.flowstar_normal_state
        assert state.symbolic_queue.J
        before = digest((current, state))

        def fail(*args, **kwargs):
            raise FloatingPointError('boundary execution regression: injected endpoint failure')

        with prepared_remainder_replay(True), packed_boundary_execution(enabled), patch.object(core.TaylorModel, 'substitute_const_with_roundoff', fail):
            rejected = step(plant, current, state, state.step_index + 1)
        assert rejected.status == 'failed' and rejected.reset_tm is None and rejected.endpoint_raw_tm is None
        assert digest((current, state)) == before, 'failed attempt committed history'
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    rows = verify_sequence(args.checkpoint, steps=args.steps, output=args.output)
    with gzip.open(args.output/'boundary_objects.jsonl.gz', 'wt') as out:
        for row in rows:
            out.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
    result = {'checkpoint': str(args.checkpoint), 'steps': [r['step'] for r in rows],
              'history_lengths': [r['history_before'] for r in rows], 'full_boundary_objects_equal': True,
              'actual_next_input_used': True, 'repeated_measurement_preserved': True,
              'both_modes_failure_rollback': True, 'both_modes_checkpoint_resume': True}
    (args.output/'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
