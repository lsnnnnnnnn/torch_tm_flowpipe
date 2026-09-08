"""Bounded boundary profiles on real frozen states, with exclusive attribution.

Wall time and tensor allocation counts are separate passes. Dispatch counting
changes execution cost substantially and is never used in speed estimates.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import cProfile
import csv
import functools
import hashlib
import json
from pathlib import Path
import os
import pstats
import subprocess
import sys
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
import torch_tm_flowpipe as core
import torch_tm_flowpipe.flowpipe as flow
import torch_tm_flowpipe.batched_dense_tm as dense
import torch_tm_flowpipe.accepted_boundary_sr as sr
import torch_tm_flowpipe.symbolic_remainder as history
import torch_tm_flowpipe.endpoint_substitution as endpoint
import torch_tm_flowpipe.polynomial as polynomial
from torch_tm_flowpipe.interval import Interval
from torch_tm_flowpipe.prepared_remainder_replay import prepared_remainder_replay
from experiments.endpoint_roundoff_repair.frozen import setup, step, ROOT


CATEGORIES = dict(A='representation_and_coefficients', B='point_substitution',
    C='endpoint_error_and_ledger', D='polynomial_interval_range',
    E='composition_and_cutoff', F='center_scale_and_maps',
    G='history_checks_pack_propagate_commit', H='other_boundary')


class Attribution:
    def __init__(self):
        self.stack = []
        self.seconds = defaultdict(float)
        self.calls = Counter()
        self.intervals = Counter()
        self.tensors = Counter()
        self.tensor_ops = Counter()
        self.copies = Counter()
        self.originals = []
        self.bindings = []

    @property
    def category(self):
        return self.stack[-1]['category'] if self.stack else 'outside_boundary'

    def reset(self):
        assert not self.stack
        for value in (self.seconds, self.calls, self.intervals, self.tensors,
                      self.tensor_ops, self.copies):
            value.clear()

    def bind(self, owner, name, category, *, root=False):
        original = getattr(owner, name)
        operation = owner.__name__ + '.' + name

        @functools.wraps(original)
        def timed(*args, **kwargs):
            if not root and not self.stack:
                return original(*args, **kwargs)
            entry = dict(category=category, children=0.)
            key = (category, operation)
            self.calls[key] += 1
            self.stack.append(entry)
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                self.stack.pop()
                self.seconds[key] += elapsed - entry['children']
                if self.stack:
                    self.stack[-1]['children'] += elapsed

        targets = [(owner, name)]
        # Module-level from-imports retain their own references. Match the
        # exact original object, so aliases in flowpipe are covered as well.
        if not isinstance(owner, type):
            for module_name, module in list(sys.modules.items()):
                if module is None or not module_name.startswith('torch_tm_flowpipe'):
                    continue
                for alias, value in list(vars(module).items()):
                    if value is original and (module, alias) not in targets:
                        targets.append((module, alias))
        for target, alias in targets:
            self.originals.append((target, alias, getattr(target, alias)))
            setattr(target, alias, timed)
        self.bindings.append(dict(operation=operation, category=category,
            bindings=[o.__name__ + '.' + n for o, n in targets], boundary_root=root))

    def __enter__(self):
        self.originals = []
        self.bindings = []
        self.bind(flow, '_flowstar_normalized_insertion_transition', 'H', root=True)
        for name in ('sparse_tmvector_to_dense', 'dense_to_sparse_tmvector'):
            self.bind(dense, name, 'A', root=True)
        self.bind(polynomial.Polynomial, 'substitute_const_with_roundoff', 'C', root=True)
        self.bind(dense.BatchedTaylorModel, 'endpoint', 'C', root=True)
        self.bind(endpoint, 'enclose_constant_substitution', 'C', root=True)
        for owner, names, category in [
            (polynomial.Polynomial, ('__init__', 'clone', 'drop_variable', 'extend_vars'), 'A'),
            (polynomial.Polynomial, ('substitute_const',), 'B'),
            (dense.BatchedPolynomial, ('substitute_const_and_drop',), 'B'),
            (polynomial.Polynomial, ('evaluate_interval', 'evaluate_interval_split'), 'D'),
            (polynomial, ('evaluate_interval_normal',), 'D'),
            (sr, ('_interval_polynomial_range',), 'D'),
            (dense.DenseRemainderLedger, ('total',), 'C'),
            (polynomial.Polynomial, ('mul_truncate', '__mul__', 'cutoff', '__add__'), 'E'),
            (flow, ('insert_ctrunc_normal_dependency_preserving', '_insert_ctrunc_normal_horner_scalar',
                    '_tm_mul_ctrunc_horner_stage', '_tm_apply_cutoff_for_horner',
                    '_polynomial_part_for_power'), 'E'),
            (sr, ('split_endpoint_taylor_map', '_linear_polynomial_image', '_add_polynomial_images',
                  '_scale_and_cutoff_right_map'), 'E'),
            (flow, ('_tmvector_constant_part', '_tmvector_rm_constants',
                    '_tmvector_shift_polynomial_constants', '_normalized_tm_from_center_scale',
                    '_add_right_map_centering_diagnostics'), 'F'),
            (sr, ('prepare_accepted_boundary_sr', 'commit_accepted_boundary_sr'), 'G'),
            (history, ('validate_accepted_boundary_sr_queue', 'accepted_boundary_sr_queue_propagate',
                       'accepted_boundary_sr_queue_commit', 'accepted_boundary_sr_queue_sha256',
                       '_validated_interval_bounds', '_pack_interval_matrices', '_pack_interval_columns',
                       '_tensorized_interval_matrix_update_and_image'), 'G'),
        ]:
            for name in names:
                self.bind(owner, name, category)
        original = Interval.__init__

        @functools.wraps(original)
        def interval_init(*a, **kw):
            self.intervals[self.category] += 1
            return original(*a, **kw)

        self.originals.append((Interval, '__init__', original))
        Interval.__init__ = interval_init
        return self

    def __exit__(self, *args):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)


class TensorCounts(TorchDispatchMode):
    def __init__(self, attribution):
        super().__init__()
        self.attribution = attribution

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        category = self.attribution.category
        tensors = [v for v in tree_flatten(result)[0] if isinstance(v, torch.Tensor)]
        # Tensor objects returned by ATen, including views. Storage allocation
        # is not inferred from this count. Explicit clone/copy events are separate.
        self.attribution.tensors[category] += len(tensors)
        self.attribution.tensor_ops[(category, str(func))] += 1
        if func._schema.name in ('aten::clone', 'aten::copy_', 'aten::_to_copy'):
            self.attribution.copies[category] += 1
        return result


def state_identity(current, state):
    return dict(step=state.step_index, current=core.tmvector_hashes(current),
        normal_pre=core.tmvector_hashes(state.tmv_pre), normal_right=core.tmvector_hashes(state.tmv_right),
        queue=core.accepted_boundary_sr_queue_sha256(state.symbolic_queue) if state.symbolic_queue else None,
        history_length=len(state.symbolic_queue.J) if state.symbolic_queue else 0,
        generation=state.symbolic_queue.generation if state.symbolic_queue else None,
        reset_count=state.symbolic_queue.reset_count if state.symbolic_queue else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plant', choices=['brusselator', 'van_der_pol'], required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--advance', type=int, default=0)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--counts', action='store_true')
    p.add_argument('--cprofile-step', type=int)
    p.add_argument('--save-input-steps', type=int, nargs='*', default=[])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config, current, state = setup(args.plant)
    checkpoint_hashes = {}
    if args.checkpoint:
        restored = core.load_terminal_checkpoint(args.checkpoint, expected_dtype='float64', expected_order=config.order)
        assert restored.contract['plant'] == args.plant
        current, state = restored.current, restored.normal_state
        checkpoint_hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in args.checkpoint.iterdir() if f.is_file()}
    with prepared_remainder_replay(True):
        for _ in range(args.advance):
            result = step(args.plant, current, state, state.step_index + 1)
            assert result.status == 'validated', result.message
            current, state = result.reset_tm, result.flowstar_normal_state
    start_state = state_identity(current, state)
    rows, operations, tensors, states = [], [], [], []
    attribution = Attribution()
    for index in range(state.step_index + 1, state.step_index + args.steps + 1):
        before = state_identity(current, state)
        if index in args.save_input_steps:
            torch.save(dict(plant=args.plant, current=current, state=state, identity=before),
                       args.output / f'input_step{index:04d}.pt')
        if args.cprofile_step == index:
            # Separate replay on the SAME pre-step object, never step+1.
            profiler = cProfile.Profile()
            with prepared_remainder_replay(True):
                profiler.enable()
                observed = step(args.plant, current, state, index)
                profiler.disable()
            assert state_identity(current, state) == before
            profiler.dump_stats(str(args.output / 'self_times.pstats'))
            stats = pstats.Stats(profiler)
            with (args.output / 'self_times.csv').open('w') as out:
                w = csv.writer(out)
                w.writerow(['file', 'line', 'function', 'primitive_calls', 'calls', 'self_seconds', 'inclusive_seconds_not_additive'])
                for key, v in sorted(stats.stats.items(), key=lambda item: item[1][2], reverse=True):
                    w.writerow([*key, *v[:4]])
            (args.output / 'cprofile_state.json').write_text(json.dumps(before, indent=2) + '\n')
        attribution.reset()
        with attribution, (TensorCounts(attribution) if args.counts else nullcontext()):
            started = time.perf_counter()
            with prepared_remainder_replay(True):
                result = step(args.plant, current, state, index)
            elapsed = time.perf_counter() - started
        assert result.status == 'validated', result.message
        assert state_identity(current, state) == before, 'input state mutated'
        current, state = result.reset_tm, result.flowstar_normal_state
        after = state_identity(current, state)
        if args.cprofile_step == index:
            assert state_identity(observed.reset_tm, observed.flowstar_normal_state) == after
        states.append(dict(before=before, after=after, replays=result.backend_counters['post_accept_replay_calls']))
        boundary_total = sum(attribution.seconds.values())
        residual = elapsed - boundary_total
        assert residual >= 0
        for category, label in {**CATEGORIES, 'outside_boundary': 'nonboundary_or_unattributed'}.items():
            seconds = (sum(v for (c, _), v in attribution.seconds.items() if c == category)
                       if category != 'outside_boundary' else residual)
            rows.append(dict(plant=args.plant, step=index, history_length=before['history_length'],
                history_length_after=after['history_length'], category=category, label=label,
                seconds=seconds, calls=sum(v for (c, _), v in attribution.calls.items() if c == category),
                interval_constructions=attribution.intervals[category],
                tensor_outputs=attribution.tensors[category] if args.counts else '',
                explicit_copy_events=attribution.copies[category] if args.counts else '',
                step_seconds=elapsed, boundary_sum_seconds=boundary_total, nonboundary_seconds=residual,
                pass_kind='dispatch_counts_not_timing' if args.counts else 'exclusive_wall_timer'))
        operations.extend(dict(plant=args.plant, step=index, category=c, operation=op, seconds=seconds,
                               calls=attribution.calls[c, op]) for (c, op), seconds in attribution.seconds.items())
        tensors.extend(dict(step=index, category=c, operation=op, calls=count)
                       for (c, op), count in attribution.tensor_ops.items())
        print(json.dumps(dict(step=index, history=before['history_length'], seconds=elapsed,
                              boundary_seconds=boundary_total)), flush=True)
    for name, data in [('breakdown.csv', rows), ('operations.csv', operations), ('tensor_operations.csv', tensors)]:
        if data:
            with (args.output / name).open('w') as out:
                w = csv.DictWriter(out, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    (args.output / 'states.json').write_text(json.dumps(states, indent=2) + '\n')
    (args.output / 'source.json').write_text(json.dumps(dict(
        sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_hashes={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'src/torch_tm_flowpipe').glob('*.py'))},
        profile_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        imported_package=core.__file__, python=sys.executable, torch=torch.__version__,
        affinity=sorted(os.sched_getaffinity(0)), threads=torch.get_num_threads(),
        prepared_remainder_replay=True, boundary_execution=False,
        checkpoint=str(args.checkpoint), checkpoint_hashes=checkpoint_hashes, initial_state=start_state,
        cprofile_step=args.cprofile_step, counts=args.counts, bindings=attribution.bindings,
        accounting='Nested elapsed minus immediate timed child elapsed; all categories plus residual equal step wall time. Interval constructors stay inside their mathematical category. ATen output tensors include views; explicit copies counted separately. Dispatch pass never supplies performance denominators.'
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
