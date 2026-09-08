"""Same-input range microexperiment before production authorization."""
import argparse
from contextlib import contextmanager
import csv
import functools
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch_tm_flowpipe.polynomial as polynomial
import torch_tm_flowpipe.accepted_boundary_sr as sr
from torch_tm_flowpipe.prepared_remainder_replay import prepared_remainder_replay
from experiments.endpoint_roundoff_repair.frozen import step
from experiments.boundary_execution.profile import state_identity, Attribution, TensorCounts
from experiments.boundary_execution.prototype_range import (
    make_plan, evaluate_polynomial, evaluate_interval_coefficients,
)


def bits(value):
    return [float(value.lo).hex(), float(value.hi).hex()]


@contextmanager
def capture_ranges(records):
    originals = []

    def bind(owner, name, kind):
        original = getattr(owner, name)

        @functools.wraps(original)
        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            if kind == 'normal':
                poly, domain = args[:2]
                table = args[2] if len(args) > 2 else kwargs.get('step_exp_table')
                assert table is None, 'prototype must explicitly support supplied power tables before use'
                time_variable = args[4] if len(args) > 4 else kwargs.get('time_var_index', 0)
                variables = args[3] if len(args) > 3 else kwargs.get('state_var_indices')
                variables = set(range(poly.n_vars)) - {time_variable} if variables is None else set(variables) - {time_variable}
                candidate = lambda: evaluate_polynomial(poly, domain, normal=True,
                    state_variables=tuple(sorted(variables)), time_variable=time_variable)
            elif kind == 'point':
                poly, domain = args
                variables, time_variable = None, None
                candidate = lambda: evaluate_polynomial(poly, domain)
            else:
                coefficients, domain = args
                variables, time_variable = None, None
                candidate = lambda: evaluate_interval_coefficients(coefficients, domain, **kwargs)
            reference = lambda: original(*args, **kwargs)
            terms = ([(list(e), [float(c).hex()] * 2) for e, c in poly.terms.items()] if kind != 'interval'
                     else [(list(e), bits(coefficients[e])) for e in sorted(coefficients)])
            record = dict(kind=kind, exponents=[e for e, _ in terms], coefficients=[v for _, v in terms],
                          domain=[bits(x) for x in domain], state_variables=sorted(variables) if variables is not None else None,
                          time_variable=time_variable, bounds=bits(result))
            records.append((reference, candidate, record))
            return result

        targets = [(owner, name)]
        if not isinstance(owner, type):
            for modname, module in list(sys.modules.items()):
                if module is None or not modname.startswith('torch_tm_flowpipe'):
                    continue
                for alias, value in list(vars(module).items()):
                    if value is original and (module, alias) not in targets:
                        targets.append((module, alias))
        for target, alias in targets:
            originals.append((target, alias, getattr(target, alias)))
            setattr(target, alias, capture)

    bind(polynomial.Polynomial, 'evaluate_interval', 'point')
    bind(polynomial, 'evaluate_interval_normal', 'normal')
    bind(sr, '_interval_polynomial_range', 'interval')
    try:
        yield
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # This .pt is a private, freshly captured local development input, not an
    # evidence verifier input. The exported range objects below use hex JSON.
    loaded = torch.load(args.input, weights_only=False)
    plant, current, state = (loaded[k] for k in ('plant', 'current', 'state'))
    identity = state_identity(current, state)
    assert identity == loaded['identity']
    records = []
    with capture_ranges(records), prepared_remainder_replay(True):
        result = step(plant, current, state, state.step_index + 1)
    assert result.status == 'validated', result.message
    assert state_identity(current, state) == identity
    timings = []
    for repeat in range(1, 4):
        for mode in ([0, 1] if repeat % 2 else [1, 0]):
            # All structure preparation, packing, scalar powers, copies and
            # interval validations happen within the timed callbacks.
            make_plan.cache_clear()
            started = time.perf_counter()
            results = [record[mode]() for record in records]
            elapsed = time.perf_counter() - started
            for actual, (_, _, record) in zip(results, records):
                assert bits(actual) == record['bounds'], record
            timings.append(dict(repeat=repeat, mode='candidate' if mode else 'baseline', seconds=elapsed,
                                calls=len(records)))
    costs = list(csv.DictReader((args.profile / 'breakdown.csv').open()))
    selected = sum(float(r['seconds']) for r in costs if r['category'] == 'D')
    total = sum(float(r['seconds']) for r in costs)
    fraction = selected / total
    pairs = [next(r['seconds'] for r in timings if r['repeat'] == repeat and r['mode'] == 'baseline') /
             next(r['seconds'] for r in timings if r['repeat'] == repeat and r['mode'] == 'candidate') for repeat in range(1, 4)]
    speedup = statistics.median(pairs)
    allocations = []
    for mode in (0, 1):
        attribution = Attribution()
        make_plan.cache_clear()
        with attribution, TensorCounts(attribution):
            attribution.stack.append({'category': 'D', 'children': 0.})
            try:
                counted_results = [record[mode]() for record in records]
            finally:
                attribution.stack.pop()
        assert all(bits(actual) == record['bounds'] for actual, (_, _, record) in zip(counted_results, records))
        allocations.append(dict(mode='candidate' if mode else 'baseline',
            interval_constructions=sum(attribution.intervals.values()),
            tensor_outputs=sum(attribution.tensors.values()),
            explicit_copy_events=sum(attribution.copies.values()),
            scope='Separate dispatch pass over exactly the same callbacks; not a time denominator.'))
    report = dict(plant=plant, step=state.step_index + 1, input_state=identity,
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(sys.modules[make_plan.__module__].__file__).read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        profile=str(args.profile), profile_sha256=hashlib.sha256((args.profile/'breakdown.csv').read_bytes()).hexdigest(),
        f=fraction, s=speedup, local_speedups=pairs, timings=timings, allocations=allocations,
        predicted_whole_window_speedup=1 / ((1 - fraction) + fraction / speedup),
        infinite_speedup_limit=1 / (1 - fraction),
        scope='Representative actual step range calls; f is the corresponding exclusive full window. Prediction is approximate, not full-run timing.',
        same_work_bitwise_equal=True, required_safety_checks_and_preparation_included=True,
        prepared_remainder_replay=True, production_boundary_enabled=False)
    (args.output/'result.json').write_text(json.dumps(report, indent=2) + '\n')
    with gzip.open(args.output/'range_inputs.jsonl.gz', 'wt') as out:
        for _, _, record in records:
            out.write(json.dumps(record, separators=(',', ':')) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_state','timings')}), flush=True)


if __name__ == '__main__':
    main()
