"""Independent full observation with explicit export/range backends and timing.

The physical range observer defaults to the existing exact Fraction evaluator.
The original feasibility observer module and its defaults are untouched.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

from experiments.whole_engine_feasibility.observe_saved import _FIELDS, _read_json, _validate
from experiments.whole_engine_feasibility.observe import observe_step
from experiments.whole_engine_feasibility import export as fraction_export
from experiments.xiangru_adoption import common
from . import outward_export


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _observe_outward_step(models, *, plant, step, h, lane=0):
    """Original row/clock semantics with the explicitly selected range backend."""
    from fractions import Fraction
    import math
    from experiments.whole_engine_feasibility.observe import _number, _interval as check_pair
    from .outward_range import measure, OBSERVER
    h = _number(h)
    if type(step) is not int or step < 1 or h <= 0:
        raise ValueError('positive fixed step and one-based accepted step required')
    start, end = (step-1)*Fraction(h), step*Fraction(h)
    rows = []
    for view in ('endpoint', 'tube'):
        bounds = measure(models[view])
        if len(bounds) != 2:
            raise ValueError('observer requires both x and y')
        for coordinate, (lo, hi) in zip(('x', 'y'), bounds):
            check_pair((lo, hi))
            width = hi-lo
            if not math.isfinite(width):
                raise ValueError('nonfinite observed width')
            rows.append(dict(plant=plant, lane=lane, step=step, view=view, coordinate=coordinate,
                             h_hex=h.hex(), t_start_exact=str(start), t_end_exact=str(end), time=float(end),
                             lo=lo, hi=hi, lo_hex=lo.hex(), hi_hex=hi.hex(), width=width, observer=OBSERVER))
    return rows


def run(input_path, output, backend, range_backend='fraction'):
    input_path, output = input_path.resolve(), output.resolve()
    source_path = input_path.with_name('summary.json')
    source = _read_json(source_path.read_text())
    lane_ids, counts = source['case']['lane_ids'], source['accepted_steps']
    requested = source['requested_steps']
    if (type(requested) is not int or requested < 1 or not lane_ids
            or source['completed'] is not True or source.get('failure') is not None
            or any(type(count) is not int for count in counts)
            or counts != [requested] * len(lane_ids)):
        raise ValueError('observation requires every lane to complete the requested horizon')
    if backend not in ('outward_binary64', 'fraction'):
        raise ValueError('unknown export backend')
    h = float(source['settings']['step'])
    exporter = outward_export if backend == 'outward_binary64' else fraction_export
    compose = exporter.outward_composition if backend == 'outward_binary64' else exporter.exact_composition
    if range_backend == 'fraction':
        range_source, observe, observer_name = common, observe_step, 'EXACT_FRACTION_COMPLETE_TM'
    elif range_backend == 'outward_binary64':
        from . import outward_range
        range_source, observe, observer_name = outward_range, _observe_outward_step, outward_range.OBSERVER
    else:
        raise ValueError('unknown range backend')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'whole-engine-explicit-export-observation/2',
              'export_backend': backend, 'range_backend': range_backend, 'observer': observer_name,
              'composition': 'full-degree interval-coefficient substitution; no cutoff or truncation',
              'input': str(input_path), 'input_sha256': sha256(input_path),
              'source_summary': str(source_path), 'source_summary_sha256': sha256(source_path),
              'exporter_source': str(Path(exporter.__file__).resolve()),
              'exporter_source_sha256': sha256(Path(exporter.__file__)),
              'range_observer_source': str(Path(range_source.__file__).resolve()),
              'range_observer_source_sha256': sha256(Path(range_source.__file__)),
              'range_arithmetic_source_sha256': (sha256(Path(outward_export.__file__))
                                                  if range_backend == 'outward_binary64' else None),
              'driver_source_sha256': sha256(Path(__file__)),
              'plant': source['plant'], 'lane_ids': lane_ids,
              'expected_accepted_steps': counts, 'observed_steps': [0] * len(lane_ids),
              'solver_completed': source['completed'], 'requested_steps': source['requested_steps'],
              'status': 'running', 'complete_observation': False, 'model_records': 0, 'bound_rows': 0,
              'export_wall_seconds': 0., 'export_cpu_seconds': 0.,
              'range_wall_seconds': 0., 'range_cpu_seconds': 0.,
              'max_export_seconds': 0., 'max_terms_per_component': 0,
              'timing': 'offline export + explicitly selected range backend + serialization; no solver timing'}
    start, cpu_start = time.perf_counter(), time.process_time()

    def checkpoint():
        report['offline_wall_seconds'] = time.perf_counter() - start
        report['offline_cpu_seconds'] = time.process_time() - cpu_start
        (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    checkpoint()
    try:
        with gzip.open(input_path, 'rt') as snapshots, gzip.open(output / 'models.jsonl.gz', 'wt') as model_file, \
                (output / 'bounds.csv').open('w', newline='') as bound_file:
            writer = csv.DictWriter(bound_file, fieldnames=_FIELDS + ['export_backend', 'range_backend'])
            writer.writeheader()
            for step, line in enumerate(snapshots, 1):
                record = _read_json(line)
                _validate(record, step, lane_ids, h)
                if step > counts[0]:
                    raise ValueError('input has excess snapshots')
                for index, lane in enumerate(lane_ids):
                    t, c = time.perf_counter(), time.process_time()
                    components = compose(record['pre'][index], record['pre_rem'][index], record['pre_exponents'],
                                         record['tmv'][index], record['tmv_rem'][index], record['tmv_exponents'])
                    models = exporter.canonical_models(components, h)
                    elapsed, cpu = time.perf_counter() - t, time.process_time() - c
                    report['export_wall_seconds'] += elapsed
                    report['export_cpu_seconds'] += cpu
                    report['max_export_seconds'] = max(report['max_export_seconds'], elapsed)
                    terms = list(map(len, components))
                    report['max_terms_per_component'] = max(report['max_terms_per_component'], *terms)
                    t, c = time.perf_counter(), time.process_time()
                    rows = observe(models, plant=source['plant'], step=step, h=h, lane=lane)
                    report['range_wall_seconds'] += time.perf_counter() - t
                    report['range_cpu_seconds'] += time.process_time() - c
                    for row in rows:
                        row['export_backend'] = backend
                        row['range_backend'] = range_backend
                    writer.writerows(rows)
                    model_file.write(json.dumps({'plant': source['plant'], 'step': step, 'lane': lane,
                                                 'export_backend': backend, 'range_backend': range_backend,
                                                 'export_seconds': elapsed,
                                                 'terms': terms, 'models': models}, separators=(',', ':'),
                                                allow_nan=False) + '\n')
                    report['model_records'] += 1
                    report['bound_rows'] += len(rows)
                    report['observed_steps'][index] += 1
                if step % 100 == 0:
                    model_file.flush()
                    bound_file.flush()
                    checkpoint()
                    print(json.dumps({'event': 'observation_progress', 'step': step,
                                      'export_backend': backend, 'offline_wall_seconds': report['offline_wall_seconds']}), flush=True)
        if report['observed_steps'] != counts or report['bound_rows'] != 4 * sum(counts):
            raise ValueError('incomplete observation')
        report.update(status='complete', complete_observation=True)
    except BaseException as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        checkpoint()
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--export-backend', choices=['outward_binary64', 'fraction'], default='outward_binary64')
    parser.add_argument('--range-backend', choices=['fraction', 'outward_binary64'], default='fraction')
    args = parser.parse_args()
    run(args.input, args.output, args.export_backend, args.range_backend)
