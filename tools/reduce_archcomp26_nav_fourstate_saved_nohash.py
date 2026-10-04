#!/usr/bin/env python3
"""Read existing NAV records to stdout, or write local tables from that JSON."""

import argparse
import csv
import json
import math
from pathlib import Path
import struct
import sys


ROOT = Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs')
STATES = ('x', 'y', 'speed', 'heading')
CELL_FIELDS = ('tube_lo', 'tube_hi', 'endpoint_lo', 'endpoint_hi',
               'partition_mean_tube_width', 'partition_max_tube_width',
               'partition_mean_endpoint_width', 'partition_max_endpoint_width')


def sources():
    for variant, boxes in (('standard', 640), ('robust', 25)):
        yield variant, boxes, 'working_p3', '2026-10-02 current P3', ROOT / f'archcomp26_20261001/nav_author_{variant}_working_p3_full30_001/ranges.bin', '<II16d'
        for method in ('huan', 'xiangru'):
            yield variant, boxes, method, '2026-09-23 historical GPU', ROOT / f'archcomp_review_20260923/suite_v1/nav_{variant}_{method}/ranges.bin', '<QQd16d'
        native = (ROOT / 'archcomp26_20261001/nav_author_standard_native_full30_001/ranges.bin'
                  if variant == 'standard' else ROOT / 'archcomp_review_20260923/native_matched/nav_robust/ranges.bin')
        yield variant, boxes, 'flowstar_native', '2026-10-02 current native' if variant == 'standard' else '2026-09-23 historical native', native, '<QQd16d'


def reduce_source(variant, boxes, method, generation, path, layout):
    record = struct.Struct(layout)
    expected = boxes * 600
    if path.stat().st_size != expected * record.size:
        raise ValueError(f'Incomplete saved file: {path}')
    next_step = [1] * boxes
    cells = [[[math.inf, -math.inf, math.inf, -math.inf, 0., 0., 0., 0.]
              for _ in STATES] for _ in range(600)]
    lane_full_tubes = [[[math.inf, -math.inf] for _ in STATES] for _ in range(boxes)]
    count = 0
    with path.open('rb') as stream:
        while chunk := stream.read(record.size):
            lane, step, *values = record.unpack(chunk)
            if layout == '<QQd16d':
                h, *values = values
                if not math.isfinite(h) or abs(h - .01) > 1e-12:
                    raise ValueError(f'Unexpected step duration in {path}')
            if not (lane < boxes and step == next_step[lane] and 1 <= step <= 600):
                raise ValueError(f'Invalid lane-step identity in {path}: {lane}, {step}')
            if not all(map(math.isfinite, values)):
                raise ValueError(f'Nonfinite range in {path}')
            for state in range(4):
                tlo, thi, elo, ehi = values[state * 4:state * 4 + 4]
                if not tlo <= elo <= ehi <= thi:
                    raise ValueError(f'Reversed interval or endpoint outside tube in {path}')
                row = cells[step - 1][state]
                row[0], row[1] = min(row[0], tlo), max(row[1], thi)
                row[2], row[3] = min(row[2], elo), max(row[3], ehi)
                tw, ew = thi - tlo, ehi - elo
                row[4] += tw
                row[5] = max(row[5], tw)
                row[6] += ew
                row[7] = max(row[7], ew)
                full = lane_full_tubes[lane][state]
                full[0], full[1] = min(full[0], tlo), max(full[1], thi)
            next_step[lane] += 1
            count += 1
    if count != expected or any(step != 601 for step in next_step):
        raise ValueError(f'Missing saved lane-step records in {path}')
    for step in cells:
        for cell in step:
            cell[4] /= boxes
            cell[6] /= boxes
    full = []
    for state in range(4):
        intervals = [lane[state] for lane in lane_full_tubes]
        widths = [hi - lo for lo, hi in intervals]
        full.append([min(lo for lo, _ in intervals), max(hi for _, hi in intervals),
                     math.fsum(widths) / boxes, max(widths)])
    return dict(instance='nav-' + variant, method=method, source_generation=generation,
                source=str(path), source_bytes=expected * record.size, record_layout=layout,
                boxes=boxes, steps=600, records=count, h=.01, t_end=6.,
                complete_saved_grid=True, interval_and_identity_errors=0,
                cells=cells, full_tube_by_state=full)


def tables(saved, output):
    data = json.loads(saved.read_text())
    if data['states'] != list(STATES) or len(data['methods']) != 8:
        raise ValueError('Expected two NAV contracts and four methods')
    identities = {(m['instance'], m['method']) for m in data['methods']}
    if identities != {('nav-' + v, m) for v in ('standard', 'robust')
                      for m in ('working_p3', 'huan', 'xiangru', 'flowstar_native')}:
        raise ValueError('Incorrect contract/method set')
    output.mkdir(parents=True, exist_ok=False)
    long_rows, widths = [], []
    for method in data['methods']:
        if not (method['complete_saved_grid'] and method['steps'] == 600 and
                method['records'] == method['boxes'] * 600 and len(method['cells']) == 600):
            raise ValueError('Incomplete saved grid')
        meta = {key: method[key] for key in ('instance', 'method', 'source_generation')}
        for step, cells in enumerate(method['cells'], 1):
            if len(cells) != 4:
                raise ValueError('Missing state')
            for state, values in zip(STATES, cells):
                if len(values) != 8 or not all(map(math.isfinite, values)):
                    raise ValueError('Invalid reduced state')
                row = dict(meta, step=step, t_start=(step - 1) / 100,
                           t_end=step / 100, state=state, **dict(zip(CELL_FIELDS, values)))
                row.update(tube_union_width=values[1] - values[0], endpoint_union_width=values[3] - values[2])
                long_rows.append(row)
        for i, state in enumerate(STATES):
            last = method['cells'][-1][i]
            full = method['full_tube_by_state'][i]
            widths.append(dict(meta, state=state, boxes=method['boxes'], terminal_time=6.,
                               terminal_endpoint_lo=last[2], terminal_endpoint_hi=last[3],
                               terminal_endpoint_union_width=last[3] - last[2],
                               terminal_partition_mean_endpoint_width=last[6],
                               terminal_partition_max_endpoint_width=last[7],
                               last_step_tube_lo=last[0], last_step_tube_hi=last[1],
                               last_step_tube_union_width=last[1] - last[0],
                               full_time_tube_lo=full[0], full_time_tube_hi=full[1],
                               full_time_tube_union_width=full[1] - full[0],
                               full_time_partition_mean_tube_width=full[2],
                               full_time_partition_max_tube_width=full[3],
                               max_step_tube_union_width=max(s[i][1] - s[i][0] for s in method['cells'])))
    for name, rows in (('all_states_saved_curves.csv', long_rows), ('absolute_widths.csv', widths)):
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    audit = {key: value for key, value in data.items() if key != 'methods'}
    audit.update(source_json=str(saved.resolve()), source_json_bytes=saved.stat().st_size,
                 csv_rows=len(long_rows), absolute_width_rows=len(widths),
                 methods=[{k: v for k, v in m.items() if k not in ('cells', 'full_tube_by_state')}
                          for m in data['methods']],
                 evidence_qualification='Saved numerical bounds only; no new solve, author checker verdict or independent NNCS certificate.',
                 units='Time is seconds; state units not explicitly declared by source contract, retained without conversion.')
    (output / 'AUDIT.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    (output / 'README.md').write_text('''# NAV 两合同四方法：四态保存界与绝对宽度

这里只读已有 `ranges.bin`；不运行控制器、求解器或新 benchmark，不计算内容摘要。
`all_states_saved_curves.csv` 有 19200 行：standard/robust × 4 方法 × 600 步 × x/y/speed/heading。
`absolute_widths.csv` 有 32 行，逐态区分 T=6 终点、最后一步 tube、[0,6] 全 tube union，
并给终点每盒 mean/max 以及每个初盒整段时间包络的宽度 mean/max。
逐步表同时保留每盒 tube/endpoint mean/max；这些统计均来自所有真实保存分盒。

P3 是当前 working P3；Huan/Xiangru 是 2026-09-23 同合同历史；standard native 是 2026-10-02，robust native 是历史。
这不是四方法同时新跑的计时比较；完整保存数值时域不升级为独立浮点 NNCS 证书。
x/y 投影图仍使用原 `nav_current_p3_saved_20261004_001/`，新增本表补齐其它物理态。
所有 8 源逐条检查有限、顺序、600 步完整、h=.01（152字节记录）、endpoint 包含在本段 tube。
源路径、字节数、真实记录数与布局见 `AUDIT.json`；精简原始派生 JSON 在其 `source_json`。
时间单位为秒，状态单位在来源合同中未显式声明，未换算。
''')
    print(json.dumps({'csv_rows': len(long_rows), 'absolute_width_rows': len(widths)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-json', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.saved_json:
        if not args.output_dir:
            parser.error('--saved-json requires --output-dir')
        tables(args.saved_json, args.output_dir)
    else:
        if args.output_dir:
            parser.error('Remote reduction writes only stdout')
        payload = {'schema': 'archcomp26-nav-fourstate-saved-nohash-v1', 'states': STATES,
                   'cell_fields': CELL_FIELDS,
                   'full_tube_fields': ('lo', 'hi', 'partition_mean_width', 'partition_max_width'),
                   'methods': [reduce_source(*args) for args in sources()]}
        json.dump(payload, sys.stdout, allow_nan=False, separators=(',', ':'))
        print()


if __name__ == '__main__':
    main()
