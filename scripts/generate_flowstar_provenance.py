from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FLOWSTAR_ROOT = Path('/srv/local/shengenli/flowstar')
CSV_PATH = ROOT / 'outputs' / 'flowstar_vdp_remainder_cutoff_sweep.csv'
LIB_PATH = FLOWSTAR_ROOT / 'flowstar-toolbox' / 'libflowstar.a'
ARTIFACT_BUNDLE_COMMIT_NOTE = (
    'This manifest is generated from the clean source tree recorded in '
    'source_tree_commit_used_for_generation. The artifact bundle commit '
    'containing this refreshed manifest may be later.'
)


def run(cmd: list[str], cwd: Path = ROOT) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    text = proc.stdout if proc.returncode == 0 else proc.stdout + proc.stderr
    return text.strip()


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: str | Path | None) -> dict[str, Any] | None:
    if path in (None, ''):
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return {
        'path': str(p),
        'exists': p.exists(),
        'size_bytes': p.stat().st_size if p.exists() and p.is_file() else None,
        'sha256': sha256(p),
    }


def safe_plot_stem(cpp_path: Path) -> str:
    stem = cpp_path.name[:-4] if cpp_path.name.endswith('.cpp') else cpp_path.name
    return re.sub(r'[^A-Za-z0-9_]+', '_', stem).strip('_')


def plot_paths_for_cpp(cpp_path: Path) -> list[Path]:
    return sorted(cpp_path.parent.glob(f'{safe_plot_stem(cpp_path)}_t_*.plt'))


def parse_setting(label: str) -> tuple[str, str]:
    m = re.fullmatch(r'rem([^_]+)_cut(.+)', label or '')
    return ('', '') if m is None else (m.group(1), m.group(2))


def md_escape(value: Any) -> str:
    return ('' if value is None else str(value)).replace('|', '/')


def load_flow_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get('tool') == 'flowstar']


def collect_representatives(flow_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    specs = [
        ('order2_loose_failed', {'setting_label': 'rem1e-4_cut1e-10', 'order': '2', 'h': '0.01', 'steps': '10', 'status': 'failed'}),
        ('order4_loose_completed', {'setting_label': 'rem1e-4_cut1e-10', 'order': '4', 'h': '0.01', 'steps': '10', 'status': 'completed'}),
        ('order8_strict_completed', {'setting_label': 'rem1e-10_cut1e-15', 'order': '8', 'h': '0.0025', 'steps': '10', 'status': 'completed'}),
    ]
    reps: list[dict[str, Any]] = []
    for label, spec in specs:
        match = next((r for r in flow_rows if all(r.get(k) == v for k, v in spec.items())), None)
        if match is None and label == 'order8_strict_completed':
            match = next((r for r in flow_rows if r.get('setting_label') == 'rem1e-10_cut1e-15' and r.get('order') == '8'), None)
        if match is None:
            reps.append({'label': label, 'found': False, 'requested': spec})
            continue
        cpp_path = Path(match['flowstar_model_path'])
        if not cpp_path.is_absolute():
            cpp_path = ROOT / cpp_path
        reps.append({
            'label': label,
            'found': True,
            'system': match.get('system'),
            'setting_label': match.get('setting_label'),
            'h': match.get('h'),
            'steps': match.get('steps'),
            'order': match.get('order'),
            'status': match.get('status'),
            'failure_reason': match.get('failure_reason'),
            'cpp': file_record(cpp_path),
            'stdout': file_record(match.get('flowstar_stdout_path')),
            'stderr': file_record(match.get('flowstar_stderr_path')),
            'plots': [file_record(p) for p in plot_paths_for_cpp(cpp_path)],
        })
    return reps


def build_manifest(flow_rows: list[dict[str, str]]) -> dict[str, Any]:
    status_counts = Counter(r.get('status', '') for r in flow_rows)
    unique_cpp = sorted({r['flowstar_model_path'] for r in flow_rows if r.get('flowstar_model_path')})
    cpp_paths = [(ROOT / p if not Path(p).is_absolute() else Path(p)) for p in unique_cpp]
    existing_cpp = [p for p in cpp_paths if p.exists()]
    cpp_texts = [p.read_text(encoding='utf-8', errors='ignore') for p in existing_cpp]
    completed_gnuplot = [r for r in flow_rows if r.get('status') == 'completed' and r.get('box_source') == 'flowstar_gnuplot_last_segment_and_tube']
    endpoint_false_ok = all(str(r.get('endpoint_box_available')).lower() == 'false' for r in completed_gnuplot)
    compiler_version = run(['g++', '--version']).splitlines()
    flowstar_status = run(['git', '-C', str(FLOWSTAR_ROOT), 'status', '--short'])
    generation_worktree_status = run(['git', 'status', '--short']) or 'clean'
    return {
        'schema_version': 1,
        'generated_utc_date': '2026-06-02',
        'scope': 'plant-only fixed-step/fixed-order Flow* toolbox provenance audit; no new reachability algorithm',
        'torch_tm_flowpipe': {
            'path': str(ROOT),
            'branch': run(['git', 'branch', '--show-current']),
            'source_tree_commit_used_for_generation': run(['git', 'rev-parse', 'HEAD']),
            'source_tree_commit_message_used_for_generation': run(['git', 'log', '-1', '--oneline']),
            'generation_worktree_status': generation_worktree_status,
            'remote_origin_main_at_generation': run(['git', 'ls-remote', 'origin', 'main']),
            'artifact_bundle_commit_note': ARTIFACT_BUNDLE_COMMIT_NOTE,
        },
        'flowstar_backend': {
            'FLOWSTAR_ROOT': str(FLOWSTAR_ROOT),
            'backend': 'toolbox_cpp',
            'description': 'chenxin415/flowstar C++ toolbox/static library backend, not a Python package',
            'remote_v': run(['git', '-C', str(FLOWSTAR_ROOT), 'remote', '-v']).splitlines(),
            'head_sha': run(['git', '-C', str(FLOWSTAR_ROOT), 'rev-parse', 'HEAD']),
            'status_short': flowstar_status.splitlines() if flowstar_status else [],
            'libflowstar_a': file_record(LIB_PATH),
            'compiler_version': compiler_version,
        },
        'generated_cpp_audit': {
            'comparison_csv': str(CSV_PATH),
            'generated_cpp_case_count_from_csv': len(unique_cpp),
            'existing_cpp_case_count': len(existing_cpp),
            'all_generated_cases_include_Continuous_h': all('#include "Continuous.h"' in t for t in cpp_texts),
            'all_generated_cases_call_ode_reach': all('ode.reach(' in t for t in cpp_texts),
            'all_generated_cases_use_setFixedStepsize': all('setting.setFixedStepsize' in t for t in cpp_texts),
            'all_generated_cases_emit_GNUPLOT_interval_plots': all('plot_2D_interval_GNUPLOT' in t for t in cpp_texts),
            'runner_links_libflowstar_a': True,
            'runner_link_evidence': 'comparisons/flowstar/run_flowstar.py compile_cmd uses -L $FLOWSTAR_ROOT/flowstar-toolbox and -lflowstar',
        },
        'result_counts': {
            'source_csv': str(CSV_PATH),
            'flowstar_row_count': len(flow_rows),
            'status_counts': dict(sorted(status_counts.items())),
            'completed_gnuplot_rows': len(completed_gnuplot),
        },
        'box_and_ratio_semantics': {
            'box_source': 'GNUPLOT interval plot files parsed from plot_2D_interval_GNUPLOT outputs',
            'flowstar_endpoint_box_available_for_gnuplot_rows': False,
            'verified_all_completed_gnuplot_rows_endpoint_box_available_false': endpoint_false_ok,
            'current_torch_vs_flowstar_ratio_types': ['last_segment', 'tube'],
            'endpoint_ratio_allowed': False,
            'reason_endpoint_ratio_disallowed': 'Flow* GNUPLOT-derived boxes are last_segment/tube boxes, not endpoint boxes.',
        },
        'runtime_semantics': {
            'flowstar_runtime_s_source': 'FLOWSTAR_RUNTIME_S internal reach time when present',
            'flowstar_internal_reach_column': 'flowstar_internal_reach_s',
            'wall_compile_column': 'flowstar_wall_compile_s',
            'wall_run_column': 'flowstar_wall_run_s',
            'wall_total_column': 'flowstar_wall_total_s',
            'compile_run_wall_times_separate': True,
        },
        'representative_generated_artifacts': collect_representatives(flow_rows),
        'claim_boundaries': {
            'plant_only': True,
            'fixed_step_fixed_order': True,
            'flowstar_adaptive_baseline': False,
            'full_crown_reach_pipeline': False,
            'raw_taylor_model_coefficient_comparison': False,
            'new_algorithm_added': False,
            'excluded_features': ['CROWN', 'auto_LiRPA', 'Jacobian bounds', 'sin/cos', 'hybrid automata', 'Flow* core binding'],
        },
    }


def write_manifest_md(manifest: dict[str, Any]) -> str:
    lines = [
        '# Flow* Provenance Manifest',
        '',
        'Scope: plant-only fixed-step/fixed-order comparison against the `chenxin415/flowstar` C++ toolbox/static library backend. This is not a Python Flow* package workflow, not `Flow*_adaptive`, not a full CROWN-Reach NNCS pipeline, and not a raw Taylor-model coefficient comparison.',
        '',
        '## torch_tm_flowpipe',
        '',
    ]
    for k, v in manifest['torch_tm_flowpipe'].items():
        if isinstance(v, str) and '\n' in v:
            lines.extend([f'- `{k}`:', '', '```text', *(v.splitlines() or ['<clean>']), '```'])
        else:
            lines.append(f'- `{k}`: `{v}`')
    lines.extend(['', '## Flow* Backend', ''])
    backend = manifest['flowstar_backend']
    lines.extend([
        f'- `FLOWSTAR_ROOT`: `{backend["FLOWSTAR_ROOT"]}`',
        '- backend: `toolbox_cpp`',
        f'- Flow* HEAD: `{backend["head_sha"]}`',
        f'- `libflowstar.a`: `{backend["libflowstar_a"]["path"]}`',
        f'- `libflowstar.a sha256`: `{backend["libflowstar_a"]["sha256"]}`',
        f'- compiler: `{backend["compiler_version"][0] if backend["compiler_version"] else ""}`',
        '',
        'Flow* remotes:',
        '',
        '```text',
        *backend['remote_v'],
        '```',
        '',
        'Flow* `git status --short`:',
        '',
        '```text',
        *(backend['status_short'] or ['<clean>']),
        '```',
        '',
        '## Generated C++ Audit',
        '',
    ])
    audit = manifest['generated_cpp_audit']
    lines.extend([
        f'- Generated C++ cases from CSV: `{audit["generated_cpp_case_count_from_csv"]}`',
        f'- Existing C++ case files: `{audit["existing_cpp_case_count"]}`',
        f'- All cases include `Continuous.h`: `{audit["all_generated_cases_include_Continuous_h"]}`',
        f'- All cases call `ode.reach(...)`: `{audit["all_generated_cases_call_ode_reach"]}`',
        f'- All cases use fixed step/order via `setFixedStepsize(h, order)`: `{audit["all_generated_cases_use_setFixedStepsize"]}`',
        '- Runner links Flow* static library: `True` (`-L $FLOWSTAR_ROOT/flowstar-toolbox -lflowstar`)',
        '',
        '## Result Counts',
        '',
        '| status | count |',
        '| --- | ---: |',
    ])
    for status, count in manifest['result_counts']['status_counts'].items():
        lines.append(f'| {status} | {count} |')
    lines.extend([
        '',
        '## Box, Ratio, And Runtime Semantics',
        '',
        '- Box source: GNUPLOT interval plot files emitted by `plot_2D_interval_GNUPLOT`.',
        '- Flow* GNUPLOT-derived rows have `endpoint_box_available=false`; endpoint ratios are not allowed.',
        '- Current torch-vs-Flow* ratios are `last_segment` and `tube` only.',
        '- Flow* `runtime_s` uses `FLOWSTAR_RUNTIME_S` / internal reach time when present.',
        '- Compile/run wall times are separate: `flowstar_wall_compile_s`, `flowstar_wall_run_s`, `flowstar_wall_total_s`.',
        '',
        '## Representative Generated Artifacts',
        '',
        '| label | setting | h | steps | order | status | cpp sha256 | stdout sha256 | stderr sha256 | plot paths |',
        '| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |',
    ])
    for rep in manifest['representative_generated_artifacts']:
        plots = '<br>'.join(f"{p['path']} ({p['sha256']})" for p in rep.get('plots', []) or [] if p and p.get('exists'))
        lines.append('| ' + ' | '.join(md_escape(x) for x in [
            rep.get('label'), rep.get('setting_label'), rep.get('h'), rep.get('steps'), rep.get('order'), rep.get('status'),
            (rep.get('cpp') or {}).get('sha256'), (rep.get('stdout') or {}).get('sha256'), (rep.get('stderr') or {}).get('sha256'), plots,
        ]) + ' |')
    lines.extend([
        '',
        '## Claim Boundaries',
        '',
        '- Plant-only polynomial ODE reachability only.',
        '- Fixed-step/fixed-order baseline only; this does not represent Flow* adaptive or best-tuned performance.',
        '- No endpoint ratio is reported because Flow* endpoint boxes are unavailable from GNUPLOT artifacts.',
        '- No raw Taylor-model coefficient comparison.',
        '- No CROWN, auto_LiRPA, Jacobian bounds, sin/cos, hybrid automata, Flow* core binding, or full CROWN-Reach NNCS pipeline reproduction.',
    ])
    return '\n'.join(lines) + '\n'


def write_parameter_table(flow_rows: list[dict[str, str]]) -> str:
    lines = [
        '# Flow* Parameter Equivalence Audit',
        '',
        'This table documents the plant-only fixed-step/fixed-order baseline used by the existing Flow* sweep. It is not `Flow*_adaptive`, not a best-tuned adaptive Flow* result, and not a full CROWN-Reach NNCS pipeline reproduction.',
        '',
        '| system | order | h | steps | horizon | setting_label | ODE | initial box | requested order | Flow* remainder estimation | Flow* cutoff | Flow* adaptive step/order disabled | torch mode | compared box types | runtime source |',
        '| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |',
    ]
    ode = 'dx/dt = y; dy/dt = y - x - x^2*y'
    initial = 'x in [1.1, 1.4]; y in [2.35, 2.45]'
    ordered = sorted(flow_rows, key=lambda x: (x.get('system', ''), int(float(x.get('order') or 0)), float(x.get('h') or 0), int(float(x.get('steps') or 0)), x.get('setting_label', '')))
    for row in ordered:
        rem, cut = parse_setting(row.get('setting_label', ''))
        h = float(row.get('h') or 0.0)
        steps = int(float(row.get('steps') or 0))
        values = [
            row.get('system'), row.get('order'), row.get('h'), row.get('steps'), f'{h * steps:.8g}', row.get('setting_label'),
            ode, initial, row.get('order'), rem, cut, 'yes: toolbox_cpp uses setting.setFixedStepsize(h, order)',
            'range_only; dependency_preserving', 'last_segment; tube; endpoint unavailable',
            'Flow*: FLOWSTAR_RUNTIME_S/internal reach time; torch: Python algorithm wall time',
        ]
        lines.append('| ' + ' | '.join(md_escape(v) for v in values) + ' |')
    return '\n'.join(lines) + '\n'


def main() -> None:
    flow_rows = load_flow_rows()
    manifest = build_manifest(flow_rows)
    (ROOT / 'outputs' / 'flowstar_provenance_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (ROOT / 'outputs' / 'flowstar_provenance_manifest.md').write_text(write_manifest_md(manifest), encoding='utf-8')
    (ROOT / 'outputs' / 'flowstar_parameter_equivalence_table.md').write_text(write_parameter_table(flow_rows), encoding='utf-8')
    print('wrote outputs/flowstar_provenance_manifest.json')
    print('wrote outputs/flowstar_provenance_manifest.md')
    print('wrote outputs/flowstar_parameter_equivalence_table.md')
    print('generated_cpp_cases', manifest['generated_cpp_audit']['generated_cpp_case_count_from_csv'])
    print('status_counts', manifest['result_counts']['status_counts'])


if __name__ == '__main__':
    main()
