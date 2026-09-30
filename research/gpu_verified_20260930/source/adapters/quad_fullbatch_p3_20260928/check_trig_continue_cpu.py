"""CPU-only actual40 protocol, command preservation and cold-source diff checks."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import sys

HERE = Path(__file__).parent

def function_source(path, name):
    source = path.read_text(); node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'


def main(a):
    spec = importlib.util.spec_from_file_location('trig_continue_CPU_checked', HERE/'run_fullbatch_p3_trig_continue.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert m.sha(HERE/'run_fullbatch_p3.py') == '22f4dda16dcfd30563e6652ec8af82034b3b138a7505dbf31714a8bd55820a72'
    original = function_source(HERE/'run_fullbatch_p3.py', 'cold')
    candidate = function_source(HERE/'run_fullbatch_p3_trig_continue.py', 'cold')
    restored = candidate.replace("binding=ctx.trig_source_binding;source_identity=ctx.trig_source_identity\n    assert source.resolve()==Path(binding['source40_path']).resolve()", 'binding=own40_gate(source,ctx.identity)')
    restored = restored.replace('metadata=source_identity,host=x.h', 'metadata=ctx.identity,host=x.h')
    restored = restored.replace('compare_snapshot(x,source/name,x.args.output/name,source_identity)', 'compare_snapshot(x,source/name,x.args.output/name)')
    restored = restored.replace("compare_artifact(x,source/(name+'.pt'),x.args.output/(name+'.pt'),source_identity)", "compare_artifact(x,source/(name+'.pt'),x.args.output/(name+'.pt'))")
    assert restored == original
    assert ast.dump(ast.parse(restored), include_attributes=False) == ast.dump(ast.parse(original), include_attributes=False)
    assert 'bounded_prefix_completed' not in (HERE/'run_fullbatch_p3_trig_continue.py').read_text()
    mappings = {}
    for mode, filename, entry in [('cold20_to21','full1024_p3_phase_profile_gpu14_40_v1_COMMAND.json','run_fullbatch_p3_phase_profile_gpu14.py'), ('long1000','full1024_p3_gpu14_1000_v1_COMMAND.json','run_fullbatch_p3_gpu14.py')]:
        args, command = m.arguments(SimpleNamespace(mode=mode,base_command=HERE/filename,output=a.output/('unused_'+mode),candidate40=a.candidate40))
        argv = command['argv']; index = next(i for i,v in enumerate(argv) if Path(v).name == entry); tail=argv[index+1:]
        expected = {flag[2:].replace('-','_'):value for flag,value in zip(tail[::2],tail[1::2])}
        for key,value in expected.items():
            if key in ('output','mode') or (mode=='cold20_to21' and key=='source40'):continue
            assert str(getattr(args,key)) == value
        assert args.mode == mode
        if mode=='cold20_to21':assert args.source40 == a.candidate40
        else:assert str(args.source40) == expected['source40'] and str(args.cold) == expected['cold']
        mappings[mode] = dict(copied_fields=len(expected),base_command_sha256=m.sha(HERE/filename),all_base_numeric_paths_preserved=True,historical_base_source40_cold_preserved=mode=='long1000')
    for filename in ['INPUT.json','RESULT.json','TRIG_REUSE_RUN.json','REFERENCE40_COMPARISON.json']:
        assert m.sha(a.candidate40/filename) == m.SOURCE_FILES[filename]
    values = [m.read(a.candidate40/n) for n in ['INPUT.json','RESULT.json','TRIG_REUSE_RUN.json','REFERENCE40_COMPARISON.json']]
    profile = m.read(a.baseline_input)['source_identity']; short = profile['algorithm_identity']
    actual = m.source40_fields(*values, short, profile)
    rejected = []
    mutations = [
        ('legacy_status', lambda v:v[1].update(status='bounded_prefix_completed')),
        ('wrong_main', lambda v:v[1].update(script_sha256='0'*64)),
        ('wrong_wrapper', lambda v:v[2].update(wrapper_sha256='0'*64)),
        ('wrong_candidate', lambda v:v[2].update(candidate_sha256='0'*64)),
        ('failed_receipt', lambda v:v[2].update(status='failed')),
        ('missing_accepted', lambda v:v[1].update(accepted_lane_steps=40959)),
        ('missing_NN', lambda v:v[1].update(controller_calls=1)),
        ('zero_reuse', lambda v:v[2]['candidate_python_counters'].update(reuse_hits=0)),
        ('wrong_plan', lambda v:v[2]['candidate_last_plan'].update(unique_direct_plans=7)),
        ('failed_width_pairing', lambda v:v[3].update(observer_all_tensor_bytes_equal=False)),
        ('wrong_component_count', lambda v:v[3]['snapshots']['committed_20'].update(component_count=13)),
    ]
    for label, mutate in mutations:
        v=copy.deepcopy(values);mutate(v)
        try:m.source40_fields(*v, short, profile)
        except AssertionError:rejected.append(label)
        else:raise AssertionError('Invalid protocol admitted: '+label)
    for label, bad_short, bad_profile in [('wrong_short_ancestry',dict(short,policy='wrong'),profile),('wrong_profile_ancestry',short,dict(profile,policy='wrong'))]:
        try:m.source40_fields(*values,bad_short,bad_profile)
        except AssertionError:rejected.append(label)
        else:raise AssertionError('Invalid ancestry admitted: '+label)
    assert not any(n=='flowstar_gpu' or n.startswith('flowstar_gpu.') for n in sys.modules)
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(status='passed_CPU_actual40_protocol_argument_and_cold_source_checks',script_sha256=m.sha(__file__),entry_sha256=m.sha(HERE/'run_fullbatch_p3_trig_continue.py'),
        exact_cold_body_equal_after_only_declared_identity_boundary_changes=True,source_diff_sha256=m.sha(HERE/'trig_continue_cold_sequence.diff'),
        command_mappings=mappings,actual40_files_sha256={k:v for k,v in m.SOURCE_FILES.items() if k!='phase_profile.jsonl'},actual40_identity_sha256=m.canonical(actual),
        actual40_protocol_passed=True,rejected=rejected,CUDA_executed=False,base_initialize_executed=False,cold_numerical_run_executed=False,long_numerical_run_executed=False,
        scope='Actual downloaded40 protocol/ancestry and preserved commands; frozen cold numerical source exact after four boundary edits. Not cold replay or long qualification.')
    m.write(a.output/'RESULT.json',result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate40',type=Path,required=True);p.add_argument('--baseline-input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
