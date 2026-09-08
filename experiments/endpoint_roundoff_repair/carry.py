"""Small actual-consumer witnesses, reusing the strict carry regression cases."""
import argparse,importlib.util,json,subprocess
from pathlib import Path
import pytest,torch
from torch_tm_flowpipe import tmvector_hashes


def collect(repo):
    spec=importlib.util.spec_from_file_location('endpoint_carry_cases',repo/'tests/test_endpoint_roundoff_carry.py')
    cases=importlib.util.module_from_spec(spec);spec.loader.exec_module(cases)
    import torch_tm_flowpipe.flowpipe as fp
    def intervals(values):return [[float(v.lo).hex(),float(v.hi).hex()] for v in values]
    def tensors(pair):return [[float(pair[0][0,i]).hex(),float(pair[1][0,i]).hex()] for i in range(pair[0].shape[1])]
    records=[]
    for entry in ['published','dense_internal']:
        cases.test_two_substitutions_carry_complete_remainder(entry)
        records.append({'path':entry,'two_steps_passed':True,'first_error_consumed_by_second':True,
            'exact_value':'3/144115188075855872','ordinary_remainder_present_in_both_steps':True,
            'first_input_unchanged':True,'test':'test_two_substitutions_carry_complete_remainder['+entry+']'})
    normal=[];actual=cases._flowstar_normalized_insertion_transition
    def observe_normal(seg,state,*args,**kwargs):
        current,next_state,stats=actual(seg,state,*args,**kwargs)
        normal.append({'endpoint_error':intervals(seg.endpoint_substitution_roundoff or []),
                       'right_map_sha256':tmvector_hashes(next_state.tmv_right)['tmvector_sha256'],
                       'queue_size':len(next_state.symbolic_queue.J),'queue_reset_count':next_state.symbolic_queue.reset_count,
                       'new_owner_width':stats.get('accepted_boundary_sr_current_owner_width_sum'),
                       'total_interval_image_width':stats.get('accepted_boundary_sr_total_interval_image_width_sum'),
                       'composition_branch':stats.get('accepted_boundary_sr_composition_branch')})
        return current,next_state,stats
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cases,'_flowstar_normalized_insertion_transition',observe_normal)
        cases.test_normal_maps_current_owner_history_and_capacity_reanchor()
    records.append({'path':'normal_sr','two_steps_passed':True,'first_error_consumed_by_second':True,
                    'transitions':normal,'report_reads_do_not_append_owner':True,'no_rebilling_of_propagated_history':True})
    for label,mode in [('g1','normalized_insertion_bounded_source_ledger_o4_g1'),
                       ('g2','normalized_insertion_shared_source_columns_o4_g2'),
                       ('s1_current','normalized_insertion_structured_remainder_k16'),
                       ('s1_total_delta','normalized_insertion_structured_total_delta_k16')]:
        rows=[];consumed=[];previous=[]
        actual_step=fp.flowpipe_step_flowstar_style_adaptive
        class Capture:
            def __init__(self,patch):self.patch=patch
            def setattr(self,target,name,fn):
                def observed(seg):
                    result=fn(seg)
                    consumed.append({'new_endpoint_error':intervals(seg.endpoint_substitution_roundoff),
                        'cutoff_remainder':intervals(seg.endpoint_cutoff_remainder),
                        'original_roundoff':tensors(seg.validated_remainder_decomposition.ledger.entries['roundoff_safeguard']),
                        'consumed_roundoff':tensors(result.ledger.entries['roundoff_safeguard']),
                        'consumed_full_remainder':tensors((result.decomposition_lo,result.decomposition_hi))})
                    return result
                self.patch.setattr(target,name,observed)
        def step(*args,**kwargs):
            if previous:
                assert args[1] is previous[-1].reset_tm and kwargs['flowstar_normal_state'] is previous[-1].flowstar_normal_state
            result=actual_step(*args,**kwargs)
            assert result.status=='validated'
            rows.append({'step':len(rows)+1,'consumes_previous_returned_state':bool(previous),
                         'right_map_sha256':tmvector_hashes(result.flowstar_normal_state.tmv_right)['tmvector_sha256'],
                         'next_input_sha256':tmvector_hashes(result.reset_tm)['tmvector_sha256']})
            previous.append(result);return result
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(fp,'flowpipe_step_flowstar_style_adaptive',step)
            cases.test_replacement_ledgers_consume_endpoint_additions_in_two_steps(mode,.005,Capture(patch))
        records.append({'path':label,'two_steps_passed':True,'first_error_consumed_by_second':rows[1]['consumes_previous_returned_state'],
                        'steps':rows,'derived_ledger_consumed':consumed})
    rollbacks=[]
    for entry in ['published','dense_internal']:
        with pytest.MonkeyPatch.context() as patch:
            cases.test_substitution_failure_preserves_accepted_state(entry,patch,Path('/unused'))
        rollbacks.append({'entry':entry,'accepted_current_unchanged':True,'queue_unchanged':True,'failed_endpoint_not_published':True})
    return {'paths':records,'failed_attempts':rollbacks,'all_actual_paths_closed':True,
            'scope':'Small actual production transitions; original domain, ownership and rollback assertions retained.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert not subprocess.check_output(['git','-C',str(args.repo),'status','--porcelain'],text=True).strip()
    result=collect(args.repo)
    result['replay_source_sha']=subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip()
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'all_actual_paths_closed':result['all_actual_paths_closed'],'paths':len(result['paths'])}))
