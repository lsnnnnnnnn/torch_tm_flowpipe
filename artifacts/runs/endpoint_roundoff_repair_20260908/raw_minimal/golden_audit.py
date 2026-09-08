import importlib.util,json,subprocess
from pathlib import Path
import torch
from torch_tm_flowpipe import *
from torch_tm_flowpipe.ode_examples import van_der_pol_ode
from experiments.endpoint_roundoff_repair.local_oracles import assert_model_contains

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
root=Path.cwd()
sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert sha=='196a50e9131336d68df07ad0af353deca0092d19'
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
def proof(segment):
    assert segment.status=='validated'
    return [assert_model_contains(a,b,segment.h) for a,b in zip(segment.tm,segment.endpoint_raw_tm)]
state=FlowstarNormalFlowpipeState.from_exact_decimal_box([('1.1','1.4'),('2.35','2.45')],4)
current=state.normalized_initial_tm(4)
rows=[]
for i in range(2):
    seg=flowpipe_step_flowstar_style_adaptive(van_der_pol_ode,current,h=.01,h_min=.01,h_max=.01,
        order=4,target_remainder_radius=1e-4,cutoff_threshold=1e-10,
        reset_mode=NORMALIZED_INSERTION_DEPENDENCY_PRESERVING,flowstar_normal_state=state)
    check=proof(seg)
    rows.append({'step':i+1,'full_function_exact_check':check,
                 'reset_sha256':tmvector_hashes(seg.reset_tm)['tmvector_sha256'],
                 'right_sha256':tmvector_hashes(seg.flowstar_normal_state.tmv_right)['tmvector_sha256']})
    current,state=seg.reset_tm,seg.flowstar_normal_state
spec=importlib.util.spec_from_file_location('gate_audit',root/'experiments/audit_s1_boundary_drift.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
original=audit.runner._run_lane_step
checks=[]
def checked(*args,**kwargs):
    seg=original(*args,**kwargs)
    if seg.status=='validated':checks.append(proof(seg))
    return seg
audit.runner._run_lane_step=checked
schedule=json.loads((root/'outputs/s1_prefix_integrated_complete_o4_20260810/20260810T095423Z/04_frozen_schedule_prefix/frozen_schedule.json').read_text())
gate=audit.replay_control('C3',schedule,max_attempt_index=12)
assert gate['status']=='domain_gate_failure' and gate['failure_boundary']==3
result={'scientific_sha':sha,'source_clean':True,'c2_two_steps':rows,'c3_no_renormalization':gate,
        'c3_accepted_endpoint_fraction_checks':checks,'reason':'Required endpoint coefficient error changes consumed right maps and normalized domain margins; strict gate remains unchanged.',
        'old_no_renormalization_failure_boundary':11,'new_no_renormalization_failure_boundary':3}
out=root.parent/'evidence/raw_minimal/affected_golden_audit.json'
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'rows':rows,'boundary':gate['failure_boundary'],'states':len(gate['states']),'checks':len(checks)}))
