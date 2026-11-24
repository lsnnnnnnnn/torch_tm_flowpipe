"""256 original cells/C64, unchanged whole diagnostics, delayed same-T subboxes."""
from pathlib import Path
from types import SimpleNamespace
import time
from residual import DelayedReLUResidual
from gate import check


def install(torch, module, iv, batch_functions, *, save_gate, qa_mode=False):
    if type(qa_mode) is not bool:raise ValueError('controller qa_mode must be an explicit bool')
    if torch.__version__.split('+')[0] != '2.5.1':
        raise RuntimeError('preloaded Torch 2.5.1 required')
    original, affine = module.partitioned_affine_residual_interval,module.affine_residual_interval
    if original.__globals__ is not vars(module) or affine.__globals__ is not vars(module):
        raise RuntimeError('unwrapped original DP controller required')
    active=armed=restored=False
    core=owners=identities=None
    references=[]
    counts=dict.fromkeys(('calls','completed','failed','whole_calls','batched_subbox_calls',
        'subboxes_evaluated','subbox_lane_rows','peak_batch_rows','live_reference_checks',
        'partition_batch_assertions'),0)
    def identity(x):
        return (id(x),x.data_ptr(),x._version,tuple(x.shape),tuple(x.stride()),x.dtype,x.device)
    def guard(layers):
        current=tuple(x for pair in layers for x in pair)
        if len(current)!=len(owners) or any(a is not b for a,b in zip(current,owners)) or tuple(map(identity,current))!=identities:
            raise RuntimeError('original DP model identity/storage/version changed')
    def candidate(hull,T,layers,parts=2):
        nonlocal active,core,owners,identities
        if active or restored or (qa_mode and not armed):raise RuntimeError('inactive or reentered delayed DP binding')
        active=True;counts['calls']+=1
        try:
            if core is None:
                owners=tuple(x for pair in layers for x in pair);identities=tuple(map(identity,owners))
                if qa_mode:
                    gate_component=DelayedReLUResidual(torch,iv,layers)
                    torch.cuda.synchronize(hull.device)
                    gate_started=time.perf_counter()
                    gate_receipt=check(torch,iv,affine,gate_component,hull,T)
                    torch.cuda.synchronize(hull.device)
                    gate_receipt['wall_s']=time.perf_counter()-gate_started
                    gate_receipt['timing_included_in_driver']=True
                    save_gate(gate_receipt)
                core=DelayedReLUResidual(torch,iv,layers,qa_mode=qa_mode)
            guard(layers)
            call=0
            def dispatch(h,t,current_layers):
                nonlocal call
                call+=1
                guard(current_layers)
                if call==1:return affine(h,t,current_layers)
                result=core.evaluate(h,t)
                if qa_mode and not references:
                    torch.cuda.synchronize(h.device)
                    started=time.perf_counter()
                    expected=affine(h,t,current_layers)
                    torch.cuda.synchronize(h.device)
                    elapsed=time.perf_counter()-started
                    for name,value in [('original_network_interval',expected[0]),('original_residual_interval',expected[1])]:
                        if not torch.equal(result[name].contiguous().view(torch.uint8),value.contiguous().view(torch.uint8)):
                            raise AssertionError('first actual C64 deferred original arithmetic differs')
                    if not bool(iv.contains(result['original_residual_interval'],result['intersected_residual_interval']).all()):
                        raise AssertionError('new same-T intersection left old residual')
                    references.append(dict(batch=h.shape[0],production_B=hull.shape[0],parts=parts,
                        original_forward_tensor_bytes_equal=True,same_domain_same_T_intersection=True,
                        reference_wall_s=elapsed,whole_public_diagnostics_from_original=True,
                        reference_timing_scope='fresh actual first14400 original affine; included in driver'))
                    counts['live_reference_checks']+=1
                return (result['original_network_interval'] if qa_mode else None),result['intersected_residual_interval'],None
            result=batch_functions.evaluate(torch,iv,dispatch,hull,T,layers,parts=parts,counts=counts)
            if call!=5:raise RuntimeError('one whole + four C64 calls required')
            counts['completed']+=1
            return result
        except BaseException:
            counts['failed']+=1
            raise
        finally:active=False
    def arm_live_reference():
        nonlocal armed
        if active or armed or restored:raise RuntimeError('cannot arm current binding')
        armed=True
    def restore():
        nonlocal restored
        if restored:return
        if active or module.partitioned_affine_residual_interval is not candidate:raise RuntimeError('binding owner changed')
        module.partitioned_affine_residual_interval=original;restored=True
    module.partitioned_affine_residual_interval=candidate
    return SimpleNamespace(counters=counts,reference_records=references,arm_live_reference=arm_live_reference,
        restore=restore,core_statistics=lambda:dict(core.counters) if core is not None else {},
        graph_statistics=lambda:core.graph_statistics() if core is not None else {},
        receipt=dict(qa_mode=qa_mode,development_checks_separated_from_production=True,
            policy='dp_delayed_relu_v2_same_domain_same_T_intersection',source=str(Path(__file__).resolve()),
            partition_source=str(Path(batch_functions.__file__).resolve()),controller_source=module.__file__,
            residual_graph_dispatch=True,cell_chunk=64,subboxes_per_input=256,maximum_enlarged_batch=14400,
            whole_box_original_function_and_public_diagnostics=True,original_256_hull_order=True,
            original_interval_groups=20,new_interval_groups=16,anchor_groups=1,alpha_groups=2,
            successful_host_validity_reads_per_subbox_batch=1,
            model_identity_storage_version_guard=True,new_numerical_algorithm=True,
            intersection_before_strict_injection_same_NN_minus_Tx=True,
            Pu_recentered_by_original_strict_injection=True,different_Pu_remainders_never_intersected=True,
            end_to_end_strict_certificate=False))
