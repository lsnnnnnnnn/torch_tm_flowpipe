"""New actual-model CUDA gate called once on the first new control invocation."""
from fractions import Fraction
from itertools import product


def check(torch, iv, affine, component, hull, T):
    if hull.device.type != 'cuda' or hull.shape != (225,4,2):
        raise ValueError('actual new DP225 CUDA input required')
    # Three actual initial domains; this is a fresh mathematical gate inside
    # the new run, not an invocation of any saved experiment or checker.
    h, t = hull[:3].clone(), T[:3].clone()
    result = component.evaluate(h,t)
    ref = affine(h,t,component.layers)
    for name, expected in [('original_network_interval',ref[0]),('original_residual_interval',ref[1])]:
        if not torch.equal(result[name].contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8)):
            raise AssertionError('deferred original arithmetic differs on actual CUDA model')
    model = [([[Fraction(float(v)) for v in row] for row in w.cpu().tolist()],
              [Fraction(float(v)) for v in b.cpu().tolist()]) for w,b in component.layers]
    cpu_h, cpu_t = h.cpu().tolist(),t.cpu().tolist()
    fields = ('network_interval','original_network_interval','residual_interval',
              'original_residual_interval','intersected_residual_interval')
    bounds = {name:result[name].cpu().tolist() for name in fields}
    outputs = 0
    for lane in range(3):
        lo,hi = [[Fraction(float(x[edge])) for x in cpu_h[lane]] for edge in (0,1)]
        points = [[hi[k] if corner[k] else lo[k] for k in range(4)] for corner in product((0,1),repeat=4)]
        points += [[(lo[k]+hi[k])/2 for k in range(4)],[(3*lo[k]+hi[k])/4 for k in range(4)]]
        for point in points:
            value = point
            for index,(w,b) in enumerate(model):
                value = [sum((value[i]*w[i][j] for i in range(len(value))),b[j]) for j in range(len(b))]
                if index<2:value = [max(Fraction(0),v) for v in value]
            for out in range(2):
                residual = value[out]-sum((Fraction(float(cpu_t[lane][out][k]))*point[k] for k in range(4)),Fraction(0))
                for name in fields:
                    expected = residual if 'residual' in name else value[out]
                    low,high=map(Fraction,bounds[name][lane][out])
                    if not low<=expected<=high:raise AssertionError('actual-model exact Fraction escaped '+name)
                outputs += 1
    saved=result['intersected_residual_interval'].clone()
    component.evaluate(h,t*1.03125)
    if not torch.equal(saved.contiguous().view(torch.uint8),result['intersected_residual_interval'].contiguous().view(torch.uint8)):
        raise AssertionError('prior owned residual overwritten')
    return dict(status='PASSED_NEW_DP_DELAYED_RELU_V2_ACTUAL_MODEL_CUDA_GATE',
        actual_model=True,device=str(hull.device),batch=3,model_shape=[4,25,25,2],
        exact_fraction_outputs=outputs,interval_containment_checks=outputs*5,
        original_arithmetic_bytes_equal=True,prior_output_owned=True,
        source_input='first 3 lanes of actual new production first control domain/T',
        gate_component_separate_from_production=True,old_checker_executed=False,
        end_to_end_strict_certificate=False)
