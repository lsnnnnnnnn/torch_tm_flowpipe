"""Independent directed DP ReLU residual with delayed error concretization.

Deferred v2: one numerical host validity read per successful subbox call.
The unchanged tensor body uses the existing Huan GraphCache only at C64/14400.
No CROWN certificate, solver or model importer is used.
The fixed 4-25-25-2 model has point binary64 weights in (input,output) order.
"""


class DelayedReLUResidual:
    def __init__(self, torch, iv, layers, *, qa_mode=False):
        if type(qa_mode) is not bool:raise ValueError('explicit residual graph QA mode required')
        self._graph_qa_mode = qa_mode
        self._residual_graph = None
        self._graph_reference_records = []
        self.torch, self.iv = torch, iv
        expected = (((4, 25), (25,)), ((25, 25), (25,)), ((25, 2), (2,)))
        if len(layers) != 3:
            raise ValueError('the fixed DP 4-25-25-2 network is required')
        self.device = layers[0][0].device
        for pair, shapes in zip(layers, expected):
            if len(pair) != 2:
                raise ValueError('weight/bias pair required')
            for value, shape in zip(pair, shapes):
                if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                        or value.dtype != torch.float64 or value.device != self.device
                        or not bool(torch.isfinite(value).all())):
                    raise ValueError('finite point binary64 DP model parameters required')
        self.layers = tuple(tuple(x.detach().clone() for x in pair) for pair in layers)
        self._owners = tuple(x for pair in self.layers for x in pair)
        self._versions = tuple(self._identity(x) for x in self._owners)
        self.counters = dict(calls=0, completed=0, refused=0, host_validity_checks=0,
                             original_interval_predicates=0, new_interval_predicates=0,
                             anchor_predicates=0, alpha_predicates=0)

    @staticmethod
    def _identity(value):
        return (id(value), value.data_ptr(), value._version, tuple(value.shape),
                tuple(value.stride()), value.dtype, value.device)

    def _guard(self, hull, T):
        torch = self.torch
        current = tuple(x for pair in self.layers for x in pair)
        if (len(current) != len(self._owners)
                or any(a is not b for a,b in zip(current,self._owners))
                or tuple(self._identity(x) for x in self._owners) != self._versions):
            raise RuntimeError('owned model parameters changed')
        if (not isinstance(hull, torch.Tensor) or hull.dtype != torch.float64
                or hull.device != self.device or hull.ndim != 3
                or hull.shape[1:] != (4, 2) or hull.shape[0] < 1
                or not isinstance(T, torch.Tensor) or T.dtype != torch.float64
                or T.device != self.device or T.shape != (hull.shape[0], 2, 4)):
            raise ValueError('matching binary64 hull[B,4,2] and point T[B,2,4] required')


    def _tensor_body(self, hull, T):
        """Return two independent same-domain residuals and their intersection.

        Existing forward arithmetic and every original finite/order/alpha
        predicate are retained to bound each hidden preactivation. Additional
        delayed arithmetic keeps all coefficient roundoff as intervals.
        """
        torch, iv = self.torch, self.iv
        flags, labels = [], []
        def valid(value, label):
            flags.append(torch.stack((~torch.isnan(value).any(), ~torch.isinf(value).any(),
                                      ~(value[..., 0] > value[..., 1]).any())))
            labels.append(label)
        valid(hull, 'DP delayed residual physical domain')
        valid(iv.from_point(T), 'DP finite point anchor')
        B = hull.shape[0]
        with torch.no_grad():
            A = iv.from_point(torch.eye(4, dtype=torch.float64, device=self.device).expand(B, -1, -1))
            b = iv.from_point(torch.zeros((B, 4), dtype=torch.float64, device=self.device))
            R = b.clone()
            slopes, errors, preactivations = [], [], []
            original_interval_predicates, alpha_predicates = 1, 0
            for index, (weight, bias) in enumerate(self.layers):
                A = iv.dot_point_iv(weight.T[None, :, None, :], A.permute(0, 2, 1, 3).unsqueeze(1), dim=-1)
                b = iv.add(iv.dot_point_iv(weight.T.unsqueeze(0), b.unsqueeze(1), dim=-1),
                           iv.from_point(bias).unsqueeze(0))
                R = iv.dot_point_iv(weight.T.unsqueeze(0), R.unsqueeze(1), dim=-1)
                for name, value in (('A', A), ('b', b), ('R', R)):
                    valid(value, f'DP original affine layer {index+1} {name}')
                    original_interval_predicates += 1
                if index < 2:
                    pre = iv.add(iv.add(iv.sum(iv.mul(A, hull.unsqueeze(1)), dim=-1), b), R)
                    valid(pre, f'DP original preactivation {index+1}')
                    original_interval_predicates += 1
                    low, high = pre[..., 0], pre[..., 1]
                    inactive, active = high <= 0, low >= 0
                    crossing = ~(inactive | active)
                    denom = torch.where(crossing, high-low, torch.ones_like(high))
                    alpha = torch.where(crossing, high/denom, active.to(torch.float64))
                    flags.append(torch.stack((torch.isfinite(alpha).all(),
                        ((alpha >= 0) & (alpha <= 1)).all(),
                        torch.ones((), dtype=torch.bool, device=hull.device))))
                    labels.append(f'DP original ReLU slope {index+1}')
                    alpha_predicates += 1
                    neg_alpha_low = iv.neg(iv.mul_point(iv.from_point(low), alpha))
                    one_minus_alpha = iv.sub(iv.from_point(torch.ones_like(alpha)), iv.from_point(alpha))
                    one_minus_alpha_high = iv.mul(one_minus_alpha, iv.from_point(high))
                    error_upper = torch.maximum(neg_alpha_low[..., 1], one_minus_alpha_high[..., 1])
                    error_upper = torch.maximum(error_upper, torch.zeros_like(error_upper))
                    e = torch.stack((torch.zeros_like(error_upper), error_upper), -1)
                    scaled_A = iv.mul_point(A, alpha.unsqueeze(-1))
                    scaled_b = iv.mul_point(b, alpha)
                    scaled_R = iv.add(iv.mul_point(R, alpha), e)
                    A = torch.where(active[..., None, None], A,
                                    torch.where(inactive[..., None, None], torch.zeros_like(A), scaled_A))
                    b = torch.where(active[..., None], b,
                                    torch.where(inactive[..., None], torch.zeros_like(b), scaled_b))
                    R = torch.where(active[..., None], R,
                                    torch.where(inactive[..., None], torch.zeros_like(R), scaled_R))
                    for name, value in (('A', A), ('b', b), ('R', R)):
                        valid(value, f'DP original post-ReLU {index+1} {name}')
                        original_interval_predicates += 1
                    # Stable active/inactive ReLUs have exact zero error.
                    # At {0}, the original priority chooses alpha=1, also exact.
                    error = torch.where(crossing[..., None], e, torch.zeros_like(e))
                    valid(error, f'DP retained ReLU error {index+1}')
                    slopes.append(alpha)
                    errors.append(error)
                    preactivations.append(pre)
            original_network = iv.add(iv.add(iv.sum(iv.mul(A, hull.unsqueeze(1)), dim=-1), b), R)
            delta_A = iv.sub(A, iv.from_point(T))
            original_residual = iv.add(iv.add(iv.sum(iv.mul(delta_A, hull.unsqueeze(1)), dim=-1), b), R)
            for name, value in (('network', original_network), ('residual', original_residual)):
                valid(value, 'DP original '+name)
                original_interval_predicates += 1

            # W1:[4,25], W2:[25,25], W3:[25,2] (transposes in proof).
            # C1 carries one first-layer error across BOTH downstream layers.
            (W1, b1), (W2, b2), (W3, b3) = self.layers
            D1, D2 = slopes
            C2D2 = iv.mul_point(iv.from_point(W3.T).unsqueeze(0), D2.unsqueeze(1))
            C1 = iv.dot_point_iv(W2[None, None, :, :], C2D2[:, :, None, :, :], dim=-1)
            C1D1 = iv.mul_point(C1, D1.unsqueeze(1))
            delayed_A = iv.dot_point_iv(W1[None, None, :, :], C1D1[:, :, None, :, :], dim=-1)
            term_b1 = iv.dot_point_iv(b1[None, None, :], C1D1, dim=-1)
            term_b2 = iv.dot_point_iv(b2[None, None, :], C2D2, dim=-1)
            delayed_b = iv.add(iv.add(term_b1, term_b2), iv.from_point(b3).unsqueeze(0))
            first_error = iv.sum(iv.mul(C1, errors[0].unsqueeze(1)), dim=-1)
            second_error = iv.dot_point_iv(W3.T.unsqueeze(0), errors[1].unsqueeze(1), dim=-1)
            delayed_R = iv.add(first_error, second_error)
            delayed_delta = iv.sub(delayed_A, iv.from_point(T))
            delayed_network = iv.add(iv.add(iv.sum(iv.mul(delayed_A, hull.unsqueeze(1)), dim=-1), delayed_b), delayed_R)
            delayed_residual = iv.add(iv.add(iv.sum(iv.mul(delayed_delta, hull.unsqueeze(1)), dim=-1), delayed_b), delayed_R)
            intermediates = dict(C2D2=C2D2, C1=C1, C1D1=C1D1, A=delayed_A,
                term_b1=term_b1, term_b2=term_b2, b=delayed_b, first_error=first_error,
                second_error=second_error, R=delayed_R, delta_A=delayed_delta,
                network=delayed_network, residual=delayed_residual)
            for name, value in intermediates.items():
                valid(value, 'DP delayed '+name)
            intersection = torch.stack((torch.maximum(original_residual[..., 0], delayed_residual[..., 0]),
                                        torch.minimum(original_residual[..., 1], delayed_residual[..., 1])), -1)
            valid(intersection, 'DP same-domain same-T residual intersection')
            if B == 14400 and not self._graph_qa_mode:
                # Only this tensor is consumed by the C64 partition union.
                # The surrounding graph dispatch gives it one owned clone.
                return dict(intersected_residual_interval=intersection,
                    diagnostics=dict(B=B, original_interval_predicates=original_interval_predicates,
                    alpha_predicates=alpha_predicates, new_interval_predicates=len(intermediates)+3,
                    first_relu_errors_preserved_until_output=True,
                    same_domain_and_point_T=True, coefficient_roundoff_retained=True,
                    external_nn_certificate_used=False, solver_integrated=False,
                    independent_end_to_end_nncs_certificate=False)), torch.stack(flags), labels
            result = dict(network_interval=delayed_network.clone(), residual_interval=delayed_residual.clone(),
                original_network_interval=original_network.clone(), original_residual_interval=original_residual.clone(),
                intersected_residual_interval=intersection.clone(), certified_domain=hull.clone(), point_T=T.clone(),
                slopes=[x.clone() for x in slopes], relu_errors=[x.clone() for x in errors],
                preactivation_intervals=[x.clone() for x in preactivations],
                diagnostics=dict(B=B, original_interval_predicates=original_interval_predicates,
                    alpha_predicates=alpha_predicates, new_interval_predicates=len(intermediates)+3,
                    first_relu_errors_preserved_until_output=True,
                    same_domain_and_point_T=True, coefficient_roundoff_retained=True,
                    external_nn_certificate_used=False, solver_integrated=False,
                    independent_end_to_end_nncs_certificate=False))

            return result, torch.stack(flags), labels


    def _owned_graph_result(self, result):
        # Every tensor leaf must survive the next C64 replay of this one key.
        return {name: dict(value) if name == 'diagnostics' else
                [item.clone() for item in value] if isinstance(value, list) else value.clone()
                for name, value in result.items()}

    def _dispatch_tensor_body(self, hull, T):
        if not hull.is_cuda or hull.shape[0] != 14400:
            return self._tensor_body(hull, T)
        if self._residual_graph is None:
            from flowstar_gpu.graphing import GraphCache
            self._residual_graph = GraphCache(str(hull.device))
            if not self._residual_graph.enabled:
                raise RuntimeError('existing Huan graph-compatible kernels required for C64 residual')
        result, flags, labels = self._residual_graph.run(
            ('delayed_residual_C64_14400',), self._tensor_body, [hull, T],
            keepalive=(self.layers, self._owners, self.iv))
        result = self._owned_graph_result(result)
        if self._graph_qa_mode and not self._graph_reference_records:
            expected, expected_flags, expected_labels = self._tensor_body(hull, T)
            if result.keys() != expected.keys() or result['diagnostics'] != expected['diagnostics']:
                raise AssertionError('C64 delayed graph result diagnostics differ')
            tensor_fields = 0
            for name, value in result.items():
                if name == 'diagnostics':
                    continue
                reference = expected[name]
                values = value if isinstance(value, list) else [value]
                references = reference if isinstance(reference, list) else [reference]
                if len(values) != len(references):
                    raise AssertionError('C64 delayed graph tensor list differs: ' + name)
                for actual, wanted in zip(values, references):
                    if (actual.shape != wanted.shape or actual.dtype != wanted.dtype
                            or not self.torch.equal(actual.contiguous().view(self.torch.uint8),
                                                    wanted.contiguous().view(self.torch.uint8))):
                        raise AssertionError('C64 delayed graph tensor bytes differ: ' + name)
                    tensor_fields += 1
            if labels != expected_labels or not self.torch.equal(flags, expected_flags):
                raise AssertionError('C64 delayed graph all39 validity flags differ')
            self._graph_reference_records.append(dict(batch=14400, tensor_leaves=tensor_fields,
                all_tensor_bytes_equal=True, all39_flags_equal=True, diagnostic_dict_equal=True,
                actual_inputs='same live hull and T', development_QA_only=True))
        return result, flags, labels

    def graph_statistics(self):
        cache = self._residual_graph
        captured = next(iter(cache._segs.values()))[2][0] if cache is not None and cache._segs else {}
        leaves = [item for name,value in captured.items() if name != 'diagnostics'
                  for item in (value if isinstance(value,list) else [value])]
        return dict(policy='unchanged_delayed39_tensor_body_Huan_GraphCache_C64',
            captures=0 if cache is None else cache.captures,
            replays=0 if cache is None else cache.hits,
            cached_shapes=0 if cache is None else len(cache._segs),
            fixed_batch=14400, mutable_inputs=['hull', 'T'],
            weights_strong_keepalive=True, recursive_owned_tensor_outputs=True,
            diagnostic_dict_copied=True, successful_host_validity_reads_per_evaluate=1,
            output_keys=list(captured),output_tensor_leaves=len(leaves),
            output_tensor_bytes=sum(value.numel()*value.element_size() for value in leaves),
            qa_mode=self._graph_qa_mode, reference_records=list(self._graph_reference_records))

    def evaluate(self, hull, T):
        """Release owned results only after all original and new flags pass."""
        self._guard(hull, T)
        self.counters['calls'] += 1
        result, flags, labels = self._dispatch_tensor_body(hull, T)
        if len(labels) != 39:
            raise RuntimeError('expected 20 original + 16 new + 1 anchor + 2 alpha predicates')
        for key, count in [('original_interval_predicates',20), ('new_interval_predicates',16),
                           ('anchor_predicates',1), ('alpha_predicates',2)]:
            self.counters[key] += count
        self.counters['host_validity_checks'] += 1
        if not bool(flags.all()):
            self.counters['refused'] += 1
            # Diagnostic synchronization is allowed only after refusal. No
            # intermediate or result is returned on the failure path.
            rows = flags.detach().cpu().tolist()
            first = next(i for i, row in enumerate(rows) if not all(row))
            raise FloatingPointError('DP delayed deferred validity rejected '+labels[first])
        self.counters['completed'] += 1
        result['diagnostics'].update(deferred_numeric_validity=True,
            successful_host_validity_reads=1, total_predicate_groups=39,
            original_interval_predicates=20, new_interval_predicates=16, anchor_predicates=1)
        return result
