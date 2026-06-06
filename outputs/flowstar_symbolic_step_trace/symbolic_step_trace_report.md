# Symbolic Step Trace Report

Source run: `flowstar_style_o6_candidate8_output6_insert_symqueue`.
Failed segment index: `37` with h_try=`0.00234375`.
PyTorch rejection reason: `initial or cutoff remainder exceeds target remainder`.

## Answers

Is PyTorch rejecting because propagated symbolic width is counted as ordinary target remainder? yes.
Is it rejecting because insertion cutoff/truncation width is counted too early? no direct evidence in this trace; the dominant trigger is the materialized propagated symbolic channel unless the component table says otherwise.
Which component Flow* likely keeps symbolic? `propagated_symbolic` is treated as symbolic/output materialization under the local source map when it is propagated queue width.
If removed from ordinary target check, does the remaining ordinary remainder fit target? yes.
Exact implementation change: keep propagated queue width out of the ordinary seed-remainder precheck, keep carrying it in the symbolic queue, and materialize it into reported range/output boxes.

## Components

| component | width_x | width_y | width_sum | target_sum | Flow* channel | old PyTorch channel | exceeds? |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| ordinary_initial_remainder | 0.00019908269804765267 | 0.000289723330474081 | 0.0004888060285217336 | 0.0004000000000000001 | ordinary seed only | ordinary target precheck | yes |
| cutoff_uncertainty |  |  | 1e-323 | 0.0004000000000000001 | J_ip1 symbolic/current insertion | ordinary target precheck via reset remainder | no |
| insertion_truncation |  |  | 1.8793901660490429e-06 | 0.0004000000000000001 | J_ip1 symbolic/current insertion | ordinary target precheck via reset remainder | no |
| propagated_symbolic | 0.00019908269804765264 | 0.00028972333047408097 | 0.0004888060285217336 | 0.0004000000000000001 | symbolic queue J_i | ordinary reset remainder | yes |
| materialized_symbolic | 0.00019908269804765267 | 0.000289723330474081 | 0.0004888060285217336 | 0.0004000000000000001 | output/range materialization | ordinary reset remainder | yes |
| new_symbolic | 0.015597624657322865 | 0.03789158698085342 | 0.053489211638176284 | 0.0004000000000000001 | queued J_ip1 | queued for future propagation | yes |
| target_remainder | 0.00020000000000000004 | 0.00020000000000000004 | 0.0004000000000000001 | 0.0004000000000000001 | ordinary Picard target | ordinary Picard target | no |

This is a local comparator over PyTorch artifacts. It does not copy Flow* code and uses the source map document for channel classification.
