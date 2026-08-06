# Xiangru complete-Q3 source callgraph

```text
run_s0_tora_static_partition_sweep.main
  -> _run_lane(method="complete_q3")                         [run_s0:327-345]
     -> complete_total_degree_support(3, variables=6)        [generic_fixed_basis:94-108]
     -> StaticRouteTables.build(support)                     [run_s0:345]
     -> step_closed_loop                                     [run_s0:456-462]
        -> TensorFixedBasisKernel.run_tora_remainder_picard   [tensor_fixed_basis:838-940]
           -> polynomial_rhs / integrate, exactly twice      [tensor_fixed_basis:745-772,847-858]
           -> seed interval remainder                        [tensor_fixed_basis:860-873]
           -> tm_rhs / integrate and 10 DR-RP rounds         [tensor_fixed_basis:874-927]
           -> evaluate full tube and endpoint separately     [tensor_fixed_basis:928-940]
        -> property margins 2 - abs(bound)                    [tensor_closed_loop:521-522]
     -> boundary_for_controller / worker.control every 10 h  [run_s0:407-451]
```

Variable slot 0 is local time. Slots 1-5 parameterize `x1,x2,x3,x4,u1`; the complete degree-3 support therefore contains `C(6+3,3)=84` monomials.
