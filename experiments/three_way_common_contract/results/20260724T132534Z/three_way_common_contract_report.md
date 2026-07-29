# Three-way low-order reachability comparison under common external contracts

## Result status

**All strict correctness gates passed.**

| gate | checks | violations | passed |
| --- | --- | --- | --- |
| all_analytic_intervals_contained | 11954 | 0 | True |
| configuration_coverage | 64 | 0 | True |
| deterministic_high_accuracy_trajectory_sanity | 407646 | 0 | True |
| diffreach_real_upstream_operations | 26 | 0 | True |
| endpoint_vs_whole_tube_extraction | 8023 | 0 | True |
| flowstar_order_and_extraction_workaround | 60 | 0 | True |
| harmonic_one_step_exact_containment | 22 | 0 | True |
| identical_initial_boxes_and_state_order | 327 | 0 | True |
| identical_ode_point_evaluations | 45 | 0 | True |
| native_validation_status_consistency | 16219 | 0 | True |
| riccati_one_step_exact_containment | 11 | 0 | True |

Deterministic trajectory checks are sanity checks only; they do not establish soundness. Native validation and analytic containment are reported independently.

## Direct answers

1. **Literal common internal order:** No. Torch order 1, the DiffReach affine flag, and Flow* fixed order 2 have different retained support and validation semantics.
2. **Identical one-step input:** Torch gives the smallest valid Riccati widths at every tested `h`; DiffReach gives the smallest valid harmonic and Van der Pol widths at every tested `h`/state. Flow* has no accepted row at some larger steps, and those configurations are reported as `validation_failed` rather than ranked.
3. **Common componentwise-box carry:** Torch is tightest at the Riccati checkpoints. DiffReach is tightest at every harmonic and Van der Pol common checkpoint/state where the primary tools can be compared. Flow* fails before harmonic `t=4`, Van der Pol `t=0.08` at `h=0.005`, and the first Van der Pol step at `h=0.01`.
4. **Van der Pol validation horizon:** The farthest primary tool depends on protocol and `h`, as summarized immediately below. The supplemental default DiffReach quasi-quadratic native variant reaches the requested `T=1` for both step sizes.

| protocol | h | farthest_primary_tool(s) | failure_horizon_or_censor |
| --- | --- | --- | --- |
| multi_step_common_box_carry | 0.005 | DiffReach, Torch TM | 0.63 |
| multi_step_common_box_carry | 0.01 | Torch TM | 0.6 |
| native_low_order | 0.005 | DiffReach | 0.62 |
| native_low_order | 0.01 | DiffReach | 0.54 |

5. **Throughput after one-time work:** There is no system-independent winner. Flow* is fastest on scalar Riccati at roughly `0.07 ms/step`; DiffReach is typically about `0.1–0.3 ms/step` after JIT and is faster than Flow* on the two-state harmonic runs; Torch eager execution is roughly `8–24 ms/step`. These steady rates exclude Flow* builds (about `1.7 s` per generated executable) and DiffReach JIT (about `0.4–1.6 s` per configuration).
6. **Native versus controlled carry:** Native carry materially changes widths and horizons because each tool retains its own dependencies. Most notably, the supplemental default DiffReach quasi-quadratic variant reaches Van der Pol `T=1`, while the primary affine flag and all common-box primary runs fail earlier.

## Tool identity and semantics

- Torch repository: `/srv/local/shengenli/torch_tm_flowpipe_three_way_comparison` at `c12de37d5ada53fdab17b98ab9526f48c6cc31c4`.
- DiffReach repository: `/srv/local/shengenli/DiffReach` at `dd628eb443b517d6415de93e7035b4baef73963e`.
- Flow* repository/static library: `/srv/local/shengenli/flowstar` at `b85a3211748cb77b736fe4ad42ee02d8d2b81148`.

The tools cannot be compared under a literal common internal order. Torch retains complete total-degree order 1, DiffReach's affine flag has transient restricted quasi-quadratic support before its final projection, and Flow* rejects fixed order 1 and runs at fixed order 2.

| tool | tool_variant | protocol | local_order | local_retained_basis | carried_representation | reset_policy | validator |
| --- | --- | --- | --- | --- | --- | --- | --- |
| torch_tm_flowpipe | complete_total_degree_order_1 | one_step_common_input | 1 | complete_total_degree_1(local_time,state_generators) | none_one_segment | not_applicable | torch_native_picard_growth |
| torch_tm_flowpipe | complete_total_degree_order_1 | multi_step_common_box_carry | 1 | complete_total_degree_1(local_time,state_generators) | componentwise_axis_aligned_box | endpoint_box_exact_no_inflation | torch_native_picard_growth |
| torch_tm_flowpipe | complete_total_degree_order_1 | native_low_order | 1 | complete_total_degree_1(local_time,state_generators) | dependency_preserving_taylor_model | none_native_dependency_carry | torch_native_picard_growth |
| diffreach | affine_flag | one_step_common_input | affine_flag_with_transient_quasi_quadratic | stock_affine_flag_final_{1,t,z};transient_{t^2,t*z} | none_one_segment | not_applicable | DiffReach_upstream_remainder_picard_initial_contraction |
| diffreach | affine_flag | multi_step_common_box_carry | affine_flag_with_transient_quasi_quadratic | stock_affine_flag_final_{1,t,z};transient_{t^2,t*z} | componentwise_axis_aligned_box | endpoint_box_exact_no_inflation | DiffReach_upstream_remainder_picard_initial_contraction |
| diffreach | affine_flag | native_low_order | affine_flag_with_transient_quasi_quadratic | stock_affine_flag_final_{1,t,z};transient_{t^2,t*z} | upstream_normalized_affine_symbolic_carry | upstream_symbolic_linear_normalization | DiffReach_upstream_remainder_picard_initial_contraction |
| diffreach | default_restricted_quasi_quadratic | native_low_order | restricted_quasi_quadratic | stock_restricted_quasi_quadratic_{1,t,z,t^2,t*z} | upstream_restricted_quasi_quadratic_symbolic_carry | upstream_symbolic_linear_normalization | DiffReach_upstream_remainder_picard_initial_contraction |
| flowstar | minimum_supported_fixed_order_2 | one_step_common_input | 2 | complete_total_degree_2(local_time,normalized_generators) | none_one_segment | not_applicable | Flowstar_public_advance_initial_Picard_candidate_inclusion |
| flowstar | minimum_supported_fixed_order_2 | multi_step_common_box_carry | 2 | complete_total_degree_2(local_time,normalized_generators) | componentwise_axis_aligned_box | endpoint_box_exact_no_inflation | Flowstar_public_advance_initial_Picard_candidate_inclusion |
| flowstar | minimum_supported_fixed_order_2 | native_low_order | 2 | complete_total_degree_2(local_time,normalized_generators) | native_Flowstar_Taylor_model_flowpipe | native_QR_normalized_Taylor_model_carry | Flowstar_public_advance_initial_Picard_candidate_inclusion |

## Proof of real upstream DiffReach execution

The primary adapter calls `src.reachability.CT_Dyn_Reach.step_once` from `/srv/local/shengenli/DiffReach/src/reachability.py` at source line 127. The callable identity gate is `True` and the full run recorded 24 upstream JAX trace invocations.

Picard validation resolves to `src.picard.remainder_picard` in `/srv/local/shengenli/DiffReach/src/picard.py`; Taylor-model operations resolve to `src.taylor_model.QuadTM` in `/srv/local/shengenli/DiffReach/src/taylor_model.py`. The optional `jax_verify` shim is fail-fast and only satisfies imports for unused neural-bound paths. The external DiffReach repository was not modified.

## Protocol A: identical one-step input

| tool | system | h | state_name | status | lower | upper | width | exact_inflation_ratio | native_validation_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torch_tm_flowpipe | riccati | 0.005 | x | validated | -6.25276e-09 | 0.10005 | 0.10005 | 1 | validated |
| torch_tm_flowpipe | riccati | 0.01 | x | validated | -2.50151e-08 | 0.1001 | 0.1001 | 1 | validated |
| torch_tm_flowpipe | riccati | 0.02 | x | validated | -1.00114e-07 | 0.100201 | 0.100201 | 1.00001 | validated |
| torch_tm_flowpipe | riccati | 0.05 | x | validated | -6.26759e-07 | 0.100506 | 0.100506 | 1.00004 | validated |
| torch_tm_flowpipe | harmonic | 0.005 | x1 | validated | -0.100503 | 0.100503 | 0.201006 | 1.00004 | validated |
| torch_tm_flowpipe | harmonic | 0.005 | x2 | validated | -0.100503 | 0.100503 | 0.201006 | 1.00004 | validated |
| torch_tm_flowpipe | harmonic | 0.01 | x1 | validated | -0.101013 | 0.101013 | 0.202025 | 1.00017 | validated |
| torch_tm_flowpipe | harmonic | 0.01 | x2 | validated | -0.101013 | 0.101013 | 0.202025 | 1.00017 | validated |
| torch_tm_flowpipe | harmonic | 0.02 | x1 | validated | -0.10205 | 0.10205 | 0.2041 | 1.00069 | validated |
| torch_tm_flowpipe | harmonic | 0.02 | x2 | validated | -0.10205 | 0.10205 | 0.2041 | 1.00069 | validated |
| torch_tm_flowpipe | harmonic | 0.05 | x1 | validated | -0.105313 | 0.105313 | 0.210625 | 1.00419 | validated |
| torch_tm_flowpipe | harmonic | 0.05 | x2 | validated | -0.105313 | 0.105313 | 0.210625 | 1.00419 | validated |
| torch_tm_flowpipe | van_der_pol | 0.0025 | x1 | validated | 1.10583 | 1.40615 | 0.300318 |  | validated |
| torch_tm_flowpipe | van_der_pol | 0.0025 | x2 | validated | 2.34015 | 2.45338 | 0.11323 |  | validated |
| torch_tm_flowpipe | van_der_pol | 0.005 | x1 | validated | 1.11157 | 1.41234 | 0.30077 |  | validated |
| torch_tm_flowpipe | van_der_pol | 0.005 | x2 | validated | 2.32984 | 2.45675 | 0.126912 |  | validated |
| torch_tm_flowpipe | van_der_pol | 0.01 | x1 | validated | 1.12277 | 1.42485 | 0.302082 |  | validated |
| torch_tm_flowpipe | van_der_pol | 0.01 | x2 | validated | 2.30785 | 2.4635 | 0.155653 |  | validated |
| torch_tm_flowpipe | van_der_pol | 0.02 | x1 | validated | 1.14409 | 1.45041 | 0.306326 |  | validated |
| torch_tm_flowpipe | van_der_pol | 0.02 | x2 | validated | 2.25824 | 2.477 | 0.218759 |  | validated |
| diffreach | riccati | 0.005 | x | validated | -1.25188e-05 | 0.10005 | 0.100063 | 1.00013 | validated |
| diffreach | riccati | 0.01 | x | validated | -2.50751e-05 | 0.1001 | 0.100125 | 1.00025 | validated |
| diffreach | riccati | 0.02 | x | validated | -5.03011e-05 | 0.100201 | 0.100251 | 1.00051 | validated |
| diffreach | riccati | 0.05 | x | validated | -0.000126892 | 0.100505 | 0.100632 | 1.00128 | validated |
| diffreach | harmonic | 0.005 | x1 | validated | -0.100503 | 0.100503 | 0.201005 | 1.00004 | validated |
| diffreach | harmonic | 0.005 | x2 | validated | -0.100503 | 0.100503 | 0.201005 | 1.00004 | validated |
| diffreach | harmonic | 0.01 | x1 | validated | -0.10101 | 0.10101 | 0.20202 | 1.00015 | validated |
| diffreach | harmonic | 0.01 | x2 | validated | -0.10101 | 0.10101 | 0.20202 | 1.00015 | validated |
| diffreach | harmonic | 0.02 | x1 | validated | -0.102041 | 0.102041 | 0.204082 | 1.0006 | validated |
| diffreach | harmonic | 0.02 | x2 | validated | -0.102041 | 0.102041 | 0.204082 | 1.0006 | validated |
| diffreach | harmonic | 0.05 | x1 | validated | -0.105263 | 0.105263 | 0.210526 | 1.00372 | validated |
| diffreach | harmonic | 0.05 | x2 | validated | -0.105263 | 0.105263 | 0.210526 | 1.00372 | validated |
| diffreach | van_der_pol | 0.0025 | x1 | validated | 1.10586 | 1.40613 | 0.300273 |  | validated |
| diffreach | van_der_pol | 0.0025 | x2 | validated | 2.34054 | 2.44627 | 0.105729 |  | validated |
| diffreach | van_der_pol | 0.005 | x1 | validated | 1.11169 | 1.41228 | 0.300591 |  | validated |
| diffreach | van_der_pol | 0.005 | x2 | validated | 2.33093 | 2.44261 | 0.111678 |  | validated |
| diffreach | van_der_pol | 0.01 | x1 | validated | 1.12324 | 1.42461 | 0.301377 |  | validated |
| diffreach | van_der_pol | 0.01 | x2 | validated | 2.31121 | 2.43549 | 0.124283 |  | validated |
| diffreach | van_der_pol | 0.02 | x1 | validated | 1.14588 | 1.44949 | 0.303604 |  | validated |
| diffreach | van_der_pol | 0.02 | x2 | validated | 2.26967 | 2.42233 | 0.152665 |  | validated |
| flowstar | riccati | 0.005 | x | validated | -0.000112497 | 0.100138 | 0.10025 | 1.002 | validated |
| flowstar | riccati | 0.01 | x | validated | -0.000124987 | 0.100175 | 0.1003 | 1.002 | validated |
| flowstar | riccati | 0.02 | x | validated | -0.00014995 | 0.10025 | 0.1004 | 1.00199 | validated |
| flowstar | riccati | 0.05 | x | validation_failed |  |  |  |  | failed |
| flowstar | harmonic | 0.005 | x1 | validated | -0.1006 | 0.1006 | 0.2012 | 1.00101 | validated |
| flowstar | harmonic | 0.005 | x2 | validated | -0.1006 | 0.1006 | 0.2012 | 1.00101 | validated |
| flowstar | harmonic | 0.01 | x1 | validated | -0.1011 | 0.1011 | 0.2022 | 1.00104 | validated |
| flowstar | harmonic | 0.01 | x2 | validated | -0.1011 | 0.1011 | 0.2022 | 1.00104 | validated |
| flowstar | harmonic | 0.02 | x1 | validated | -0.1021 | 0.1021 | 0.2042 | 1.00118 | validated |
| flowstar | harmonic | 0.02 | x2 | validated | -0.1021 | 0.1021 | 0.2042 | 1.00118 | validated |
| flowstar | harmonic | 0.05 | x1 | validation_failed |  |  |  |  | failed |
| flowstar | harmonic | 0.05 | x2 | validation_failed |  |  |  |  | failed |
| flowstar | van_der_pol | 0.0025 | x1 | validated | 1.10487 | 1.40712 | 0.30225 |  | validated |
| flowstar | van_der_pol | 0.0025 | x2 | validated | 2.3399 | 2.44701 | 0.107109 |  | validated |
| flowstar | van_der_pol | 0.005 | x1 | validated | 1.11072 | 1.41322 | 0.3025 |  | validated |
| flowstar | van_der_pol | 0.005 | x2 | validated | 2.3307 | 2.44292 | 0.112219 |  | validated |
| flowstar | van_der_pol | 0.01 | x1 | validation_failed |  |  |  |  | failed |
| flowstar | van_der_pol | 0.01 | x2 | validation_failed |  |  |  |  | failed |
| flowstar | van_der_pol | 0.02 | x1 | validation_failed |  |  |  |  | failed |
| flowstar | van_der_pol | 0.02 | x2 | validation_failed |  |  |  |  | failed |

Tightest valid endpoint by configuration/state:

| system | h | state | tightest_valid_tool | width |
| --- | --- | --- | --- | --- |
| harmonic | 0.005 | x1 | DiffReach | 0.201005 |
| harmonic | 0.005 | x2 | DiffReach | 0.201005 |
| harmonic | 0.01 | x1 | DiffReach | 0.20202 |
| harmonic | 0.01 | x2 | DiffReach | 0.20202 |
| harmonic | 0.02 | x1 | DiffReach | 0.204082 |
| harmonic | 0.02 | x2 | DiffReach | 0.204082 |
| harmonic | 0.05 | x1 | DiffReach | 0.210526 |
| harmonic | 0.05 | x2 | DiffReach | 0.210526 |
| riccati | 0.005 | x | Torch TM | 0.10005 |
| riccati | 0.01 | x | Torch TM | 0.1001 |
| riccati | 0.02 | x | Torch TM | 0.100201 |
| riccati | 0.05 | x | Torch TM | 0.100506 |
| van_der_pol | 0.0025 | x1 | DiffReach | 0.300273 |
| van_der_pol | 0.0025 | x2 | DiffReach | 0.105729 |
| van_der_pol | 0.005 | x1 | DiffReach | 0.300591 |
| van_der_pol | 0.005 | x2 | DiffReach | 0.111678 |
| van_der_pol | 0.01 | x1 | DiffReach | 0.301377 |
| van_der_pol | 0.01 | x2 | DiffReach | 0.124283 |
| van_der_pol | 0.02 | x1 | DiffReach | 0.303604 |
| van_der_pol | 0.02 | x2 | DiffReach | 0.152665 |

## Protocol B: common componentwise-box carry

Widths below are compared only at the same absolute checkpoint. A `validation_failed` entry contains no substituted earlier width.

| tool | system | h | checkpoint | state_name | status | lower | upper | width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torch_tm_flowpipe | riccati | 0.01 | 0.1 | x | validated | -2.53576e-07 | 0.101011 | 0.101012 |
| torch_tm_flowpipe | riccati | 0.01 | 0.5 | x | validated | -1.3493e-06 | 0.10527 | 0.105272 |
| torch_tm_flowpipe | riccati | 0.01 | 1 | x | validated | -2.92993e-06 | 0.111127 | 0.11113 |
| torch_tm_flowpipe | harmonic | 0.01 | 1 | x1 | validated | -0.27385 | 0.27385 | 0.547699 |
| torch_tm_flowpipe | harmonic | 0.01 | 1 | x2 | validated | -0.27385 | 0.27385 | 0.547699 |
| torch_tm_flowpipe | harmonic | 0.01 | 2 | x1 | validated | -0.749936 | 0.749936 | 1.49987 |
| torch_tm_flowpipe | harmonic | 0.01 | 2 | x2 | validated | -0.749936 | 0.749936 | 1.49987 |
| torch_tm_flowpipe | harmonic | 0.01 | 4 | x1 | validated | -5.62403 | 5.62403 | 11.2481 |
| torch_tm_flowpipe | harmonic | 0.01 | 4 | x2 | validated | -5.62403 | 5.62403 | 11.2481 |
| torch_tm_flowpipe | van_der_pol | 0.005 | 0.02 | x1 | validated | 1.14563 | 1.44956 | 0.303924 |
| torch_tm_flowpipe | van_der_pol | 0.005 | 0.02 | x2 | validated | 2.26529 | 2.47686 | 0.211565 |
| torch_tm_flowpipe | van_der_pol | 0.005 | 0.05 | x1 | validated | 1.21058 | 1.52491 | 0.314329 |
| torch_tm_flowpipe | van_der_pol | 0.005 | 0.05 | x2 | validated | 2.11695 | 2.51647 | 0.399522 |
| torch_tm_flowpipe | van_der_pol | 0.005 | 0.08 | x1 | validated | 1.27062 | 1.60147 | 0.330851 |
| torch_tm_flowpipe | van_der_pol | 0.005 | 0.08 | x2 | validated | 1.94086 | 2.55537 | 0.614508 |
| torch_tm_flowpipe | van_der_pol | 0.01 | 0.02 | x1 | validated | 1.1451 | 1.44985 | 0.304749 |
| torch_tm_flowpipe | van_der_pol | 0.01 | 0.02 | x2 | validated | 2.26291 | 2.47691 | 0.213994 |
| torch_tm_flowpipe | van_der_pol | 0.01 | 0.05 | x1 | validated | 1.2091 | 1.52565 | 0.316551 |
| torch_tm_flowpipe | van_der_pol | 0.01 | 0.05 | x2 | validated | 2.11052 | 2.51661 | 0.40609 |
| torch_tm_flowpipe | van_der_pol | 0.01 | 0.08 | x1 | validated | 1.268 | 1.60268 | 0.334681 |
| torch_tm_flowpipe | van_der_pol | 0.01 | 0.08 | x2 | validated | 1.92975 | 2.55563 | 0.625876 |
| diffreach | riccati | 0.01 | 0.1 | x | validated | -0.000253605 | 0.101011 | 0.101265 |
| diffreach | riccati | 0.01 | 0.5 | x | validated | -0.00133488 | 0.105268 | 0.106603 |
| diffreach | riccati | 0.01 | 1 | x | validated | -0.00285448 | 0.111123 | 0.113977 |
| diffreach | harmonic | 0.01 | 1 | x1 | validated | -0.2732 | 0.2732 | 0.5464 |
| diffreach | harmonic | 0.01 | 1 | x2 | validated | -0.2732 | 0.2732 | 0.5464 |
| diffreach | harmonic | 0.01 | 2 | x1 | validated | -0.746382 | 0.746382 | 1.49276 |
| diffreach | harmonic | 0.01 | 2 | x2 | validated | -0.746382 | 0.746382 | 1.49276 |
| diffreach | harmonic | 0.01 | 4 | x1 | validated | -5.57086 | 5.57086 | 11.1417 |
| diffreach | harmonic | 0.01 | 4 | x2 | validated | -5.57086 | 5.57086 | 11.1417 |
| diffreach | van_der_pol | 0.005 | 0.02 | x1 | validated | 1.14616 | 1.44889 | 0.302728 |
| diffreach | van_der_pol | 0.005 | 0.02 | x2 | validated | 2.27102 | 2.41858 | 0.147559 |
| diffreach | van_der_pol | 0.005 | 0.05 | x1 | validated | 1.21225 | 1.52095 | 0.308704 |
| diffreach | van_der_pol | 0.005 | 0.05 | x2 | validated | 2.13933 | 2.36286 | 0.223534 |
| diffreach | van_der_pol | 0.005 | 0.08 | x1 | validated | 1.27415 | 1.59125 | 0.3171 |
| diffreach | van_der_pol | 0.005 | 0.08 | x2 | validated | 1.99239 | 2.29845 | 0.306052 |
| diffreach | van_der_pol | 0.01 | 0.02 | x1 | validated | 1.14608 | 1.44909 | 0.30301 |
| diffreach | van_der_pol | 0.01 | 0.02 | x2 | validated | 2.2706 | 2.41978 | 0.149179 |
| diffreach | van_der_pol | 0.01 | 0.05 | x1 | validated | 1.21203 | 1.52154 | 0.309513 |
| diffreach | van_der_pol | 0.01 | 0.05 | x2 | validated | 2.13802 | 2.36597 | 0.227945 |
| diffreach | van_der_pol | 0.01 | 0.08 | x1 | validated | 1.27374 | 1.59232 | 0.318578 |
| diffreach | van_der_pol | 0.01 | 0.08 | x2 | validated | 1.98985 | 2.30362 | 0.313772 |
| flowstar | riccati | 0.01 | 0.1 | x | validated | -0.00125666 | 0.101759 | 0.103016 |
| flowstar | riccati | 0.01 | 0.5 | x | validated | -0.00643951 | 0.109004 | 0.115443 |
| flowstar | riccati | 0.01 | 1 | x | validated | -0.0132954 | 0.118559 | 0.131854 |
| flowstar | harmonic | 0.01 | 1 | x1 | validated | -0.28753 | 0.28753 | 0.575059 |
| flowstar | harmonic | 0.01 | 1 | x2 | validated | -0.28753 | 0.28753 | 0.575059 |
| flowstar | harmonic | 0.01 | 2 | x1 | validated | -0.794762 | 0.794762 | 1.58952 |
| flowstar | harmonic | 0.01 | 2 | x2 | validated | -0.794762 | 0.794762 | 1.58952 |
| flowstar | harmonic | 0.01 | 4 | x1 | validation_failed |  |  |  |
| flowstar | harmonic | 0.01 | 4 | x2 | validation_failed |  |  |  |
| flowstar | van_der_pol | 0.005 | 0.02 | x1 | validated | 1.14228 | 1.45264 | 0.310368 |
| flowstar | van_der_pol | 0.005 | 0.02 | x2 | validated | 2.27033 | 2.41962 | 0.149295 |
| flowstar | van_der_pol | 0.005 | 0.05 | x1 | validated | 1.20255 | 1.53034 | 0.327788 |
| flowstar | van_der_pol | 0.005 | 0.05 | x2 | validated | 2.13911 | 2.36405 | 0.224945 |
| flowstar | van_der_pol | 0.005 | 0.08 | x1 | validation_failed |  |  |  |
| flowstar | van_der_pol | 0.005 | 0.08 | x2 | validation_failed |  |  |  |
| flowstar | van_der_pol | 0.01 | 0.02 | x1 | validation_failed |  |  |  |
| flowstar | van_der_pol | 0.01 | 0.02 | x2 | validation_failed |  |  |  |
| flowstar | van_der_pol | 0.01 | 0.05 | x1 | validation_failed |  |  |  |
| flowstar | van_der_pol | 0.01 | 0.05 | x2 | validation_failed |  |  |  |
| flowstar | van_der_pol | 0.01 | 0.08 | x1 | validation_failed |  |  |  |
| flowstar | van_der_pol | 0.01 | 0.08 | x2 | validation_failed |  |  |  |

Smallest valid common-time width by configuration/state:

| system | h | time | state | smallest_valid_width_tool | width |
| --- | --- | --- | --- | --- | --- |
| harmonic | 0.01 | 1 | x1 | DiffReach | 0.5464 |
| harmonic | 0.01 | 1 | x2 | DiffReach | 0.5464 |
| harmonic | 0.01 | 2 | x1 | DiffReach | 1.49276 |
| harmonic | 0.01 | 2 | x2 | DiffReach | 1.49276 |
| harmonic | 0.01 | 4 | x1 | DiffReach | 11.1417 |
| harmonic | 0.01 | 4 | x2 | DiffReach | 11.1417 |
| riccati | 0.01 | 0.1 | x | Torch TM | 0.101012 |
| riccati | 0.01 | 0.5 | x | Torch TM | 0.105272 |
| riccati | 0.01 | 1 | x | Torch TM | 0.11113 |
| van_der_pol | 0.005 | 0.02 | x1 | DiffReach | 0.302728 |
| van_der_pol | 0.005 | 0.02 | x2 | DiffReach | 0.147559 |
| van_der_pol | 0.005 | 0.05 | x1 | DiffReach | 0.308704 |
| van_der_pol | 0.005 | 0.05 | x2 | DiffReach | 0.223534 |
| van_der_pol | 0.005 | 0.08 | x1 | DiffReach | 0.3171 |
| van_der_pol | 0.005 | 0.08 | x2 | DiffReach | 0.306052 |
| van_der_pol | 0.01 | 0.02 | x1 | DiffReach | 0.30301 |
| van_der_pol | 0.01 | 0.02 | x2 | DiffReach | 0.149179 |
| van_der_pol | 0.01 | 0.05 | x1 | DiffReach | 0.309513 |
| van_der_pol | 0.01 | 0.05 | x2 | DiffReach | 0.227945 |
| van_der_pol | 0.01 | 0.08 | x1 | DiffReach | 0.318578 |
| van_der_pol | 0.01 | 0.08 | x2 | DiffReach | 0.313772 |

## Failure horizons and each method's own final valid step

| tool | protocol | system | h | run_status | first_failure_time | final_valid_time | width_at_own_final_valid_step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| torch_tm_flowpipe | multi_step_common_box_carry | riccati | 0.01 | success |  | 1 | 0.11113 |
| torch_tm_flowpipe | multi_step_common_box_carry | harmonic | 0.01 | success |  | 4 | 11.2481 |
| torch_tm_flowpipe | multi_step_common_box_carry | van_der_pol | 0.005 | validation_failed | 0.63 | 0.625 | 14.2607 |
| torch_tm_flowpipe | multi_step_common_box_carry | van_der_pol | 0.01 | validation_failed | 0.6 | 0.59 | 8.44401 |
| torch_tm_flowpipe | native_low_order | riccati | 0.01 | success |  | 1 | 0.111438 |
| torch_tm_flowpipe | native_low_order | harmonic | 0.01 | success |  | 4 | 24.2863 |
| torch_tm_flowpipe | native_low_order | van_der_pol | 0.005 | validation_failed | 0.485 | 0.48 | 8.69714 |
| torch_tm_flowpipe | native_low_order | van_der_pol | 0.01 | validation_failed | 0.47 | 0.46 | 6.03999 |
| diffreach | multi_step_common_box_carry | riccati | 0.01 | success |  | 1 | 0.113977 |
| diffreach | multi_step_common_box_carry | harmonic | 0.01 | success |  | 4 | 11.1417 |
| diffreach | multi_step_common_box_carry | van_der_pol | 0.005 | validation_failed | 0.63 | 0.625 | 1.65312 |
| diffreach | multi_step_common_box_carry | van_der_pol | 0.01 | validation_failed | 0.55 | 0.54 | 1.19003 |
| diffreach | native_low_order | riccati | 0.01 | success |  | 1 | 0.113899 |
| diffreach | native_low_order | harmonic | 0.01 | success |  | 4 | 11.1417 |
| diffreach | native_low_order | van_der_pol | 0.005 | validation_failed | 0.62 | 0.615 | 1.62973 |
| diffreach | native_low_order | van_der_pol | 0.01 | validation_failed | 0.54 | 0.53 | 1.16747 |
| flowstar | multi_step_common_box_carry | riccati | 0.01 | success |  | 1 | 0.131854 |
| flowstar | multi_step_common_box_carry | harmonic | 0.01 | validation_failed | 2.23 | 2.22 | 1.9834 |
| flowstar | multi_step_common_box_carry | van_der_pol | 0.005 | validation_failed | 0.08 | 0.075 | 0.344051 |
| flowstar | multi_step_common_box_carry | van_der_pol | 0.01 | validation_failed | 0.01 |  |  |
| flowstar | native_low_order | riccati | 0.01 | success |  | 1 | 0.131854 |
| flowstar | native_low_order | harmonic | 0.01 | success |  | 4 | 1.33827 |
| flowstar | native_low_order | van_der_pol | 0.005 | validation_failed | 0.09 | 0.085 | 0.336716 |
| flowstar | native_low_order | van_der_pol | 0.01 | validation_failed | 0.01 |  |  |

On Van der Pol, the farthest validated horizon depends on protocol; the following table retains the protocol and never ranks widths from different failure times:

| tool | protocol | h | run_status | failure_horizon_or_censor | final_valid_time |
| --- | --- | --- | --- | --- | --- |
| torch_tm_flowpipe | multi_step_common_box_carry | 0.005 | validation_failed | 0.63 | 0.625 |
| torch_tm_flowpipe | multi_step_common_box_carry | 0.01 | validation_failed | 0.6 | 0.59 |
| torch_tm_flowpipe | native_low_order | 0.005 | validation_failed | 0.485 | 0.48 |
| torch_tm_flowpipe | native_low_order | 0.01 | validation_failed | 0.47 | 0.46 |
| diffreach | multi_step_common_box_carry | 0.005 | validation_failed | 0.63 | 0.625 |
| diffreach | multi_step_common_box_carry | 0.01 | validation_failed | 0.55 | 0.54 |
| diffreach | native_low_order | 0.005 | validation_failed | 0.62 | 0.615 |
| diffreach | native_low_order | 0.01 | validation_failed | 0.54 | 0.53 |
| flowstar | multi_step_common_box_carry | 0.005 | validation_failed | 0.08 | 0.075 |
| flowstar | multi_step_common_box_carry | 0.01 | validation_failed | 0.01 |  |
| flowstar | native_low_order | 0.005 | validation_failed | 0.09 | 0.085 |
| flowstar | native_low_order | 0.01 | validation_failed | 0.01 |  |

## Protocol C: native low-order supplement

| tool | tool_variant | system | h | state_name | run_status | final_valid_time | width_at_own_final_valid_step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| torch_tm_flowpipe | complete_total_degree_order_1 | riccati | 0.01 | x | success | 1 | 0.111438 |
| torch_tm_flowpipe | complete_total_degree_order_1 | harmonic | 0.01 | x1 | success | 4 | 24.2863 |
| torch_tm_flowpipe | complete_total_degree_order_1 | harmonic | 0.01 | x2 | success | 4 | 24.2863 |
| torch_tm_flowpipe | complete_total_degree_order_1 | van_der_pol | 0.005 | x1 | validation_failed | 0.48 | 8.69714 |
| torch_tm_flowpipe | complete_total_degree_order_1 | van_der_pol | 0.005 | x2 | validation_failed | 0.48 | 237.197 |
| torch_tm_flowpipe | complete_total_degree_order_1 | van_der_pol | 0.01 | x1 | validation_failed | 0.46 | 6.03999 |
| torch_tm_flowpipe | complete_total_degree_order_1 | van_der_pol | 0.01 | x2 | validation_failed | 0.46 | 111.16 |
| diffreach | affine_flag | riccati | 0.01 | x | success | 1 | 0.113899 |
| diffreach | default_restricted_quasi_quadratic | riccati | 0.01 | x | success | 1 | 0.113895 |
| diffreach | affine_flag | harmonic | 0.01 | x1 | success | 4 | 11.1417 |
| diffreach | affine_flag | harmonic | 0.01 | x2 | success | 4 | 11.1417 |
| diffreach | default_restricted_quasi_quadratic | harmonic | 0.01 | x1 | success | 4 | 0.303186 |
| diffreach | default_restricted_quasi_quadratic | harmonic | 0.01 | x2 | success | 4 | 0.303186 |
| diffreach | affine_flag | van_der_pol | 0.005 | x1 | validation_failed | 0.615 | 1.62973 |
| diffreach | affine_flag | van_der_pol | 0.005 | x2 | validation_failed | 0.615 | 7.93661 |
| diffreach | default_restricted_quasi_quadratic | van_der_pol | 0.005 | x1 | success | 1 | 0.118264 |
| diffreach | default_restricted_quasi_quadratic | van_der_pol | 0.005 | x2 | success | 1 | 0.140647 |
| diffreach | affine_flag | van_der_pol | 0.01 | x1 | validation_failed | 0.53 | 1.16747 |
| diffreach | affine_flag | van_der_pol | 0.01 | x2 | validation_failed | 0.53 | 4.97027 |
| diffreach | default_restricted_quasi_quadratic | van_der_pol | 0.01 | x1 | success | 1 | 0.128936 |
| diffreach | default_restricted_quasi_quadratic | van_der_pol | 0.01 | x2 | success | 1 | 0.151332 |
| flowstar | minimum_supported_fixed_order_2 | riccati | 0.01 | x | success | 1 | 0.131854 |
| flowstar | minimum_supported_fixed_order_2 | harmonic | 0.01 | x1 | success | 4 | 1.33827 |
| flowstar | minimum_supported_fixed_order_2 | harmonic | 0.01 | x2 | success | 4 | 1.33827 |
| flowstar | minimum_supported_fixed_order_2 | van_der_pol | 0.005 | x1 | validation_failed | 0.085 | 0.336716 |
| flowstar | minimum_supported_fixed_order_2 | van_der_pol | 0.005 | x2 | validation_failed | 0.085 | 0.307825 |
| flowstar | minimum_supported_fixed_order_2 | van_der_pol | 0.01 | x1 | validation_failed |  |  |
| flowstar | minimum_supported_fixed_order_2 | van_der_pol | 0.01 | x2 | validation_failed |  |  |

Native results differ from controlled box carry because Torch retains initial-generator dependency, DiffReach retains its upstream symbolic normalization state, and Flow* retains a normalized Taylor-model flowpipe. Protocol B deliberately erases all of those dependencies at every boundary.

## Runtime decomposition

Build, JIT, first execution, and steady execution are separate. The steady column is the relevant implementation-throughput measure after one-time work; no combined total-runtime ranking is claimed.

| tool | tool_variant | protocol | build_time_s | jit_compile_time_s | first_execution_time_s | steady_runtime_per_step_s |
| --- | --- | --- | --- | --- | --- | --- |
| diffreach | affine_flag | multi_step_common_box_carry | 0 | 0.650279 | 0.000633583 | 0.000146418 |
| flowstar | minimum_supported_fixed_order_2 | multi_step_common_box_carry | 1.74199 | 0 | 0.0002035 | 0.0001885 |
| torch_tm_flowpipe | complete_total_degree_order_1 | multi_step_common_box_carry | 0 | 0 | 0.0194073 | 0.0186529 |
| diffreach | affine_flag | native_low_order | 0 | 0.657095 | 0.00176537 | 0.000216442 |
| diffreach | default_restricted_quasi_quadratic | native_low_order | 0 | 1.17619 | 0.00176807 | 0.000263769 |
| flowstar | minimum_supported_fixed_order_2 | native_low_order | 1.75274 | 0 | 0.0002095 | 0.0002055 |
| torch_tm_flowpipe | complete_total_degree_order_1 | native_low_order | 0 | 0 | 0.0193008 | 0.018581 |
| diffreach | affine_flag | one_step_common_input | 0 | 0.510024 | 0.000645546 | 0.000173684 |
| flowstar | minimum_supported_fixed_order_2 | one_step_common_input | 1.72546 | 0 | 0.000203 | 0.0001965 |
| torch_tm_flowpipe | complete_total_degree_order_1 | one_step_common_input | 0 | 0 | 0.013827 | 0.0130709 |

## Figures

- [One-step exact inflation ratio](plots/one_step_exact_inflation_ratio_vs_h.png)
- [Common-box endpoint width curves](plots/multi_step_common_box_carry_width_vs_time.png)
- [Common-time grouped widths](plots/common_time_grouped_width_bars.png)
- [First validation-failure horizon](plots/first_validation_failure_horizon.png)
- [Runtime decomposition](plots/runtime_decomposition.png)
- [Native low-order width curves](plots/native_low_order_width_curves.png)
- [Semantics table](plots/semantics_table.png)

## Remaining limitations

- DiffReach uses floating-point JAX interval-style arithmetic rather than MPFR-directed interval arithmetic; analytic and sampled gates detect tested violations but do not turn sampling into a proof.
- Flow* extraction restores the validated initial Picard candidate instead of exporting the stock un-revalidated refinement image. This is conservative and clearly labeled in every row.
- Local polynomial bases and validators remain tool-specific. The external contracts align inputs and carry/reset policies, not internal algorithms.
- Runtime values are machine- and build-dependent. Flow* compilation, DiffReach JIT, and Torch eager orchestration must remain separate.

## Reproduction

```bash
cd /srv/local/shengenli/torch_tm_flowpipe_three_way_comparison
experiments/three_way_common_contract/run_smoke.sh
experiments/three_way_common_contract/run_all.sh
# or after the interactive smoke gate:
experiments/three_way_common_contract/launch_background.sh
tmux attach -t tm_three_way_common_contract
```

The exact canonical specification copied into this result directory is `benchmark_spec.yaml`; adapter logs and generated Flow* sources are under `logs/`.
