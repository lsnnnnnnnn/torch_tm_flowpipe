# Q3/order semantics

`complete_q3` is the complete total-degree-3 support over six variables, with local time in slot 0. It has 84 slots. Products or time integrations whose exponent is absent are evaluated into interval overflow. The checked monomial retention predicate is exactly `sum(exponent) <= 3`, so the small predicate test agrees with Torch `Polynomial.truncate(3)`.

That predicate agreement is necessary but not sufficient for algorithmic equivalence. Xiangru uses a dense fixed basis, exactly two polynomial Picard iterates, a seed remainder, and ten DR-RP remainder rounds. Torch uses sparse polynomial maps in its current VDP lane and defaults polynomial Picard iterations to `order`, hence three iterations at order 3. The plant/controller contracts also differ. Therefore this audit records `retention_predicate_equivalent=true` but `full_algorithmic_order_semantics_equivalent=false`.

Sources: Xiangru `generic_fixed_basis.py:94-108`, `tensor_fixed_basis.py:149-162,605-648,838-928`; Torch `polynomial.py:235-250`, `flowpipe.py:1783-1805`; Flow* `Continuous.cpp:2781,2843,2920-2940,2962`.
