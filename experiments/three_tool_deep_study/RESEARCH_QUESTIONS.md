# Research questions

The study answers six questions.

1. With identical one-step inputs and raw tube/endpoint semantics, how tight is
   each local enclosure?
2. If every solver may carry only `x = c + A ξ + I`, which local construction
   controls error best?
3. Which retained monomials, resets, symbolic mechanisms, and preconditioners
   explain native low-order behavior?
4. What width/runtime/successful-horizon Pareto frontier does each solver
   achieve when allowed to use documented native strengths?
5. Which differences are attributable to basis, polynomial construction, range
   bounding, validator, remainder refinement, reset, preconditioning, or
   backend?
6. Which concrete changes would most improve `torch_tm_flowpipe`?

A literal same-order ranking is excluded because the integer called “order” has
different basis and lifecycle meaning in the three implementations. Controlled
and native questions are reported separately; no universal winner is assumed.
