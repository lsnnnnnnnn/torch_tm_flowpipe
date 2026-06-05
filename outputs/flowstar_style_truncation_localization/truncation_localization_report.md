# Truncation Localization Report

Failure dimension near the final rejected attempts: `y`.
Is truncation dominated by a few dropped monomials or many small ones? `few dropped monomials`.
Are dropped terms mostly from x*x*y or Picard integration? Dominant source=`(x*x)*y`; source totals: (x*x)*y=0.089145, Picard integrated y=0.00230383, x*x=0.000132639, Picard integrated x=1.84776e-05.
Is containment failure caused by width or residual shift? `residual shift`.
What symmetric target radius would be needed for the failed step? `0.00010037083520361246`.
Minimal asymmetric interval needed: `[-9.574283820161121e-05, 0.00010037083520361246]`.
Would a tighter range bound on dropped terms likely fix it? `possibly`.

## Output Files

- `truncation_localization_summary.csv` summarizes the final failure mechanism.
- `truncation_top_terms.csv` lists top dropped monomial interval contributions near rejected attempts after t>2.
- `dropped_terms_near_failure.png` and `residual_shift_near_failure.png` visualize term size and residual shift.
