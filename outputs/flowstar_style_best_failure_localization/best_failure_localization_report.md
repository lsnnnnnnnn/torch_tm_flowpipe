# Best Failure Localization Report

Failure dimension near the final rejected attempts: `y`.
Is truncation dominated by a few dropped monomials or many small ones? `few dropped monomials`.
Are dropped terms mostly from x*x*y or Picard integration? Dominant source=`(x*x)*y`; source totals: (x*x)*y=0.0159603, Picard integrated y=3.38118e-05, x*x=1.34041e-05, Picard integrated x=1.04208e-07.
Is containment failure caused by width or residual shift? `residual width`.
What symmetric target radius would be needed for the failed step? `0.00012293271852974451`.
Minimal asymmetric interval needed: `[-0.00012293271852974451, 0.00011527539184973732]`.
Would a tighter range bound on dropped terms likely fix it? `possibly`.

## Output Files

- `truncation_localization_summary.csv` summarizes the final failure mechanism.
- `truncation_top_terms.csv` lists top dropped monomial interval contributions near rejected attempts after t>2.
- `dropped_terms_near_failure.png` and `residual_shift_near_failure.png` visualize term size and residual shift.
