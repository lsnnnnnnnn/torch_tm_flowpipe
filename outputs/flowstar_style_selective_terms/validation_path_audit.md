# Selective Validation Path Audit

Conclusion: retained degree >6 monomials are present during residual validation.
After-selective max high-degree term count: `16.0`.
Inside-validation max high-degree term count: `16.0`.
After-internal max high-degree term count: `16.0`.
Do after-selective and inside-validation term hashes match where comparable? no.
Do inside-validation and after-internal term hashes match where comparable? yes.

The audit hashes the candidate polynomial before selective retention, after selective retention, inside the Picard residual validator, and after internal validation Taylor-model operations.

## Stage Counts

| stage | rows | max_high_degree_terms |
| --- | ---: | ---: |
| after_selective | 800 | 16.0 |
| inside_validation | 800 | 16.0 |
| after_internal | 800 | 16.0 |
