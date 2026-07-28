# Original Flow* Van der Pol parity

The local upstream benchmark used for parity is
`benchmarks/continuous/vanderpol/vanderpol_2d.cpp`, not a settings claim copied
from an earlier report.

Its effective configuration is:

| Item | Value |
|---|---|
| Equations | `x1' = x2`; `x2' = x2 - x1 - x1^2*x2` |
| Initial box | `x1 ∈ [1.10,1.40]`, `x2 ∈ [2.35,2.45]` |
| Horizon | 10 |
| Step policy | adaptive, 0.002 through 0.1 |
| Order | 4 |
| Cutoff | 1e-10 |
| Candidate remainder radius | 1e-4 |
| Preconditioning | QR |
| Symbolic remainder window | 100 |
| Output | stock composed Flowpipe tube; endpoint extracted without mutation |

Three executions are compared step-by-step:

A. the actual original executable;
B. a generated C++ harness with identical settings;
C. the repaired generic harness with identical settings.

The audited run produced 290 accepted segments in each implementation, reached
`T=10`, matched accepted schedules to `5e-7`, and matched generated-versus-
generic interval bounds to `1e-14`. Runtime and build measurements are retained
as parity metadata, not solver rankings.

This proves the installed Flow* library and repaired harness can reproduce a
known-working configuration. It disproves H3 as a general limitation claim,
while leaving configuration-specific fixed-order failures visible.
