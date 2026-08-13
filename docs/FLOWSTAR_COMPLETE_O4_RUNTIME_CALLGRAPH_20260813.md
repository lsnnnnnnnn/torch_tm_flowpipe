# Flow* complete-O4 runtime call graph — 2026-08-13

Status: runtime path closed for the pinned Van der Pol trace.

## Provenance and observed features

The source is the isolated detached worktree
`/srv/local/shengenli/flowstar_source_carry_20260813` at
`b85a3211748cb77b736fe4ad42ee02d8d2b81148`. Its tracked source is clean. The
GCC-15 observation archive is `libflowstar.a` SHA256
`03cda3ac1a685a07098ca5ef53da8591676e7adf79e1e2d5261babd7bf05fc40`;
the trace probe is SHA256
`2c149080d941a2d175f37a7c9bb9d7cee99aa52556e618f8a70ed6b40591c24a`.
The historical baseline
used the same source SHA and the clean GCC-11 archive SHA256
`b5ff500af66354b0518cf12e7d951f4525f435e8e2d695cf84b91821992c9d9a`.
The observation build uses the normal `-O3 -g -std=c++11` flags plus
`-fpermissive` for a pre-existing, uninstantiated derivative-template body; it
does not patch arithmetic. Full, unshortened hashes and `file`/`ldd`/compiler
records are in `00_provenance/provenance.json`.

Runtime fields prove that symbolic remainder is enabled, the `J` queue becomes
active after the first boundary and reaches its configured maximum size 100,
and the expression Picard path is used. The benchmark supplies no invariant,
so invariant remainder contraction is inactive. No QR preconditioning or
shrink wrapping is invoked. The active preconditioning is the diagonal
range-magnitude scaling in `Flowpipe::advance`.

## Benchmark-to-output path

| stage | source/function | input → output mathematical object | dependency/rounding consequence |
|---|---|---|---|
| Model and initial set | `benchmarks/continuous/vanderpol/vanderpol.cpp:7-89 main` | `ODE<Real>(y, y-x-x²y, 1)` and interval box → `Flowpipe initialSet`; `Symbolic_Remainder(initialSet,100)` | The symbolic queue is explicitly passed to reachability. |
| Fixed reach loop | `flowstar-toolbox/Continuous.h:832-895 ODE::reach_symbolic_remainder` | current accepted `Flowpipe` → `currentFlowpipe.advance(..., symbolic_remainder)` → accepted chain | The same queue survives boundaries and resets at size 100. |
| Previous endpoint | `Continuous.cpp:2130-2148 Flowpipe::advance` | previous `tmvPre` evaluated at local `tau=h`, then constant removed | `tmvPre` is the previous time-parametric solution; `tmv` is its normalized domain map. |
| Linear/nonlinear split | `Continuous.cpp:2151-2159` | centered endpoint TM → `x0_linear`, `x0_other`, matrix `Phi_L_i` | Linear old sources are separated before nonlinear insertion. |
| Old-source propagation | `Continuous.cpp:2161-2177` | old `Phi_L[]` and `J[]` → matrix-propagated `J_i` | Each queued old linear remainder source is propagated once through its accumulated linear maps. |
| Nonlinear carry | `Continuous.cpp:2184-2251` | `x0_other` (queue active) or full endpoint (first step) inserted into normalized `tmv` | Nonlinear sources are still intervalized; Flow* does not retain a fully symbolic nonlinear source algebra. |
| Horner insertion | `TaylorModel.h:4213-4243 HornerForm::insert_ctrunc_normal` | recursive Horner form and inner TMs → composed TM | Expression grouping and truncation happen along Horner multiplication paths. |
| Local TM multiply | `TaylorModel.h:797-866 TaylorModel::mul_insert_ctrunc_normal` | `(P1,I1)*(P2,I2)` → `P1P2` plus `P1I2+P2I1+I1I2`, truncation, cutoff | Ordinary remainders are interval objects here; shared nonlinear identity is not preserved through this primitive. |
| New source and normalization | `Continuous.cpp:2179,2191,2241,2289,2292-2322` | new local insertion remainder `J_ip1`; `range_of_x0` → diagonal scale `S`, normalized `result.tmv` | `J_ip1` becomes a new queue entry. Scaling changes the next step domain but is not QR. |
| No-remainder Picard polynomial | `Continuous.cpp:2324-2338` | center plus diagonal generators → O1…O4 polynomial seed and target remainder | Produces the O4 polynomial candidate before interval validation. |
| Picard image and roundoff difference | `Continuous.cpp:2340-2368` | `Picard_ctrunc_normal` interval TM plus polynomial difference → candidate remainder subset test | Cutoff/truncation/roundoff intervals enter the validator. |
| Remainder refinement | `Continuous.cpp:2382-2410` | prior accepted remainder → refined `Picard_ctrunc_normal_remainder` | Subset and width-ratio refinement decides stored ordinary remainder. |
| Accepted flowpipe | `Continuous.cpp:2412-2416` | validated `x` and local domain → `result.tmvPre/domain` | This is the object consumed at the next boundary and by output extraction. |
| Range extraction | `Continuous.cpp:415-454 Flowpipe::compose_normal/intEvalNormal`; `TaylorModel.h:448-452` | accepted `tmvPre` composed with `tmv`, evaluated on `tau∈[0,h]` or `tau=h` | Polynomial range and stored remainder are both included. |
| Directed endpoint conversion | `Interval.cpp:851-859 Interval::sup/inf` | MPFR interval endpoints → binary64 | Lower uses `MPFR_RNDD`; upper uses `MPFR_RNDU`. |
| Probe serialization | `experiments/flowstar_probe/flowstar_vdp_step_trace_probe.cpp:400-408,613-624,695-724,1496-1506` | direct endpoint/tube bounds → 17-significant-digit CSV | Endpoint and segment are separate range evaluations; no low-precision collapse occurs. |

## Object responsibilities

`tmvPre` is the accepted local-time Taylor model vector for the current segment.
At the next step it is evaluated at `tau=h`. `tmv` is the normalized map from
the current local domain variables into the physical initial set. `domain`
contains local time plus normalized uncertainty variables. `J_ip1` is the new
ordinary interval remainder created by nonlinear insertion; `J_i` is the sum of
older queued sources after their linear maps. `Phi_L` retains the associated
linear propagation matrices. The accepted `tmvPre` remainder after Picard is
ordinary validation remainder, while the queue gives a separate cross-step
decomposition of parameterization uncertainty.

The exact conclusion is deliberately narrower than “Flow* preserves shared
remainder symbols”: Flow* preserves the identity of linear old sources in
`Phi_L/J`, while `x0_other.insert_ctrunc_normal` and local TM multiplication
intervalize nonlinear source interactions. Its tighter result can therefore
come from the linear-source queue, Horner grouping, and decomposition together;
none alone proves soundness.

## Soundness qualification

The pinned source has a separate strict analytic/256-bit-MPFR scalar-affine
under-enclosure witness documented in
`FLOWSTAR_SCALAR_AFFINE_CORRECTNESS_CLOSURE.md`: the accepted endpoint misses an
analytic bound by up to `3.4938679727147814e-10`, first during Picard remainder
refinement. The Van der Pol sampling audit found no point witness, but that does
not close the existing correctness gate. Consequently these Flow* widths are
valid source-behavior evidence and comparison measurements, not an unconditional
sound tightness oracle.
