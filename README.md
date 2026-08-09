# torch-tm-flowpipe: TORA-Q3 stage-parity closure

This branch is the clean, self-contained review surface for the TORA-Q3
stage-parity, fused-kernel, and native-hierarchy investigation. The final
classification is **Case C**: the fused common-control plant passes P0--P5 and
the internal 10x gates, while every Torch-native lane fails the strict T5 gate.
The best Torch certified horizon is `4.4 s`; T10/T20 are therefore `NOT_RUN`.

The branch starts at clean commit
`63efe66cfe7bdda907f8255ba23cebaa9b878233`, descends from parentless clean
root `9fc45344c4379422244b75af705dffd17304f824`, and has no merge base with
blocked historical tip `c49d74bbf48d1004f7f3818174e7f40b6200b142`.
Controller bytes, Xiangru raw source and traces, observation patches, raw
per-leaf tensors, compilation caches, and server paths are not in Git.

The historical Van der Pol failure near `t=6.397083942944808` remains an
unresolved, independent task. Nothing on this branch claims a VDP fix.

## Current result

The earliest numerical difference is A2 point-sine outward rounding
(`4.22e-15`). The first material difference is A3 sine-composition remainder
routing: `0.0145973` maximum width difference at segment 1, leaf 0. At T1,
`99.924210%` of the direct `0.014211021942602` endpoint difference is already
present before projection. At segment 40, the `1.2186185882` maximum carried
remainder is categorized as `composition_overflow`; the current-step Picard
residual is only `0.00126978318`.

The independently implemented `algorithm_aligned_q3` lane uses a centered
quadratic sine polynomial, signed input-remainder propagation, a complete
line-segment third-derivative bound, and separated outward remainder routing.
It passes one-leaf, B48 one-step, T1, and common-control T20. It does not pass
native T5. The evidence-selected h=.05 fallback improves the native horizon
from `4.3 s` to `4.4 s`, but also fails T5 by the unchanged physical property;
the numerical certificate still passes at the first failure.

The four-stage, 13-invocation fused tensor path has zero graph breaks inside
each full graph. It reduces a B48 logical step from `0.508397 s` to
`0.129653 s` and common-control T20 from `105.480052 s` to `26.185731 s`.
That is a `19.553566x` internal speedup over the frozen `512.024427 s` Torch
baseline, but it remains a descriptive `21.6992x` slower than the matched
Xiangru `1.20676 s` common-control plant runtime.

## Portable quick start

```bash
python -m pip install -e ".[test]"
python -m compileall -q src tests experiments scripts examples
pytest -q
python examples/tora_q3_one_step.py
python scripts/check_readme_surface.py
python scripts/summarize_tora_q3_stage_parity.py
python scripts/summarize_tora_q3_fused_runtime.py
```

The example is CPU/float64, uses one leaf and a held-control interval, and
requires no external controller or private trace. Optional tests skip with an
explicit reason when private inputs or CUDA are absent; formal GPU results are
never inferred from such skips.

## External integration contract

External checks are enabled only by explicit environment variables:

```bash
export XIANGRU_ROOT
export TORA_CONTROLLER_PATH
export TORA_CONTROLLER_TRACE_PATH
export TORA_XIANGRU_STAGE_TRACE_PATH
export TORA_TORCH_STAGE_TRACE_PATH
pytest -q -m external_integration
```

`XIANGRU_ROOT` must be the independently frozen Xiangru tree at commit
`27d29050a5f214b56f211ca9cb411e734ed80230`. The controller SHA-256 must be
`52a50c6bc6b1b45b89319edc809cd4d3baca06c32f9fd2ddceb2e95007414418`,
and the controller-trace SHA-256 must be
`89a225add6e2c02ecb3e84b2182b2f7ea872b064dd9e5e534444552485a091d9`.
Missing optional variables skip; supplied but missing or drifted assets fail
closed.

## Evidence map

- [Stage-parity root cause](TORA_Q3_STAGE_PARITY_ROOT_CAUSE_REPORT.md)
- [Algorithm-aligned implementation](TORA_Q3_ALGORITHM_ALIGNED_IMPLEMENTATION_REPORT.md)
- [Fused-kernel runtime](TORA_Q3_FUSED_KERNEL_RUNTIME_REPORT.md)
- [Native T20 closure](TORA_Q3_NATIVE_T20_CLOSURE_REPORT.md)
- [Final handoff](handoff.md)
- [Stage-parity public summary](outputs/tora_q3_stage_parity_fused_20260809/stage_parity/summary.json)
- [Common-control public summary](outputs/tora_q3_stage_parity_fused_20260809/common_control/summary.json)
- [Final comparison](outputs/tora_q3_stage_parity_fused_20260809/comparison/summary.json)
- [Native hierarchy](outputs/tora_q3_stage_parity_fused_20260809/native_full_loop/hierarchical_gates.json)
- [Final validation](outputs/tora_q3_stage_parity_fused_20260809/tests/final_validation.json)
- [Final requirement audit](outputs/tora_q3_stage_parity_fused_20260809/provenance/final_requirement_audit.json)
- [Complete manifest](outputs/tora_q3_stage_parity_fused_20260809/manifest.sha256)

Common-control is a period-local plant replay and excludes controller time.
Native closed-loop results include controller execution and are separately
reported. No common-control result substitutes for a native T5/T10/T20 gate,
and no enclosure is interpolated onto a formal time point.
