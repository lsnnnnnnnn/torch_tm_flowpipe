# Flowstar / Torch complete-O4 fixed-schedule common prefix

Date: 2026-08-11

## Outcome

`FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_ONLY`.

## Eligibility

The 632 accepted fixed-schedule steps are eligible for schedule-controlled
empirical common-prefix endpoint/tube comparison. Only the initial state is
same-prestate. The pinned Flow* build remains formally ineligible because its
scalar-affine under-enclosure gate is open. Cross-tool timing is unavailable.

## Contract

B1, VDP, complete total-degree O4, `h=0.01` exactly, 1,000 requested steps,
target remainder `[-0.0001,0.0001]`, cutoff `[-1e-10,1e-10]`, no adaptive
fallback, no endpoint repair. Endpoint, segment tube, and prefix tube are
distinct columns.

## What was actually run

The unmodified pinned Flow* core with a read-only observer and the Torch
complete-O4 fixed-schedule runner each ran cold from the frozen initial box.
Both attempted every step until Torch's first rejection; Flow* continued to
the requested T10 capability endpoint.

## Exact results

- Flow*: 1,000 accepted steps, validated horizon 10.0, cold process wall 6.29
  seconds, peak RSS 97,280,000 bytes.
- Torch: 632 accepted steps, rejected candidate 633 at pre-time 6.32,
  validated horizon 6.32, core wall 407.56588636524975 seconds, peak RSS
  642,637,824 bytes.
- Torch failed because the raw-remainder-compatible residual was not a subset
  of the target at fixed minimum step 0.01; y margin was
  `-8.441898798e-06` in the recorded decision artifact.
- Common prefix: 632 rows. Widths first differ at step 1; margin ordering first
  changes at step 23.

## What is comparable

For steps 1--632: fixed schedule, initial box, representation order, endpoint
widths, segment-tube widths, prefix-tube widths, target margins, and diagnostic
cumulative runtime to the same accepted index.

## What remains unavailable

There is no same-prestate claim after step 1, no Torch value after its failure,
no fixed-schedule T10 tightness row, and no eligible one-cold/ten-warm timing
ratio.

## Negative results

Both tools did not complete the same fixed-schedule T10 contract. Failure-time
values are not replaced by zero or joined to fictional later rows. Flow*'s
T10 capability cannot be promoted to a formal oracle under the open build
qualification.

## Limitations

The comparison is empirical/build-qualified and schedule-controlled, not a
proof of cross-tool enclosure ordering. A wider row does not establish greater
soundness, and the runtime figures include different implementation stacks.

## Evidence paths

Under the final package `outputs/three_tool_full_horizon_pairwise_carry_closure_20260811/20260811T191549Z/`:

- `04_flowstar_torch_fixed_schedule/common_prefix/summary.json`;
- `04_flowstar_torch_fixed_schedule/common_prefix/common_prefix.csv`;
- `12_pairwise_tables/table_mf_flowstar_torch_common_prefix.csv`;
- `13_figures/flowstar_torch_endpoint_tube_widths.svg` and source CSV;
- `13_figures/flowstar_torch_y_margin.svg` and source CSV.

## Reproduction commands

```bash
g++ --version
python experiments/run_vdp_dense_backend.py --help
python experiments/compare_flowstar_torch_fixed_schedule.py --help
```

## Next authorized action

Do not tune the frozen contract. The only authorized follow-up is an
independent numerical qualification of the pinned Flow* build or a separately
specified representation-semantics study.
