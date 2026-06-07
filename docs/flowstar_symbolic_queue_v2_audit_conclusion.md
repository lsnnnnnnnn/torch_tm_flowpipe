# Flowstar Symbolic Queue V2 Audit Conclusion

## Executive Conclusion

v2 is a cleaner diagnostic symbolic queue, but it does not rescue h10. The best v2 h10 run reaches `t=7.4960392581387341`, matching the best no-queue and split-symbolic-queue runs. It passes the 500-sample containment sanity check, while the local one-step Flow* oracle after v2 still reports `not_completed` for orders 4, 6, and 8 on the exported local box. Treat this as diagnostic evidence, not Flow* parity.

Decision: `NEEDS_MORE_WORK`. The useful result is reproducibility and channel attribution, not a merge-ready reachability improvement. The next mechanism should stop queue variants for now and focus on Flow* preconditioning, Horner/source-order insertion semantics, or a local-box reducer driven by the oracle failure.

## Commands Used

```bash
cd /srv/local/shengenli/torch_tm_flowpipe
python experiments/flowstar_queue_state_audit.py --out-dir outputs/flowstar_queue_state_audit
```

The h10 v2 outputs already existed from the previous v2 run under `outputs/flowstar_symbolic_queue_v2_h10` and have been copied into the preferred evidence directory `outputs/flowstar_normalized_insertion_symqueue_v2_h10`. Future regeneration should use the preferred directory name:

```bash
cd /srv/local/shengenli/torch_tm_flowpipe
export FLOWSTAR_ROOT=/srv/local/shengenli/flowstar
conda run -n py11 python experiments/flowstar_style_rescue_vanderpol.py \
  --out-dir outputs/flowstar_normalized_insertion_symqueue_v2_h10 \
  --max-horizon 10 \
  --wall-cap-s 300 \
  --configs flowstar_style_o4_target_insert_symqueue_v2 flowstar_style_o6_candidate8_output6_insert_symqueue_v2
```

## Artifact Inventory

| Artifact | Purpose |
| --- | --- |
| `docs/flowstar_source_queue_semantics_audit.md` | Source-guided Flow* queue semantics note |
| `outputs/flowstar_queue_state_audit/queue_state_trace.csv` | Three-way per-segment queue/channel trace |
| `outputs/flowstar_queue_state_audit/queue_state_summary.csv` | Three-way source/run summary |
| `outputs/flowstar_queue_state_audit/queue_state_report.md` | Three-way audit answers |
| `outputs/flowstar_normalized_insertion_symqueue_v2_h10/symqueue_v2_summary.csv` | v2 h10 summary |
| `outputs/flowstar_normalized_insertion_symqueue_v2_h10/symqueue_v2_segments.csv` | v2 h10 segment diagnostics |
| `outputs/flowstar_normalized_insertion_symqueue_v2_h10/symqueue_v2_report.md` | v2 h10 report |
| `outputs/flowstar_normalized_insertion_symqueue_v2_h10/sample_containment_summary.csv` | 500-sample v2 containment result |
| `outputs/flowstar_symbolic_queue_v2_h10/` | Legacy source directory from the original v2 run |
| `outputs/flowstar_one_step_oracle_after_symqueue_v2/oracle_after_symqueue_v2_report.md` | Existing local one-step Flow* oracle result |

## Three-Way Comparison

Common horizon for best-run width comparisons: `7.4960392581387341`.

| source | best run_id | status | last_validated_t | reached_h10 | max_reset_width_sum_common_horizon | max_right_map_range_width_sum_common_horizon | max_output_only_symbolic_width_sum | sample_containment_status | oracle_status |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| no_queue | `flowstar_style_o6_candidate8_output6_insert` | failed | 7.4960392581387341 | no | 21.902999702465412 | unavailable | unavailable | not rerun here | not compared here |
| split_symqueue | `flowstar_style_o6_candidate8_output6_insert_symqueue_split` | failed | 7.4960392581387341 | no | 21.902999702465412 | unavailable | 5.0175141128736733 | passed in existing artifact | rejected/not_completed after split |
| v2_symqueue | `flowstar_style_o6_candidate8_output6_insert_symqueue_v2` | failed | 7.4960392581387341 | no | 21.902999702465415 | 21.883875748631645 | 0.24507717887847935 | passed, 0 violations over 75,000 checks | orders 4/6/8 not_completed |

## Answers

- Does v2 beat no_queue or split? No. Best validated time is tied at `7.4960392581387341`.
- Does v2 reduce reset width? No. The best-run common-horizon reset width is effectively unchanged versus no_queue/split.
- Does v2 reduce right-map range width? Not comparable from the old no_queue/split raw fields; v2 records `21.883875748631645` for its best run.
- Does v2 only add output symbolic width? Yes for the recorded v2 diagnostics: target-check contribution remains about zero, and output-only symbolic width is positive.
- Is h10 still not reached? Yes. All v2 h10 configs remain failed before horizon 10.
- Did v2 move Flow* width-ratio crossings later? No. The audit reports first `width_ratio>=1` at `3.206019346657988` for v2 versus `4.0459460349724967` for no_queue and split.

## Queue Channel Interpretation

The ordinary target channel is the clean normalized reset used for target-remainder validation. The symbolic output-only channel is propagated older queue width added back for output/range accounting. v2 stores `J`, `Phi_L`, inverse scalars, current linear map entries/norms, queue counts, and target/output channel widths.

Source-guided max-size behavior is append-then-reset-before-next-step: when the accepted-step queue reaches `max_size`, the returned clean-room next-step state is reset. Source-guided scalar behavior is inverse magnitude applied as right column scaling of the next linear map. Source-guided Phi_L behavior is current-left multiplication of older accumulated maps.

## Flow* Source Audit Summary

Consulted files include `flowstar-toolbox/Continuous.h`, `flowstar-toolbox/Continuous.cpp`, `flowstar-toolbox/Discrete.h`, `flowstar-toolbox/Matrix.h`, and `flowstar-toolbox/TaylorModel.h` in `/srv/local/shengenli/flowstar`. The v2 implementation matches the queue state shape, inverse scalar direction, reset threshold interpretation, and linear-map multiplication order at a clean-room diagnostic level.

What remains missing is exact parity for Flow* preconditioning, invariant contraction, Horner insertion/source order, and nonlinear queue treatment. The one-step oracle result suggests the exported local box is still too wide or missing a deeper Flow* mechanism.

## Decision

`NEEDS_MORE_WORK`. Keep v2 as diagnostic evidence because it makes the channel accounting reproducible and tests source-guided queue invariants. Do not claim Flow* parity or h10 success. The next task should focus on Flow* preconditioning/Horner/source-order semantics rather than another symbolic queue variant.
