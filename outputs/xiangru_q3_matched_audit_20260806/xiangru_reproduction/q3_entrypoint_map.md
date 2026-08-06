# Xiangru complete-Q3 entrypoint map

| Question | Authoritative path | Evidence |
|---|---|---|
| Where is Q3 selected? | `experiments/remainder_ablation/run_s0_tora_static_partition_sweep.py:341-342` | Method `complete_q3` calls `complete_total_degree_support(3)`. |
| What receives it? | `run_s0...py:345-348,456-462` | The support builds static routes, then enters `step_closed_loop`. |
| What mathematical operation changes? | `generic_fixed_basis.py:94-108`; `tensor_fixed_basis.py:149-162,605-648` | Six-variable complete total-degree support; integration/products outside support are intervalized. |
| What completed baseline is used here? | `s3c_fair_timing_raw/rep1_q3/s3r_q3_b48_rep1.json` | Homogeneous TORA, B48 static partition, complete-Q3, 200 segments. |
| What does “completed” mean? | raw cell fields plus `run_s0...py:688-715` | Cell `VERIFIED`, no first failure/retries, 200/200 segments, certified horizon 20.0; top-level `PARTIAL_VALIDATION` only records selected matrix scope. |

The authoritative frozen command is in `original_commands.sh`; its config and controller bytes are in `original_config_snapshot/`.
