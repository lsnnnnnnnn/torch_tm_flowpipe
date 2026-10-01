# ACC participant-order native, corrected VAR tail, full 50 periods

This is a **new** isolated 2026-10-01 attempt. The complete original evidence in this directory was copied from remote `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/acc_native_var_tail_full50_001`. The one-period plumbing run is separately preserved at `../acc_native_var_tail_smoke1_001/`; its short-horizon `VERIFIED` label is not a 5 s benchmark verdict. No content digest or SHA-256 check was performed in either new attempt.

Contract: [named ACC participant-order profile](../../../../ARCHCOMP26_ACC_PARTICIPANT_CONTRACT_20261001.md). The input map is `[30,1.4,v_ego,x_lead-x_ego,v_lead-v_ego]`; the fixed 2026 controller copy is `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/acc_prep_001/official_acc_controller_5_20.onnx`. This is the participant source's explicit relative-velocity sign; the 2026 paper leaves that sign unnamed. The Flow* binary `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/acc_native_var_tail_build_001/acc_native_var_tail` was freshly compiled against the saved isolated VAR-tail-corrected library. The paired RPC server was `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/acc_native_var_tail_build_001/observed_server.py`, on port 5102, GPU 2, CPUs 10–13. The corrected binary and library remain on the server; they were not transferred into this evidence directory.

## Completion and property

- [START.json](START.json) and [RESULT.json](RESULT.json): supervisor completed, exit 0, wall time 7.930 s, timeout false; the paired launcher recorded the service and native PIDs and port ownership. The native log shows Steps 0–49 and `VERIFIED`, with its own solver time 4.419 s.
- [controller_rpc.jsonl](controller_rpc.jsonl) contains 50 controller requests; [server.log](server.log) has 50 POST records. The first input lower vector is `[30,1.4,30.000000000000004,79,1.8000000000000007]`, consistent with the declared feature transform. The final request's fifth input lower bound is `-6.200586805235841`.
- [ranges.bin](ranges.bin) has 50 records, one lane and consecutive steps 1–50, each with local `h=0.1`. Each 216-byte record is little-endian `uint64 lane`, `uint64 step`, `float64 h`, then for each of six physical states four float64 values `(tube_lo,tube_hi,endpoint_lo,endpoint_hi)`. It is the native local-domain axis-aligned range observer, not an independently recomputed Taylor model.
- An independent scan of those recorded **tube** boxes evaluated the safety margin lower bound `x_lead_lo - x_ego_hi - 1.4*v_ego_hi - 10` for all 50 records. Its minimum was **16.25475375921129 > 0**, in agreement with the author's `VERIFIED` checker for the all-time halfspace. This uses only the saved state interval boxes and is conservative with respect to correlations. It does not independently prove the neural-network bounder or every floating-point operation.

## Recorded physical ranges

| State | Full `t∈[0,5]` tube union | T=5 endpoint interval | T=5 width |
| --- | --- | --- | ---: |
| `x_lead` | `[89.99997107492605,250.05933191284015]` | `[229.03713723127746,250.04918866762728]` | 21.01205143634982 |
| `v_lead` | `[22.816705888415587,32.20005787304437]` | `[22.81672433252317,23.01848275887569]` | 0.20175842635252295 |
| `a_lead` | `[-2.035271618160528,0.0005712737196437219]` | `[-2.0289686179857127,-2.0282260735541815]` | 0.0007425444315312113 |
| `x_ego` | `[9.999785471157484,159.603440312928]` | `[154.1047791270118,159.60277792945882]` | 5.497998802447029 |
| `v_ego` | `[27.184807808393042,30.20042918838096]` | `[27.18484543562199,29.20037374341775]` | 2.0155283077957584 |
| `a_ego` | `[-1.1246481743826773,0.05894171852068421]` | `[-1.1246481743826773,0.05647573581455831]` | 1.1811239101972355 |

The paper/ONNX input semantic boundary and the targeted scope of the saved VAR-tail fix remain the contract limitations described in the linked audit. This run is a complete author-checker outcome for the explicitly named participant-order variant, not an end-to-end independently certified floating-point NN proof.
