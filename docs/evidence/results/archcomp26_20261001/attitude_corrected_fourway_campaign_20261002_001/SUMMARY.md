# Attitude Control corrected unsafe box: four-method process campaign

Run ID: `attitude_corrected_fourway_campaign_20261002_001`. Original server directory: `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/attitude_corrected_fourway_campaign_20261002_001`. This is a new 24-process timing campaign, separate from the earlier single full-horizon runs.

The six physical-state initial box and the closed official unsafe box are recorded in `PLAN.json`. All four methods used the official torch ONNX, 30 controller periods, 60 saved `h=0.05` segments, physical GPU 2, CPU affinity 10–13, and sequential fresh processes. Flow* native used exclusive RPC port 5103. The P3 launcher was staged with only its physical-GPU guard and error message changed from GPU 3 to GPU 2; `original_p3_source.py` preserves the source used for that plain-text comparison.

| Method | First fresh process wall (s) | Later 5 median (s) | Later 5 min–max (s) | Saved tube verdict |
| --- | ---: | ---: | ---: | --- |
| Flow* native | 6.233379421 | 6.181134824 | 5.982561410–6.182213624 | corrected author checker `VERIFIED`; 30 RPC and HTTP 200 |
| Huan | 6.985901689 | 6.834693036 | 6.833845183–7.036363389 | author checker silence, 60 accepted segments |
| Xiangru | 6.884525434 | 6.884600563 | 6.783010526–6.986936340 | author checker silence, 60 accepted segments |
| Our P3 | 13.205139695 | 12.951999972 | 12.851352740–13.154223252 | author checker silence, 60 accepted segments |

`SUMMARY.json` reports 24/24 completed and valid with no first failure or resource stop. The independent local audit scanned all 1,440 saved six-state range records; every finite, ordered segment tube was disjoint from the **closed corrected six-dimensional unsafe box**. Its minimum positive box separation gap was 0.072724112698 for native, 0.072695128080 for Huan and Xiangru, and 0.072798355056 for P3, identical across the six process repetitions of each method. The native logs show 30 periods, 60 segments, 30 RPC requests and 30 HTTP 200 responses per run. Each GPU run has 30 recorded controller steps and 60 accepted substeps. The audit checked campaign event order, nonoverlapping timestamps, fixed device/CPU/model, generated unsafe constraints and the staged P3 guard edit.

Machine-readable evidence: `PLAN.json`, `events.jsonl`, `SUMMARY.json`, `RUNS.csv`, `INDEPENDENT_AUDIT.json`, and each of the 24 `first00_*`/`later0*_*` raw run directories. The independent scanner is `audit_campaign.py`; `runner.py` is the actual saved remote campaign launcher. This audit performs no content digest.

The wall boundary is supervisor child process start to reap. Native time includes RPC server startup; GPU time includes Python, NN and CUDA startup. First means the first fresh process in this campaign, not a rebooted host or GPU. This shared-host run gives descriptive timings, not a hardware-isolated speed ranking. Saved tube checks and the author checker do not independently certify the floating-point NN bound or the whole NNCS implementation.
