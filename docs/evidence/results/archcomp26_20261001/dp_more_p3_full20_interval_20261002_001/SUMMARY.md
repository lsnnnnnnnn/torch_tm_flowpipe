# ARCH-COMP26 Double Pendulum more — P3 full-horizon attempt

This is a **new, isolated attempt** at the fixed 2026 more-robust continuous contract. The remote original is `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/dp_more_p3_full20_interval_20261002_001`; [COPY.json](COPY.json) lists the saved local source and raw files with byte counts. The old native, Huan, Xiangru, and P3 first-period jobs were inspected but not restarted.

The requested task was the full `[1,1.3]^4` initial box partitioned `5×5×3×3=225`, the fixed 2026 more-robust `[4,25,25,2]` ONNX controller, official four-state DP equations, **20 control periods × 0.02 s = T=0.4**, four 0.005 s ODE substeps per period, and the whole-tube safety band `[-1.5,1.5]^4`. [The generated run config](attempt/data/config.yaml) matches the saved full contract YAML except for its two comment lines. The copied source also contains `preflight_config.yaml` from the earlier one-period gate; that file was **not** used for this run. The method was work-P3/point-P2/validation-P4 with the existing directed interval controller residual, native-f64 control transfer and first numerical rejection stop. The only source change from the accepted first-period gate was reserving physical **GPU 2** instead of GPU 3; CPUs 10–13 and a 900 s outer limit were used. [Outer START](attempt/START.json) and [inner START](attempt/data/START.json) record the actual command and dependencies.

The attempt **stopped at its first numerical rejection**, cumulative ODE substep **9** (control period index 2, `t∈[0.04,0.045]`). All 225 lanes were accepted on each of steps 1–8, so **1,800 of the required 18,000 lane-substeps** were accepted through `t=0.04`. At step 9, **all 225 lanes had solver status 1 = `FAILED_CONTRACTION` and zero were accepted**. The stop handler saved that rejection row and raised an exception before the author driver could print a final property verdict. [Outer RESULT](attempt/RESULT.json) is `failed`, exit 1, no timeout, wall **16.6163630746305 s**; [inner RESULT](attempt/data/RESULT.json) records the first-rejection reason and inner wall **15.594604306854308 s**. These are failed-attempt times, not full-horizon benchmark times. `stdout.log` contains `Step 0`, `Step 1`, `Step 2` and **no** final `Unknown.`, `Unsafe.`, or `VERIFIED` line.

The [independent scan](INDEPENDENT_INTERVAL_SCAN_NOHASH.json), reproducible with [scan_saved.py](scan_saved.py), checked all **7,200 accepted four-state tube/endpoint pairs** for finite values, ordering, exact endpoint containment in each corresponding tube, and safety flags. The first four saved observations are value-for-value equal to the prior all-225-box one-period gate. Rejected step 9 retains the prior state in the driver and contributes **no valid new tube**.

| Cumulative ODE step | Accepted lanes | Lanes with all four saved tubes inside safety band | `θ̇₁` crossing lanes | `θ̇₂` crossing lanes |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 225 | 225 | 0 | 0 |
| 2 | 225 | 225 | 0 | 0 |
| 3 | 225 | 181 | 44 | 0 |
| 4 | 225 | 120 | 105 | 0 |
| 5 | 225 | 14 | 211 | 0 |
| 6 | 225 | 2 | 223 | 0 |
| 7 | 225 | 0 | 225 | 49 |
| 8 | 225 | 0 | 225 | 161 |
| 9 | 0 | 0 | — | — |

The first saved safety-band crossing is lane 6 at step 3, `θ̇₁∈[0.9912745512752368,1.5106057605641459]`. No accepted saved tube lies **entirely** outside the safety band on any of the eight accepted steps. The saved intervals therefore establish a numerical safety prefix only through `t=0.01`; afterward the box property is **undecided**. The prior one-period author driver printed `Unknown.`; this longer attempt has no final author property verdict because of the explicit first-rejection stop. Neither a real unsafe trajectory nor a full `T=0.4` flowpipe follows from these records. There is no full-horizon endpoint width, verification result, end-to-end floating-point NNCS proof, or four-method speed ranking from this attempt.

The third control refresh's directed controller residual width reached maximum **22.95178739274622** (median **15.935593892139352**) in the inner RESULT; this coincides with the first subsequent plant-step contraction failure. The saved data do not isolate a narrower causal threshold. The fixed 2026 more-robust contract and the other three new attempts remain in [the report draft](../../../../ARCHCOMP26_FINAL_REPORT_DRAFT.md); their early stops are separate outcomes.
