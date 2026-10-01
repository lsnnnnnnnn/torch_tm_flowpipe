# ARCH-COMP26 Double Pendulum more — Huan / Xiangru attempts

Date: 2026-10-01. These are new attempts under the fixed 2026 more-robust controller and the **full** `[1,1.3]^4` initial box partitioned into 225 cells. The old more runs used a corner and the wrong less controller; none of their time or width data is used here. No content digest was calculated.

The two methods used distinct saved author plant engines but the same Xiangru controller driver, native `[N,4]→[N,2]` layout, box/same-slope CROWN, RPC-float32 transfer, strict plant mode, and 0.005 s ODE step at Taylor order 4. The run-generated YAML was parsed field-by-field against the independent [new shared more YAML](../../../../../benchmarks/archcomp26/configs/double_pendulum_more_robust_paper_p4.yaml): all values agree. The GPU and native 225-cell ledgers have identical values and order. GPU2/CPU10–13 were used; raw supervisor, child, metrics, stdout and range files are saved in each local run subdirectory, with remote originals under `runs/archcomp26_20261001/author_dp_more_v1/`.

| Method | 225-box one-period smoke | 20-period attempt | Process wall | Core reported time | Saved coverage | Checker output |
| --- | --- | --- | ---: | ---: | --- | --- |
| Huan | 900/900 box-substeps accepted, tube inside safe band | Early stop after Step 17 | 11.685967 s | 4.958002 s | 225 × 72 ODE steps | `Unsafe.` |
| Xiangru | 900/900 box-substeps accepted, tube inside safe band | Early stop after Step 17 | 11.724106 s | 4.911208 s | 225 × 72 ODE steps | `Unsafe.` |

Both full attempts exited with wrapper code 1 because the requested 80 ODE steps were **not** completed; the shared driver itself returned 0 after its property early stop. `broken=0` and all 16,200 observed lane-substeps were accepted numerically. The two saved `ranges.bin` files are directly byte-equal. These are one-sample *early-stop* times, not T=0.4 completion times or independent correctness confirmations. The original controller coupling retains unqualified round-to-nearest arithmetic.

## Common safe prefix and transition

The independent [range scanner](../../../../../tools/archcomp26_scan_dp_ranges_nohash.py) found every lane/step pair present, finite and ordered at 0.005 s. In both methods, all saved four-state tubes through cumulative step 60 (`T=0.3`) lie in the `[-1.5,1.5]^4` band. The first crossing is step 61, in the lower bound of `θ̇₁`. The counts of crossing lanes in steps 61–72 are `1,20,40,45,52,74,88,90,90,113,135,135`. At step 72 (`t∈[0.355,0.36]`), 15 lanes have the **entire saved `θ̇₁` tube** below `−1.5`; these are lanes `207,208,209,210,211,213,216,217,218,219,220,221,222,223,224`. This supports why the shared checker printed `Unsafe.`, but is not an independently established trajectory counterexample or end-to-end floating-point proof. The native more attempt stopped at Step 15 with `UNKNOWN` and saved only 64 ODE steps, so it did not reach this time segment.

At the shared valid prefix `T=0.3`, the Huan and Xiangru saved endpoint intervals and widths are identical:

| State | Endpoint union at T=0.3 | Union width | Per-box width mean | Per-box width maximum |
| --- | --- | ---: | ---: | ---: |
| θ₁ | [0.9669259101684384, 1.2715233751919472] | 0.3045974650235088 | 0.0734120873231281 | 0.07949113776321393 |
| θ₂ | [0.8164686638510671, 1.090715775544863] | 0.274247111693796 | 0.06678010225557998 | 0.07298935093329884 |
| θ̇₁ | [-1.4857433529205841, -1.0125541894903975] | 0.4731891634301866 | 0.15663785101552707 | 0.18637696254415714 |
| θ̇₂ | [-1.4546853320806452, -0.9724255940484228] | 0.48225973803222244 | 0.1743587082864292 | 0.2025469413403017 |

The scanner's `COMMON60_SCAN.json` in each directory includes the corresponding whole-time tube unions and the exact crossing counts. As in the less runs, the shared GPU observer has small endpoint-versus-same-step-tube discrepancies; the maximum here is `4.6629367034256575e-15`. Do not claim exact endpoint-in-tube inclusion from these two files. Both endpoint and tube intervals are separately inside the safety band through step 60. `T=0.4` endpoints and widths do not exist for these attempts.
