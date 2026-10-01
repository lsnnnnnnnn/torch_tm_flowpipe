# Double Pendulum less robust: four saved full runs

This is a read-only comparison of the **2026 continuous** `[1,1.3]^4` initial box split into 225 boxes, with 20 held-control periods and 100 ODE substeps of 0.01 s through `T=1`. Physical state order is `(θ₁, θ₂, θ̇₁, θ̇₂)`; the all-time safety box is `[-1.7,2]^4`. No old experiment was restarted. The [generator](../../../../../tools/archcomp26_dp_less_fourway_saved_nohash.py) reads the four saved numerical outputs directly. Its [endpoint CSV](endpoint_stats.csv) keeps full decimal values, and [tube CSV](tube_union.csv) keeps every method, dimension, and step union. The [PNG](fourway_tube_union.png), [PDF](fourway_tube_union.pdf), and [MATLAB script](fourway_tube_union.m) show the four physical dimensions. In each panel, a black segment identifies the initial `[1,1.3]` interval at `t=0`, and pale green identifies the visible part of the Safe `[-1.7,2]` band; its boundaries extend beyond the cropped y-axis. The MATLAB script was generated but not run in MATLAB/Octave.

## Sources and coverage

| Method | Saved numerical source | Supervisor wall | Driver/solver time | Saved coverage |
| --- | --- | ---: | ---: | --- |
| native | [ranges.bin](../native_dp_less_full20_001/ranges.bin), [result](../native_dp_less_full20_001/RESULT.json), [checker log](../native_dp_less_full20_001/native.log) | 1107.127423 s | solver log 1100.101000 s | 22,500 unique lane-step records; checker `VERIFIED`; binary records omit acceptance/status |
| Huan | [ranges.bin](../author_dp_less_v1/huan_full20_001/payload/ranges.bin), [child result](../author_dp_less_v1/huan_full20_001/payload/RESULT.json), [supervisor result](../author_dp_less_v1/huan_full20_001/RESULT.json) | 9.539174 s | 8.669310 s | 225 × 100 accepted |
| Xiangru | [ranges.bin](../author_dp_less_v1/xiangru_full20_001/payload/ranges.bin), [child result](../author_dp_less_v1/xiangru_full20_001/payload/RESULT.json), [supervisor result](../author_dp_less_v1/xiangru_full20_001/RESULT.json) | 8.387790 s | 7.488113 s | 225 × 100 accepted |
| ours/P3 | [observations.jsonl](../dp_p3_affine_split4_v1/full225_smoke_001/attempt/data/observations.jsonl), [child result](../dp_p3_affine_split4_v1/full225_smoke_001/attempt/data/RESULT.json), [supervisor result](../dp_p3_affine_split4_v1/full225_smoke_001/attempt/RESULT.json) | 74.274085 s | 73.304367 s | 225 × 100 accepted, status 0; strict end-to-end certificate false |

All four saved data sources cover steps `1–100`, lanes `0–224`, with finite ordered intervals. Each method's 90,000 physical tube **coordinate intervals** are within the safety band. The P3 saved JSONL also records every lane accepted, valid tube/endpoint, status 0 and within the safety box. The native binary has no accepted/status field; its `VERIFIED` is the author checker's separate log verdict. Huan/Xiangru acceptance counts come from their adjacent child results, not from the binary range format. The Huan and Xiangru four-state saved range arrays are exactly equal across all 22,500 records. Their shared numerical source and driver mean this equality is not independent corroboration. Huan/Xiangru have 2,927 endpoint-lower and 368 endpoint-upper coordinate values just outside the corresponding recorded tube intervals, with maximum gap `5.551115123125783e-15`; both interval views remain in the safety band. Native and P3 have no such mismatch in the saved records.

## Absolute interval widths at T=1

The endpoint union is the hull of **225 saved endpoint boxes at step 100**. Per-box mean and maximum use 225 individual interval widths; the all-time tube union scans every saved tube box at steps `1–100`. These are interval enclosures, not sampled trajectories. Values below are rounded for display; use the CSV for the saved decimal values.

| Method | State | Endpoint union at T=1 | Union width | Per-box mean | Per-box max | All-time tube union |
| --- | --- | --- | ---: | ---: | ---: | --- |
| native | θ₁ | [1.293126787, 1.777127132] | 0.484000345 | 0.194361064 | 0.329011998 | [0.999999692, 1.904628248] |
| native | θ₂ | [-0.124833765, 0.441682210] | 0.566515975 | 0.194340174 | 0.329318718 | [-0.124833765, 1.357787325] |
| native | θ̇₁ | [-0.840506751, -0.044472647] | 0.796034104 | 0.234146755 | 0.743550142 | [-0.840506755, 1.696912177] |
| native | θ̇₂ | [-1.618490165, -0.601905269] | 1.016584896 | 0.315835065 | 1.016584896 | [-1.618490165, 1.300042761] |
| Huan | θ₁ | [1.290764319, 1.776794969] | 0.486030651 | 0.192945369 | 0.333805930 | [0.999435247, 1.904318218] |
| Huan | θ₂ | [-0.124282314, 0.444401061] | 0.568683375 | 0.192918900 | 0.337692186 | [-0.124297173, 1.357918762] |
| Huan | θ̇₁ | [-0.865377576, -0.012398205] | 0.852979372 | 0.237215787 | 0.807804621 | [-0.865413574, 1.700873199] |
| Huan | θ̇₂ | [-1.663990572, -0.554202697] | 1.109787875 | 0.317152859 | 1.109787875 | [-1.667565097, 1.318757489] |
| Xiangru | θ₁ | [1.290764319, 1.776794969] | 0.486030651 | 0.192945369 | 0.333805930 | [0.999435247, 1.904318218] |
| Xiangru | θ₂ | [-0.124282314, 0.444401061] | 0.568683375 | 0.192918900 | 0.337692186 | [-0.124297173, 1.357918762] |
| Xiangru | θ̇₁ | [-0.865377576, -0.012398205] | 0.852979372 | 0.237215787 | 0.807804621 | [-0.865413574, 1.700873199] |
| Xiangru | θ̇₂ | [-1.663990572, -0.554202697] | 1.109787875 | 0.317152859 | 1.109787875 | [-1.667565097, 1.318757489] |
| ours/P3 | θ₁ | [1.283893321, 1.782483200] | 0.498589879 | 0.211252269 | 0.320056205 | [0.999437221, 1.906080152] |
| ours/P3 | θ₂ | [-0.134899028, 0.433984352] | 0.568883381 | 0.211966101 | 0.305682489 | [-0.134911067, 1.358125658] |
| ours/P3 | θ̇₁ | [-0.788599580, -0.117156994] | 0.671442586 | 0.287469227 | 0.634061308 | [-0.788637798, 1.701834617] |
| ours/P3 | θ̇₂ | [-1.491083926, -0.691993912] | 0.799090015 | 0.382361002 | 0.784384879 | [-1.563737338, 1.318928245] |

The P3 velocity endpoint unions are narrower than the other saved unions, while its mean per-box widths are larger. Neither statistic alone establishes a method ranking. The four supervisor times come from one run each and have different engine, hardware, and observation costs. P3's driver time explicitly includes per-lane diagnostic observation and JSONL writing. Huan/Xiangru still use unqualified round-to-nearest coefficient injection and CROWN floating bias; P3's own result marks the end-to-end strict certificate false. The saved-box scan checks interval consistency and the safety band, not full neural-network reachability soundness. Source files and adjacent run declarations are cited by path and size; no content identity claim is made.
