# Xiangru 2026 experiment code audit

## Result

The expected private checkout `/srv/local/shengenli/CROWN-Reach_Development` is absent. No matching private archive or raw B12/B48 timing JSON/log was found at the workspace root. Therefore the private source identity, runner-to-result mapping, B12/B48 meaning, completed horizon, property verdict, and timing boundaries are all `blocked_missing_source`.

The machine-readable inventory is `outputs/three_tool_reaudit/20260804T060058Z/raw/xiangru_source_inventory.json`. It explicitly sets `headline_b48_ratio_allowed=false` and `historical_timing_recomputed=false`.

## Public release is not a substitute

The distinct public repository exists at `/srv/local/shengenli/CROWN-Reach`, remote `https://github.com/Verified-Intelligence/CROWN-Reach`, SHA `7b90f308...`, clean `main`. Nothing in the available filesystem establishes that it is the missing private 2026 experiment tree.

The public C++ controller route calls CROWN, unpacks `T`, `u_max`, and `u_min`, and constructs Flowstar Taylor models (`src/CrownReach.cpp:77-115`). It uses JSON `asFloat()` for all three coefficient families (`:98-101`). This records a float32 transfer on that historical public route; it does not quantify the private runner's behavior.

The requested private paths could not be audited:

- `src/crown.py` plus the private controller/Flowstar route;
- `experiments/reachability/` native CROWN-Reach/DiffReach runner;
- `experiments/adaptive_tm/` complete-Q3, DR-RP, tensor, and GPU routes;
- result JSON producer identity and B12/B48 workload definition;
- controller period, internal step count, T=20 completion, and property verdict;
- `core/controller/dynamics/compile/cold/warm/process` timers and CUDA synchronization;
- `nextafter` coverage and high-precision leaf-step replay;
- float32 `0.10000000149011612` versus binary64 `0.1` model-error handling.

## Historical numbers

The approximate figures in the goal text—Flowstar B12 core `~2.072 s`, Q3/DR-RP B48 core `~2.185 s`, and fresh process `~5.865 s` versus `~39.475 s`—were not present as raw artifacts and were not recomputed. They remain unverified targets, not findings. In particular, no ratio between a B12 and B48 workload is reported.

## Minimum unblock condition

Provide the private/archive source plus raw JSON/logs and hashes. If the source is an archive without `.git`, its archive SHA256 can be recorded, but no commit may be inferred. A future matched B48 run must replay identical 48 leaves and controller bands, check `lA/uA` slopes and band domains, unify the `0.1` dtype, complete T=20 and the same property, and collect 1 cold plus 10 synchronized steady timings. Until then Xiangru has no eligible row in any headline comparison.
