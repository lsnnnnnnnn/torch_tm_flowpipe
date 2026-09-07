# 旧副本盘点

只读检查，没有切分支、拉取、清理、安装或重编译旧目录。进程检查仅检查本用户可读的 cwd 和进程名；不读取完整参数。

| 角色 | 绝对路径 | HEAD | 分支 | 工作树 | 本用户 cwd 任务 |
|---|---|---|---|---|---|
| our_repository | `/srv/local/shengenli/torch_tm_flowpipe` | `26a254ef585a9dee394b7e41922c06bf8799f501` | codex/flowstar-raw-remainder-compat | 有修改/未跟踪内容 | 0 |
| our_frozen_evidence | `/srv/local/shengenli/torch_tm_flowpipe_vdp_c3_huan_20260827` | `ed9c305dc39c25eab23a96f4fb3775cc2d13d396` | codex/torch-flowstar-brusselator-live-range-c5-20260828 | 干净 | 0 |
| our_performance_evidence | `/srv/local/shengenli/torch_tm_flowpipe_c4_perf_batch_20260829` | `df50c55ef640b0ca9d90b132c88235b8ec07510b` | codex/c4-reference-performance-batch-foundation-20260829 | 干净 | 0 |
| our_exact_witness_evidence | `/srv/local/shengenli/torch_tm_flowpipe_huan_proof_closure_20260826` | `6037cfeac496d606a72e9c14ecbfd979b1cd7479` | detached | 干净 | 0 |
| old_xiangru | `/srv/local/shengenli/CROWN-Reach_Development` | `84184de6c2b3f1ff2da6755f732d91925037025d` | 2026_experiment | 干净 | 0 |
| old_xiangru_native | `/srv/local/shengenli/CROWN-Reach_Development_native_27d2905` | `27d29050a5f214b56f211ca9cb411e734ed80230` | detached | 干净 | 0 |
| old_xiangru_native | `/srv/local/shengenli/CROWN-Reach_Development_native_9bf2cce` | `9bf2ccea781cd47eca1d5ab5954f4e508ee8192f` | detached | 干净 | 0 |
| native_flowstar | `/srv/local/shengenli/flowstar` | `b85a3211748cb77b736fe4ad42ee02d8d2b81148` | master | 有修改/未跟踪内容 | 0 |
| old_huan_reference | `/srv/local/shengenli/flowstar-gpu` | `d5f0b68fcd36ba5f582733624f074728fe9720d8` | main | 干净 | 0 |
| old_huan_patched_reference | `/srv/local/shengenli/flowstar-gpu-proof-closure-20260826` | `743f6205e6408072193ad76e940e7f15030e8d3c` | codex/strict-proof-contract-closure-20260826 | 干净 | 0 |

完整的已修改文件名、脱敏 remote、worktree 清单和旧证据位置保存在本轮 raw_minimal/old_clones_inventory.json。
主要旧 Xiangru 为 84184de6；新版新增仓库内 GPU 引擎、相关测试和来源许可记录。旧 Xiangru 工作树干净，没有必须移植的未提交修补。原生 Flow* 和用户主工作树有本地改动，均原样保留。
