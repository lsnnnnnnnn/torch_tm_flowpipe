# 2026 TORA remain：Huan / Xiangru 新尝试合同与入口

状态（2026-10-01）：**CPU-only 无哈希预检通过；新原生与新 P3 均完整运行 T=20，Huan/Xiangru 新 full 尝试均出现盒拒绝与 checker `Unknown.`**。本页只对应 `tora-remain`，不涵盖 TORA 的两个 reach 变体，也不复用旧运行时间或性质判定。原生完整结果见[独立摘要](evidence/results/archcomp26_20261001/native_tora_remain_full20_001/SUMMARY.md)，Huan/Xiangru 原始尝试与前缀见[作者方法摘要](evidence/results/archcomp26_20261001/author_tora_remain_v1/SUMMARY.md)，P3 见[新摘要](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp26_20261001/p3_tora_remain_v1/SUMMARY.md)。

## 固定任务与接口

| 项 | 本次选择 |
| --- | --- |
| 初集 | `(x1,x2,x3,x4) ∈ [0.6,0.7] × [-0.7,-0.6] × [-0.4,-0.3] × [0.5,0.6]`，保存的 x1 四分、x2 三分，共 12 盒 |
| 辅助状态 | `t=0, u1=0`；顺序 `x1,x2,x3,x4,t,u1` |
| ODE | `x1'=x2; x2'=-x1+0.1 sin(x3); x3'=x4; x4'=u1-10; t'=1; u1'=0` |
| 控制 | 每 1 秒更新一次；ONNX 原始输出 `f(x)` 注入 `u1`，由 ODE **仅减一次 10** |
| 时域和性质 | `T=20`；四个物理状态在整个 `[0,20]` 内都须位于 `[-2,2]` |
| 本次方法设置 | 保存配置的 Taylor order 3、ODE 小步 0.1、cutoff `1e-6`、初始余项 `[-0.01,0.01]`、SR queue 1000；两方同一共享控制驱动，分别加载 Huan / Xiangru plant engine；strict plant、box/same-slope CROWN、native ONNX 输入布局、RPC float32 控制系数传输 |

`research/gpu_verified_20260930/report/configs/arch_tora_homogeneous.yaml` 和同目录 12 盒 JSON 与远端保存配置、盒 ledger 分别经过直接字节比较，结果相同；没有计算内容摘要。远端实际共享驱动 `.../xiangru_upstream/src/flowstar_gpu/integrations/crown_reach.py` 也与本地冻结的 `research/gpu_verified_20260930/source/integration/crown_reach.py` 直接字节相同。保存的 2024 路径 ONNX 与本次 prep 中官方 2026 ONNX 在远端直接字节相同；新运行仍显式指向独立 prep 的官方 2026 文件。以上只记录比较发生时的直接相等，不提供不可变内容绑定。

驱动的 `build_raw_net` 读取 ONNX；`crown_bounds` 在 `output_scale=1, output_offset=0` 下返回原始网络包络；`inject_controls_s` 将其写入末尾 `u1` 行；`compile_ode` 编译保存的 `u1-10`。官方模型的动态输入是四维 `input`，输出形状 `[1,1]`，尾部 `Conv→Relu→Flatten`；输入 `Sub` 的 `input_Mean` 为全零，网络末尾没有额外的减 10 操作。CPU 中心点 `[0.65,-0.65,-0.35,0.55]` 得到原始 `f(x)=10.022441531381736`，对应 plant 输入 `0.022441531381735658`。这是一点接口预检，不是区间正确性或性质证明。

共享驱动在每个 ODE 小步对 `constraints_safe` 进行全时间段检查；无 `constraints_safe_from/_until` 窗口。它在安全集合不能证明时打印 `Unknown.`，证明完全越界时打印 `Unsafe.`；静默成功须连同完整步数和全部盒接受数才能解释。两个方法共享控制驱动和大部分数值实现，所以相同范围不能当作两份独立正确性证明。控制器浮点包络注入仍无端到端 NNCS 证书。

## CPU-only 预检与新运行入口

[预检记录](evidence/results/archcomp26_20261001/author_tora_remain_v1/PREFLIGHT.json)由 [独立入口](../tools/archcomp26_tora_remain_author_nohash.py) 在远端保存 Python 环境运行产生；`gpu_initialized=false`，12 盒、配置、模型输入输出和原始控制边界通过。远端脚本位于 `N/runs/archcomp26_20261001/author_tora_remain_v1/`，其中 `N=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z`。

CPU 重算命令（`CUDA_VISIBLE_DEVICES=` 保持空）：

```bash
cd "$N/runs/archcomp26_20261001/author_tora_remain_v1"
CUDA_VISIBLE_DEVICES= "$N/nncs_env/bin/python" -B archcomp26_tora_remain_author_nohash.py \
  --preflight --output PREFLIGHT_recheck.json
```

GPU 运行前须重新检查 GPU 与 CPU 占用。2026-10-01 的实际 Huan/Xiangru 新尝试在原生 TORA 退出、GPU2 和端口 5100 复查为空后，依序使用 GPU2/CPU10–13。每种方法先做新的 `smoke1`，再在独立目录做 `full`；不要重启任何旧作业。外层使用现有 `tools/run_archcomp26_nohash.py` 监督，`--run-dir` 必须是新路径。以下是已执行的 Huan smoke 入口（只供审计，不要对同一目录重复启动）：

```bash
cd "$N/runs/archcomp26_20261001/author_tora_remain_v1"
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 10-13 \
  "$N/nncs_env/bin/python" -B run_archcomp26_nohash.py \
  --run-dir "$PWD/huan_smoke1_001" --instance tora-remain --method huan \
  --contract-label tora-remain-2026-b12-order3-v1 --cwd "$PWD" --timeout-s 1800 \
  --env CUDA_VISIBLE_DEVICES=2 --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 -- \
  "$N/nncs_env/bin/python" -B archcomp26_tora_remain_author_nohash.py \
  --backend huan --mode smoke1 --output huan_smoke1_001/payload
```

把 `huan` 换为 `xiangru` 可启动另一方；`smoke1` 完成后把模式换成 `full`，并使用全新的 `*_full20_001` 目录。入口用 `mkdir(exist_ok=False)` 和外层 no-hash supervisor 的新目录要求拒绝覆盖已有尝试。两个新目录各自保存 `START.json`、`RESULT.json`、stdout/stderr、`payload/config.yaml`、`payload/ranges.bin`、逐小步 `payload/observations.jsonl` 和 `payload/metrics.json`。结构化范围中每个 `(lane,substep)` 存四个状态的 whole-step tube 与 step endpoint 两种区间；`payload/RESULT.json` 还记录进程 wall、接受计数、已观察前缀 tube union、最后观察到的 endpoint union 与作者 checker 的实际输出解释。只有完整 `T=20` 才填全时 tube 和终点 endpoint 字段；`smoke1` 只到 `T=1`。

Huan/Xiangru 的 T=1 smoke 均接受 120/120 盒步；T=20 尝试各观察到 200 个小步，但只接受 2357/2400 盒步，首次保存 tube 越出安全带在第 185 步，首次盒拒绝在第 190 步，checker 均为 `Unknown.`。两份失败 full 尝试的进程 wall 分别为 8.371523 s 和 8.270158 s；没有合格的全初集终点宽度或完整安全结论，不能与新原生完成时间作全程速度排名。保存区间和静默/Unknown 判定均不构成独立端到端浮点 NNCS 证明。

## P3 新尝试：完整 12 盒安全范围

使用固定同一份配置、12 盒 ledger 与官方 ONNX，P3 在 GPU3/CPU14–17 的新隔离目录 `N/runs/archcomp26_20261001/p3_tora_remain_v1/` 先过一周期 smoke：120/120 盒步接受，只构成 `T=1` 前缀。随后完整 `T=20` 运行接受 2400/2400 盒步；作者 checker 无失败输出，进程 wall 10.411001097 s，driver elapsed 6.216789040 s。保存的 2400 条四态 whole-step tube 与 endpoint 经[独立扫描](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp26_20261001/p3_tora_remain_v1/full20_001/INDEPENDENT_INTERVAL_SCAN.json)全部有限、有序，endpoint 位于同段 tube，四态 tube 全时处于 `[-2,2]`。最后端点 union 为 `x1∈[-0.12790393433470768,0.00945220392405179]`、`x2∈[-0.2733529513870905,-0.09659560631766524]`、`x3∈[-0.0652936777633828,1.0333656496375623]`、`x4∈[-0.8155275940424564,0.4176773220886647]`；全时 tube union 与更多收据见[P3 摘要](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp26_20261001/p3_tora_remain_v1/SUMMARY.md)。这些是单次浮点数值运行及保存盒的重检，无独立端到端 NNCS 证书；与原生/Huan/Xiangru 的不同观察器、数值路径和完成状态不能直接作稳定速度排名。
