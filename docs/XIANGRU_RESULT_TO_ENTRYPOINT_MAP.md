# Xiangru 保存结果到原生入口映射

本文件只建立复现对象与作者入口之间的映射。它不把进程退出码等同于复现，
也不把 one-step 结果当作端到端复现标准。

## 证据边界

- 审计仓库：`/srv/local/shengenli/CROWN-Reach_Development`
- 远端：`https://github.com/xiangruzh/CROWN-Reach_Development.git`
- 分支：`2026_experiment`
- fresh fetch 后本地与远端 SHA：
  `84184de6c2b3f1ff2da6755f732d91925037025d`
- 当前分支干净、无 submodule。完整 tracked-file inventory 在
  `outputs/native_reproduction_no_adapters/20260804T081205Z/xiangru/tracked_file_inventory/stdout.log`。
  该 inventory 有 867 行，其中 4 个 README、106 个 experiment Python/shell
  文件、411 个 tracked result 文件、22 个 config/benchmark YAML/JSON。
- 主要保存结果自身记录了不同的生成 SHA。因此历史结果必须在对应 detached
  worktree 上复现，不能用当前 `84184de` 的运行替代。
- 保存结果中出现的 `/home/xiangru4/...` 是原机器绝对路径；本轮若使用 symlink，
  只允许把相同输入字节映射到该路径，并在 manifest 中记录。

## B12、B24 与 B48 的确切含义

这些标签不是 batch-size tuning 的抽象名字，而是 TORA 初始集合的静态
Cartesian partition 数量。代码入口为
`experiments/remainder_ablation/tora_static_grid.py`，保存的 S0 控制字段为：

| 标签 | `(x1,x2,x3,x4)` splits | 初始 leaves |
|---|---:|---:|
| `b12_archcomp` | `(4,3,1,1)` | 12 |
| `b24_static` | `(8,3,1,1)` | 24 |
| `b48_static` | `(8,6,1,1)` | 48 |

每个控制周期有 10 个 0.1 秒 plant segment，目标 horizon 为 20 秒；所以完整
B48 run 有 200 segments、9,600 leaf-segment attempts 和 960 controller boxes。
`batch` 在 tensor 实现中恰好承载这些 leaves，但 B 标签的实验语义是 initial
partitions，不是任意吞吐 batch。

## 结果—入口映射

### X1：CROWN-Reach controller + Flow* plant，TORA B12

- claim/result：
  `experiments/reachability/results/s3c_fair_timing_raw/flowstar_rep1_reachbench/run.json`
  (`sha256=99de29173076e5f10c40a22cfec4dea5675be197ff3d3c4daa2403bb9c494aa6`)
- raw stdout：相邻 `stdout.log`
  (`d64b1b86ffd88fa0a1d9b4c3b380725a36928889edbe2361a4a413bb1514bb22`)
- generating source SHA：`27d29050a5f214b56f211ca9cb411e734ed80230`
- native entrypoint：`scripts/reachbench_crown_docker.sh` 调用仓库已有
  `reachbench.py`、submit-package `crown.py` 与 `./Tora` binary。
- exact outer command：

  ```bash
  scripts/reachbench_crown_docker.sh run tora_homogeneous \
    --backend crownreach_submit --profile full --timeout 180
  ```

- config：保存的 `crownreach_submit_config.json`
  (`9aa0d59810a54fcd0accdaaed53746fed8fb578c9021985a724f2aedca410178`)
- benchmark manifest：`experiments/reachability/benchmarks/tora_homogeneous.json`
  (`d6510b5bf927a8ffeb8455e62ed25c06a0d7d504c8a1708580f2fcc04c9e0d10`)
- controller：`ARCH-COMP2024/benchmarks/Benchmark9-Tora/controllerTora.onnx`
  (`52a50c6bc6b1b45b89319edc809cd4d3baca06c32f9fd2ddceb2e95007414418`)
- expected output：12 partitions，20 controller updates，requested/reached
  horizon 20，`VERIFIED`，无 `Flow* terminated`。
- success definition：完整 20 秒 tube 的四个 state 均在 `[-2,2]`；不能只看
  exit 0 或最后 endpoint。
- fresh outcome：作者 27d Dockerfile 的 rootless image 已重建并核验；20/20
  updates、T=20、`VERIFIED`、无 Flow* termination。指定字段 exact comparison
  通过，状态 `reproduced_exact`。证据在
  `xiangru/x1_crownreach_flowstar_full_rootless/`。

### X2：Xiangru 的 upstream DiffReach/JAX whole-verifier TORA run

- claim/result：`experiments/remainder_ablation/results/u0_diffreach_native_gpu.json`
  (`8ea83095c9947b75c4b74bf0e675f45f9db81d95210e4d61a13c503b35518dc4`)
- generating CROWN-Reach provenance SHA：
  `9bf2ccea781cd47eca1d5ab5954f4e508ee8192f`
- upstream DiffReach SHA：`dd628eb443b517d6415de93e7035b4baef73963e`
- native entrypoint：DiffReach 自带 `run_ctl.py ... --ver`，不是本轮新增 runner。
- exact command：

  ```bash
  CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  <diffreach-python> run_ctl.py \
    <byte-identical-u0_diffreach_gpu_config.json> --ver
  ```

- config hash：
  `354b8f88c8f45161e10cc6e528a355aa29b15c3faf486d7cc662cb63f5f478fb`。
  它固定 12 partitions、200 个 0.1 秒 steps、controller period 1 秒、
  `init_remainder=0.01`。
- expected output：进程完成，但第一个 false shrink mask 位于 segment `[7.9,8.0]`；
  最多只到 7.9 秒的 Picard certificate，不能标 20 秒 verified。
- reference success definition：这是保存的失败边界；fresh run 只有重现同一失败
  语义时才可能是 `reference_failure_reproduced`。
- important scope：config 使用 Xiangru reachbench 已有的
  `controller_transformed.onnx`。因此这是“Xiangru-DiffReach experiment”，不是
  upstream 官方 TORA benchmark，也不进入无 adapter 的跨工具 comparison。
- fresh outcome：现场建立了 exact JAX/JAXlib 0.8.3 dependency stack。V100 GPU
  进入 verify 后在 cuDNN float64 convolution autotune 报
  `NOT_FOUND: No valid config found`，T=0，状态 `environment_failed`（GPU backend
  execution failure，不是 native algorithm rejection）。
  分开的 CPU supplemental 完成 200 steps，但只有 66.22% shrink flags，fresh
  NPZ 与作者 CPU NPZ byte-identical；这是 `reference_failure_reproduced`，不是
  verified。

### X3：PyTorch complete-Q3 + DR-RP + CUDA，B48 全 20 秒

- claim/result：
  `experiments/reachability/results/s3c_fair_timing_raw/rep1_q3/s3r_q3_b48_rep1.json`
  (`5ae2abf56e7ba25c34931db12b5d51e95aedcdc15a1eb00518c778d1b939b15a`)
- wrapper：相邻 `lane_result.json`
  (`7709facd5995e5faca14ec9544197c2c08e43f4960d1a68f13829d06cf55853d`)
- generating source SHA：`27d29050a5f214b56f211ca9cb411e734ed80230`
- native author entrypoint：
  `experiments/remainder_ablation/run_s0_tora_static_partition_sweep.py`
  (`sha256` at that SHA:
  `9789664a933490cdbf198063cff02b078a1e60e08099e87c9fcc1870f2be5623`)
- exact command：

  ```bash
  <crownreach-python> experiments/remainder_ablation/run_s0_tora_static_partition_sweep.py \
    --device cuda --policies b48_static --methods complete_q3 --cap 200 \
    --q3-engine dynamic --controller-backend autolirpa \
    --controller-platform cuda --controller-mode eager \
    --controller-composition outward --diffreach-root <DiffReach-root> \
    --diffreach-python <diffreach-python> \
    --output-json <fresh.json> --output-markdown <fresh.md>
  ```

- expected output：`VERIFIED`，200 segments，20 秒，48 leaves，无 retry segment，
  full-tube property 通过。
- configuration：complete total-degree-3 support、2 polynomial Picard rounds、
  DR-RP 10 remainder rounds、default seed 0.01、retry seed 0.1、natural range
  validator、symbolic capacity 200、float64。
- timing boundaries：保存结果分开记录 cache/JIT warm、controller worker startup、
  controller、dynamics、validation-excluded 和 process wall；本轮不会把它们与
  Flow* 的不同 timing boundary 混成 speedup。
- fresh outcome：Python 3.11.15、PyTorch 2.8.0+cu128、exact auto_LiRPA
  `5a098e8...` 与 V100 上完成 200/200 segments。2,850 个非 runtime 数值字段
  max abs `1.421e-13`，低于作者 `CONTROLLER_TOLERANCE=1e-6`；状态
  `reproduced_with_declared_tolerance`。设备不同，因此不比较 timing。

### X4：历史 B12/B24/B48 static sweep

- claim/result：
  `experiments/remainder_ablation/results/s0_tora_static_partition_sweep.json`
  (`ebf60960332294d4dd5d3a7664a5d25b88e18780b09dd3d5b13aea326076faeb`)
- native entrypoint：同 X3，保存 command 为
  `<crownreach-python> .../run_s0_tora_static_partition_sweep.py --device cuda`。
- saved outcomes：complete-Q3 B12 在 segment 151 失败，B24 在 segment 181
  失败，B48 完成 200 segments 并 `VERIFIED`。
- source identity blocker：结果记录 `crown_reach_commit=0a2cfcf...` 同时记录
  `crown_reach_dirty=true`，却没有保存当时 patch/diff。按照复现标准，该具体
  result 的 source identity 不可恢复。clean-commit 重跑只能作为另一条 native
  run，不能声称精确复现这个 dirty reference。
- fresh clean-27d outcomes：B12 attempted segment 151 失败、certified T=15.0；
  B24 attempted segment 181 失败、certified T=18.0。两行均保持
  `source_identity_unknown`。

### X5：PyTorch DR13/complete-Q3 static-route CUDA implementation gate

- claim/result：`experiments/remainder_ablation/results/m4_q3_gpu.json`
  (`a8ba4f21a5a70b7805b6db7d23cf586451580ddd62e374ab6175c8717783a7eb`)
- source SHA：`4be3d2b493a124e77497380d42a9553d0bb75207`
- entrypoint：`experiments/remainder_ablation/run_m4_q3_gpu.py`，无参数。
- scope：B12 first control period/10 segments 加 repeated-leaf throughput，
  CPU/PyTorch/CUDA field equivalence tolerance `1e-12`。
- expected output：两种 support 均 `PASS`；它是 implementation gate，
  不是 long-horizon reproduction，也不证明 formal outward rounding。

### X6：仓库已有 Flow*—PyTorch shared first-step diagnostic

- claim/result：
  `experiments/remainder_ablation/results/m4_flowstar_shared_step_cpu.json`
  (`20268daa2f2360c88015b3548688ac13c028bb25c87f2bdae7cf2b4fb350eab5`)
- analysis source SHA：`e84c84ed01f4b3ccfa22539a3c54160822451396`
- entrypoint：`experiments/remainder_ablation/run_m4_flowstar_shared_step_export.py`。
- scope：B12、首个 0.1 秒、Flow* order 3；只读分析现有 semantic trace。
- eligibility：trace 来自 `1ef9428...` 的 observation-only patch，保存结果自己也
  记录 `observer_patch`。因此若重跑，只能是 `patched_diagnostic_only`，绝不进入
  native reproduction matrix，也不能支撑 long-horizon/timing 结论。
- execution decision：X1 与 Q3 end-to-end fresh run 已完成；源码与 raw
  end-to-end evidence 已能定位 basis/remainder/reset 差异，故本轮为
  `not_needed`，不扩张 patched trace。

## Cache 与旧结果读取审计

- `reachbench.py` 为每次 run 创建 timestamped result directory；它运行真实 backend，
  不会仅复制旧 JSON。但其结果目录位于 Xiangru repo 的 ignored `results/`，本轮必须
  在完成后复制到主仓库 fresh run directory 并 hash。
- S0 runner会读取被冻结的 boundary、prior DR/Q3 result、matrix 与 plan 进行输入和
  source-classification checks；它仍实际执行 propagation。所有读取文件及 hash 必须
  与目标 generating SHA 对应。
- S3C Q3 lane可复用预填 TorchInductor cache。保存 repetition 的 exact command 使用
  shared cache；单次 native reproduction若没有 byte-identical cache，只能比较数值与
  completion 字段，不能声称 cache/timing 精确复现。
- U0 的 `run_ctl.py` 实际运行 verifier 并写 fresh `flowpipe_ver.npz`；配置中的 output
  directory 必须是 fresh。由于 config 有硬编码绝对路径，路径映射不得改变 controller、
  dynamics 或 config 字节。

## 当前命令确定性结论

X1、X2、X3、X5、X6 的命令由保存 command/`gnu_time.txt`、runner 和 README 共同
确定。X4 的 command 可确定，但 dirty source bytes 不可确定，故其阻断是
`source_identity_unknown`，不是 `reference_command_ambiguous`。最终状态以
`benchmarks/native_reproduction_registry.json` 为准。
