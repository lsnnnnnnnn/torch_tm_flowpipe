# Huan QUAD：40 步 strict / parity 诊断（2026-10-01）

两臂顺序使用服务器 `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/huan_quad_stage_a_40_20261001` 中全新的 `parity/`、`strict/` 目录；历史 1000 步作业未重启。原 Huan 分块引擎路径为 `N/engine_huan_sr_chunk`，使用 `N/nncs_env/bin/python`、GPU3（V100，UUID `GPU-1ad11bb9-50d4-6b9d-22f8-fc8c33180e56`）、CPU 14–17。GPU3 在两臂启动前均瞬时空闲，结束后也空闲。每臂外层 `timeout 600s`。本轮按用户要求没有做文件哈希或重新核对源码及二进制身份；历史身份仅来自已存记录，预编译扩展的路径、大小和修改时间见 `launcher_result.json`。

`quad_40.yaml` 只把原 `quad_official.yaml` 的 `steps: 50` 改为 `steps: 2`，并把原控制器 `model_dir` 写成同一个 ONNX 文件的绝对路径；逐字对照见 `config.diff`。其余配置含 B=1024 初盒、order 2、控制周期 0.1、ODE 步长 0.005、box / same-slope。`launcher.py` 直接加载已有三个 CUDA `.so`，在进程内禁止 JIT 构建，调用原 `gpu_driver.py` 并启用 `--metrics-json`。两臂使用独立新进程，strict 只多一个 `--strict` 参数。

| 单次诊断 | parity | strict |
|---|---:|---:|
| 外层退出码 | 0 | 0 |
| 内部求解 `metrics.elapsed_s` | 1.412598 s | 1.504988 s |
| `driver.main()` wall | 2.841967 s | 2.970675 s |
| `/usr/bin/time -v` 全进程 wall | 5.61 s | 5.70 s |
| `metrics.broken` | 0 | 0 |
| GPU allocator 峰值 allocated | 3,132,353,024 B | 3,132,646,912 B |
| GPU allocator 峰值 reserved | 5,779,750,912 B | 5,779,750,912 B |
| 最大 RSS | 1,326,700 KiB | 1,312,176 KiB |
| 终点各 lane 宽度和的均值 | 6.213007919 | 6.237998456 |

两份 metrics 都记录 B=1024、2 个控制期、每期 20 个 ODE 子步、两个控制期均 1024 条活跃 lane、`broken=0`；stdout 有 `Step 0`、`Step 1`，无 `Flow* terminated`。`broken=0` 是 driver 报告的终止分支数，单凭退出码不证明逐 lane 每步资格。

第 0 控制期两模式的状态/控制器宽度相同。第 1 控制期 x3 输入 hull 均值宽度为 parity `0.169481192484`、strict `0.170740534096`；u1 box 均值宽度为 `0.090817629713`、`0.091432635943`。t=0.2 的 x3 终点区间分别为 `[-0.277482570903, 0.544458403719]` 与 `[-0.278764581508, 0.545745560046]`，宽度分别为 `0.821940974622` 与 `0.824510141554`，strict 宽约 0.313%。终点 16 个变量的 strict 聚合区间均包含 parity 区间；完整原值在两份 `metrics.json`。

两份 stdout 都显示 `FALSIFIED`，因为把 `steps` 改为 2 后，原终点目标在 **t=0.2** 检查。这不是原 T=5 目标的判定。所有时间与宽度均为单次短程诊断，不可当成 1000 步正式测速或同保证的完整比较。

本地保存了 `quad_40.yaml`、`config.diff`、`launcher.py`，以及两臂的 `metrics.json`、`stdout.log`、`stderr.log`、`launcher_result.json`、`time.txt`、起止 UTC 与退出码。外层全进程 wall 包括 Python 启动与预载；driver wall 包括 driver 初始化及最终输出；内部计时从原 driver 的求解循环起点开始，在最终 verdict/hull 输出前结束。

## 独立 parity 阶段计时

随后另起 `profile_parity/` 新进程，同一配置与原 parity 模式，保留无 JIT 预载。`profile_launcher.py` 仅包装原 driver 的五个全局调用，按原参数调用、原样返回，并在调用前后同步 CUDA 与累计 wall；没有改动引擎或 driver 文件。外层仍为 600 s 上限、GPU3、CPU 14–17。五个包装点在 driver 主流程顺序调用，没有相互嵌套；其中 `hull_ranges_s` 的末尾 3 次发生在原内部计时结束后，因此五项合计不能直接当作内部耗时分解。

| 包装的调用 | 次数 | CUDA 同步 wall |
|---|---:|---:|
| `hull_ranges_s` | 5 | 0.001926 s |
| `crown_bounds` | 2 | 0.316279 s |
| `inject_controls_s` | 2 | 0.005563 s |
| `advance_sparse` | 40 | 1.051305 s |
| `prune_state` | 40 | 0.015192 s |

阶段计时这一次的内部 `metrics.elapsed_s` 为 1.405465 s，driver wall 为 2.864922 s，全进程 wall 为 5.54 s；外层退出码 0、`broken=0`。本地直接读取两份 JSON 比较，profile 与原 parity 的 `B`、`steps`、`substeps`、`broken`、全部 `ctrl_steps`、完整 `final_hull` 及 `final_hull_width_sum_mean` **数值相同**；五类调用次数也与 2 控制期 / 40 子步吻合。计时器的 CUDA 同步会扰动执行，且只是一条短程样本，不能外推为 1000 步正式成本比例。`profile_parity/` 与 `profile_launcher.py` 已复制到此本地目录，原始阶段秒数见 `phase_timing.json`。

## 执行命令

服务器上设 `N=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z`、`D=$N/runs/huan_quad_stage_a_40_20261001`。每条命令分别将 stdout/stderr 定向到相应子目录的 `stdout.log`/`stderr.log`；外层同时保存 `start_utc.txt`、`end_utc.txt`、`exit_code.txt`。原样的 driver 参数、GPU 映射、CPU affinity、扩展路径和尺寸也记录于各臂 `launcher_result.json`。

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONDONTWRITEBYTECODE=1 timeout 600s taskset -c 14-17 /usr/bin/time -v -o "$D/parity/time.txt" "$N/nncs_env/bin/python" -B "$D/launcher.py" parity "$D"
CUDA_VISIBLE_DEVICES=3 PYTHONDONTWRITEBYTECODE=1 timeout 600s taskset -c 14-17 /usr/bin/time -v -o "$D/strict/time.txt" "$N/nncs_env/bin/python" -B "$D/launcher.py" strict "$D"
CUDA_VISIBLE_DEVICES=3 PYTHONDONTWRITEBYTECODE=1 timeout 600s taskset -c 14-17 /usr/bin/time -v -o "$D/profile_parity/time.txt" "$N/nncs_env/bin/python" -B "$D/profile_launcher.py" profile_parity "$D"
```
