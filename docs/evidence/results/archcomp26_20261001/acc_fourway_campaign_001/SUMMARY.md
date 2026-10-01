# ACC 四方新进程计时 campaign（2026 participant-order 合同）

本目录是 2026-10-01 在服务器上新建的 24 次独立运行证据副本；远端原始目录：`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/acc_fourway_campaign_001`。本轮只按原始事件、日志和保存范围复核，没有运行内容摘要校验。

## 合同与计时口径

- 合同：`docs/ARCHCOMP26_ACC_PARTICIPANT_CONTRACT_20261001.md` 中明确命名的 participant-order profile。控制器输入为 `[30, 1.4, v_ego, x_lead-x_ego, v_lead-v_ego]`；官方固定 2026 ONNX，`T=5 s`、每次 50 个 `h=0.1 s` 控制期，连续全时安全半空间 `x_lead-x_ego-1.4*v_ego-10 >= 0`。论文没有定义 `v_rel` 的方向，因此这些结果只能称为该参与者输入顺序 profile。
- 全部按一盒六物理态、同一物理 GPU 2、CPU 10–13 顺序运行；native 额外使用独占 RPC 端口 5102。每次启动前有资源/端口预检，进程之间间隔 2 s。GPU 1 上有另一项 QUAD 作业，因此这些是共享主机下的描述性计时。
- `outer_wall_s` 来自同一个 supervisor 的单调时钟，从启动子进程前到子进程回收后。native 子进程包含配对 RPC 服务启动与求解；GPU 子进程包含 Python、ONNX/CUDA 初始化与驱动。期与期之间的 2 s 间隔不计入单次 wall。子进程等待有轮询粒度，小于约 0.05 s 的差异不应解释为方法性能。每次上限 120 s。
- round 0 是本 campaign 每方法首个**新进程**（cold_process）；主机和 GPU 之前已使用，并非重启后的冷机。round 1–5 是每方法五次独立新进程（steady_process），每轮轮换方法顺序，不是在一个进程里循环。

## 审计结果

24/24 个子进程 exit 0、未超时，24/24 完成 50 期。native 每次保存 50 条范围、50 次 RPC 请求及 50 条 HTTP 200 响应，日志 `Step 0`–`Step 49` 和作者 `VERIFIED`。其余三方每次 50 条 accepted 范围、50 条安全事件与 50 次 feature/injection 调用，作者 checker 为全时安全。独立按每段保存的六态轴对齐 tube 计算安全半空间下界，24/24 均严格大于零；这只是已保存范围的性质复核，不独立证明 NN 浮点界或所有底层运算。

| 方法 | 首次进程 wall (s) | 后 5 次 median (s) | 后 5 次 min–max (s) | 保存 tube 最小安全余量 |
| --- | ---: | ---: | ---: | ---: |
| Flow* native RPC | 7.686284 | 7.736304 | 7.636556–7.786681 | 16.254753759211 |
| Huan GPU | 8.087964 | 8.237208 | 8.138279–8.337823 | 16.165034707564 |
| Xiangru GPU | 7.887486 | 7.987532 | 7.937743–8.086614 | 16.165034707564 |
| ours/P3 | 8.589842 | 8.740510 | 8.637965–8.840520 | 16.434858569717 |

## 逐次原始事件

下表的 wall 来自各运行目录的 `RESULT.json`，与 [`events.jsonl`](events.jsonl) 逐项一致。`RPC` 只表示 native 实际 HTTP 控制器请求；GPU 行的“控制器调用”是适配器 feature/injection 调用，不混记为 RPC。方法输出均为一盒完整 `T=5 s` 的保存结果。

| 轮次 | 方法 | 阶段 | wall (s) | 期数 | RPC | 控制器调用 | 性质结论 | 原始目录 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 0 | Flow* native RPC | 首个新进程 | 7.686284 | 50 | 50 | 50 | 作者 VERIFIED；保存 tube 安全 | [cold00_native](cold00_native/) |
| 0 | Huan GPU | 首个新进程 | 8.087964 | 50 | — | 50 | 作者安全；保存 tube 安全 | [cold00_huan](cold00_huan/) |
| 0 | Xiangru GPU | 首个新进程 | 7.887486 | 50 | — | 50 | 作者安全；保存 tube 安全 | [cold00_xiangru](cold00_xiangru/) |
| 0 | ours/P3 | 首个新进程 | 8.589842 | 50 | — | 50 | 作者安全；保存 tube 安全 | [cold00_ours_p3](cold00_ours_p3/) |
| 1 | Huan GPU | 后续新进程 | 8.138279 | 50 | — | 50 | 作者安全；保存 tube 安全 | [steady01_huan](steady01_huan/) |
| 1 | Xiangru GPU | 后续新进程 | 8.086614 | 50 | — | 50 | 作者安全；保存 tube 安全 | [steady01_xiangru](steady01_xiangru/) |
| 1 | ours/P3 | 后续新进程 | 8.637965 | 50 | — | 50 | 作者安全；保存 tube 安全 | [steady01_ours_p3](steady01_ours_p3/) |
| 1 | Flow* native RPC | 后续新进程 | 7.735500 | 50 | 50 | 50 | 作者 VERIFIED；保存 tube 安全 | [steady01_native](steady01_native/) |
| 2 | Xiangru GPU | 后续新进程 | 7.937743 | 50 | — | 50 | 作者安全；保存 tube 安全 | [steady02_xiangru](steady02_xiangru/) |
| 2 | ours/P3 | 后续新进程 | 8.740510 | 50 | — | 50 | 作者安全；保存 tube 安全 | [steady02_ours_p3](steady02_ours_p3/) |
| 2 | Flow* native RPC | 后续新进程 | 7.736864 | 50 | 50 | 50 | 作者 VERIFIED；保存 tube 安全 | [steady02_native](steady02_native/) |
| 2 | Huan GPU | 后续新进程 | 8.337823 | 50 | — | 50 | 作者安全；保存 tube 安全 | [steady02_huan](steady02_huan/) |
| 3 | ours/P3 | 后续新进程 | 8.639428 | 50 | — | 50 | 作者安全；保存 tube 安全 | [steady03_ours_p3](steady03_ours_p3/) |
| 3 | Flow* native RPC | 后续新进程 | 7.636556 | 50 | 50 | 50 | 作者 VERIFIED；保存 tube 安全 | [steady03_native](steady03_native/) |
| 3 | Huan GPU | 后续新进程 | 8.187671 | 50 | — | 50 | 作者安全；保存 tube 安全 | [steady03_huan](steady03_huan/) |
| 3 | Xiangru GPU | 后续新进程 | 8.037365 | 50 | — | 50 | 作者安全；保存 tube 安全 | [steady03_xiangru](steady03_xiangru/) |
| 4 | Flow* native RPC | 后续新进程 | 7.736304 | 50 | 50 | 50 | 作者 VERIFIED；保存 tube 安全 | [steady04_native](steady04_native/) |
| 4 | Huan GPU | 后续新进程 | 8.337536 | 50 | — | 50 | 作者安全；保存 tube 安全 | [steady04_huan](steady04_huan/) |
| 4 | Xiangru GPU | 后续新进程 | 7.987532 | 50 | — | 50 | 作者安全；保存 tube 安全 | [steady04_xiangru](steady04_xiangru/) |
| 4 | ours/P3 | 后续新进程 | 8.840520 | 50 | — | 50 | 作者安全；保存 tube 安全 | [steady04_ours_p3](steady04_ours_p3/) |
| 5 | Huan GPU | 后续新进程 | 8.237208 | 50 | — | 50 | 作者安全；保存 tube 安全 | [steady05_huan](steady05_huan/) |
| 5 | Xiangru GPU | 后续新进程 | 7.987427 | 50 | — | 50 | 作者安全；保存 tube 安全 | [steady05_xiangru](steady05_xiangru/) |
| 5 | ours/P3 | 后续新进程 | 8.790101 | 50 | — | 50 | 作者安全；保存 tube 安全 | [steady05_ours_p3](steady05_ours_p3/) |
| 5 | Flow* native RPC | 后续新进程 | 7.786681 | 50 | 50 | 50 | 作者 VERIFIED；保存 tube 安全 | [steady05_native](steady05_native/) |

## 数据与限制

- [`PLAN.json`](PLAN.json) 给出顺序、资源和固定模型；[`RUNS.csv`](RUNS.csv) 与 [`INDEPENDENT_AUDIT.json`](INDEPENDENT_AUDIT.json) 是机器可读 24 次明细和聚合；[`audit_campaign.py`](audit_campaign.py) 从保存的原始文件重算，不启动实验。各子目录保留 START/RESULT、原生或 GPU 日志与范围证据。
- 未发现运行失败、超时、缺失/重复事件、RPC/步数错配或作者 checker 与保存 tube 扫描矛盾。Huan 与 Xiangru 六轮保存的 `ranges.jsonl` 逐字节相同；二者仍使用分别记录的 engine 路径和独立进程。
- native 范围来自原生 `ranges.bin` 的每段 local-domain tube；GPU 三方来自各自的 `ranges.jsonl`。观察器路径与内部算法不同，绝对宽度可在相同六态坐标下并列查看，但不能用这里的 wall 直接声称稳定速度排名或形式化证明强弱。所有方法均缺独立的端到端浮点 NN 证书。
- 独立扫描将保存的 binary64 端点作为精确有理数、系数 `1.4` 按精确 `7/5` 评价，再显示为十进制浮点。它只覆盖保存的 50 段轴对齐 tube；状态间相关性未恢复。
