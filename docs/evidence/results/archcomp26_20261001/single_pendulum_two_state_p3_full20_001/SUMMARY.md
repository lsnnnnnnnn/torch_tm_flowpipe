# Single Pendulum 两物理态：ours/working-P3 新全程尝试

原始远端目录为 `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/single_pendulum_two_state_p3_full20_001`；本地同名目录是小型原始证据镜像。独立的一盒一期 plumbing smoke 在 [`../single_pendulum_two_state_p3_smoke1_001`](../single_pendulum_two_state_p3_smoke1_001/)：exit 0、5/5 小步接受、外层 wall 4.677 s；只运行到 `t=0.05`，没有检查 `[0.5,1]` 性质。两个目录均为新建，旧实验未重启；未执行内容摘要校验或 CUDA 扩展重编译。

执行合同见[固定两物理态预检](../single_pendulum_prep_001/SUMMARY.md)：完整一盒 `x1∈[1,1.175], x2∈[0,0.2]`，`x1'=x2, x2'=2 sin(x1)+8u1`，20 个 `0.05 s` 控制期、每期 5 个 `0.01 s` ODE 小步、Taylor order 2；仅在闭时间窗 `[0.5,1]` 全时要求 `x1∈[0,1]`。辅助 `t(0)=0,t'=1` 只给 checker 定时，`u1'=0` 使控制周期内保持；NN 只读 `x1,x2`，没有把辅助时钟当成论文第三物理态。固定 2026 ONNX 是 `../single_pendulum_prep_001/controller_single_pendulum.onnx`。新 [P3 launcher](../../../../../tools/archcomp26_sp_two_state_p3_nohash.py) 复用已运行 ACC P3 的预编译 CUDA 库/strict endpoint，实用 `engine_quad_normalization_center` 与共享 Xiangru driver；它在调用链入口阻断 SHA-256 构造和 CUDA JIT。源配置为[固定两物理态 YAML](../../../../../benchmarks/archcomp26/configs/single_pendulum_paper_two_state.yaml)。运行时使用物理 GPU2、CPU10–13。

## 实际结果与口径

- 外层 [`RESULT.json`](RESULT.json)：supervisor exit 0、无超时、完整新进程 wall **6.583 s**。内层 [`data/RESULT.json`](data/RESULT.json)：driver exit 0、100/100 小步接受、`metrics_broken=0`，driver 自报积分计时 2.389 s，内层 wall 5.704 s；峰值 CUDA allocated 71,251,968 B、reserved 111,149,056 B。冷启动单次进程时间不是五次稳态比较。
- [`stdout.log`](stdout.log) 打印 Step 0–19，且明确 `SPEC WINDOW safeset t in [0.5, 1]: checked on 50 of 100 substeps, 0 partial`；没有 `Unsafe.` 或 `Unknown.`。保存的 [`data/safety.jsonl`](data/safety.jsonl) 有 50 个窗口安全事件，各自作者违例表达式的上界均 `≤0`，其中最大为 `-0.006759317707212498`。
- 对保存的 [`data/ranges.jsonl`](data/ranges.jsonl) 做[独立区间扫描](INDEPENDENT_SCAN.json)：100 个连续小步、100 个 `accepted=true`、有限有序的两物理态 tube/endpoint。第 51–100 小步覆盖 `[0.5,1]`；这 50 个保存 tube 的 `x1` union 是 **`[0.5645452654370386,0.9932406822927875]`**，全部在闭 Safe 区间内。`t=0.5` 端点 `x1=[0.8159451523188869,0.9936458548877724]`；`t=1` 直接局部端点 `x1=[0.5645582112532347,0.7062300774306092]`、`x2=[-0.5618548977541915,-0.41488126056280245]`。原始 terminal `HULL` 行由 driver 的另一 endpoint 例程计算，未取代上述逐小步观察值。

此结果是明确命名的两物理态 profile 下的作者 checker 成功及保存区间再扫描。扫描不能独立证明 NN 浮点边界、P3 数值声称或端到端 NNCS 证书。Huan/Xiangru 的同合同全程结果仍在各自独立目录；P3 的这个单次冷启动时间不可直接充作稳定速度排名。
