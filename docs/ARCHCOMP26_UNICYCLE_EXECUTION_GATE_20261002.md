# Unicycle reach：2026 新四方执行门

日期：2026-10-02。此页只读核对来源与历史入口；本轮没有启动 Unicycle 实验，没有计算文件摘要。

## 直接核对的 2026 纸面合同

本地固定报告 `../../reference/ARCH_COMP26_AINNCS.pdf` 的 PDF 第 7 页、印刷第 91 页，式 (3) 写为

```
x1' = x4 cos(x3)     x2' = x4 sin(x3)
x3' = u2             x4' = u1 + w
u_i = f_i(x) - 20    w ∈ 10^-4[-1,1]
```

报告同页给出初集 `[9.5,9.55] × [-4.5,-4.45] × [2.1,2.11] × [1.5,1.51]`，每 `0.2 s` 更新控制器，目标是 `10 s` 内进入 `[-0.6,0.6] × [-0.2,0.2] × [-0.06,0.06] × [-0.3,0.3]`。报告没有说 `w` 是每条轨迹的固定参数，还是可随时间变化的有界扰动。这里的“10 s 内”也未指定旧程序采用的终点包含检查是否为唯一判据；终点包含可作充分证据，终点未包含不能证明整个时间窗失败。

保存的固定 2026 ONNX 为本机 `../../../../results/archcomp26_20261001/official_unicycle_controller_2026.onnx`，远端副本是 `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/unicycle_prep_001/controllerB_2026.onnx`。用远端 `onnx.load` 只读检查，真正状态输入 `input` 为 `[1,1,1,4]`，输出 `relu_2_Flatten` 为 `[1,2]`，图节点依次为 `Sub, Conv, Relu, Conv, Relu, Flatten`；其余五个 graph inputs 是图内均值/权重/偏置参数，亦列为 initializer。远端固定 2026 文件与保存的 2024 `Benchmark10-Unicycle/controllerB.onnx` 用 `cmp -s` 直接逐字节比较相等。该事实仅冻结控制器文件，不解决动力学冲突。

## 四方历史入口和官方执行源码的差异

| 来源 | yaw 导数 | speed 导数 | 扰动语义 |
| --- | --- | --- | --- |
| 2026 报告式 (3)，本轮直接读 PDF | `u2` | `u1+w` | 仅给范围，时间语义未定义 |
| 固定官方 `dynamics10.m`，见 [2026-10-01 来源审计](ARCHCOMP26_UNICYCLE_SOURCE_CONFLICT_20261001.md) | `f2(x)-20` | `f1(x)-20` | 可执行 RHS 不含 `w`；此项本轮未重取远端源文件 |
| 旧 Huan/Xiangru/P3 共享 [GPU YAML](../research/gpu_verified_20260930/report/configs/arch_unicycle.yaml) | `u2+w-20` | `u1+w-20` | 扩充 `w'=0`，初值 `[-10^-4,10^-4]` |
| 旧 [Flow* native C++](../../../../results/archcomp_review_20260923/evidence_v2/suite_build/archcomp/unicycle/matched_threads4.cpp) | `u2+w-20` | `u1+w-20` | 同一个常值 `w`，原生状态顺序有置换 |

旧 GPU YAML 明示单初盒、`0.2 × 50` 期、Flow* `h=.02/order2`、目标终点八条半空间，模型形状 `[-1,1,1,4]→[-1,2]`。旧 native C++ 也只在最后调用 `fp_end_of_time.isInTarget`。历史四方记录可用于分析旧的 `w-in-both` 参与者合同，不可填入本轮 `paper-w-in-speed` 或 `official-matlab-no-w` 的新结果。旧原生初盒 x2 另有 1 ULP 欠包审计，见 [旧一步审计](../research/gpu_verified_20260930/report/evidence/unicycle_native_one_step.md)。旧 P3 后续首步诊断曾停在扩展加载，亦非本轮 2026 新作业。

## 具体阻断与资源决定

要声称执行 2026 论文合同，须取得权威说明或明确选择：`w` 是轨迹常值，还是允许随时间变化；并确定四方使用报告式 (3) 的 `w` 仅在 `x4'`，还是执行官方 MATLAB 的无扰动 RHS。若选择轨迹常值，可单独命名 `paper-speed-w-constant` 比较 profile；若选择时变有界扰动，旧 `w'=0` 实现不够，须另定并审计四方输入/集合推进语义。若选择 MATLAB 无扰动 RHS，则须单独命名该 profile，不能标成纸面式 (3)。新 reach checker 还须明确它只给 `T=10` 终点充分条件，或实现时间窗存在性检查。

目前没有无歧义的纸面四方合同。**不安排 GPU、CPU、RPC 端口，也不启动首周期 smoke**。固定 ONNX 已具备；缺的是上述 plant 与扰动/性质语义，不能从旧成绩推断。
