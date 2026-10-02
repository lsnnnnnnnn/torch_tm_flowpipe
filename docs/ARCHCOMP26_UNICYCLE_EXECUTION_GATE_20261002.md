# Unicycle reach：2026 新四方执行门

日期：2026-10-02。此页先核对来源与历史入口，随后记录用户选定合同下的新实验；没有计算文件摘要。

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

## 用户指定后冻结的新主表合同

用户随后明确指定：新主表使用**2026 论文式 (3)**；同一条轨迹的 `w∈[-0.0001,0.0001]` 在全时域恒定，并且只进入速度导数；`T=10` 的完整终点可达集若包含于目标盒，只作为“10 秒内到达”的充分证据。新 profile 命名为 `unicycle-paper-speed-w-constant-v1`。这是用户选择的可执行语义，并非声称论文自己说明了 `w` 的时间量词；固定官方 MATLAB 无扰动版本与历史两导数加扰动版本仍须分列。

| 项 | 冻结值 |
| --- | --- |
| 物理态与初盒 | `(x1,x2,x3,x4)=(x,y,yaw,speed)`；`[9.5,9.55]×[-4.5,-4.45]×[2.1,2.11]×[1.5,1.51]`，一整盒。 |
| 参数与辅助态 | GPU 配置顺序 `(x1,x2,x3,x4,w,t,u1,u2)`，其中 `w(0)∈[-1e-4,1e-4]`、`w'=0`；`t(0)=0,t'=1`；两个 raw 网络输出 `u_i=f_i(x)` 在每个 `0.2 s` 周期内保持，`u_i'=0`。原生若采用不同内部顺序，须在入口和保存台账中逐名映射并核验初盒覆盖。 |
| 动力学 | `x1'=x4 cos(x3)`、`x2'=x4 sin(x3)`、`x3'=u2-20`、`x4'=u1-20+w`；`w` **不**进入朝向导数。控制器输入按原始 `(x1,x2,x3,x4)` 顺序，不含 `w/t/u`；固定 2026 ONNX 输出两路 raw `f_i`，每路只减 20 一次。 |
| 时域/数值设置 | 50 个控制周期×`0.2 s`=`10 s`；四方法隔离入口先用历史可执行参数 `h=0.02 s`、Flow* 阶数 2、cutoff `1e-6`、余项初估 `[-0.01,0.01]`、SR 容量 1000。它们是明确记录的方法参数，不是论文模型方程。计划 500 个内积分步。 |
| 性质与结论 | 目标盒 `[-0.6,0.6]×[-0.2,0.2]×[-0.06,0.06]×[-0.3,0.3]`。只有完整数值 `T=10` 且保存终点全盒包含于目标时，才能写“终点包含给出论文时间窗到达的充分证据”；若终点不包含、数值早停或未完成全盒覆盖，性质为 `UNKNOWN`，不据此断言未在 10 秒内到达。独立浮点 NNCS 证明另行限定。 |

旧四方的 `x3'=u2+w-20` 结果不得移入此 profile。新实验只用新 run ID，先 CPU 模型与完整初盒预检、再首周期有界诊断；通过后才启动各方法全 500 小步配置，首个数值拒绝即停，保存原始区间和日志。原生历史 x2 初盒曾有一 ULP 欠包，新的原生入口必须逐态确认**实际表示**覆盖请求初盒；仅打印相同十进制盒不足以过门。不得把单周期结果、终点目标 checker 的不确定结果或早停壁钟时间写成全时域四方成绩。

## 已执行的 Huan/Xiangru 新合同门检与全程

原始 [证据目录](evidence/results/archcomp26_20261001/unicycle_paper_speed_w_constant_v1/SUMMARY.md) 保留 CPU 预检首个配置拒绝 `_001`、修正后通过的 `_002`、Huan/Xiangru 各自的首周期和全程 run ID。首周期各 10/10 小步接受，仅覆盖 `t∈[0,0.2]`。随后 Huan `huan_full50_001` 与 Xiangru `xiangru_full50_001` 各完成 50 个控制周期、500/500 小步到 `T=10`，外层壁钟分别为 9.594389 s 与 9.895126 s。这是各一次作业的观测时间，不构成四方速度排名。

独立扫描逐条检查两份 500 步的 8 态保存 tube 与 endpoint：区间有限、有序，首段物理初盒与常值参数初区间被 tube 包含，每个 endpoint 均落在同段 tube 内。两份终点物理盒相同：`x1=[0.4365198254,0.5277617930]`、`x2=[-0.1859375613,-0.0507807157]`、`x3=[-0.1035952958,0.0189019927]`、`x4=[-0.3227584415,-0.2088621331]`。后两维没有整体落入目标 `[-0.06,0.06]×[-0.3,0.3]`；程序输出 `UNKNOWN`。这只说明终点包含这一充分判据未成立，不能据此判定 10 秒内从未到达。两方法共用同一控制器驱动，独立浮点 NNCS 证明尚未建立。

## P3 新合同完整数值运行

[P3 一期](evidence/results/archcomp26_20261001/unicycle_paper_speed_w_constant_v1/p3_smoke1_001/SUMMARY.md)为 10/10 步、完整初盒门检；[独立全程](evidence/results/archcomp26_20261001/unicycle_paper_speed_w_constant_v1/p3_full50_001/SUMMARY.md)在新 run ID 接受 500/500 步到 `T=10`，外层 wall 17.819782 s。独立扫描 500 条八态范围，首段覆盖初盒，`w` 每步包住初始参数，所有区间有限有序且 endpoint 位于对应 tube 内。保存的最终物理 endpoint 四维均在目标盒内，构成已选“10 秒内到达”语义的**数值充分条件观察**，但没有独立端到端浮点 NNCS 证书。

P3 工作阶数 3 / 验证阶数 4、严格端点与两控制量注入；Huan/Xiangru 使用阶数 2。数值阶数不同，单次 wall 和宽度不作同参数方法排名。P3 全程原始 `P3_METHOD.json` 误沿用一期文字，错误写“首周期”；原件保留，[更正审计](evidence/results/archcomp26_20261001/unicycle_paper_speed_w_constant_v1/p3_full50_001/SCOPE_ERRATUM.json)列明其与 `START/RESULT` 及 500 条保存区间的矛盾。判定完整时域依据后者。

## Flow* native 新合同完整数值运行

原生[初盒审计与首周期](evidence/results/archcomp26_20261001/native_unicycle_paper_speed_smoke1_001/SUMMARY.md)先确认内部状态次序 `x1,x2,x3,x4,t,u1,u2,w` 与论文物理态映射、8/8 个实际仿射初盒覆盖；旧原生 x2 的 1 ULP 欠包通过双侧半径修复，仅 x2 初始半径改变。独立[全程作业](evidence/results/archcomp26_20261001/native_unicycle_paper_speed_full50_001/SUMMARY.md)随后完成 50/50 期、500/500 小段，50 次 RPC；外层 wall 10.397308 s。原生日志打印 `VERIFIED`。独立重读 500 条物理 tube/endpoint 与 RPC，确认网格完整、区间有限有序、逐段终点在 tube 内，保存 `T=10` 物理终点四维全盒落在目标内。它支持用户选定的终点充分判据，仍不构成独立端到端浮点 NNCS 证书。

本 profile 四方法均有完整 10 秒数值时域：Huan/Xiangru 的终点包含判据未过、性质为 `UNKNOWN`；P3/原生的保存终点在目标内。四份单次 wall 在资源、实现与 P3 数值阶数不一致的条件下只记录，不排序。
