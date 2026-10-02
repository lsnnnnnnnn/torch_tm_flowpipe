# ARCH-COMP26 TORA 两个 reach 实例：四方执行门

日期：2026-10-02。范围为 `tora-reach-sigmoid`、`tora-reach-tanh` 的来源、已选择的主合同及隔离运行。基线来源为 [2026 AINNCS 报告 §3.2，印刷页 90–91](https://easychair.org/publications/paper/GsKW/download)、[已冻结的官方来源审计](ARCHCOMP26_TORA_REACH_SOURCE_CONFLICT_20261001.md)、[控制器数值预检](ARCHCOMP26_TORA_REACH_CONTROLLER_PREFLIGHT_NOHASH_20261001.md)。

## 已经固定、可以共用的部分

两实例均为四态 `(x1,x2,x3,x4)`，ODE 为 `x1'=x2, x2'=-x1+0.1 sin(x3), x3'=x4, x4'=u`；完整初盒是 `[-0.77,-0.75] × [-0.45,-0.43] × [0.51,0.54] × [-0.3,-0.28]`；控制器每 0.5 s 刷新一次，10 期到 `T=5 s`；目标为 `x1∈[-0.1,0.2]` 且 `x2∈[-0.9,-0.6]`。论文、官方规格与[旧四方保存配置](../research/gpu_verified_20260930/report/configs/arch_tora_sigmoid.yaml)在这些字段一致；[另一个配置](../research/gpu_verified_20260930/report/configs/arch_tora_relu_tanh.yaml)也一致。官方 `dynamicsTora.m` 的 **注释**仍写了 TORA remain 初盒，实际函数体只给上述 ODE；初盒应取规格和论文，不取注释。

固定官方两份 `.mat` 各含 `4→20→20→20→1` 的四层网络。2024 旧 ONNX 与相应 `.mat` 的各层权重、偏置、激活序列逐元素相同，22 个点的前向输出误差均低于 `1.12e-16`，见[预检收据](ARCHCOMP26_TORA_REACH_CONTROLLER_PREFLIGHT_NOHASH_20261001.md)。本次再直接解读两份官方 `.txt`：各 969 行，其中 6 行头部、961 个按“每神经元权重后接偏置”排列的参数、末尾 `0` 和 `11`；961 个参数分别与同名 `.mat` **逐元素完全相等**。这确认了已保存参数的数值关系；`.txt` 本身没有逐层激活标签，不能用它推断输出激活。比对仅为普通数值读取与逐元素比较。

## 来源差异与已选择的主合同

| 来源 | sigmoid 实例 | tanh 实例 | 注入 plant 的 `u` |
| --- | --- | --- | --- |
| 2026 论文的合并文字 | 三层 sigmoid 隐层、tanh 输出 | 同一描述覆盖第二个 reach 网络 | 两者 `11 f(x)` |
| 固定官方 `.mat` 激活与 `.txt` 尾部 | 四层全 sigmoid | 三层 ReLU 隐层、tanh 输出 | 两者标记为 `11 f(x)` |
| 冻结旧 native、Huan、Xiangru、P3 入口 | 2024 `nn_tora_sigmoid.onnx`，原始网络同官方 `.mat` | 2024 `nn_tora_relu_tanh.onnx`，原始网络同官方 `.mat` | sigmoid 为 `22(f(x)-0.5)`；tanh 为 `11 f(x)` |

**用户已确认新版主表采用两份 2026 官方模型及 `u=11f`。** 因此 sigmoid 为官方 `.mat` 四层 sigmoid，tanh 为官方 `.mat` 三层 ReLU 隐层加 tanh 输出；论文合并激活文字和旧 sigmoid `22(f-0.5)` 均单列为不同合同。旧 native 的 `crown_sigmoid.py`/`crown_relu_tanh.py` 与旧 Huan/Xiangru/P3 的 `output_scale`、`output_offset` 均按 `u=(f-offset)·scale` 实施；[共享驱动实现](../research/gpu_verified_20260930/source/integration/crown_reach.py)明确写出该公式。旧 sigmoid 500 步属于不同闭环，不能移入新主表：初盒中心官方 `.mat` 原始 `f≈0.461650624`，新 `11f≈5.078156863` 与旧 `22(f-0.5)≈-0.843686273` 已给不同输入。旧 tanh 网络参数、激活、`11f`、ODE、初盒和时域则与已选主合同相同，可作为**标明历史来源的同合同证据**复用，不记为本轮新 attempt。

论文的性质是“5 s 内到达”；旧 native 两个 C++ 入口仅对最终 `fp_end_of_time` 调用 `isInTarget`，[旧 GPU 驱动](../research/gpu_verified_20260930/source/integration/crown_reach.py)也仅在最终盒检查 `constraints_target`。因此旧 `VERIFIED` 只能支持终点包含目标（它足以证明“窗内到达”），而旧 `UNKNOWN`/`FALSIFIED` 不能自动解释为整个时间窗的判定。全时域图须保存 tube；目标判断不能从终点以外的未保存部分补写。

## 启动门与性质收据口径

1. 两个变体的控制器主合同已按用户决定冻结；[显式参数构造器](../tools/archcomp26_tora_reach_controller_nohash.py)由官方 `.mat` 生成输出已是 plant `u=11f` 的 float64 ONNX，四方均须关闭旧入口的额外缩放。CPU 前向和接口预检之后，以完整初盒、10 期、500 小步保存各方法的独立 `START/RESULT`、接受计数与 tube/endpoint。失败前缀不得作为完整计时或性质结果；保持旧任务不动。
2. `R(5)⊆G` 是论文“5 秒内到达”的**充分条件**；完整数值流管加终点包含可以形成候选充分证据，但仍须区分作者性质 checker 和端到端浮点证书。若终点不包含目标，不能推出“5 秒内未到达”。若要对整个时间窗给出否定判定，仍缺明确时间量词及跨时段 checker；该缺件不阻挡当前已冻结主合同的完整数值实验。
3. 新工作矩阵须同时写明每格是本轮新尝试还是同合同历史复用，不把旧性能时间与新 run 直接混作四方排名。

## 后续独立短程诊断

在合同选择前，对明确命名的官方文件 `tora_reach_sigmoid_official2026_mat_u11_firstperiod_diag` 做了[完整初盒一期 Huan 诊断](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_firstperiod_diag/SUMMARY.md)：由固定 `.mat` 构造四层 sigmoid 加图内 `u=11f` 的 ONNX，外部缩放设为 1/0，50/50 个 0.01 秒数值小步接受。其范围、日志、配置与独立扫描单列，没有检查 `T=5` 目标。

## 2026-10-02 后续数值与复用审计

同一具名官方 sigmoid `u=11f` profile 的新[Huan 全初盒数值运行](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_full500_huan_002/SUMMARY.md)完成 500/500 小步，保存终点 `x1,x2` 落入目标；性质 checker 未启用，故这是已选主合同的数值区间充分条件观察，尚非作者 checker 判定或端到端证明。独立的[第一次全程入口](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_full500_huan_001/SUMMARY.md)因误选保存 Python 环境在第一个 ODE 步前失败，原始收据不删除。

同合同的 [P3 完整初盒首周期门检](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_p3_firstperiod_001/SUMMARY.md)接受 50/50 小步；随后另立[全程 run ID](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_p3_full500_001/SUMMARY.md)完成 10 期、500/500 小步。两个运行的四态 tube/endpoint 均经独立保存区间扫描确认有限、有序且逐步包含。全程保存 `T=5` 的 `x1=[0.13452581591208781,0.16060475161859264]`、`x2=[-0.8764374242552789,-0.8505126711494655]` 落入目标；working P3 / validation P4 使用严格 endpoint/injection。外层 wall 13.741791 s、内层 11.278381 s，性质 checker 未运行，也不据单次时间判稳定排名。[Xiangru 一期](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_xiangru_firstperiod_001/SUMMARY.md)及其[全程](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_xiangru_full500_001/SUMMARY.md)分别保存，完整 500 步末端也包含目标。现有数值区间观察须与独立端到端证明分开；Huan/Xiangru 使用共享作者驱动，不据区间相同宣称独立证明。

原生 Flow* 的[隔离构建预检](evidence/results/archcomp26_20261001/native_tora_reach_sigmoid_u11_build_001/PREFLIGHT.json)确认完整编译初盒、四态 ODE、P6/0.01 s，以及官方四层 sigmoid 图内 `u=11f`、RPC 外部 `1/0`；另用端口 5111。独立[一期门检](evidence/results/archcomp26_20261001/native_tora_reach_sigmoid_u11_smoke1_001/SUMMARY.md)为 1 RPC、50/50 保存范围，短时域作者 checker `UNKNOWN`；后续新[全程 10 期](evidence/results/archcomp26_20261001/native_tora_reach_sigmoid_u11_full10_001/SUMMARY.md)为 10 RPC、500/500 保存范围、外层 wall 8.991666 s、作者**终点 checker `VERIFIED`**。独立扫描的 `T=5` 保存 endpoint 为 `x1=[0.1345319317307225,0.16059887233324702]`、`x2=[-0.8763648122763305,-0.8505857060659042]`，在目标内。原生作者 `VERIFIED`、四方完整数值时域、终点包含和独立端到端浮点证明是不同口径；旧 `22(f−0.5)` 原生任务未重启，不能混入新版主表或由一次 wall 排名。

官方 tanh 模型 `ReLU³/tanh, u=11f` 的新[一期门检](evidence/results/archcomp26_20261001/tora_reach_tanh_official2026_mat_u11_firstperiod_diag_001/SUMMARY.md)接受 50/50 步；其 800 个四态 tube/endpoint 边界数与旧 Huan 全程前 50 步逐值相同，且旧四方法同合同均有 500 步完整历史记录。因此没有重启同一 tanh 全程作业；旧四方全程应作为**历史同合同证据**审计引用，仍需逐项区分作者 `VERIFIED`、完整数值时域和端到端证书。

后来另立的[当前 working P3 首周期门检](evidence/results/archcomp26_20261001/tora_reach_tanh_official2026_mat_u11_workingp3_firstperiod_20261002_001/SUMMARY.md)和[当前 working P3 500 步全程](evidence/results/archcomp26_20261001/tora_reach_tanh_official2026_mat_u11_workingp3_full500_20261002_001/SUMMARY.md)属于**新代引擎** `engine_quad_normalization_center`，不是重启旧 `engine_linear_leaf_v2` 作业。前者 50/50、后者 500/500 接受；新 P3 保存 `T=5` 完整初盒终点的 `x1/x2` 数值盒在目标内，性质 checker 未运行。固定主表用新 P3，加 Huan/Xiangru/原生三方同合同历史全程；旧 P3 原结果仅保留为代际对照。新旧数值阶数和资源不同，不据单次时间排名。
