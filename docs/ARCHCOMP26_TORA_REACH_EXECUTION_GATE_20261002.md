# ARCH-COMP26 TORA 两个 reach 实例：四方执行门

日期：2026-10-02。范围仅为 `tora-reach-sigmoid`、`tora-reach-tanh` 的来源与旧入口核对；没有启动新可达性实验。基线来源为 [2026 AINNCS 报告 §3.2，印刷页 90–91](https://easychair.org/publications/paper/GsKW/download)、[已冻结的官方来源审计](ARCHCOMP26_TORA_REACH_SOURCE_CONFLICT_20261001.md)、[控制器数值预检](ARCHCOMP26_TORA_REACH_CONTROLLER_PREFLIGHT_NOHASH_20261001.md)。

## 已经固定、可以共用的部分

两实例均为四态 `(x1,x2,x3,x4)`，ODE 为 `x1'=x2, x2'=-x1+0.1 sin(x3), x3'=x4, x4'=u`；完整初盒是 `[-0.77,-0.75] × [-0.45,-0.43] × [0.51,0.54] × [-0.3,-0.28]`；控制器每 0.5 s 刷新一次，10 期到 `T=5 s`；目标为 `x1∈[-0.1,0.2]` 且 `x2∈[-0.9,-0.6]`。论文、官方规格与[旧四方保存配置](../research/gpu_verified_20260930/report/configs/arch_tora_sigmoid.yaml)在这些字段一致；[另一个配置](../research/gpu_verified_20260930/report/configs/arch_tora_relu_tanh.yaml)也一致。官方 `dynamicsTora.m` 的 **注释**仍写了 TORA remain 初盒，实际函数体只给上述 ODE；初盒应取规格和论文，不取注释。

固定官方两份 `.mat` 各含 `4→20→20→20→1` 的四层网络。2024 旧 ONNX 与相应 `.mat` 的各层权重、偏置、激活序列逐元素相同，22 个点的前向输出误差均低于 `1.12e-16`，见[预检收据](ARCHCOMP26_TORA_REACH_CONTROLLER_PREFLIGHT_NOHASH_20261001.md)。本次再直接解读两份官方 `.txt`：各 969 行，其中 6 行头部、961 个按“每神经元权重后接偏置”排列的参数、末尾 `0` 和 `11`；961 个参数分别与同名 `.mat` **逐元素完全相等**。这确认了已保存参数的数值关系；`.txt` 本身没有逐层激活标签，不能用它推断输出激活。比对仅为普通数值读取与逐元素比较。

## 尚未统一的闭环合同

| 来源 | sigmoid 实例 | tanh 实例 | 注入 plant 的 `u` |
| --- | --- | --- | --- |
| 2026 论文的合并文字 | 三层 sigmoid 隐层、tanh 输出 | 同一描述覆盖第二个 reach 网络 | 两者 `11 f(x)` |
| 固定官方 `.mat` 激活与 `.txt` 尾部 | 四层全 sigmoid | 三层 ReLU 隐层、tanh 输出 | 两者标记为 `11 f(x)` |
| 冻结旧 native、Huan、Xiangru、P3 入口 | 2024 `nn_tora_sigmoid.onnx`，原始网络同官方 `.mat` | 2024 `nn_tora_relu_tanh.onnx`，原始网络同官方 `.mat` | sigmoid 为 `22(f(x)-0.5)`；tanh 为 `11 f(x)` |

旧 native 的 `crown_sigmoid.py`/`crown_relu_tanh.py` 与旧 Huan/Xiangru/P3 的 `output_scale`、`output_offset` 均按 `u=(f-offset)·scale` 实施；[共享驱动实现](../research/gpu_verified_20260930/source/integration/crown_reach.py)明确写出该公式。旧记录每项均曾覆盖单盒 500 个 ODE 小步，但这些是**旧闭环**结果，不能填入新版两实例的八个四方单元。尤其 sigmoid 初盒中心，官方 `.mat` 原始 `f≈0.461650624`，`11f≈5.078156863` 与旧 `22(f-0.5)≈-0.843686273` 已是不同控制输入。激活按论文字面改造时也将改变网络，不可称为原官方训练模型。

论文的性质是“5 s 内到达”；旧 native 两个 C++ 入口仅对最终 `fp_end_of_time` 调用 `isInTarget`，[旧 GPU 驱动](../research/gpu_verified_20260930/source/integration/crown_reach.py)也仅在最终盒检查 `constraints_target`。因此旧 `VERIFIED` 只能支持终点包含目标（它足以证明“窗内到达”），而旧 `UNKNOWN`/`FALSIFIED` 不能自动解释为整个时间窗的判定。全时域图须保存 tube；目标判断不能从终点以外的未保存部分补写。

## 启动门与尚需的决定

1. **用户须为两个变体分别选定主合同。** sigmoid 有“论文三 sigmoid + tanh、`11f`”“官方四 sigmoid、`11f`”“旧四 sigmoid、`22(f-0.5)`”三种不同系统；tanh 有“论文三 sigmoid + tanh、`11f`”与“官方/旧三 ReLU + tanh、`11f`”两种。若选择论文文字，必须显式标记为从官方权重改激活构造的新控制器。现有对 QUAD 论文方程的决定不涵盖 TORA 网络。其他合同可作为单列对照。
2. **须确定性质收据口径。** 若以 `R(5)⊆G` 为充分证明标准，可沿用终点 checker，但不能把未证实终点包含说成整个窗内不可能到达；若要完整判断“窗内到达”，须先固定时间量词及跨时段 checker。无需为已确定的 ODE、初盒或 0.5 s 采样再做来源选择。
3. 选定后用[显式参数构造器](../tools/archcomp26_tora_reach_controller_nohash.py)由官方 `.mat` 构造**同一份**输出已是 plant `u` 的 float64 ONNX 给四方，关闭旧入口的额外缩放；先做 CPU 前向与接口预检，再用全初盒、10 期、500 小步分别取得四方的新 `START/RESULT`、接受计数、tube/endpoint、checker 和进程时间。失败前缀不得作为完整计时或性质结果。保持现有旧任务和收据不动。
