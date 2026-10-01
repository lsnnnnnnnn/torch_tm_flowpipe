# 2026 TORA reach：控制器合同冲突与执行门槛

日期：2026-10-01。对象是 `tora-reach-sigmoid` 和 `tora-reach-tanh`，不包含已运行的 `tora-remain`。本页为固定来源审计；尚未启动这两个 reach 实例的新数值任务，也未进行内容摘要校验。

## 已一致的物理任务

[2026 AINNCS 报告，印刷页 90–91](https://easychair.org/publications/paper/GsKW/download) 和[固定官方实例规格](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Tora_Heterogeneous/Specifications.txt)给出四态顺序 `(x1,x2,x3,x4)`，初始盒 `[-0.77,-0.75] × [-0.45,-0.43] × [0.51,0.54] × [-0.3,-0.28]`，10 次控制刷新、周期 0.5 s、终点 5 s，以及目标 `x1∈[-0.1,0.2]`、`x2∈[-0.9,-0.6]`。[固定官方动力学](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Tora_Heterogeneous/dynamicsTora.m)与报告式 (2) 对应 `x1'=x2, x2'=-x1+0.1 sin(x3), x3'=x4, x4'=u`。

报告把性质写成“在 5 s 时间窗内到达目标”，冻结旧原生 CROWN/Flow* `tora_sigmoid.cpp` 和 `tora_relu_tanh.cpp` 则在最后一次推进后只调用 `fp_end_of_time.isInTarget`。因此旧 checker 的 `VERIFIED` 仅证明其终点目标语义；若采用真正的“某时刻到达”，须单独定义并实现跨时间的 reach checker。终点包含于时间窗，所以一份完整的终点包含证明可以作为到达的充分条件，反向则不成立。

## 控制器来源并不一致

| 来源 | reach-sigmoid | reach-tanh / ReLU-tanh | plant 输入缩放 |
| --- | --- | --- | --- |
| 2026 报告印刷页 90 的合并文字 | 对“另两个控制器”写三层 sigmoid 隐层、tanh 输出 | 同一句文字同样覆盖本项 | 两者 `u=11·f(x)` |
| 固定官方 `.mat` 中 `act_fcns` | `sigmoid,sigmoid,sigmoid,sigmoid` | `relu,relu,relu,tanh` | `.mat` 只给网络层，不给 plant 缩放 |
| 固定官方 `.txt` 与 `Specifications.txt` | 文字称 sigmoid 激活；文本尾部 offset `0`、scale `11` | 文字称 ReLU 隐层、tanh 输出；文本尾部 offset `0`、scale `11` | 规格对应 `u=11·f(x)` |
| 冻结旧 CROWN 服务器 | 加载 2024 `nn_tora_sigmoid.onnx` | 加载 2024 `nn_tora_relu_tanh.onnx` | sigmoid `u=22·(f(x)-0.5)`；ReLU/tanh `u=11·f(x)` |

本地已直接解析固定官方 `.mat` 中四个 `act_fcns`，并读取 `.txt` 的末尾 offset/scale。原始文件在 `results/archcomp26_20261001/tora_reach_official_2026/`；官方 2026 目录仅提供 `.mat` 与 `.txt`，没有 ONNX。随后完成的[无哈希控制器预检](ARCHCOMP26_TORA_REACH_CONTROLLER_PREFLIGHT_NOHASH_20261001.md)逐层比较固定官方 `.mat` 与保存的 2024 ONNX，两个网络的全部权重、偏置和激活序列均相同；22 点独立前向计算与 ONNX Runtime 的最大绝对差不超过 `1.12e-16`。这解决了原始网络参数的数值身份，**没有**解决 plant 缩放、论文激活文字或区间网络证明。预检工具也能按显式选择构造新的 ONNX，接入时必须避免再由旧执行器二次缩放。

为量化差别，仅在完整初始盒的中心 `(-0.76,-0.44,0.525,-0.29)` 用固定官方 `.mat` 权重作一次普通双精度前向计算：sigmoid 输出网络的原始值约 `0.461650624`；官方四层 sigmoid 加 `11·f` 给 `u≈5.078156863`，同一网络沿旧 `22·(f-0.5)` 给 `u≈-0.843686273`，把最后一层改成论文写的 tanh 并取 `11·f` 给 `u≈-1.677504295`。三种控制输入在同一点已明显不同。这只是合同差异演示，不是可达域实验、验证结果或替代 ONNX 等价性检查。

## 新四方主表的决策状态

`tora-reach-sigmoid` 的主要冲突是输出激活与缩放：三种来源会定义不同闭环系统。用户已被请求选择主合同；选定前不启动该实例。`tora-reach-tanh` 还有报告“sigmoid 隐层”与固定官方 ReLU 隐层冲突，须在主合同选择后明确对应网络；不得把 `.mat` 的 ReLU 层暗称论文 sigmoid 层。

在两项均未有可执行的共同模型、确切缩放、目标 checker 和全初盒/全时域端到端收据前，16×4 新版无哈希工作矩阵保留“未尝试”。冻结旧运行可作为历史路径参照，不能移作本次新版结果。
