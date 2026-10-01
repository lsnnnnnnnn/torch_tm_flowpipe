# 2026 Unicycle：扰动位置与时间语义的合同审计

日期：2026-10-01。本页是 `unicycle-reach` 新四方实验前的来源核对；没有启动新数值任务或重新运行旧实验，也没有计算文件摘要。

## 来源给出的共同部分

[2026 AINNCS 报告，印刷页 91](https://easychair.org/publications/paper/GsKW/download)、[固定官方规格](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Benchmark10-Unicycle/Specifications.txt)和[固定官方 MATLAB 动力学](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Benchmark10-Unicycle/dynamics10.m)一致采用物理顺序 `(x1,x2,x3,x4)=(x,y,yaw,speed)`，初集 `[9.5,9.55]×[-4.5,-4.45]×[2.1,2.11]×[1.5,1.51]`，控制周期 0.2 s，50 次刷新到 10 s。目标是四态分别进入 `[-0.6,0.6]×[-0.2,0.2]×[-0.06,0.06]×[-0.3,0.3]`。三方的前两条 plant 方程都是 `x1'=x4 cos(x3), x2'=x4 sin(x3)`；网络输出在进入 plant 前按 `u_i=f_i(x)-20` 转换。

固定官方 `controllerB.onnx` 大小为 15,026 字节。本轮取固定 2026 文件上传服务器后，与保存的 2024 同名 ONNX 做直接逐字节比较，`cmp` 返回相等；没有进行内容摘要校验。ONNX 图的实际非参数输入是形状 `[1,1,1,4]` 的 `input`，输出为 `[1,2]`；图中依次有 Sub、两层 Conv、两次 ReLU、Flatten。该字节相等结论仅限这两条实际路径，不解决下面的 plant 扰动差异。

## 三份 plant 实际不同

| 来源 | yaw 导数 | speed 导数 | 扰动解释 |
| --- | --- | --- | --- |
| 2026 报告式 (3) | `u2` | `u1+w` | `w∈[-10^-4,10^-4]`；未明确它是每条轨迹恒定还是可随时间变化 |
| 固定官方 `dynamics10.m` | `f2(x)-20` | `f1(x)-20` | 注释提到 `w`，执行表达式完全没有它；注释上限还写成 `10^4`，与论文相冲突 |
| 保存的 CROWN/Flow* `Unicycle.cpp` 与旧 GPU YAML | `f2(x)-20+w` | `f1(x)-20+w` | 把同一 `w` 加入两条导数，并通过初始 `w∈[-10^-4,10^-4], w'=0` 实现恒定参数 |

因此三个来源不是同一闭环系统。旧四方结果采用最后一行，不能搬进 2026 论文式 (3) 的新主表。要运行论文式 (3)，需要在四方法的隔离配置中都把 `w` 只加到 speed 导数；若声称覆盖任意随时间变化的 bounded error，固定 `w'=0` 并不足够，必须有外部扰动语义和相应集合推进实现。

论文把性质写为“10 s 时间窗内到达”，旧原生代码仅在最后一次推进后调用 `fp_end_of_time.isInTarget`。完整的 `T=10` 终点包含是“某时刻到达”的充分条件，但不同于整个时间窗的精确事件检查；失败时不能反推时间窗到达失败。新 checker 须写明所用充分条件和如何处理未决。

## 当前执行门槛

可明确命名两个候选：`paper-w-in-speed`（仍须定 `w` 的时间语义）和 `official-matlab-no-w`；旧 `w-in-both` 只能是历史参与者合同。选主合同前不启动新的数值结果。仍缺一份能规定 `w` 随时间语义、以及 2026 参与者实际采用哪行 plant 的权威执行源码或用户选择。固定 2026 ONNX 已具备，不是此项阻断。
