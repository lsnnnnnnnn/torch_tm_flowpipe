# 旧作者 QUAD 合同：P3 的 40 步 observer 消融（无摘要、无 JIT）

这是独立于 2026 论文修正方程的**新短前缀诊断**。输入 [quad_40.yaml](quad_40.yaml) 与已存 Huan Stage A 40 步 YAML 经直接逐字节比较相同，保留其旧作者 x2/x4/x5 方程、1,024 初盒、控制周期 0.1 s 与 ODE 小步 0.005 s；本入口将工作阶设为 P3、ODE point 阶保持 P2、validation 阶为 P4，并显式设 SR queue 1000。它不是 2026 paper QUAD，也不是旧 P3 全程作业的重启。

隔离 [runner.py](runner.py) 和同目录的无摘要适配器沿用已完成的新 P3 入口的数值推进、控制、SR 和工作图逐出路径，使用八个既存 CUDA `.so`。摘要构造器、CUDA 构建与 `torch.compile`/trace 被拦截；auto_LiRPA 的 `@torch.jit.script` 被替换成纯 Python 恒等装饰器，**不执行 TorchScript 编译**。两成功臂均使用该相同非 JIT 路径。这改变了冻结全程 P3 的执行环境，因此以下计时不能与冻结的 1,000 步 P3 时间相减作优化归因。

## 运行与直接数值核对

| 新目录 | 结果 | driver 内部 `elapsed_s` | 外层进程 wall | observer 区域 wall |
| --- | --- | ---: | ---: | ---: |
| [observer_on_001](observer_on_001/RESULT.json) | **0/40**，导入 auto_LiRPA 时 JIT 守卫拦截装饰器；保留失败日志 | — | 2.521601 s | — |
| [observer_on_002](observer_on_002/RESULT.json) | **40/40**，40,960/40,960 盒步接受，`broken=0` | 53.554101 s | 60.379012 s | 0.182725 s |
| [observer_off_002](observer_off_002/RESULT.json) | **40/40**，40,960/40,960 盒步接受，`broken=0` | 53.323556 s | 60.028937 s | 0 s |

两臂顺序占用物理 GPU3、CPU14–17，各为独立进程，外层 600 s 上限。`observer_off` 只跳过**每个 ODE 小步之后**的 `hull_ranges_s`、`rows_range_over_time_sparse`、接受盒的 12 态 pooled 区间归约与导出；它仍保存每步接受数、状态计数及 SR 长度/epoch。原 driver 为控制器输入及终点检查所需的范围计算、最终指标、真实工作图逐出和数值推进仍执行。两个臂都在 driver 内部计时结束后另外保存终步逐盒 `final_tube_12x2.npy`、`final_endpoint_12x2.npy` 和 `final_status.npy`，供直接比较。

[比较脚本](compare_arms_nohash.py) 对两份原始日志与数组逐值检查，结果在 [COMPARISON.json](COMPARISON.json)：40 步的接受数/状态计数/SR 长度与 epoch 相同；两次 NN 控制记录、终点 16 变量聚合 hull 和其均值宽度相同；终步 1,024 盒 × 12 态的 tube、endpoint 数组及状态数组也完全相同。两臂共同的 t=0.2 `x3` 终点聚合为 `[-0.27693202339139644, 0.543910762761349]`；短前缀打印的 `FALSIFIED` 只是把 T=5 终点目标提前用于 t=0.2，**不是完整任务性质结论**。这些直接观测没有覆盖全程，也没有比较每步完整内部 plant/SR 张量。

单次 on−off 的 driver 时间差为 **0.230546 s**，外层进程 wall 差为 **0.350075 s**。on 臂包围逐步 observer 计算、同步和 CPU 转移的区域累计 **0.182725 s**；该区域不是纯数学求界核时间。每臂仅一次、固定运行顺序，JIT 被禁用，图/内存调度与同步可能扰动计时；不能据此估计 1,000 步净收益、稳定速度排名，或把旧冻结 working-prune 混合区的 191.723381 s 当成可移除时间。两臂均已没有旧 `mathematical_signature()`，所以**本实验没有测量签名 on/off 消融**。

判断：在此 40 步前缀，observer 导出是可见但较小的开销；删除它会失去逐步证据。没有支持把删除 observer 或旧签名路径移植为全程加速结论。若以后继续，需单独核对无 JIT 与原 controller 的数值等价、覆盖后期 SR 历史并作匹配重复计时；本次没有启动长程实验或执行任何内容摘要校验。
