# TORA reach-sigmoid：官方 2026 控制器的一期数值诊断

2026-10-02（中国时间）。此目录只记录新合同的一次有限诊断，**不是**论文主表的 5 秒结果或性质判定。没有启动旧 `u=22(f-0.5)` 合同、tanh 变体或 500 小步实验，也没有计算文件摘要。

## 合同与缩放

[2026 官方规格](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Tora_Heterogeneous/Specifications.txt)和固定官方 [`.mat`](nn_tora_sigmoid.mat)对应四态 `(x1,x2,x3,x4)`、完整单初盒 `[-0.77,-0.75] × [-0.45,-0.43] × [0.51,0.54] × [-0.3,-0.28]`、`x1'=x2, x2'=-x1+0.1sin(x3), x3'=x4, x4'=u`。控制器为 `4→20→20→20→1`，四层均 sigmoid；本诊断取官方规格的 `u=11f(x)`。论文合并文字对隐藏层/输出层的描述与此官方文件不一致，因此本诊断标为 **official2026_mat_u11**，不冒称论文字面网络。

从保存的 `.mat` 经 [构造器](controller_builder.py)生成 [ONNX](controller_plant_u.onnx)，把 `Mul(11)` 和 `Add(0)` 放在网络内部；[导出收据](controller_plant_u.onnx.json)的 22 点最大前向绝对误差为 `8.881784197001252e-16`。独立 [CPU 预检](PREFLIGHT.json)核对了四个 Sigmoid、内部 `11/0`、盒与动力学，并在中心点取得 plant 输入 `u=5.078156863296165`；该预检没有初始化 GPU。[运行配置](config.yaml)的外部 `output_scale=1, output_offset=0`，plant 直接使用 `x4'=u1`，所以没有二次缩放。配置仅含 `steps=1`，控制周期 0.5 秒，ODE 小步 0.01 秒、Taylor order 6，并去掉了只适用于完整时域的目标 checker。[原冻结配置](source_config.yaml)保留作来源对照。

## 唯一短程运行

Huan sparse plant 引擎与保存的作者 CROWN 驱动在物理 GPU 2、CPU 18–19 上执行一个完整初盒、一个控制周期；首次数值拒绝即停。原始 [START](START.json)、[RESULT](RESULT.json)、[stdout](stdout.log)、[stderr](stderr.log)、[metrics](metrics.json)、[逐步观察](observations.jsonl)与二进制 [ranges](ranges.bin)均保存。结果是 **50/50 小步接受**，监督进程 wall `5.345883 s`；驱动内部 `time cost` `1.673462 s`。这两个一次性诊断时间均不能形成四方速度排名。

[独立扫描](INDEPENDENT_INTERVAL_SCAN.json)从保存的 50 条唯一 `(lane,step)` 区间记录重算，全部有限、有序、接受，且每步末端区间包含于同一步 tube。数值如下：

| 状态 | 已保存 `t∈[0,0.5]` tube union | `t=0.5` 末端区间 |
| --- | --- | --- |
| x1 | `[-0.8850621763,-0.7498588580]` | `[-0.8850144075,-0.8573825037]` |
| x2 | `[-0.4501136567,0.0219013111]` | `[-0.0066841850,0.0219011293]` |
| x3 | `[0.5008911954,1.0343247573]` | `[0.9952107353,1.0343247573]` |
| x4 | `[-0.3001543770,2.2616427765]` | `[2.2364991939,2.2616427765]` |

运行仅覆盖 `t≤0.5`，**没有**评估论文 `T=5` 内进入目标 `x1∈[-0.1,0.2]、x2∈[-0.9,-0.6]` 的性质，也没有端到端 NNCS 浮点证书。运行源为 [一期入口](run_first_period.py)、[区间扫描器](scan_saved_ranges.py)、[保存的作者驱动](author_crown_reach_driver.py)及 [GPU 支持入口](author_support.py)。
