# ARCH-COMP26 Docking：完整初盒、三种 GPU 方法的全程诊断

记录时间：2026-10-01 UTC。全部是新 run ID；没有重启旧实验，也没有做内容摘要或哈希校验。本文详述三种 GPU 方法；Flow* native 独立的[新全盒结果与审计](native_docking_full40_001/SUMMARY.md)另列，不以本文替代其证据。

## 冻结合同与独立性质检查

以 [2026 AINNCS 报告 §3.10](https://easychair.org/publications/paper/GsKW/open) 和固定官方 [Docking 规格](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Docking/specification.txt)、[动力学](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Docking/dynamics.m)、[ONNX](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Docking/model.onnx) 为准：单个完整初盒 `[70,106]²×[-0.28,0.28]²`；状态 `(sx,sy,vx,vy)`，控制 `(Fx,Fy)`；模型直接接收原始四态并输出物理力，不额外归一化或裁剪。每个整数 `k=0,…,39` 更新一次控制并保持 1 s，连续动力学为

```text
sx'=vx,  sy'=vy,
vx'=2·0.001027·vy + 3·0.001027²·sx + Fx/12,
vy'=-2·0.001027·vx + Fy/12.
```

全时含端点安全性质是 `q = sqrt(vx²+vy²) - 0.2 - 0.002054·sqrt(sx²+sy²) ≤ 0`，即 `t∈[0,40]`。每周期分为 10 个 0.1 s ODE 子步，3 阶 Taylor 设置、同一单盒 ledger；0.1 s 是明示的方法设置，不是官方 NN 更新周期。实际运行的旧参与者 ONNX 文件已在先前合同审计中与固定官方 ONNX 直接逐字节比较相同；本轮没有再作文件身份计算。CPU 上原始 `[N,4]→[N,2]` 前向与完整初盒的 CROWN 包络预检通过，中心 `(88,88,0,0)` 输出约 `(-0.99375157,-0.89423484)`。

原通用区间 AST 把两个平方项的零下界作向外加法后得到微小负数，从而把零速处的平方根误判为定义域错误。隔离入口仅对本性质使用有向区间范数：平方和的下界按其数学非负性与零相交，再求有向区间平方根及 `q`；四维每段 tube 的轴对齐盒是输入。它给出保守包络，不能利用速度和位置之间的相关性。初盒单独检查 `q∈[-0.5079082336541202,-0.007355828533536612]`，安全。

## 全部新尝试

下表 wall 为外层进程从启动到退出的单次时间，包含加载和记录开销；三法共用驱动框架并且服务器有其他并行任务，因此不能按这些冷启动数值排名。`accepted` 只表示数值步骤得到有效包络，不能把 Unknown 改写为 Verified 或真实反例。

| 尝试 | 外层 wall (s) | accepted/预期 | 全时性质结论 | 原始证据 |
|---|---:|---:|---|---|
| Huan smoke #1，导入前失败 | 0.0641 | 0/10 | 没有数值结果；公共模块路径未设置 | [记录](docking_huan_smoke1_001/RESULT.json)、[stderr](docking_huan_smoke1_001/stderr.log) |
| Huan smoke #2 | 4.7301 | 10/10 | Unknown | [结果](docking_huan_smoke1_002/detail/RESULT.json)、[tube](docking_huan_smoke1_002/detail/ranges.jsonl)、[性质](docking_huan_smoke1_002/detail/safety.jsonl) |
| **Huan full 40 s** | **12.6988** | **400/400** | **Unknown** | [结果](docking_huan_full40_001/detail/RESULT.json)、[tube](docking_huan_full40_001/detail/ranges.jsonl)、[性质](docking_huan_full40_001/detail/safety.jsonl) |
| Xiangru smoke | 4.9786 | 10/10 | Unknown | [结果](docking_xiangru_smoke1_001/detail/RESULT.json)、[tube](docking_xiangru_smoke1_001/detail/ranges.jsonl)、[性质](docking_xiangru_smoke1_001/detail/safety.jsonl) |
| **Xiangru full 40 s** | **12.6681** | **400/400** | **Unknown** | [结果](docking_xiangru_full40_001/detail/RESULT.json)、[tube](docking_xiangru_full40_001/detail/ranges.jsonl)、[性质](docking_xiangru_full40_001/detail/safety.jsonl) |
| P3 smoke | 5.1780 | 10/10 | Unknown | [结果](docking_p3_smoke1_001/detail/RESULT.json)、[方法](docking_p3_smoke1_001/detail/P3_METHOD.json) |
| **P3 full 40 s** | **17.4118** | **400/400** | **Unknown** | [结果](docking_p3_full40_001/detail/RESULT.json)、[tube](docking_p3_full40_001/detail/ranges.jsonl)、[性质](docking_p3_full40_001/detail/safety.jsonl)、[方法](docking_p3_full40_001/detail/P3_METHOD.json) |

三份完整尝试均已逐行复核：`ranges.jsonl` 与 `safety.jsonl` 分别有 400 行，子步编号连续 `1..400`，两份记录的同段 `q` 区间一致、上下界有序。三法首次 `q` 上界大于零均为第 1 个子步，即 `t∈[0,0.1]`：Huan/Xiangru `[-0.5079914003077841,+0.016082847580048576]`，P3 `[-0.5079914000561839,+0.01608284732764942]`。它们均横跨零，只说明此保守包络不能证明安全。400 段内的最大 `q` 上界，Huan/Xiangru 为 `+8.808307740090513`，P3 为 `+8.806147900413093`。

末时 `t=40` 的四态 endpoint 轴对齐包络，Huan/Xiangru 为 `sx∈[-52.610230807548426,201.84748319980218]`、`sy∈[-51.83858371010658,205.10465085079335]`、`vx∈[-6.403299104826275,5.433949211484817]`、`vy∈[-6.3361951041994695,5.538329874480357]`；P3 对应 `sx∈[-53.4848979912236,202.74696059717763]`、`sy∈[-52.5454688333426,205.83698662805992]`、`vx∈[-6.401580056763278,5.433522939081758]`、`vy∈[-6.334861661219265,5.53810686231263]`。Huan 与 Xiangru 两个运行的保存数值摘要一致，未以内容摘要工具作比较。

三法 `q` 上界全程图：[PNG](plots/nohash_saved_20261001/docking_fullbox_3method_q_upper.png)、[PDF](plots/nohash_saved_20261001/docking_fullbox_3method_q_upper.pdf)、[MATLAB 脚本](plots/nohash_saved_20261001/docking_fullbox_3method_q_upper.m)、[几何 JSON](plots/nohash_saved_20261001/docking_fullbox_3method_q_upper.geometry.json)、[渲染记录](plots/nohash_saved_20261001/docking_fullbox_3method_q_upper.render.json)。曲线读取各自 `safety.jsonl` 的 400 段保守上界，黑点是初盒安全上界；图中的正上界只表示 Unknown，不是真实轨迹反例。

另存四方扩展图：[PNG](plots/nohash_saved_20261001/docking_fullbox_4method_q_upper.png)、[PDF](plots/nohash_saved_20261001/docking_fullbox_4method_q_upper.pdf)、[MATLAB 脚本](plots/nohash_saved_20261001/docking_fullbox_4method_q_upper.m)、[几何 JSON](plots/nohash_saved_20261001/docking_fullbox_4method_q_upper.geometry.json)、[渲染记录](plots/nohash_saved_20261001/docking_fullbox_4method_q_upper.render.json)。第四条 Flow* native 读取其[原始 `safety.tsv`](native_docking_full40_001/safety.tsv)，`global_substep=s` 的 q 区间覆盖闭段 `[(s−1)/10,s/10]`。其 400/400 数值 tube 已完成，但独立 checker 为 UNKNOWN、外层 exit 2；四方图只是保守 box 上界的对齐呈现，具体原生结论以[独立摘要](native_docking_full40_001/SUMMARY.md)为准。两图均以淡绿色标明 `q≤0` 安全区域。

## 方法与限制

- Huan/Xiangru：各自已保存的 GPU plant engine；共享 CROWN-Reach Python 控制驱动；box/same-slope CROWN；RPC float32 投影；严格 plant 模式。入口源 `tools/archcomp26_docking_author_nohash.py`。
- P3：保存的 working-P3 plant core、严格端点及 `(7,4,2)` 双输出严格控制注入；同一 box/same-slope CROWN 与 RPC float32 投影。入口源 `tools/archcomp26_docking_p3_nohash.py`，每个 P3 run 的 `P3_METHOD.json` 列出引擎和预建库。
- 三法的控制 NN 包络/注入尚无独立完整浮点证明，`end_to_end_floating_point_nn_certificate=false`。以上是新合同下的记录完整性和保守性质诊断，不是严格端到端安全证书。末端 ONNX Tanh 真值范围不能被偷偷当作对已计算 CROWN 包络的外部裁剪。
- 性质 checker 只用每段四态轴对齐 tube；更紧的耦合 TM checker、合法全盒分箱或别的精化也许改变 Unknown，但必须另立 run ID、保存覆盖 ledger 和方法设置，不能追认本轮为 Verified。论文自身也指出 Docking 的非线性性质在 2026 年仍很难验证。
