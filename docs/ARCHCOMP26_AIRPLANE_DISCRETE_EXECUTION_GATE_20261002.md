# ARCH-COMP26 Airplane discrete：来源核对与四方执行门

本页只核对固定来源和现有入口，未启动数值实验。`airplane-discrete` 仍是独立的未尝试格；连续版的 flowpipe、早停或失败记录均不能填写这一格。用户禁止内容摘要校验，本次没有执行此类操作。

## 1. 已确定的 2026 来源合同

| 项目 | 已确定内容 | 证据及边界 |
|---|---|---|
| 离散化 | 所有报告中的离散模型由连续导数用 forward Euler 得到，印刷页 89 写 `x(k+1)=x(k)+f(x) Δt`。Airplane 的 `Δt=0.1 s`。 | [2026 AINNCS 报告，PDF 页 5 与 11](https://easychair.org/publications/paper/GsKW/download)。页 89 只写通用 `f(x)`，没有给出参与者实际 NN 调用与更新程序。 |
| 物理态与初集 | `x=(sx,sy,sz,vx,vy,vz,phi,theta,psi,r,p,q)`；`sx=sy=sz=r=p=q=0`，`vx,vy,vz,phi,theta,psi` 六维各为 `[0,1]`。完整初集是一个未分割的 12 维盒。 | [报告 §3.7，印刷页 95](https://easychair.org/publications/paper/GsKW/download)、[固定官方规格副本](evidence/results/archcomp26_20261001/airplane_prep_001/specifications.txt)、[完整初盒 ledger](evidence/results/archcomp26_20261001/airplane_prep_001/INITIAL_FULL_BOX.json)。旧 `vx=vy=vz=1,phi=theta=psi=0.9` 单点不覆盖该盒。 |
| 连续导数 `f` | 固定官方 `dynamics.m` 返回 12 项，输入为 12 态与六控制 `(Fx,Fy,Fz,Mx,My,Mz)`。参数 `m=Ix=Iy=Iz=g=1,Ixz=0`。输出位置 10、11、12 分别为 `r'=Mz,p'=Mx,q'=My`。 | [固定官方 MATLAB 副本](evidence/results/archcomp26_20261001/airplane_prep_001/dynamics.m)、[逐式对照](ARCHCOMP26_NEXT_CONTRACT_SOURCE_AUDIT_20261001.md#2-airplane连续-2-秒与离散-20-次转移)。`dynamics.m` 是连续右端函数，本身不是离散转移程序。 |
| 控制器 | 固定官方 `controller_airplane.onnx`，实际检查的接口是 `FLOAT[N,12]→FLOAT[N,6]`；首层前没有图内显式预处理。MATLAB 的 action 槽位顺序是 `(Fx,Fy,Fz,Mx,My,Mz)`；历史连续入口将六个 NN 输出依次填入这些槽位，离散共同比较可明确沿用此映射。 | [官方固定文件](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/Airplane/controller_airplane.onnx)、[已有 ONNX 图检查记录](evidence/results/archcomp26_20261001/airplane_prep_001/CONTROLLER_INSPECTION.json)、[历史连续 C++ 入口](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/sources/native/submit/CROWN-Reach/archcomp/Airplane/airplane.cpp)。ONNX 接口本身不标出六个输出的物理名称，亦不证明四方运行时的控制包络。 |
| 索引与性质 | 20 次转移产生 `x_0,…,x_20`，在 **全部 21 个离散索引** `k=0..20` 检查 `sy,phi,theta,psi∈[-1,1]`。对应物理终点时间 2 秒；离散性质没有要求两索引之间的连续时刻。 | [报告 §3.7，印刷页 96](https://easychair.org/publications/paper/GsKW/download)、[官方规格副本](evidence/results/archcomp26_20261001/airplane_prep_001/specifications.txt)。顶层 README 将连续版写为 `[0,20]`，与报告和实例规格冲突；它不把离散 20 步变成 20 秒。 |

报告 PDF 第 89、95、96 页已经按原版面检查，不能仅凭文字抽取重排后的公式判断。固定官方 Airplane 目录的[此前目录审计](ARCHCOMP26_AIRPLANE_2026_ENTRY_AUDIT.md)只找到控制器格式、`dynamics.m` 和 `specifications.txt`，没有可选择的离散转移实现。此次网络页面直接读取未成功，所以“目录没有离散文件”沿用该固定版本审计，而不是声称做了新的在线目录证明。

## 2. 可以明确命名并实现的四方共同 Euler 比较合同

下面是**新比较约定**，不是已经核实的 2026 参与者离散执行记录。若选用，运行名、manifest、表格和图均须标为 `airplane-discrete / paper-Euler-controller-first`：

```text
X_0 = {12 维官方完整初盒}；先检查 Safe(X_0)
for k = 0,...,19:
    在旧状态 X_k 上，按固定 12 态顺序求 U_k = NN(X_k)
    同时用同一个旧状态 X_k 和 U_k 评估 MATLAB 的全部 12 个 f 分量
    X_{k+1} = X_k + 0.1 f(X_k,U_k)     # 12 维同步赋值，不逐坐标原地更新
    保存 X_{k+1} 的物理态包络；检查 Safe(X_{k+1})
```

点值闭环映射为 `x_{k+1}=x_k+0.1 f(x_k,NN(x_k))`。集合实现可保留 `x` 与 `NN(x)` 的相关性，或用包含它们的外包络计算，但必须保守包含上述点值映射。六个输出是本次转移所用控制，不能把 `NN(X_{k+1})` 回填本次转移，也不能将 12 个更新分量按新旧混合状态依次求值。每一步的 `cos(theta)` 分母必须在当前外包络上排除零；若不能排除，立即记录 `Unknown/undefined`，不得默默继续。`theta∈[0,1]` 的初盒满足此域条件，但未来步要再次核验。

完整初盒可以按明确的共同分区精确覆盖；原始单盒和分区 ledger 都需保存。四方均须有真正的离散转移入口，保存 `k=0..20` 的 endpoint 集、NN 调用次数、每步接受/停止状态和性质状态。安全只有在所有初盒、所有 21 个索引的可达集均落入闭带时才可报通过；包络碰到带外而没有有效反例只能报 `Unknown`。性能和宽度对比以同一离散索引、同一初集覆盖和同一物理 12 态为准，不把连续 tube 当成离散 endpoint。ODE 内步长、Taylor 时间阶数及连续 tube 余项在此合同中均不适用。

这个顺序是根据报告的 Euler 规则、同周期控制间隔和历史连续入口的“周期起点 NN → 推进”**推导的可执行选择**。报告与固定 MATLAB 文件并未独立写出 `U_k=NN(X_k)` 的实际程序先后，因此不能把该选择标作“官方离散提交的忠实复现”。

## 3. 四方现有入口及不能代用之处

| 方法 | 现有 Airplane 入口 | 与真正离散版的差额 |
|---|---|---|
| Flow* native | 历史 [`airplane.cpp`](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/sources/native/submit/CROWN-Reach/archcomp/Airplane/airplane.cpp) 第 118–169 行在周期起点求 NN，写入六个 held-control TM，再调用 `dynamics.reach(...,0.1,...)`；第 66–69 行是旧单点。 | 它积分连续 ODE，且 `COMPLETED_UNKNOWN` 分支未推进 `initial_set` 也未退出。需要单独的同步 Euler 12 态转移、完整初盒和离散索引 checker；改初值或改 `h` 仍不是离散版。 |
| Huan、Xiangru | 旧[配置](../research/gpu_verified_20260930/report/configs/arch_airplane.yaml)与[盒 ledger](../research/gpu_verified_20260930/report/configs/arch_airplane_boxes.json)是 19 变量、`0.01 s` ODE 内步、20 个 `0.1 s` 连续周期和上述单点。新完整初盒 smoke 仍走连续 NNCS 驱动。 | 两法均缺已核实的 Airplane 离散 Euler 入口。连续 smoke 首小步拒绝或建表内存失败，不代表离散映射已经失败。 |
| 我方 P3 | 新[完整初盒连续 P3 入口](../tools/archcomp26_airplane_continuous_p3_nohash.py)为 19 变量连续求解；已有 CPU 六输出注入预检。 | P3 连续验证阶表限制与首步拒绝只约束该连续入口。不能只删除 `h` 参数就声称有离散实现，仍需带包络的 12 态同步 Euler 映射。 |

当前[四方连续全初盒短程摘要](ARCHCOMP26_AIRPLANE_P3_NATIVE_ENTRY_AUDIT_20261002.md)只说明连续入口状态，不能给 `airplane-discrete` 填运行时间、宽度或性质结论。现有[工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)中该实例四格均为未尝试。

## 4. 执行门与明确缺件

1. **若目标是复现 2026 实际离散提交**：仍缺该提交使用的离散转移源码或同等权威运行记录，尤其是控制器取样发生在 `x_k` 还是更新后的状态、六输出施加在哪一次转移、12 态是否同步更新。当前固定论文及目录不能给出这份程序证据；不能猜定成官方顺序。
2. **若接受新的四方共同比较**：可冻结上节明示的 `paper-Euler-controller-first` 合同，分别为 P3、Huan、Xiangru、Flow* native 建新离散入口。先做不推进数值的 CPU 合同预检：读固定模型接口、12 项 MATLAB 右端、全初盒、20 次控制更新与 21 个检查索引；再在独立 run ID 做一步 smoke。任何一法未实现必须保留为未支持/失败，不能用连续记录填空。
3. **两种路径都需要的运行凭据**：固定 2026 模型实际加载路径、四方法对 `12→6` 及输出次序的检查、共同初集覆盖、离散映射表达式、每步端点/状态、控制器包络与舍入策略、性质判定和停止原因。这些可直接记录为文本与小数据，不要求内容摘要校验。现有 `discrete_execution_contract_v1` 的离散字段列表见[manifest](../benchmarks/archcomp26/manifest.json)；旧启动器的身份门并非本次无摘要实验的现成入口。

截至本审计，没有证据支持填写四方离散成绩；已经确定的是一份可实施的**新**比较约定及其与官方复现身份的界线。

## 后续有界入口诊断

此审计之后，按上面的新命名约定执行了两个**四方方法之外**的 CPU directed 区间诊断。[原始一步](evidence/results/archcomp26_20261001/airplane_discrete_paper_euler_interval_smoke1_001/AUDIT.md)在 `k=1` 因精确零角速度经过通用外舍入而触及闭带外一个浮点格；保留其原始记录。[精确零恒等式诊断](evidence/results/archcomp26_20261001/airplane_discrete_paper_euler_exactzero_prefix_001/AUDIT.md)在固定动力学代数上证明首步角度导数为零后，得到 `k=1` 安全端点，随后 `k=2` 外包络显著越带而首个 `Unknown` 即停。该结果只给 **1/20** 次转移的安全端点前缀，不是 P3/Huan/Xiangru/native 任何一方的完整离散运行，也不解决参与者实际控制/转移顺序的来源缺件。
