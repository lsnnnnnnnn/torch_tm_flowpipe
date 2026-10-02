# Airplane discrete：新 paper-Euler-controller-first 一次转移诊断

独立运行目录：`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/airplane_discrete_paper_euler_interval_smoke1_001`。原始 [`RESULT.json`](RESULT.json) 已原样镜像。入口为 [`tools/archcomp26_airplane_discrete_euler_interval_smoke_nohash.py`](../../../../../tools/archcomp26_airplane_discrete_euler_interval_smoke_nohash.py)。没有再次启动已有连续 Airplane 实验。

本运行采用**新比较配置**：固定 2026 官方完整 12 维初盒，在旧状态调用固定 12→6 ONNX，再依固定 `dynamics.m` 的 12 个方程同步计算 `x₁=x₀+(1/10)f(x₀,NN(x₀))`。区间网络与物理转移使用服务器现有 CPU directed 区间算术；`cos(theta)` 初盒 `[0.5403023058681393,1]`，已排除零。只执行 `k=0→1`，不是已核实的参赛者离散执行次序，也不是 P3/Huan/Xiangru/Flow* native 任一方法的成绩。

结果：原始 status 为 `first_transfer_property_unknown`，wall `0.037324500270187855 s`。`sy₁∈[-0.08414709848078979,0.31453676396724295]` 在安全带内；`phi₁,theta₁,psi₁` 的包络下界有约 `10⁻³²³` 的负值，上界均为 `1.0000000000000002`，故区间 checker 对这三个边界变量报 Unknown。初盒的 `r=p=q=0` 使实数 Euler 首步角速度恰为零；这个窄幅溢出来自逐运算外舍入，不能记录为真实反例。运行按首次 Unknown 停止，没有推进到 `k=2`。

控制器六输出的初盒区间相当宽，例如 `Fx∈[-197.1800425294152,146.23170353268074]`；这使速度分量在首步明显扩张。三个点值 NN 计算共 18 个输出值落在保存包络内，仅作为程序检查。现有区间三角函数源码注明其舍入余量按另一 PyTorch 版本校准，本次环境为 `torch 2.5.1+cu121`；本诊断不建立独立端到端浮点证书。完整 20 次转移、四方法数值成绩及运行时间排名均未得到。

若要填写 2026 实际离散提交复现身份，仍缺参赛者的离散转移/控制施加顺序源码或同等权威运行记录。这个新比较配置即便继续，也须保留自己的名称与证据链。
