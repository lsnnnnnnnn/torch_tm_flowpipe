# Airplane continuous：P3 与 Flow* native 全初盒入口只读审计

本页先记录 P3/native 新入口的只读可行性检查，文末补记随后两次独立 P3 和三次独立 native smoke。旧单点成绩未被复用，没有做内容摘要或哈希校验。

## 固定合同与已有证据

固定 2026 [规格](evidence/results/archcomp26_20261001/airplane_prep_001/specifications.txt)、[动力学](evidence/results/archcomp26_20261001/airplane_prep_001/dynamics.m) 和[来源审计](ARCHCOMP26_AIRPLANE_2026_ENTRY_AUDIT.md)给出一个未分割的 12 物理态盒：按 `(x,y,z,u,v,w,phi,theta,psi,r,p,q)` 排列，`x=y=z=r=p=q=0`，`u,v,w,phi,theta,psi∈[0,1]`。在 `t=0,0.1,…,1.9` 从当时的 12 态集合求 `[N,12]→[N,6]` 固定 ONNX，六输出按 `(Fx,Fy,Fz,Mx,My,Mz)` 注入并保持 0.1 s；连续 ODE 共 20 周期至 `T=2 s`。初态及整个闭时间区间的 tube 均要求 `y,phi,theta,psi∈[-1,1]`。本地 [ONNX 检查记录](evidence/results/archcomp26_20261001/airplane_prep_001/CONTROLLER_INSPECTION.json)给出 FLOAT 接口、网络层 `12→100→100→20→6` 和无图内显式输入预处理；旧参与者模型与所选官方文件的直接字节比较由前次审计记录，本页不重新计算身份。

[CPU-only 预检](evidence/results/archcomp26_20261001/airplane_continuous_order3_fullbox_20261002/PREFLIGHT.json)已检查一盒 ledger、19 条含辅助变量的 ODE、8 条安全表达式和模型形状。辅助 `t,Fx,Fy,Fz,Mx,My,Mz` 从零初始化，不是额外的官方不确定物理维度。候选 `h=0.01`、Taylor order 3/6 是明示方法设置，并非官方强制值。初始 `theta∈[0,1]` 内 `cos(theta)>0`；后续 tube 仍须逐段检查分母定义域。

## 已有尝试不能代填

历史 native、Huan、Xiangru、P3 Airplane 运行用 `u=v=w=1, phi=theta=psi=0.9` 单点；旧本地来源 `results/archcomp_review_20260923/sources/native/submit/CROWN-Reach/archcomp/Airplane/airplane.cpp` 第 66–69 行直接写死该初值。它们不覆盖官方六维完整初盒。旧 C++ 对 `COMPLETED_UNKNOWN` 只改标志而未更新 `initial_set` 或退出，不能作为新入口直接使用。

三个较新的全初盒作者方法 smoke 已有[独立摘要](evidence/results/archcomp26_20261001/airplane_continuous_order3_fullbox_20261002/SUMMARY.md)：Huan order 6 在零小步阶段因 monomial 表构建超过 54 GiB RSS 而停止；Huan/Xiangru order 3 均在第一个 `0.01 s` 小步对唯一完整初盒返回 `accepted=false`，保存范围为空。其内部验证失败细因尚无记录，因此无法推断 P3 或 native 会通过，更不能得出安全/不安全结论。

## P3 入口成本与明确缺件

已保存 P3 驱动和预建 CUDA 扩展可为新入口提供工作三阶核心、严格 endpoint 与控制注入、`rpc-float32` 传输和同斜率 box CROWN。当前 [`strict_injection`](../tools/archcomp26_dp_p3_nohash.py)只准许既有 `(n,输入维,输出维,控制行)` 布局，**不含 Airplane 所需 `(19,12,6,(13,14,15,16,17,18))`**；应在隔离入口显式加入并用六输出、一个完整盒作形状/数值有限性预检，不能仅改名字。已有 19 变量 order-3 配置与作者驱动形状相符，但 P3 对此维度和这些三角/倒数项尚无实际小步结果。

当前引擎三阶工作基底含 `C(23,3)=1,771` 个单项式，成对乘法表有 `1,771²=3,136,441` 项；六阶相应为 `C(26,6)=230,230` 个基底和约 530 亿成对项。因此 P3 最小数值诊断应明确为 **order 3**，不能把结果写成旧 order-6 方法的复现。三阶工作/四阶验证的整数编码及首个数值步接受情况见文末新增尝试；需独立目录、预设内存/超时限制并保存首次拒绝的引擎内部状态。

## Flow* native 入口成本与明确缺件

服务器已有的历史 `archcomp/airplane/matched.cpp` 含同一 19 变量 ODE、12→6 RPC、`0.1 s` 采样保持和四个坐标的 8 条性质约束，且有 `matched_reach.h` 和 `arch_ranges.h` 可在固定 `0.01 s` 子步记录 tube/endpoint。它的初始化仍是上述旧单点，`COMPLETED_UNKNOWN` 后继续循环；因此需要隔离源码与编译目录，写入完整 `[0,1]` 六维盒，保留辅助七维初值零，修正每周期结果判定与 `initial_set` 更新，只对已接受且可继续的 endpoint 进入下一周期。RPC 应显式使用已选 2026 模型文件、独占端口，并记录 20 次周期起点的 12→6 调用及六个 held-control 输出。随后已有隔离构建和一周期 smoke，结果见文末；旧单点成绩仍不能推出新全初盒编译/运行表现。

## 最小新 smoke 顺序

两法各自使用新 run ID 与**同一个未分箱官方初盒**；先存 `START`、初盒 ledger、实际 ONNX 路径/形状、实际配置、控制输出顺序和资源限制。只运行第一控制周期 `[0,0.1]`，每个 `0.01 s` 子步保存完整 12 物理态 tube 与 endpoint，并对初盒、每段完整 tube 的 `y,phi,theta,psi` 八个带约束分别记录保守上下界。任何 NN 包络错误、`cos(theta)` 分母域错误或数值拒绝都立即停止，保存首个原因/时刻。性质 Unknown 时可以保存数值前缀并报告；若十个小步均数值接受，再结合资源和性质记录决定是否另立 full20 完整数值诊断，仍不能把 Unknown 当作已验证。区间越界不能充当真实轨迹反例。两法都需单独说明控制 NN 与浮点区间链是否有完整端到端证明。

## 随后 P3 新尝试

完整原始证据、CPU 预检和两次独立 run ID 见[P3 smoke 摘要](evidence/results/archcomp26_20261001/AIRPLANE_P3_FULLBOX_SMOKES_20261002.md)。默认 `solution_plus_one` 在首步前创建四阶验证表时遇到 `9^20 >= 2^63` 整数编码限制，0 个已接受段。新 `solution_order` 严格同阶验证越过该阻断，但唯一全初盒的首个 `0.01 s` 数值段 `accepted=false`，0/10 段可用，且当前记录未细分内部验证失败原因。入口立即停止，两次均没有全时性质结论；P3 full20 未启动。

Flow* native 的[三次独立完整初盒 smoke](evidence/results/archcomp26_20261001/native_airplane_fullbox_smokes_20261002/SUMMARY.md)各完成一次控制 RPC，但首个 `0.01 s` Flow* 子步均为状态 4 `UNCOMPLETED_SAFE`、零已接受 tube，原始外层 `failed/exit2`。历史 order 6、order 3、加宽余项 `[-1,1]` 的 order 3 是三种明确不同数值设置；最后一种仅为参数诊断。保存源码可把 status 4 的停止追到 Picard 余项未被预置估计包含，但失败坐标和计算所得余项界没有日志，不应猜其具体值。性质没有可用 tube，native full20 未启动。
