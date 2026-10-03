# Huan QUAD / 绘图 / ARCH-COMP26 执行进度

日期：2026-10-01。此页记录本次接续后的实际状态；完整目标仍在进行中。

## 已核对的接续点

- GitHub 交接分支为 `codex/gpu-verified-handoff-20260930`；本地已有后续执行分支 `codex/huan-plot-archcomp-execution-20260930`，包含 A 阶段报告草稿、绘图功能、16×4 新版矩阵和最终报告草稿。接续前的旧矩阵 64 个格子均未填入新实验结果；本轮新增尝试另见下方无哈希矩阵，旧成绩没有挪用。
- 服务器旧 Huan parity QUAD 完成 1,024 盒 × 1,000 步；旧 strict 首次在第 597 步拒绝。当前 P3 和 trig 版均有 1,000 步完成记录。原生 QUAD 六小时作业自然超时，只记录 600 步共同完整前缀；旧进程已退出。没有重启这些原作业。
- 用户要求本轮**不做任何哈希校验或 SHA-256**。本轮读取历史身份记录，但没有重新核验源码或预编译二进制的内容身份。一次读取远端 Git 提交号的请求被自动审批以此理由拒绝，此后未重试或变通。

## 本轮新增结果

1. 用全 1,024 初盒做了新的 Huan 40 步 parity / strict 两臂短程诊断，另做一次 parity 函数级计时。三次都是新目录、新进程，预载已有 CUDA 扩展并禁止 JIT。两臂均 `broken=0`。内部时间 parity 1.412598 s、strict 1.504988 s；进程时间 5.61 s、5.70 s。strict 的终点每盒 16 变量宽度和均值比 parity 高约 0.40%。阶段计时中，40 次 `advance_sparse` 共 1.051305 s，2 次 `crown_bounds` 共 0.316279 s。全部原始文件和计时边界见 [40 步诊断摘要](evidence/results/huan_quad_stage_a_40_20261001/SUMMARY.md)；分析见 [Huan 模式与速度报告](HUAN_QUAD_SPEED_AND_MODES.md)。这些数字不能外推为完整 5 秒的速度或性质结论。
2. 用户选定 2026 论文方程为新版 QUAD 四方主合同。论文与官方仓库/旧作者代码在 `x2'`、`x4'`、`x5'` 三处不同；旧 Huan 约 80 秒结果仍按旧作者合同解释。逐式来源和未定控制器见 [QUAD 合同决策](ARCHCOMP26_QUAD_PAPER_CONTRACT_DECISION_20261001.md)。
3. 已从论文、官方规格/动力学/控制器和旧入口核出 [Double Pendulum less-robust 的首个新版共同合同](ARCHCOMP26_DOUBLE_PENDULUM_LESS_CONTRACT_20261001.md)：四物理态、全初盒 `[1,1.3]^4`、225 子盒、20 个 0.05 s 控制期、全程四维安全带 `[-1.7,2]`。旧三套 GPU 结果只是回归线索，不能充当新版重跑。
4. 已添加 [无哈希单格执行器](../tools/run_archcomp26_nohash.py)，可独占新目录、保存命令与原始日志、拒绝重复运行并在超时时清理进程组。假子进程自检通过。现有 ARCH-COMP 预检/矩阵/报告程序依赖哈希绑定，本轮不会把无哈希数据伪装成它们的合格结果。
5. 对完成的 P3+trig 运行做了只读逐步计时审计：19 次非空图逐出步骤的混合计时合计 191.723381 s，其余 981 次合计 1.229124 s。前者包括全状态诊断签名和必要的 CUDA 同步/图逐出，不能当成净可省时间；调用链、候选改动和无哈希入口缺口见 [P3 优化审计](evidence/results/huan_quad_stage_a_40_20261001/P3_OPTIMIZATION_AUDIT_NOHASH.md)。没有改动旧数值源码或启动优化测速。

## 绘图和剩余边界

已有 QUAD 的时间—状态、状态—状态、原生保存范围的 box 投影，以及 MATLAB `.m`、PNG、PDF；见 [绘图说明](flowpipe_plotting.md) 和 [保存数据验证](FLOWPIPE_PLOT_VALIDATION_20261001.md)。本机与服务器 PATH 都没有 MATLAB/Octave，因此尚无 `.m` 实跑证据。保存的 P3 observer 只有逐坐标区间，无法恢复带相关性的 octagon；未来要在接受步保留的 Taylor 模型上导出八方向支持界，并单独验证。

新增 [无哈希绘图入口](../src/torch_tm_flowpipe/flowpipe_plot_nohash.py) 可从 native `ranges.bin` 直接生成几何 JSON、MATLAB `.m`、PNG/PDF 与只记录路径/大小的收据。其调用链在 SHA 构造器拒绝测试下通过。新完成的 DP less 原生全程数据已另存本地小型数据包并生成 [t–θ₁ tube 图](evidence/results/archcomp26_20261001/native_dp_less_full20_001/plots/dp_less_native_225x100_t_theta1_tube.png) 和 [θ₁–θ₂ endpoint 图](evidence/results/archcomp26_20261001/native_dp_less_full20_001/plots/dp_less_native_225x100_theta1_theta2_endpoint.png) 等三组。图中初盒为 `[1,1.3]^4`，Safe region 是全时 `[-1.7,2]^4`；native 范围记录本身不含 accepted/status，图上仅声明记录覆盖。

Double Pendulum 旧 native 300 秒作业超时。只读审计发现其二进制请求端口是 5200，而同目录的 server/launcher 监听 5100；旧 5625 条范围记录不能作为已配对的四方结果。已从服务器旧离线轮包在新目录提取 `gevent`、`tinyrpc` 等 RPC 依赖并通过导入 smoke。端口一致的新 native **225 盒、1 控制期 plumbing smoke** 在独立目录 `.../runs/archcomp26_20261001/native_dp_less_smoke1_001` 完成：53.211 秒、1 次 RPC、1125 条范围记录（225 盒 × 5 个 0.01 s ODE 小步），服务与求解器正常退出。这不是完整 benchmark 成绩。

随后另一新目录 `.../runs/archcomp26_20261001/native_dp_less_full20_001` 的原生 **225 盒 × 20 控制期**正式尝试已完成：进程 wall **1107.127423 s**（求解器自报 1100.101 s），20 次 RPC，22,500 条唯一且完整的 `(lane,step)` 范围记录（225 盒 × 100 小步），无超时或数值拒绝。作者 checker 报 `VERIFIED`；独立读取保存区间，四个物理态全时 tube 均位于官方 `[-1.7,2]` 安全带，未发现 NaN、逆序或 endpoint 越出 tube。完整 T=1 endpoint/tube 绝对区间、每盒宽度与计时边界见 [native DP less 摘要](evidence/results/archcomp26_20261001/native_dp_less_full20_001/SUMMARY.md)。这不构成独立的浮点 NN 证明，也不能单独填满四方比较。其它新版实例的模型、控制器和性质冲突仍需逐项冻结，不能猜测或删行。

本轮新尝试的机器可读索引是 [无哈希 attempt 记录](evidence/archcomp26_nohash_attempts_20261001.json)，与旧版摘要绑定 16×4 矩阵分开，明确短程 smoke、完整单次结果和失败预检的不同资格。

## 随后完成的新尝试

- **论文方程 QUAD / Huan**：独立新 [配置](../benchmarks/archcomp26/configs/quad_paper_p2_huan.yaml) 下，1,024 盒 × 50 控制周期、1,000 ODE 小步全部完成，`broken=0`，进程 wall 94.583 s，driver 90.471 s。作者现有终点 checker 给出 `VERIFIED`，T=5 的 `x3` union `[0.967434441417146,1.015176258384569]` 位于论文目标内。旧约75秒用的是不同动力学，不能直接作速度变化。见 [新运行摘要](evidence/results/archcomp26_20261001/quad_paper_huan_full50_001/SUMMARY.md)；reach-and-remain 的完整时间语义与独立浮点 NN 证明仍未闭合。
- **DP less / 四方实际尝试**：原生完整结果如上。新 Huan/Xiangru 共享驱动各完成 225 盒 × 20 周期、22,500 接受盒步，进程 wall 分别为 9.539174 s 和 8.387790 s；两份保存区间逐字节相同，独立扫描的 tube 与 endpoint 均在安全带内。它们共享驱动且各只有一个样本，不构成独立证明或稳定速度排名；保存 observer 的 endpoint 相对同小步 tube 有最多 `5.55e-15` 的微小越界，见 [两作者新结果](evidence/results/archcomp26_20261001/author_dp_less_v1/SUMMARY.md)。新版 P3 先用独立有向区间 NN residual 跑完整盒尝试，第 2 期首小步 225 盒全因收缩失败；更紧的有向仿射 ReLU residual 后已推进 60/100 小步，但从小步48起性质未决，并出现局部收缩失败。两者都保留为失败/部分前缀，见 [P3 诊断](ARCHCOMP26_DP_P3_DIRECTED_DIAGNOSTIC_20261001.md)。
- **DP more / 正确独立 controller**：新原生 225 盒跑到第16个控制周期，进程 wall 715.978257 s，45盒 `UNKNOWN` 后停；共同安全前缀到 `T=0.3`，无 T=0.4 结果，见 [原生摘要](evidence/results/archcomp26_20261001/native_dp_more_full20_001/SUMMARY.md) 和 [状态图](evidence/results/archcomp26_20261001/native_dp_more_full20_001/plots/dp_more_native_theta1dot_status.png)。新 Huan/Xiangru 用相同 225 盒和固定2026 more ONNX，均在第18期后由共享 checker 输出 `Unsafe.` 并早停；进程 wall 11.685967 s、11.724106 s，保存区间逐字节相同。独立区间扫描显示 `T≤0.3` 安全，步61初见未决，步72有15盒的整个保存 `θ̇₁` tube 位于安全带下方；这尚不是独立端到端浮点 NN 反例证明。见 [两作者早停摘要](evidence/results/archcomp26_20261001/author_dp_more_v1/SUMMARY.md)。

截至上节时，主要缺口是论文 QUAD 的其它三方法、新版其余 13 个非 VCAS 配置的逐项合同和四方运行、P3 严格控制器与更完整的可验证数值资格，以及正式多次计时。Single Pendulum 的第三时钟初值、Airplane 离散控制更新先后、QUAD reach-and-remain 检查的原始参与者执行源码仍未取得，已向用户具体询问；不由现有论文/README猜造执行语义。当时阶段性 DOCX/PDF 正在制作；旧哈希绑定 16×4 矩阵仍保留为恢复前快照，新数据以独立 [无哈希 attempt 索引](evidence/archcomp26_nohash_attempts_20261001.json)登记。

## 后续新证据（12:30 UTC 左右）

- **论文 QUAD / Xiangru**：另一独立新目录 `.../quad_paper_xiangru_v1/full50_001` 的 1,024 盒 × 1,000 步全部接受，进程 wall 108.018 s、driver 103.686 s，最终 16 维 hull 与新 Huan 论文合同 JSON 相同。两者共用作者数学核心和 P2 parity，不能充当彼此独立的正确性证明；各一次 wall 不能建立稳定排名。见 [Xiangru 摘要](evidence/results/archcomp26_20261001/quad_paper_xiangru_v1/SUMMARY.md)。原生论文 QUAD 在独立 `.../native_quad_paper_full50_001` 运行中；1,024 盒一期预检有 20,480 条完整范围记录，未重复启动旧六小时作业。
- **Single Pendulum / 论文两物理态 profile**：新 [配置](../benchmarks/archcomp26/configs/single_pendulum_paper_two_state.yaml) 对完整单盒、`0.05×20`、全时 `t∈[0.5,1]` 的 `0≤x1≤1` 设闭时间窗。Huan 与 Xiangru 各有一份完整 100/100 小步、`broken=0` 的新运行，wall 5.431 s / 5.529 s；两份保存区间完全相同，窗口内 `x1` tube union `[0.5663832766836561,0.9925703395905962]`。见 [SP 两方摘要](evidence/results/archcomp26_20261001/single_pendulum_prep_001/SUMMARY.md)。此 profile 把第三状态仅作辅助时钟；若要称为 2026 参与者 MATLAB 三态复现，仍缺第三初值/执行源码。新 native 仍在单独预检/正式运行。
- **TORA remain / 原生**：逐字节确认选定的固定 2026 ONNX 与服务器已有旧同名模型相同；原生隔离 build 的 12 盒一期 plumbing smoke 完成。新的 12 盒 × 20 期、200 小步正式单次原生运行在 8.339151 s 完成，20 RPC、2,400 条唯一范围记录，作者 checker `VERIFIED`。独立 [范围扫描](evidence/results/archcomp26_20261001/native_tora_remain_full20_001/SUMMARY.md) 所见全时四态 tube 都在 `[-2,2]` 内，端点也未越出同小步 tube。四方 [合同](ARCHCOMP26_TORA_REMAIN_CONTRACT_20261001.md) 已冻结；截至本节写作时 Huan/Xiangru 尚待结果，其后续终态见下方补记。此单次时间不进入四方排名。

无哈希 [16×4 工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md) 单独从新 attempt 索引重算，反映已尝试/运行中/早停/失败；旧冻结矩阵没有被改写。阶段性 [DOCX](evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_STAGE_REPORT_DRAFT_20261001.docx) 与 [PDF](evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_STAGE_REPORT_DRAFT_20261001.pdf) 已生成，仍是**未完成全套实验的阶段稿**；下一版须纳入本节新增运行。

## 21:55 本地补记：QUAD P3、SP 与 TORA 两方

- **2026 论文方程 QUAD / P3**：独立新目录 `quad_paper_p3_nohash_v1/full50_001` 完成 1,024 盒 × 1,000 小步，1,024,000/1,024,000 盒步接受，50 次控制刷新、`broken=0`，外层进程 wall 1,357.555 s；`T=5` 的 `x3` endpoint union 为 `[0.9584732146312492,1.0256989633476477]`，在目标 `[0.94,1.06]` 内。驱动打印 `VERIFIED`，但明确记录 `end_to_end_strict_certificate=false`，不能把该诊断提升为独立端到端浮点 NNCS 证书。其终点宽 0.06722574871639841，比新 Huan/Xiangru P2 parity 的 0.04774181696742297 宽；各只有一次样本、设备与路线不同，不给速度排名。见[原始证据摘要](evidence/results/archcomp26_20261001/quad_paper_p3_nohash_v1/SUMMARY.md)。该时点论文方程 native 全程作业仍在运行；最终结果见下方 10 月 2 日补记。
- **Single Pendulum / 原生补完**：两物理态加辅助时钟的原生全程已完成 1 盒 × 100 小步，wall 4.726027 s；Huan/Xiangru 分别为 5.430762 / 5.528731 s。三方在性质要求的闭时间窗 `t∈[0.5,1]` 保存的 `x1` tube 均位于 `[0,1]`。见[原生摘要](evidence/results/archcomp26_20261001/native_sp_two_state_full20_001/SUMMARY.md)和[两方摘要](evidence/results/archcomp26_20261001/single_pendulum_prep_001/SUMMARY.md)。
- **TORA remain / 作者两方新尝试**：原生已全程完成并由作者 checker 报 `VERIFIED`，进程 wall 8.339151 s、2,400/2,400 条范围，见[原生摘要](evidence/results/archcomp26_20261001/native_tora_remain_full20_001/SUMMARY.md)。Huan/Xiangru 各观察到 200 小步，但只接受 2357/2400 盒步，作者 checker 均报 `Unknown.`；首次保存 tube 出安全带在步185，首次拒绝在步190。两方全部盒接受且保存 tube 安全的共同前缀仅 `t≤18.4`。其失败进程 wall 8.371523 / 8.270158 s 不与原生完整时间排名，区间出带不是独立实际轨迹反例。见[两方摘要](evidence/results/archcomp26_20261001/author_tora_remain_v1/SUMMARY.md)。
- 新[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)当前由 35 条 attempt 重算：10 个方法格有单次完整时域数值记录、1 格运行中、5 格早停或未完成。它独立于旧哈希绑定矩阵，完成格也尚不构成四方同资源、多次测量或端到端证明。阶段 [DOCX](evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_STAGE_REPORT_DRAFT_20261001.docx) 与 [PDF](evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_STAGE_REPORT_DRAFT_20261001.pdf) 已按这批事实更新并逐页检查，仍非最终完整报告。

## 继续推进：ACC 与未冻结的来源差异

- **ACC participant-order / 修复 VAR 尾项的原生 Flow\***：固定 2026 ONNX 与保存的旧同名文件直接逐字节相同；保存的两份参与者源码均明确第 5 个 NN 特征为 `v_lead-v_ego`，论文没有定义相对速度符号，故新[合同](ARCHCOMP26_ACC_PARTICIPANT_CONTRACT_20261001.md)明确命名这一输入顺序。独立一期 smoke 后，隔离修复版完成全初盒 B=1、50 控制期、T=5，50 RPC、50 条范围、作者 checker `VERIFIED`，外层 wall **7.930 s**。独立读取保存 tube 后，安全半空间 `x_lead-x_ego-1.4*v_ego-10` 的保守下界最小 **16.25475375921129>0**。见[新原生摘要](evidence/results/archcomp26_20261001/acc_native_var_tail_full50_001/SUMMARY.md)。这只证明本次数值记录与作者 checker 一致，不提升旧 VAR 修复针对性单步测试为端到端控制器证明；Huan/Xiangru 新入口仍在准备。
- [NAV 初次来源审计](ARCHCOMP26_NAV_PAPER_SOURCE_CONTRACT_20261001.md)发现论文状态文字 `(x,y,θ,ν)` 与固定官方 MATLAB 的 `(x,y,ν,θ)` 顺序冲突；当时新 NAV 数值任务尚未启动。后来找到控制器作者训练与闭环执行源码，后续合同及两次新短程尝试见[补充审计](ARCHCOMP26_NAV_AUTHOR_EXECUTION_CONTRACT_20261002.md)。
- [TORA reach 来源冲突](ARCHCOMP26_TORA_REACH_SOURCE_CONFLICT_20261001.md)涉及 sigmoid 最后一层和 plant 缩放；[无哈希网络预检](ARCHCOMP26_TORA_REACH_CONTROLLER_PREFLIGHT_NOHASH_20261001.md)已证实官方 2026 `.mat` 与旧 ONNX 的逐层参数、激活一致，并提供需显式选择缩放的构造器。这是选择前的来源审计截点；用户现已选官方 2026 模型与 `u=11f`，后续数值结果见文末。
- [Unicycle 来源冲突](ARCHCOMP26_UNICYCLE_SOURCE_CONFLICT_20261001.md)涉及扰动加入 yaw/speed 的位置与 `w` 是否随时间变化；固定 2026 ONNX 与保存版直接字节相同，但不能替代 plant 选择。[Airplane 入口审计](ARCHCOMP26_AIRPLANE_2026_ENTRY_AUDIT.md)亦已核出完整初盒和固定 ONNX，而离散更新顺序在官方目录没有执行程序；没有把旧单点连续任务或自拟 Euler 先后伪称官方离散复现。

当前[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)已收录 37 条新尝试：11 格单次完整数值时域、1 格仍运行的原生论文 QUAD、5 格早停、1 格失败。所有完整格的稳定多次比较和端到端浮点 NNCS 证明仍未完成。

## 后续补记：ACC 三方、Attitude checker 与 QUAD 长作业

- **ACC participant-order 三方全程**：Huan/Xiangru 各自在独立新目录完成一期 smoke 和全 50 期，均 50/50 小步接受，50 次安全评价，保存 tube 的半空间保守下界最小都是 `16.16503470756421>0`；两方全程区间记录和安全记录逐字节一致。进程 wall 分别为 7.259717 s、7.311630 s。与原生修复 VAR 尾项的 7.930 s 共同形成这个明确命名合同的三方单次完整数值记录；独立证书及稳定多次计时尚缺。见[三方证据汇总](evidence/results/archcomp26_20261001/ACC_PARTICIPANT_PROFILE_20261001.md)。我方 P3 适配正在独立评估，未填完整结果。
- **Attitude Control 来源错误**：旧 native/Xiangru checker 把官方危险盒的 `x4≥-0.7` 误写成 `x4≥-0.4`，而上界仍为 `x4≤-0.6`，所检危险集为空。旧 `VERIFIED` 不可沿用为 2026 正确规格结果；新隔离修正与预检继续进行。
- **论文方程 QUAD / native 当时状态**：只读确认独立新作业运行到第 26 个控制期；未启动第二份或重启旧六小时实验。该时点完整 T=5 结果尚未产生，最终结果见下方 10 月 2 日补记。

最新[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)由 **41 条**新尝试重算：**13 格**有单次完整数值时域记录、1 格运行中、5 格早停、1 格失败，其余 44 格尚无尝试。上述先前“35 条”“37 条”是各阶段快照，不表示当前计数。

## ACC 四方数值记录闭合

- 我方 working-P3 ACC 最小适配在新独立目录先完成一期 smoke，再完成全单盒 50 期、50/50 小步接受，50 次控制特征构造/注入和 50 次安全事件。单次外层 wall **7.933729 s**，保存全时 tube 盒的安全半空间保守下界最小 **16.43485856971698>0**；见[P3 证据摘要](evidence/results/archcomp26_20261001/acc_p3_full50_001/SUMMARY.md)与[四方汇总](evidence/results/archcomp26_20261001/ACC_PARTICIPANT_PROFILE_20261001.md)。该入口复用 working-P3 数值核心与已有 CUDA 库，不能继承 QUAD 的全批次资格；用户要求的无摘要路径被保持。
- ACC participant-order 是第一个四方法均完成完整时域的实例。其“参与者顺序”指 `v_rel=v_lead-v_ego`，论文未定义符号；四条单次 wall 仅作各自进程事实，不能据此排名。四方独立端到端浮点 NNCS 证明、统一多次计时和每维完整宽度对比仍待完成。

当前[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)由 **43 条**新尝试重算：**14 格**单次完整、1 格运行中、5 格早停、1 格失败，43 格尚未尝试。此前 41 条是上一阶段快照。

## Attitude 官方 unsafe 修正后的原生全程

旧 native/Xiangru 源码把 `x4` 的危险盒下界 `-0.7` 误写成 `-0.4`，结合上界 `-0.6` 形成空集；详见[合同和逐式来源](ARCHCOMP26_ATTITUDE_CONTROL_CONTRACT_20261001.md)。在独立新构建和运行目录修正这条约束后，原生先完成一期 smoke，再完成全部 30 控制期、60 个 ODE 小段、30 RPC，进程 wall **6.281498 s**。作者 checker 给 `VERIFIED`；保存的 60 个六维 tube 盒均与官方真正的 unsafe 盒不相交，`x4` 全时上界为 `-0.710800050132425`，低于危险盒下界 `-0.7`。见[保存盒扫描](evidence/results/archcomp26_20261001/native_attitude_avoid_full30_001/SCAN.json)和[危险盒独立判交](evidence/results/archcomp26_20261001/native_attitude_avoid_full30_001/BOX_UNSAFE_AUDIT.json)。旧 `VERIFIED` 不纳入新成绩；新数值结果也不声称独立端到端浮点 NN 证明。Huan/Xiangru 的修正 profile 正在另立目录预检。

目前[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)由 **45 条**新尝试重算：**15 格**单次完整、1 格运行中、5 格早停、1 格失败，42 格尚未尝试。此前 43 条是上一阶段快照。

新增[已保存流管绘图汇总](evidence/results/archcomp26_20261001/plots/nohash_saved_20261001/SUMMARY.md)：[ACC 四方 t–安全距离裕量 PNG](evidence/results/archcomp26_20261001/plots/nohash_saved_20261001/acc_four_method_t_safe_distance_margin_tube.png)直接从 50 期六态 tube 盒计算半空间保守像，并画安全阈值 0；[论文方程 QUAD P3 t–x3 PNG](evidence/results/archcomp26_20261001/plots/nohash_saved_20261001/quad_paper_p3_1024x1000_t_x3_pooled_tube.png)使用 1,000 行每步 1,024 盒 union，目标仅是 T=5 终点。两者另有 PDF、MATLAB `.m`、逐步几何 JSON 和不含内容摘要的出图收据；MATLAB 脚本仍未实跑，QUAD 图不冒充逐盒或 octagon。

[Docking 与 Balancing/CartPole 来源审计](ARCHCOMP26_DOCKING_BALANCING_SOURCE_CONTRACT_20261001.md)已完成。Docking 的固定官方附带说明可冻结四原态输入、两力输出、全初盒、1 秒×40 周期及非线性全时安全不等式；缺四方可执行入口与该不等式的流管 checker。Balancing 的论文五特征控制器与固定四输入 ONNX/MATLAB 不一致，闭时间窗 `[8,10]` 与规格 `8<t≤10` 也不同；可另立仓库四原态 profile，但不能冒充论文五特征主合同。这两格暂不填数值成绩。

修正危险集的 **Attitude Huan/Xiangru** 也在各自独立 smoke 后完成 30/30 期、60/60 接受小段。进程 wall 分别为 7.094217 / 6.933376 s；两者新保存范围直接逐字节相同，各 60 个六维 tube 盒与官方真正的 unsafe 盒不相交，修正后的作者 checker 未输出 Unsafe/Unknown。原生显式 `VERIFIED` 和独立扫描如上；旧空集 checker 成绩不升格。两作者 [保存结果](evidence/results/archcomp26_20261001/author_attitude_avoid_v1/huan_full30_001/payload/RESULT.json) 仍缺 P3 同合同方法格及独立浮点 NN 证明。

最新[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)由 **49 条**新尝试重算：**17 格**单次完整、1 格运行中、5 格早停、1 格失败，40 格尚未尝试。先前 45 条是阶段快照。

用于 GitHub 审阅的[本轮证据镜像](evidence/results/README.md)已开始放入分支，包含已完成的小运行收据、保存范围、图、CSV 和阶段报告；官方第三方控制器模型不重复发布，原生 QUAD 长作业的完整二进制范围仍保留服务器原路径。新短任务产生的后续收据继续同步，不借用历史数据填新结果。

**Single Pendulum 两物理态 / P3** 在新独立 smoke 后完成全初盒一分区的 20 控制期、100/100 接受小步，外层 wall **6.583305 s**。闭窗 `[0.5,1]` 的 50 次作者安全事件全部满足，保存 tube 的 `x1` union 为 `[0.5645452654370386,0.9932406822927875]`；见[新 P3 证据摘要](evidence/results/archcomp26_20261001/single_pendulum_two_state_p3_full20_001/SUMMARY.md)。P3/Huan/Xiangru/native 现构成该**两物理态加辅助时钟**合同的四方单次完整数值记录，不能挪作缺第三初值的官方三态 MATLAB 复现或稳态速度排名。

最新[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)由 **51 条**新尝试重算：**18 格**单次完整、1 格运行中、5 格早停、1 格失败，39 格尚未尝试。先前 49 条是阶段快照。

**Attitude ours/P3 修正 unsafe** 在独立新目录完成一期 smoke 后，全程完成 30 控制期、60/60 接受小段，外层 wall **13.017036 s**；修正后的作者 checker 未输出 Unsafe/Unknown，保存的 60 个六维 tube 盒逐段与官方非空 unsafe 盒分离。见[P3 保存结果](evidence/results/archcomp26_20261001/p3_attitude_avoid_v1/full30_001/payload/RESULT.json)及[合同与旧 checker 错误](ARCHCOMP26_ATTITUDE_CONTROL_CONTRACT_20261001.md)。P3/Huan/Xiangru/native 因此均有该修正合同的单次完整数值记录；旧空集 `VERIFIED` 未借用，四方仍缺独立浮点 NN 证明和重复计时。

最新[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)由 **53 条**新尝试重算：**19 格**单次完整、1 格运行中、5 格早停、1 格失败，38 格尚未尝试。先前 51 条是阶段快照。

## 10 月 2 日凌晨补记：ACC 重复计时与 TORA P3

- **ACC 24 次独立进程 campaign**：在同一明确命名的 participant-order 合同下，四种方法各有 1 次本 campaign 首轮新进程和 5 次后续新进程；轮次轮换，共 24/24 全时域完整、已保存 tube 安全。统一外层 `Popen` 前至子进程回收 wall，GPU 2、CPU 10–13 顺序执行。后五次中位数分别为原生 **7.736304 s**、Huan **8.237208 s**、Xiangru **7.987532 s**、P3 **8.740510 s**；各自逐次 min/max、首轮时间、50 步/RPC/控制器调用和 24 个独立原始目录在[正式 campaign 摘要](evidence/results/archcomp26_20261001/acc_fourway_campaign_001/SUMMARY.md)。首轮不是重启主机后的冷机；GPU 1 同时运行 QUAD 长作业，且缺独立端到端浮点 NN 证明，这组时间只作当前资源条件下的描述，暂不宣称稳定速度排名。最初各方法的一次单独运行仍保留，没有混入这 24 个样本。
- **TORA remain / P3**：独立一期 smoke 后，新全程尝试完成 12 初盒 × 20 控制期 × 10 小步 = 2400/2400 盒步接受，外层 wall **10.411001 s**。对保存的 2400 条范围独立扫描，四态全时 tube 均在 `[-2,2]^4`，无拒绝、首越界或终点超出同段 tube；作者 checker 无失败输出。末端四态绝对 `lo/hi/width` 与原始收据见[新 P3 摘要](evidence/results/archcomp26_20261001/p3_tora_remain_v1/SUMMARY.md)。原生也完整，Huan/Xiangru 仍是 `t≤18.4` 共同安全前缀；[四方绝对区间和空值表](evidence/results/archcomp26_20261001/tora_remain_fourway_common_prefix_20261002/SUMMARY.md)区分了这两类结局。失败进程 wall 不和完整时间排名；P3 的安全数值记录不等于独立端到端 NNCS 浮点证书。

当时[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)由 **79 条**新尝试重算：**20 格**记录完整数值时域、1 格论文方程 QUAD 原生仍在运行、5 格早停、1 格失败，37 格尚未尝试。先前 53 条是上一阶段快照。

TORA remain 的[四方 t–x4 保存流管图](evidence/results/archcomp26_20261001/tora_remain_fourway_common_prefix_20261002/plots/SUMMARY.md)已从新运行原始范围逐步重画，提供 PNG/PDF/MATLAB `.m` 与几何 JSON。原生/P3 展示完整 200 步；Huan/Xiangru 只画 184 步合格前缀，后续标为未知，不把幸存盒延伸到 T=20。

## 10 月 2 日接续：Airplane 全初盒阻断与 Docking 四方全程

- **Airplane continuous**：按固定官方完整 12 态初盒、20×0.1 s 合同另立入口，旧点初盒结果没有复用。order-6 Huan 一周期预检在任何 ODE 小步之前，19 变量六阶单项式配对构建期间 RSS 超过 54,006,540 KiB，252.247 s 后只终止该子进程组；order-3 诊断 profile 的 Huan/Xiangru 均在首个 0.01 s 小步拒绝唯一全初盒，分别 6.785/6.985 s、0 个接受步。未启动完整 T=2，也没有性质结论；内部首步拒绝子类未记录。原始记录和两种阶数差异见[三次尝试摘要](evidence/results/archcomp26_20261001/airplane_continuous_order3_fullbox_20261002/SUMMARY.md)。
- **Airplane continuous P3**：六输出严格注入的 CPU 预检通过且未初始化 GPU；两个新独立完整初盒、一周期 smoke 均未得到可接受流管。默认工作 P3/验证 P4 在首步前的单项式表编码遇到 `9^20 >= 2^63` 限制，外层 wall 13.758 s、0 个数值段；另立严格同阶 `solution_order` P3 profile 越过该入口阻断，但唯一全盒在首个 0.01 s 小步 `accepted=false`，外层 wall 12.500 s、0/10 接受。首拒立即停止，未启动 T=2 full，也没有性质结论；内部拒绝子类没有记录。两次原始记录和方法区别见[P3 摘要](evidence/results/archcomp26_20261001/AIRPLANE_P3_FULLBOX_SMOKES_20261002.md)及[入口审计](ARCHCOMP26_AIRPLANE_P3_NATIVE_ENTRY_AUDIT_20261002.md)。
- **Airplane continuous Flow* native**：同一官方单完整初盒另立三次一期 smoke，依次为历史 order 6 / remainder `[-0.01,0.01]`、order 3 / 同余项、order 3 / 加宽余项 `[-1,1]`。三次均各完成一次 12→6 控制 RPC，但首个 0.01 s 段 `UNCOMPLETED_SAFE`（状态 4）、0/10 接受，外层原始状态各为 `failed/exit2`，wall 4.877421/3.925546/3.925656 s；性质无任何已接受 tube 可检查，日志末尾的 `UNKNOWN` 不可改写为安全或真实反例。加宽余项只是参数诊断，不是完整官方结果。原始记录与来源级首拒说明见[原生三次摘要](evidence/results/archcomp26_20261001/native_airplane_fullbox_smokes_20261002/SUMMARY.md)。
- **Docking P3/Huan/Xiangru**：官方单完整初盒、40×1 s、每期 10 个数值小步，三个新 full run 各 400/400 接受，外层 wall 分别 17.411771/12.698839/12.668129 s。三方对非线性全时约束的 checker 均为 `Unknown.`；首段保守安全裕量 `q` 上界约 +0.016083，不能当作实际轨迹越界反例或安全证书。先前 Huan 一次 smoke 因公共模块导入路径缺失在数值推进前失败，保留该记录，随后新 smoke 成功；七次尝试及末端四态绝对区间见[三方原始证据摘要](evidence/results/archcomp26_20261001/DOCKING_FULLBOX_3METHODS_SUMMARY.md)。原生新入口由独立工作继续。

当前[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)登记 **89 条**新尝试：64 个方法单元中 **23 格**有完整数值时域、1 格原生论文方程 QUAD 运行中、5 格早停、3 格失败、32 格尚未尝试。这里的“完整”只指数值覆盖，Docking 性质仍未知；上述 79 条/20 格是上一阶段快照。

**Docking / 原生第四方法** 在独立一期 smoke 和全时域新目录中运行。full 走完 40/40 期、400/400 段、40 RPC，外层 wall 9.139416 s；原始 `RESULT` 仍是 `failed/exit 2`，因为原生性质 checker 返回 `UNKNOWN`。逐段保守径向安全裕量 `q` 的 400 个区间都跨零，首段上界 +0.016082965、末段最大上界 +8.954347227；区间相交不等于实际轨迹反例。40 次 CROWN 调用的上下仿射斜率逐元素相同，保存范围与 400 条性质记录独立核对；见[原生 Docking 摘要](evidence/results/archcomp26_20261001/native_docking_full40_001/SUMMARY.md)。四方法现在都有完整数值时域，但没有一家证明全时安全或独立浮点 NNCS 证书。

**旧作者 QUAD / P3 观测器成对消融** 使用旧 x2/x4/x5 方程、同一 1,024 盒与 40 小步，两个隔离无 JIT/无内容摘要新进程各接受 40,960/40,960 盒步；终步 1,024×12 的 tube、endpoint、status 数组和控制记录直接逐值一致。每步 observer 开/关的 driver 时间为 53.554101/53.323556 s，只有一对样本，不能外推全程加速；首次 on 入口被 TorchScript 守卫在 0 步前拦截也保留。见[成对原始记录](evidence/results/huan_quad_stage_a_40_20261001/observer_pair_v1/SUMMARY.md)与[速度模式报告](HUAN_QUAD_SPEED_AND_MODES.md)。当前证据不支持为省去短前缀约 0.23 s 而删除绘图所需的逐步 observer。

加入 Airplane P3 前的矩阵快照对应 **91 条**新尝试：**24 格**完整数值时域、1 格原生论文方程 QUAD 仍运行、5 格早停、3 格失败、31 格尚未尝试。此“完成”包括原生 Docking 的数值完成/性质未知/外层 exit 2，三者在 attempt 记录中分开表示。上面的 89 条/23 格为更早快照。

加入两个 Airplane P3 新 smoke 后，当时矩阵为 **93 条**新尝试：64 格中 **24 格**完整数值时域、1 格论文方程 QUAD 原生仍运行、5 格早停、**4 格失败**、**30 格未尝试**。P3 连续 Airplane 格归类为入口/首小步失败，旧单点成绩没有填入。

再加入三次 Airplane Flow* native 完整初盒新 smoke 后，最新[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)为 **96 条**新尝试：64 格中 **24 格**完整数值时域、1 格论文方程 QUAD 原生仍运行、5 格早停、**5 格失败**、**29 格未尝试**。Airplane continuous 四方方法单元均已有实际失败尝试，但均没有完整 T=2 数值流管或全时性质结论；上面的 93 条为此前快照。

## 10 月 2 日补记：NAV 作者执行顺序与首周期 smoke

- [NAV 作者执行合同补充审计](ARCHCOMP26_NAV_AUTHOR_EXECUTION_CONTRACT_20261002.md)从控制器作者训练脚本和 CORA 闭环例确认：网络原序输入为 `[x,y,speed,heading]`，输出为 `[speed_rate,heading_rate]`；固定官方方程同此顺序。论文文字写 `[x,y,heading,speed]`，隐藏层也写 `64/64`，而固定官方 ONNX 为 `64/32`。新尝试按固定官方 ONNX 与作者可执行顺序显式命名；仍不声称作者仓库 Git LFS 模型与官方模型的内容身份已比对。
- 新的 Huan standard 和 robust 各取历史分区台账**首盒**，用固定官方 point/set ONNX 跑首个 `0.2 s` 周期，GPU 2 / CPU 10–13、无 TCP RPC 端口。两个独立 ID [`nav_author_standard_huan_smoke1_001`](evidence/results/archcomp26_20261001/nav_author_standard_huan_smoke1_001/RESULT.json) 和 [`nav_author_robust_huan_smoke1_001`](evidence/results/archcomp26_20261001/nav_author_robust_huan_smoke1_001/RESULT.json) 均接受 20/20 个 `0.01 s` 小步，保存 tube 与官方障碍盒分离，进程 wall 分别为 4.142439 s 与 4.417727 s。各自独立读回的区间检查与配置在[运行档案](evidence/results/archcomp26_20261001/nav_author_smoke_v1/README.md)；**未覆盖 640/25 全部分块、余下 29 期及 `t=6` 终点目标**。[旧 Huan 全程合同逐项比对](ARCHCOMP26_NAV_AUTHOR_EXECUTION_CONTRACT_20261002.md#旧-huan-全程记录与当前可执行合同逐项比对)确认两条历史结果在模型内容、方程、初集分区、采样、性质与数值设置上同合同，故不重复启动全盒作业；历史证据和本轮新 attempt 索引仍分开。
- [Unicycle 执行门](ARCHCOMP26_UNICYCLE_EXECUTION_GATE_20261002.md)明确了阻断：2026 论文只在速度导数加入 `w`，固定官方 MATLAB 右端没有 `w`，旧四方在朝向和速度两导数均加入同一个常值 `w`。论文未定 `w` 是轨迹常值还是可随时间变化，也未冻结“10 秒内到达”的检查语义；没有开启新 Unicycle 数值尝试，旧成绩不移入新格。

当前[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)为 **98 条**新尝试：24 格完整数值时域、1 格运行中、5 格早停、5 格失败、**2 格仅短前缀**、27 格未尝试。两条 NAV smoke 只使 Huan 两格变成“仅短前缀”，没有增加完整结果数。

## 10 月 2 日补记：离散 Airplane 与 TORA reach 启动门

- [Airplane discrete 执行门](ARCHCOMP26_AIRPLANE_DISCRETE_EXECUTION_GATE_20261002.md)核对论文的同步 forward Euler、12 态完整初盒、20 次 `0.1 s` 转移和 `k=0..20` 共 21 个安全检查索引。四方现有 Airplane 程序都推进连续 ODE，不能充当离散成绩。`x_k` 先取控制、再同步 Euler 可作为**新命名的四方共同比较合同**；若要求忠实复现 2026 参与者离散提交，仍缺离散转移及控制应用顺序的执行源码或等价权威记录。此审计未启动作业。
- [TORA 两个 reach 执行门](ARCHCOMP26_TORA_REACH_EXECUTION_GATE_20261002.md)直接逐元素核对了固定官方 `.txt` 和 `.mat` 的各 961 个控制器参数，并沿用此前 `.mat` 与旧 ONNX 的数值预检。模型文件已到位，问题是合同：sigmoid 的论文/官方/旧三种激活与缩放组合不同，tanh 的论文与官方激活不同；“5 秒内到达”也要确定用终点包含作为充分证据，还是要求完整时间窗判定。未选定前不把旧四方 500 步运行填为新版成绩。
- [Balancing/CartPole 执行门](ARCHCOMP26_BALANCING_EXECUTION_GATE_20261002.md)核对论文五特征 `f(x1,x2,sin x3,cos x3,x4)` 与固定仓库 ONNX 四原态输入 `[x1,x2,x3,x4]` 的冲突。论文忠实主合同仍缺五输入控制器或作者权威的五到四映射；固定仓库四输入版本已明确命名 `balancing-fixed-repo-raw4`，可单列准备四方入口与全初盒、10 秒性质窗检查，不能用旧一秒小盒诊断替代。

## 10 月 2 日补记：Balancing 四输入仓库 profile 的实际尝试

- 新隔离 [Huan raw4 入口](../tools/archcomp26_balancing_raw4_huan_nohash.py)先完成 CPU 模型预检：固定四输入、一输出 Gemm/Tanh 图可读取，未初始化 CUDA。第一次一期 smoke 的性质窗 `[8,10]` 与其 0.02 秒前缀不相交，驱动在 ODE 前明确失败；第二次一期 smoke 关闭不适用的性质检查，全初盒 4/4 小步接受。其原始收据把零次检查误标 `VERIFIED_BY_SAVED_BOX_CHECKS`，原文件保留，并在[审计勘误](evidence/results/archcomp26_20261001/balancing_fixed_raw4_huan/AUDIT.md)明确改读为**性质不适用**。
- 同一新命名 profile 的 500 期独立作业在第 99 个 ODE 内步拒绝，前 98/2000 计划小步接受，最后有效流管到 `t=0.49 s`；外层 wall 8.218304 s。`[8,10]` 性质窗未进入，0 次检查，结论 **UNKNOWN/incomplete**。原始记录没有内部拒绝状态码，不从流管变宽推测原因。论文五特征控制器仍缺，不把这次早停当作论文主合同成绩或完整性能样本。
- 加入上述 Balancing 三条尝试时，[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)有 **101 条**新尝试：64 格中 24 格完整数值时域、1 格原生论文方程 QUAD 运行中、6 格早停、5 格失败、2 格仅短前缀、26 格未尝试；两条旧 NAV 同合同全程证据仍单列在索引外。

## 10 月 2 日补记：Double Pendulum less 的第四方完整数值运行

- P3 的新[有向仿射分区审计](ARCHCOMP26_DP_P3_PARTITION_DIAGNOSTIC_20261002.md)保持官方四态 ODE、225 初盒、20 期、100 小步、全时 `[-1.7,2]^4` 安全带及原 P3 数值阶数。控制器 residual 使用原全局仿射图 `T`；每维二分的 16 个闭子盒完整覆盖输入盒，首次性质未决/收缩失败分别推迟到小步 75/84，仍未完成。随后独立的每维四分 256 子盒尝试完成 **100/100 小步、22,500/22,500 盒步接受、broken=0**；逐条保存 tube 的独立读回确认 22,500 个四态盒均在安全带内，最小盒裕量 **+0.093919848**。外层 wall **74.274085 s** 包含逐盒观察写盘，不能与其它三方单次 wall 做稳定排名。
- 这使 DP less 在原生、Huan、Xiangru 和 P3 四方都有同初集、同 1 秒时域的完整数值记录。先前 P3 失败尝试保持原始状态，不被成功尝试覆盖；控制器 residual 的局部有向包含论证不等于 P3 七变量 plant、倒数/三角、余项、端点与性质组合的端到端浮点证明，`end_to_end_strict_certificate=false` 保留。当前[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)为 **103 条**尝试、**25 格**完整数值时域、1 格运行中、6 格早停、4 格失败、2 格仅短前缀、26 格未尝试。
- 已从四方**保存的全量数值数据**生成[DP less 四方绝对宽度汇总](evidence/results/archcomp26_20261001/dp_less_fourway_split4_20261002/SUMMARY.md)、16 行[终点统计 CSV](evidence/results/archcomp26_20261001/dp_less_fourway_split4_20261002/endpoint_stats.csv)、1,600 行[逐步 tube union CSV](evidence/results/archcomp26_20261001/dp_less_fourway_split4_20261002/tube_union.csv)及[四态叠加图](evidence/results/archcomp26_20261001/dp_less_fourway_split4_20261002/fourway_tube_union.png)。Huan/Xiangru 的保存区间直接相同；P3 在第四物理态的 `T=1` endpoint union 宽为 0.799090015，原生为 1.016584896，两作者为 1.109787875；每盒平均宽度的次序并不一致，不能用单一 union 宽度宣布优劣。PDF/MATLAB `.m` 同目录，脚本未在 MATLAB/Octave 实跑。

## 10 月 2 日补记：DP more 首周期与独立数值点诊断

- 固定官方 more-robust ONNX、官方四态 ODE 和完整 `5×5×3×3=225` 初盒的新[P3 首控制期诊断](evidence/results/archcomp26_20261001/dp_more_p3_firstperiod_interval_20261002_001/SUMMARY.md)只运行 `T=0.02`，4 个 0.005 s 小步、900/900 盒步均数值接受。作者驱动打印 `Step 0`、`Unknown.`；原始 JSONL 独立重扫的四步安全盒数依次为 `[225,225,181,120]`，第 3 步首次有 44 盒的 `θ̇₁` 上界超过 1.5，第 4 步增至 105 盒。首例 lane 6 保存区间 `[0.9912745512752368,1.5106057605641459]` 只表示区间无法证明安全，非实际轨迹反例。未启动 P3 `T=0.4` 全程，也未重复旧 Huan/Xiangru/原生区间作业。当前矩阵由 **104 条**尝试重算：25 格完整、1 格运行中、6 格早停、4 格失败、**3 格仅短前缀**、25 格未尝试。
- 先前固定官方 ONNX 的[名义角点 `(1.3)^4` 数值重放](evidence/results/archcomp26_20261001/dp_more_point_candidate_20261002/SUMMARY.md)显示越界，但 binary64 `1.3` 比精确初盒上界 `13/10` 大约 `4.44e-17`，故单凭这条记录不能称为精确初盒内的候选反例。为修正初点资格，另做唯一[严格内点 `(1.299)^4` 数值重放](evidence/results/archcomp26_20261001/dp_more_interior_point_candidate_20261002/SUMMARY.md)：每 0.02 s 以 float32 ONNX 控制后持值、官方 ODE 积分；DOP853 估计 `θ̇₁=-1.5` 下穿在 `t=0.3248652630`，`t=0.325` 网格值为 `-1.5004299419`，`t=0.36` 值为 `-1.6071729076`，位于已保存 Huan lane 224 endpoint 区间内。两档 RK4 与 DOP853 的保存期末差小于 `1.05e-14`。这是**合同内初点的数值候选**，一致性不是严格误差界，控制器浮点语义与 ODE 逐期包络仍需验证，不称为严格反例，也不作为第五种 flowpipe 方法计入四方索引。
- 已将 103 条截点的可编辑 [DOCX](evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_STAGE_REPORT_DRAFT_20261001.docx)和同名 [PDF](evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_STAGE_REPORT_DRAFT_20261001.pdf) 更新到 DP less 四方完整数值记录和六幅图；Word 导出的 17 页 PDF 已逐页目视检查，避免替代排版器漏中文。新 DP more 诊断晚于该固定阶段快照，先记在本进展和 Markdown 完整草稿中。

## 10 月 2 日补记：Airplane 离散命名约定的有界入口

- 对已明确命名、但尚非参与者官方执行身份的 `paper-Euler-controller-first` 约定，使用固定 12→6 ONNX、完整 12 维单盒和 12 态同步 Euler 映射做了[首次一步 CPU 区间诊断](evidence/results/archcomp26_20261001/airplane_discrete_paper_euler_interval_smoke1_001/AUDIT.md)。`k=1` 的 `sy` 包络在安全带内；三个角度因通用外舍入各比精确 `[0,1]` 多一个极小浮点格，原始 checker 为 `Unknown`。固定 MATLAB 动力学的 `p=q=r=0` 使首步角度导数代数上确为零；保留原始结果后，[第二次新诊断](evidence/results/archcomp26_20261001/airplane_discrete_paper_euler_exactzero_prefix_001/AUDIT.md)只对该精确零恒等式特殊处理。`k=1` 四个安全坐标均在闭带，随后 `k=2` 包络大幅越带，首次实质性 `Unknown` 即停；安全端点前缀只有 **1/20 次转移**。两次均是四方方法之外的 CPU 区间入口，不给 Airplane 离散主表四格填成绩、证书或完整运行时间。参与者真实的离散控制取样与转移顺序仍缺。

## 10 月 2 日补记：TORA reach-sigmoid 官方文件 profile 的一期结果

- 为避免把旧 `u=22(f−0.5)` 成绩误认成 2026 官方 `u=11f`，另建唯一[TORA 官方文件一期 Huan 诊断](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_firstperiod_diag/SUMMARY.md)。固定 `.mat` 四层 sigmoid 经显式构造后，ONNX 图内 `Mul(11)+Add(0)`，外部 `output_scale=1/output_offset=0`，完整单初盒的一个 0.5 秒控制期、50/50 ODE 小步全部接受。原始 `START/RESULT`、50 条范围、日志与独立有限/有序扫描均保存；`T=0.5` 末端 `x1∈[-0.8850144075,-0.8573825037]`、`x2∈[-0.0066841850,0.0219011293]`，外层单次 wall `5.345883 s`。没有判定 5 秒内到达目标，没有启动旧 500 步或 tanh，且没有速度排名资格。由于论文和官方网络激活冲突，仍须用户选主合同；本结果按官方文件 profile 单列。纳入此短前缀后工作矩阵为 **105 条**尝试、25 格完整、1 格运行中、6 格早停、4 格失败、**4 格仅短前缀**、24 格未尝试。
- 原有论文方程 QUAD native 长作业在 2026-10-02 00:26:29 UTC 的只读复核仍为 `RESULT_PENDING`：日志在第 49 个控制期的 Flow* 计算，原始 `ranges.bin` 已写前 **980/1000** 小步，native 与 RPC 进程仍存活且 CPU 时间持续增长。未启动副本、重启或干预；没有第 50 期的完整时间、终点宽度或性质结论。

## 10 月 2 日补记：原生论文方程 QUAD 自然完成及四方数值边界

- 原有 `native_quad_paper_full50_001` 在 **2026-10-02 00:30:15 UTC** 自然完成，原始 `RESULT.json` 为 `completed`、exit 0、未超时，外层 wall **47058.886571 s**（单次）；native 日志打印 `VERIFIED`、内部 `time cost: 46918.890000 s`。这是上条 00:26 仍运行作业的**同一原始运行**，没有重启旧六小时实验、增加副本或重复登记 attempt。原始 START/RESULT、日志、50 次 RPC、构建源码及[只读范围扫描](evidence/results/archcomp26_20261001/native_quad_paper_full50_001/SUMMARY.md)已保存到分支的小证据包，417,792,000 字节 `ranges.bin` 保持在原服务器路径。
- 原始范围按“50 期→1024 盒→期内 20 小步”精确覆盖 **1,024,000/1,024,000** 条唯一 `(盒,小步)` 记录；有限性、区间顺序、endpoint 含于对应 tube、盒/步/步长检查均为零异常。`T=5` 的 `x3` endpoint union `[0.965771839016746,1.0167484756616678]`、宽 0.0509766366449218，位于目标 `[0.94,1.06]`。最后一小步 tube 宽 0.05097738677735586，与 endpoint 分列；全时 `x3` tube `[−0.40817151558335585,1.452252535346294]` 不能冒充终点。原生源码 checker 只用 `fp_end_of_time.isInTarget` 终点判定；step 775–1000 保存 tube 连续落目标带只是额外数值诊断，**论文 reach-and-remain 时间语义与独立端到端浮点 NNCS 证明仍未闭合**。
- 2026 论文方程 QUAD 的 P3、Huan、Xiangru、原生现各有一次完整 1024×1000 数值记录；四方 `T=5` 的十二态 endpoint 绝对下/上界与并集宽见[48 行 CSV](evidence/archcomp26_quad_paper_endpoint_4methods_20261002.csv)。四方 `x3` endpoint 宽依次为 **0.067225749 / 0.047741817 / 0.047741817 / 0.050976637**；运行路线、资源和观察器不同，不据单次时间或宽度做速度/正确性排名。原生原作业的索引行已就地更新，工作矩阵仍为 **105 条尝试**、64 个方法单元中 **26 格完整数值时域、0 格运行中、6 格早停、4 格失败、4 格仅短前缀、24 格未尝试**。
- 新[四方 QUAD t–x3 图与数据包](evidence/results/archcomp26_20261001/quad_paper_fourway_saved_20261002/SUMMARY.md)以原生和 P3 各 1,000 小步的完整初盒保存 tube 画曲线；Huan/Xiangru 没有保存逐步范围，只以 T=5 endpoint 标记。目标 `[0.94,1.06]` 只作为终点目标，MATLAB `.m` 已导出但没有 MATLAB/Octave 实跑证据，也不能由轴对齐投影恢复 Flow* octagon。
- [阶段 DOCX](evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_STAGE_REPORT_DRAFT_20261001.docx)及 [Word 导出 PDF](evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_STAGE_REPORT_DRAFT_20261001.pdf)已从旧 103/25 截点刷新为 **105 条尝试、26 格完整**，纳入 QUAD 四方图，并把 DP more P3 与 TORA reach-sigmoid Huan 如实标为短前缀。PDF 共 17 页，中文、表格、六幅图及最终修改页已目视核查；它仍是阶段草稿，不是完整四方最终成绩。细节见[交付说明](evidence/results/archcomp26_20261001/stage_report/README.md)。

## 10 月 2 日补记：旧 NAV Xiangru 同合同复查与 Balancing P3 一期

- 对旧 NAV standard/robust Xiangru 原始文件作[逐字段合同及保存范围复查](ARCHCOMP26_NAV_XIANGRU_HISTORICAL_CONTRACT_AUDIT_20261002.md)：固定官方 point/set ONNX 与旧使用文件直接逐字节相同，物理态原序、方程、640/25 完整初盒、30×0.2 秒及障碍/终点性质均符合明确命名的“固定官方模型 + 作者可执行顺序”合同。原始两条各完成 600/600 小步、384,000/15,000 盒步接受；独立重读的所有保存区间有限、有序，tube 与障碍分离且末端入目标。旧 Huan/Xiangru 对应范围直接逐字节相同，共享数值核心，不是两份独立证明；本轮没有重复启动旧 NAV Xiangru 作业，也没有把历史运行增记为新 attempt。
- 固定仓库四输入 `balancing-fixed-repo-raw4` 的 [P3 完整初盒一期诊断](evidence/results/archcomp26_20261001/balancing_fixed_raw4_p3_smoke1_001/SUMMARY.md)在 GPU 3 接受 4/4 个 0.005 秒小步、无首拒，外层进程 wall 5.230873 秒。独立读回四条 tube/endpoint 均有限、有序且逐步包含；第 8–10 秒性质窗尚未进入，`property_checks=0` 表示不适用。此短前缀不覆盖 500 期，也不解决论文五特征控制器与固定四输入 ONNX 的身份差异，不能参与四方速度排名。
- NAV standard 原生 Flow* 的新隔离[首盒首周期检查](evidence/results/archcomp26_20261001/nav_author_standard_native_smoke1_001/INDEPENDENT_SAVED_RANGE_SCAN.json)完成 1×20/20 小步，exit 0、外层 wall 3.875128 秒。其后 640 盒×30 期、3600 秒上限的独立原生作业 `nav_author_standard_native_full30_001` **自然完成**，原始 RESULT 为 `completed`、exit 0、未超时，外层 wall **1478.865919 秒**，作者日志打印 `COMPLETED_PERIODS 30/30` 和 `VERIFIED`；旧两条 300 秒超时作业没有重启或覆写。原始保存范围 58,368,000 字节留在服务器，[本地原始小收据与独立扫描](evidence/results/archcomp26_20261001/nav_author_standard_native_full30_001/SUMMARY.md)核对 640×600=384,000/384,000 条有限、有序、唯一的盒步，endpoint 均含于对应 tube。保存 x/y tube 与闭障碍 `[1,2]²` 相交 0 条，末端 640 个盒均在闭目标 `[-0.5,0.5]²` 内；`T=6` 的 x/y endpoint union 分别为 `[-0.09175765699647812,-0.004467673970412854]`、`[0.08836682447943034,0.35454798657692854]`，宽分别约 0.087289983 / 0.266181162。作者标签和保存区间检查不是独立端到端浮点 NNCS 证明；单次时间也不参与稳定速度排名。
- [旧 NAV P3/原生证据审计](ARCHCOMP26_NAV_P3_NATIVE_HISTORICAL_AUDIT_20261002.md)核对旧 `ours` standard/robust 两条 640/25×600 完整历史数值运行及旧原生 robust 25×600；后者的 15,000 条保存范围全量重读，未见非有限、倒序、端点越 tube、障碍相交或末端不入目标。旧 `ours` 使用 `engine_linear_leaf_v2`，不能冒称当前工作 P3 的新运行；这些历史记录不进入本轮新 attempt 格。
- 两个 NAV 变体的[四方保存范围 x/y 图、宽度图及 MATLAB 脚本](evidence/results/archcomp26_20261001/nav_fourway_historical_vs_new_20261002/README.md)从八份原始 `ranges.bin` 只读归约：standard 每法 640×600 条、robust 每法 25×600 条；两个 4,800 行紧凑 CSV 另保留逐步 pooled tube/endpoint 与分块 mean/max 宽度。standard 三条 GPU 曲线为 9 月历史结果，原生为 10 月 2 日新作业；robust 四条均为历史结果。x/y 单轴投影不单独证明联合 `[1,2]²` 避障；逐盒二维扫描另确认障碍相交 0、末端目标外 0。图不支持速度排名或当前 P3 完成声明。
- 为定位 Airplane continuous 完整初盒 P3 旧 smoke 的首步拒绝，另立唯一[带内部回调的首拒诊断](evidence/results/archcomp26_20261001/airplane_p3_first_reject_trace_smoke1_001/SUMMARY.md)：首个 `h=0.01` 小步的 `initial_self_map` 和四次 `recentered_self_map` 均在物理 `x,y,z` 三分量上超出各自 `[-0.01,0.01]` 余项初猜，其余分量未触发该拒绝；记录值有限，最终状态码 1=`FAILED_CONTRACTION`，0/10 步接受、范围文件为空。这只确定该**带回调 eager 诊断路径**的数值自映射失败，不代表真实轨迹违反性质或未经回调作业的逐位内部原因；回调影响耗时，不用 12.250824 秒做速度比较。
- Airplane 另立只修改 `x,y,z` 余项初猜为 `[-0.1,0.1]` 的[有界数值 profile](evidence/results/archcomp26_20261001/airplane_p3_xyz_rem0p1_trace_smoke1_001/SUMMARY.md)：首步自映射与细化在 trace 中通过、引擎返回 `accepted=true`，但随后独立保存检查抛 `invalid accepted interval at 1`，原始 observations/ranges 均为空、**0 个可用保存步**。原检查把非有限、区间次序和 endpoint 含于 tube 合并为一个异常，且未保存原始 bounds；现有证据无法进一步区分具体谓词，不猜测，也没有全时性质结论。这次带回调外层 wall 11.700555 秒不作排名。
- NAV standard 的[当前工作 P3 首盒首周期诊断](evidence/results/archcomp26_20261001/nav_author_standard_working_p3_smoke1_001/INDEPENDENT_SAVED_RANGE_AUDIT.json)使用固定官方 point ONNX 和作者可执行顺序，在 GPU 1 接受 20/20 小步，独立重读 20 条保存区间的有限性、顺序、endpoint 包含、首盒含于首步 tube 和障碍避让均无异常，外层 wall 4.497773 秒。随后新隔离的[全部 640 盒首周期门检](evidence/results/archcomp26_20261001/nav_author_standard_working_p3_fullgrid_firstperiod_001/INDEPENDENT_SAVED_RANGE_AUDIT.json)也自然完成 20/20 小步、**12,800/12,800** 盒步接受，外层 wall 5.243638 秒；独立重扫原始 12,800 条记录，有限性、区间顺序、endpoint 在同小步 tube 内、640 初盒各含于首步 tube、二维障碍相交均为零异常。两个 P3 运行都只覆盖 30 期中的首期，**未检查 `t=6` 目标**；旧 `ours` 全程不能代替这个新 P3 profile 的全时结果。
- 此时[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)按 **112 条**新尝试重算：64 个方法单元中 **27 格完整数值时域**、0 格运行中、6 格早停、4 格失败、6 格仅短前缀、21 格未尝试。旧同合同 NAV 全程证据只在历史审计中陈述，不当作新的尝试。
- 该截点的 NAV P3 一期入口强制 `steps=1` 并关闭 6 秒终点目标，因此不能把它原样延成 full30；当时仍需在新隔离入口恢复 30 期与最终目标 checker、添加 640×600 盒步全接受及保存区间独立完成 guard。后续新 run 的完成结果见下条。
- NAV 原四方法[同轴叠加图](evidence/results/archcomp26_20261001/nav_fourway_historical_vs_new_20261002/README.md)中四色几乎重合，是已核对的数值结果相似，不是缺数据；旧 Huan 与旧 Xiangru 的逐步界完全相同。主图保留同轴比较，宽度差放大图与按方法分列图作为附图；它们只读取既有 CSV，未为重画图启动实验。
- 为精确定位 Airplane continuous 的放宽 xyz 余项 profile 在观察器中的首步拒绝，又立[保留原始四界的单次诊断](evidence/results/archcomp26_20261001/airplane_p3_xyz_rem0p1_observer_smoke1_001/SUMMARY.md)。与前次同一模型/配置，首步引擎报告 `accepted=true`，但原保存 guard 前的 48 个界虽全部有限且 endpoint 自身有序，`phi` 的 endpoint 下界低于 tube 下界 1 个 binary64 相邻值，`y/z/theta/psi` 的 endpoint 上界高于 tube 上界 1/2/1/1 个相邻值。原 guard 如实失败、0/10 个可用保存步，外层 `failed/exit1`；这不等于真实轨迹反例或安全证明。离线向外合并 observer 界不会更改 plant 传播，但候选 tube 的三个角度分量仍超安全阈值 1，因此没有为这个观察器差异再开全时运行。
- NAV standard 当前 working P3 的新隔离[640 盒×30 期完整作业](evidence/results/archcomp26_20261001/nav_author_standard_working_p3_full30_001/SUMMARY.md)已恢复原终点 checker，另加 384,000 盒步全接受与首拒停 guard，在 GPU1/CPU6–9、11 GiB 分配上限、3600 秒监督时限下自然完成。原始 RESULT 为 `completed/exit0`、600/600 小步及 384,000/384,000 盒步接受、单次外层 wall **28.999885 秒**，作者 checker 打印 `VERIFIED`；独立重读 384,000 条保存范围未发现非有限、逆序、endpoint 越 tube、首步初盒不含于 tube、全时二维障碍相交或终点目标遗漏。`T=6` endpoint 并集 x `[-0.09058780124346533,-0.005241272342260411]`、y `[0.08845848499512873,0.3542626232863495]`；原始 52,224,000 字节范围仍留服务器，新 1,200 行 x/y 投影 CSV 已保存供同轴叠加图使用。旧 `engine_linear_leaf_v2` NAV 作业没有重启，也不等于此新 P3；这一单次时间不构成稳定四方速度排名或独立端到端浮点 NNCS 证明。
- 该截点[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)曾为 **114 条**新尝试、64 个方法单元：**28 格完整数值时域**、0 格运行中、6 格早停、4 格失败、5 格仅短前缀、21 格未尝试。上面的 112/27 记录保留为先前截点。
- NAV robust 当前 working P3 另立[25 盒×30 期全程作业](evidence/results/archcomp26_20261001/nav_author_robust_working_p3_full30_001/SUMMARY.md)，固定官方 set ONNX 与作者可执行状态顺序，600/600 小步、15,000/15,000 盒步全接受，原始 `completed/exit0`、未超时、作者 `VERIFIED`，单次外层 wall **18.703898 秒**。本地镜像原始 2,040,000 字节范围并二次重读：15,000 条身份完整、有限、有序、endpoint 在同小步 tube 内、25 个初盒含于首步 tube，保存二维障碍相交及末端目标遗漏均为 0。`T=6` x/y endpoint 并集分别为 `[0.10709245590899041,0.1982240186155469]`、`[-0.06337556479191862,-0.04827867885822918]`；1,200 行小 CSV 单列，未把旧 `ours` 充作当前 P3。单次时间与作者标签仍无稳定四方速度排名或独立端到端浮点证明。
- Balancing 固定仓库四输入 raw4 的新[当前 P3 完整时域尝试](evidence/results/archcomp26_20261001/balancing_fixed_raw4_p3_full500_001/SUMMARY.md)在第 **87/2000** 个 ODE 小步首次 `FAILED_CONTRACTION`，只接受前 86 小步至 `t=0.43`，独立重读 87 条原始台账，其中 86 条已接受四态 tube/endpoint 有限、有序、endpoint 被 tube 包含；失败步无新有效 tube。外层 `failed/exit2`、wall 13.806936 秒，8–10 秒性质窗 0/400 检查，性质 Unknown/incomplete；旧 Huan raw4 在第 99 步早停记录原样保留。两者都不代表论文五特征控制器或全程成绩。
- DP more 当前 P3 的新[225 盒×20 期全时域尝试](evidence/results/archcomp26_20261001/dp_more_p3_full20_interval_20261002_001/SUMMARY.md)在累计第 **9/80** 小步 225 盒全部 `FAILED_CONTRACTION`，前 8 小步共 1,800/18,000 盒步接受，仅到 `t=0.04`。独立扫描 7,200 个已接受四态 tube/endpoint 数值均有限、有序并互相包含；安全盒从第 3 小步起与边界相交，性质 Unknown，不能叫实际反例。原始外层 `failed/exit1`、wall 16.616363 秒、无终局作者性质标签；首拒即停，无 `T=0.4` 宽度或全程计时。
- 该截点[工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)曾为 **117 条**新尝试、64 格中 **29 格完整数值时域、8 格早停、4 格失败、3 格仅短前缀、20 格未尝试、0 格运行中**；后续更新见文末。未尝试格与外部材料/合同选择仍按下表逐项保留，不用早停前缀补完整格。

### 仍需外部材料或明确选择的字段

| 实例 | 具体缺件或选择 | 在此之前可独立推进的工作 |
| --- | --- | --- |
| QUAD reach-and-remain | 2026 参与者实际运行的全时间窗 checker 源码或同等权威执行记录；目前四方完整数值时域及原生终点检查已保存，仍不足以固定全时间窗语义。 | 保留四方终点和原生逐步 tube；所见 step 775–1000 连续入带仅作数值诊断。 |
| Single Pendulum 官方三态 | 第三状态的初值和三态 MATLAB 闭环执行入口；现有四方仅是明确命名的两物理态加辅助时钟合同。 | 保留两态 profile 的四方结果及图。 |
| Airplane discrete | 参与者离散转移与控制更新次序的源码或权威记录。 | 可单列实施已明示的 `paper-Euler-controller-first` 四方比较约定。 |
| Balancing 论文五特征 | 五输入控制器文件，或作者明确的五特征到固定四输入模型映射/执行源码。 | 可单列实施 `balancing-fixed-repo-raw4`。 |
| TORA reach 两变体 | 用户已选官方 2026 两个模型与 `u=11f`；若要给出参与者的完整时间窗性质标签，仍需实际 checker 或同等权威执行语义。 | sigmoid 四方新全程、tanh 当前 P3 新全程及三方历史同合同全程均已保存；各方目标坐标终点数值盒入目标只作“5 秒内到达”的充分观察，不补写未运行的 checker 标签。 |
| NAV standard/robust | 已执行的官方 point/set ONNX 加作者可执行状态顺序已冻结；若要宣称作者另一 Git LFS 仓库模型与官方文件二进制同一，仍缺该 LFS 模型原件或权威来源记录。 | 当前新全程和旧同合同复用按已核文件身份分别列示，LFS 身份未定不重启旧实验。 |

这些来源冲突均未用旧实验或自拟默认值填补新版主表。其余未完成格仍需各方法实现、接受/失败记录、宽度和重复计时；这些是当前工程工作，不当作待用户提供的文件。

## 10 月 2 日：QUAD 已保存区间的持续入带复核

只读复算已有论文方程 QUAD 的 1,000 步 `x3` 数值包络；没有启动或重启求解作业。原生全部 1,024 盒 pooled tube 自第 775 步（名义 `t=3.87 s`）至终点连续落入 `[0.94,1.06]`；当前 P3 的保存 tube 与 endpoint 合并包络自第 791 步（`t=3.95 s`）至终点连续入带。两法的紧前一步上界仍超带。P3 有 174 个小步的保存 `x3` endpoint 超出同小步 tube 至多 `8.88e-16`，故诊断明确合并两种保存界；后 210 步中的 9 个末位差异仍不改变入带结论。详见[只读重算和数值收据](evidence/results/archcomp26_20261001/quad_paper_fourway_saved_20261002/SUMMARY.md)。Huan/Xiangru 无逐步坐标包络，不能从终点推断持续入带；上述观察不替代缺失的参与者全时间窗 checker 或独立浮点 NNCS 证明。

同样只读重算 Docking 四方各 400 步保存范围，整理 `sx,sy,vx,vy` 的 `T=40` endpoint 上下界、绝对宽度与全时 tube union 至[逐维 CSV](evidence/results/archcomp26_20261001/docking_fourway_saved_widths_20261002.csv)，并在报告中并排呈现。四方法仍是完整数值时域、非线性性质 `UNKNOWN`；没有为宽度表重启实验。

论文方程 QUAD 的四方 `x1`–`x12` 终点绝对宽度也已从现存驱动终态与原生最后一步保存 endpoint 整理到[逐态 CSV](evidence/results/archcomp26_20261001/quad_paper_fourway_saved_20261002/terminal_12states_fourway.csv)。同文件另列 P3 最后一步观察器 endpoint；它与 P3 驱动终态不是同一保存对象，在 `x5` 一个界上最大相差约 `5.69e-6`，不可混列成第五方法或不注明来源的宽度值。

**Single Pendulum 两物理态 24 次新进程轮换计时**：四方法各 1 次本 campaign 首轮和 5 次后续独立进程，24/24 全部完成 1 盒×100 小步，独立扫描 2,400 条有限有序 tube/endpoint，闭性质窗全部保存 `x1` tube 落在 `[0,1]`；原生合计 120/120 RPC 与 HTTP200。统一使用物理 GPU2、CPU10–13 顺序运行。后五次进程 wall 中位数为 native **4.729023 s**、Huan **5.380765 s**、Xiangru **5.480807 s**、P3 **6.183896 s**；逐次和 min/max 见[原始及独立审计](evidence/results/archcomp26_20261001/sp_two_state_fourway_campaign_20261002_001/SUMMARY.md)。这只属于明确命名的两物理态加辅助时钟合同，不是缺第三状态执行材料的官方三态复现，也不是稳定速度排名或独立端到端 NNCS 证书。24 条逐次 attempt 已在无哈希索引登记，旧作业未重启。

**Balancing 固定仓库 raw4 / Xiangru**：新入口一期 4/4 接受；独立全 500 期配置在第 99/2000 ODE 小步 `FAILED_CONTRACTION` 首拒，98 步接受至 `t=0.49 s`，8–10 秒性质窗 0/400 检查，结论 Unknown/incomplete。独立扫描还发现已接受步的保存 endpoint 在 189 处超过同一步 tube 最多 `2.49e-14`；这些观察不归因于未证明的实现原因，引擎接受与严格 tube 包含分别陈述。见[独立证据](evidence/results/archcomp26_20261001/balancing_fixed_raw4_xiangru_20261002/SUMMARY.md)。Huan 原有 raw4 作业也在第 99 步首拒，但它是另一原始运行；论文五特征控制器仍未取得。Xiangru 的一期与完整尝试已入索引，旧实验未重启。

**TORA reach 两变体执行门**：官方四层 sigmoid、图内 `u=11f` 的新 Huan 全初盒运行在另立 run ID 完成 10 期/500 小步，独立扫描 500 条保存区间有限、有序且 endpoint 含于 tube；`T=5` 的 `x1=[0.1346564224,0.1604743267]`、`x2=[-0.8762353073,-0.8507148906]` 数值 endpoint 在目标内，进程 wall 11.285899 s。性质 checker 未执行；此前误选 Python 环境的 0 步失败原始收据另保留。见[全程与失败两条证据](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_full500_huan_002/SUMMARY.md)。tanh 官方 ReLU³/tanh、`u=11f` 新一期 50/50 与旧 Huan 全程前 50 步保存的 800 个四态 tube/endpoint 边界数逐值相同；旧四方同合同已有 500 步原始记录，故未重复启动 tanh 全程，见[复用审计](evidence/results/archcomp26_20261001/tora_reach_tanh_official2026_mat_u11_firstperiod_diag_001/SUMMARY.md)。论文合并文字的激活与官方两模型冲突；用户已选官方 2026 两个模型和 `u=11f` 为新版主表合同。具名官方 profile 与旧合同证据分列，保存区间不构成端到端证明。三条新尝试已入索引。

**Attitude 修正危险集四方 24 次独立新进程**：四方法各 1 次首轮、5 次后续运行，24/24 均完成全初盒、30 控制期、60 小段；同一物理 GPU2/CPU10–13 顺序执行。独立审计扫描 1,440 条六态保存范围，全部有限有序、首步覆盖初盒、端点包含于同段 tube，所有 tube 与修正后的闭危险盒分离；原生合计 180 次 RPC/HTTP200。后五次完整进程 wall 中位数：native 6.181135 s、Huan 6.834693 s、Xiangru 6.884601 s、P3 12.952000 s。见[原始运行、逐次表和独立审计](evidence/results/archcomp26_20261001/attitude_corrected_fourway_campaign_20261002_001/SUMMARY.md)。P3 只改启动器的 GPU 设备守卫，原源码另存；24 条逐次结果已入新索引。时长是具名合同下的描述统计，保存盒和作者 checker 仍不构成独立端到端证明或稳定四方排名。

**Balancing 固定仓库 raw4 / Flow* native**：首次隔离入口在首步前解析失败（exit 139、0 条范围），原始失败与构建源码保留；修正表达式文本后新一期 4/4 小段完成。新 500 期请求只保存 83/2000 小段，前 20 期完整、第 21 期前三段后返回 status 4 `UNCOMPLETED_SAFE`，约至 `t=0.415 s`，外层 exit 2、wall 5.731613 s。独立扫描 83 条五坐标范围均有限、有序且 endpoint 包含于同段 tube；8–10 秒性质窗 0/400 检查，Unknown/incomplete。见[三条原始 run、源码和审计](evidence/results/archcomp26_20261001/native_balancing_raw4_20261002/SUMMARY.md)。四方法 raw4 完整请求都已首拒即停；论文五特征控制器仍缺，四条前缀时间不能用于全程速度排名。

此截点的[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)曾为 **175 条**新尝试：64 格中 **31 格完整数值时域、10 格早停、4 格失败、3 格仅短前缀、16 格未尝试、0 格运行中**。后续更新见下方；各格状态只描述新尝试，历史全程证据继续按原资格单列复用。

**TORA reach-sigmoid / Xiangru 官方主合同**：新一期门检 50/50 步接受；另立 500/500 步全程运行 wall 11.335332 s。独立扫描 500 条四态保存 tube/endpoint 有限、有序且逐步包含，`T=5` 的 `x1=[0.1346564224,0.1604743267]`、`x2=[-0.8762353073,-0.8507148906]` 数值 endpoint 在目标内；和 Huan 对应保存区间逐字节相同，共享数值驱动不能据此得到独立正确性证明。性质 checker 未运行。见[一期](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_xiangru_firstperiod_001/SUMMARY.md)与[全程原始记录和扫描](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_xiangru_full500_001/SUMMARY.md)。两条新尝试已入索引；P3 与原生的随后结果见下条。

**TORA reach-sigmoid / P3 和 Flow* native 官方主合同**：P3 首周期完整初盒 50/50 通过后，独立新作业完成 10 期/500 小步；外层 13.741791 s，保存 `T=5` 的 `x1=[0.1345258159,0.1606047516]`、`x2=[-0.8764374243,-0.8505126711]` 在目标内。原生另立官方图内 `u=11f`、RPC 外部 `1/0` 的入口，CPU/一期门检后完成 10 期、500/500 小段及 10 RPC，外层 8.991666 s，原生终点 checker `VERIFIED`；保存终点 `x1=[0.1345319317,0.1605988723]`、`x2=[-0.8763648123,-0.8505857061]` 也在目标内。两方法的全部保存范围经独立有限/有序/逐段包含扫描。见[P3 原始全程](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_p3_full500_001/SUMMARY.md)、[原生全程与构建](evidence/results/archcomp26_20261001/native_tora_reach_sigmoid_u11_full10_001/SUMMARY.md)和[四方终点宽度](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_terminal_fourway.csv)。P3/Huan/Xiangru 未执行性质 checker，原生 `VERIFIED` 仅按终点 checker 口径；单次耗时不排名，四方均无独立端到端浮点证明。旧 sigmoid 缩放实验没有重启。

**Unicycle / 用户选定论文速度扰动主合同**：四方法均新建隔离入口并走完 50 期、500/500 小步到 `T=10`，原始保存区间独立扫描通过。Huan/Xiangru 终点物理盒相同，`x3` 与 `x4` 未全入目标，作者 verdict `UNKNOWN`；不能据此断言整个 10 秒时间窗未到达。P3 与 Flow* native 的终点全盒包含于目标，后者作者终点 checker 打印 `VERIFIED`；这只支持选定的数值终点充分条件，不是端到端浮点证明。P3 原始方法元数据错误写“首周期”，原件保留并由完整 500 步、START/RESULT 与[更正审计](evidence/results/archcomp26_20261001/unicycle_paper_speed_w_constant_v1/p3_full50_001/SCOPE_ERRATUM.json)明确纠正。原生旧 x2 初盒欠包通过实际赋值后系数的 8/8 有理数覆盖审计修复。四方法每维终点上下界/宽度见[CSV](evidence/results/archcomp26_20261001/unicycle_paper_speed_w_constant_v1/terminal_fourway.csv)，全部原始 run 与方法差异见[执行门](ARCHCOMP26_UNICYCLE_EXECUTION_GATE_20261002.md)。P3 工作阶数 3/验证阶数 4，作者 GPU 使用阶数 2，单次 wall 不作排名。

当时[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)由 **187 条新尝试**重算：64 格中 **37 格完整数值时域、10 格早停、4 格失败、3 格仅短前缀、10 格未尝试、0 格运行中**。当时 10 个未尝试格中，Airplane discrete 四格缺参与者权威离散转移/控制更新顺序；其余六格连同三个短前缀格均有已核同合同历史全程证据（NAV 五格、TORA reach-tanh 四格），保留原资格且没有重跑。此为已过阶段截点；链接中的矩阵随后按新作业更新。完整数值时域、作者 checker、保存区间性质观察、独立端到端证明与稳定速度排名在报告中分口径陈述。

## 10 月 2 日接续：历史覆盖、数值拒绝和真实八方向出图

- 新[16×4 覆盖附表](evidence/archcomp26_coverage_overlay_20261002.md)当时从 187 条 attempt 的工作矩阵与九条旧原始 `result.json`、合同审计重算：64 格为 **37 格本轮新完整数值时域、9 格已核同合同历史完整数值时域、14 格无完整数值时域、4 格 Airplane discrete 合同阻塞**。此为已过阶段截点，附表文件已在后续重算；九条历史原始记录不计入新尝试，不重启旧实验。NAV 作者仓库 Git LFS 控制器身份与旧 P3 代际仍有具名限制。附表同时有[CSV](evidence/archcomp26_coverage_overlay_20261002.csv)、[JSON](evidence/archcomp26_coverage_overlay_20261002.json)及只读重算脚本。
- TORA remain 两作者的[只读数值拒绝审计](ARCHCOMP26_TORA_REMAIN_NUMERIC_STOP_AUDIT_20261002.md)核对每方 200/200 观察步、2357/2400 盒步接受；第 185 步首次保存 tube 出 `[-2,2]^4`，第 190 步初盒 2 首次被数值拒绝，最后各仅 6/12 盒接受。共享驱动的 `safe_unknown` 只决定末尾 `Unknown.`，不使积分停止；关闭性质检查不能恢复全盒 `T=20`，故未启动这种重复实验。原记录未保存具体数值失败子类。
- [真实 Taylor 模型八方向导出器](../src/torch_tm_flowpipe/tm_octagon_nohash.py)在接受段上评估 `x,y,x+y,x−y` 的关联 TM 范围，分别保存 tube/endpoint 的八方向支持界，并从保存几何重画 MATLAB `.m`、PDF、PNG。[三步 plant-only 谐振子证据](evidence/results/archcomp26_20261001/tm_octagon_harmonic_smoke_20261002_001/SUMMARY.md)到 `T=0.15` 全接受；独立有限解析采样/半平面检查 108 项通过，第三步 endpoint 显示多边形面积为坐标盒的 0.8488663。它不是新 NNCS 四方实例；现有 QUAD 保存范围无 TM 相关性，不能伪装成 octagon 或为绘图重启旧长实验。MATLAB/Octave 尚未执行。
- 新[两页固定阶段 DOCX](evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_STATUS_20261002.docx)与[Word 导出 PDF](evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_STATUS_20261002.pdf)纳入当时 37+9+14+4 覆盖、主要结果与 14 格实测停止点；两页用 Word 和 PDFium 逐页查看，中文字形、表格与页边界正常。先前 105/26 的 17 页阶段稿仍是原截点；新的完整逐实例报告仍在[Markdown 草稿](ARCHCOMP26_FINAL_REPORT_DRAFT.md)中，目标尚未完成。

## 10 月 2 日后续：数值步长变体、流式方向界与逐实例报告

- TORA remain 原 `h=0.1` Huan/Xiangru 的第 190 步数值盒拒绝保留在固定主表。另立[仅减半 ODE 小步的 `h=0.05` 数值 profile](evidence/results/archcomp26_20261001/author_tora_remain_h005_probe_20261002/SUMMARY.md)：官方 ONNX、12 初盒、动力学、1 秒控制周期、20 秒时域和全时安全性质不变；两方各 400/400 小步、4800/4800 盒步全部接受，独立扫描的全部保存 tube 在 `[-2,2]^4` 内，作者 checker 无输出。单次 wall 各约 10.899 秒。此结果单列为参数诊断，在当时不改写 187 条固定主表尝试、37+9+14+4 覆盖或同设置排名；也无独立端到端 NNCS 证书。
- CPU Taylor 模型求解器新增接受段回调，配套[八方向 JSONL 流式观察器](../src/torch_tm_flowpipe/tm_octagon_nohash.py)在每个已接受段直接保存 tube 与用于传播的 `final_tm` 终点方向界；绘图时可从保存支持界重建几何，不重算 ODE。[隔离三步 plant-only 短例](evidence/results/archcomp26_20261001/tm_octagon_stream_harmonic_smoke_20261002_001/SUMMARY.md)3/3 到 `T=0.15`，带观察器与无观察器对照的所有方向界逐项相等，108 个有限解析采样在界内。此接口仍只接 CPU scalar TM；求解器仍保留 segment 列表，尚非大规模 GPU/native 常驻内存优化。旧 QUAD 轴盒数据不能追算 octagon，MATLAB 脚本尚未在 MATLAB/Octave 执行。
- [逐实例 Markdown 阶段报告](ARCHCOMP26_FINAL_REPORT_DRAFT.md)已把恢复前写“未启动”的 16 节旧模板替换为当时 187 条尝试和 9 格历史复用的阶段覆盖、配置、来源、性质及具体缺项。它仍不是完整最终成绩：Airplane continuous、DP more、Balancing、固定 `h=0.1` TORA remain 等有未完成格，Airplane discrete 及官方模型身份还有具名缺件，四方稳定排名和独立端到端证明尚无资格。Huan 速度报告另补五次旧作者方程运行的完整进程/内部计时，不与新论文 QUAD 合同混比。
- [TORA reach-sigmoid 官方 `u=11f` 四方保存流管图](evidence/results/archcomp26_20261001/tora_reach_sigmoid_u11_fourway_saved_figure_20261002/SUMMARY.md)从既有 Huan、Xiangru、P3、原生各 500 条原始四态区间直接画 `x1/x2` 整步 tube，并在同图比较 T=5 终点上下界相对 Huan 的 `10⁻⁴` 量级差异；四方终点盒均入目标。Huan/Xiangru 保存曲线重合。前三方性质 checker 未运行，原生仅有作者终点 `VERIFIED`；图中轴盒无相关方向支持，不能据图得独立端到端浮点证明或速度排名。没有重跑旧实验；MATLAB 脚本生成但未执行。
- [Unicycle 四方全部保存终点时间窗审计](evidence/results/archcomp26_20261001/unicycle_paper_speed_w_constant_v1/SAVED_ENDPOINT_WINDOW_AUDIT_20261002.md)只读检查各 500 个已保存步的完整初盒四态 endpoint 与整步 tube：P3 最早在 `t=9.72 s`、原生最早在 `t=9.80 s` 有全盒 endpoint 入目标；对应最早整步 tube 分别为第 487 步 `[9.72,9.74] s` 与第 490 步 `[9.78,9.80] s`。Huan/Xiangru 的 500 个已保存 endpoint 无四态同时全入目标，性质仍 `UNKNOWN`，不能据此断言整个时间窗不可达。原生二进制不带逐步接受字段；此为保存轴盒的数值充分条件审计，没有新实验或独立端到端证书。
- [TORA reach-tanh 同合同历史四方保存流管图](evidence/results/archcomp26_20261001/tora_reach_tanh_historical_fourway_saved_20261002/SUMMARY.md)只读复用 2026 官方 ReLU³/tanh、`u=11f` 四份旧完整记录：每方 500 条四态 `0.01 s` 保存范围，2,000 条原始记录有限、有序、endpoint 均含于同段 tube；并与旧保存轨迹的 12,000 行区间值逐项对照。四态 `T=5` 绝对终点和宽度已导出 CSV，四方 `x1/x2` 终点数值盒都在目标内。Huan/Xiangru 全部保存界重合；旧我方明确是 `engine_linear_leaf_v2`，不是当前 working P3。前三方 GPU result 无性质 verdict，原生作者终点 checker `VERIFIED`；同图右列放大细小差异，不能提升为独立端到端证书或稳定速度排名。旧实验未重启，当时 187 条新尝试和 37+9+14+4 覆盖不变。

## 10 月 2 日再接续：TORA 同步长补充对照及原生方向界门禁

- 固定主表的 TORA remain `h=0.1` 作者两法早停原样保留。另立的[`h=0.05` 四方补充对照](evidence/results/archcomp26_20261001/tora_remain_h005_fourway_saved_20261002/SUMMARY.md)现由 P3、Huan、Xiangru、原生各一条完整 `T=20` 新作业组成；四方法各保存 12×400 盒步，独立扫描均为有限有序、endpoint 在同段 tube 内，所有保存四态 tube 均在 `[-2,2]^4`。原生新 C++ 源与旧源仅 `setFixedStepsize(0.1,3)→(0.05,3)` 一行改变，先过一期门检再跑 20 期；P3 的新 YAML 对旧全程亦仅变 ODE 步长。原生作者 checker 打印 `VERIFIED`，前三方没有明确 checker 文字输出。四态绝对终点/每盒平均最大宽度、全时 tube union 见[16 行 CSV](evidence/results/archcomp26_20261001/tora_remain_h005_fourway_saved_20261002/terminal_and_tube_fourway.csv)，[同轴 t–x4 图](evidence/results/archcomp26_20261001/tora_remain_h005_fourway_saved_20261002/tora_remain_h005_fourway_t_x4.png)显示 Huan/Xiangru 保存曲线重合。它不替换当时固定主表的 187 条新尝试或 37+9+14+4 覆盖，也不构成独立证明或单次速度排名。[P3 执行入口勘误](evidence/results/archcomp26_20261001/p3_tora_remain_h005_probe_20261002/SUMMARY.md)保留实际执行源码，并说明事前“首拒即停”在该版本未强制实现；本次无任何盒拒绝，维护入口已补首拒抛错。
- [原生 Flow* 八方向生产门禁](ARCHCOMP26_NATIVE_OCTAGON_PRODUCTION_GATE_20261002.md)先在普通 `ODE::reach` 的 order-4 简谐振子找到 28/1512 项有限解析样本越界；源码定位到初次 Picard 捕获的截断余项被后续余项细化丢失，**复制库**跳过该细化后同样本 0/1512 越界，但首步界明显变宽。另用冻结论文方程 QUAD 的真实 symbolic-remainder 路径，仅首盒、一次 CROWN 控制、一个 `h=0.005` 小步做 observer 开/关门检：两臂 `COMPLETED_SAFE`、1/1 接受，RPC/终态轴界逐值相同；然而 513 个属于 C++ 注入的 CROWN 仿射余项放松控制集的数值样本在 `x7/x8` 末界有 2048/20520 项检查越界，最大 `1.98259×10⁻⁶`，DOP853/Radau 最大参考差 `2.054×10⁻¹⁵`。SR **复制库**跳过对应细化后样本 0/20520 越界，但 `x7` 界从约 `[1.638,1.655]×10⁻⁶` 扩至 `[-5.576,5.609]×10⁻⁴`，不是生产修复。原生真八方向生产接入仍 fail closed；冻结全程作业、源码和库未动，现有 189 条主表尝试及 38 格新完整数值时域计数不因该方法独立诊断改写。该常值控制未证明为真实 NN 输出，不据短门断言原全程每条范围错误，但原生 accepted/`VERIFIED` 不可提升为独立正确性证明。CPU Python TM 短例不能清除此原生门。
- [16 节可编辑阶段 DOCX 与 18 页 PDF](evidence/results/archcomp26_20261001/stage_report/README.md)已由当时 Markdown 的 187 条新尝试截点生成，10 幅图、中文表格和分页经 Word/PDFium 逐页检查。它是固定阶段快照；上述后续 `h=0.05` 对照和原生门禁以最新 Markdown/原始证据为准。尚有 14 格无完整时域和 Airplane discrete 四格合同阻塞，不能标为最终完成。
- [Unicycle 论文常值速度扰动四方同轴图](evidence/results/archcomp26_20261001/unicycle_paper_speed_w_constant_v1/plots/fourway_saved_20261002/SUMMARY.md)只读绘出四法全部 500 步的 `x1/x3` 整步 tube，另在右列比较 T=10 的 `x3/x4` endpoint；Huan/Xiangru 对应边界相同，曲线重合。终点目标只在 T=10 标示；四态同步入目标由独立时间窗审计判定。PNG/PDF/几何 JSON/MATLAB 脚本已保存，MATLAB 未实跑，没有新求解作业。

## 10 月 2 日当前截点：TORA reach-tanh 新 working P3

用户选定的官方 ReLU³/tanh、`u=11f` 合同下，先从固定官方 `.mat` 在新 ID 构造 ONNX 并做 CPU 合同/22 点前向预检（最大差 `9.99e-16`、GPU 未初始化），然后[新 working P3 首周期门检](evidence/results/archcomp26_20261001/tora_reach_tanh_official2026_mat_u11_workingp3_firstperiod_20261002_001/SUMMARY.md)完整初盒 50/50 小步接受。另一独立[10 期/500 小步全程作业](evidence/results/archcomp26_20261001/tora_reach_tanh_official2026_mat_u11_workingp3_full500_20261002_001/SUMMARY.md)在 600 秒外层监督下 500/500 接受、外层 exit 0、内层/外层 wall **10.998221/13.456629 s**。独立保存范围扫描确认 500 条四态 tube/endpoint 有限有序、逐段包含；`T=5` 的 `x1=[0.06791520996448479,0.09301949154372924]`、`x2=[-0.8033787322621515,-0.7759834292813406]` 数值终点盒在目标内，[新 P3 加三方历史的四态绝对终点表](evidence/results/archcomp26_20261001/tora_reach_tanh_official2026_mat_u11_workingp3_full500_20261002_001/terminal_current_p3_historical_author_fourway_T5.csv)按证据代际逐行标明。新 P3 是 `engine_quad_normalization_center` working P3/validation P4，旧 `engine_linear_leaf_v2` P3 全程只作历史对照；旧实验未重启。性质 checker 未运行，不从数值终点补写作者 `VERIFIED` 或独立端到端浮点证明，也不据混合代际单次时间排名。

当前[189 条新尝试工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)为 64 格中 **38 格新完整数值时域、3 格仅短前缀、10 格早停、4 格失败、9 格未尝试、0 格运行中**。[16×4 覆盖附表](evidence/archcomp26_coverage_overlay_20261002.md)为 **38 格新全程、8 格同合同历史全程、14 格无完整时域、4 格 Airplane discrete 合同阻塞**；历史八格为 NAV 五格和 TORA reach-tanh Huan/Xiangru/原生三格。旧 187 条与 37+9+14+4 的[两页状态文件](evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_STATUS_20261002.pdf)和[18 页完整阶段稿](evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_FULL_STAGE_DRAFT_20261002.pdf)保留明确的旧截点，不用新数字改写已冻结的 DOCX/PDF。

[189 条截点的 16 节可编辑 DOCX 与 Word 导出 PDF](evidence/results/archcomp26_20261001/stage_report/README.md)已在原生 QUAD 短门禁结论定点写入 Markdown 后重新生成：21 页 A4、13 幅保存数据图，PDFium 逐页检查无截断。它保留了放松控制样本与真实神经网络轨迹的区别，仍为可审阅阶段稿；上述 187 条截点文件继续作为历史快照。

## 10 月 2 日再接续：Airplane 原生首拒数值定位

在不重启旧三次 smoke 的前提下，两次新隔离原生诊断仅运行官方完整单盒、order-3 配置、一次 12→6 CROWN 调用和首个 `h=0.01` 小步。[_004 对照](evidence/results/archcomp26_20261001/native_airplane_first_reject_trace_20261002_004/README.md)把只读插桩放在未调用的 Interval SR 重载，真实作业仍 status 4、0 段接受，没有新 Picard 数值；原始失败保留。[_005 有效追踪](evidence/results/archcomp26_20261001/native_airplane_first_reject_trace_20261002_005/README.md)改在实际 `Expression<Real>` SR 重载，记录 19 个有限有序 Picard 提议及独立重读审计。默认旧余项 `[-0.01,0.01]` 在 `x/y/z/phi/theta` 五个坐标不包含首轮提议，`x` 提议为 `[-0.069390255181559155,0.070282484851510674]`；原生仍 `UNCOMPLETED_SAFE`、0 段、外层 exit 2。两次初盒和 RPC 逐值一致。此结论只定位这个三阶数值 profile 的第一步自映射拒绝，不是轨迹不安全证据，也不证明加宽初猜就能走完 `T=2`。

两条实际 solver run 均已登记，[当时无哈希矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)为 **191 条新尝试**、64 格中 **38 格新完整数值时域、3 格短前缀、10 格早停、4 格失败、9 格未尝试、0 格运行中**；[覆盖附表](evidence/archcomp26_coverage_overlay_20261002.md)仍是 **38 新全程＋8 同合同历史全程＋14 无全程＋4 Airplane 离散合同阻塞**。189 条截点的 21 页阶段 DOCX/PDF 保持历史快照；当前 Markdown 与原始证据更新。另有原生 QUAD 的历史两处 VAR 截断尾项补丁已通过当前冻结源码适配检查并在新复制库重建。[隔离短门禁](evidence/results/archcomp26_20261001/native_quad_var_tail_repair_gate_20261002_007/README.md)中，普通谐振子周期式/单次式各 3 步与 1,512 项解析有限样本检查通过；论文 QUAD 首盒、一次控制、一个 `h=0.005` 小步的 observer 开/关结果一致，513 条放松控制数值参考的 20,520 项检查无越界。修补版 32 个方向/终点比较区间都落入简单跳过细化版，30 个严格更窄，`x7/x8` 首步终点宽度约缩小 139 倍；原库漏包界仍保留为失败证据。有限样本不能证明整个初盒、控制放松集、整步时间或 50 期闭环；原生生产门维持关闭，不重复原长作业。

## 10 月 2 日：Single Pendulum 已保存四方图

[具名两物理态 Single Pendulum 四方同图与逐态宽度表](evidence/results/archcomp26_20261001/sp_two_state_fourway_saved_20261002/README.md)只读读取 P3、Huan、Xiangru、原生各 100 条已保存小步；逐方法核对原始完成状态、顺序、tube/endpoint 有限有序且逐段包含，并扫描性质窗 `t∈[0.5,1]` 的 `x1` 保存 tube 全在 `[0,1]`。同一张图上排叠四方完整 `x1/x2` tube，下排画 `t=0.8–1` endpoint 对 Huan 的细微差异；Huan 与 Xiangru 保存数值逐项一致。八行 CSV 保存两个物理态的 `T=1` 绝对界及性质窗宽度。该图只读、没有新增求解尝试；它不能补官方 MATLAB 未给出的第三态执行身份、独立浮点 NNCS 证明或稳定速度排名。MATLAB 示例已生成但未执行。

## 10 月 2 日：Airplane 加宽余项首拒与 QUAD 独立整步门

原生 Airplane order-3 的历史 `[-1,1]` 加宽余项一期 smoke 原样保存。[另立的 `_006` 首步追踪](evidence/results/archcomp26_20261001/native_airplane_rem1_first_reject_trace_20261002_006/README.md)只运行同一官方完整单盒、一次固定控制 RPC 和一个 `h=0.01` 小步：外层 exit 2、原生 `UNCOMPLETED_SAFE(4)`、0 保存段，wall 3.625502 秒。实际 `Expression<Real>` 路径的 19 个 Picard 提议有限有序，但 `x/y/phi/theta/psi` 超出 `[-1,1]` 旧初猜；首拒 `x=[-2.52122191537212,2.5218805989602933]`，部分角度区间提议约 `10²⁰`。独立保存文件审计确认新旧初盒及 RPC 逐值一致。大区间是数值提议，不是实际轨迹反例，也没有 T=2 性质结果。此真实求解作为第 **192 条**新尝试登记，[矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)仍为 38 新完整、3 短前缀、10 早停、4 失败、9 未尝试、0 运行；[覆盖附表](evidence/archcomp26_coverage_overlay_20261002.md)仍为 **38 新全程＋8 同合同历史全程＋14 无全程＋4 离散合同阻塞**。

[QUAD 保存 RPC 域整段 `0.005 s` 独立区间检查](evidence/results/archcomp26_20261001/native_quad_independent_interval_gate_20261002_001/README.md)不启动 Flow*，而是从保存 RPC 构造输入区间和 `T·x+` 余项控制包络，以 1000 个区间 Picard 小步覆盖整段时间。当时记录原库 23/32、两处 VAR-tail 修补复制库 27/32 个方向/轴界比较通过，另五项未判定。**后续发现该版 Decimal 一元变号/绝对值可因默认精度而向内舍入，所以这份原审计本身不具严格有向包含资格；同文件的计数须以后文修正版独立重审为准。**此门也未覆盖源码首盒或验证 CROWN、后续控制与端到端 NNCS；原生生产门保持关闭，方法诊断不计入当时 192 条主表尝试。

## 10 月 2 日：旧作者 QUAD P3 trig 后 40 步阶段剖面

[隔离阶段诊断与失败入口收据](evidence/results/huan_quad_stage_a_40_20261001/post_trig_phase_profile_v1/SUMMARY.md)保留首次脚本名 `profile.py` 遮蔽标准库导致 0/40 的失败，以及新身份的成功运行。成功进程按旧作者 QUAD 方程完成 40/40 步和 40,960/40,960 盒步，最终每盒 tube/endpoint/status 数组、40 行观察、部分 driver metrics 与此前保存的 observer-on 对照逐值相同。成功进程外层 60.134 秒；插入逐步 CUDA 同步后的 52.674 秒 advance 流跨度中，weighted accepted 30.503 秒、ordinary validation 17.250 秒，合计 90.66%；图逐出另计 0.090 秒。这是带同步的单次短前缀剖面，不属于 ARCH-COMP26 新实例尝试，也不能推成旧 1000 步成本、稳定速度排名或可删除验证步骤。没有重启旧完整实验。

## 10 月 2 日：DP more 严格内点越界候选的验证门

[独立定点向外舍入见证尝试](evidence/results/archcomp26_20261001/dp_more_validated_witness_20261002_001/AUDIT.md)从已保存严格内点 `(binary64 1.299)^4` 出发，显式包含固定 float32 NN 的一种运行算术误差、每 `0.02 s` 控制期保持及论文 plant 方程，用 Picard 自包含和二阶区间 Taylor 在新方法诊断内接受 1,954 个 `h=0.0001 s` 子步。它在 `t=0.1954` 的下一步首次不能自包含，按首拒规则停止；此前 `t=0.18` 的 `θ̇₁` 盒仍在 `[-0.869633,-0.217428]`，不能触及数值候选约 `t=0.325` 的越界。盒式状态/NN 包络失去相关性、越走越宽，因此**没有严格反例**、完整流管或性质结论；该诊断不计入四方法 192 条尝试。需更紧且固定运行算术合同的验证积分才可能定案，不能将已有数值点轨迹改写为证明。

## 10 月 3 日接续：Airplane 子盒门与 TORA remain 首拒

- [Airplane 六维二分子盒 `_007`](evidence/results/archcomp26_20261001/native_airplane_binary6_firststep_20261002_007/README.md)只求 64 盒规划中的全低角盒 `[0,0.5]^6`：一次 RPC、19/19 首轮 Picard 自包含、一个 `h=0.01` 小步接受，status 2、外层 exit 0，保存 tube 在安全盒内。随后单独[_008 全高角盒](evidence/results/archcomp26_20261001/native_airplane_binary6_coverage_gate_20261002_008/README.md)同样数值接受一个小步，但 `phi/theta/psi` tube 上界超过 1，原生 status 3 `COMPLETED_UNKNOWN`、外层 exit 2；这是性质 Unknown，不是数值拒绝或实际不安全轨迹。其余 62 盒未运行，两个首步不能填完整初集或 T=2。[_009 固定 ONNX 高角点重放](evidence/results/archcomp26_20261001/native_airplane_highcorner_point_replay_20261002_009/README.md)的 11 个数值采样点未越带，只是方法外点诊断，不入四方法索引。
- 原 TORA remain Huan `h=0.1`、统一余项的[第 190 步只读追踪](evidence/results/archcomp26_20261001/author_tora_remain_h01_refusal_trace_20261002_001/README.md)先在 189 步后遇插桩 API 不匹配，原始失败保留；改用已安装引擎支持的只读 hook 后，前 190 步范围与原运行逐字节相同，初盒 2 的 `x2` Picard 提议 `[-0.012253595542717259,0.011665851915157245]` 超 `[-0.01,0.01]`，第 190 步数值首拒，2279/2280 盒步接受。只把 `x2` 余项改为 `[-0.02,0.02]` 的[独立补充 profile](evidence/results/archcomp26_20261001/author_tora_remain_h01_x2rem002_20261002_001/README.md)一期 120/120 接受，但全时请求在第 192 步初盒 2 再拒，2303/2304 接受；保存盒安全前缀仍止于第 184 步。它未补齐 `T=20`，固定主表 Huan/Xiangru 的原 `h=0.1` 早停格保持原收据和设置。
- 两条 Airplane 真实首步 run 与四条 TORA 新 solver run 均已登记在[198 条无哈希 attempt 索引](evidence/archcomp26_nohash_attempts_20261001.json)。工作矩阵显式排除四条 TORA 补充诊断作为主表所选 run，仍选原统一余项的 Huan 全时尝试；[当前矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)为 64 格中 **38 新完整、3 仅短前缀、10 早停、4 失败、9 未尝试、0 运行**，[覆盖附表](evidence/archcomp26_coverage_overlay_20261002.md)仍为 **38 新全程＋8 同合同历史全程＋14 无全程＋4 Airplane 离散合同阻塞**。远端相关进程只读检查为空，未重启旧作业；原生 QUAD 生产八方向门保持关闭。189 截点的 21 页 DOCX/PDF 仍为固定阶段快照。

## 10 月 3 日接续：旧 QUAD 两轮余项、原生短门边界与保存几何叠加

- [旧作者 QUAD P3 两轮加权细化限长诊断](evidence/results/huan_quad_stage_a_40_20261001/weighted_round_gate_v1/README.md)另立新 ID，只运行同合同 1,024 盒×40 小步，40 行观察及最终逐盒 tube/endpoint/status 与先前 `observer_on_002` 直接相等。第一轮 40,960/40,960 盒步接受并改变余项；第二轮 40,615/40,960 接受并改变，另外 345 保留第一轮结果。每步两轮重构与实际终步余项逐位相同，第二轮不能作为逐位等价的直接删减；阶段同步计时仅为早期诊断，不能外推 1,000 步或称速度收益。没有重启旧全程作业，也不把诊断计入 198 条新版 benchmark 尝试。
- [原生 QUAD 保存 RPC 域代数门](evidence/results/archcomp26_20261001/native_quad_algebraic_gate_20261003_001/README.md)结合先前逐小步区间 Picard 审计，对两处 VAR-tail 复制库候选的首个 `0.005 s` 取得 20/20 项**合成物理态**数值包含；pre 的数值列比较不算 `tmvPre` 集合证明。[初盒重心化审计](evidence/results/archcomp26_20261001/native_quad_initial_recenter_gate_20261003_001/README.md)进一步核出：冻结 C++ 首盒 x1–x3 的原始下界与所存 RPC 下界相差 1 个 binary64 ULP，RPC 更窄。故上述独立门只覆盖**保存的 RPC 输入域**，不能说覆盖源码定义的整个首盒；旧 27/32 整步审计的同类表述一并收窄。原冻结库放松控制漏包见证保留，修补候选没有原生生产资格；其余盒、后续控制、CROWN/NN 与全程性质仍未验证。
- [无哈希绘图入口](flowpipe_plot_nohash.md)现在可重复传入保存的 v1 几何文件作同轴多方法叠加，并分记几何验证/导出、MATLAB 脚本生成及 Matplotlib PNG/PDF 渲染用时。[合成双方法验收](evidence/results/flowpipe_plot_nohash_overlay_20261003_001/README.md)保留一方缺第二步的空缺、初盒和按时刻定义的 Safe/Target 图层；它只验证绘图器，不是新可达集结果。另用现有 TORA 原生几何做了单份无哈希重画回归；没有重跑求解，MATLAB/Octave 仍未实跑。
- 在另一新 ID 的[复制库重心化修补首步门](evidence/results/archcomp26_20261001/native_quad_initial_recenter_repair_gate_20261003_002/README.md)中，同时保留先前两处 VAR-tail 修补，并让 `toCenterForm` 半径向外覆盖中心两侧。首个源码物理初盒的十二维都被新 RPC 输入包住；原来缺失的 x1–x3 下边缘不再遗漏。观察器开/关各接受一个 `h=0.005` 小步、各保存一条有限有序范围且数值相同，513 条放松控制样本 20,520 项未见越界。针对**新 RPC**重做 1,000 子步定向区间 Picard 与代数相关性门：27/32 普通盒比较及 20/20 合成物理态比较通过，pre 列仅作数值诊断。该结果只限源码首盒、首个 plant 步、保存的放松控制及精确十进制论文 ODE；不验证 CROWN/NN、本机 Flow* 浮点解析、其他 1,023 盒、后续控制或全时性质。冻结原库和原长作业未动，生产门继续关闭。
- 新[原生 QUAD 全 1,024 初盒首小步隔离门](evidence/results/archcomp26_20261001/native_quad_allbox_firststep_recenter_gate_20261003_003/README.md)沿用复制库双修补候选，仅在**首个控制调用下各跑一个 `h=0.005` 小步**，一次批量 CROWN RPC 后逐盒首拒即停。12,288 个 RPC 输入坐标界全部覆盖相应源码初盒，1,024/1,024 盒数值接受；1,024 条范围、8,192 条方向区间行（每盒 tube、endpoint 各四条）和 12,288 条终态轴界完整有限有序。独立读回无缺行、倒序或同小步 endpoint 超 tube；C++ 4.618 秒只描述这 1,024 个首步，不是 `0.1 s` 完整控制期或 `T=5` 时间。此原生收据当时尚未独立检查其余盒的 plant 包含；后文修正 Decimal 的全盒首步门已补齐这一项。首期余下 19 小步、后续控制、CROWN/NN 及性质仍未通过门禁，生产门继续关闭。原长作业未重启；方法诊断不计入新版主表尝试。

截至上述阶段，主表为 **198 条新尝试**及 **38 新全程＋8 同合同历史全程＋14 无全程＋4 Airplane 离散合同阻塞**；21 页阶段 DOCX/PDF 是 189 截点快照。下文记录后续 200 条截点与新版报告。

## 10 月 3 日后续：DP more 控制包络诊断与 200 条截点

- [P3 more affine-split4 三周期新门](evidence/results/archcomp26_20261001/dp_more_p3_affine_split4_threeperiod_20261003_001/README.md)保持独立 more ONNX、全 `[1,1.3]^4` 的 225 盒、原 P3 阶数和四维 `[-1.5,1.5]` 全时安全盒，仅把控制器 residual 换成有向仿射四分外包络。12/12 小步、2,700/2,700 盒步数值接受，保存 tube 全在安全盒内，最小裕量 +0.01133738449；它只请求 3/20 期，不是 `T=0.4` 结果。
- 随后的[同配置 20 期完整请求](evidence/results/archcomp26_20261001/dp_more_p3_affine_split4_full20_20261003_001/README.md)外层 `completed/exit0`、wall 62.360384 秒，但作者 checker 在 18 期后输出 `Unsafe.`，内层只保存 **72/80 小步、16,200/18,000 盒步**；全部已存盒步数值接受、`broken=0`。独立读回 72×225 四态 tube/endpoint 均有限有序且 endpoint 含于同段 tube。所有盒保存安全的前缀为前 60 步至 `t=0.30`，第 61 步首有 tube 跨带；第 72 步一盒的 `θ̇₁` tube 完全低于 −1.5。这是作者 checker 性质早停，不是已独立核实的真实轨迹反例；外层正常退出不等于 `T=0.4` 完整数值时域。旧区间残差第 9 步数值收缩失败继续独立保留，两次未完成进程时间不作完整任务速度排名。
- 两条新真实 solver run 已入[200 条无哈希 attempt 索引](evidence/archcomp26_nohash_attempts_20261001.json)。三期门标为补充短前缀，20 期请求成为 P3 主格所选的早停记录。[工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)和[历史覆盖附表](evidence/archcomp26_coverage_overlay_20261002.md)重算后仍为 **38 新完整＋8 同合同历史完整＋14 无完整时域＋4 Airplane 离散合同阻塞**，64 格无运行中。原生 QUAD 全盒首小步只属方法门禁，不入该 attempt 计数；其生产八方向门保持关闭。
- [200 条截点的可编辑完整阶段 DOCX 与 Word PDF](evidence/results/archcomp26_20261001/stage_report/README.md)已从逐实例 Markdown 重建，23 页 A4、14 幅保存数据图和 10 个表格；PDFium 逐页检查中文、图表与分页。187/189 截点继续保留为历史快照。此稿仍是可审阅阶段记录，未把 DP more 早停、QUAD 首小步门检或数值全程升格为独立性质证明或稳定四方速度排名。

## 10 月 3 日：旧作者 QUAD 一轮加权细化短程变体

[独立 40 步一轮变体](evidence/results/huan_quad_stage_a_40_20261001/weighted_one_round40_v1/README.md)仅把两轮 `refine_accepted` 改为一轮，旧作者方程及 1024 盒、P3/P2/P4、strict 边界、两个控制周期与 trig 复用不变。它接受 40/40 小步、40,960/40,960 盒步；独立读回的 40 行保存区间都比原两轮参考宽，终步 tube 与 endpoint 各 22,528 个端点向外改变、0 个向内。外层单次 wall 45.207353 秒与不同日期、附带每轮同步/复制的两轮短诊断不能直接相减当加速；这不是数值等价优化，也无 1000 步终点/性质/完整时间。它不计入 200 条 ARCH-COMP26 新尝试，旧全程作业未重启。

## 10 月 3 日：原生 QUAD 全初盒首小步的修正后独立 plant 门

[旧 Decimal 方法守卫](evidence/results/archcomp26_20261001/native_quad_allbox_independent_plant_firststep_20261003_004/README.md)在首盒的有向控制界构造中查出一元变号/绝对值按环境 28 位精度向内舍入；该次方法首拒发生于 Picard 前，不是 Flow* 失败。[修正同一旧窄 RPC 的独立重审](evidence/results/archcomp26_20261001/native_quad_legacy_narrow_rpc_decimal_reaudit_20261003_006/README.md)重新得到原库 23/32、VAR-tail 复制库 27/32，精确代数补足后为该**窄 RPC 域**20/20 合成物理态数值包含；其输入仍缺源码首盒前三维下缘 1 ULP。[修正后首盒至第六盒的门](evidence/results/archcomp26_20261001/native_quad_allbox_independent_plant_firststep_fixed_decimal_20261003_005/README.md)各完成 1000 子步严格 Picard 与 20/20 比较；第七盒因固定 `|x11|<0.001` bootstrap 假设不成立而停在 Picard 前，原始停点保留。

[逐盒精确控制界的自适应 bootstrap 门](evidence/results/archcomp26_20261001/native_quad_allbox_independent_plant_adaptive_bootstrap_20261003_006/README.md)先通过第七盒，再在一次顺序任务中完成余下 1016 盒。合并先前七盒，**1024/1024 源定义初盒**各有 1000 个严格 Picard 子步；全部 **20,480/20,480** 个合成物理态界比较包含于复制库保存范围，12,288 个源盒/RPC 坐标比较通过。[另立账本读回](evidence/results/archcomp26_20261001/native_quad_allbox_firststep_ledger_audit_20261003_007/README.md)核实 0–1023 盒各一次、重建控制包络及首离开条件、比较计数无缺。这是**以保存的 CROWN 仿射加余项控制包络有效为前提**的首个 `h=0.005 s` plant 步结论；没有独立证明 NN/CROWN 包络、Flow* 内部浮点、首期其余 19 步、后续 49 期或 `T=5` reach-and-remain。原库的有限放松控制漏包样本仍在，原生生产八方向门继续关闭；这些检查均不增加 200 条 benchmark 尝试及 64 格完整数值时域计数。

## 10 月 3 日：原生 QUAD 首盒第二小步及保存控制包络审计

- [隔离第二小步回放](evidence/results/archcomp26_20261001/native_quad_lane0_secondstep_replay_20261003_008/README.md)只读取旧首控 RPC，先逐值复现首盒原保存的首步范围、八方向和十二终态轴，再用同一常值控制推进至 `t=0.010`，2/2 小步数值接受。独立 plant 门从原初盒连续完成两段各 1000 次严格有向 Picard 子步，重建七项首离开条件；第二小步的 tube、endpoint 及终态轴 **20/20** 项落入保存的未扩张 binary64 界。该结果只覆盖源盒 lane 0 的前两小步，并且假定保存控制包络有效；其余 1023 盒第二步、首盒余下 18 步与后续控制均未闭合。
- [保存 CROWN/NN 合同只读审计](evidence/results/archcomp26_20261001/native_quad_saved_crown_nn_contract_audit_20261003_001/README.md)确认旧 RPC 写入下界斜率与上下 bias，但没有保存上界斜率或独立 same-slope 证据。首盒第一输出的传输后 `T·X` 宽度约 3.39016，已存 residual 带宽约 0.06710；把 `f(X)` 和 `T·X` 分别做普通区间再相减，必然过宽，不能成为独立网络证书。这是检查方法的局限，不是网络反例。需要保留输入相关性的有向 sigmoid 仿射 residual 证明。生产门保持关闭；两项诊断不增加主表 200 条尝试及 38＋8＋14＋4 格计数。
- [首盒独立相关仿射 sigmoid 门](evidence/results/archcomp26_20261001/native_quad_lane0_nn_affine_certificate_20261003_009/README.md)按本地选定 ONNX 的精确 binary32 参数与保存 RPC 首盒做 80 位有向区间计算，三输出均未落入保存的 residual 带，状态 **UNDECIDED**。第一输出独立外包络宽 0.09847，保存带宽 0.06710；这只能说明当前外包络偏宽，不是实际网络违例。64 子盒尝试在完成覆盖前停止，未产生可用结论；服务器模型字节身份及 float32 推理语义也未由此检查。生产门与 200 条主表计数不变。
- [保存 CROWN 系数的 float32 传输门](evidence/results/archcomp26_20261001/native_quad_crown_f32_transport_gate_20261003_001/README.md)用精确有理数比较原仿射系数及 C++ `.asFloat()` 后的系数。即使假设原 CROWN 实数下界有效，三个输出分别有 573、615、533/1024 盒不满足其**直接继承该证书的充分条件**；若再假设未保存的 `uA=lA`，任一侧失败计数为 833、828、795/1024。它只暴露传输证书缺口，不能断言真实 NN 输出越界。lane 0 第一输出直接传输需把下 bias 向外补约 `3.240×10⁻⁷`；原生冻结 run 未改，生产门仍关闭，也不增加 benchmark 尝试数。
- 随后仅做一次[同批 1024 盒 CROWN 控制器调用](evidence/results/archcomp26_20261001/native_quad_crown_same_slope_batch_20261003_001/README.md)，补录 `uA`。新 `lA/lbias/ubias` 与旧 RPC 的 43,008 个数值逐值相同，`uA=lA` 的 36,864 个数值逐值相同；这为首批相同数值输入补上了 same-slope **记录**，没有重跑 plant 或原完整作业。它不证明 CROWN 浮点计算正确，也不消除上述 `float32` 传输充分条件失败；原生生产门继续关闭，主表仍为 200 条。
- [首批系数的逐盒条件性修正账本](evidence/results/archcomp26_20261001/native_quad_crown_transport_correction_20261003_001/README.md)依托新 `uA=lA` 记录，用精确有理数求每盒理想实数仿射证书转入原生 float32 斜率后所需残差端点，向外取最近 binary32。3,072 输出行中 2,456 行、1,024 初盒中 1,014 盒至少一端改变；三输出最大残差宽度增量约 `1.90735×10⁻⁶`、`1.97906×10⁻⁹`、`1.39698×10⁻⁹`。这仅是**假设原 CROWN 实数界正确**时的传输修正计划，未运行 C++ 中心/半径、plant 或全程；原生生产门关闭，主表计数不变。
- [Single Pendulum 官方第三态定点来源复核](ARCHCOMP26_SINGLE_PENDULUM_THIRD_STATE_SOURCE_FOLLOWUP_20261003.md)再次从固定论文、官方递归文件树、MATLAB 方程/规格、原始作者仓库和论文重复性目录追查：论文给两物理态及闭性质窗，`dynamics_sp.m` 另有 `dx3=1`，但找不到 2026 MATLAB 第三态初值、重置、NN 投影或性质 checker 的执行入口。已有四方具名两态 profile 继续保留原身份，不能升格三态 MATLAB 复现；主表计数不变。
- [条件性校正后的原生控制构造首拒](evidence/results/archcomp26_20261001/native_quad_corrected_control_construct_20261003_001/README.md)在隔离 C++ 中重建 1,024 源盒输入、按原 `.asFloat()` 和双精度中心/半径顺序构造，12,288 个输入对与原 RPC 逐值相同；对 lane 0 第一输出的更强整盒仿射参考包含检查首次拒绝，build 0/run 4，CROWN/ODE 调用均 0，其余 3,071 行未检。首行有理数重读支持中心±半径恰好等于校正端点，拒绝属于整盒参考比较；该参考可能宽于关联 TM 像，CSV 15 位精度不足以判哪侧 ULP 差。因此只是 `UNDECIDED` 的方法门，不是实际 NN/控制反例，也不使原生生产门开放。
- [首盒全精度关联控制构造门](evidence/results/archcomp26_20261001/native_quad_lane0_correlated_control_gate_20261003_001/README.md)在另一个隔离复制库只回放保存首批 RPC 的 lane 0 和三项条件性校正输出，导出原生输入/输出 TM 的精确 binary64 值；不重跑旧实验、CROWN 或 ODE。源盒/RPC 输入审计通过，build/native trace/check 退出码为 **0/0/2**。[原始 RESULT](evidence/results/archcomp26_20261001/native_quad_lane0_correlated_control_gate_20261003_001/RESULT.json)按首拒原则在第 1 输出停止：同一归一化符号的精确有理数充分包含检查下界裕量 `−3.330452308×10⁻¹⁶`、上界 `+1.104650772×10⁻¹⁶`。[同一 trace 的三输出只读分析](evidence/results/archcomp26_20261001/native_quad_lane0_correlated_control_gate_20261003_001/ALL_OUTPUTS_ANALYSIS.json)得到三个下界裕量 `−3.330452308×10⁻¹⁶`、`−5.159067004×10⁻¹⁹`、`−5.310919262×10⁻²⁰`，上界均为正；保留固定多项式及输入 TM 时，对最终 remainder 下端向外移的最小值分别是 48、1、1 个 binary64 ULP。这只说明**假设原 CROWN 实数仿射界有效**时，当前构造未通过该充分条件；改校正 bias 会重新构造 TM，这些 ULP 数并非已验证补丁，更非真实 NN/控制越界证据。NN/CROWN 独立正确性、后续控制与全时性质仍缺，生产门保持关闭；主表仍为 200 条新尝试及 38＋8＋14＋4 格。
- 旧作者 QUAD 的[加权 CUDA 图 256 行新候选](evidence/results/huan_quad_stage_a_40_20261001/weighted_chunk256_v1/README.md)先在 40/40 短门接受全 40,960 盒步，保存观察/终态与旧 128 行短参考直接一致；加权图调用 640→320 次。随后[全 1,000 步新 ID](evidence/results/huan_quad_stage_a_40_20261001/weighted_chunk256_full1000_v1/README.md)在旧合同、旧 1,000 个 observer、预编译扩展和空闲 GPU3 的自动预检后完成：1,000/1,000 小步、1,024,000/1,024,000 盒步接受、50 控制，outer/watchdog/driver wall 分别 **1093.267186/1093.192982/1086.231490 秒**。独立[逐步直接保存值对照](evidence/results/huan_quad_stage_a_40_20261001/weighted_chunk256_full1000_v1/FULL_SAVED_COMPARISON.json)使 1,000 步的逐盒 `1024×12×4` bounds、accepted、status 和 pooled 观察全部与旧 P3/trig 完整参考逐字节相同；旧参考无另存的 post-driver final NPY，未冒称那组三数组直接相同。旧参考 watchdog 单次 1533.752052 秒，异日无交错 128 行对照，故只能并列描述单次时间，不能把差值认定为分块改动的因果或稳定加速；旧作业未重启。
- 新[旧作者 QUAD P3 同包装 128 行全程对照](evidence/results/huan_quad_stage_a_40_20261001/weighted_chunk128_samewrapper_full1000_v1/README.md)在新 ID 完成 **1000/1000** 小步、**1024000/1024000** 盒步、50 控制；加权图 16,000 次映射同样覆盖 2,048,000 行、无填充，256 行同包装候选为 8,000 次。128 行 outer/watchdog/driver 单次 wall 为 **1350.901441/1350.820362/1343.874426 秒**；[全部 1000 步直接保存值审计](evidence/results/huan_quad_stage_a_40_20261001/weighted_chunk128_samewrapper_full1000_v1/FULL_SAVED_COMPARISON.json)对逐盒范围、接受、状态和汇总观察与旧完整 P3/trig 参考直接相同；[三份终态 NPY](evidence/results/huan_quad_stage_a_40_20261001/weighted_chunk128_samewrapper_full1000_v1/FINAL_ARRAYS_DIRECT_COMPARISON.json)与 256 行运行直接字节相同。256 行 outer 比 128 行单次短 **257.634255 秒**，但顺序未交错、各仅一次，可能有主机状态漂移，不能称稳定或因果加速；未比较隐藏 TM/SR 或旧参考缺失的 post-driver 终态三数组。这是旧合同速度诊断，200 条新 ARCH 主表尝试和 38＋8＋14＋4 格不变，旧作业未重启。

## 10 月 3 日：TORA reach-sigmoid 计时首槽审计误拒与 201 条截点

第一次独立四方计时 campaign 的[原始事件及首槽数值重扫](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_001/README.md)显示：仅 `first00_native` 被启动，外层 `completed/exit 0`、wall **8.989589569 秒**。原生实际保存 500/500 个四态小段、10 次控制 RPC 和 10 个 HTTP 200，作者终点 checker 打印 `VERIFIED`；直接读取保存范围确认 500 个 lane/step 无缺行，四态 tube/endpoint 有限有序且逐段包含，`T=5` 的 x1/x2 endpoint 在所选目标内。此进程不是数值失败或性质 `UNKNOWN`。

冻结的 campaign `runner.py` 在第 50 小步用 `|h−0.01|≤10⁻¹²` 作为网格判据；实际保存的 `h=0.009999999998999698`，偏差 **1.00030227157×10⁻¹²**。审计器因此误标 `valid=false` 并按首拒停掉后续 23 槽；旧事件和源码均保留。该已完成进程现为[无哈希索引](evidence/archcomp26_nohash_attempts_20261001.json)的第 **201** 条，标 `matrix_eligible=false`、`timing_eligible=false`，不混入后续独立 campaign 的正式计时样本。新 ID 的 24 槽在本截点尚未审计入账，不预填结果。[16×4 工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)和[历史覆盖附表](evidence/archcomp26_coverage_overlay_20261002.md)重算后仍为 **38 新完整＋8 同合同历史完整＋14 无完整时域＋4 Airplane 离散合同阻塞**，主格状态不变。200 条截点的 DOCX/PDF 保留原快照；当前 Markdown 报告对应 201 条截点。

同日另立的[原生 QUAD 首盒后构造余项隔离门](evidence/results/archcomp26_20261001/native_quad_lane0_correlated_remainder_gate_20261003_001/README.md)仅在保存首盒三路控制 TM 的**最终 remainder 下端**各外扩精确 `2⁻⁵⁰`，保持输入 TM、系数、bias、中心/半径与输出多项式不变。构建、原生 trace 与精确有理数检查退出码 **0/0/0**；三路同符号条件性包含的下界裕量变为 `+5.551331889×10⁻¹⁶`、`+8.876625130×10⁻¹⁶`、`+8.881253105×10⁻¹⁶`，上界亦通过。这验证该**单盒、假设原实数 CROWN 仿射界有效**的构造机制；固定外扩量来自已观察缺口，不是 1024 盒统一舍入预算或已验证生产补丁。没有运行 NN/CROWN、plant ODE 或全时性质，原生八方向生产门仍关闭，不计入 benchmark 新尝试。

## 10 月 3 日：TORA reach-sigmoid 四方 24 次有效计时与 225 条截点

[新独立 ID 的 24 个原始子进程](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/SUMMARY.md)全部自然完成；四方法各有本 campaign 首个新进程一次及后五次新进程，顺序轮换使用物理 GPU 2、CPU 10–13。独立镜像审计逐条比对 PLAN、events、SUMMARY、outer/inner START/RESULT/metrics、原生 Step/RPC/HTTP200 与二进制范围：**24/24** 外层 exit 0、每次 10 期及 **500/500** 小步；共 **12,000/12,000** 条四态 tube/endpoint 有限有序、端点落入同段 tube，24 个保存 `T=5` 目标坐标终点盒均入目标。六个原生进程共 **60** 次 RPC/HTTP200 且作者终点 checker 各打印 `VERIFIED`；18 个 GPU 进程的性质 checker 未执行。机器可读的[逐次审计](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/INDEPENDENT_AUDIT.json)和[逐次 CSV](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/RUNS.csv)均来自本轮原始保存文件；第一次 ID 的完整原生进程因审计阈值误拒，未混入这 24 个计时样本。

统一外层进程 wall 的后五次 median 按 native/Huan/Xiangru/P3 为 **8.942749/13.652781/13.607118/13.904651 秒**；各方法首轮及 min–max 见[campaign 摘要](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/SUMMARY.md)。这些只是共享主机、不同引擎/观察器路径下的描述性时间，不能推出稳定四方速度排名或独立浮点 NNCS 证书。该截点的[无哈希索引](evidence/archcomp26_nohash_attempts_20261001.json)为 **225 条新尝试**（原201条＋24条有效新进程），顺序重建的[工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)及[覆盖附表](evidence/archcomp26_coverage_overlay_20261002.md)仍为 **38 新完整＋8 同合同历史完整＋14 无完整时域＋4 Airplane 离散合同阻塞**，64 格无运行中。该 225 条截点的 DOCX/PDF 已另存并核查；后续证据见下一节。

## 10 月 3 日：Airplane 连续全分盒首步、QUAD 控制构造首拒与 287 条截点

Airplane 连续版六个不确定初态各二分，64 个闭盒拼合覆盖完整初集。继先前两角盒之后，新独立队列仅运行[其余 62 盒](evidence/results/archcomp26_20261001/native_airplane_binary6_numeric_cover_20261003_001/README.md)，没有重启旧实验；原始保存文件独立审计确认新 62/62 与总 **64/64** 均数值接受各自首个 `h=0.01 s` 小步，合计 **1216/1216** 条首轮 Picard 包含记录。64 盒中 **8 盒首步保存 tube 在安全盒内，56 盒性质 Unknown**；Unknown 不是数值失败或真实轨迹越界。结果只在保存控制包络条件下覆盖每盒第一个 plant 小步，不覆盖首个 `0.1 s` 控制期剩余步、后续控制或 `T=2 s` 全时域；完整初盒的四方法主表格仍无全程结果。

原生 QUAD [固定余项外扩量全盒隔离门](evidence/results/archcomp26_20261001/native_quad_allbox_correlated_remainder_gate_20261003_001/README.md)写出 1024 源盒首批控制 TM trace，12,288 个构造输入界与保存 RPC 一致；在先前 lane 0 三输出通过后，精确有理数同符号充分条件于 **lane 1/output 1** 首拒，下界裕量 `−1.188470239739×10⁻¹⁵`、上界为正。因此实际只核查 1 个完整盒和下一盒 1 个输出行，不把 trace 的 1024 行误写成全盒通过。固定 `2⁻⁵⁰` 外扩量没有通过此门；该负裕量不证明真实 NN 输出越界。未调用 NN/CROWN 或 ODE，也没有修补后 plant、后续控制和全时性质的证明；原生八方向生产门仍关闭，此隔离诊断不计入四方 benchmark 尝试。

[无哈希索引](evidence/archcomp26_nohash_attempts_20261001.json)在此截点为 **287 条本轮尝试**（原 225 条＋Airplane 新 62 子盒），[工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)的 64 格仍为 38 完成、3 仅短前缀、10 早停、4 失败、9 未尝试、0 运行中；[历史覆盖附表](evidence/archcomp26_coverage_overlay_20261002.md)仍分为 **38 新完整＋8 同合同历史完整＋14 无完整时域＋4 Airplane discrete 合同阻塞**。Airplane 原生主格仍选完整初盒失败入口，新子盒不构成 `T=2` 完整结果；QUAD 构造诊断不改变方法格。此前 225 条截点 DOCX/PDF 保留为独立快照；当时 Markdown 总报告对应 287 条。

## 10 月 3 日：Airplane 高角盒首控制期数值首拒与 288 条截点

在独立新 ID 的[全高角 `111111` 子盒首个常值控制期诊断](evidence/results/archcomp26_20261001/native_airplane_binary6_period_numeric_20261003_001/README.md)中，保持先前二分初盒、官方 12→6 ONNX、ODE、order 3、`h=0.01 s` 和余项初猜，只将一次求解请求延长至 `0.1 s` 的 10 个小步，并在数值调用内关闭性质早停。新旧初盒、首次控制 RPC 及首个 408-byte 保存范围逐值相同。独立离线审计确认 **4/10 小步保存至 `t=0.04 s`**，第 5 步的 `x/y` Picard 提议超出 `[-0.01,0.01]` 初猜；原生 status 4、外层 failed/exit 2。四个保存 tube 的 `cos(theta)` 下界均为正，最小 `0.5350594255572875`；保存前缀对安全盒的性质为 **Unknown**，没有实际不安全轨迹结论。此次是单个高角子盒的数值首拒，不覆盖完整首控制期、其余 63 盒后续步或 `T=2 s`，也不是独立 NN/CROWN 或浮点 Flow* 证明；旧首步任务未重启。

该作业作为[无哈希索引](evidence/archcomp26_nohash_attempts_20261001.json)第 **288** 条补充早停尝试登记，`matrix_eligible=false`；[工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)仍为 38 完成、3 仅短前缀、10 早停、4 失败、9 未尝试、0 运行中，[覆盖附表](evidence/archcomp26_coverage_overlay_20261002.md)仍为 **38 新完整＋8 同合同历史完整＋14 无完整时域＋4 Airplane discrete 合同阻塞**。原生 Airplane 连续主格仍由完整初盒入口决定。287 条截点的 DOCX/PDF 保留为独立快照，当前 Markdown 总报告按 288 条索引更新。
