# ARCH-COMP 2026：下一阶段依据

核查日期：2026-09-30。只读文献与源码核查，未启动实验。

## 最新公开报告

已核实最新版本是 **ARCH-COMP26 Category Report: Artificial Intelligence and Neural Network Control Systems (AINNCS) for Continuous and Hybrid Systems Plants**，Samuel Sasaki等，EasyChair EPiC Series in Computing 110，2026-07-03，85–130页（PDF46页），DOI `10.29007/637n`。

- [官方发布页](https://easychair.org/publications/paper/GsKW)
- [官方PDF](https://easychair.org/publications/paper/GsKW/download)
- [官方ARCH26卷](https://easychair.org/publications/volume/ARCH26)
- 本地PDF：`ARCH_COMP26_AINNCS.pdf`，3,157,362 bytes，SHA256 `964326d7d418eeb9ecc258c655f358e1764fdb08fdc1d97e0a12432cc3c7fb3e`。
- Crossref检索首5项未返回本报告，精确DOI API返回404；这是索引缺口，不影响官方发布页及完整PDF证据。

PDF、全文提取和页面图片仅保留为本地阅读依据，不复制进本轮发布仓库。可发布本说明、元数据索引、官方链接及原创goal。

## 完整非VCAS清单

来源：[官方2026 benchmark README](https://github.com/Kiguli/ARCH-COMP2026)。冻结main为 `d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26`（committer 2026-03-26）；README blob `bc191ca64418465c18b9ff9f65f3d7110a2112a1`。本文按其控制器/时间语义展开，而非按Git分支计数。

| 序号 | Benchmark / instance | 建议图坐标（官方README） |
|---|---|---|
| 1 | ACC / safe-distance | distance–time |
| 2 | Airplane / continuous | 状态2、7 |
| 3 | Airplane / discrete | 状态2、7 |
| 4 | Attitude Control / avoid | 状态1、2 |
| 5 | Balancing（CartPole）/ reach | 状态1、3 |
| 6 | Docking / constraint | 状态1–time |
| 7 | Double Pendulum / less-robust | 状态3、4 |
| 8 | Double Pendulum / more-robust | 状态3、4 |
| 9 | NAV / standard | 状态1、2 |
| 10 | NAV / robust | 状态1、2 |
| 11 | QUAD / reach | 状态3–time |
| 12 | Single Pendulum / reach | 状态1–time |
| 13 | TORA / remain | 状态1、2及3、4 |
| 14 | TORA / reach-sigmoid | 状态1、2 |
| 15 | TORA / reach-tanh | 状态1、2 |
| 16 | Unicycle / reach | 状态1、2 |

VCAS按用户要求排除。旧14配置比本清单少Docking和Airplane discrete。离散Airplane必须独立核实离散模型与性质，不能用连续ODE输出代替；其余文档、README、实际入口的差异亦应逐项记录。

## 图式：已经视觉核查

已查看PDF第27页（印刷111页，Fig22–23）与第29页（印刷113页，Fig26–27）：绿色flowpipe，浅黄色initial set，浅蓝色safe/goal set，红色obstacle。Single Pendulum安全区域从规定时间开始；QUAD图是时间采样的竖向区间。NAV同时画目标区域与障碍。

用户明确“verified box”指图中安全区域。功能应按各任务实际性质标为Safe set、Goal set或Unsafe set/Obstacle，不再解释成成功初始子盒。支持相关时间窗口、初始盒、legend以及MATLAB输出。安全区域可视化本身不替代验证结论；抽样显示需明确，验收数据仍保留全部步骤。

## QUAD文档—冻结源码差异：待核，不直接判错

报告§3.9，PDF第13页（印刷97页），已视觉核对。冻结本地源码：

`/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/sources/native/submit/CROWN-Reach/archcomp/Quadrotor/quad.cpp`

该保存CROWN-Reach版本为 `7b90f30831212f0c59c44c67aa428932d5307a81`。

| 项 | 报告公式 | 冻结源码 |
|---|---|---|
| x2'中x5和x6系数 | x5含`+cos(x7)cos(x9)`；x6含`-sin(x7)cos(x9)` | 第54行两个对应符号相反 |
| x4' | `x12*x5 - x11*x6 - g*sin(x8)` | 第56行`x12*x5 * x11*x6 - 9.81*sin(x8)` |
| x5' | `x10*x6 - x12*x4 + g*cos(x8)*sin(x7)` | 第57行`x10*x6 - x11*x6 - 9.81*sin(x8)` |

必须进一步核对2026官方模型、重复性归档、坐标约定与作者实际入口，确定新实验合同。历史作者复现保留原样；若采用文档模型，作为另一个明确命名的合同，不覆盖历史结果。`x12'=0`在本报告参数Jx=Jy、tau_psi=0下与给定方程一致，不列为差异。

## 原生绘图出处

QUAD上述文件第286–296行有注释的`transformToTaylorModels`、合并flowpipes及`t,x3`绘图调用。

服务器只读核实：`/srv/local/shengenli/flowstar_mainline_causal_b85a321_20260810/flowstar-toolbox/Continuous.cpp:9261`的`Plot_Setting::plot_2D_octagon_MATLAB`（声明`Continuous.h:2856`）输出`.m`，通过八方向界构造投影多边形；离散绘图设置取步末。可移植其输出接口思路，用当前TM投影生成图；如果只有逐坐标hull，则明确画矩形，不冒充相关性八边形。

机器可读来源、完整16行scope及SHA见`ARCH_COMP_2026_REFERENCE.json`。上文是下一阶段定位依据，不是2026全模型审计或新实验报告。
