# ARCH-COMP26 TORA remain：Flow* native 新全程尝试

日期：2026-10-01。新运行目录是服务器
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_tora_remain_full20_001`。这是固定 2026 论文/官方规格的独立单次运行；历史 TORA 作业没有重启。用户本轮要求不做哈希校验，本次没有执行内容摘要计算。

## 合同与入口

- [共享合同](../../../../ARCHCOMP26_TORA_REMAIN_CONTRACT_20261001.md)：四物理态、全初始盒、12 个覆盖子盒、20 个 1 s 控制周期、`h=0.1`、全过程 `[-2,2]^4` 安全集。原生方法使用 Taylor order 3、4 个 CPU 线程、GPU2 CROWN RPC。
- 新隔离二进制和服务端位于远端 `.../native_tora_remain_build_001/archcomp/TORA/`；[BUILD.json](BUILD.json) 保留编译命令和源路径，[START.json](START.json) 保留实际命令、模型路径和设备。模型是固定官方 2026 ONNX 的隔离副本，已与服务器现有同名旧文件逐字节对照相同。
- 原生实际 [initial_boxes.json](initial_boxes.json) 为 12×6，前四维是物理态；与保留的旧 12 盒 ledger 数值完全相同。此处只有分区可复用，旧求解结果没有复用。
- 同一 build 的一周期 12 盒 plumbing smoke 在独立 `native_tora_remain_smoke1_001` 完成，1 RPC、120 范围记录、进程 wall 4.226222 s；它不作为 T=20 结果。

## 新运行实测

| 字段 | 结果 |
|---|---:|
| 进程 wall（[RESULT.json](RESULT.json)） | 8.339151 s |
| 原生求解器自报时间（[native.log](native.log)） | 4.735000 s |
| 控制周期 / RPC | 20 / 20 |
| RPC 每次输入 / 输出 | 12 盒 × 4 态 / 12 盒 × 1 控制 |
| ODE 小步 | 200 / 200 |
| 保存范围记录 | 2,400 / 2,400 个唯一 `(lane,step)` |
| 原生 checker | `VERIFIED` |
| 扫描检查 | 有限、有序、无缺盒缺步、无 endpoint 越出同小步 tube、所有保存 tube 在 `[-2,2]^4` 内 |

[SCAN.json](SCAN.json) 是对实际 [ranges.bin](ranges.bin) 的独立读取结果。全时保存 tube 的四态 union 依次是：

| 状态 | tube union | `T=20` endpoint union | `T=20` 每盒 endpoint 平均宽度 | 最大宽度 |
|---|---|---|---:|---:|
| `x1` | `[-0.9824693945103186, 0.8523752517413353]` | `[-0.12080135448520393, 0.0014545196664676456]` | 0.08500496228437161 | 0.1133784987347304 |
| `x2` | `[-1.0095867327453434, 0.9245144713359922]` | `[-0.2689787595574199, -0.1056643590682298]` | 0.09670441169031806 | 0.12732931819764776 |
| `x3` | `[-1.1816778974066222, 1.0455706042602457]` | `[0.05603333011303763, 0.9348903254733817]` | 0.5210208003381384 | 0.8718328557605826 |
| `x4` | `[-1.404338861306241, 1.5639798371619464]` | `[-0.6762171703385176, 0.2839682090526269]` | 0.561529785385826 | 0.9556767557396498 |

原生安全集 checker 的 `VERIFIED` 与保存区间读回均支持**该方法的这一次完整运行**，尚无独立的端到端浮点 NN/CROWN 证明。服务器 CROWN 导入时仍出现现有 auto_LiRPA 的 batch-dimension constant 警告，原样保留于 [server.log](server.log)；不得将该警告静默解释为已完成资格。

## 直接生成的图

- [t–x1 全时 tube PNG](plots/tora_native_t_x1_tube.png)、[PDF](plots/tora_native_t_x1_tube.pdf)、[MATLAB 脚本](plots/tora_native_t_x1_tube.m)
- [x1–x2 五个显示时刻 endpoint PNG](plots/tora_native_x1_x2_endpoint.png)、[PDF](plots/tora_native_x1_x2_endpoint.pdf)、[MATLAB 脚本](plots/tora_native_x1_x2_endpoint.m)

两图都画了完整初始盒和 `[-2,2]` Safe region。几何为保存逐坐标范围的 box 投影，不是保留关联性的 octagon；状态—状态图只抽取 5 个显示步，性质扫描仍覆盖全部 200 步。绘图收据明确原生 `ranges.bin` 不含 accepted/status，不能由图单独推断求解器验收。MATLAB/Octave 未安装，`.m` 已生成但未实跑。
