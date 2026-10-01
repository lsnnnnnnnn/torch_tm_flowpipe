# ACC 四方已保存范围：T=5 端点与全时 tube

本目录只读取 2026-10-01 四份**新**、完整 50 期的 ACC participant-order 运行记录，未启动求解器、修改原记录或计算内容摘要。共同执行合同是六物理态顺序 `(x_lead,v_lead,a_lead,x_ego,v_ego,a_ego)`、一初盒、控制保持周期 `0.1 s`、`T=5 s`、固定 2026 ONNX 与参与者输入顺序 `[30,1.4,v_ego,x_lead-x_ego,v_lead-v_ego]`。论文没有定义最后一项的符号，故这些数字只属于[明确命名的参与者合同](../../../../ARCHCOMP26_ACC_PARTICIPANT_CONTRACT_20261001.md)。

## 数据

- [逐方法 CSV](acc_t5_endpoint_and_full_tube_long.csv)：24 行，每方法、每物理态单独保留 `T=5` 端点 `lo/hi/width` 和 50 段 whole-step tube 的全时 union `lo/hi`，并列出观察器及源文件。
- [并排 CSV](acc_t5_endpoint_and_full_tube_wide.csv)：六物理态各一行；四方法各有独立的五个数值列，不合并不相同的观察器。
- [生成脚本](make_csv.py)：只用 Python 标准库读取原始记录；要求每方法恰好 50 个连续期、`h=0.1`、六个有限且有序的区间。端点取第 50 期记录；全时 tube `lo=min(50 个 lo)`、`hi=max(50 个 hi)`；端点宽度为保存端点的 `hi-lo`，未另外扩大舍入误差。

原生输入是 [`ranges.bin`](../acc_native_var_tail_full50_001/ranges.bin)：每条小端记录为 `uint64 lane, uint64 step, float64 h`，随后每态 `(tube_lo,tube_hi,endpoint_lo,endpoint_hi)`。GPU 三方各读自己的 `ranges.jsonl`：[Huan](../acc_huan_full50_001/ranges.jsonl)、[Xiangru](../acc_xiangru_full50_001/ranges.jsonl)、[ours/P3](../acc_p3_full50_001/ranges.jsonl)。CSV 的 `source` 列相对于上一级 `archcomp26_20261001` 目录。

## 观察口径与可比边界

| 方法 | 本次保存的 tube 和 endpoint | 与其他方法的关系 |
| --- | --- | --- |
| Huan、Xiangru | 同一 GPU driver 的 `hull_ranges_s(state,eng,6)` 及 `rows_range_over_time_sparse(state,eng,[[h,h]],6)`；分别取本地整期及 `h` 时刻。 | 同一观察函数，保存的两份 `ranges.jsonl` 直接逐字节相同。逐态数字可直接对照。 |
| ours/P3 | 同样调用上述两个 GPU 观察函数，在 working-P3 数值引擎与 strict endpoint 推进后观察本地 TM。 | 物理坐标、期数、时间点及轴对齐区间口径相同；引擎和生成的 TM 不同。可比较**保存范围的描述性数值**，不能据此宣称独立正确性或稳定优劣。 |
| 原生 corrected VAR tail | Flow* 的 `ARCH_RANGE_LOG` 二进制本地范围观察，分别保存每期 tube 和 `h` 时刻 endpoint；与 GPU 不共用观察函数。 | 时间、物理坐标及轴对齐输出目标相同，但观察实现不同。各值保留为独立列；细小宽度差异不能归因于求解器本身。`ranges.bin` 不编码 accepted/status，完成和 `VERIFIED` 需看相邻运行日志。 |

GPU driver 的最终终端 `HULL` 行来自另一个 endpoint 例程，**没有混入这些 CSV**。四方 `RESULT`/作者 checker、独立安全扫描与单次进程 wall 也不在范围数值列内，不应把它们与 endpoint 宽度当作同一观察量。四方完整性、全时半空间性质和浮点 NN 证明边界见[四方运行汇总](../ACC_PARTICIPANT_PROFILE_20261001.md)。本 CSV 只重算保存区间；它既不独立验证控制器边界，也不把一轮冷启动时间解释为稳态排名。

| 物理态 | Native T=5 width | Huan T=5 width | Xiangru T=5 width | ours/P3 T=5 width |
| --- | ---: | ---: | ---: | ---: |
| `x_lead` | 21.01205143634982 | 21.012040961970598 | 21.012040961970598 | 20.99786614290832 |
| `v_lead` | 0.20175842635252295 | 0.2017538905540981 | 0.2017538905540981 | 0.19848302167509146 |
| `a_lead` | 0.0007425444315312113 | 0.0007410763491990657 | 0.0007410763491990657 | 0.0006987294776128472 |
| `x_ego` | 5.497998802447029 | 5.497975787460462 | 5.497975787460462 | 5.163069177034146 |
| `v_ego` | 2.0155283077957584 | 2.015510730280848 | 2.015510730280848 | 1.8580462893489376 |
| `a_ego` | 1.1811239101972355 | 1.1811061829727711 | 1.1811061829727711 | 1.0968973787144316 |

全时 tube 的 48 个 `lo/hi` 与 T=5 端点的 48 个 `lo/hi` 均在 CSV 中；此表只为快速看端点宽度，不能代替原始列。
