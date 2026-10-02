# NAV 旧 GPU 与原生证据的作者执行合同审计

审计日期：2026-10-02。合同依据见 [NAV 作者执行合同](ARCHCOMP26_NAV_AUTHOR_EXECUTION_CONTRACT_20261002.md)。新主表建议把固定官方可执行方程与控制器作者原序接口显式命名：`[x,y,v,θ] → [u_v,u_θ]`，`(x',y',v',θ')=(v cos θ,v sin θ,u_v,u_θ)`；控制周期 0.2 秒，总时域 6 秒；初集 `[2.9,3.1]² × {0}²`；全时避开闭障碍 `[1,2]²`，末时进入闭目标 `[-0.5,0.5]²`。论文文字的 `[x,y,θ,v]` 与两层 `64/64` 仍作为来源冲突保留；不据此重排固定官方实际 `64/32` ONNX。

## 已完成历史任务：复用原始证据，不重复运行

服务器旧配置 `runs/archcomp_review_20260923/contracts/nav_{standard,robust}.yaml` 给出上述方程、原序四维网络输入与两维控制输出、30 × 0.2 秒、Flow* `h=0.01`/order 4、cutoff `1e-6`、初余项 `[-0.1,0.1]`、SR queue 1000、全时障碍与末时目标。标准按 `40×16=640` 盒、robust 按 `5×5=25` 盒分区。旧模型路径含 `ARCH-COMP2024`，本轮对各自路径与服务器固定 2026 point/set 副本做**直接逐字节比较**，内容相同；没有计算摘要。这个判断不扩展到另一个作者仓库的 Git LFS 模型。

| 历史方法与实例 | 原始结果 | 合同与数值记录 | 限定结论 |
| --- | --- | --- | --- |
| 旧 `ours` / PyTorch GPU，standard | [result](<../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/nav_standard_ours/result.json>)、[stdout](<../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/nav_standard_ours_watch/stdout.log>) | 上述 standard YAML 与 point ONNX；`strict`、`sparse`、box/same-slope CROWN、RPC float32；640 盒、600/600 小步、384,000/384,000 lane-step accepted、exit 0；作者 checker 打印 `VERIFIED`。 | 与命名的**物理/控制器合同**同一；旧引擎 `engine_linear_leaf_v2`、其时 RN 控制器注入未独立合格，`end_to_end_strict_certificate=false`。不能当作后来工作 P3 新内核的独立合格证据，也不启动重复 smoke。 |
| 旧 `ours` / PyTorch GPU，robust | [result](<../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/nav_robust_ours/result.json>)、[stdout](<../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/nav_robust_ours_watch/stdout.log>) | 同一 robust YAML 与 set ONNX；同一数值设置；25 盒、600/600 小步、15,000/15,000 lane-step accepted、exit 0；作者 checker 打印 `VERIFIED`。 | 完整数值时域历史证据；同样无独立端到端浮点证书，也不重复启动。 |
| 旧 Flow* native，robust | [原始 result](evidence/results/archcomp26_20261001/nav_robust_native_historical_20260923/result.json)、[日志](evidence/results/archcomp26_20261001/nav_robust_native_historical_20260923/native.log)、[ranges](evidence/results/archcomp26_20261001/nav_robust_native_historical_20260923/ranges.bin) | 保存源码 `suite_build/archcomp/nav_robust/matched_threads4.cpp` 使用同一方程与集合性质、5×5 分块、`h=.01`/order 4；服务端 `observed_server.py` 加载同内容 set ONNX，box/same-slope CROWN；原始结果 25 盒×600 子步，exit 0，作者 checker 打印 `VERIFIED`。 | 旧完整运行可作历史证据，不重复启动；旧原生 checker 的标签仍是作者 checker 输出，不提升为独立端到端证书。 |

旧 robust native 原始范围本轮只读镜像到[历史证据目录](evidence/results/archcomp26_20261001/nav_robust_native_historical_20260923/)。[独立记录扫描](evidence/results/archcomp26_20261001/nav_robust_native_historical_20260923/INDEPENDENT_SAVED_RANGE_SCAN.json)确认 25×600=15,000 条完整网格、所有区间有限且有序、0 条同步端点越出同小步 tube。[NAV 保存性质扫描](evidence/results/archcomp26_20261001/nav_robust_native_historical_20260923/INDEPENDENT_NAV_PROPERTY_SCAN.json)显示 0 条保存 x/y tube 与闭障碍盒相交、25 个 `t=6` 末端 x/y 盒均落在目标内。末端联合 x 为 `[0.10708168848705917,0.19808018604018326]`，y 为 `[-0.06337211080872229,-0.048295620925280607]`。这是保存范围上的独立数值检查，未审计神经网络浮点边界的全部可靠性。

## standard native 的未完成历史前缀与唯一新作业

旧 standard native 的两个 300 秒作业分别只保存每盒 220/600 和 240/600 小步，原始结果为超时；不能作为完整 6 秒结果，也不能用作者 checker 的任何末时标签补齐。新 [首盒首周期 smoke](evidence/results/archcomp26_20261001/nav_author_standard_native_smoke1_001/SUMMARY.md) 的新 ID、原始 RESULT、20 条范围和独立扫描已经保存；它只覆盖首周期。

唯一新 [standard native 全时域原始收据及摘要](evidence/results/archcomp26_20261001/nav_author_standard_native_full30_001/SUMMARY.md)使用固定官方 point 副本、旧 640 盒原始台账、GPU 1/CPU 6–9、RPC 5110、3600 秒上限，以及完成 30 周期/600 小步 guard。其 START 明记旧两次 300 秒超时的对照字段，原任务未重启。它已自然结束：外层 `completed`、exit 0、wall `1478.865919311531 s`，原生日志 `COMPLETED_PERIODS 30/30` 与作者 checker `VERIFIED`。[独立全范围扫描](evidence/results/archcomp26_20261001/nav_author_standard_native_full30_001/INDEPENDENT_SAVED_RANGE_SCAN.json)确认 640×600=384,000 条完整、有限、有序记录；[NAV 保存性质扫描](evidence/results/archcomp26_20261001/nav_author_standard_native_full30_001/INDEPENDENT_NAV_PROPERTY_SCAN.json)报告 0 条保存 tube 与闭障碍相交、640 个末端盒目标外 0 个。末端联合 x 为 `[-0.09175765699647812,-0.004467673970412854]`，y 为 `[0.08836682447943034,0.35454798657692854]`。这是单次完整数值及保存区间的性质观察，不是独立端到端浮点证明。

旧 `ours` 结果的 `driver_wall_s`、旧 native robust 的 `native_process_wall_s` 与新作业的外层 `wall_s` 测量边界不同，且 GPU/CPU 配额不同；这些数字不能构成四方速度排序。全四方统一资源、重复次数和独立证书仍缺失。

[历史对新原生的四方法 NAV 流管及宽度图](evidence/results/archcomp26_20261001/nav_fourway_historical_vs_new_20261002/README.md)只读归约各方法已保存的 x/y 区间：standard 的旧 `ours`/Huan/Xiangru 与新 native，robust 的四条均为历史记录。两种配置各有四方法×600 小步×2 坐标的紧凑 CSV 及原始路径审计。图上的 x 或 y 单独投影不能推断二维联合障碍性质；原始每盒每小步的联合区间交叉另在来源审计及上述保存性质扫描中检查。
