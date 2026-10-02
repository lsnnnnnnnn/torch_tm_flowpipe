# TORA remain `h=0.1`：作者 Huan 第 190 步数值首拒追踪

原 [Huan 全程尝试](../author_tora_remain_v1/huan_full20_001/payload/RESULT.json)观察到 200/200 小步，但只接受 2357/2400 盒步；第 190 步初盒 2 首拒。为找出旧收据没有保存的数值失败子类，本目录用新 ID、原官方 2026 ONNX、原 12 盒、原六态 ODE、`h=0.1`、order 3、余项初猜 `[-0.01,0.01]` 和原 CROWN/strict 设置做一次**限于首拒点**的只读插桩。原作业没有重启或覆盖；新入口最多执行 190 小步，首拒即停。

第一份 [`huan_step190_trace_001`](huan_step190_trace_001/payload/RESULT.json)在完成 189 小步后因远端已安装 Huan `Settings` 不支持本地新版 `refinement_callback` 而于第 190 步调用前异常。错误、原始 stdout/stderr、189 步范围和空 trace 均保留。随后改用[第二版插桩入口](diagnostic_driver_v2.py)：只在第 190 步包裹远端 `GlueCache.run`，先调用原函数，然后从 `validpost` 返回值读取 12 盒的 Picard 提议和逐维自包含裕量，返回原张量，不改配置、源库或求解分支。

| 第二份 [`huan_step190_trace_002`](huan_step190_trace_002/payload/RESULT.json) | 保存结果 |
| --- | --- |
| 小步/盒步 | 190/200 已观察，2279/2280 接受；第 190 步初盒 2 首拒，立即停止 |
| 已安装引擎状态 | 初盒 2 的 `status=1`，远端[源行](REMOTE_STATUS_CODES.txt)对应 `FAILED_CONTRACTION`；其余 11 盒仍为 `ACTIVE=0` |
| 第 190 步初盒 2 的 `x2` | 原余项初猜 `[-0.01,0.01]`；Picard 提议 `[-0.012253595542717259,0.011665851915157245]`；左右包含裕量分别 `-0.002253595542717259` 与 `-0.0016658519151572446` |
| 同步检查 | 12 盒 × 6 分量中只有初盒 2 的 `x2` 首次自包含失败；`x1,x3,x4,t,u1` 以及其他 11 盒各维通过 |

[逐盒原始 `validpost` 记录](huan_step190_trace_002/payload/refinement_step190.jsonl)、[逐步状态](huan_step190_trace_002/payload/observations.jsonl)、[直接字节及裕量审计](AUDIT.json)保留具体证据。审计将两份新运行各自保存的全部 `ranges.bin` 与原全程运行等长前缀直接逐字节比较：前 189/190 步均相等，四项观察字段也逐步相等；没有计算内容摘要。[独立区间扫描](huan_step190_trace_002/INDEPENDENT_INTERVAL_SCAN.json)复查 2280 条已观察记录，接受盒区间有限有序、同段 endpoint 落在 tube 内；保存 tube 首次越出 `[-2,2]^4` 仍为第 185 步。

这明确定位**该数值设置下**第 190 步的 `x2` Picard 自包含失败；不说明真实网络轨迹越带，也不提供完整 `T=20` 流管或性质证明。新追踪的 7.94 秒外层失败进程时间不能与完整结果排名。
