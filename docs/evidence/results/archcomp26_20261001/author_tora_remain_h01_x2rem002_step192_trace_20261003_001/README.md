# TORA remain：`h=0.1`、`x2` 初猜 ±0.02 的第 192 步首拒追踪

**状态：一次新 ID 诊断已结束并保存原始收据。** 新 ID 为 `author_tora_remain_h01_x2rem002_step192_trace_20261003_001/huan_step192_trace_001`；启动前只读确认服务器该目录不存在、旧变体已自然结束且 GPU3/CPU14–17 空闲。原 [Huan 补充变体](../author_tora_remain_h01_x2rem002_20261002_001/README.md)在第 192 步初盒 2 因 `FAILED_CONTRACTION` 首拒；此前的原主表第 190 步 `x2` 首拒已由[插桩](../author_tora_remain_h01_refusal_trace_20261002_001/README.md)定位。旧目录均未修改或重启。

新[外层 RESULT](huan_step192_trace_001/RESULT.json)为 `failed/exit 1`、非超时、wall **7.986703 s**，属于首拒即停的预期诊断退出。[内层 RESULT](huan_step192_trace_001/payload/RESULT.json)为 `stopped_first_numerical_rejection`：观察 192/200 小步，接受 **2303/2304** 盒步，第 192 步只有初盒 2 被拒，状态码 `1 = FAILED_CONTRACTION`；没有 `T=20` 完整数值时域或作者性质结论。CPU 合同[预检](preflight.json)在启动前通过。无残留诊断进程。

[直接比较审计](AUDIT.json)逐字比较生成 YAML、192 条逐步观察及 **2304 个 152-byte 范围记录**，均与旧 `x2±0.02` 补充变体一致；外层原始失败/非超时另行核对。只在第 192 步读取一次 12 盒×6 维 `validpost`，唯一不包含是初盒 2 的 `x2`：旧初猜 `[-0.02,0.02]`，新 Picard 提议 **`[-0.03005730939678987,0.028837305588174926]`**，左右裕量分别为 `−0.010057309396789869` 和 `−0.008837305588174926`。其余 71 个盒维组合本轮首轮自映射检查通过。这个结果只定位数值余项初猜失败，不给出另一个足以完成全程的半径，更不是实际轨迹越带证据。

[新驱动](diagnostic_driver.py)从已成功的第 190 步追踪复制，只加入原 `x2=±0.02` 的六维余项配置，并将唯一 `validpost` 插桩及硬停止点移到第 192 步。只在原 `GlueCache.run` 返回后读取全部 12 盒的六维初猜、Picard 提议、差界、自包含布尔值及两侧裕量，原张量不变地返回。模型、初盒、ODE、`h=0.1`、order 3、控制周期、CROWN/strict 与安全性质均保持该 **补充变体**的配置；不修改服务器引擎，不启用自映射重试。这不是冻结主表的统一余项设置。

运行前已再次只读核对旧 ID 原始 `RESULT`、新 ID 不存在、GPU3 / CPU14–17 空闲以及相关进程。实际沿用服务器现存 `author_tora_remain_v1/run_archcomp26_nohash.py` 的排他建目录和进程组超时监督，限制 120 秒、192 小步、至多 12×192 盒步。指定 `CUDA_VISIBLE_DEVICES=3`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1` 和 CPU14–17。原完整请求未重启，旧目录未覆盖。外层保存 `START.json`、`RESULT.json`、`stdout.log`、`stderr.log`；内层保存 `payload/START.json`、`config.yaml`、`observations.jsonl`、`ranges.bin`、`refinement_step192.jsonl` 和 `RESULT.json`。

[直接比较脚本](audit_saved.py)只读旧新保存文件并写出上述审计；没有内容摘要检查。第 185 步起旧保存 tube 已无法证实全时安全；本追踪不是 `T=20` 完整流管、安全证明或真实轨迹反例。详细事前合同与资源界见 [INTENT.json](INTENT.json)。

只读核对冻结服务器源码：Huan 的 `/srv/local/shengenli/flowstar-gpu/src/flowstar_gpu/sparse_exec.py:1506–1534` 与 Xiangru 的 `/srv/local/shengenli/xiangru_adoption_20260907T032448Z/xiangru_upstream/src/flowstar_gpu/sparse_exec.py:1572–1604` 都是一次 `rem_est`→`validpost`→`_refine_dispatch`，没有可启用的 `FLOWSTAR_SELF_MAP_RETRIES` 路径。该开关只出现在较新的独立源码快照，不能当作这两份冻结引擎的现成修复。当前两次首拒只支持具体失败位置；不据此猜测继续增大余项初猜就会完成 `h=0.1` 全时域。
