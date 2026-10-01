# Stage A: P3+trig 后处理优化审计（只读，未启动新实验）

日期：2026-10-01。对象是冻结的 1,024 盒、P3/point2/validation4、trig-reuse、1,000 步完成运行。该审计读取已有源码与结果，不执行任何内容哈希或 SHA-256 校验，也不修改冻结源码或证据。

## 已有计时能定位到哪里

冻结 `research/gpu_verified_20260930/report/evidence/quad_trig_RESULT.json` 记录进程内时间 1,531.395193 秒，其中 advance 1,234.852041 秒、observer 75.132491 秒、working-prune timed region 192.952505 秒。该 region 的 1,000 条已有逐步记录可进一步拆成：

| 类型 | 步数 | 记录时间之和 |
|---|---:|---:|
| 待逐出图项非空，执行完整数学状态签名 | 19 | 191.723381 秒 |
| 无待逐出图项，只执行存储/版本/容量检查 | 981 | 1.229124 秒 |

19 个步骤是 2、3、4、9、10、20、28、30、33、41、45、82、138、139、141、402、590、842、944。晚期步骤 402、590、842、944 分别记录约 21.878、31.803、45.993、51.440 秒；与随 SR/host 历史增长的完整签名路径相符，但现有计时没有把签名与 CUDA 同步、图逐出、`empty_cache()`、日志写入逐项分离。**191.723381 秒只是这些混合步骤的整段上限，不能当作预计净收益。**75.132491 秒的 observer 也同时包括原始 tube/endpoint 求界、GPU 同步、张量搬运和每步导出；没有子阶段证据表明观察器计算或导出分别可省多少。

## 精确触发与调用链

1. `run_fullbatch_p3_selective_audit_long.py:142-170` 的 `LongRuntime.prune_state()` 先调用原始 `b.w.Runtime.prune_state()`，再用 `helper.peek()` 取得逐出计划；`full=bool(plan['stale_count'])` 决定是否在逐出前后各调用一次 `b.probe.mathematical_signature(self)`。函数随后调用 `binding.finish_step(self.eng)`、检查存储/版本/容量，最后调用 `eager_segments.finish_step()`。
2. `selective_prune_audit.py:28-37` 与 `working_graph_prune.py:55-59` 都以 `stale=current-used` 判断工作图项。`working_graph_prune.py:65-69` 在非空时执行必要的 CUDA 同步、删除旧图项、`torch.cuda.empty_cache()`；这些操作必须保留，不能把整段计时视为纯诊断。
3. `probe_working_graph_prune.py:31-35` 的完整签名覆盖 plant 状态、cap、接受标志、进度及 `ledger._snapshot_views(sr)` 的活跃 SR/host 历史。实际 `Runtime.signature()` 位于 `quad_fullbatch_sr_20260928/run_fullbatch.py:145-153`，逐块拷贝 CUDA 张量到 CPU 并做 SHA-256。随历史长度增长，其代价有合理的尺寸机制，但尚无净收益测量。
4. `run_fullbatch_p3_observer64.py:89-107` 的 `observe()` 调用 tube 求界、64 lane endpoint helper，并用 `artifact()` 写每步 PT；基类 `run_fullbatch.py:160-166` 对张量及 PT 文件做内容摘要。此路径在任何新增实验中也必须处理用户的无哈希要求。

## 最小候选及边界

为**新的独立性能实验身份**构造 `LongRuntime.prune_state()` 的 clean arm：保持原始 `prune_state`、`helper.peek`/逐出计划、`binding.finish_step`、存储/版本/容量检查和 `eager_segments.finish_step` 的调用顺序，只去掉两次 `mathematical_signature()`、与旧参考数学签名的比较及对应签名日志。不要改变数值运算、控制器、SR 队列、图逐出、观察器范围或接受判定。这个候选针对的是诊断/导出开销；即使测得较快，也应称为同数值路径的运行器性能变化，不能称为求解器算法提速。现有 40 步参考已证明原冻结版本在指定步骤的字节一致；新候选仍需自己的无哈希数值对照。

建议两个分开的短前缀进程：qualification arm 在 40 步结束后，用 `torch.equal`/逐元素相等检查每步 observer 数值、接受/status、关键 plant 与 SR/host 活跃张量（可以使用已有 PT 数值，不能计算摘要）；clean arm 仅保留必要图逐出和轻量结构检查，在相同输入、设备、预载模块和计时边界测时间。资格检查中的 GPU→CPU 拷贝和比较放在计时窗外；否则不能从差值推断性能。若 40 步差值不足以代表后期长 SR 历史，可从已有合法快照做单步/短段 continuation，但其读取和保存路径也须先改成完全无哈希，并标明它是不同时间边界。

## 为什么现在不能直接启动

当前 `run_fullbatch_p3_trig_continue.py` 的装载、旧资格门、冷启动、结果/快照对照和结束收据都调用 `sha()`；`run_fullbatch_p3_gpu14.py` 及其多层基类在导入、初始化、`artifact()`、`save()`、结束阶段也调用 SHA-256。`working_graph_prune.py:15-16`、`selective_prune_audit.py:15-20` 和 `observer_endpoint64.py:19-25` 自身还有来源哈希。仅在最上层关闭 working-prune 数学签名，仍会违反本轮“不要做任何哈希校验和 sha256”的要求。

CUDA 加载也不能默认称为无 JIT：`cuda_kernels.py:488-530` 用 `load_inline()` 懒加载扩展，`sr_kernels.py:106-113` 经同一路径，`horner_edge_kernels.py:108-125` 在 `load()` 内做来源摘要并调用 `load_inline()`；`glue.py:110` 可调用 `torch.compile()`。缓存命中不等于从未进入 JIT 路径。要运行上述 P3 短前缀，先在隔离副本中建立完整的无哈希入口与纯预编译模块加载，检查所有实际调用路径后再启动。普通 `tools/run_archcomp26_nohash.py` 只保证它自己的监控包装不做内容哈希，不能保证被它启动的冻结 P3 子进程无哈希。

因此本阶段**没有**启动改版 P3 测速，也没有将 191.723381 秒或 75.132491 秒报告为可移除开销。下一个可执行动作是构造并审阅隔离的无哈希、无 JIT 短前缀入口；在入口满足用户限制之前保留冻结结果原样。
