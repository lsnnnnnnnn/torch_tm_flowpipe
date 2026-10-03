# TORA reach-sigmoid 第一次四方计时 campaign：首槽审计器误拒

本目录镜像服务器独立 ID `tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_001` 的原始 `PLAN.json`、`events.jsonl`、`SUMMARY.json`、冻结 `runner.py` 与唯一已启动的 [`first00_native`](first00_native/) 原始收据、日志、RPC 记录和 `ranges.bin`。该 campaign 在第一槽后停下；余下 23 槽没有启动。新 ID 的后续计时不得混入本目录的统计。

这一个原生进程的 [outer RESULT](first00_native/RESULT.json) 为 `completed`、exit 0、`wall_s=8.989589569158852`。`native.log` 记录 `Step 0`–`Step 9` 和作者终点 checker `VERIFIED`；控制日志及 RPC 服务各记录 10 次请求/HTTP 200。直接重读 `ranges.bin` 的 500 行：lane 0、step 1–500 齐全，四态 tube/endpoint 均有限有序，每个 endpoint 包含于相应 tube；`T=5` 的 `x1=[0.1345319317307225,0.16059887233324702]`、`x2=[-0.8763648122763305,-0.8505857060659042]` 落在所选目标内。这是完整数值时域与保存终点观察，并非独立浮点 NNCS 证明。

旧 [`runner.py`](runner.py) 第 197 行要求每段 `abs(h-0.01)≤1e-12`；第 50 行及最终多处保存 `h=0.009999999998999698`，与 `0.01` 的最大差值 `1.0003022715698862e-12`，仅多约 `3.02e-16`。因此旧事件把这个已完成的进程写成 `valid=false`、`failure="saved grid mismatch at 50"`，并依首拒规则停下。该标签是 **campaign 审计器的容差误拒**，不是求解器拒绝、区间缺失或性质 `UNKNOWN`。当前修正源把这个非性质判据放宽到 `1e-9`，需在新的独立 campaign 中重新审计；原始 `runner.py` 和失败事件保留不改。

这次运行是第 201 条新 attempt，记为 `status=completed`、`matrix_eligible=false`、`timing_eligible=false`。其 `wall_s` 只用于原始进程事实，不进入后续正式 24 次计时表或四方速度排名。新 campaign 尚未纳入本目录或第 201 条截点。
