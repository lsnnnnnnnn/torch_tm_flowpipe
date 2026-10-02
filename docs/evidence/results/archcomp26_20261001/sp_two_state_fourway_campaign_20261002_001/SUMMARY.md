# Single Pendulum 两物理态：四方独立进程轮换计时

远端原始目录为 `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/sp_two_state_fourway_campaign_20261002_001`；本目录镜像了全部 24 个原始 run 子目录及 [`PLAN.json`](PLAN.json)、[`events.jsonl`](events.jsonl)、[`SUMMARY.json`](SUMMARY.json)。新 campaign 使用已完成四方单次运行的**具名论文两物理态合同**：物理初盒 `x1∈[1,1.175]、x2∈[0,0.2]`，第三变量仅是 `t(0)=0,t'=1` 的检查器时钟；20 个 `0.05 s` 控制期，每期 5 个 `0.01 s` ODE 小步，只在闭窗口 `t∈[0.5,1]` 要求全时 `x1∈[0,1]`。固定 2026 ONNX 的路径、完整参数与顺序写在计划及逐次 START 中。原先各方法的一次完整运行未重启、未纳入本 campaign。

四法按轮次轮换并**顺序**使用物理 GPU 2、CPU 10–13；native 独占 RPC 端口 5101。每法第 0 轮是 campaign 内第一个新进程，第 1–5 轮是另五个新进程；进程间隔 2 秒。统一外层 wall 取 supervisor 启动子进程前到回收后，native 包含 RPC 服务启动与求解，GPU 法包含 Python、NN/CUDA 初始化与 driver。每次超时上限 120 秒，任一无效样本即停止后续轮次。服务器原有 GPU 0 的其他服务仍在运行，因此这是共享主机描述性计时；“首个进程”不等于主机/GPU 重启后的冷机。

24/24 个新进程 exit 0，均完成 20 个控制期、100/100 个 ODE 小步。native 每次有 `Step 0`–`Step 19`、`COMPLETED_PERIODS 20/20`、作者 `VERIFIED`、20 个 RPC 请求与 20 个 HTTP 200；其余三法每次有 20 个控制调用、作者闭窗口 50/100 小步检查与完整接受状态，P3 还保存 50 个非正违例上界事件。独立[扫描](INDEPENDENT_AUDIT.json)重新读取原始二进制/JSON 区间，确认 2,400/2,400 条两物理态 tube/endpoint 有限有序、每个 endpoint 包于同段 tube、初盒被首段 tube 包含；`t=0.5` 的端点及后 50 段完整 tube 均在 `x1∈[0,1]` 内。该扫描没有独立证明浮点 NN 包络或端到端 NNCS 证书。

| 方法 | 首个新进程 wall (s) | 后五次 median (s) | 后五次 min–max (s) |
| --- | ---: | ---: | ---: |
| Flow* native RPC | 4.727866 | 4.729023 | 4.728539–4.729131 |
| Huan GPU | 5.381102 | 5.380765 | 5.280744–5.681419 |
| Xiangru GPU | 5.631141 | 5.480807 | 5.380850–5.581295 |
| ours/working-P3 | 6.284114 | 6.183896 | 6.133769–6.483611 |

逐次 wall、原始目录、控制/RPC 次数、闭窗口 tube union 与均值/最大宽度在 [`RUNS.csv`](RUNS.csv) 和 [`INDEPENDENT_AUDIT.json`](INDEPENDENT_AUDIT.json)。[`audit_campaign.py`](audit_campaign.py) 是不启动求解器的独立再扫描入口；新的[运行器](../../../../../tools/archcomp26_sp_fourway_campaign_nohash.py)及[审计器](../../../../../tools/audit_archcomp26_sp_fourway_campaign_nohash.py)也保存在分支。没有执行内容摘要校验。

此表只评价**两物理态加辅助时钟**的命名合同。2026 参与者 MATLAB 三态执行的第三状态初值和入口仍未取得，故不能把此表称作官方三态复现，也不能从共享主机的五个样本推断稳定的四方速度排名。
