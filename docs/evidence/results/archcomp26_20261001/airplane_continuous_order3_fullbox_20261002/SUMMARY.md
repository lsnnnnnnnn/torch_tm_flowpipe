# Airplane continuous：官方全初盒执行预检与三次新短程尝试

本页只覆盖连续版。固定官方 2026 规格给一个未分割 12 物理态初盒：`x,y,z,r,p,q=0`，`u,v,w,phi,theta,psi∈[0,1]`；控制周期 `0.1 s`、20 期、终点 `T=2 s`，全时要求 `y,phi,theta,psi∈[-1,1]`。实际固定 ONNX 为 `airplane_prep_001/controller_airplane.onnx`，接口 `[N,12]→[N,6]`，输出按 `Fx,Fy,Fz,Mx,My,Mz` 注入。合同来源与旧点初盒/旧 C++ UNKNOWN 续跑问题见[来源审计](../../../../ARCHCOMP26_AIRPLANE_2026_ENTRY_AUDIT.md)。没有复用旧单点成绩，也没有执行内容摘要校验。

CPU-only [order-6 预检](../airplane_continuous_fullbox_20261002/PREFLIGHT.json)及本目录 [order-3 预检](PREFLIGHT.json)均通过：固定官方模型由 `onnx.checker` 检查，19 条 ODE 经当前引擎静态解析，8 条安全约束经同一表达式语法解析；预检没有初始化 GPU 或做数值推进。[order-6 配置](../airplane_continuous_fullbox_20261002/config.yaml)与[order-3 配置](config.yaml)在直接 YAML 语义对照下**仅 `ode_order` 不同**：旧方法候选为 6，本诊断 profile 为 3；二者均用同一完整初盒、`h=0.01`、20×0.1 s 和相同性质。Taylor 阶数不是官方强制字段，order-3 结果不能并入 order-6 数值候选。

| 独立新尝试 | 远端原始 run_dir（均在 `N/runs/archcomp26_20261001/`） | 外层 wall | 外层 status / exit | 已接受 ODE 小步 | 精确停止原因 |
| --- | --- | ---: | --- | ---: | --- |
| [Huan order-6 smoke](../airplane_continuous_huan_smoke1_001/RESULT.json) | `airplane_continuous_huan_smoke1_001` | 252.246969 s | failed / -15 | 0 | 表构建时 RSS 超过 54,006,540 KiB；仅终止本子进程组；[资源采样](../airplane_continuous_huan_smoke1_001/RESOURCE_OBSERVATION.json) |
| [Huan order-3 smoke](../airplane_continuous_order3_huan_smoke1_001/RESULT.json) | `airplane_continuous_order3_huan_smoke1_001` | 6.784922 s | failed / 1 | 0 | 第 1 个 `0.01 s` 小步唯一全初盒 `accepted=false`；[观测](../airplane_continuous_order3_huan_smoke1_001/payload/observations.jsonl) |
| [Xiangru order-3 smoke](../airplane_continuous_order3_xiangru_smoke1_001/RESULT.json) | `airplane_continuous_order3_xiangru_smoke1_001` | 6.985431 s | failed / 1 | 0 | 同样第 1 个 `0.01 s` 小步唯一全初盒 `accepted=false`；[观测](../airplane_continuous_order3_xiangru_smoke1_001/payload/observations.jsonl) |

`N=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z`。三次均为独占新目录、GPU2/CPU10–13；order-3 子进程额外受 32 GiB 虚拟地址上限及 300 s 外层超时保护，均未触限。order-6 尝试的外层超时是 600 s；因资源增长在 252.247 s 手动 SIGTERM，外层保留失败 RESULT，但尚未写出 payload RESULT。所有小原始 START/RESULT/stdout/stderr、实际配置、观察日志和空 `ranges.bin` 已分别镜像在本地三个运行目录；原始远端目录保持不变。

order-6 的资源阻断来自当前 GPU 引擎 `build_tables(n=19,k=6)` 的全单项式配对：工作基底 `C(26,6)=230,230`，两两索引 `53,005,852,900` 项，单是两个 int64 索引理论存储约 848 GB。它尚未进入任何 ODE 小步，所以不是性质 `UNKNOWN/UNSAFE`。order-3 两方已经进入 `Step 0`，但在首个小步验证返回拒绝；没有可接受区间，`ranges.bin` 为空。入口按预设策略当场停止，未让旧 C++ 那种 UNKNOWN 后继续推进发生。**拒绝的内部细分原因本轮没有被记录**，不能猜成实际轨迹越界或 NN 错误。

因此两种数值 profile 都没有通过完整一周期 smoke，未启动任何 `T=2` full；无全时安全结论、无全初盒终点宽度，也没有可供四方计时排名的完成样本。后续若要推进，需要在新诊断身份下记录首步验证失败的内部状态，并选择能够对该完整初盒产生已接受小步的明示方法参数/实现；旧点初盒结果不得代填。
