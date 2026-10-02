# NAV standard 原生完整时域单次新运行

新运行 ID：`nav_author_standard_native_full30_001`。服务器原目录为 `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/nav_author_standard_native_full30_001`。[START](START.json)、[RESULT](RESULT.json)、[原生日志](native.log)、[服务日志](server.log)及[外层 stdout](stdout.log)均从原目录只读镜像。58,368,000 字节的 `ranges.bin` 原件保留在该服务器目录；本地保存其独立[全记录扫描](INDEPENDENT_SAVED_RANGE_SCAN.json)和[NAV 性质投影扫描](INDEPENDENT_NAV_PROPERTY_SCAN.json)。控制器 RPC 原始日志在服务器目录，本地另有无损压缩镜像 `controller_rpc.jsonl.gz`。

本次任务使用固定官方 2026 point ONNX、作者可执行状态/网络原序、旧 640 盒原始台账、30 个 0.2 秒控制周期、Flow* 0.01 秒固定步长与阶数 4。原生进程 CPU 6–9、控制器 GPU 1、RPC 端口 5110；3600 秒上限。它是一个**新 ID 的单次**完整尝试，动机是旧 standard native 两次各受 300 秒限制、每盒只保存 220/600 和 240/600 小步；旧任务没有重启。新源码增加 30 周期与各盒 600 小步完成 guard，源码及构建记录在 [`native_nav_standard_author_build_001`](../native_nav_standard_author_build_001/BUILD.json)。

外层 RESULT 为 `completed`、exit 0、`timed_out=false`、wall `1478.865919311531 s`；原生日志有 `COMPLETED_PERIODS 30/30`，作者 checker 打印 `VERIFIED`，并报告内部 `time cost: 1458.200000`。独立读取保存范围得到 640×600=384,000 条完整 lane-step 网格，记录大小 152 字节，0 非有限、0 区间倒序、0 同步端点超出同小步 tube。保存的 x/y tube 与闭障碍 `[1,2]²` 相交 0 条；640 个 `t=6` 末端 x/y 盒落在闭目标 `[-0.5,0.5]²` 之外 0 个。

四个物理态的绝对末端宽度也保存在可机读的 [CSV](terminal_widths.csv)：

| 物理态 | 末端联合区间 | 联合宽度 | 分块平均宽度 | 分块最大宽度 |
| --- | ---: | ---: | ---: | ---: |
| x | `[-0.09175765699647812,-0.004467673970412854]` | `0.08728998302606526` | `0.02975178896891136` | `0.0823040664383668` |
| y | `[0.08836682447943034,0.35454798657692854]` | `0.2661811620974982` | `0.02937692508869974` | `0.05590697511963416` |
| speed | `[-0.1511771073005031,0.015832430317627044]` | `0.16700953761813014` | `0.03806200786237967` | `0.16700953761813014` |
| heading | `[-0.09846913962904279,0.6104767306829899]` | `0.7089458703120327` | `0.14238464256617242` | `0.37262866207903256` |

这些是本次保存区间上的完整数值观察，作者 checker 的 `VERIFIED` 与本地读回并不构成独立端到端浮点 NNCS 证书。本次单样本在不同预算、GPU、CPU 和计时边界下不可用于四方法速度排名。
