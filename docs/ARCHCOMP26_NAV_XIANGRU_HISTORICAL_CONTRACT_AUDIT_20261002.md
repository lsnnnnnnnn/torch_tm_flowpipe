# NAV Xiangru 历史全程结果与作者可执行合同审计

审计于 2026-10-02，只读检查 2026-09-23 保存的 NAV standard 与 robust Xiangru 原始作业。**没有启动新的 Xiangru smoke 或全程实验，没有做哈希校验。**本文件补充 [NAV 官方来源与作者执行顺序审计](ARCHCOMP26_NAV_AUTHOR_EXECUTION_CONTRACT_20261002.md)；它记录历史同合同证据，不增加本轮新尝试索引的条数。

## 逐字段合同核对

| 字段 | 旧 Xiangru 原始运行 | 当前明确命名的“固定官方 + 作者可执行顺序” profile |
| --- | --- | --- |
| 标准与 robust 模型 | 原始 `result.json` 分别指向旧路径 `nn-nav-point.onnx` 与 `nn-nav-set.onnx`。只读 `cmp` 将每个旧文件与已保存的固定官方 2026 ONNX 副本逐字节比较，两对均相同。 | 分别使用固定官方 point 与 set 文件；路径年份不造成模型内容差异。作者仓库 Git LFS 文件未与这两份内容比对。 |
| 状态、网络接口、动力学 | 原始 YAML 有七变量 `[x1,x2,x3,x4,t,u1,u2]`，`num_nn_input=4`、`num_nn_output=2`，原始输入布局、输出 scale 1 和 offset 0；`x1'=x3 cos(x4), x2'=x3 sin(x4), x3'=u1, x4'=u2`，附加时钟和采样保持。保存的 runner 从 YAML 加载模型并将盒原序传入共享作者 driver；`--official-controller` 选 native 布局，box/same-slope/rpc-float32，`--strict`。 | 物理态与网络输入均为 `[x,y,speed,heading]`，输出为 `[speed_rate,heading_rate]`，与固定官方方程和控制器作者可执行入口相符。论文文字写的另一状态顺序及隐藏层宽是来源冲突，不能归到这个 profile。 |
| 初集、分块 | 两份 YAML 均为 `x1,x2∈[2.9,3.1]`、其他初值 0。原始分区台账逐盒检查：standard 40×16=640、robust 5×5=25，均为完整笛卡尔网格，邻盒连续。standard 台账末边约为 `3.1000000000000134`，对官方上界有极小二进制浮点外包。 | 覆盖完整官方初集；standard 的微小外包已明确保留，不暗称十进制端点逐位相等。 |
| 时域与求解参数 | 两份 YAML 均 `steps=30`、采样 `0.2 s`、ODE 步长 `0.01 s`、四阶、cutoff `1e-6`、余项 `[-0.1,0.1]`、SR queue 1000。原始启动参数均为 strict sparse 引擎。 | 30 周期至 6 秒、每期 20 个 ODE 小步；数值参数是这份可执行 profile，不称为论文唯一强制选择。 |
| 性质 | 两份 YAML 均把闭盒 `x,y∈[1,2]` 作为全时障碍，把 `x,y∈[-0.5,0.5]` 作为末时目标。 | 与固定官方 2026 两项性质相同；robust 指集合训练控制器，不表示 plant 有额外扰动。 |

原始运行文件：[standard result](<../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/nav_standard_xiangru/result.json>)、[standard watch](<../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/nav_standard_xiangru_watch/process.json>)、[standard stdout](<../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/nav_standard_xiangru_watch/stdout.log>)；[robust result](<../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/nav_robust_xiangru/result.json>)、[robust watch](<../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/nav_robust_xiangru_watch/process.json>)、[robust stdout](<../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/nav_robust_xiangru_watch/stdout.log>)。服务器原始配置及范围位于 `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp_review_20260923/{contracts,suite_v1}`。运行时的共享 driver 为 `/srv/local/shengenli/xiangru_adoption_20260907T032448Z/xiangru_upstream/src/flowstar_gpu/integrations/crown_reach.py`。

## 原始结果与保存范围复查

| 实例 | 原始状态 | 数值覆盖 | 作者输出 | driver wall / 外层进程 wall | `T=6` 保存 endpoint 的 `(x,y)` 并集 |
| --- | --- | --- | --- | --- | --- |
| standard | `status=completed`, `complete=true`, exit 0 | 640 盒×600 小步；384,000/384,000 接受 | `VERIFIED` | 14.505592 / 17.907006 s | `[-0.08803818270400292,-0.007714880465252999]`；`[0.08895643289004677,0.3534297034666618]` |
| robust | `status=completed`, `complete=true`, exit 0 | 25 盒×600 小步；15,000/15,000 接受 | `VERIFIED` | 12.583398 / 15.461834 s | `[0.1074805687559851,0.1977835916197828]`；`[-0.06319188958683246,-0.04838890242461219]` |

原始 `result.json` 的 600 个小步编号连续，standard 每步有 640 个 true、robust 每步有 25 个 true，状态码全为 0。[独立保存范围扫描](evidence/results/archcomp26_20261001/nav_author_historical_xiangru_20261002/INDEPENDENT_SAVED_RANGE_AUDIT.json)按原始 152 字节记录布局直接顺序读取服务器的 `ranges.bin`：standard 恰有 384,000 条、robust 恰有 15,000 条；步号和盒号按顺序、`h=0.01`、所有物理态 tube 与 endpoint 均为有限有序区间。逐条保存 tube 的 `(x,y)` 盒均与障碍盒分离；各盒末端 endpoint 落入目标盒，且 endpoint 没有超出同小步 tube。此处只复查了**保存的数值包络**和作者 checker 输出；未构造独立端到端浮点 NNCS 证书。

两次旧 Xiangru 运行都用 GPU 3、CPU 14–17；外层进程时间含启动和收尾，与 driver 时间口径不同，且其资源与本轮多方法作业不同。旧 Huan 的相应 standard/robust `ranges.bin` 与旧 Xiangru 保存范围**直接逐字节相同**，两方沿用共享控制驱动与数学核心，因此不能当作相互独立的正确性证明，也不能据单次旧时间作稳定速度排名。

**结论：** 两条旧 Xiangru 结果可按其历史、同合同、完整数值时域的原始资格引用；本轮不再启动 Xiangru NAV 作业，也不把旧运行包装成新的 2026 attempt 格。仍缺的是独立端到端浮点 NNCS 证明与统一资源、多次计时；论文文字状态顺序/层宽与官方可执行来源的冲突继续明确标出。
